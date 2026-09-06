"""Perceive one remaining source object and associate its labelled region.

The identity table -- what each label looks like and what its graspable part
is called -- is supplied by the graph through ``identity_hints`` and
``canonical_targets``; the script itself knows no object.
"""

import json
import os
import re
import sys
from typing import Any, TypedDict

import numpy as np
from gap import NodeContext

# The runtime loads each node script standalone, so a sibling is not importable
# by package path. Put this script's own directory on the path and import it by
# name -- the same thing the runtime does for the entry module.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sorting_cv import bounds as _bounds  # noqa: E402
from sorting_cv import box as _box  # noqa: E402
from sorting_cv import camera as _camera  # noqa: E402


class Output(TypedDict, total=False):
    status: str
    attempted_json: str
    target_name: str
    target_label: str
    target_obb: dict[str, Any]
    target_mask: np.ndarray
    target_cloud: dict[str, Any]
    destination_obb: dict[str, Any]


_DEFAULT_IDENTITY_HINT = "Identify the narrow graspable handle or graspable body."


def _normalize(label: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(label).lower()).strip()


def _parse_label_list(value: str) -> set[str]:
    """A comma-separated or JSON list of labels, normalized."""
    text = str(value or "").strip()
    if not text:
        return set()
    if text.startswith("["):
        items = json.loads(text)
    else:
        items = text.split(",")
    return {_normalize(item) for item in items if _normalize(item)}


def _parse_canonical_targets(value: str) -> dict[str, str]:
    text = str(value or "").strip()
    if not text:
        return {}
    table = json.loads(text)
    if not isinstance(table, dict):
        raise ValueError("canonical_targets must be a JSON object mapping label to TARGET phrase")
    return {_normalize(label): str(target).strip().lower() for label, target in table.items()}


def _empty_result(status: str) -> Output:
    empty_obb = {
        "center": {"x": 0.0, "y": 0.0, "z": 0.0},
        "extent": {"x": 0.0, "y": 0.0, "z": 0.0},
        "orientation": {"w": 1.0, "x": 0.0, "y": 0.0, "z": 0.0},
    }
    return {
        "status": status,
        "target_name": "",
        "target_label": "",
        "target_obb": empty_obb,
        "target_mask": np.zeros((1, 1), dtype=np.uint8),
        "target_cloud": {"points": np.empty((0, 3), dtype=np.float32)},
        "destination_obb": empty_obb,
    }


def _load_attempts(attempted_json: str) -> list[dict[str, Any]]:
    """The attempt log, or an empty one. A malformed log is an empty log.

    The log is how a *looping* graph remembers what it already tried. Excluding
    by destination *label* cannot express it: several objects can share a label,
    so "I tried the hammer" wrongly retires every hammer in the bin, and one
    object that refuses to be picked wrongly retires the rest of its kind.
    """
    try:
        loaded = json.loads(str(attempted_json or "[]"))
    except (TypeError, ValueError):
        return []
    return [entry for entry in loaded if isinstance(entry, dict)] if isinstance(loaded, list) else []


def _matching_attempt(
    attempts: list[dict[str, Any]], px: float, py: float, rx: float, ry: float
) -> dict[str, Any] | None:
    """The logged attempt at this spot, if there is one.

    An object is identified by *where it is*, in full-frame pixels, because
    nothing else about it is stable across passes -- the label is shared and the
    model's TARGET phrase is free text. The tolerance is the object's own
    half-extent (floored, so a tiny object still has a catchment), which is the
    same reasoning `verifying-grasps.verify_reach` uses: what counts as "the
    same place" is a question about the size of the thing.
    """
    for entry in attempts:
        try:
            dx = abs(float(entry["px"]) - px)
            dy = abs(float(entry["py"]) - py)
        except (KeyError, TypeError, ValueError):
            continue
        if dx <= max(rx, 8.0) and dy <= max(ry, 8.0):
            return entry
    return None


def _exhausted_clause(attempts: list[dict[str, Any]] | None, crop_x: float, crop_y: float,
                      crop_w: float, crop_h: float) -> str:
    """Tell the model which spots are spent, in the frame it is being shown.

    Without this the loop cannot make progress: a deterministic model asked the
    same question about the same image returns the same object forever. The
    spots are rendered back into the crop's own 0-1000 coordinates because that
    is what the prompt asks the model to answer in.
    """
    if not attempts:
        return ""
    tried, settled = [], []
    for entry in attempts:
        try:
            x = (float(entry["px"]) - crop_x) / max(crop_w, 1.0) * 1000.0
            y = (float(entry["py"]) - crop_y) / max(crop_h, 1.0) * 1000.0
        except (KeyError, TypeError, ValueError):
            continue
        if not (0.0 <= x <= 1000.0 and 0.0 <= y <= 1000.0):
            continue
        (settled if entry.get("reason") == "settled" else tried).append(f"({x:.0f},{y:.0f})")
    parts = []
    if tried:
        parts.append("Objects at these points have already been attempted and must not be "
                     f"chosen again: {', '.join(tried)}.")
    if settled:
        parts.append("Objects at these points are already in their destination and are not "
                     f"unsorted: {', '.join(settled)}.")
    return (" " + " ".join(parts)) if parts else ""


def _settled_region(obb: dict[str, Any], regions: list[dict[str, Any]], margin: float):
    """The destination this object is already sitting in, if it is in one.

    A looping graph re-reads the whole scene each pass, so an object it has
    already delivered is still visible and still matches its label. Without this
    it gets picked up and put down again forever. Axis-aligned containment of
    the object's centre, with *margin* undoing the inset a layout builder
    applies to a compartment box so the test matches the region as drawn.
    """
    try:
        cx = float(obb["center"]["x"])
        cy = float(obb["center"]["y"])
    except (KeyError, TypeError, ValueError):
        return None
    for region in regions:
        box = region.get("obb") or {}
        try:
            rx, ry = float(box["center"]["x"]), float(box["center"]["y"])
            ex, ey = float(box["extent"]["x"]) + margin, float(box["extent"]["y"]) + margin
        except (KeyError, TypeError, ValueError):
            continue
        if abs(cx - rx) <= ex and abs(cy - ry) <= ey:
            return region
    return None


def _finished(attempts: list[dict[str, Any]] | None) -> Output:
    """The exit that says there is nothing left to pick.

    Carries the attempt log on when there is one. ``None`` means the caller is
    excluding by label and has no log, and then this publishes exactly the keys
    it always did -- a new output key is a change to what the node publishes,
    and a caller that did not ask for the loop should not have to see one.
    """
    result = _empty_result("finished")
    if attempts is not None:
        result["attempted_json"] = json.dumps(attempts)
    return result


def _source_box(ctx: NodeContext, image: Any, query: str) -> dict[str, Any]:
    """Choose the tight source container/cluster, not a box spanning the whole scene."""
    detections = (
        ctx.tool("grounding-dino.detect", image=image, query=query, box_threshold=0.10).get(
            "detections"
        )
        or []
    )
    plausible = []
    for detection in detections:
        box = _box(detection)
        x1, y1, x2, y2 = _bounds(box)
        width, height = x2 - x1, y2 - y1
        label = str(detection.get("label", "")).lower()
        if (
            width >= 100
            and height >= 100
            and any(token in label for token in ("source", "tool", "bin", "pile"))
        ):
            plausible.append((width * height, -float(detection.get("score", 0.0)), box))
    if not plausible:
        raise ValueError(f"could not localize {query!r}")
    return min(plausible, key=lambda item: (item[0], item[1]))[2]


def run(
    ctx: NodeContext,
    observation: dict[str, Any],
    instruction: str,
    layout_json: str,
    source_description: str,
    camera_name: str = "overhead",
    exclude_label_1: str = "",
    exclude_label_2: str = "",
    exclude_label_3: str = "",
    identity_hints: str = "",
    canonical_targets: str = "",
    allowed_labels: str = "",
    exclusion_mode: str = "label",
    attempted_json: str = "[]",
    max_attempts_per_instance: int = 2,
    max_reselects: int = 3,
    max_total_attempts: int = 40,
    settled_margin_m: float = 0.0,
    min_box_px: float = 8.0,
    min_box_fraction: float = 0.0,
) -> Output:
    """Select one unsorted object and its destination region.

    ``identity_hints`` is the graph's visual-identity text for the labels in
    play (what each looks like and what TARGET phrase to answer); when empty
    a generic "narrow graspable handle or body" cue is used.
    ``canonical_targets`` is a JSON object mapping a label to the exact
    TARGET phrase the model must return for it; a mismatch is an error only
    for labels in the table. ``allowed_labels`` (comma-separated or JSON
    list) restricts the layout to those labels before anything is asked.

    ``exclusion_mode`` decides how the caller says "not that one again".
    ``"label"`` -- the default, and what this always did -- retires a
    *destination label* through ``exclude_label_1..3``, which suits a graph that
    unrolls one round per label. ``"instance"`` keeps a per-object attempt log
    in ``attempted_json`` and hands it back on every result, which is what a
    graph that *loops* needs: several objects can share a label, so retiring the
    label would retire the rest of its kind along with the one that would not be
    picked. In instance mode the node re-asks up to ``max_reselects`` times,
    telling the model which spots are spent, and calls it ``finished`` after
    ``max_total_attempts`` in total or when every visible object is exhausted.

    ``min_box_px`` and ``min_box_fraction`` set the smallest box a reply may
    give, as ``max(min_box_px, min_box_fraction * crop side)``. A flat floor is
    a claim about pixels, and pixels are not a fixed size: on a 1280x960 frame
    the same object covers four times the area it does at 640x480, so a floor
    tuned for one resolution rejects real answers at the other. Measured on a
    drawn sorting scene, a small cell read as a 15 x 50 box in the model's own
    frame and was refused as "too small" while being exactly what was asked
    for. The default is the flat 8 px this always used; a caller that knows its
    frame passes a fraction instead.

    ``settled_margin_m`` above zero turns on the already-delivered test: an
    object whose centre lies inside a destination region is not unsorted, and a
    looping graph that re-reads the whole scene would otherwise pick it up and
    put it down again forever. The margin undoes the inset a layout builder
    applies to a compartment box; zero (the default) leaves the test off.
    """
    regions = json.loads(layout_json)
    if not regions:
        raise ValueError("sorting layout has no labelled regions")
    camera = _camera(observation, camera_name)
    image = camera["rgb"]
    allowed = _parse_label_list(allowed_labels)
    if allowed:
        regions = [region for region in regions if _normalize(region["label"]) in allowed]
        if not regions:
            raise ValueError(f"none of the layout labels is in allowed_labels {sorted(allowed)!r}")
    labels = [str(region["label"]) for region in regions]
    excluded = {
        _normalize(value) for value in (exclude_label_1, exclude_label_2, exclude_label_3) if value
    }
    remaining_labels = [label for label in labels if _normalize(label) not in excluded]
    if not remaining_labels:
        return _empty_result("finished")
    canonical = _parse_canonical_targets(canonical_targets)
    identity = str(identity_hints).strip() or _DEFAULT_IDENTITY_HINT
    by_instance = str(exclusion_mode).strip().lower() == "instance"
    attempts = _load_attempts(attempted_json) if by_instance else None
    all_regions = json.loads(layout_json)
    if by_instance and len(attempts) >= max(1, int(max_total_attempts)):
        return _finished(attempts)

    source_box = _source_box(ctx, image, source_description)
    sx1, sy1, sx2, sy2 = _bounds(source_box)
    image_array = np.asarray(image)
    # Detector boxes often hug the dense pile and clip handles protruding over
    # a bin wall. Preserve a visual margin while keeping the labelled
    # destination outside the crop.
    crop_x, crop_y = max(0, int(sx1) - 40), max(0, int(sy1) - 55)
    crop_x2 = min(image_array.shape[1], int(np.ceil(sx2)) + 15)
    crop_y2 = min(image_array.shape[0], int(np.ceil(sy2)) + 35)
    source_crop = image_array[crop_y:crop_y2, crop_x:crop_x2]
    # Fixed for every pass: the crop is taken once, before the loop.
    crop_h, crop_w = source_crop.shape[:2]
    floor_w = max(float(min_box_px), float(min_box_fraction) * crop_w)
    floor_h = max(float(min_box_px), float(min_box_fraction) * crop_h)
    # One ask per pass. In label mode there is exactly one pass, which is what
    # this always did; in instance mode a pass that lands on a spot already
    # spent re-asks with that spot named, because a deterministic model shown
    # the same image answers the same way until it is told otherwise.
    for _reselect in range(max(1, int(max_reselects)) if by_instance else 1):
        answer = ctx.tool(
            "vlm.query",
            image=source_crop,
            prompt=(
                f"Instruction: {instruction}\nInspect only {source_description}. "
                f"Destination labels not yet attempted are: {', '.join(remaining_labels)}. Select the "
                "most accessible visible unsorted object matching one of those labels. Occlusion order is "
                "the first priority: choose an object visibly resting on top of overlapping objects, and "
                "never reach underneath another tool merely because the lower tool has a narrower handle. "
                "Among equally topmost objects, prefer a clearly visible pinch section narrower than the "
                "parallel-jaw gripper's 8 cm opening and isolated from neighbours. Ignore objects "
                f"already in the destination.{_exhausted_clause(attempts, crop_x, crop_y, crop_w, crop_h)} "
                f"{identity} "
                "Return its tight pixel box and one foreground pixel clearly inside that same part, "
                "using coordinates in this cropped image. Reply exactly TARGET: <grasp part phrase>; "
                "LABEL: <one valid label>; BOX: <x1>,<y1>,<x2>,<y2>; PIXEL: <x>,<y>. "
                "If no unsorted object remains, reply DONE."
            ),
        )["text"]
        if str(answer).strip().upper().startswith("DONE"):
            return _finished(attempts)
        # The separator is whichever the model reached for: the prompt asks for
        # semicolons and some replies come back on newlines instead -- the same
        # decision, refused on punctuation.
        #
        # PIXEL is optional to parse, not to ask for. It is a point hint inside
        # the box the same answer already gave, so a reply that ends after the
        # box still says everything the segmenter needs -- and a reply that ends
        # in the middle of the pixel is the shape a truncated answer takes.
        # Falling back to the box centre costs a point hint that was always
        # inside the box anyway; refusing the whole decision costs the run.
        match = re.search(
            r"TARGET\s*:\s*(.+?)\s*[;\r\n]+\s*LABEL\s*:\s*(.+?)\s*[;\r\n]+\s*"
            r"BOX\s*:\s*([\d.]+)\s*,\s*([\d.]+)\s*,\s*([\d.]+)\s*,\s*([\d.]+)"
            r"(?:\s*[;\r\n]+\s*PIXEL\s*:\s*([\d.]+)\s*,\s*([\d.]+))?",
            str(answer),
            re.I,
        )
        if not match:
            raise ValueError(f"VLM returned an unparseable sorting decision: {answer!r}")
        target_name, requested_label = (part.strip() for part in match.groups()[:2])
        bx1, by1, bx2, by2 = (float(value) for value in match.groups()[2:6])
        raw_px, raw_py = match.groups()[6:8]
        px = float(raw_px) if raw_px is not None else (bx1 + bx2) / 2.0
        py = float(raw_py) if raw_py is not None else (by1 + by2) / 2.0
        normalized = {_normalize(label): region for label, region in zip(labels, regions, strict=True)}
        key = _normalize(requested_label)
        if key not in normalized:
            raise ValueError(f"VLM chose {requested_label!r}, not one of {labels!r}")
        if key in excluded:
            raise ValueError(f"VLM repeated excluded label {requested_label!r}")
        canonical_target = canonical.get(key)
        if canonical_target and target_name.lower() != canonical_target:
            raise ValueError(
                f"VLM returned TARGET {target_name!r} for LABEL {requested_label!r}; "
                f"expected {canonical_target!r}"
            )


        def _fits(box: tuple[float, float, float, float]) -> bool:
            """Whether this reading lands inside the crop with a usable box."""
            x1, y1, x2, y2 = box
            return (
                min(x1, x2) >= 0.0
                and max(x1, x2) <= crop_w - 1.0
                and min(y1, y2) >= 0.0
                and max(y1, y2) <= crop_h - 1.0
                and abs(x2 - x1) >= floor_w
                and abs(y2 - y1) >= floor_h
            )

        # The prompt asks for pixels "in this cropped image" and some models
        # answer in their own convention instead: coordinates normalised to
        # 0-1000 over the image shown, or the full frame's pixels. Clamping
        # those collapses the box below the 8 px gate and reports a model that
        # answered correctly as a failure. Read as candidates rather than
        # sniffed: the as-asked reading is tried first and only a reading that
        # fits replaces it, so an answer already in crop pixels is never
        # rescaled.
        scale_x, scale_y = crop_w / 1000.0, crop_h / 1000.0
        for box, point in (
            ((bx1, by1, bx2, by2), (px, py)),
            (
                (bx1 * scale_x, by1 * scale_y, bx2 * scale_x, by2 * scale_y),
                (px * scale_x, py * scale_y),
            ),
            (
                (bx1 - crop_x, by1 - crop_y, bx2 - crop_x, by2 - crop_y),
                (px - crop_x, py - crop_y),
            ),
        ):
            if _fits(box):
                (bx1, by1, bx2, by2), (px, py) = box, point
                break

        bx1, bx2 = sorted((max(0.0, min(bx1, crop_w - 1.0)), max(0.0, min(bx2, crop_w - 1.0))))
        by1, by2 = sorted((max(0.0, min(by1, crop_h - 1.0)), max(0.0, min(by2, crop_h - 1.0))))
        if bx2 - bx1 < floor_w or by2 - by1 < floor_h:
            raise ValueError(f"VLM grasp-part box is too small: {match.group(0)!r}")
        full_box = {"x1": bx1 + crop_x, "y1": by1 + crop_y, "x2": bx2 + crop_x, "y2": by2 + crop_y}
        full_px = max(full_box["x1"], min(px + crop_x, full_box["x2"]))
        full_py = max(full_box["y1"], min(py + crop_y, full_box["y2"]))
        segmented = ctx.tool(
            "sam3.segment_box",
            image=image,
            box=full_box,
            pixel_x=full_px,
            pixel_y=full_py,
            use_point=True,
        )
        if not segmented.get("masks"):
            raise ValueError("SAM3 returned no mask for the selected sorting object")
        candidates = []
        for index, candidate in enumerate(segmented["masks"]):
            candidate_mask = np.asarray(candidate)
            mask_area = int(np.count_nonzero(candidate_mask))
            # Segmentation can return high-confidence one-pixel fragments. Such
            # fragments cannot support an OBB or 3D feature fit and should not
            # outrank a slightly lower-scoring complete grasp part.
            if mask_area >= 100:
                candidates.append((float(segmented["scores"][index]), index))
        if not candidates:
            raise ValueError(f"SAM3 found {target_name!r}, but no instance lies in the source bin")
        mask_index = max(candidates)[1]
        mask = np.asarray(segmented["masks"][mask_index], dtype=np.uint8)
        cloud = ctx.tool(
            "geometry.mask_to_world_points",
            mask=mask,
            depth=camera["depth"],
            intrinsics=camera["intrinsics"],
            camera_pose=camera["pose"],
        )["points"]
        obb = ctx.tool("geometry.filter_and_compute_obb", points=cloud)["obb"]
        if by_instance:
            # Identify the object by where it is; see _matching_attempt.
            half_x = (full_box["x2"] - full_box["x1"]) / 2.0
            half_y = (full_box["y2"] - full_box["y1"]) / 2.0
            previous = _matching_attempt(attempts, full_px, full_py, half_x, half_y)
            settled = (
                _settled_region(obb, all_regions, settled_margin_m)
                if settled_margin_m > 0.0
                else None
            )
            if settled is not None:
                # Already delivered: retire the spot outright rather than
                # spending an attempt on it, and ask again.
                if previous is None:
                    attempts.append({"label": str(requested_label), "px": full_px, "py": full_py,
                                     "rx": half_x, "ry": half_y,
                                     "attempts": max(1, int(max_attempts_per_instance)),
                                     "reason": "settled"})
                else:
                    previous.update({"attempts": max(1, int(max_attempts_per_instance)),
                                     "reason": "settled"})
                continue
            if previous is not None and int(previous.get("attempts", 0)) >= max(
                1, int(max_attempts_per_instance)
            ):
                continue
            if previous is None:
                attempts.append({"label": str(requested_label), "px": full_px, "py": full_py,
                                 "rx": half_x, "ry": half_y, "attempts": 1, "reason": "tried"})
            else:
                previous["attempts"] = int(previous.get("attempts", 0)) + 1
                previous.update({"px": full_px, "py": full_py, "rx": half_x, "ry": half_y})
        found: Output = {
            "status": "found",
            "target_name": target_name,
            "target_label": str(normalized[key].get("label", requested_label)),
            "target_obb": obb,
            "target_mask": mask,
            "target_cloud": cloud,
            "destination_obb": normalized[key]["obb"],
        }
        if by_instance:
            found["attempted_json"] = json.dumps(attempts)
        return found
    # Every pass landed on something already attempted or already delivered.
    return _finished(attempts)
