"""Perceive one remaining source object and associate its labelled region.

The identity table -- what each label looks like and what its graspable part
is called -- is supplied by the graph through ``identity_hints`` and
``canonical_targets``; the script itself knows no object.
"""

import json
import re
from typing import Any, TypedDict

import numpy as np
from gap import NodeContext
from PIL import Image, ImageDraw


class Output(TypedDict, total=False):
    status: str
    target_name: str
    target_label: str
    target_obb: dict[str, Any]
    target_mask: np.ndarray
    target_cloud: dict[str, Any]
    destination_obb: dict[str, Any]


_DEFAULT_IDENTITY_HINT = "Identify the narrow graspable handle or graspable body."


def _camera(observation: dict[str, Any], name: str) -> dict[str, Any]:
    cameras = observation.get("cameras") or []
    if isinstance(cameras, dict):
        cameras = list(cameras.values())
    return next(camera for camera in cameras if camera.get("name") == name)


def _box(detection: dict[str, Any]) -> dict[str, Any]:
    return detection.get("box") or detection.get("bbox") or detection


def _bounds(box: dict[str, Any]) -> tuple[float, float, float, float]:
    return tuple(float(box[key]) for key in ("x1", "y1", "x2", "y2"))


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


def _best_box(
    ctx: NodeContext, image: Any, query: str, inside: tuple[float, ...] | None = None
) -> dict[str, Any]:
    detections = (
        ctx.tool("grounding-dino.detect", image=image, query=query, box_threshold=0.10).get(
            "detections"
        )
        or []
    )
    candidates = []
    for detection in detections:
        box = _box(detection)
        x1, y1, x2, y2 = _bounds(box)
        cx, cy = 0.5 * (x1 + x2), 0.5 * (y1 + y2)
        if inside is not None and not (
            inside[0] <= cx <= inside[2] and inside[1] <= cy <= inside[3]
        ):
            continue
        candidates.append((float(detection.get("score", 0.0)), box))
    if not candidates:
        raise ValueError(f"could not localize {query!r}")
    candidates.sort(key=lambda item: item[0], reverse=True)
    candidates = candidates[:6]
    crops = []
    height, width = np.asarray(image).shape[:2]
    overview = Image.fromarray(np.asarray(image, dtype=np.uint8)).convert("RGB")
    draw = ImageDraw.Draw(overview)
    for index, (_, box) in enumerate(candidates):
        x1, y1, x2, y2 = _bounds(box)
        draw.rectangle((x1, y1, x2, y2), outline=(255, 255, 0), width=3)
        draw.text(
            (x1 + 3, y1 + 3),
            str(index),
            fill=(255, 0, 0),
            stroke_width=2,
            stroke_fill=(255, 255, 255),
        )
        xa, ya = max(0, int(x1)), max(0, int(y1))
        xb, yb = min(width, int(np.ceil(x2))), min(height, int(np.ceil(y2)))
        crops.append(np.asarray(image)[ya:yb, xa:xb])
    answer = ctx.tool(
        "vlm.query",
        images=[np.asarray(overview), *crops],
        prompt=(
            f"The first image is the full scene with candidate boxes numbered 0 through "
            f"{len(crops) - 1}; the remaining images are those crops in the same order. "
            f"Which numbered box most specifically encloses the complete {query}, rather than "
            "another tool, a printed label, or a multi-object group? Prefer a tight complete-object "
            "box over a larger group box. Reply exactly INDEX: <number>."
        ),
    )["text"]
    match = re.search(r"INDEX\s*:\s*(\d+)", str(answer), re.I)
    if not match or int(match.group(1)) >= len(candidates):
        raise ValueError(f"VLM did not select a valid {query!r} candidate: {answer!r}")
    return candidates[int(match.group(1))][1]


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
) -> Output:
    """Select one unsorted object and its destination region.

    ``identity_hints`` is the graph's visual-identity text for the labels in
    play (what each looks like and what TARGET phrase to answer); when empty
    a generic "narrow graspable handle or body" cue is used.
    ``canonical_targets`` is a JSON object mapping a label to the exact
    TARGET phrase the model must return for it; a mismatch is an error only
    for labels in the table. ``allowed_labels`` (comma-separated or JSON
    list) restricts the layout to those labels before anything is asked.
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
            f"already in the destination. {identity} "
            "Return its tight pixel box and one foreground pixel clearly inside that same part, "
            "using coordinates in this cropped image. Reply exactly TARGET: <grasp part phrase>; "
            "LABEL: <one valid label>; BOX: <x1>,<y1>,<x2>,<y2>; PIXEL: <x>,<y>. "
            "If no unsorted object remains, reply DONE."
        ),
    )["text"]
    if str(answer).strip().upper().startswith("DONE"):
        return _empty_result("finished")
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

    crop_h, crop_w = source_crop.shape[:2]

    def _fits(box: tuple[float, float, float, float]) -> bool:
        """Whether this reading lands inside the crop with a usable box."""
        x1, y1, x2, y2 = box
        return (
            min(x1, x2) >= 0.0
            and max(x1, x2) <= crop_w - 1.0
            and min(y1, y2) >= 0.0
            and max(y1, y2) <= crop_h - 1.0
            and abs(x2 - x1) >= 8.0
            and abs(y2 - y1) >= 8.0
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
    if bx2 - bx1 < 8.0 or by2 - by1 < 8.0:
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
    return {
        "status": "found",
        "target_name": target_name,
        "target_label": str(normalized[key].get("label", requested_label)),
        "target_obb": obb,
        "target_mask": mask,
        "target_cloud": cloud,
        "destination_obb": normalized[key]["obb"],
    }
