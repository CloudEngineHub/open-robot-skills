"""Discover a labelled 2x2 destination layout from one calibrated RGB-D view."""

import json
import logging
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
from sorting_cv import box as _box  # noqa: E402
from sorting_cv import camera as _camera  # noqa: E402


class Output(TypedDict):
    layout_json: str


logger = logging.getLogger(__name__)

_POSITIONS = ("top-left", "top-right", "bottom-left", "bottom-right")


def _specific_container_box(
    detections: list[dict[str, Any]], image_shape: tuple[int, ...]
) -> dict[str, Any]:
    """Prefer the tight destination container over a higher-scoring parent scene box."""
    # The detector often returns a compact box around the four printed labels
    # as well as an oversized "destination bin" box spanning both source and
    # target. The former is the reliable footprint of this labelled 2x2
    # container.
    image_height, image_width = image_shape[:2]
    # Smaller, resolution-independent gates: about 115x96 for 640x480.
    label_min_width = 0.18 * image_width
    label_min_height = 0.20 * image_height
    label_groups = []
    for detection in detections:
        box = _box(detection)
        width = float(box["x2"]) - float(box["x1"])
        height = float(box["y2"]) - float(box["y1"])
        label = str(detection.get("label", "")).lower()
        if width >= label_min_width and height >= label_min_height and "printed label" in label:
            label_groups.append((width * height, box))
    if label_groups:
        return min(label_groups, key=lambda item: item[0])[1]

    container_min_width = 0.14 * image_width
    container_min_height = 0.16 * image_height
    plausible = []
    for detection in detections:
        box = _box(detection)
        width = float(box["x2"]) - float(box["x1"])
        height = float(box["y2"]) - float(box["y1"])
        label = str(detection.get("label", "")).lower()
        if (
            width >= container_min_width
            and height >= container_min_height
            and ("destination" in label or "compartment" in label)
        ):
            plausible.append((width * height, box))
    if plausible:
        return min(plausible, key=lambda item: item[0])[1]
    return _box(max(detections, key=lambda item: float(item.get("score", 0.0))))


def _read_labels(
    ctx: NodeContext, image: Any, layout_description: str, attempts: int = 1
) -> dict[str, str]:
    """The four printed labels, re-asking up to *attempts* times.

    **Why a retry is worth having.** The failure this absorbs is not a hard one:
    the model answers with the structure intact and simply stops before the last
    label --

        LAYOUT: top-left=SOCKET HEAD SCREW M6X60; top-right=WIRE CUTTER WALL
        MOUNT; bottom-left=MULTIMETER; bottom-right

    -- and the node then raises, the subgraph routes ambiguous, and the episode
    aborts having done nothing. It is not a token budget (the reply is about a
    hundred characters against a budget of a thousand tokens); it is a draw that
    came out short, and the next draw usually does not.

    It matters more the longer the labels are. A layout labelled with one or two
    short words is read in one go; a layout labelled `socket head screw m6x60`
    and `wire cutter wall mount` is where the read gets hard -- so the tier that
    most needs the read to be robust was the one least tolerant of it.

    **Default 1, which is the single ask this always did.** The retry is
    something a graph asks for, because a re-ask costs a model call and only the
    caller knows whether its labels are the hard kind. Note the asymmetry this
    closes: `select_pair`, in the same bundle and often the same graph, has
    always retried an unparseable reply three times.
    """
    answer = ""
    for attempt in range(max(1, int(attempts))):
        answer = ctx.tool(
            "vlm.query",
            image=image,
            prompt=(
                f"Locate {layout_description}. Read the printed label in each of its four "
                "compartments. Use IMAGE coordinates, not world directions. Reply exactly: "
                "LAYOUT: top-left=<label>; top-right=<label>; "
                "bottom-left=<label>; bottom-right=<label>."
            ),
        )["text"]
        found: dict[str, str] = {}
        for position in _POSITIONS:
            match = re.search(rf"{position}\s*=\s*([^;\n.]+)", str(answer), re.I)
            if match:
                found[position] = re.sub(r"\s+", " ", match.group(1).strip()).lower()
        if len(found) == 4 and len(set(found.values())) == 4:
            return found
        if attempt + 1 < max(1, int(attempts)):
            logger.warning(
                "[discover_regions] re-asking for the layout labels: %r", answer
            )
    raise ValueError(f"could not read four unique destination labels: {answer!r}")


def _region_obb(
    ctx: NodeContext, camera: dict[str, Any], bounds: dict[str, Any], position: str
) -> dict[str, Any]:
    x1, y1, x2, y2 = (int(round(float(bounds[key]))) for key in ("x1", "y1", "x2", "y2"))
    inset_x = max(3, int(0.08 * (x2 - x1)))
    inset_y = max(3, int(0.08 * (y2 - y1)))
    middle_x, middle_y = (x1 + x2) // 2, (y1 + y2) // 2
    left, top = position.endswith("left"), position.startswith("top")
    xa, xb = (x1 + inset_x, middle_x - inset_x) if left else (middle_x + inset_x, x2 - inset_x)
    ya, yb = (y1 + inset_y, middle_y - inset_y) if top else (middle_y + inset_y, y2 - inset_y)
    mask = np.zeros(np.asarray(camera["depth"]).shape[:2], dtype=np.uint8)
    mask[max(0, ya) : max(0, yb), max(0, xa) : max(0, xb)] = 255
    cloud = ctx.tool(
        "geometry.mask_to_world_points",
        mask=mask,
        depth=camera["depth"],
        intrinsics=camera["intrinsics"],
        camera_pose=camera["pose"],
    )["points"]
    points = np.asarray(cloud["points"], dtype=np.float64).reshape(-1, 3)
    points = points[np.isfinite(points).all(axis=1)]
    if len(points) < 20:
        raise ValueError(f"too few depth points for destination region {position}")
    floor_cut = np.quantile(points[:, 2], 0.45)
    floor = points[points[:, 2] <= floor_cut + 0.004]
    if len(floor) < 10:
        floor = points
    low, high = np.quantile(floor, [0.10, 0.90], axis=0)
    center = 0.5 * (low + high)
    extent = np.maximum(0.5 * (high - low), [0.025, 0.025, 0.002])
    return {
        "center": dict(zip(("x", "y", "z"), map(float, center), strict=True)),
        "extent": dict(zip(("x", "y", "z"), map(float, extent), strict=True)),
        "orientation": {"w": 1.0, "x": 0.0, "y": 0.0, "z": 0.0},
    }


def run(
    ctx: NodeContext,
    observation: dict[str, Any],
    layout_description: str,
    camera_name: str = "overhead",
    max_read_attempts: int = 1,
) -> Output:
    camera = _camera(observation, camera_name)
    image = camera["rgb"]
    labels = _read_labels(ctx, image, layout_description, max_read_attempts)
    detections = (
        ctx.tool(
            "grounding-dino.detect", image=image, query=layout_description, box_threshold=0.10
        ).get("detections")
        or []
    )
    if not detections:
        raise ValueError(f"could not localize {layout_description!r}")
    bounds = _specific_container_box(detections, np.asarray(image).shape)
    regions = [
        {
            "label": label,
            "image_position": position,
            "obb": _region_obb(ctx, camera, bounds, position),
        }
        for position, label in labels.items()
    ]
    return {"layout_json": json.dumps(regions)}
