"""Locate a thin shaft protruding from a mounting plane with DINO, SAM3 and RGB-D.

A detector proposes boxes for ``fixture_description``; only labels carrying
``label_keyword`` with an elongated box survive. SAM3 refines each box, but
often keeps the thick mount and misses the thin front half, so the complete
protrusion is recovered from calibrated depth inside the detector's tight
box: the mounting plane is the far-X plane of the ROI and the shaft is the
connected foreground in front of it (protruding toward decreasing world X).
Physical gates on shaft length and transverse extent reject look-alikes on
the table. The result is the distal tip, the plane-to-tip axis, the usable
shaft length and the outer radius the mating skill needs.
"""

from typing import Any, TypedDict

import numpy as np
from gap import NodeContext


class Output(TypedDict):
    route: str
    fixture_kind: str
    fixture_obb: dict[str, Any]
    fixture_mask: np.ndarray
    fixture_cloud: dict[str, Any]
    shaft_tip: dict[str, float]
    fixture_axis: dict[str, float]
    fixture_feature: dict[str, Any]


def _camera(observation: dict[str, Any], name: str) -> dict[str, Any]:
    cameras = observation.get("cameras") or []
    if isinstance(cameras, dict):
        cameras = list(cameras.values())
    for camera in cameras:
        if camera.get("name") == name:
            return camera
    raise ValueError(f"observation has no camera named {name!r}")


def _backproject(ctx: NodeContext, camera: dict[str, Any], mask: Any) -> dict[str, Any]:
    return ctx.tool(
        "geometry.mask_to_world_points",
        mask=np.asarray(mask, dtype=np.uint8),
        depth=camera["depth"],
        intrinsics=camera["intrinsics"],
        camera_pose=camera["pose"],
    )["points"]


def _box_mask(shape: tuple[int, ...], box: dict[str, Any]) -> np.ndarray:
    mask = np.zeros(shape[:2], dtype=np.uint8)
    x1 = max(0, int(np.floor(float(box["x1"]))))
    y1 = max(0, int(np.floor(float(box["y1"]))))
    x2 = min(shape[1], int(np.ceil(float(box["x2"]))) + 1)
    y2 = min(shape[0], int(np.ceil(float(box["y2"]))) + 1)
    mask[y1:y2, x1:x2] = 255
    return mask


def _feature(
    kind: str,
    tip: dict[str, float],
    axis: dict[str, float],
    radius_outer: float = 0.0,
    seating_margin: float = 0.0,
) -> dict[str, Any]:
    feature = {
        "kind": kind,
        "pose": {"position": dict(tip), "rotation": {"w": 1.0, "x": 0.0, "y": 0.0, "z": 0.0}},
        "axis": dict(axis),
        "radius_outer": float(radius_outer),
        "confidence": 1.0,
    }
    if seating_margin > 0.0:
        feature["seating_margin"] = float(seating_margin)
    return feature


def _not_found() -> Output:
    zero = {"x": 0.0, "y": 0.0, "z": 0.0}
    return {
        "route": "not_found",
        "fixture_kind": "",
        "fixture_obb": {
            "center": dict(zero),
            "extent": dict(zero),
            "orientation": {"w": 1.0, "x": 0.0, "y": 0.0, "z": 0.0},
        },
        "fixture_mask": np.zeros((1, 1), dtype=np.uint8),
        "fixture_cloud": {"points": np.empty((0, 3), dtype=np.float32)},
        "shaft_tip": dict(zero),
        "fixture_axis": dict(zero),
        "fixture_feature": {},
    }


def run(
    ctx: NodeContext,
    observation: dict[str, Any],
    fixture_description: str,
    feature_type: str = "shaft",
    label_keyword: str = "hook",
    aspect_min: float = 2.5,
    shaft_length_min: float = 0.025,
    shaft_length_max: float = 0.15,
    transverse_max: float = 0.04,
    seating_margin: float = 0.015,
    required: bool = True,
    camera_name: str = "overhead",
) -> Output:
    """Find the shaft; raise when ``required`` and nothing passes the gates.

    With ``required=False`` a miss returns ``route="not_found"`` and empty
    outputs so a graph can fall back to another fixture rule.
    """
    camera = _camera(observation, camera_name)
    detections = ctx.tool(
        "grounding-dino.detect",
        image=camera["rgb"],
        query=fixture_description,
        box_threshold=0.18,
        text_threshold=0.18,
    )["detections"]

    keyword = str(label_keyword).strip().lower()
    candidates = []
    for detection in detections[:8]:
        label = str(detection.get("label", "")).strip().lower()
        # The detector can return the whole mounting board with a higher
        # score than the requested small shaft. A large scene box must never
        # enter the geometric fitter merely because it is elongated. Accept
        # phrases containing the keyword; the geometric gates below handle
        # shaft-like false positives.
        if keyword and keyword not in label:
            continue
        box = detection["box"]
        width = max(1.0, float(box["x2"] - box["x1"]))
        height = max(1.0, float(box["y2"] - box["y1"]))
        if max(width / height, height / width) < aspect_min:
            continue
        segmented = ctx.tool("sam3.segment_box", image=camera["rgb"], box=box)
        if not segmented["masks"] or float(segmented["scores"][0]) < 0.50:
            continue

        # Recover the complete protrusion from calibrated depth inside the
        # tight box: the mounting plane is the far-X plane and the shaft is
        # the connected foreground in front of it.
        roi_mask = _box_mask(camera["depth"].shape, box)
        roi_cloud = _backproject(ctx, camera, roi_mask)
        roi_points = np.asarray(roi_cloud["points"], dtype=np.float64).reshape(-1, 3)
        if len(roi_points) < 40:
            continue
        board_x = float(np.percentile(roi_points[:, 0], 80))
        points = roi_points[roi_points[:, 0] < board_x - 0.003]
        if len(points) < 20 or float(np.ptp(points[:, 0])) < 0.025:
            continue
        shaft_length = float(board_x - np.percentile(points[:, 0], 5))
        transverse_extent = min(float(np.ptp(points[:, 1])), float(np.ptp(points[:, 2])))
        # Physical plausibility gates for a small protruding shaft. They
        # reject an elongated object on the table that the detector labelled
        # as the fixture, while remaining independent of the fixture's pose.
        if not (
            shaft_length_min <= shaft_length <= shaft_length_max
            and transverse_extent <= transverse_max
        ):
            continue
        cloud = {"points": points.astype(np.float32)}
        candidates.append((float(detection.get("score", 0.0)), board_x, roi_mask, cloud, points))

    if not candidates:
        if required:
            raise ValueError(
                f"no thin SAM3-refined {label_keyword or 'shaft'} candidate was found for "
                f"{fixture_description!r}"
            )
        return _not_found()
    _, board_x, mask, cloud, points = max(candidates, key=lambda item: item[0])
    obb = ctx.tool("geometry.filter_and_compute_obb", points=cloud)["obb"]

    # The shaft projects from the plane toward decreasing world X. The very
    # foremost points belong to a rounded or upturned nose, so their Z is
    # systematically off the shank centreline. Recover distal X from those
    # points, but transverse Y/Z from the straight 10--30% section behind it.
    cutoff = np.percentile(points[:, 0], 5)
    tip_points = points[points[:, 0] <= cutoff]
    shank_lo, shank_hi = np.percentile(points[:, 0], [10, 30])
    shank_points = points[(points[:, 0] >= shank_lo) & (points[:, 0] <= shank_hi)]
    if len(shank_points) < 8:
        shank_points = tip_points
    tip = np.array(
        [
            float(np.median(tip_points[:, 0])),
            float(np.median(shank_points[:, 1])),
            float(np.median(shank_points[:, 2])),
        ],
        dtype=np.float64,
    )
    # Direction is observed from the local mounting plane to the distal tip;
    # nothing assumes a fixed world mating direction.
    board_point = np.array([board_x, tip[1], tip[2]], dtype=np.float64)
    axis = tip - board_point
    axis /= np.linalg.norm(axis)
    tip_dict = {"x": float(tip[0]), "y": float(tip[1]), "z": float(tip[2])}
    axis_dict = {"x": float(axis[0]), "y": float(axis[1]), "z": float(axis[2])}
    # Perception reports the physical distal tip and the observed usable
    # shaft length. The mating skill chooses how far a loop must cross and
    # travel along the shaft.
    shaft_length = float(np.linalg.norm(board_point - tip))
    fixture_feature = _feature(
        feature_type,
        tip_dict,
        axis_dict,
        radius_outer=max(0.001, min(float(obb["extent"]["y"]), float(obb["extent"]["z"]))),
        seating_margin=seating_margin,
    )
    fixture_feature["usable_length"] = shaft_length
    return {
        "route": "found",
        "fixture_kind": str(feature_type),
        "fixture_obb": obb,
        "fixture_mask": mask,
        "fixture_cloud": cloud,
        "shaft_tip": tip_dict,
        "fixture_axis": axis_dict,
        "fixture_feature": fixture_feature,
    }
