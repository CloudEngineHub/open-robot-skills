"""Locate a profile-declared fixture feature with semantic RGB-D geometry.

Three strategies, each a declaration the calling graph owns through
``fixture_profiles`` (keyed by ``source_kind``) rather than an object-name
branch here:

- ``depth_support``: a support rail read from calibrated depth inside a
  declared workspace box -- the foreground in front of the board plane, its
  tip at the far percentile, crossed at a declared clearance above its top.
- ``parent_inferred_aperture``: a container located by text, its opening
  inferred from the parent as a whole (its horizontal centre and its dense
  upper depth band), because a second segmentation of "open interior" locks
  onto the front wall or an inserted tool once one is in.
- ``thin_projection``: a pegboard hook proposed by Grounding DINO, refined by
  SAM3 inside the box, its thin protrusion recovered from depth in front of
  the board plane, with physical-plausibility gates on length and thickness.

``correspondence_anchor`` is the coordinate along the scene's correspondence
axis that ``perceiving-relational-correspondences`` chose for this source; a
strategy uses it to pick the matching fixture (the hook nearest it), to
narrow the depth window, or to give repeated placements distinct landing
points inside a shared container. It is an anchor, not a y: the axis is the
graph's declaration.
"""

from typing import Any, TypedDict

import numpy as np
from gap import NodeContext


class Output(TypedDict):
    fixture_kind: str
    fixture_obb: dict[str, Any]
    fixture_mask: np.ndarray
    fixture_cloud: dict[str, Any]
    hook_tip: dict[str, float]
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


def _points(cloud: dict[str, Any]) -> np.ndarray:
    return np.asarray(cloud["points"], dtype=np.float64).reshape(-1, 3)


def _box_mask(shape, box) -> np.ndarray:
    mask = np.zeros(shape[:2], dtype=np.uint8)
    x1 = max(0, int(np.floor(float(box["x1"]))))
    y1 = max(0, int(np.floor(float(box["y1"]))))
    x2 = min(shape[1], int(np.ceil(float(box["x2"]))) + 1)
    y2 = min(shape[0], int(np.ceil(float(box["y2"]))) + 1)
    mask[y1:y2, x1:x2] = 255
    return mask


def _feature(kind, tip, axis, radius_outer=0.0, seating_margin=0.0) -> dict[str, Any]:
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


def _depth_support_points(workcell: np.ndarray, profile: dict[str, Any]) -> np.ndarray:
    """Foreground support points in front of the board plane, or a quality error."""
    if len(workcell) < 100:
        raise ValueError("too few calibrated board-region depth points")
    board_x = float(np.percentile(workcell[:, 0], float(profile.get("board_percentile", 85))))
    points = workcell[workcell[:, 0] < board_x - float(profile.get("foreground_margin", 0.004))]
    if len(points) < 30:
        raise ValueError("support foreground was not recovered")
    return points


def _with_mating_profile(feature: dict[str, Any], profile: dict[str, Any]) -> dict[str, Any]:
    if profile.get("mating_profile"):
        feature["mating_profile"] = dict(profile["mating_profile"])
    return feature


def _depth_support(ctx, camera, profile, feature_type, correspondence_anchor) -> Output:
    full_mask = np.full(camera["depth"].shape, 255, dtype=np.uint8)
    all_points = _points(_backproject(ctx, camera, full_mask))
    bounds = profile["workspace"]
    unanchored = all_points[
        (all_points[:, 0] > float(bounds["x_min"])) & (all_points[:, 0] < float(bounds["x_max"]))
        & (all_points[:, 1] > float(bounds["y_min"])) & (all_points[:, 1] < float(bounds["y_max"]))
        & (all_points[:, 2] > float(bounds["z_min"])) & (all_points[:, 2] < float(bounds["z_max"]))
    ]
    axis_index = {"x": 0, "y": 1, "z": 2}[str(profile.get("correspondence_axis", "y"))]
    workcell = unanchored
    if correspondence_anchor is not None:
        workcell = workcell[
            np.abs(workcell[:, axis_index] - float(correspondence_anchor))
            < float(profile.get("anchor_tolerance", 0.065))
        ]
    try:
        points = _depth_support_points(workcell, profile)
    except ValueError:
        # A source may sit farther from the centreline than its destination.
        # When the narrow anchor window fails, optionally fall back to the
        # declared workspace partition; a successful anchored estimate is
        # never changed.
        if correspondence_anchor is None or not bool(profile.get("anchor_fallback_partition", False)):
            raise
        split = float(profile.get("partition_split", 0.0))
        margin = max(0.0, float(profile.get("partition_margin", 0.0)))
        if float(correspondence_anchor) >= split:
            fallback = unanchored[unanchored[:, axis_index] >= split + margin]
        else:
            fallback = unanchored[unanchored[:, axis_index] <= split - margin]
        points = _depth_support_points(fallback, profile)
    cutoff = np.percentile(points[:, 0], float(profile.get("tip_percentile", 3)))
    tip = np.median(points[points[:, 0] <= cutoff], axis=0)
    # Cross the support at a height clear of its top edge, then let the
    # feature-mating skill lower the handles onto it after the crossing.
    tip[0] += float(profile.get("tip_axis_offset", 0.003))
    tip[2] = float(np.percentile(points[:, 2], 99)) + float(profile.get("top_clearance", 0.018))
    cloud = {"points": points.astype(np.float32)}
    obb = ctx.tool("geometry.filter_and_compute_obb", points=cloud)["obb"]
    tip_dict = {"x": float(tip[0]), "y": float(tip[1]), "z": float(tip[2])}
    axis_dict = dict(zip(("x", "y", "z"), map(float, profile.get("axis", [-1.0, 0.0, 0.0])), strict=True))
    feature = _feature(feature_type, tip_dict, axis_dict, seating_margin=float(profile.get("seating_margin", 0.015)))
    return {
        "fixture_kind": "support", "fixture_obb": obb, "fixture_mask": full_mask, "fixture_cloud": cloud,
        "hook_tip": tip_dict, "fixture_axis": axis_dict, "fixture_feature": _with_mating_profile(feature, profile),
    }


def _parent_inferred_aperture(ctx, camera, profile, feature_type, fixture_description, correspondence_anchor) -> Output:
    parent = ctx.tool(
        "sam3.segment_text", image=camera["rgb"], query=fixture_description,
        max_results=int(profile.get("max_results", 3)),
    )
    if not parent.get("masks"):
        raise ValueError("declared parent fixture was not visible")
    # Text segmentation occasionally ranks the whole pegboard above the small
    # box, especially after the first placed tool changes the scene: keep only
    # candidates compact in 3-D. These are scale gates, not position gates.
    span_max = np.asarray(profile.get("span_max", [0.20, 0.22, 0.30]), dtype=float)
    scores = list(parent.get("scores") or [])
    candidates = []
    for index, candidate_mask in enumerate(parent["masks"]):
        candidate_cloud = _backproject(ctx, camera, candidate_mask)
        points = _points(candidate_cloud)
        if len(points) < 40:
            continue
        lower, upper = np.percentile(points, [1, 99], axis=0)
        if np.any(upper - lower > span_max):
            continue
        candidates.append((float(scores[index]) if index < len(scores) else 0.0, candidate_mask, candidate_cloud))
    if not candidates:
        raise ValueError("parent fixture candidates were not compact in 3-D")
    _, fixture_mask, cloud = max(candidates, key=lambda item: item[0])
    obb = ctx.tool("geometry.filter_and_compute_obb", points=cloud)["obb"]
    parent_points = _points(cloud)
    # The opening is inferred from the box as a whole: its horizontal centre
    # locates it and its dense upper depth band locates the rim (sparse
    # still-higher points are the mounting hooks).
    top_lo, top_hi = np.percentile(parent_points[:, 2], profile.get("rim_percentiles", [85, 90]))
    top_band = parent_points[(parent_points[:, 2] >= top_lo) & (parent_points[:, 2] <= top_hi)]
    if len(top_band) < 20:
        raise ValueError("container parent cloud did not expose a stable top rim")
    axis_name = str(profile.get("correspondence_axis", "y"))
    axis_index = {"x": 0, "y": 1, "z": 2}[axis_name]
    center_along = float(obb["center"][axis_name])
    target_along = center_along
    if correspondence_anchor is not None:
        # Preserve the source's ordering at a much smaller scale inside a
        # shared container, so repeated placements land at distinct points.
        lo, hi = np.percentile(parent_points[:, axis_index], profile.get("lateral_percentiles", [2, 98]))
        lateral = min(float(profile.get("lateral_offset_max", 0.012)),
                      float(profile.get("lateral_offset_fraction", 0.25)) * 0.5 * float(hi - lo))
        if abs(float(correspondence_anchor) - center_along) > 1e-4:
            target_along += float(np.sign(float(correspondence_anchor) - center_along)) * lateral
    tip_dict = {k: float(obb["center"][k]) for k in ("x", "y", "z")}
    tip_dict[axis_name] = target_along
    tip_dict["z"] = float(np.median(top_band[:, 2]))
    axis_dict = {"x": 0.0, "y": 0.0, "z": -1.0}
    feature = _feature(feature_type, tip_dict, axis_dict)
    feature["radius_inner"] = float(profile.get("radius_inner", 0.025))
    # A container drop needs the tip just inside the rim before release, or a
    # long object can lean back out; the guide depth derives from the opening.
    feature["insertion_depth"] = float(profile.get("insertion_depth_radius_scale", 0.8)) * feature["radius_inner"]
    feature["description"] = str(profile.get("feature_description", "open interior"))
    return {
        "fixture_kind": "container", "fixture_obb": obb, "fixture_mask": fixture_mask, "fixture_cloud": cloud,
        "hook_tip": tip_dict, "fixture_axis": axis_dict, "fixture_feature": _with_mating_profile(feature, profile),
    }


def _thin_projection(ctx, camera, profile, feature_type, fixture_description, correspondence_anchor) -> Output:
    detections = ctx.tool(
        "grounding-dino.detect", image=camera["rgb"], query=fixture_description,
        box_threshold=float(profile.get("box_threshold", 0.18)),
        text_threshold=float(profile.get("text_threshold", 0.18)),
    )["detections"]
    label_token = str(profile.get("required_label_token", "")).strip().lower()
    length_bounds = profile.get("length_bounds", [0.025, 0.15])
    candidates = []
    for detection in detections[:8]:
        # Grounding DINO can return the whole pegboard above the small hook;
        # a label that does not name a hook never reaches the geometric gates.
        if label_token and label_token not in str(detection.get("label", "")).strip().lower():
            continue
        box = detection["box"]
        width = max(1.0, float(box["x2"] - box["x1"]))
        height = max(1.0, float(box["y2"] - box["y1"]))
        if max(width / height, height / width) < float(profile.get("minimum_aspect_ratio", 2.5)):
            continue
        segmented = ctx.tool("sam3.segment_box", image=camera["rgb"], box=box)
        if not segmented["masks"] or float(segmented["scores"][0]) < float(profile.get("minimum_mask_score", 0.50)):
            continue
        # SAM often captures the thick mount and misses the thin front half:
        # recover the whole protrusion from depth inside the tight box, the
        # board being the far-X plane and the hook the foreground before it.
        roi_mask = _box_mask(camera["depth"].shape, box)
        roi_points = _points(_backproject(ctx, camera, roi_mask))
        if len(roi_points) < 40:
            continue
        board_x = float(np.percentile(roi_points[:, 0], 80))
        points = roi_points[roi_points[:, 0] < board_x - float(profile.get("foreground_margin", 0.003))]
        if len(points) < 20 or float(np.ptp(points[:, 0])) < 0.025:
            continue
        shaft_length = float(board_x - np.percentile(points[:, 0], 5))
        transverse = min(float(np.ptp(points[:, 1])), float(np.ptp(points[:, 2])))
        if not (float(length_bounds[0]) <= shaft_length <= float(length_bounds[1])
                and transverse <= float(profile.get("transverse_extent_max", 0.04))):
            continue
        candidates.append((float(detection.get("score", 0.0)), board_x, roi_mask,
                           {"points": points.astype(np.float32)}, points))
    if not candidates:
        raise ValueError("no thin SAM3-refined hook candidate was found")
    axis_index = {"x": 0, "y": 1, "z": 2}[str(profile.get("correspondence_axis", "y"))]
    if correspondence_anchor is None:
        _, board_x, mask, cloud, points = max(candidates, key=lambda item: item[0])
    else:
        _, board_x, mask, cloud, points = min(
            candidates, key=lambda item: abs(float(np.median(item[4][:, axis_index])) - float(correspondence_anchor))
        )
    obb = ctx.tool("geometry.filter_and_compute_obb", points=cloud)["obb"]
    # The foremost points are the rounded nose, above the shank centreline:
    # distal X comes from them, transverse Y/Z from the straight 10--30 %
    # section immediately behind.
    tip_points = points[points[:, 0] <= np.percentile(points[:, 0], 5)]
    shank_lo, shank_hi = np.percentile(points[:, 0], [10, 30])
    shank = points[(points[:, 0] >= shank_lo) & (points[:, 0] <= shank_hi)]
    if len(shank) < 8:
        shank = tip_points
    radius_outer = max(0.001, min(float(obb["extent"]["y"]), float(obb["extent"]["z"])))
    z_low, z_high = np.percentile(shank[:, 2], [2, 98])
    if float(z_high - z_low) > 0.008:
        # Both faces of the rectangular arm are visible: move from the
        # silhouette midpoint to the arm centre by half the excess span.
        shank_z = 0.5 * float(z_low + z_high) + max(
            0.5 * float((z_high - z_low) - 2.0 * radius_outer), 3.0 * radius_outer
        )
    else:
        # Only the lower return face is visible: the arm centre lies two
        # transverse half-thicknesses above the depth median.
        shank_z = float(np.median(shank[:, 2]) + 2.0 * radius_outer)
    tip = np.array([float(np.median(tip_points[:, 0])), float(np.median(shank[:, 1])), shank_z])
    # Direction is observed from the local board plane to the distal tip.
    board_point = np.array([board_x, tip[1], tip[2]], dtype=np.float64)
    axis = tip - board_point
    axis /= np.linalg.norm(axis)
    tip_dict = {"x": float(tip[0]), "y": float(tip[1]), "z": float(tip[2])}
    axis_dict = {"x": float(axis[0]), "y": float(axis[1]), "z": float(axis[2])}
    feature = _feature(feature_type, tip_dict, axis_dict, radius_outer=radius_outer,
                       seating_margin=float(profile.get("seating_margin", 0.015)))
    # Perception reports the tip and the usable shaft length; the mating skill
    # decides how far a loop crosses and travels.
    feature["usable_length"] = float(np.linalg.norm(board_point - tip))
    return {
        "fixture_kind": "hook", "fixture_obb": obb, "fixture_mask": mask, "fixture_cloud": cloud,
        "hook_tip": tip_dict, "fixture_axis": axis_dict, "fixture_feature": _with_mating_profile(feature, profile),
    }


def run(
    ctx: NodeContext,
    observation: dict[str, Any],
    target_kind: str = "",
    fixture_description: str = "fixture",
    feature_type: str = "shaft",
    correspondence_anchor: float | None = None,
    fixture_profiles: list[dict[str, Any]] | None = None,
    camera_name: str = "overhead",
) -> Output:
    camera = _camera(observation, camera_name)
    profiles = {str(profile["source_kind"]): profile for profile in (fixture_profiles or [])}
    if target_kind not in profiles:
        raise ValueError(f"no fixture profile declared for source kind {target_kind!r}")
    profile = profiles[target_kind]
    strategy = str(profile["strategy"])
    fixture_description = str(profile.get("fixture_description", fixture_description))
    feature_type = str(profile.get("feature_type", feature_type))
    if strategy == "depth_support":
        return _depth_support(ctx, camera, profile, feature_type, correspondence_anchor)
    if strategy == "parent_inferred_aperture":
        return _parent_inferred_aperture(ctx, camera, profile, feature_type, fixture_description, correspondence_anchor)
    if strategy == "thin_projection":
        return _thin_projection(ctx, camera, profile, feature_type, fixture_description, correspondence_anchor)
    raise ValueError(f"unsupported fixture feature strategy {strategy!r}")
