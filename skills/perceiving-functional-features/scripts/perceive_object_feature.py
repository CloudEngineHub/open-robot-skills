"""Localize one of several described objects and fit its planar functional feature.

The generic core of language-driven object-plus-feature perception from one
RGB-D view: every candidate row is segmented by text, the strongest detection
wins, its mask is back-projected and optionally completed down to the support
surface it rests on, and -- when the winner names a feature -- that feature is
segmented, centred and fitted as a planar loop/aperture frame whose normal
sign is resolved by the camera position.

Object-specific landmark rules (CAD offsets, a measured shaft axis from a
handle to a tip, the choice of mating relation) are not part of this script.
A graph supplies them as glue downstream of ``target_cloud``/``target_obb``
and overrides ``feature_center``/``feature_pose`` where it knows better.
"""

import json
from typing import Any, TypedDict

import numpy as np
from gap import NodeContext
from scipy.ndimage import binary_dilation
from scipy.spatial.transform import Rotation

#: The winning object detection must score at least this to be trusted.
OBJECT_SCORE_MIN = 0.12
#: Feature segmentation floor used when a candidate row does not state one.
DEFAULT_FEATURE_SCORE_MIN = 0.05


class Output(TypedDict):
    target_kind: str
    target_obb: dict[str, Any]
    target_mask: np.ndarray
    target_cloud: dict[str, Any]
    feature_center: dict[str, float]
    feature_pose: dict[str, Any]
    functional_feature: dict[str, Any]


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


def _complete_to_support(
    ctx: NodeContext, camera: dict[str, Any], mask: Any, cloud: dict[str, Any]
) -> dict[str, Any]:
    """Add a phantom bottom layer so a thin object's OBB reaches its support."""
    fg = np.asarray(mask) > 0
    ring = binary_dilation(fg, iterations=8) & ~binary_dilation(fg, iterations=2)
    support = _backproject(ctx, camera, ring.astype(np.uint8) * 255)
    pts = np.asarray(cloud["points"], dtype=np.float32).reshape(-1, 3)
    support_pts = np.asarray(support["points"], dtype=np.float32).reshape(-1, 3)
    if len(pts) < 10 or len(support_pts) < 20:
        return cloud
    support_z = float(np.median(support_pts[:, 2]))
    top_z = float(np.percentile(pts[:, 2], 75))
    if not 0.001 < top_z - support_z < 0.08:
        return cloud
    bottom = pts.copy()
    bottom[:, 2] = support_z + 0.001
    return {"points": np.concatenate([pts, bottom], axis=0).astype(np.float32)}


def _mask_center_world(ctx: NodeContext, camera: dict[str, Any], mask: Any) -> dict[str, float]:
    pts = np.asarray(_backproject(ctx, camera, mask)["points"], dtype=np.float64).reshape(-1, 3)
    if len(pts) < 8:
        raise ValueError("feature mask contains too few depth points")
    center = np.median(pts, axis=0)
    return {"x": float(center[0]), "y": float(center[1]), "z": float(center[2])}


def _best_mask(ctx: NodeContext, image: np.ndarray, query: str, threshold: float) -> np.ndarray:
    result = ctx.tool("sam3.segment_text", image=image, query=query, max_results=3)
    if not result.get("masks") or float(result["scores"][0]) < threshold:
        raise ValueError(f"SAM3 could not localize {query!r}")
    return result["masks"][0]


def _point_pose(center: dict[str, float]) -> dict[str, Any]:
    """A point feature has no observed orientation; mark it with identity."""
    return {
        "position": dict(center),
        "rotation": {"w": 1.0, "x": 0.0, "y": 0.0, "z": 0.0},
    }


def _parse_candidates(candidates: str) -> list[dict[str, Any]]:
    try:
        rows = json.loads(candidates) if isinstance(candidates, str) else list(candidates)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"candidates must be a JSON list of objects: {exc}") from exc
    if not isinstance(rows, list) or not rows:
        raise ValueError("candidates must be a non-empty JSON list")
    parsed = []
    for row in rows:
        if not isinstance(row, dict) or not str(row.get("object_description", "")).strip():
            raise ValueError(f"candidate row needs an object_description: {row!r}")
        description = str(row["object_description"]).strip()
        parsed.append(
            {
                "kind": str(row.get("kind") or description),
                "object_description": description,
                "feature_description": str(row.get("feature_description") or "").strip(),
                "feature_type": str(row.get("feature_type") or "").strip(),
                "feature_score_min": float(row.get("feature_score_min", DEFAULT_FEATURE_SCORE_MIN)),
            }
        )
    return parsed


def run(
    ctx: NodeContext,
    observation: dict[str, Any],
    candidates: str,
    instruction: str = "",
    camera_name: str = "overhead",
    complete_to_support: bool = True,
) -> Output:
    """Segment every candidate, keep the strongest, fit its named feature.

    ``candidates`` is a JSON list of ``{kind, object_description,
    feature_description, feature_type, feature_score_min}`` rows. When
    ``instruction`` mentions one or more candidate ``kind`` names, only those
    rows are tried. A row with an empty ``feature_description`` yields the
    OBB centre as a point feature for a downstream landmark rule to refine.
    """
    camera = _camera(observation, camera_name)
    rows = _parse_candidates(candidates)
    instruction_lower = str(instruction).lower()
    mentioned = [row for row in rows if row["kind"].lower() in instruction_lower]
    if mentioned:
        rows = mentioned

    detections = []
    for row in rows:
        result = ctx.tool(
            "sam3.segment_text",
            image=camera["rgb"],
            query=row["object_description"],
            max_results=1,
        )
        if result.get("masks") and result.get("scores"):
            detections.append((float(result["scores"][0]), row, result["masks"][0]))
    if not detections:
        raise ValueError("SAM3 could not localize any of the described candidate objects")
    score, row, target_mask = max(detections, key=lambda item: item[0])
    if score < OBJECT_SCORE_MIN:
        raise ValueError(f"best object detection was too weak ({row['kind']}: {score:.3f})")

    cloud = ctx.tool(
        "geometry.mask_to_world_points",
        mask=target_mask,
        depth=camera["depth"],
        intrinsics=camera["intrinsics"],
        camera_pose=camera["pose"],
    )["points"]
    if complete_to_support:
        cloud = _complete_to_support(ctx, camera, target_mask, cloud)
    obb = ctx.tool("geometry.filter_and_compute_obb", points=cloud)["obb"]

    feature_query = row["feature_description"]
    feature_kind = row["feature_type"] or "loop"
    if feature_query:
        feature_mask = _best_mask(ctx, camera["rgb"], feature_query, row["feature_score_min"])
        feature_center = _mask_center_world(ctx, camera, feature_mask)
        # The camera sees the broad face of the feature. Its optical direction
        # resolves the otherwise unavoidable +/- plane-normal ambiguity.
        camera_position = camera["pose"]["position"]
        hint = {
            "x": float(camera_position["x"] - feature_center["x"]),
            "y": float(camera_position["y"] - feature_center["y"]),
            "z": float(camera_position["z"] - feature_center["z"]),
        }
        fit = ctx.tool(
            "geometry.fit_planar_feature",
            points=_backproject(ctx, camera, feature_mask),
            normal_hint=hint,
            fit_circle_center=True,
        )
        feature_pose = fit["pose"]
        feature_radius = float(fit.get("radius_inner", fit.get("radius", 0.0)))
    else:
        feature_center = {key: float(obb["center"][key]) for key in ("x", "y", "z")}
        feature_pose = _point_pose(feature_center)
        feature_radius = 0.0

    fq = feature_pose["rotation"]
    feature_axis = Rotation.from_quat([fq["x"], fq["y"], fq["z"], fq["w"]]).as_matrix()[:, 2]
    return {
        "target_kind": row["kind"],
        "target_obb": obb,
        "target_mask": target_mask,
        "target_cloud": cloud,
        "feature_center": feature_center,
        "feature_pose": feature_pose,
        "functional_feature": {
            "kind": feature_kind,
            "description": feature_query,
            "pose": feature_pose,
            "axis": {
                "x": float(feature_axis[0]),
                "y": float(feature_axis[1]),
                "z": float(feature_axis[2]),
            },
            "confidence": float(score),
            "radius_inner": feature_radius,
        },
    }
