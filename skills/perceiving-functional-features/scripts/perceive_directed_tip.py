"""Localize an elongated object and its directed insertion tip from one RGB-D view.

The object is segmented by text and back-projected. Its long axis gets a
direction from a visible marker at one end (``direction_marker_description``,
for example a coloured cap) or, when no marker is described or found, from the
narrower of the two ends. The tip is the extreme of the cloud along that
directed axis; the returned pose has local Z pointing from the object body
through the tip, which is the insertion direction a mating skill needs.
"""

from typing import Any, TypedDict

import numpy as np
from gap import NodeContext
from scipy.spatial.transform import Rotation


class Output(TypedDict):
    target_obb: dict[str, Any]
    target_mask: np.ndarray
    target_cloud: dict[str, np.ndarray]
    insertion_pose: dict[str, Any]
    marker_center: dict[str, float]
    functional_feature: dict[str, Any]


def _camera(observation: dict[str, Any], name: str) -> dict[str, Any]:
    cameras = observation.get("cameras") or []
    if isinstance(cameras, dict):
        cameras = list(cameras.values())
    for camera in cameras:
        if camera.get("name") == name:
            return camera
    raise ValueError(f"observation has no camera named {name!r}")


def _mask(
    ctx: NodeContext, image: Any, query: str, threshold: float, optional: bool = False
) -> np.ndarray | None:
    result = ctx.tool("sam3.segment_text", image=image, query=query, max_results=3)
    if not result.get("masks") or float(result["scores"][0]) < threshold:
        if optional:
            return None
        raise ValueError(f"could not perceive {query!r}")
    return np.asarray(result["masks"][0], dtype=np.uint8)


def _points(ctx: NodeContext, camera: dict[str, Any], mask: np.ndarray) -> np.ndarray:
    cloud = ctx.tool(
        "geometry.mask_to_world_points",
        mask=mask,
        depth=camera["depth"],
        intrinsics=camera["intrinsics"],
        camera_pose=camera["pose"],
    )["points"]
    return np.asarray(cloud["points"], dtype=np.float64).reshape(-1, 3)


def _narrow_endpoint(points: np.ndarray) -> np.ndarray:
    """The end of the principal axis whose cross-section is narrower."""
    origin = np.median(points, axis=0)
    _, _, basis = np.linalg.svd(points - origin, full_matrices=False)
    axis = basis[0]
    along = (points - origin) @ axis
    tails = [
        points[along <= np.percentile(along, 15)],
        points[along >= np.percentile(along, 85)],
    ]
    widths = []
    for tail in tails:
        centered = tail - np.median(tail, axis=0)
        perpendicular = centered - np.outer(centered @ axis, axis)
        widths.append(float(np.percentile(np.linalg.norm(perpendicular, axis=1), 90)))
    return np.median(tails[int(np.argmin(widths))], axis=0)


def run(
    ctx: NodeContext,
    observation: dict[str, Any],
    object_description: str,
    direction_marker_description: str = "",
    feature_type: str = "tip",
    object_score_min: float = 0.08,
    marker_score_min: float = 0.06,
    camera_name: str = "overhead",
) -> Output:
    camera = _camera(observation, camera_name)
    target_mask = _mask(ctx, camera["rgb"], object_description, object_score_min)
    points = _points(ctx, camera, target_mask)
    if len(points) < 30:
        raise ValueError(f"{object_description!r} mask contains too few depth points")
    obb = ctx.tool("geometry.filter_and_compute_obb", points={"points": points.astype(np.float32)})[
        "obb"
    ]
    marker_mask = None
    if direction_marker_description:
        marker_mask = _mask(
            ctx, camera["rgb"], direction_marker_description, marker_score_min, optional=True
        )
    marker_points = (
        _points(ctx, camera, marker_mask) if marker_mask is not None else np.empty((0, 3))
    )
    marker = (
        np.median(marker_points, axis=0) if len(marker_points) >= 8 else _narrow_endpoint(points)
    )
    center = np.array([obb["center"][k] for k in ("x", "y", "z")], dtype=np.float64)
    axis = marker - center
    axis /= max(float(np.linalg.norm(axis)), 1.0e-12)
    along = (points - center) @ axis
    tip = np.median(points[along >= np.percentile(along, 98)], axis=0)
    reference = (
        np.array([0.0, 0.0, 1.0]) if abs(float(axis[2])) < 0.9 else np.array([1.0, 0.0, 0.0])
    )
    x_axis = np.cross(reference, axis)
    x_axis /= np.linalg.norm(x_axis)
    y_axis = np.cross(axis, x_axis)
    quat = Rotation.from_matrix(np.column_stack((x_axis, y_axis, axis))).as_quat()
    insertion_pose = {
        "position": {"x": float(tip[0]), "y": float(tip[1]), "z": float(tip[2])},
        "rotation": {
            "w": float(quat[3]),
            "x": float(quat[0]),
            "y": float(quat[1]),
            "z": float(quat[2]),
        },
    }
    centered = points - tip
    radial = centered - np.outer(centered @ axis, axis)
    radius = float(np.percentile(np.linalg.norm(radial, axis=1), 35))
    return {
        "target_obb": obb,
        "target_mask": target_mask,
        "target_cloud": {"points": points.astype(np.float32)},
        "marker_center": {"x": float(marker[0]), "y": float(marker[1]), "z": float(marker[2])},
        "insertion_pose": insertion_pose,
        "functional_feature": {
            "kind": feature_type,
            "pose": insertion_pose,
            "axis": {"x": float(axis[0]), "y": float(axis[1]), "z": float(axis[2])},
            "radius_outer": max(radius, 1.0e-4),
            "confidence": 1.0,
            "description": direction_marker_description or f"tip of {object_description}",
        },
    }
