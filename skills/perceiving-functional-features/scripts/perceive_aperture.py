"""Measure an aperture's centre and inward axis on a fixture's fitted lid plane.

The fixture and its opening are segmented by text. The lid is fitted as a
plane from the upper depth quantile of the fixture cloud (the camera looks
down on it), with the normal signed toward the camera and the returned axis
pointing inward. Depth pixels inside a hole belong to its interior and are
parallax-shifted, so the opening is measured by intersecting calibrated
camera rays through its 2-D mask with the fitted lid plane: the centroid ray
gives the centre and sampled rim rays give the radius.
"""

from typing import Any, TypedDict

import numpy as np
from gap import NodeContext
from scipy.spatial.transform import Rotation


class Output(TypedDict):
    fixture_mask: np.ndarray
    fixture_cloud: dict[str, np.ndarray]
    aperture_mask: np.ndarray
    aperture_cloud: dict[str, np.ndarray]
    aperture_center: dict[str, float]
    fixture_axis: dict[str, float]
    aperture_radius: float
    rim_z: float
    fixture_feature: dict[str, Any]


def _camera(observation: dict[str, Any], name: str) -> dict[str, Any]:
    cameras = observation.get("cameras") or []
    if isinstance(cameras, dict):
        cameras = list(cameras.values())
    for camera in cameras:
        if camera.get("name") == name:
            return camera
    raise ValueError(f"observation has no camera named {name!r}")


def _mask(ctx: NodeContext, image: Any, query: str, threshold: float) -> np.ndarray:
    result = ctx.tool("sam3.segment_text", image=image, query=query, max_results=3)
    if not result.get("masks") or float(result["scores"][0]) < threshold:
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


def _pose_matrix(pose: dict[str, Any]) -> np.ndarray:
    p, q = pose["position"], pose["rotation"]
    out = np.eye(4)
    out[:3, :3] = Rotation.from_quat([q["x"], q["y"], q["z"], q["w"]]).as_matrix()
    out[:3, 3] = [p["x"], p["y"], p["z"]]
    return out


def run(
    ctx: NodeContext,
    observation: dict[str, Any],
    fixture_description: str,
    feature_description: str,
    feature_type: str = "aperture",
    fixture_score_min: float = 0.08,
    feature_score_min: float = 0.04,
    camera_name: str = "overhead",
) -> Output:
    camera = _camera(observation, camera_name)
    box_mask = _mask(ctx, camera["rgb"], fixture_description, fixture_score_min)
    aperture_mask = _mask(ctx, camera["rgb"], feature_description, feature_score_min)
    box_points = _points(ctx, camera, box_mask)
    aperture_points = _points(ctx, camera, aperture_mask)
    if len(box_points) < 30:
        raise ValueError(f"{fixture_description!r} mask contains too few depth points")

    # Fit the visible lid rather than assuming it is horizontal. The camera is
    # above the fixture, so the upper depth quantile isolates the lid from its
    # walls.
    z_cut = np.percentile(box_points[:, 2], 78.0)
    lid = box_points[box_points[:, 2] >= z_cut]
    origin = np.median(lid, axis=0)
    _, _, basis = np.linalg.svd(lid - origin, full_matrices=False)
    outward = basis[-1]
    camera_position = _pose_matrix(camera["pose"])[:3, 3]
    if float(outward @ (camera_position - origin)) < 0.0:
        outward = -outward
    inward = -outward

    # The hole's depth pixels lie below the lid and are parallax-shifted. Use
    # its 2-D mask centroid to form a calibrated camera ray, then intersect
    # that ray with the measured lid plane.
    rows, cols = np.nonzero(aperture_mask > 0)
    if len(rows) < 8:
        raise ValueError("aperture mask contains too few pixels")
    u, v = float(np.median(cols)), float(np.median(rows))
    intrinsics = np.asarray(camera["intrinsics"], dtype=np.float64).reshape(3, 3)
    ray_camera = np.linalg.inv(intrinsics) @ np.array([u, v, 1.0])
    camera_world = _pose_matrix(camera["pose"])
    ray_world = camera_world[:3, :3] @ ray_camera
    denominator = float(outward @ ray_world)
    if abs(denominator) < 1.0e-8:
        raise ValueError("aperture camera ray is parallel to the perceived lid")
    distance = float(outward @ (origin - camera_position) / denominator)
    aperture = camera_position + distance * ray_world
    # Measure the opening on the fitted lid plane. Depth inside a hole belongs
    # to its interior, so its raw 3-D cloud cannot measure the rim reliably.
    # Project sampled mask pixels onto the lid plane using calibrated rays.
    sample = np.linspace(0, len(rows) - 1, min(len(rows), 256)).round().astype(int)
    plane_points = []
    for row, col in zip(rows[sample], cols[sample], strict=True):
        ray = camera_world[:3, :3] @ (
            np.linalg.inv(intrinsics) @ np.array([float(col), float(row), 1.0])
        )
        denominator = float(outward @ ray)
        if abs(denominator) < 1.0e-8:
            continue
        travel = float(outward @ (origin - camera_position) / denominator)
        plane_points.append(camera_position + travel * ray)
    if len(plane_points) < 8:
        raise ValueError("could not measure aperture radius on the lid plane")
    radial = np.linalg.norm(np.asarray(plane_points) - aperture, axis=1)
    aperture_radius = float(np.percentile(radial, 90.0))
    aperture_center = {"x": float(aperture[0]), "y": float(aperture[1]), "z": float(aperture[2])}
    fixture_axis = {"x": float(inward[0]), "y": float(inward[1]), "z": float(inward[2])}
    aperture_pose = {
        "position": aperture_center,
        "rotation": {"w": 1.0, "x": 0.0, "y": 0.0, "z": 0.0},
    }
    return {
        "fixture_mask": box_mask,
        "fixture_cloud": {"points": box_points.astype(np.float32)},
        "aperture_mask": aperture_mask,
        "aperture_cloud": {"points": aperture_points.astype(np.float32)},
        "aperture_center": aperture_center,
        "fixture_axis": fixture_axis,
        "aperture_radius": aperture_radius,
        "rim_z": float(aperture[2]),
        "fixture_feature": {
            "kind": feature_type,
            "pose": aperture_pose,
            "axis": fixture_axis,
            "radius_inner": aperture_radius,
            "confidence": 1.0,
            "description": feature_description,
        },
    }
