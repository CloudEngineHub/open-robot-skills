"""Plan orientation-locked approach and linear feature engagement."""

from typing import Any, TypedDict

import numpy as np
from gap import NodeContext
from scipy.spatial.transform import Rotation


class Output(TypedDict):
    placement_plan: dict[str, Any]


def _matrix(pose: dict[str, Any]) -> np.ndarray:
    p, q = pose["position"], pose["rotation"]
    out = np.eye(4)
    out[:3, :3] = Rotation.from_quat([q["x"], q["y"], q["z"], q["w"]]).as_matrix()
    out[:3, 3] = [p["x"], p["y"], p["z"]]
    return out


def _visible_tip(
    ctx: NodeContext,
    observation: dict[str, Any],
    object_description: str,
    direction_marker_description: str,
) -> np.ndarray | None:
    """Return the best wrist-observed physical endpoint of the held object.

    The endpoint lies along the object's principal axis, directed toward its
    visible direction marker (a cap, a coloured band, a head).
    """
    best = None
    for camera in observation.get("cameras", []):
        if "eye_in_hand" not in camera.get("name", ""):
            continue
        body = ctx.tool(
            "sam3.segment_text", image=camera["rgb"], query=object_description, max_results=3
        )
        marker = ctx.tool(
            "sam3.segment_text",
            image=camera["rgb"],
            query=direction_marker_description,
            max_results=3,
        )
        score = float((body.get("scores") or [0.0])[0])
        if not body.get("masks") or not marker.get("masks") or score < 0.04:
            continue
        marker_score = float((marker.get("scores") or [0.0])[0])
        if marker_score < 0.04 or (best is not None and score + marker_score <= best[0]):
            continue
        best = (score + marker_score, camera, body["masks"][0], marker["masks"][0])
    if best is None:
        return None
    _, camera, body_mask, marker_mask = best

    def points(mask):
        cloud = ctx.tool(
            "geometry.mask_to_world_points",
            mask=np.asarray(mask, dtype=np.uint8),
            depth=camera["depth"],
            intrinsics=camera["intrinsics"],
            camera_pose=camera["pose"],
        )["points"]
        return np.asarray(cloud["points"], dtype=np.float64).reshape(-1, 3)

    body_points, marker_points = points(body_mask), points(marker_mask)
    if len(body_points) < 20 or len(marker_points) < 6:
        return None
    center = np.median(body_points, axis=0)
    _, _, basis = np.linalg.svd(body_points - center, full_matrices=False)
    axis = basis[0]
    if float(axis @ (np.median(marker_points, axis=0) - center)) < 0.0:
        axis = -axis
    along = (body_points - center) @ axis
    return np.median(body_points[along >= np.percentile(along, 98)], axis=0)


def run(
    ctx: NodeContext,
    held_feature_in_tcp: dict[str, Any],
    fixture_feature: dict[str, Any],
    relation: str,
    world_config: dict[str, Any],
    attached_object: dict[str, Any],
    precontact_clearance_m: float = 0.02,
    engagement_depth_m: float = 0.035,
    observation: dict[str, Any] | None = None,
    object_description: str = "",
    direction_marker_description: str = "",
) -> Output:
    if relation not in {
        "shaft_into_aperture",
        "tip_through_aperture",
        "insert_through",
        "loop_over_shaft",
        "feature_to_fixture",
    }:
        raise ValueError(f"unsupported feature relation {relation!r}")
    axis = np.array([fixture_feature["axis"][k] for k in ("x", "y", "z")], dtype=float)
    axis /= max(float(np.linalg.norm(axis)), 1.0e-12)
    center = np.array(
        [fixture_feature["pose"]["position"][k] for k in ("x", "y", "z")], dtype=float
    )
    ee = ctx.tool("robot.get_ee_pose")["pose"]
    ee_matrix, feature_matrix = _matrix(ee), _matrix(held_feature_in_tcp)
    rotation, feature_tcp = ee_matrix[:3, :3], feature_matrix[:3, 3]
    if observation is not None and object_description and direction_marker_description:
        # Re-observe the held feature right before engagement: the object may
        # have settled in the hand since registration, and the wrist view of
        # the physical endpoint is more precise than the carried estimate.
        visible_tip = _visible_tip(
            ctx, observation, object_description, direction_marker_description
        )
        if visible_tip is not None:
            feature_tcp = rotation.T @ (visible_tip - ee_matrix[:3, 3])

    def hand_pose(feature_target: np.ndarray) -> dict[str, Any]:
        position = feature_target - rotation @ feature_tcp
        return {
            "position": dict(zip(("x", "y", "z"), map(float, position), strict=True)),
            "rotation": dict(ee["rotation"]),
        }

    precontact = center - max(0.005, float(precontact_clearance_m)) * axis
    engaged = center + max(0.005, float(engagement_depth_m)) * axis
    return {
        "placement_plan": {
            "waypoints": [
                {"pose": hand_pose(precontact), "mode": "planned_joint", "cartesian": False},
                {
                    "pose": hand_pose(engaged),
                    "mode": "planned_linear",
                    "cartesian": True,
                    "allow_goal_contact": True,
                },
            ],
            "world_config": world_config,
            "attached_object": attached_object,
        }
    }
