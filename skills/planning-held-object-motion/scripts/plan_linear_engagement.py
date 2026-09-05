"""Plan orientation-locked approach and linear feature engagement."""

from typing import Any

import numpy as np
from gap import NodeContext
from scipy.spatial.transform import Rotation


def _matrix(pose: dict[str, Any]) -> np.ndarray:
    p, q = pose["position"], pose["rotation"]
    out = np.eye(4)
    out[:3, :3] = Rotation.from_quat([q["x"], q["y"], q["z"], q["w"]]).as_matrix()
    out[:3, 3] = [p["x"], p["y"], p["z"]]
    return out


def run(ctx: NodeContext, held_feature_in_tcp: dict[str, Any],
        fixture_feature: dict[str, Any], relation: str,
        world_config: dict[str, Any], attached_object: dict[str, Any],
        precontact_clearance_m: float = 0.02,
        engagement_depth_m: float = 0.035) -> dict[str, Any]:
    if relation not in {"shaft_into_aperture", "tip_through_aperture", "insert_through",
                        "loop_over_shaft", "feature_to_fixture"}:
        raise ValueError(f"unsupported feature relation {relation!r}")
    axis = np.array([fixture_feature["axis"][k] for k in ("x", "y", "z")], dtype=float)
    axis /= max(float(np.linalg.norm(axis)), 1.0e-12)
    center = np.array([fixture_feature["pose"]["position"][k] for k in ("x", "y", "z")], dtype=float)
    ee = ctx.tool("robot.get_ee_pose")["pose"]
    ee_matrix, feature_matrix = _matrix(ee), _matrix(held_feature_in_tcp)
    rotation, feature_tcp = ee_matrix[:3, :3], feature_matrix[:3, 3]

    def hand_pose(feature_target: np.ndarray) -> dict[str, Any]:
        position = feature_target - rotation @ feature_tcp
        return {"position": dict(zip(("x", "y", "z"), map(float, position))),
                "rotation": dict(ee["rotation"])}

    precontact = center - max(0.005, float(precontact_clearance_m)) * axis
    engaged = center + max(0.005, float(engagement_depth_m)) * axis
    return {"placement_plan": {
        "waypoints": [
            {"pose": hand_pose(precontact), "mode": "planned_joint", "cartesian": False},
            {"pose": hand_pose(engaged), "mode": "planned_linear", "cartesian": True,
             "allow_goal_contact": True},
        ],
        "world_config": world_config,
        "attached_object": attached_object,
    }}
