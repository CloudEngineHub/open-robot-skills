"""Lift, reobserve the held object, and reject an empty or slipped grasp.

After a short Cartesian lift the object is looked for from the wrist camera
first and the overhead camera second, by ``object_description`` and, when
given, ``marker_description``. Confidence only admits a candidate: metric
depth, table clearance and hand proximity decide. A small held object can
occupy only a few dozen wrist pixels, so the score floor is deliberately low.
"""

from typing import Any, TypedDict

import numpy as np
from gap import NodeContext


class Output(TypedDict):
    route: str
    verified: bool
    observed_center: dict[str, float]
    point_count: int
    camera: str
    reason: str


def _not_held(
    reason: str, camera: str = "", point_count: int = 0, center: np.ndarray | None = None
) -> Output:
    if center is None:
        observed = {"x": 0.0, "y": 0.0, "z": 0.0}
    else:
        observed = {"x": float(center[0]), "y": float(center[1]), "z": float(center[2])}
    return {
        "route": "not_held",
        "verified": False,
        "observed_center": observed,
        "point_count": int(point_count),
        "camera": camera,
        "reason": reason,
    }


def run(
    ctx: NodeContext,
    object_description: str,
    marker_description: str = "",
    lift_m: float = 0.04,
    score_min: float = 0.005,
    min_points: int = 20,
    min_above_table_m: float = 0.03,
    max_hand_distance_m: float = 0.20,
    wrist_camera_keyword: str = "eye_in_hand",
    overhead_camera_name: str = "overhead",
) -> Output:
    """Route ``verified`` when the object rides with the lifted hand, else ``not_held``."""
    ee = ctx.tool("robot.get_ee_pose")["pose"]
    lifted = {"position": dict(ee["position"]), "rotation": dict(ee["rotation"])}
    lifted["position"]["z"] = float(lifted["position"]["z"]) + float(lift_m)
    ctx.tool("robot.go_to_pose_cartesian", pose=lifted)

    observation = ctx.tool("robot.get_observation")
    cameras = observation.get("cameras") or []
    if isinstance(cameras, dict):
        cameras = list(cameras.values())
    wrist = next((c for c in cameras if wrist_camera_keyword in str(c.get("name", ""))), None)
    if wrist is None:
        raise RuntimeError(
            f"grasp verification requires an RGB-D camera whose name contains "
            f"{wrist_camera_keyword!r}"
        )
    candidates = [wrist]
    overhead = next((c for c in cameras if c.get("name") == overhead_camera_name), None)
    if overhead is not None:
        candidates.append(overhead)
    queries = [q for q in (object_description, marker_description) if q]

    camera: dict[str, Any] | None = None
    result: dict[str, Any] | None = None
    for candidate in candidates:
        for query in queries:
            detected = ctx.tool(
                "sam3.segment_text", image=candidate["rgb"], query=query, max_results=3
            )
            if (
                detected.get("masks")
                and detected.get("scores")
                and float(detected["scores"][0]) >= float(score_min)
            ):
                camera, result = candidate, detected
                break
        if camera is not None:
            break
    if camera is None or result is None:
        return _not_held(f"no camera sees {object_description!r} after lifting")

    camera_name = str(camera.get("name", "unknown"))
    cloud = ctx.tool(
        "geometry.mask_to_world_points",
        mask=np.asarray(result["masks"][0], dtype=np.uint8),
        depth=camera["depth"],
        intrinsics=camera["intrinsics"],
        camera_pose=camera["pose"],
    )["points"]
    points = np.asarray(cloud["points"], dtype=np.float64).reshape(-1, 3)
    if len(points) < int(min_points):
        return _not_held(
            f"{camera_name} mask of {object_description!r} has too few valid depth points "
            f"({len(points)} < {int(min_points)})",
            camera=camera_name,
            point_count=len(points),
        )

    center = np.median(points, axis=0)
    hand = np.array([lifted["position"][key] for key in ("x", "y", "z")], dtype=np.float64)
    workspace = ctx.tool("robot.describe_workspace")
    above_table = float(center[2] - float(workspace["surface_z"]))
    hand_distance = float(np.linalg.norm(center - hand))
    if above_table < float(min_above_table_m) or hand_distance > float(max_hand_distance_m):
        return _not_held(
            f"{camera_name} reobservation shows {object_description!r} was not lifted with "
            f"the hand (above table {above_table * 1000.0:.1f} mm, hand distance "
            f"{hand_distance * 1000.0:.1f} mm)",
            camera=camera_name,
            point_count=len(points),
            center=center,
        )
    return {
        "route": "verified",
        "verified": True,
        "observed_center": {"x": float(center[0]), "y": float(center[1]), "z": float(center[2])},
        "point_count": int(len(points)),
        "camera": camera_name,
        "reason": "",
    }
