"""Plan lift, minimum reorientation, and orientation-locked fixture transit."""

from typing import Any, TypedDict

import numpy as np
from gap import NodeContext
from scipy.spatial.transform import Rotation


class Output(TypedDict):
    reorientation_plan: dict[str, Any]


def _rotation(pose: dict[str, Any]) -> Rotation:
    q = pose["rotation"]
    return Rotation.from_quat([q["x"], q["y"], q["z"], q["w"]])


def _pose(position: np.ndarray, rotation: Rotation) -> dict[str, Any]:
    q = rotation.as_quat()
    return {"position": dict(zip(("x", "y", "z"), map(float, position))),
            "rotation": {"w": float(q[3]), "x": float(q[0]),
                         "y": float(q[1]), "z": float(q[2])}}


def _vec(value: dict[str, float]) -> np.ndarray:
    return np.array([value[key] for key in ("x", "y", "z")], dtype=np.float64)


def run(ctx: NodeContext, held_feature_in_tcp: dict[str, Any],
        fixture_feature: dict[str, Any], relation: str,
        world_config: dict[str, Any], attached_object: dict[str, Any],
        support_normal: dict[str, float] | None = None,
        approach_clearance_m: float = 0.08) -> Output:
    if relation not in {"shaft_into_aperture", "tip_through_aperture", "insert_through",
                        "loop_over_shaft", "feature_to_fixture"}:
        raise ValueError(f"unsupported feature relation {relation!r}")
    axis_value = fixture_feature.get("axis")
    if not axis_value:
        raise ValueError("fixture feature has no directed axis")
    fixture_pose = fixture_feature["pose"]
    axis = _vec(axis_value); axis /= max(float(np.linalg.norm(axis)), 1.0e-12)
    normal = _vec(support_normal or {"x": 0.0, "y": 0.0, "z": 1.0})
    normal /= max(float(np.linalg.norm(normal)), 1.0e-12)

    ee = ctx.tool("robot.get_ee_pose")["pose"]
    current_rotation = _rotation(ee)
    feature_rotation = _rotation(held_feature_in_tcp)
    feature_offset = _vec(held_feature_in_tcp["position"])
    current_axis = current_rotation.apply(feature_rotation.apply([0.0, 0.0, 1.0]))
    current_axis /= max(float(np.linalg.norm(current_axis)), 1.0e-12)
    correction, _ = Rotation.align_vectors([axis], [current_axis])
    aligned = correction * current_rotation

    fixture_center = _vec(fixture_pose["position"])
    clearance = max(0.02, float(approach_clearance_m))
    # A shaft axis points from its base toward its free end, so a loop stages
    # farther along that axis. An aperture axis points into the opening, so an
    # inserting tip stages on the opposite side.
    direction = 1.0 if relation == "loop_over_shaft" else -1.0
    feature_target = fixture_center + direction * clearance * axis
    candidates = []
    for angle in np.deg2rad([0.0, 45.0, -45.0, 90.0, -90.0, 135.0, -135.0, 180.0]):
        rotation = Rotation.from_rotvec(axis * float(angle)) * aligned
        hand_position = feature_target - rotation.apply(feature_offset)
        candidate = _pose(hand_position, rotation)
        try:
            check = ctx.tool("motion.plan_joint", pose=candidate, orientation="lock")
        except Exception:
            continue
        if not check.get("planned"):
            continue
        if float(check.get("position_error_m", 0.0)) > 0.006:
            continue
        if float(check.get("rotation_error_rad", 0.0)) > np.deg2rad(4.0):
            continue
        # Geodesic TCP rotation is the stable, robot-independent score. It
        # prevents symmetry from producing an arbitrary large wrist turn.
        turn = float((rotation * current_rotation.inv()).magnitude())
        candidates.append((turn, abs(float(angle)), candidate, rotation))
    if not candidates:
        raise RuntimeError("no feasible orientation aligns the held feature with the fixture")
    _, _, transit_pose, goal_rotation = min(candidates, key=lambda item: item[:2])

    def sphere_extent(sphere: dict[str, Any]) -> float:
        center = sphere.get("center") or {}
        offset = (np.asarray(center, dtype=float).reshape(3) if isinstance(center, (list, tuple))
                  else np.array([float(center.get(k, 0.0)) for k in ("x", "y", "z")]))
        return float(np.linalg.norm(offset)) + float(sphere.get("radius", 0.0))
    extents = [sphere_extent(s) for s in attached_object.get("spheres", [])]
    escape_distance = max(0.04, max(extents, default=0.025) + 0.015)
    current_position = _vec(ee["position"])
    escape_position = current_position + escape_distance * normal
    rotate_pose = _pose(escape_position, goal_rotation)
    return {"reorientation_plan": {
        "time_scale": 2.0,
        "waypoints": [
            {"pose": _pose(escape_position, current_rotation),
             "mode": "contact_transition", "cartesian": True},
            {"pose": rotate_pose, "mode": "planned_joint", "cartesian": False,
             "use_attachment": False},
            {"pose": transit_pose, "mode": "planned_joint", "cartesian": False},
        ],
        "world_config": world_config,
        "attached_object": attached_object,
    }}
