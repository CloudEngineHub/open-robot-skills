"""Execute typed planner, Cartesian-servo and contact waypoints of a placement plan."""

import math
from typing import Any, TypedDict

from gap import NodeContext

#: Fixture engagement has millimetres of usable radial clearance. A 10 mm
#: "already there" shortcut discards the wrist-registration correction and
#: starts insertion almost against the aperture rim, so only a genuinely
#: zero-length waypoint is skipped: about 1.5 mm and 2 degrees.
_REACHED_DISTANCE_M = 0.0015
_REACHED_DOT = 0.99985
#: After a final wrist re-registration only a small correction remains. A
#: fresh joint-space trajectory can make a loosely held object slip while the
#: TCP moves a few millimetres; the robot's smooth Cartesian servo is used for
#: such local corrections and the sampled planner kept for larger motions.
_SERVO_DISTANCE_M = 0.03
_SERVO_DOT = 0.995
#: The sampled planner can miss a narrow, valid engagement corridor that it
#: finds on another seed. Replanning is cheap and never weakens or discards
#: the collision world.
_PLAN_ATTEMPTS = 3
#: 10 mm tracking suits free-space carry but consumes almost all of a
#: ring/shaft clearance: engagement waypoints are tracked to 2 mm.
_TRACK_TOLERANCE_M = 0.002
_MAX_STEPS_PER_WAYPOINT = 60


class Output(TypedDict):
    final_pose: dict[str, Any]


def _pose_distance(current: dict[str, Any], target: dict[str, Any]) -> tuple[float, float]:
    """Return (translation distance in metres, |quaternion dot|) between two poses."""
    cp, tp = current["position"], target["position"]
    distance = math.sqrt(sum((float(cp[axis]) - float(tp[axis])) ** 2 for axis in ("x", "y", "z")))
    cq, tq = current["rotation"], target["rotation"]
    dot = abs(sum(float(cq[axis]) * float(tq[axis]) for axis in ("w", "x", "y", "z")))
    return distance, dot


def _pose_reached(current: dict[str, Any], target: dict[str, Any]) -> bool:
    distance, dot = _pose_distance(current, target)
    return distance <= _REACHED_DISTANCE_M and dot >= _REACHED_DOT


def run(ctx: NodeContext, placement_plan: dict[str, Any]) -> Output:
    waypoints = list(placement_plan.get("waypoints") or [])
    if not waypoints:
        raise ValueError("placement plan has no waypoints")
    world = placement_plan.get("world_config")
    attachment = placement_plan.get("attached_object")
    final_pose: dict[str, Any] | None = None
    for waypoint in waypoints:
        final_pose, mode = waypoint["pose"], waypoint.get("mode")
        if mode is None:
            mode = "planned_linear" if waypoint.get("cartesian", False) else "planned_joint"
        if mode == "contact_seat":
            ctx.tool("robot.move_cartesian_until_contact", pose=final_pose)
            continue
        if mode == "cartesian_cross":
            # Crossing a fixture mouth must not stop at the first incidental
            # touch. The pose is a short local segment, typically recomputed
            # from the final wrist observation, for which Cartesian servoing
            # is smoother and more repeatable than a new sampled plan.
            ctx.tool("robot.go_to_pose_cartesian", pose=final_pose)
            continue
        if mode not in {"planned_joint", "planned_linear"}:
            raise ValueError(f"unknown placement waypoint mode {mode!r}")
        # The preceding carry plan normally terminates at the first engagement
        # waypoint. Do not ask the planner to solve a zero-motion problem
        # there: at fixture clearance its start-contact check can reject it
        # even though no swept motion is requested.
        current_pose = ctx.tool("robot.get_ee_pose")["pose"]
        if _pose_reached(current_pose, final_pose):
            continue
        distance, orientation_dot = _pose_distance(current_pose, final_pose)
        if (
            mode == "planned_joint"
            and distance <= _SERVO_DISTANCE_M
            and orientation_dot >= _SERVO_DOT
        ):
            ctx.tool("robot.go_to_pose_cartesian", pose=final_pose)
            continue
        trajectory = None
        for _ in range(_PLAN_ATTEMPTS):
            tool = "motion.plan_linear" if mode == "planned_linear" else "motion.plan_to_pose"
            inputs = (
                {"end": final_pose, "orientation": "lock"}
                if mode == "planned_linear"
                else {"pose": final_pose}
            )
            result = ctx.tool(
                tool,
                **inputs,
                world_config=world,
                attached_object=attachment,
                allow_start_contact=bool(waypoint.get("allow_start_contact", False)),
                allow_goal_contact=bool(waypoint.get("allow_goal_contact", False)),
                contact_margin=float(waypoint.get("contact_margin", 0.005)),
            )
            trajectory = result.get("trajectory") if result.get("planned") else None
            if trajectory and trajectory.get("waypoints"):
                break
        if not trajectory or not trajectory.get("waypoints"):
            raise RuntimeError(
                f"planner refused a typed fixture-engagement waypoint after {_PLAN_ATTEMPTS} attempts"
            )
        ctx.tool(
            "robot.execute_trajectory",
            trajectory=trajectory,
            tolerance=_TRACK_TOLERANCE_M,
            max_steps_per_waypoint=_MAX_STEPS_PER_WAYPOINT,
        )
    assert final_pose is not None
    return {"final_pose": final_pose}
