"""Execute only the first (free-space approach) waypoint of a placement plan.

Used when the graph wants to re-observe the held object at the approach pose
(for example to re-register the wrist before a tight engagement) and then hand
the remaining waypoints to ``executing-feature-mating``.

Without an ``execution_profile`` this is the original body: up to three
sampled plans to the approach pose, then the trajectory tracked to 2 mm. A
profile adds, and only when present: a local Cartesian shortcut when the hand
is already within ``local_cartesian_max_distance_m`` /
``local_cartesian_max_rotation_deg`` of the pose, a goal-contact retry when
the free-space plan is refused, a Cartesian recovery within a wider band, and
the before/after TCP reads that fill ``waypoint_report``. Those reads are
recorded tool calls, which is why the whole layer is gated on the profile
rather than on defaults. ``arm_id`` is threaded as an absent keyword when
unset.
"""

import math
from typing import Any, TypedDict

from gap import NodeContext

#: The sampled planner can miss a narrow, valid corridor it finds on the next
#: seed; replanning is cheap and never weakens the collision world.
_PLAN_ATTEMPTS = 3
#: Track the approach waypoint to millimetre precision: the engagement that
#: follows has only millimetres of radial clearance.
_TRACK_TOLERANCE_M = 0.002
_MAX_STEPS_PER_WAYPOINT = 60


class Output(TypedDict):
    approach_pose: dict[str, Any]
    waypoint_report: dict[str, Any]


def _errors(current: dict[str, Any], target: dict[str, Any]) -> tuple[float, float]:
    """Translation [m] and rotation [deg] between two poses."""
    distance = math.sqrt(sum(
        (float(current["position"][key]) - float(target["position"][key])) ** 2 for key in ("x", "y", "z")
    ))
    dot = min(1.0, abs(sum(
        float(current["rotation"][key]) * float(target["rotation"][key]) for key in ("w", "x", "y", "z")
    )))
    return distance, math.degrees(2.0 * math.acos(dot))


def run(
    ctx: NodeContext,
    placement_plan: dict[str, Any],
    arm_id: int | None = None,
    execution_profile: dict[str, Any] | None = None,
    registration_uncertainty_m: float | None = None,
) -> Output:
    waypoints = list(placement_plan.get("waypoints") or [])
    if not waypoints:
        raise ValueError("placement plan has no approach waypoint")
    waypoint = waypoints[0]
    pose = waypoint["pose"]
    attachment = placement_plan.get("attached_object")
    if arm_id is None and isinstance(attachment, dict) and attachment.get("arm_id") is not None:
        arm_id = int(attachment["arm_id"])
    on_arm: dict[str, Any] = {} if arm_id is None else {"arm_id": int(arm_id)}
    if registration_uncertainty_m is None:
        registration_uncertainty_m = (
            float(attachment.get("translation_uncertainty_m", 0.0)) if isinstance(attachment, dict) else 0.0
        )
    profile = execution_profile or {}

    def plan(allow_goal_contact: bool, margin: float, attempts: int):
        trajectory = None
        for _ in range(attempts):
            result = ctx.tool(
                "motion.plan_to_pose",
                pose=pose,
                world_config=placement_plan.get("world_config"),
                attached_object=attachment,
                allow_start_contact=False,
                allow_goal_contact=allow_goal_contact,
                contact_margin=margin,
                **on_arm,
            )
            trajectory = result.get("trajectory") if result.get("planned") else None
            if trajectory and trajectory.get("waypoints"):
                return trajectory
        return None

    def track(trajectory: dict[str, Any]) -> None:
        ctx.tool(
            "robot.execute_trajectory",
            trajectory=trajectory,
            tolerance=float(profile.get("trajectory_tolerance_m", _TRACK_TOLERANCE_M)),
            max_steps_per_waypoint=int(profile.get("max_steps_per_waypoint", _MAX_STEPS_PER_WAYPOINT)),
            **on_arm,
        )

    if not profile:
        trajectory = plan(False, float(waypoint.get("contact_margin", 0.005)), _PLAN_ATTEMPTS)
        if trajectory is None:
            raise RuntimeError("planner refused the pre-insertion approach waypoint")
        track(trajectory)
        return {
            "approach_pose": pose,
            "waypoint_report": {"index": 0, "mode": "approach", "method": "planned",
                                "attempts": _PLAN_ATTEMPTS, "registration_uncertainty_m": float(registration_uncertainty_m)},
        }

    current = ctx.tool("robot.get_ee_pose", **on_arm)["pose"]
    distance, angle = _errors(current, pose)
    method, attempts = "cartesian", 1
    if (distance <= float(profile.get("local_cartesian_max_distance_m", 0.050))
            and angle <= float(profile.get("local_cartesian_max_rotation_deg", 25.0))):
        ctx.tool("robot.go_to_pose_cartesian", pose=pose, **on_arm)
    else:
        attempts = max(1, min(int(profile.get("planner_attempts", _PLAN_ATTEMPTS)), 3))
        trajectory = plan(False, float(waypoint.get("contact_margin", profile.get("contact_margin_m", 0.005))), attempts)
        if trajectory is None:
            goal_attempts = max(0, min(int(profile.get("goal_contact_attempts", 2)), 3))
            trajectory = plan(True, float(profile.get("goal_contact_margin_m", 0.002)), goal_attempts)
            attempts += goal_attempts
        if trajectory is not None:
            method = "planned"
            track(trajectory)
        elif (distance <= float(profile.get("recovery_max_distance_m", 0.100))
              and angle <= float(profile.get("recovery_max_rotation_deg", 35.0))):
            method = "cartesian_recovery"
            ctx.tool("robot.go_to_pose_cartesian", pose=pose, **on_arm)
        else:
            raise RuntimeError(f"planner refused approach ({distance * 1000:.1f} mm / {angle:.1f} deg)")
    reached = ctx.tool("robot.get_ee_pose", **on_arm)["pose"]
    residual, rotation_error = _errors(reached, pose)
    if method == "cartesian_recovery" and residual > float(profile.get("recovery_position_tolerance_m", 0.015)):
        raise RuntimeError(f"Cartesian approach recovery left {residual * 1000:.1f} mm error")
    return {
        "approach_pose": pose,
        "waypoint_report": {
            "index": 0, "mode": "approach", "method": method, "attempts": attempts,
            "position_error_m": residual, "rotation_error_deg": rotation_error,
            "registration_uncertainty_m": float(registration_uncertainty_m),
        },
    }
