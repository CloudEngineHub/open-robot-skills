"""Execute only the first (free-space approach) waypoint of a placement plan.

Used when the graph wants to re-observe the held object at the approach pose
(for example to re-register the wrist before a tight engagement) and then hand
the remaining waypoints to ``executing-feature-mating``.
"""

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


def run(ctx: NodeContext, placement_plan: dict[str, Any]) -> Output:
    waypoints = list(placement_plan.get("waypoints") or [])
    if not waypoints:
        raise ValueError("placement plan has no approach waypoint")
    waypoint = waypoints[0]
    pose = waypoint["pose"]
    trajectory = None
    for _ in range(_PLAN_ATTEMPTS):
        result = ctx.tool(
            "motion.plan_to_pose",
            pose=pose,
            world_config=placement_plan.get("world_config"),
            attached_object=placement_plan.get("attached_object"),
            allow_start_contact=False,
            allow_goal_contact=False,
            contact_margin=float(waypoint.get("contact_margin", 0.005)),
        )
        trajectory = result.get("trajectory") if result.get("planned") else None
        if trajectory and trajectory.get("waypoints"):
            break
    if not trajectory or not trajectory.get("waypoints"):
        raise RuntimeError("planner refused the pre-insertion approach waypoint")
    ctx.tool(
        "robot.execute_trajectory",
        trajectory=trajectory,
        tolerance=_TRACK_TOLERANCE_M,
        max_steps_per_waypoint=_MAX_STEPS_PER_WAYPOINT,
    )
    return {"approach_pose": pose}
