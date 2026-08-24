"""Execute a clearance-first pose sequence while preserving the grasp."""

from typing import Any, TypedDict

import numpy as np
from gap import NodeContext


class Output(TypedDict):
    final_pose: dict[str, Any]


def _time_scale(trajectory: dict[str, Any], scale: float) -> dict[str, Any]:
    waypoints = list(trajectory.get("waypoints") or [])
    if scale <= 1.0 or len(waypoints) < 2:
        return trajectory
    rows = np.asarray([waypoint["positions"] for waypoint in waypoints], dtype=float)
    count = max(2, int(round((len(rows) - 1) * scale)) + 1)
    source, target = np.linspace(0.0, 1.0, len(rows)), np.linspace(0.0, 1.0, count)
    scaled = np.stack([np.interp(target, source, rows[:, j]) for j in range(rows.shape[1])], axis=1)
    return {"waypoints": [{"positions": row.tolist()} for row in scaled]}


def run(ctx: NodeContext, reorientation_plan: dict[str, Any]) -> Output:
    waypoints = list(reorientation_plan.get("waypoints") or [])
    if not waypoints:
        raise ValueError("reorientation_plan must contain at least one waypoint")

    world = reorientation_plan.get("world_config")
    attachment = reorientation_plan.get("attached_object")
    scale = max(1.0, float(reorientation_plan.get("time_scale", 1.0)))
    final_pose: dict[str, Any] | None = None
    for index, waypoint in enumerate(waypoints):
        final_pose = waypoint["pose"]
        mode = waypoint.get("mode")
        if mode is None:
            mode = "planned_linear" if waypoint.get("cartesian", False) else "planned_joint"
        if mode == "contact_transition":
            ctx.tool("robot.go_to_pose_cartesian", pose=final_pose)
            continue
        if mode not in {"planned_joint", "planned_linear"}:
            raise ValueError(f"unsupported reorientation waypoint mode {mode!r}")
        use_attachment = attachment if waypoint.get("use_attachment", True) else None
        attempts = max(1, min(int(waypoint.get("max_attempts", 1)), 3))
        trajectory = None
        for _ in range(attempts):
            planner_tool = "motion.plan_linear" if mode == "planned_linear" else "motion.plan_to_pose"
            inputs = (
                {"end": final_pose, "orientation": "lock"}
                if mode == "planned_linear"
                else {"pose": final_pose}
            )
            inputs.update(
                world_config=world,
                attached_object=use_attachment,
                allow_start_contact=bool(waypoint.get("allow_start_contact", False)),
                allow_goal_contact=bool(waypoint.get("allow_goal_contact", False)),
                contact_margin=float(waypoint.get("contact_margin", 0.005)),
            )
            result = ctx.tool(planner_tool, **inputs)
            trajectory = result.get("trajectory") if result.get("planned") else None
            if trajectory and trajectory.get("waypoints"):
                break
        if not trajectory or not trajectory.get("waypoints"):
            raise RuntimeError(f"collision-aware reorientation failed at waypoint {index} ({mode})")
        ctx.tool("robot.execute_trajectory", trajectory=_time_scale(trajectory, scale))

    assert final_pose is not None
    return {"final_pose": final_pose}
