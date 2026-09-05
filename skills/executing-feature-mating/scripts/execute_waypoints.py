"""Execute an ordered, collision-aware feature-mating waypoint sequence."""

import math
from typing import Any, TypedDict

from gap import NodeContext


class Output(TypedDict):
    final_pose: dict[str, Any]
    waypoint_count: int


def _pose_reached(current: dict[str, Any], target: dict[str, Any]) -> bool:
    cp, tp = current["position"], target["position"]
    distance = math.sqrt(sum((float(cp[k]) - float(tp[k])) ** 2 for k in ("x", "y", "z")))
    cq, tq = current["rotation"], target["rotation"]
    dot = abs(sum(float(cq[k]) * float(tq[k]) for k in ("w", "x", "y", "z")))
    return distance <= 0.01 and dot >= 0.995


def run(ctx: NodeContext, placement_plan: dict[str, Any]) -> Output:
    waypoints = list(placement_plan.get("waypoints") or [])
    if not waypoints:
        raise ValueError("placement_waypoints must contain at least one pose")

    world = placement_plan.get("world_config")
    attachment = placement_plan.get("attached_object")
    final_pose: dict[str, Any] | None = None
    for index, item in enumerate(waypoints):
        wrapped = "pose" in item
        pose = item["pose"] if wrapped else item
        if not isinstance(pose, dict) or "position" not in pose or "rotation" not in pose:
            raise ValueError(f"invalid placement waypoint: {item!r}")
        mode = item.get("mode") if wrapped else None
        if mode is None:
            mode = "planned_linear" if wrapped and item.get("cartesian", False) else "planned_joint"
        if mode == "contact_seat":
            ctx.tool("robot.move_cartesian_until_contact", pose=pose)
            final_pose = pose
            continue
        if mode not in {"planned_joint", "planned_linear"}:
            raise ValueError(f"unsupported fixture waypoint mode {mode!r}")
        current = ctx.tool("robot.get_ee_pose")["pose"]
        if _pose_reached(current, pose):
            final_pose = pose
            continue
        attempts = max(1, min(int(item.get("max_attempts", 1)), 3))
        trajectory = None
        for _ in range(attempts):
            tool = "motion.plan_linear" if mode == "planned_linear" else "motion.plan_to_pose"
            inputs = ({"end": pose, "orientation": "lock"}
                      if mode == "planned_linear" else {"pose": pose})
            inputs.update(
                world_config=world,
                attached_object=attachment if item.get("use_attachment", True) else None,
                allow_start_contact=bool(item.get("allow_start_contact", False)),
                allow_goal_contact=bool(item.get("allow_goal_contact", False)),
                contact_margin=float(item.get("contact_margin", 0.005)),
            )
            result = ctx.tool(tool, **inputs)
            trajectory = result.get("trajectory") if result.get("planned") else None
            if trajectory and trajectory.get("waypoints"):
                break
        if not trajectory or not trajectory.get("waypoints"):
            raise RuntimeError(f"collision-aware fixture motion failed at waypoint {index} ({mode})")
        ctx.tool("robot.execute_trajectory", trajectory=trajectory)
        final_pose = pose

    assert final_pose is not None
    return {"final_pose": final_pose, "waypoint_count": len(waypoints)}
