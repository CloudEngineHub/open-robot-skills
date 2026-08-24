"""Turn typed feature-mate geometry into a typed fixture-engagement plan."""
from typing import Any, TypedDict
from gap import NodeContext

class Output(TypedDict):
    placement_plan: dict[str, Any]

def run(ctx: NodeContext, approach_pose: dict[str, Any], engaged_pose: dict[str, Any],
        mate_pose: dict[str, Any], relation: str, world_config: dict[str, Any],
        attached_object: dict[str, Any]) -> Output:
    del ctx
    supported = {"loop_over_shaft", "shaft_into_aperture", "tip_through_aperture",
                 "insert_through", "feature_to_fixture"}
    if relation not in supported:
        raise ValueError(f"unsupported feature relation {relation!r}")
    waypoints = [{"pose": approach_pose, "mode": "planned_joint"}]
    if relation in {"loop_over_shaft", "feature_to_fixture"}:
        # Crossing a peg/hook is the intended contact operation. A free-space
        # planner may correctly classify the engaged endpoint as collision;
        # execute the short local legs with the bounded contact servo instead.
        waypoints.append({"pose": engaged_pose, "mode": "contact_seat"})
        waypoints.append({"pose": mate_pose, "mode": "contact_seat"})
    else:
        waypoints.append({"pose": engaged_pose, "mode": "planned_linear",
                          "allow_goal_contact": True})
        waypoints.append({"pose": mate_pose, "mode": "planned_linear",
                          "allow_goal_contact": True})
    return {"placement_plan": {"relation": relation, "waypoints": waypoints,
                               "world_config": world_config,
                               "attached_object": attached_object}}
