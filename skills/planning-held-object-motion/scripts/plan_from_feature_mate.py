"""Turn a feature-mate approach pose into a direct held-object carry plan."""

from typing import Any, TypedDict

from gap import NodeContext


class Output(TypedDict):
    reorientation_plan: dict[str, Any]


def run(ctx: NodeContext, approach_pose: dict[str, Any], world_config: dict[str, Any],
        attached_object: dict[str, Any]) -> Output:
    """Plan directly to the requested fixture approach pose."""
    del ctx
    return {"reorientation_plan": {
        "time_scale": 2.0,
        "waypoints": [{"pose": approach_pose, "mode": "planned_joint", "cartesian": False}],
        "world_config": world_config,
        "attached_object": attached_object,
    }}
