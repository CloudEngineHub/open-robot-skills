"""Release at the mate, then retreat away from the fixture without sweeping it."""

from typing import Any, TypedDict

from gap import NodeContext
import numpy as np


class Output(TypedDict):
    released: bool


def run(ctx: NodeContext, final_pose: dict[str, Any], retract_m: float = 0.08,
        retreat_axis: dict[str, float] | None = None,
        attached_object: dict[str, Any] | None = None,
        relation: str = "loop_over_shaft") -> Output:
    ctx.tool("robot.open_gripper", settle_steps=80)
    retreat = {
        "position": dict(final_pose["position"]),
        "rotation": dict(final_pose["rotation"]),
    }
    axis = np.array([0.0, 0.0, 1.0])
    if retreat_axis is not None:
        axis = np.array([retreat_axis[k] for k in ("x", "y", "z")], dtype=float)
        axis /= max(float(np.linalg.norm(axis)), 1.0e-12)
        if relation in {"shaft_into_aperture", "tip_through_aperture", "insert_through"}:
            axis = -axis
    spheres = list((attached_object or {}).get("spheres") or [])
    extent = max((float(np.linalg.norm([s["center"][k] for k in ("x", "y", "z")]))
                  + float(s.get("radius", 0.0)) for s in spheres), default=0.0)
    distance = max(float(retract_m), extent + 0.01)
    for key, component in zip(("x", "y", "z"), axis, strict=True):
        retreat["position"][key] = float(retreat["position"][key]) + distance * component
    if retreat_axis is not None:
        retreat["position"]["z"] = float(retreat["position"]["z"]) + 0.5 * distance
    ctx.tool("robot.go_to_pose_cartesian", pose=retreat)
    ctx.tool("robot.wait_steps", steps=120)
    return {"released": True}
