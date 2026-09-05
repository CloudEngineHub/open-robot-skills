"""Release at the mate, then retreat away from the fixture without sweeping it."""

from typing import Any, TypedDict

import numpy as np
from gap import NodeContext

#: Minimum retreat when the caller gives no ``retract_m``.
_DEFAULT_RETRACT_M = 0.08
#: Extra clearance past the farthest attached sphere so the opened gripper
#: leaves the released object's envelope.
_CLEARANCE_MARGIN_M = 0.01
#: Relations where the held feature went *into* the fixture: retreating along
#: the fixture axis would push further in, so the retreat is reversed.
_INSERTED_RELATIONS = frozenset({"shaft_into_aperture", "tip_through_aperture", "insert_through"})


class Output(TypedDict):
    released: bool


def run(
    ctx: NodeContext,
    final_pose: dict[str, Any],
    retract_m: float = 0.0,
    retreat_axis: dict[str, float] | None = None,
    attached_object: dict[str, Any] | None = None,
    relation: str = "loop_over_shaft",
    open_settle_steps: int = 80,
    settle_steps: int = 120,
) -> Output:
    ctx.tool("robot.open_gripper", settle_steps=int(open_settle_steps))
    retreat = {
        "position": dict(final_pose["position"]),
        "rotation": dict(final_pose["rotation"]),
    }
    axis = np.array([0.0, 0.0, 1.0])
    if retreat_axis is not None:
        axis = np.array([retreat_axis[k] for k in ("x", "y", "z")], dtype=float)
        axis /= max(float(np.linalg.norm(axis)), 1.0e-12)
        if relation in _INSERTED_RELATIONS:
            axis = -axis
    spheres = list((attached_object or {}).get("spheres") or [])
    extent = max(
        (
            float(np.linalg.norm([s["center"][k] for k in ("x", "y", "z")]))
            + float(s.get("radius", 0.0))
            for s in spheres
        ),
        default=0.0,
    )
    minimum = float(retract_m) if float(retract_m) > 0.0 else _DEFAULT_RETRACT_M
    distance = max(minimum, extent + _CLEARANCE_MARGIN_M)
    for key, component in zip(("x", "y", "z"), axis, strict=True):
        retreat["position"][key] = float(
            float(retreat["position"][key]) + distance * float(component)
        )
    if retreat_axis is not None:
        # Lift while backing off along the fixture axis so the open fingers
        # clear the released object instead of dragging it along the fixture.
        retreat["position"]["z"] = float(retreat["position"]["z"]) + 0.5 * distance
    ctx.tool("robot.go_to_pose_cartesian", pose=retreat)
    # Let the released object come to rest before the graph reports success:
    # a freshly released object can still be swinging on its fixture.
    ctx.tool("robot.wait_steps", steps=int(settle_steps))
    return {"released": True}
