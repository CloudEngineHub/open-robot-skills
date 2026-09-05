"""Release into a container: open the gripper, then a LINEAR retract straight up.

Opens with a settle so the object lands inside the container, then retracts
vertically with ``robot.go_to_pose_cartesian`` (cuRobo ``plan_directed_linear``)
instead of a free-space ``robot.go_home`` swing. Leaving the arm at hover above
the container is fine for a loop: the next grasp's rise/XY segments carry it to
the next object, and an overhead camera sees the table clearly.

The hover height is derived from the placement pose (``hover_z`` left at its
``-1.0`` sentinel → 10 cm above it) rather than being a resting literal for one
workspace, and the retract keeps whichever downward wrist rotation the
transport arrived with (``robot.get_ee_pose``) instead of forcing a canonical
one — a vertical retract that also turns the wrist can fail after the object
has already been released.
"""

from typing import TypedDict

from gap import NodeContext
from gap_core.types import Vec3


class Output(TypedDict):
    done: bool


def run(ctx: NodeContext, place_position: Vec3, hover_z: float = -1.0) -> Output:
    # Open + hold so the released object settles in the container BEFORE retracting.
    ctx.tool("robot.open_gripper", settle_steps=60)
    p = place_position
    # Derive the retract height from the placement pose unless pinned.
    if hover_z < 0.0:
        hover_z = float(p["z"]) + 0.10
    # Preserve whichever downward yaw the transport selected as reachable.
    # Forcing a canonical yaw here turns a vertical retract into an unnecessary
    # wrist rotation and can fail after the object has already been released.
    current_rotation = ctx.tool("robot.get_ee_pose")["pose"]["rotation"]
    # Straight-up retract to hover (Z only).
    ctx.tool(
        "robot.go_to_pose_cartesian",
        pose={
            "position": {"x": float(p["x"]), "y": float(p["y"]), "z": float(hover_z)},
            "rotation": current_rotation,
        },
    )
    return {"done": True}
