"""Compute the align-pose for direct-IK grasping.

Constructs an SE(3) pose above the grasp with the grasp's own rotation, at
``(grasp_pose.x, grasp_pose.y, max(obb_top, grasp_z) + clearance)``. Used by
the ``grasping-direct-ik`` skill: the gripper rotates into the grasp
orientation at this align-pose first, then descends straight down to the
actual grasp pose.

``clearance`` is asked for when it is not given: ``robot.describe_workspace()``
reports ``align_clearance_m`` -- the fingers' reach plus the housing standing
above them plus the pad thickness plus a margin, off the hand that is actually
on the arm. A hand with a taller housing gets a taller clearance instead of a
collision. Pass ``clearance`` explicitly to pin it, which is what a workflow
does when the *held* object rather than the hand is what needs the room (a
long tool hanging below the fingertips).

The clearance is measured from whichever is higher, the OBB's top face or the
grasp itself. For a grasp inside the box (the usual case) that is the top
face; for a grasp lifted above it -- a thin object whose grasp was clamped to
a fingertip floor -- the hover still stands the full clearance above the
fingers rather than dipping toward them.
"""

from typing import TypedDict

from gap import NodeContext
from gap_core.types import OrientedBoundingBox, Se3Pose


class Output(TypedDict):
    align_pose: Se3Pose


def run(
    ctx: NodeContext,
    grasp_pose: Se3Pose,
    target_obb: OrientedBoundingBox,
    clearance: float = 0.0,
) -> Output:
    if clearance <= 0.0:
        clearance = float(ctx.tool("robot.describe_workspace")["align_clearance_m"])
    obb_top = float(target_obb["center"]["z"]) + float(target_obb["extent"]["z"])
    grasp_z = float(grasp_pose["position"]["z"])
    approach_z = max(obb_top, grasp_z) + clearance
    align_pose: Se3Pose = {
        "position": {
            "x": grasp_pose["position"]["x"],
            "y": grasp_pose["position"]["y"],
            "z": approach_z,
        },
        "rotation": grasp_pose["rotation"],
    }
    return {"align_pose": align_pose}
