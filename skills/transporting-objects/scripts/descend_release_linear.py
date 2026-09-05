"""Linear-descent variant of ``descend_release`` using the connector's TCP-aware
``robot.go_to_pose_cartesian``.

Drop-in replacement for ``descend_release.py`` — same node-level contract:
takes ``drop_position`` (and optionally ``drop_rotation``), descends, opens
the gripper, retreats a short way straight up, retracts home.

Routes through ``robot.go_to_pose_cartesian``: this applies the configured TCP
offset / TCP rotation, plans a straight Cartesian line at the IK link, and
falls back internally to a single-pose collision-aware plan when the linear
plan cannot solve. A ``curobo.plan_linear`` bundle call would interpret
``drop_position`` as a link target and silently drop the TCP offset — a
source of vertical misses when the workflow really did mean "put the
fingertips at this XYZ".

When ``drop_rotation`` is not given, "straight down" is asked of the live hand
(``robot.grasp_frame`` with no arguments) rather than written as a literal: the
literal this replaces was top-down for a hand whose approach axis is its
tool-local +z and a half turn wrong for one whose is not.

After the release the wrist keeps its orientation and takes one short (5 cm)
Cartesian retreat, then settles a few steps, before the large return-home
motion, so the fingers cannot strike or drag the just-released object.

When to use this variant:
- For ANY subpart-grasp + place-ON task (frypan handle → stove, kettle
  spout → trivet), and for a release above a fixture the object was engaged
  with. The yaw-preserving ``compute_drop_pose`` + linear descent combination
  gives the cleanest release dynamics.
- A repair pass can flip ``descend_release`` → ``descend_release_linear``
  when the place stage shows visible release artefacts (object tilting on
  touch-down) even though the geometric drop pose is correct.

When NOT to use it:
- When the descent path needs to avoid an obstacle. ``go_to_pose_cartesian``
  refuses to plan around obstacles — it only checks the straight line and
  falls back to a collision-free single-pose plan if the line fails. Use
  the ``descend_release.py`` variant for cluttered scenes that need
  smoother IK-only descent.

Inputs:
- ``drop_position`` (Vec3): TCP target XYZ (fingertip position), same as
  descend_release.
- ``drop_rotation`` (Quaternion, optional): TCP orientation held through
  the descent. Defaults to this hand's top-down grasp frame.
"""

from typing import TypedDict

from gap import NodeContext
from gap_core.types import Quaternion, Se3Pose, Vec3


class Output(TypedDict):
    drop_position: Vec3


def run(
    ctx: NodeContext,
    drop_position: Vec3,
    drop_rotation: Quaternion | None = None,
) -> Output:
    # Straight down for *this* hand, not for a hand whose approach axis happens
    # to be its tool-local +z. `robot.grasp_frame` with no arguments composes it
    # from the live hand's measured axes.
    rotation = (
        drop_rotation if drop_rotation is not None else ctx.tool("robot.grasp_frame")["rotation"]
    )
    end_pose: Se3Pose = {"position": drop_position, "rotation": rotation}
    ctx.tool("robot.go_to_pose_cartesian", pose=end_pose)
    # Open + settle long enough for the object to land before anything moves.
    ctx.tool("robot.open_gripper", settle_steps=60)
    # Clear the released object and container before the large return-home
    # motion. Preserve the wrist orientation and use one short Cartesian IK
    # retreat so the fingers cannot strike or drag the newly released object.
    current = ctx.tool("robot.get_ee_pose")["pose"]
    retreat_position = dict(current["position"])
    retreat_position["z"] = float(retreat_position["z"]) + 0.05
    retreat_pose: Se3Pose = {
        "position": retreat_position,
        "rotation": dict(current["rotation"]),
    }
    ctx.tool("robot.go_to_pose_cartesian", pose=retreat_pose)
    ctx.tool("robot.wait_steps", steps=12)
    ctx.tool("robot.go_home")
    return {"drop_position": drop_position}
