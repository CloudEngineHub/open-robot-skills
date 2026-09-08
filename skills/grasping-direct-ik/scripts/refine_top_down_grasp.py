"""Refine a top-down grasp against the hand that is actually on the arm.

Two corrections, both read off ``robot.describe_gripper`` rather than assumed:

1. **Close-axis rotation.** The grasp rotation is rebuilt so the hand's
   declared closing axis lies along the shorter horizontal axis of the target
   OBB. ``robot.grasp_frame`` composes that orientation from the hand's
   measured approach and close axes and its TCP twist, so it is right for a
   hand that closes along tool-local x as well as one that closes along
   tool-local y -- a literal quaternion is right for exactly one of them, and
   a hand whose TCP carries a twist would otherwise lock the jaws a quarter
   turn away from the object. Working from the OBB's own axes also handles an
   arbitrarily rotated object and avoids taking the first (world-aligned) pose
   of a generic grasp-candidate fan.
2. **Fingertip floor.** The TCP is kept high enough that the fingertips below
   it do not enter the support surface: ``support_z + finger.reach_m +
   finger.clearance_m``. Thin objects otherwise command a geometrically
   centred grasp that jams the jaws against the table before they can close.
   Applied only when the hand states its finger envelope, so a hand that
   never measured it is left alone rather than trusted to a zero.

Returns the refined ``grasp_pose``. Feed it to ``compute_align_pose`` (the
pre-rotated hover) and to the descend state, so hover and grasp share one
rotation.
"""

from typing import TypedDict

import numpy as np
from gap import NodeContext
from gap_core.types import OrientedBoundingBox, Se3Pose
from scipy.spatial.transform import Rotation


class Output(TypedDict):
    grasp_pose: Se3Pose


def run(
    ctx: NodeContext,
    grasp_pose: Se3Pose,
    target_obb: OrientedBoundingBox,
) -> Output:
    refined: Se3Pose = {
        "position": dict(grasp_pose["position"]),
        "rotation": dict(grasp_pose["rotation"]),
    }
    gripper = ctx.tool("robot.describe_gripper")

    # The OBB's shorter horizontal axis is where the jaws should close.
    q = target_obb["orientation"]
    obb_rotation = Rotation.from_quat([q["x"], q["y"], q["z"], q["w"]]).as_matrix()
    extents = np.array([target_obb["extent"][k] for k in ("x", "y", "z")], dtype=np.float64)
    vertical = int(np.argmax(np.abs(obb_rotation.T @ np.array([0.0, 0.0, 1.0]))))
    horizontal = [i for i in range(3) if i != vertical]
    short = min(horizontal, key=lambda i: extents[i])
    short_axis = obb_rotation[:, short].copy()
    short_axis[2] = 0.0
    if np.linalg.norm(short_axis) > 1.0e-6:
        short_axis /= np.linalg.norm(short_axis)
        # ``close_axis`` is measured in the physical hand frame while commanded
        # poses are stated at the TCP; the connector composes the two, so the
        # heading is asked for as a world direction rather than rotated by hand.
        heading_deg = float(np.degrees(np.arctan2(short_axis[1], short_axis[0])))
        grasp_frame = ctx.tool(
            "robot.grasp_frame",
            approach={"x": 0.0, "y": 0.0, "z": -1.0},
            close_heading_deg=heading_deg,
        )
        refined["rotation"] = dict(grasp_frame["rotation"])

    # Keep the fingertips below the TCP out of the support surface.
    finger = gripper.get("finger") or {}
    if finger.get("stated"):
        support_z = float(target_obb["center"]["z"]) - float(target_obb["extent"]["z"])
        min_tcp_z = (
            support_z + float(finger.get("reach_m", 0.0)) + float(finger.get("clearance_m", 0.0))
        )
        refined["position"]["z"] = max(float(refined["position"]["z"]), min_tcp_z)
    return {"grasp_pose": refined}
