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

**The optional grasp profile.** A graph that carries one profile per object
kind (the tool-hanging graphs: ``grasp_profile`` directly, or a
``grasp_profiles`` table keyed by ``target_kind``) can ask for two more things,
both off unless the profile says so, so a graph that binds only ``grasp_pose``
and ``target_obb`` gets exactly the pose and the calls it always got:

- ``use_embodiment_calibration``: turn the grasp so the hand's declared closing
  axis lies along the OBB's shorter horizontal axis, composed through
  ``robot.grasp_frame`` -- the hand's own TCP twist is what makes composing it
  by hand wrong -- and keep the fingertip above the support surface using the
  reach and clearance ``robot.describe_gripper`` states. This is what lets an
  arbitrarily rotated tool be grasped across rather than along.
- ``approach_clearance_m``: a profile-pinned clearance, same meaning as the
  ``clearance`` argument, which it overrides.

``arm_id`` names the hand on a bimanual cell and is threaded as an ABSENT
keyword when unset, for the promotion-parity gate's sake. The second output,
``grasp_pose``, is the (possibly re-oriented and lifted) grasp; without a
profile it is the input, unchanged.
"""

from typing import Any, TypedDict

import numpy as np
from gap import NodeContext
from gap_core.types import OrientedBoundingBox, Se3Pose


class Output(TypedDict):
    align_pose: Se3Pose
    grasp_pose: Se3Pose


def _profile(
    grasp_profile: dict[str, Any] | None,
    target_kind: str,
    grasp_profiles: list[dict[str, Any]],
) -> dict[str, Any]:
    """Resolve a declarative profile without embedding category-specific logic.

    ``grasp_profile`` is the canonical interface. The keyed ``grasp_profiles``
    list is a compatibility adapter for already-materialised graphs and can go
    once their orchestration passes the selected profile directly.
    """
    if grasp_profile is not None:
        return dict(grasp_profile)
    if not target_kind and not grasp_profiles:
        return {}
    profiles = {str(item["source_kind"]): item for item in grasp_profiles}
    if target_kind not in profiles:
        raise ValueError(f"no grasp profile declared for profile key {target_kind!r}")
    return dict(profiles[target_kind])


def _calibrated_grasp(
    ctx: NodeContext,
    grasp_pose: dict[str, Any],
    target_obb: dict[str, Any],
    on_arm: dict[str, Any],
) -> dict[str, Any]:
    """Re-orient across the OBB's short axis and lift the fingertip off the table."""
    from scipy.spatial.transform import Rotation  # noqa: PLC0415

    try:
        gripper = ctx.tool("robot.describe_gripper", **on_arm)
    except Exception:
        # Geometry-only fallback on a connector that exposes no hand metadata:
        # the supplied pose is retained.
        return grasp_pose
    q = target_obb["orientation"]
    obb_rotation = Rotation.from_quat([q["x"], q["y"], q["z"], q["w"]]).as_matrix()
    extents = np.array([target_obb["extent"][k] for k in ("x", "y", "z")], dtype=np.float64)
    vertical = int(np.argmax(np.abs(obb_rotation.T @ np.array([0.0, 0.0, 1.0]))))
    horizontal = [i for i in range(3) if i != vertical]
    short = min(horizontal, key=lambda i: extents[i])
    short_axis = obb_rotation[:, short].copy()
    short_axis[2] = 0.0
    if np.linalg.norm(short_axis) > 1.0e-6 and gripper:
        short_axis /= np.linalg.norm(short_axis)
        # Composed through the embodiment's grasp frame: ``close_axis`` is
        # measured in the physical hand frame while commanded poses are stated
        # at the TCP, and a TCP with a calibrated z twist turns a hand-composed
        # quaternion away from the perceived short axis.
        heading_deg = float(np.degrees(np.arctan2(short_axis[1], short_axis[0])))
        grasp_frame = ctx.tool(
            "robot.grasp_frame",
            approach={"x": 0.0, "y": 0.0, "z": -1.0},
            close_heading_deg=heading_deg,
            **on_arm,
        )
        grasp_pose["rotation"] = dict(grasp_frame["rotation"])
    # Keep the fingertip below the TCP out of the support surface: a thin tool
    # otherwise commands a geometrically centred grasp that jams the jaws on
    # the table before they close. Reach and clearance come from the hand, so
    # the correction survives a hand change.
    finger = gripper.get("finger") or {}
    if finger.get("stated"):
        support_z = float(target_obb["center"]["z"]) - float(target_obb["extent"]["z"])
        min_tcp_z = support_z + float(finger.get("reach_m", 0.0)) + float(finger.get("clearance_m", 0.0))
        grasp_pose["position"]["z"] = max(float(grasp_pose["position"]["z"]), min_tcp_z)
    return grasp_pose


def run(
    ctx: NodeContext,
    grasp_pose: Se3Pose,
    target_obb: OrientedBoundingBox,
    clearance: float = 0.0,
    grasp_profile: dict[str, Any] | None = None,
    arm_id: int | None = None,
    target_kind: str = "",
    grasp_profiles: list[dict[str, Any]] | None = None,
) -> Output:
    profile = _profile(grasp_profile, target_kind, grasp_profiles or [])
    on_arm: dict[str, Any] = {} if arm_id is None else {"arm_id": int(arm_id)}
    grasp: dict[str, Any] = {
        "position": dict(grasp_pose["position"]),
        "rotation": dict(grasp_pose["rotation"]),
    }
    if bool(profile.get("use_embodiment_calibration", bool(target_kind and grasp_profiles))):
        grasp = _calibrated_grasp(ctx, grasp, target_obb, on_arm)

    if "approach_clearance_m" in profile:
        clearance = float(profile["approach_clearance_m"])
    if clearance <= 0.0:
        clearance = float(ctx.tool("robot.describe_workspace", **on_arm)["align_clearance_m"])
    obb_top = float(target_obb["center"]["z"]) + float(target_obb["extent"]["z"])
    grasp_z = float(grasp["position"]["z"])
    approach_z = max(obb_top, grasp_z) + clearance
    align_pose: Se3Pose = {
        "position": {
            "x": grasp["position"]["x"],
            "y": grasp["position"]["y"],
            "z": approach_z,
        },
        "rotation": grasp["rotation"],
    }
    return {"align_pose": align_pose, "grasp_pose": grasp}  # type: ignore[typeddict-item]
