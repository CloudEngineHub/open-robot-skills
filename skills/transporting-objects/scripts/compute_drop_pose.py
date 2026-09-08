"""Compute a safe drop pose and approach pose above a container OBB.

Given a container's oriented bounding box, produce:
- ``drop_position`` — Vec3 at the container's XY center, slightly above its top
- ``drop_pose`` — Se3Pose at the drop position with a top-down gripper orientation
- ``approach_pose`` — Se3Pose well above the container for a safe lateral approach

The top-down orientation is this hand's own (``robot.grasp_frame``), which
points the gripper straight down regardless of the container's orientation —
except when ``ee_pose_at_grasp`` is supplied, in which case the grasp-time
wrist YAW is preserved (see ``_yaw_only_topdown``). The wrist-to-TCP distance
the Z math needs is read off ``robot.describe_arm`` / ``robot.describe_gripper``
unless ``wrist_to_tcp`` pins it.
"""

from typing import TypedDict

from gap import NodeContext
from gap_core.types import OrientedBoundingBox, Quaternion, Se3Pose, Vec3


class Output(TypedDict):
    drop_position: Vec3
    drop_pose: Se3Pose
    approach_pose: Se3Pose


def _yaw_only_topdown(ctx: NodeContext, ee_pose: Se3Pose) -> Quaternion:
    """A top-down grasp frame carrying only the world-z heading of ``ee_pose``.

    Aligns the drop wrist with whatever yaw the grasp acquired -- so the planner
    does not unspool it mid-transport -- while keeping pitch and roll strictly
    top-down, so the held object lands flat however angled the grasp was.

    **Both halves used to be one hand's arithmetic.** The heading was read off
    the rotated matrix's *first column*, which is the closing axis only on a
    hand whose closing axis is tool-local x; and the frame was rebuilt as
    ``R_z(yaw) · R_x(pi)``, where ``R_x(pi)`` is that same hand's top-down. On a
    hand whose grasp frame is a half turn from its wrist, the first is the
    wrong axis and the second is 180 degrees out.

    So the heading is measured against the axis the hand *says* it closes along,
    and the frame is asked for rather than composed: ``robot.grasp_frame`` takes a
    world heading for the jaws and returns the orientation that puts this hand's
    measured axes there.
    """
    import math

    from scipy.spatial.transform import Rotation as _R

    q = ee_pose["rotation"]
    # gap Quaternion is wxyz; scipy expects (x, y, z, w).
    rotation = _R.from_quat([q["x"], q["y"], q["z"], q["w"]]).as_matrix()
    close = ctx.tool("robot.describe_gripper")["close_axis"]
    world_close = rotation @ [float(close["x"]), float(close["y"]), float(close["z"])]
    yaw = math.atan2(float(world_close[1]), float(world_close[0]))
    return ctx.tool("robot.grasp_frame", close_heading_deg=math.degrees(yaw))["rotation"]


def run(
    ctx: NodeContext,
    container_obb: OrientedBoundingBox,
    container_interior_obb: OrientedBoundingBox | None = None,
    ee_pose_at_grasp: Se3Pose | None = None,
    drop_clearance: float = 0.05,
    approach_height: float = 0.20,
    held_obb: OrientedBoundingBox | None = None,
    wrist_to_tcp: float = -1.0,
) -> Output:
    # What this used to be: ``panda_hand_to_tcp = 0.1029``, subtracted from a
    # height that had already been measured at the tool centre -- one hand's
    # link-to-TCP distance from a stack whose observation returned the *hand
    # link* pose. Asked for instead: the hand's own tool-centre offset along
    # its own approach axis, which is 0 for any hand whose TCP body *is* the
    # point it reports.
    if wrist_to_tcp < 0.0:
        arm = ctx.tool("robot.describe_arm")
        gripper = ctx.tool("robot.describe_gripper")
        offset, approach = arm["tcp_offset"], gripper["approach_axis"]
        wrist_to_tcp = abs(sum(float(offset[k]) * float(approach[k]) for k in ("x", "y", "z")))
    # Container-top reference (always defined). Used as the descent
    # target for the no-held-geometry fallback so the gripper releases
    # *just above* the rim — the legacy contract callers without
    # ``held_obb`` still rely on.
    container_top = container_obb["center"]["z"] + container_obb["extent"]["z"]

    # Target the interior placement zone when available; otherwise fall
    # back to the exterior top.
    if container_interior_obb is not None:
        zone_center = container_interior_obb["center"]
        zone_floor = zone_center["z"] - container_interior_obb["extent"]["z"]
        zone_ceiling = zone_center["z"] + container_interior_obb["extent"]["z"]
    else:
        zone_center = container_obb["center"]
        zone_floor = container_top  # exterior top
        zone_ceiling = zone_floor + 0.10  # arbitrary headroom for fallback path

    # Measure the at-grasp EE height LIVE. This node runs right after the
    # grasp subgraph closes the gripper and before any lift, so the
    # current EE pose IS the at-grasp pose the invariant below requires.
    # Generated graphs typically wire ``ee_pose_at_grasp`` from an
    # ``observe`` node that runs at the pre-grasp HOVER (between approach
    # and plan) — recorded traces show that observation ~0.12-0.20 m above
    # the true at-grasp height, which inflates ``ee_to_obj_z`` and made
    # the release happen ~30 cm above the rim (items bounced out of the
    # basket or rolled away on landing). The wired value is kept for yaw
    # preservation and as a fallback when the live read fails.
    ee_z_at_grasp: float | None = None
    try:
        live = ctx.tool("robot.get_ee_pose")
        ee_z_at_grasp = float(live["pose"]["position"]["z"])
    except Exception:
        if ee_pose_at_grasp is not None:
            ee_z_at_grasp = float(ee_pose_at_grasp["position"]["z"])

    if held_obb is not None and ee_z_at_grasp is not None:
        # A containment predicate checks that the object's CENTER is inside
        # the container region's 3D AABB. Target a held-object Z that:
        #   (a) keeps the held object's BOTTOM clear of the basket walls
        #       (modelled as a 2 cm shallow lip at the basket floor) so
        #       cuRobo's collision check passes during the descent,
        #   (b) keeps the held object's CENTER strictly inside the zone
        #       (so the predicate fires after release), and
        #   (c) gives a TCP target that's well within the arm's reachable
        #       volume — short objects need a higher held center than the
        #       wall-clearance floor would suggest, otherwise the end-leg IK
        #       has too few feasible joint configurations.
        margin = max(0.03, drop_clearance)  # 3 cm above wall top
        desired_obj_z = zone_floor + margin + held_obb["extent"]["z"]
        # Hard ceiling: never push the held object's center past the zone
        # top — the containment predicate would fail.
        if desired_obj_z > zone_ceiling - 0.001:
            desired_obj_z = zone_ceiling - 0.001
        # Soft floor: if the zone is so shallow that the bottom margin
        # forces the center past the ceiling, give up on the margin.
        if desired_obj_z < zone_center["z"]:
            desired_obj_z = max(desired_obj_z, zone_floor + 0.001 + held_obb["extent"]["z"])
        # The held cuboid is rigidly attached to the EE link. With a
        # top-down grip, the world Z difference between EE and the
        # held-object center is preserved across the trajectory:
        #   ee_z_at_drop - held_z_at_drop == ee_z_at_grasp - obj_z_at_grasp
        # ``held_obb["center"]["z"]`` IS ``obj_z_at_grasp`` because the OBB
        # was captured before the gripper closed; ``ee_z_at_grasp`` is the
        # live measurement taken above.
        ee_to_obj_z = ee_z_at_grasp - held_obb["center"]["z"]
        ee_z_at_drop = desired_obj_z + ee_to_obj_z
        tcp_z = ee_z_at_drop - wrist_to_tcp
    elif held_obb is not None:
        # Fallback: assume cuRobo's primary grasp_z_offset (0.04). Less
        # accurate but doesn't require ee_pose_at_grasp.
        desired_obj_z = max(
            zone_center["z"],
            zone_floor + drop_clearance + held_obb["extent"]["z"],
        )
        tcp_z = desired_obj_z + held_obb["extent"]["z"] - 0.04
    else:
        # No held geometry: place TCP just above the container top by
        # ``drop_clearance``. This matches the pre-redesign contract so
        # legacy callers (static workflows that don't pass ``held_obb``)
        # keep their behavior — the ``descend_release`` script
        # go_to_pose's *to* this height before opening the gripper, so it
        # must be a real release height near the rim, not a
        # high-headroom waypoint.
        tcp_z = container_top + drop_clearance

    # Preserve only the grasp-time wrist YAW (z-axis rotation) — keep
    # pitch/roll top-down. Without this the drop uses a fixed yaw=0
    # top-down frame, forcing curobo to unspool the wrist
    # yaw the gripper acquired to grip a non-axis-aligned subpart (e.g.
    # a horizontal frypan handle); the planner swings the arm through
    # a redundant-joint reconfiguration to do it — visible as a
    # circular elbow motion between pick and place. Preserving the FULL
    # grasp rotation (yaw + pitch + roll) is too aggressive: it carries
    # any grasp-angle tilt into the drop, so a slanted grasp ends up
    # releasing the held object on edge. Yaw-only keeps pitch/roll at
    # top-down so the held object lands flat regardless of how the
    # grasp was angled.
    drop_rotation = (
        _yaw_only_topdown(ctx, ee_pose_at_grasp)
        if ee_pose_at_grasp is not None
        else ctx.tool("robot.grasp_frame")["rotation"]
    )
    drop_position: Vec3 = {
        "x": zone_center["x"],
        "y": zone_center["y"],
        "z": tcp_z,
    }
    drop_pose: Se3Pose = {"position": drop_position, "rotation": drop_rotation}
    approach_pose: Se3Pose = {
        "position": {
            "x": zone_center["x"],
            "y": zone_center["y"],
            "z": tcp_z + approach_height,
        },
        "rotation": drop_rotation,
    }
    return {
        "drop_position": drop_position,
        "drop_pose": drop_pose,
        "approach_pose": approach_pose,
    }
