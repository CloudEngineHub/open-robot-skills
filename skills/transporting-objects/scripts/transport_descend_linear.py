"""Straight-Z placement INTO a walled container: lift (Z) → XY over the
container → LINEAR descend to inside the walls.

The generalized form of the reference packing recipe, for any container with
an interior (basket, bin, box, tote). Lift/XY are straight-line cartesian
(``robot.go_to_pose_cartesian``); the descend uses cuRobo's AXIS-CONSTRAINED
linear move (``curobo.plan_directed_linear``, ``allowed_axes=["Z"]``,
``orientation_mode="LOCK"``) — a guaranteed straight vertical drop. The descend
lowers the TCP to ``container_top − place_offset`` (with ``place_offset`` small
or negative to sit just above the rim so the held object clears the walls on the
way down and drops in) so the object is placed IN the container, not from a
lateral swing. For placement ONTO a surface or into a described sub-region/zone,
use the ``compute_drop_pose → waypoint_move → descend_release_linear`` path
instead — this script is the walled-container fast path.

Heights are derived, not resting literals: with ``lift_z`` / ``transport_z``
left at their ``-1.0`` sentinels both come from the perceived container's rim
and the wrist's current height, so the same script places on a table at any
height. Every cartesian leg is checked against ``robot.get_ee_pose`` afterwards
(a servo that stops 3 cm short is a leg that did not happen), and the lateral
leg searches the container's reachable interior when its centre is out of
reach.
"""

import logging
import math
from typing import TypedDict

from gap import NodeContext
from gap_core.types import OrientedBoundingBox, Quaternion, Vec3

logger = logging.getLogger(__name__)


class Output(TypedDict):
    place_position: Vec3


def _cartesian(ctx: NodeContext, x: float, y: float, z: float, rotation: Quaternion) -> None:
    pose = {"position": {"x": float(x), "y": float(y), "z": float(z)}, "rotation": rotation}
    try:
        ctx.tool("robot.go_to_pose_cartesian", pose=pose)
    except Exception as exc:  # noqa: BLE001
        # A Cartesian servo can hit a joint-limit basin even when another arm
        # configuration reaches the same clear-space pose. Use the reusable
        # CuRobo single-pose planner for a genuine joint-space reconfiguration.
        logger.warning(
            "[transport_move] straight-line leg to (%.3f, %.3f, %.3f) failed "
            "(%s); falling back to curobo.plan_to_pose.",
            x,
            y,
            z,
            exc,
        )
        observation = ctx.tool("robot.get_observation")
        plan = ctx.tool(
            "curobo.plan_to_pose",
            target_pose=pose,
            start_joint_position=observation["arms"][0]["joint_state"],
        )
        if not plan.get("success") or not plan.get("trajectory"):
            raise RuntimeError("CuRobo could not plan the clear-space transport leg") from exc
        ctx.tool(
            "robot.execute_trajectory", trajectory=plan["trajectory"], max_steps_per_waypoint=60
        )
    reached = ctx.tool("robot.get_ee_pose")["pose"]["position"]
    error = math.sqrt(
        (float(reached["x"]) - float(x)) ** 2
        + (float(reached["y"]) - float(y)) ** 2
        + (float(reached["z"]) - float(z)) ** 2
    )
    if error > 0.03:
        raise RuntimeError(
            f"transport leg did not reach its requested pose: error={error:.3f} m, "
            f"requested=({x:.3f}, {y:.3f}, {z:.3f}), "
            f"reached=({reached['x']:.3f}, {reached['y']:.3f}, {reached['z']:.3f})"
        )


def _descend_linear(ctx: NodeContext, target_z: float, from_z: float) -> None:
    # FINGERTIP-frame heights. Distance is from_z - target_z, NOT
    # get_ee_pose().z - target_z: on a hand whose reported link sits above the
    # fingertip, the latter overshoots the descent by the TCP offset and rams
    # the object into the container floor.
    dist = float(from_z) - float(target_z)
    if dist <= 0.002:
        return
    js = ctx.tool("robot.get_observation")["arms"][0]["joint_state"]
    res = ctx.tool(
        "curobo.plan_directed_linear",
        start_joint_position=js,
        endpoint_mode="DISTANCE",
        explicit_direction={"x": 0.0, "y": 0.0, "z": -1.0},
        distance=dist,
        allowed_axes=["Z"],
        orientation_mode="LOCK",
    )
    if res.get("success") and res.get("trajectory"):
        ctx.tool(
            "robot.execute_trajectory", trajectory=res["trajectory"], max_steps_per_waypoint=60
        )
    else:
        ee_pose = ctx.tool("robot.get_ee_pose")["pose"]
        ee = ee_pose["position"]
        _cartesian(ctx, ee["x"], ee["y"], target_z, ee_pose["rotation"])


def _plan_reachable_interior(
    ctx: NodeContext,
    center: Vec3,
    extent: Vec3,
    z: float,
    preferred_x: float,
    preferred_y: float,
    rotation: Quaternion,
) -> tuple[float, float]:
    """Move to a reachable point inside the perceived destination footprint."""
    margin = 0.015
    xr = max(0.0, float(extent["x"]) - margin)
    yr = max(0.0, float(extent["y"]) - margin)
    cx, cy = float(center["x"]), float(center["y"])
    candidates = [
        (preferred_x, preferred_y),
        (cx - xr, cy - yr),
        (cx - xr, cy + yr),
        (cx + xr, cy - yr),
        (cx + xr, cy + yr),
        (cx, cy),
    ]
    seen: set[tuple[float, float]] = set()
    for x, y in candidates:
        key = (round(x, 5), round(y, 5))
        if key in seen:
            continue
        seen.add(key)
        try:
            _cartesian(ctx, x, y, z, rotation)
            return x, y
        except Exception:  # noqa: BLE001
            logger.debug("destination candidate (%.3f, %.3f) unreachable", x, y)
    raise RuntimeError("CuRobo found no reachable point inside the destination OBB")


def run(
    ctx: NodeContext,
    container_obb: OrientedBoundingBox,
    transport_z: float = -1.0,
    lift_z: float = -1.0,
    place_offset: float = 0.06,
) -> Output:
    c = container_obb["center"]
    e = container_obb["extent"]
    place_z = float(c["z"]) + float(e["z"]) - float(place_offset)
    current_pose = ctx.tool("robot.get_ee_pose")["pose"]
    cur = current_pose["position"]
    # Preserve the current grasp orientation throughout lift, transfer, and
    # release; do not add a task-irrelevant yaw over a destination box.
    transport_rotation = current_pose["rotation"]
    # Prefer the container centre. Perceived masks can slightly overestimate
    # a compartment footprint; clamping the current hand position into that
    # footprint biases release toward a divider and can look like a label swap.
    # _plan_reachable_interior still tries alternative interior points if this
    # centre pose is genuinely unreachable.
    bx = float(c["x"])
    by = float(c["y"])
    # Derive safe heights from perceived geometry unless the graph supplies
    # explicit world-frame heights. A resting literal belongs to one workspace
    # and is below (or far above) the work surface of the next.
    if lift_z < 0.0:
        # Do not add clearance blindly above an already-clear grasp pose: that
        # pushes a fixed-down wrist to the upper workspace boundary. Seven
        # centimetres above the release pose clears the rim; otherwise retain
        # the current height so the lateral leg never descends.
        lift_z = max(place_z + 0.07, min(float(cur["z"]), place_z + 0.08))
    if transport_z < 0.0:
        # The release pose can itself be above the rim (negative place_offset),
        # so clearance must be measured from place_z rather than added twice.
        # Eight centimetres gives a held object a little room over the source
        # container and any dividers on the way across.
        transport_z = place_z + 0.08
    # First lift the grasped item straight up so it clears the table before the
    # lateral move (avoids dragging it across the scene). lift_z is kept BELOW
    # transport_z on purpose, but even a MODEST in-place lift can be infeasible
    # at a far-edge grasp XY: the arm grasps low there yet has almost no
    # vertical room, so BOTH the straight-line and the planned solve fail. So
    # the lift is BEST-EFFORT: if it can't be solved we skip it and go straight
    # up-and-over -- the lateral leg lifts the item anyway as it moves toward
    # the central, reachable container. The lift must never abort the place;
    # only the lateral leg and the descend are essential.
    try:
        _cartesian(ctx, cur["x"], cur["y"], lift_z, transport_rotation)
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "[transport_move] in-place lift to z=%.3f infeasible at (%.3f, %.3f) "
            "(%s); skipping lift, going straight up-and-over.",
            lift_z,
            cur["x"],
            cur["y"],
            exc,
        )
    bx, by = _plan_reachable_interior(
        ctx,
        c,
        e,
        transport_z,
        bx,
        by,
        transport_rotation,
    )  # up & over the container
    _descend_linear(ctx, place_z, transport_z)  # descend INTO it (constrained Z)
    return {"place_position": {"x": bx, "y": by, "z": place_z}}
