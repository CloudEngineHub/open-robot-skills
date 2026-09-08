"""Did the hand actually arrive? If not, do not close the jaws on air.

**The failure this exists for.** A graph drives the tool to a computed grasp
pose and then closes. A position servo typically gives up only after several
*consecutive* solve failures and resets that counter on any success, so an arm
that crawls to a halt short of the goal returns normally, having never failed
often enough in a row. The close then happens wherever the hand stopped.

Measured on a sorting scene, three attempts at a small cylindrical cell::

    commanded (-0.0027, -0.3730, 0.8187)  actual (0.0135, -0.3786, 0.8352)
    error (+16.2, -5.6, +16.5) mm

The cell is 14.8 mm across. Sixteen millimetres of lateral error puts the pads
entirely beside it, and the width check after the close read an empty hand on
all three attempts. The same episode then reported a planner refusal to place
the object -- an empty hand failing to place nothing, a motion failure that is
really this one wearing a disguise.

Grasps that work do not look like this. On another scene the same measurement
reads -3.9 mm and -2.7 mm for two objects that both held.

**The tolerance is the object's own half-extent, not a constant.** What matters
is not how far the hand missed in millimetres but whether the object is still
between the pads, and that is a question about the object's size. The smallest
half-extent of its box is the tightest dimension it offers a jaw, so an error
larger than that means the object cannot be between the pads however the hand
is turned. On the four measurements above this separates them cleanly -- 13-16
mm against a 7.4 mm half-width (missed), 3.9 against 9.2 (reached), 2.7 against
7.0 (reached) -- without a magic number that needs recalibrating per object.

**It routes rather than raises**, because the caller has somewhere better to
go: a graph that learns the hand is not where it was sent can re-home and pick
again, spending seconds instead of a full carry-and-release of nothing. On a
scene with many goals and one unreachable object, that is the difference
between nothing and whatever the rest are worth.

**Falling back open.** With no box, or an unreadable pose, this reports
``reached`` rather than inventing a tolerance: a missing input is not evidence
of a missed grasp.
"""

from __future__ import annotations

import logging
import math
from typing import Any, TypedDict

from gap import NodeContext
from gap_core.types import OrientedBoundingBox, Se3Pose

logger = logging.getLogger(__name__)


class Output(TypedDict):
    route: str
    reached: bool
    error_m: float
    tolerance_m: float
    reason: str


def _position(pose: Any) -> dict[str, float] | None:
    """The position out of a pose, whichever shape it arrives in."""
    if isinstance(pose, dict):
        if "position" in pose:
            return _position(pose["position"])
        if {"x", "y", "z"} <= set(pose):
            return {k: float(pose[k]) for k in "xyz"}
    if isinstance(pose, (list, tuple)) and len(pose) >= 3:
        return {k: float(pose[i]) for i, k in enumerate("xyz")}
    return None


def _tolerance(
    target_obb: OrientedBoundingBox | None, floor_m: float, ceiling_m: float
) -> float:
    """How far the tool may be off and still have the object between the pads."""
    if not target_obb:
        return -1.0
    extent = target_obb.get("extent") if isinstance(target_obb, dict) else None
    if not extent:
        return -1.0
    smallest = min(float(extent["x"]), float(extent["y"]), float(extent["z"]))
    return max(floor_m, min(ceiling_m, smallest))


def run(
    ctx: NodeContext,
    commanded: Se3Pose | None = None,
    observed: Any = None,
    target_obb: OrientedBoundingBox | None = None,
    object_name: str = "",
    floor_m: float = 0.003,
    ceiling_m: float = 0.020,
) -> Output:
    """Route ``missed`` when the hand is too far from where it was sent to hold anything.

    ``floor_m`` is the smallest tolerance this will ever demand: a very thin
    object -- a washer, a blade -- has a half-extent of a millimetre or two, and
    demanding the servo land inside that would fail every grasp of it. Three
    millimetres is under the smallest error seen on a grasp that worked
    (2.7 mm), so it does not manufacture failures.

    ``ceiling_m`` is the largest it will ever allow: a big object forgives a big
    error, but not an unbounded one. Past two centimetres the hand is somewhere
    else entirely and the pose is not the one that was planned, whatever the
    object's size.
    """
    want = _position(commanded)
    got = _position(observed)
    tolerance = _tolerance(target_obb, floor_m, ceiling_m)
    if want is None or got is None or tolerance < 0.0:
        reason = "no commanded pose, no observed pose, or no target box"
        logger.warning("[reach] no opinion for %s (%s)", object_name or "?", reason)
        return {
            "route": "reached",
            "reached": True,
            "error_m": -1.0,
            "tolerance_m": -1.0,
            "reason": reason,
        }

    error = math.dist([want[k] for k in "xyz"], [got[k] for k in "xyz"])
    if error > tolerance:
        # Both numbers, because the ratio is what says whether the tolerance is
        # wrong or the motion is.
        reason = (
            f"the hand stopped {error * 1000:.1f} mm from the grasp pose for "
            f"{object_name or 'the target'}, past the {tolerance * 1000:.1f} mm this "
            f"object allows; closing here would grip nothing"
        )
        logger.info(
            "[reach] missed err_mm=%.1f tol_mm=%.1f object=%s",
            error * 1000.0, tolerance * 1000.0, object_name or "?",
        )
        return {
            "route": "missed",
            "reached": False,
            "error_m": error,
            "tolerance_m": tolerance,
            "reason": reason,
        }
    logger.info(
        "[reach] reached err_mm=%.1f tol_mm=%.1f object=%s",
        error * 1000.0, tolerance * 1000.0, object_name or "?",
    )
    return {
        "route": "reached",
        "reached": True,
        "error_m": error,
        "tolerance_m": tolerance,
        "reason": "",
    }
