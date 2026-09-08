"""What the jaw gap says about whether anything is between the pads.

The cheapest of the three checks in this bundle and the only one that needs no
camera: read the hand's own opening and compare it with the mechanical stop. A
close that ran all the way to the stop closed on nothing.

**The two conventions are inverted, and ``describe_gripper`` says so.** Its
``command`` block reports ``0 = open, 1 = fully closed`` for what you *write*,
while ``robot.get_gripper`` reports ``1.0 = OPEN`` for what you *read*. This
takes the read convention. Getting it backwards makes a hand closed on air look
like a hand holding something, which is the exact confusion this check ends.

**``expected_width_m`` is context, not a test.** The box carries half-extents,
so twice the smallest of the three is the thinnest way through the object --
what a jaw would close on if it found the thin part. A grasp planner that aims
at a fitted line's *centre* does not aim there: on a hammer-shaped object the
centre is near the head rather than the handle. A held width well above the
expectation is therefore not an error, it is the broad part of the body. That
gap is the point of the column: one measured episode held a multimeter at
65.7 mm against an expected 18.4, twice, at 85% of the hand's usable span,
while a box cutter came back at 12.4 against 11.6 -- a clean grip through the
thin axis. The pathology is object-specific, and seeing it is what the number
is for.

**An unreadable hand routes ``unknown``, not ``empty``.** A gap in the
instrument is not a failed grasp and must not be reported as one.
"""

from __future__ import annotations

import logging
from typing import Any, TypedDict

from gap import NodeContext
from gap_core.types import OrientedBoundingBox

logger = logging.getLogger(__name__)


class Output(TypedDict):
    route: str
    status: str
    grip_width_m: float
    grip_fraction: float
    expected_width_m: float
    object_name: str


def _mm(metres: float) -> str:
    """Millimetres for the log row, or ``n/a``.

    The -1.0 sentinels mean "not measured" and must not reach the row as
    ``-1000.0``: these lines exist to be read back into a table, and a
    plausible-looking negative millimetre is the kind of value that survives a
    ``float()`` and quietly becomes a data point.
    """
    return "n/a" if metres < 0.0 else f"{metres * 1000.0:.1f}"


def _width_from_fraction(gripper: dict[str, Any], fraction: float) -> float:
    """The jaw gap in metres for an observed open fraction.

    Linear interpolation between the measured open and closed widths. When
    ``width_fit.source`` is ``"span"`` rather than ``"measured"`` those numbers
    are the hand's declared span and a zero, so the result is a proportion of
    span -- still monotonic in the gap, which is all the verdict uses.
    """
    fit = gripper.get("width_fit") or {}
    at_open = float(fit.get("at_open_m") or gripper.get("span_m") or 0.0)
    at_closed = float(fit.get("at_closed_m") or 0.0)
    return at_closed + max(0.0, min(1.0, fraction)) * (at_open - at_closed)


def _expected_width(target_obb: OrientedBoundingBox | None) -> float:
    """The object's smallest dimension [m], or -1.0."""
    if not target_obb:
        return -1.0
    extent = target_obb.get("extent") if isinstance(target_obb, dict) else None
    if not extent:
        return -1.0
    return 2.0 * min(float(extent["x"]), float(extent["y"]), float(extent["z"]))


def run(
    ctx: NodeContext,
    target_obb: OrientedBoundingBox | None = None,
    object_name: str = "",
    arm_id: int = 0,
    empty_margin_m: float = 0.003,
) -> Output:
    """Route ``held``, ``empty`` or ``unknown`` from the jaw gap alone.

    ``empty_margin_m`` is how far above the mechanical stop still counts as
    "closed on nothing". Three millimetres, and it is a judgement: it absorbs
    the settle the jaws do not quite finish, not anything about an object. The
    one empty close measured came in at 1.8 mm, comfortably inside it.
    """
    status = "unknown"
    fraction = width = -1.0
    expected = _expected_width(target_obb)

    try:
        fraction = float(ctx.tool("robot.get_gripper", arm_id=arm_id)["position"])
        gripper = ctx.tool("robot.describe_gripper", arm_id=arm_id)
        width = _width_from_fraction(gripper, fraction)
        floor = float((gripper.get("width_fit") or {}).get("at_closed_m") or 0.0)
        status = "empty" if width <= floor + empty_margin_m else "held"
    except Exception as exc:  # noqa: BLE001
        logger.warning("[grip] could not read the gripper (%s)", exc)

    # One row per grasp, shaped to be read back as a table.
    logger.info(
        "[grip] status=%s width_mm=%s frac=%s expected_mm=%s object=%s",
        status,
        _mm(width),
        "n/a" if fraction < 0.0 else f"{fraction:.3f}",
        _mm(expected),
        object_name or "?",
    )
    return {
        "route": status,
        "status": status,
        "grip_width_m": width,
        "grip_fraction": fraction,
        "expected_width_m": expected,
        "object_name": object_name,
    }
