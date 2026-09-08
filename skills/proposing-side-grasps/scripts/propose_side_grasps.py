"""Propose horizontal (side-entry) grasps swept around an object's box.

The rung to reach for when top-down is not available: an overhead shelf, a lid,
a bin wall, or an object whose vertical dimension is the only one the jaws can
span. A top-down proposer declines those by measuring a width; it cannot offer
an alternative, because approaching from the side is a different strategy
rather than a different number.

Twelve azimuths, ranked by **jaw margin** — how much room is left over after
the object's width in that closing direction. Ranked by margin and by nothing
else, deliberately: which side is actually *open* is a fact about the scene,
not about the object, and it belongs to whoever selects from this list with a
clearance check. A proposer that quietly preferred the side it guessed was
reachable would be answering the selector's question badly instead of
answering its own well.

The width in a direction is the box's support function -- the sum over its
three axes of the half-extent projected onto that direction -- so an oriented
box lying at any angle is measured correctly rather than by its world AABB.
"""

from typing import Any, TypedDict

import math

from gap import NodeContext
from gap_core.types import OrientedBoundingBox

_AZIMUTHS = 12
"""Approach directions swept around the vertical axis, evenly spaced (30 deg).

Twelve rather than four because a box at 45 degrees to the world has no good
axis-aligned side, and rather than thirty-six because the jaw margin varies
smoothly and a selector walking the list pays IK for every entry it tries."""


class Output(TypedDict):
    route: str
    candidates: list[dict[str, Any]]
    reason: str


def _axes(quat: dict[str, float]) -> list[tuple[float, float, float]]:
    w, x, y, z = quat["w"], quat["x"], quat["y"], quat["z"]
    return [
        (1 - 2 * (y * y + z * z), 2 * (x * y + z * w), 2 * (x * z - y * w)),
        (2 * (x * y - z * w), 1 - 2 * (x * x + z * z), 2 * (y * z + x * w)),
        (2 * (x * z + y * w), 2 * (y * z - x * w), 1 - 2 * (x * x + y * y)),
    ]


def _width_along(axes, half, direction) -> float:
    """The box's full width along *direction* — its support function, doubled."""
    return 2.0 * sum(
        h * abs(sum(a * d for a, d in zip(axis, direction)))
        for axis, h in zip(axes, half)
    )


def run(
    ctx: NodeContext,
    target_obb: OrientedBoundingBox,
    gripper: dict[str, Any] | None = None,
) -> Output:
    """Side-entry candidates around *target_obb*, widest margin first.

    Returns:
        ``candidates`` entries carry ``approach`` (a horizontal unit vector, to
        hand to ``robot.grasp_frame(approach=…)``), ``close_heading_deg`` (to
        hand to the same call), ``position`` (the box's centre height), and
        ``width_m``/``margin_m`` so a caller can see why the order is the order.
        ``route`` is ``"declined"`` when the hand cannot span the object from
        any azimuth.
    """
    centre = target_obb["center"]
    extent = target_obb["extent"]
    axes = _axes(target_obb["orientation"])
    half = [extent["x"], extent["y"], extent["z"]]
    span = float((gripper or {}).get("max_grasp_width_m") or 0.0)

    candidates: list[dict[str, Any]] = []
    for index in range(_AZIMUTHS):
        theta = 2.0 * math.pi * index / _AZIMUTHS
        approach = (math.cos(theta), math.sin(theta), 0.0)
        # The jaws close across the approach, in the horizontal plane.
        closing = (-math.sin(theta), math.cos(theta), 0.0)
        width = _width_along(axes, half, closing)
        if span and width > span:
            continue
        candidates.append({
            "approach": {"x": approach[0], "y": approach[1], "z": 0.0},
            "close_heading_deg": math.degrees(math.atan2(closing[1], closing[0])),
            "position": {"x": centre["x"], "y": centre["y"], "z": centre["z"]},
            "width_m": width,
            "margin_m": (span - width) if span else None,
        })

    if not candidates:
        return {
            "route": "declined",
            "candidates": [],
            "reason": (
                f"this hand opens to {1e3 * span:.0f} mm and the object is wider than that "
                f"from every azimuth — no side grasp closes on it. A part of the object "
                f"narrower than its whole box is the remaining option."
            ),
        }
    candidates.sort(key=lambda c: (c["margin_m"] is None, -(c["margin_m"] or 0.0)))
    return {"route": "proposed", "candidates": candidates, "reason": ""}
