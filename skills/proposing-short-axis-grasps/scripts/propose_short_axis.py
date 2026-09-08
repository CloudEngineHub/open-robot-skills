"""Propose top-down grasps whose jaws close ACROSS an object's short axis.

The one geometric fact this encodes: an elongated body is grasped across its
narrow dimension, not along its length. That sounds obvious and is the single
most expensive mistake in this stack's history — ``sim.get_object_obb`` returns
``grasp_yaw``, the heading of the object's *long* axis, and every consumer fed
it to ``robot.grasp_frame(close_heading_deg=…)``, which is the direction the
jaws close *along*. Ninety degrees out, silent, and it validates and executes:
the fingers close down the length of the handle and meet each other. Measured
on the hammer task, holding the wrist at ``grasp_yaw`` scores 0/12 and holding
it square to the long axis scores 8/12.

So the heading is derived here, from the box, rather than read off a field that
means something else. It is derived from an *oriented box*, not from a
privileged pose, which is what makes this usable at every privilege tier: the
box can come from ``sim.get_object_obb`` or from ``geometry.compute_obb`` over
a segmented point cloud, and this script cannot tell which.

What it does NOT do is choose *where along* the long axis to close. It offers a
fan of stations spaced along it, best-first from the centre, and the caller
picks — because which station holds is a question about this object's mass
distribution and this hand's span, and answering it here would be inventing a
measurement rather than proposing a strategy.
"""

from typing import Any, TypedDict

import math

from gap import NodeContext
from gap_core.types import OrientedBoundingBox

_STATION_FRACTIONS = (0.0, -0.25, 0.25, -0.45, 0.45)
"""Where along the long axis to offer stations, as a fraction of its half-extent.

Centre first: it is the only station that is correct for a symmetric body, and
it is the best guess for an asymmetric one absent any other information. The
rest walk outward in pairs so a caller that simply takes the next candidate
after a failure moves somewhere meaningfully different rather than by a
millimetre.
"""


class Output(TypedDict):
    route: str
    close_heading_deg: float
    grip_width_m: float
    positions: list[dict[str, float]]
    reason: str


def _axes(quat: dict[str, float]) -> list[tuple[float, float, float]]:
    """The box's three local axes as world unit vectors (the rotation's columns)."""
    w, x, y, z = quat["w"], quat["x"], quat["y"], quat["z"]
    return [
        (1 - 2 * (y * y + z * z), 2 * (x * y + z * w), 2 * (x * z - y * w)),
        (2 * (x * y - z * w), 1 - 2 * (x * x + z * z), 2 * (y * z + x * w)),
        (2 * (x * z + y * w), 2 * (y * z - x * w), 1 - 2 * (x * x + y * y)),
    ]


def run(
    ctx: NodeContext,
    target_obb: OrientedBoundingBox,
    gripper: dict[str, Any] | None = None,
    min_z: float | None = None,
) -> Output:
    """Stations and a closing heading for a top-down grasp across the short axis.

    Args:
        target_obb: The object's oriented box. ``extent`` is half-extents.
        gripper: ``robot.describe_gripper``'s reading, for the jaw span. Omitted,
            the width is reported and nothing is refused for being too wide —
            a caller without the hand's geometry should not have a refusal
            invented for it.
        min_z: A floor for the grasp height, if the caller has one (a work
            surface plus finger clearance). The stations are otherwise placed
            at the box's own mid-height.

    Returns:
        ``route`` is ``"proposed"`` or ``"declined"``; on ``"proposed"``,
        ``close_heading_deg`` goes straight to
        ``robot.grasp_frame(close_heading_deg=…)`` and ``positions`` are the
        stations, best first.
    """
    extent = target_obb["extent"]
    centre = target_obb["center"]
    axes = _axes(target_obb["orientation"])
    half = [extent["x"], extent["y"], extent["z"]]

    # The two most horizontal axes are the ones a top-down grasp chooses
    # between; the third is the one the hand descends along.
    horizontality = [1.0 - abs(axis[2]) for axis in axes]
    order = sorted(range(3), key=lambda i: horizontality[i], reverse=True)
    long_i, short_i = sorted(order[:2], key=lambda i: half[i], reverse=True)

    long_axis, short_axis = axes[long_i], axes[short_i]
    grip_width = 2.0 * half[short_i]

    if gripper:
        widest = gripper.get("max_grasp_width_m")
        if widest is not None and grip_width > float(widest):
            return {
                "route": "declined",
                "close_heading_deg": 0.0,
                "grip_width_m": grip_width,
                "positions": [],
                "reason": (
                    f"the short axis measures {1e3 * grip_width:.0f} mm and this hand opens to "
                    f"{1e3 * float(widest):.0f} mm — no top-down grasp across it can close. "
                    f"Try a side grasp, or a part of the object narrower than its box."
                ),
            }

    # The jaws close ALONG the short axis, so that is the heading, not the long
    # axis's. This is the ninety degrees the module docstring is about.
    heading = math.degrees(math.atan2(short_axis[1], short_axis[0]))
    height = centre["z"] if min_z is None else max(centre["z"], float(min_z))
    positions = [
        {
            "x": centre["x"] + fraction * half[long_i] * long_axis[0],
            "y": centre["y"] + fraction * half[long_i] * long_axis[1],
            "z": height,
        }
        for fraction in _STATION_FRACTIONS
    ]
    return {
        "route": "proposed",
        "close_heading_deg": heading,
        "grip_width_m": grip_width,
        "positions": positions,
        "reason": "",
    }
