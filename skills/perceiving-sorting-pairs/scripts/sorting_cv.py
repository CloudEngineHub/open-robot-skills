"""What both sorting scripts need to read one calibrated RGB-D view.

Three helpers, and they are here because they were in both scripts, character
for character. A helper that exists twice drifts: the two copies of a detector's
box accessor are one bug fix apart from disagreeing about what a detection is,
and nothing would notice until a graph that discovers a layout and a graph that
picks from it read the same frame two different ways.

Not a canonical script -- no ``run``, nothing a workflow names. The runtime
links a bundle's unnamed siblings in beside whatever scripts a graph takes from
that bundle, so importing this by name works wherever those scripts run.
"""

from __future__ import annotations

from typing import Any


def camera(observation: dict[str, Any], name: str) -> dict[str, Any]:
    """The named frame, whether the observation lists or maps its cameras."""
    cameras = observation.get("cameras") or []
    if isinstance(cameras, dict):
        cameras = list(cameras.values())
    return next(frame for frame in cameras if frame.get("name") == name)


def box(detection: dict[str, Any]) -> dict[str, Any]:
    """A detection's box, under whichever key the detector used for it."""
    return detection.get("box") or detection.get("bbox") or detection


def bounds(source: dict[str, Any]) -> tuple[float, float, float, float]:
    """``(x1, y1, x2, y2)`` as floats."""
    return tuple(float(source[key]) for key in ("x1", "y1", "x2", "y2"))
