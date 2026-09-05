# SPDX-License-Identifier: Apache-2.0
"""curve tool bundle -- ordered centrelines of deformable linear objects.

Two ``@tool`` functions over ``_impl.py``: ``curve.fit_centerline`` fits an
ordered, arc-length-parameterised 3D centreline to a cable, rope or hose from
its mask and a depth frame, and ``curve.track_centerline`` carries a known
centreline onto a new frame through occlusion. The maths lives in ``_impl.py``;
this module is the typed boundary: numpy arrays + :mod:`gap_core.types`
TypedDicts in, a :class:`gap_core.types.Centerline` out.

No model, no GPU -- CPU numpy/scipy/scikit-image/cv2. ``_impl`` is imported
inside the functions that need it (the way the geometry bundle does), so
importing this module is always cheap.
"""

from __future__ import annotations

import numpy as np
from gap_core.tools import tool
from gap_core.types import Centerline, Mask, Se3Pose, pose_to_matrix

#: Defaults mirrored from ``_impl.DEFAULT_NODES`` / ``_impl.SMOOTH`` -- spelled
#: out here because ``_impl`` is imported lazily; ``tests/test_curve.py``
#: asserts the two agree.
_DEFAULT_NODES = 40
_SMOOTH = 2.0


def _camera_matrix(camera_pose: Se3Pose | np.ndarray) -> np.ndarray:
    """A camera-to-world 4x4 from either an ``Se3Pose`` dict or a 4x4 array."""
    if isinstance(camera_pose, dict):
        return pose_to_matrix(camera_pose)
    return np.asarray(camera_pose, dtype=np.float64).reshape(4, 4)


@tool(
    name="curve.fit_centerline",
    summary=(
        "Fit an ordered, arc-length-parameterised 3D centreline to a deformable linear "
        "object from its mask and a depth frame."
    ),
    tags=("perception",),
)
def fit_centerline(
    mask: Mask,
    depth: np.ndarray,
    intrinsics: np.ndarray,
    camera_pose: Se3Pose | np.ndarray,
    nodes: int = _DEFAULT_NODES,
    smooth: float = _SMOOTH,
) -> Centerline:
    """Skeletonise ``mask`` (any nonzero = object, [H, W]), recover one traversal
    order through breaks and crossings, back-project the ordered chain through
    ``depth`` (float [H, W], metres) and the pinhole ``intrinsics`` (3x3),
    smooth it with a spline and resample to ``nodes`` points uniformly in arc
    length, then push the curve from the object's camera-facing surface onto
    its axis. ``camera_pose`` is camera-to-world, as an ``Se3Pose`` or a 4x4.

    Returns a :class:`~gap_core.types.Centerline`: ``points`` as ``[[x, y, z],
    ...]`` in world metres in order along the object, ``arclength_m``,
    ``radius_m`` (mask area over length), ``nodes`` and ``ordered``. An empty
    or too-thin mask returns ``points == []`` and ``ordered == False``.
    """
    if mask is None or depth is None or intrinsics is None or camera_pose is None:
        raise ValueError("curve.fit_centerline needs mask, depth, intrinsics and camera_pose")
    from gap_skills.tools.curve import _impl

    return _impl.fit_centerline(
        np.asarray(mask),
        np.asarray(depth, dtype=np.float64),
        np.asarray(intrinsics, dtype=np.float64),
        _camera_matrix(camera_pose),
        nodes=int(nodes),
        smooth=float(smooth),
    )


@tool(
    name="curve.track_centerline",
    summary=(
        "Advance a known centreline onto a new frame, imputing motion for occluded "
        "stretches from their visible neighbours."
    ),
    tags=("perception",),
)
def track_centerline(
    prior: list[list[float]],
    mask: Mask,
    depth: np.ndarray,
    intrinsics: np.ndarray,
    camera_pose: Se3Pose | np.ndarray,
) -> Centerline:
    """Move the nodes of ``prior`` (the ``points`` of an earlier fit or track:
    ``[[x, y, z], ...]`` in world metres; a whole ``Centerline`` dict is
    unwrapped) onto the object as ``mask`` + ``depth`` see it now. Each node
    takes a Gaussian-weighted correspondence to the back-projected cloud, the
    displacement field is smoothed along the rod so hidden stretches move with
    their visible neighbours, a bending prior keeps them from folding, and the
    nodes are re-spaced to the length they had.

    Returns a :class:`~gap_core.types.Centerline` with ``visibility`` (per
    node, 0..1 -- route on it) and ``tracked: True``. A prior of fewer than
    three points falls back to a cold ``fit_centerline`` (no ``visibility``).
    """
    if prior is None or mask is None or depth is None or intrinsics is None or camera_pose is None:
        raise ValueError(
            "curve.track_centerline needs prior, mask, depth, intrinsics and camera_pose"
        )
    from gap_skills.tools.curve import _impl

    if isinstance(prior, dict):
        prior = prior.get("points") or []
    return _impl.track_centerline(
        prior,
        np.asarray(mask),
        np.asarray(depth, dtype=np.float64),
        np.asarray(intrinsics, dtype=np.float64),
        _camera_matrix(camera_pose),
    )
