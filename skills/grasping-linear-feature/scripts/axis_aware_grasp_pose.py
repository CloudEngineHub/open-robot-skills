"""Convert a fitted full-3D linear feature into an inclined grasp pose."""

from typing import TypedDict

import numpy as np
from gap import NodeContext
from gap_core.types import Se3Pose, Vec3
from scipy.spatial.transform import Rotation


class Output(TypedDict):
    grasp_pose: Se3Pose
    pregrasp_pose: Se3Pose
    approach_axis: Vec3


def _array(value: Vec3) -> np.ndarray:
    return np.array([value["x"], value["y"], value["z"]], dtype=float)


def _vec(value: np.ndarray) -> Vec3:
    return {"x": float(value[0]), "y": float(value[1]), "z": float(value[2])}


def run(
    ctx: NodeContext,
    feature_center: Vec3,
    feature_axis: Vec3,
    standoff: float = 0.12,
    surface_inset: float = 0.012,
) -> Output:
    """Approach perpendicular to the feature, as close to world-down as possible."""
    center = _array(feature_center)
    axis = _array(feature_axis)
    axis_norm = float(np.linalg.norm(axis))
    if axis_norm < 1e-9:
        raise ValueError("feature_axis must be nonzero")
    axis /= axis_norm

    down = np.array([0.0, 0.0, -1.0])
    approach = down - float(down @ axis) * axis
    if float(np.linalg.norm(approach)) < 0.20:
        preferred = np.array([-1.0, 0.0, 0.0])
        approach = preferred - float(preferred @ axis) * axis
    approach_norm = float(np.linalg.norm(approach))
    if approach_norm < 1e-9:
        raise ValueError("could not construct an approach perpendicular to feature_axis")
    approach /= approach_norm
    if approach[2] > 0.0:
        approach = -approach

    # Local Y follows the elongated feature; local Z is the approach. The
    # resulting local X closes across the feature on a parallel-jaw gripper.
    tool_y = axis
    tool_x = np.cross(tool_y, approach)
    tool_x /= max(float(np.linalg.norm(tool_x)), 1e-9)
    tool_y = np.cross(approach, tool_x)
    tool_y /= max(float(np.linalg.norm(tool_y)), 1e-9)
    rotation = np.column_stack((tool_x, tool_y, approach))
    q = Rotation.from_matrix(rotation).as_quat()  # xyzw
    quaternion = {"w": float(q[3]), "x": float(q[0]), "y": float(q[1]), "z": float(q[2])}

    grasp_position = center + approach * float(surface_inset)
    pregrasp_position = grasp_position - approach * float(standoff)
    return {
        "grasp_pose": {"position": _vec(grasp_position), "rotation": quaternion},
        "pregrasp_pose": {"position": _vec(pregrasp_position), "rotation": quaternion},
        "approach_axis": _vec(approach),
    }
