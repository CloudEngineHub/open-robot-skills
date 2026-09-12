"""Convert a fitted full-3D linear feature into an inclined grasp pose."""

from typing import Any, TypedDict

import numpy as np
from gap import NodeContext
from gap_core.types import Se3Pose, Vec3
from scipy.spatial.transform import Rotation


class Output(TypedDict):
    grasp_pose: Se3Pose
    pregrasp_pose: Se3Pose
    approach_axis: Vec3
    grasp_axis_overlay: np.ndarray


def _array(value: Vec3) -> np.ndarray:
    return np.array([value["x"], value["y"], value["z"]], dtype=float)


def _vec(value: np.ndarray) -> Vec3:
    return {"x": float(value[0]), "y": float(value[1]), "z": float(value[2])}


def _render_overlay(
    points: np.ndarray,
    center: np.ndarray,
    axis: np.ndarray,
    approach: np.ndarray,
    grasp_position: np.ndarray,
    standoff: float,
) -> np.ndarray:
    """Top-down view of the point cloud with feature axis and jaw direction overlaid."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(5, 5), dpi=96)
        if len(points) > 0:
            ax.scatter(points[:, 1], points[:, 0], s=2, c="steelblue", alpha=0.4, linewidths=0)

        half = max(0.05, float(np.linalg.norm((points.max(axis=0) - points.min(axis=0))[:2])) * 0.6) if len(points) else 0.1
        ax.annotate(
            "",
            xy=(center[1] + axis[1] * half, center[0] + axis[0] * half),
            xytext=(center[1] - axis[1] * half, center[0] - axis[0] * half),
            arrowprops=dict(arrowstyle="<->", color="orange", lw=2),
        )

        jaw_dir = np.cross(axis, approach)
        jaw_dir /= max(float(np.linalg.norm(jaw_dir)), 1e-9)
        jaw_half = half * 0.4
        ax.plot(
            [center[1] - jaw_dir[1] * jaw_half, center[1] + jaw_dir[1] * jaw_half],
            [center[0] - jaw_dir[0] * jaw_half, center[0] + jaw_dir[0] * jaw_half],
            color="red", lw=2.5, label="jaw close",
        )

        ax.plot(center[1], center[0], "o", color="orange", ms=6, zorder=5)

        pregrasp = grasp_position - approach * standoff
        ax.annotate(
            "",
            xy=(grasp_position[1], grasp_position[0]),
            xytext=(pregrasp[1], pregrasp[0]),
            arrowprops=dict(arrowstyle="->", color="green", lw=1.5),
        )

        ax.set_aspect("equal")
        ax.set_xlabel("Y (m)")
        ax.set_ylabel("X (m)")
        ax.set_title("axis grasp — top-down\norange=feature axis  red=jaw close  green=approach")
        ax.legend(fontsize=7, loc="upper right")
        fig.tight_layout()

        fig.canvas.draw()
        w, h = fig.canvas.get_width_height()
        buf = fig.canvas.buffer_rgba()
        overlay = np.frombuffer(buf, dtype=np.uint8).reshape(h, w, 4)[:, :, :3].copy()
        plt.close(fig)
        return overlay
    except Exception:
        import logging, traceback
        logging.getLogger(__name__).warning("grasp_axis_overlay render failed:\n%s", traceback.format_exc())
        return np.zeros((480, 480, 3), dtype=np.uint8)


def run(
    ctx: NodeContext,
    feature_center: Vec3,
    feature_axis: Vec3,
    standoff: float = 0.12,
    grasp_z_offset: float = 0.015,
    bin_floor_z: float | None = None,
    target_cloud: Any = None,
) -> Output:
    """Approach perpendicular to the feature, as close to world-down as possible."""
    center = _array(feature_center)
    axis = _array(feature_axis)
    axis_norm = float(np.linalg.norm(axis))
    if axis_norm < 1e-9:
        raise ValueError("feature_axis must be nonzero")
    axis /= axis_norm

    # Always approach straight down — a partial surface view from above gives
    # unreliable Z information, so we only use the XY projection of the feature
    # axis for in-plane gripper orientation.
    approach = np.array([0.0, 0.0, -1.0])

    axis_xy = np.array([axis[0], axis[1], 0.0])
    xy_norm = float(np.linalg.norm(axis_xy))
    if xy_norm < 1e-9:
        axis_xy = np.array([1.0, 0.0, 0.0])
    else:
        axis_xy /= xy_norm

    # Local X closes the jaws perpendicular to the feature; local Y follows the
    # elongated feature; local Z is straight down.
    # cross(approach, axis_xy) matches the sign convention used by _rotation_top_down
    # in contour_concave_grasp and produces IK-reachable poses for this robot.
    tool_x = np.cross(approach, axis_xy)
    tool_x /= max(float(np.linalg.norm(tool_x)), 1e-9)
    tool_y = np.cross(approach, tool_x)
    tool_y /= max(float(np.linalg.norm(tool_y)), 1e-9)
    rotation = np.column_stack((tool_x, tool_y, approach))
    q = Rotation.from_matrix(rotation).as_quat()  # xyzw
    quaternion = {"w": float(q[3]), "x": float(q[0]), "y": float(q[1]), "z": float(q[2])}

    grasp_z = (float(bin_floor_z) + float(grasp_z_offset)) if bin_floor_z is not None \
        else float(center[2])
    grasp_position = np.array([center[0], center[1], grasp_z])
    pregrasp_position = grasp_position - approach * float(standoff)

    pts = np.empty((0, 3), dtype=np.float32)
    if target_cloud is not None:
        try:
            raw = target_cloud.get("points", target_cloud) if isinstance(target_cloud, dict) else target_cloud
            pts = np.asarray(raw, dtype=np.float32).reshape(-1, 3)
            pts = pts[np.isfinite(pts).all(axis=1)]
        except Exception:
            pass

    overlay = _render_overlay(pts, center, axis, approach, grasp_position, float(standoff))

    return {
        "grasp_pose": {"position": _vec(grasp_position), "rotation": quaternion},
        "pregrasp_pose": {"position": _vec(pregrasp_position), "rotation": quaternion},
        "approach_axis": _vec(approach),
        "grasp_axis_overlay": overlay,
    }
