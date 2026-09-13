"""Top-down grasp from pairs of concave sub-contours on the object's silhouette.

Algorithm
---------
1. Fetch the full surface point cloud and project all points to 2-D (x, y).
2. Compute the **alpha shape** of the projected points: a Delaunay triangulation
   filtered to triangles whose circumradius < *alpha_radius*.  The boundary of
   the surviving triangles is the actual silhouette of the object, including
   genuine concavities that the convex hull would erase.
3. Walk the ordered boundary polygon and find **concave (reflex) vertices** --
   vertices where the boundary turns inward (clockwise for a CCW polygon).
   Group consecutive reflex vertices into sub-contours and take each group's
   centroid as a grip candidate.
4. For each pair of grip candidates whose secant distance fits the jaw span,
   build a straight-down grasp centred at the midpoint of the pair with the
   jaw closing along the secant.
5. Rank by confidence and return up to *max_candidates* poses.

Why alpha shape?
----------------
A convex hull has no concavities by construction, so measuring deviation from
it is an indirect proxy.  The alpha shape directly represents the object's
actual boundary and makes reflex vertices explicit.
"""

from __future__ import annotations

import logging
from typing import TypedDict

import numpy as np

from gap import NodeContext
from gap_core.types import OrientedBoundingBox, Se3Pose

logger = logging.getLogger(__name__)

__all__ = ["Output", "run"]

_STANDOFF_M = 0.12


class Output(TypedDict):
    grasp_pose: Se3Pose
    pregrasp_pose: Se3Pose
    candidates: list[dict]
    grip_width: float
    confidence: float
    grasp_index: int
    num_object_points: int
    grip_height: float
    contour_boundary_overlay: np.ndarray  # uint8 [H, W, 3] — auto-saved by tracing


# ---------------------------------------------------------------------------
# Alpha shape helpers
# ---------------------------------------------------------------------------

def _alpha_boundary(xy: np.ndarray, alpha_radius: float) -> np.ndarray | None:
    """Ordered outer boundary of the alpha shape, or None if degenerate."""
    from scipy.spatial import Delaunay  # noqa: PLC0415

    if len(xy) < 4:
        return None

    tri = Delaunay(xy)
    boundary: set[tuple[int, int]] = set()

    for simplex in tri.simplices:
        pts = xy[simplex]
        a = float(np.linalg.norm(pts[1] - pts[0]))
        b = float(np.linalg.norm(pts[2] - pts[1]))
        c = float(np.linalg.norm(pts[0] - pts[2]))
        s = (a + b + c) / 2.0
        area_sq = s * (s - a) * (s - b) * (s - c)
        if area_sq <= 0.0:
            continue
        circumradius = a * b * c / (4.0 * np.sqrt(area_sq))
        if circumradius > alpha_radius:
            continue
        for i, j in ((0, 1), (1, 2), (2, 0)):
            edge = (min(simplex[i], simplex[j]), max(simplex[i], simplex[j]))
            if edge in boundary:
                boundary.discard(edge)
            else:
                boundary.add(edge)

    if len(boundary) < 3:
        return None

    # Build adjacency map.
    adj: dict[int, list[int]] = {}
    for a, b in boundary:
        adj.setdefault(a, []).append(b)
        adj.setdefault(b, []).append(a)

    # Find the largest connected component.
    visited: set[int] = set()
    best: list[int] = []
    for seed in adj:
        if seed in visited:
            continue
        comp: list[int] = []
        stack = [seed]
        while stack:
            v = stack.pop()
            if v in visited:
                continue
            visited.add(v)
            comp.append(v)
            stack.extend(adj[v])
        if len(comp) > len(best):
            best = comp

    if len(best) < 3:
        return None

    comp_set = set(best)

    # Walk the boundary of the component into an ordered polygon.
    start = best[0]
    ordered = [start]
    prev: int | None = None
    curr = start
    for _ in range(len(best) + 1):
        neighbors = [n for n in adj[curr] if n in comp_set and n != prev]
        if not neighbors:
            break
        nxt = neighbors[0]
        if nxt == start and len(ordered) > 2:
            break
        ordered.append(nxt)
        prev, curr = curr, nxt

    if len(ordered) < 3:
        return None

    return xy[ordered]


def _concave_grip_pts(polygon: np.ndarray,
                      min_arc_m: float,
                      min_concavity: float = 0.26) -> list[dict]:
    """Midpoints of concave (reflex) sub-contours on the boundary polygon."""
    n = len(polygon)

    # Ensure CCW orientation via signed area.
    area = sum(
        polygon[i, 0] * polygon[(i + 1) % n, 1]
        - polygon[(i + 1) % n, 0] * polygon[i, 1]
        for i in range(n)
    )
    if area < 0:
        polygon = polygon[::-1]

    # Flag each vertex as reflex (concave) or convex using a window of k steps.
    # Adjacent-vertex curvature fires on every tessellation zigzag; a wider
    # window only fires on features that span multiple triangle edges.
    k = max(1, n // 32)
    reflex = np.zeros(n, dtype=bool)
    for i in range(n):
        prev = polygon[(i - k) % n]
        nxt = polygon[(i + k) % n]
        d1 = polygon[i] - prev
        d2 = nxt - polygon[i]
        d1_len = float(np.linalg.norm(d1))
        d2_len = float(np.linalg.norm(d2))
        if d1_len * d2_len < 1e-10:
            continue
        sin_angle = (d1[0] * d2[1] - d1[1] * d2[0]) / (d1_len * d2_len)
        reflex[i] = sin_angle < -min_concavity

    # Group consecutive reflex vertices; handle wrap-around.
    # Duplicate the array index sequence to catch runs that span the seam.
    grip_pts: list[dict] = []
    seen_starts: set[int] = set()
    for start in range(n):
        if not reflex[start] or start in seen_starts:
            continue
        run = []
        i = start
        while reflex[i % n]:
            idx = i % n
            if idx in seen_starts and i != start:
                break
            seen_starts.add(idx)
            run.append(polygon[idx])
            i += 1
            if i - start >= n:
                break
        if not run:
            continue
        # Filter sub-contours shorter than min_arc_m.
        arc_len = sum(
            float(np.linalg.norm(run[k + 1] - run[k]))
            for k in range(len(run) - 1)
        )
        if len(run) > 1 and arc_len < min_arc_m:
            continue
        grip_pts.append({"point": np.mean(run, axis=0)})

    return grip_pts


# ---------------------------------------------------------------------------
# Debug visualization
# ---------------------------------------------------------------------------

def _visualize(
    xy: np.ndarray,
    polygon: np.ndarray,
    grip_pts: list[dict],
    candidates: list[dict],
    object_name: str,
) -> np.ndarray:
    """Render the alpha-shape contour, grip candidates, and best grasp to a uint8 RGB array.

    Also appends a JSON entry and saves a PNG to GRASP_DEBUG_LOG when that env
    var is set. Returns a 1x1 black pixel on any rendering failure so the caller
    can always include it in Output without a None check.
    """
    import json  # noqa: PLC0415
    import os    # noqa: PLC0415
    from pathlib import Path  # noqa: PLC0415

    _BLANK = np.zeros((1, 1, 3), dtype=np.uint8)

    log_path = os.environ.get("GRASP_DEBUG_LOG", "")

    # --- structured JSON entry (only when GRASP_DEBUG_LOG is set) ---
    if log_path:
        entry = {
            "event": "contour_concave",
            "object_name": object_name,
            "boundary": polygon.tolist(),
            "grip_candidates": [g["point"].tolist() for g in grip_pts],
            "best_grasp": candidates[0]["grasp_pose"] if candidates else None,
        }
        try:
            with open(log_path, "a") as fh:
                fh.write(json.dumps(entry) + "\n")
        except OSError:
            pass

    # --- render figure to numpy array ---
    try:
        import matplotlib  # noqa: PLC0415
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt  # noqa: PLC0415

        fig, ax = plt.subplots(figsize=(6, 6))

        # Raw point cloud (faint)
        ax.scatter(xy[:, 0], xy[:, 1], s=2, c="lightgray", zorder=1)

        # Alpha shape boundary
        closed = np.vstack([polygon, polygon[0]])
        ax.plot(closed[:, 0], closed[:, 1], "b-", lw=1.5, label="alpha boundary", zorder=2)

        # Grip candidate midpoints
        if grip_pts:
            pts = np.array([g["point"] for g in grip_pts])
            ax.scatter(pts[:, 0], pts[:, 1], c="orange", s=60, zorder=4,
                       label=f"grip candidates ({len(grip_pts)})")

        # Best grasp: draw jaw line and centre.
        if candidates:
            best = candidates[0]
            pos = best["grasp_pose"]["position"]
            cx, cy = pos["x"], pos["y"]
            p1 = best.get("_p1")
            p2 = best.get("_p2")
            jaw_dir = best.get("_jaw_dir")
            if p1 and p2:
                ax.plot([p1[0], p2[0]], [p1[1], p2[1]],
                        "g-", lw=2.5, label="best jaw", zorder=5)
                ax.scatter(*p1, c="green", s=60, zorder=6)
                ax.scatter(*p2, c="green", s=60, zorder=6)
            elif jaw_dir is not None:
                half = float(best.get("grip_width", 0.0)) * 0.5 or 0.03
                dx, dy = jaw_dir[0] * half, jaw_dir[1] * half
                ax.plot([cx - dx, cx + dx], [cy - dy, cy + dy],
                        "g-", lw=2.5, label="best jaw (axis)", zorder=5)
                ax.scatter([cx - dx, cx + dx], [cy - dy, cy + dy],
                           c="green", s=60, zorder=6)
            ax.scatter([cx], [cy], c="red", s=80, zorder=7, label="grasp centre")

        ax.set_aspect("equal")
        ax.set_title(f"contour_concave — {object_name}")
        ax.legend(fontsize=8)
        fig.tight_layout()

        # Render to numpy before (optionally) saving to disk.
        fig.canvas.draw()
        w, h = fig.canvas.get_width_height()
        buf = fig.canvas.buffer_rgba()
        img = np.frombuffer(buf, dtype=np.uint8).reshape(h, w, 4)[:, :, :3].copy()

        if log_path:
            try:
                log_dir = Path(log_path).parent
                fig.savefig(log_dir / f"contour_concave_{object_name}.png", dpi=100)
            except OSError:
                pass

        plt.close(fig)
        return img

    except Exception:  # noqa: BLE001
        return _BLANK  # visualization is best-effort; never abort a pick


# ---------------------------------------------------------------------------
# Rotation helper
# ---------------------------------------------------------------------------

def _rotation_top_down(secant: np.ndarray) -> dict:
    """Quaternion for a straight-down tool with jaw along *secant* (unit 2-D vector)."""
    from scipy.spatial.transform import Rotation  # noqa: PLC0415

    dx, dy = float(secant[0]), float(secant[1])
    x_tool = np.array([dx, dy, 0.0])
    z_tool = np.array([0.0, 0.0, -1.0])
    y_tool = np.cross(z_tool, x_tool)
    y_norm = float(np.linalg.norm(y_tool))
    if y_norm < 1e-9:
        return {"w": 1.0, "x": 0.0, "y": 0.0, "z": 0.0}
    y_tool /= y_norm
    R = np.column_stack([x_tool, y_tool, z_tool])
    q = Rotation.from_matrix(R).as_quat()
    return {"w": float(q[3]), "x": float(q[0]), "y": float(q[1]), "z": float(q[2])}


# ---------------------------------------------------------------------------
# Axis-based grasp fallback (mirrors axis_aware_grasp_pose.py logic)
# ---------------------------------------------------------------------------

def _axis_grasp_candidate(
    ctx: Any,
    target_cloud: dict,
    standoff: float = 0.12,
    surface_inset: float = 0.012,
) -> dict | None:
    """Fit a linear feature axis and return a single grasp candidate, or None on failure."""
    from scipy.spatial.transform import Rotation  # noqa: PLC0415

    try:
        fit = ctx.tool("geometry.fit_linear_feature", points=target_cloud)
    except Exception:
        return None

    center = np.array([fit["center"]["x"], fit["center"]["y"], fit["center"]["z"]])
    axis = np.array([fit["axis"]["x"], fit["axis"]["y"], fit["axis"]["z"]], dtype=float)

    # Project feature axis to XY plane to get the in-plane jaw direction.
    axis_2d = np.array([axis[0], axis[1], 0.0])
    axis_2d_norm = float(np.linalg.norm(axis_2d))
    if axis_2d_norm < 1e-6:
        return None
    axis_2d /= axis_2d_norm

    approach = np.array([0.0, 0.0, -1.0])
    # jaw_dir: closing direction in XY, perpendicular to the feature axis.
    # This is the same vector stored in _jaw_dir and drawn as the visualization line,
    # so the rotation and the figure are guaranteed to use the same direction.
    jaw_dir = np.cross(axis_2d, approach)   # = [-ay, ax, 0], unit vector
    y_tool = np.cross(approach, jaw_dir)    # right-hand completion
    R = np.column_stack([jaw_dir, y_tool, approach])
    q = Rotation.from_matrix(R).as_quat()
    rotation = {"w": float(q[3]), "x": float(q[0]), "y": float(q[1]), "z": float(q[2])}

    grasp_pos = center + approach * surface_inset
    pregrasp_pos = grasp_pos - approach * standoff
    grasp_pose: Se3Pose = {
        "position": {"x": float(grasp_pos[0]), "y": float(grasp_pos[1]), "z": float(grasp_pos[2])},
        "rotation": rotation,
    }
    pregrasp_pose: Se3Pose = {
        "position": {"x": float(pregrasp_pos[0]), "y": float(pregrasp_pos[1]), "z": float(pregrasp_pos[2])},
        "rotation": rotation,
    }
    return {
        "cand_index": 0,
        "grasp_pose": grasp_pose,
        "pregrasp_pose": pregrasp_pose,
        "grip_width": 0.0,
        "confidence": 0.0,
        "_p1": None,
        "_p2": None,
        "_jaw_dir": [float(jaw_dir[0]), float(jaw_dir[1])],
    }


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run(
    ctx: NodeContext,
    target_obb: OrientedBoundingBox,
    target_cloud: dict,
    object_name: str = "",
    bin_floor_z: float | None = None,
    alpha_radius: float = 0.015,
    min_arc_m: float = 0.005,
    min_concavity: float = 0.26,
    max_candidates: int = 24,
    grasp_z_offset: float = 0.014,
) -> Output:
    # ------------------------------------------------------------------ cloud
    name = object_name
    points = np.array(target_cloud["points"], dtype=np.float64)
    n_pts = len(points)
    if n_pts < 16:
        raise RuntimeError(
            f"contour_concave grasp: only {n_pts} points in target_cloud for '{name}'."
        )

    xy = points[:, :2]

    # -------------------------------------------------- alpha shape boundary
    polygon = _alpha_boundary(xy, alpha_radius)
    if polygon is None:
        raise RuntimeError(
            f"contour_concave grasp: alpha shape produced no usable boundary "
            f"for '{name}' at alpha_radius={alpha_radius:.3f} m. "
            "Try increasing alpha_radius."
        )

    # ------------------------------------------- concave sub-contour midpoints
    grip_pts = _concave_grip_pts(polygon, min_arc_m, min_concavity)
    # ----------------------------------------------- gripper span
    gripper = ctx.tool("robot.describe_gripper")
    span = float(gripper.get("max_grasp_width_m") or gripper["span_m"])
    centroid = polygon.mean(axis=0)

    candidates: list[dict] = []

    # Antipodal fallback disabled — axis grasp handles convex objects.
    # if len(grip_pts) < 2:
    #     logger.info(
    #         "select_pair: contour_concave falling back to antipodal pairs for '%s' "
    #         "(only %d concave sub-contour(s) found)",
    #         name, len(grip_pts),
    #     )
    #     n_dir = 36
    #     for k in range(n_dir):
    #         theta = k * np.pi / n_dir
    #         d = np.array([np.cos(theta), np.sin(theta)])
    #         proj = np.dot(polygon - centroid, d)
    #         p1 = polygon[int(np.argmax(proj))]
    #         p2 = polygon[int(np.argmin(proj))]
    #         dist = float(np.linalg.norm(p2 - p1))
    #         if dist > span:
    #             continue
    #         center = 0.5 * (p1 + p2)
    #         secant = (p2 - p1) / dist
    #         rotation = _rotation_top_down(secant)
    #         xy_dists_fb = np.linalg.norm(xy - center, axis=1)
    #         near_fb = xy_dists_fb < max(float(dist), 0.02)
    #         if not np.any(near_fb):
    #             near_fb = xy_dists_fb < (xy_dists_fb.min() * 3 + 0.005)
    #         z_grasp_fb = (bin_floor_z if bin_floor_z is not None
    #                       else float(np.median(points[near_fb, 2]))) + grasp_z_offset
    #         grasp_pose_fb: Se3Pose = {
    #             "position": {"x": float(center[0]), "y": float(center[1]),
    #                          "z": z_grasp_fb},
    #             "rotation": rotation,
    #         }
    #         pregrasp_pose_fb: Se3Pose = {
    #             "position": {"x": float(center[0]), "y": float(center[1]),
    #                          "z": z_grasp_fb + _STANDOFF_M},
    #             "rotation": rotation,
    #         }
    #         dist_from_centroid = float(np.linalg.norm(center - centroid))
    #         candidates.append({
    #             "cand_index": len(candidates),
    #             "grasp_pose": grasp_pose_fb,
    #             "pregrasp_pose": pregrasp_pose_fb,
    #             "grip_width": dist,
    #             "confidence": 1.0 / (1.0 + dist_from_centroid),
    #             "_p1": p1.tolist(),
    #             "_p2": p2.tolist(),
    #         })

    # -------------------------------------------------- pair → grasp pose
    n = len(grip_pts)

    for i in range(n):
        for j in range(i + 1, n):
            p1 = grip_pts[i]["point"]
            p2 = grip_pts[j]["point"]
            dist = float(np.linalg.norm(p2 - p1))

            if dist > span:
                continue

            secant = (p2 - p1) / dist
            center = 0.5 * (p1 + p2)
            # closing_dir is perpendicular to the secant in XY — consistent with
            # the axis fallback (jaw_dir = cross(axis_2d, approach)).
            closing_dir = np.array([-secant[1], secant[0]])
            rotation = _rotation_top_down(closing_dir)

            # Grasp z: median z of the 3D points nearest to the grasp center XY.
            # A global height fraction (e.g. 75% of z-range) places the grasp
            # in the wrong region when the graspable feature is at a different
            # height than the object's overall maximum (e.g. a screwdriver shaft
            # near the bottom vs a wide handle near the top).
            xy_dists = np.linalg.norm(xy - center, axis=1)
            near_mask = xy_dists < max(float(dist), 0.02)
            if not np.any(near_mask):
                near_mask = xy_dists < (xy_dists.min() * 3 + 0.005)
            z_grasp = (bin_floor_z if bin_floor_z is not None
                       else float(np.median(points[near_mask, 2]))) + grasp_z_offset

            grasp_pose: Se3Pose = {
                "position": {"x": float(center[0]), "y": float(center[1]),
                             "z": z_grasp},
                "rotation": rotation,
            }
            pregrasp_pose: Se3Pose = {
                "position": {"x": float(center[0]), "y": float(center[1]),
                             "z": z_grasp + _STANDOFF_M},
                "rotation": rotation,
            }

            dist_from_centroid = float(np.linalg.norm(center - centroid))
            confidence = 1.0 / (1.0 + dist_from_centroid)

            candidates.append({
                "cand_index": len(candidates),
                "grasp_pose": grasp_pose,
                "pregrasp_pose": pregrasp_pose,
                "grip_width": dist,
                "confidence": confidence,
                "_p1": p1.tolist(),
                "_p2": p2.tolist(),
            })

    if not candidates:
        logger.info(
            "contour_concave: falling back to axis grasp for '%s' — "
            "all pairs exceed the %.0f mm gripper span",
            name, span * 1000,
        )
        axis_cand = _axis_grasp_candidate(ctx, target_cloud)
        if axis_cand is None:
            _visualize(xy, polygon, grip_pts, [], name)  # disk only; no overlay to return
            raise RuntimeError(
                f"contour_concave grasp: no valid grasp candidates for '{name}' — "
                f"contour, antipodal, and axis fallback all failed."
            )
        candidates = [axis_cand]

    candidates.sort(key=lambda c: -c["confidence"])
    candidates = candidates[:max_candidates]
    for idx, c in enumerate(candidates):
        c["cand_index"] = idx

    best = candidates[0]
    z_centre = float(target_obb["center"]["z"])
    z_grasp = float(best["grasp_pose"]["position"]["z"])

    overlay = _visualize(xy, polygon, grip_pts, candidates, name)

    return {
        "grasp_pose": best["grasp_pose"],
        "pregrasp_pose": best["pregrasp_pose"],
        "candidates": candidates,
        "grip_width": best["grip_width"],
        "confidence": best["confidence"],
        "grasp_index": 0,
        "num_object_points": n_pts,
        "grip_height": z_grasp - z_centre,
        "contour_boundary_overlay": overlay,
    }
