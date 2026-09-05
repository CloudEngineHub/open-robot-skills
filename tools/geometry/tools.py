"""geometry tool bundle — pure-math perception/planning geometry ops.

Each operation is exposed as a ``@tool`` function, including the two
scalar helpers ``geometry.iou`` / ``geometry.pose_distance``. The math
lives in ``_impl.py``; this module is the typed boundary: numpy arrays +
:mod:`gap.types` TypedDicts in and out.

No model, no GPU — everything here is CPU numpy/scipy/Open3D/sklearn/cv2.
Heavy optional imports (open3d, sklearn, cv2) happen inside the functions
that need them, so importing this module is always cheap.
"""

from __future__ import annotations

import math
from typing import TypedDict

import numpy as np
from gap_core.tools import tool
from gap_core.types import (
    CameraFrame,
    GraspCandidates,
    JointState,
    Mask,
    OrientedBoundingBox,
    PointCloud,
    Quaternion,
    Se3Pose,
    Vec3,
    WorldConfig,
    pose_to_matrix,
)

# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


class PointCloudResult(TypedDict):
    points: PointCloud


class ObbResult(TypedDict):
    obb: OrientedBoundingBox


class PoseResult(TypedDict):
    pose: Se3Pose


class PointResult(TypedDict):
    point: Vec3


class PositionResult(TypedDict):
    position: Vec3


class QuatResult(TypedDict):
    quat: Quaternion


class DistanceResult(TypedDict):
    distance: float


class IouResult(TypedDict):
    iou: float


class GraspCandidatesResult(TypedDict):
    candidates: GraspCandidates


class FrontGraspResult(TypedDict):
    grasp_pose: Se3Pose
    pre_grasp_pose: Se3Pose
    approach_direction: Vec3
    slide_axis: Vec3


class ObjectMaskEntry(TypedDict):
    """Named segmentation mask for build_world_config."""

    name: str
    mask: Mask
    camera_index: int


class WorldConfigResult(TypedDict):
    config: WorldConfig
    mesh_names: list[str]


class PlanarFeatureResult(TypedDict):
    pose: Se3Pose
    center: Vec3
    normal: Vec3
    radius: float
    radius_inner: float
    radius_outer: float
    planarity: float
    inlier_count: int


class LinearFeatureResult(TypedDict):
    center: Vec3
    axis: Vec3
    endpoint_min: Vec3
    endpoint_max: Vec3
    length: float
    linearity: float
    inlier_count: int


class AttachmentResult(TypedDict):
    attached_object: dict[str, Any]


class FeatureMateResult(TypedDict):
    mate_pose: Se3Pose
    approach_pose: Se3Pose
    engaged_pose: Se3Pose
    approach_axis: Vec3
    seating_distance: float
    minimum_clearance: float


def _pc(points: np.ndarray) -> PointCloud:
    return {"points": np.asarray(points, dtype=np.float32).reshape(-1, 3)}


# ---------------------------------------------------------------------------
# Back-projection / transforms
# ---------------------------------------------------------------------------


@tool(
    name="geometry.depth_to_point_cloud",
    summary="Convert a metric depth image to a 3D point cloud in the camera frame.",
    tags=("perception",),
)
def depth_to_point_cloud(depth: np.ndarray, intrinsics: np.ndarray) -> PointCloudResult:
    """Back-project ``depth`` (float32 [H, W], meters) through the pinhole
    ``intrinsics`` (float64 [3, 3]). Pixels with depth <= 0 are dropped."""
    from gap_skills.tools.geometry import _impl

    points = _impl._depth_to_points(
        np.asarray(depth, dtype=np.float32), np.asarray(intrinsics, dtype=np.float64)
    )
    return {"points": _pc(points)}


@tool(
    name="geometry.mask_to_world_points",
    summary="Back-project a 2D segmentation mask to 3D world points using depth + camera calibration.",
    tags=("perception",),
)
def mask_to_world_points(
    mask: Mask,
    depth: np.ndarray,
    intrinsics: np.ndarray,
    camera_pose: Se3Pose,
) -> PointCloudResult:
    """Foreground pixels of ``mask`` (uint8 0/255 [H, W]) with valid depth in
    [0.015, 20.0] m (HyRL bounds) become world-frame points via the
    camera-to-world ``camera_pose``."""
    from gap_skills.tools.geometry import _impl

    points = _impl.mask_to_world_points(
        _impl.as_mask_bool(mask),
        np.asarray(depth, dtype=np.float32),
        np.asarray(intrinsics, dtype=np.float64),
        pose_to_matrix(camera_pose),
    )
    return {"points": _pc(points)}


@tool(
    name="geometry.pixel_to_world_point",
    summary="Back-project a single pixel to a 3D world point using depth + camera calibration.",
    tags=("perception",),
)
def pixel_to_world_point(
    pixel_x: float,
    pixel_y: float,
    depth: np.ndarray,
    intrinsics: np.ndarray,
    camera_pose: Se3Pose,
) -> PointResult:
    """Raises ToolError when the pixel is out of bounds or has invalid depth."""
    from gap_skills.tools.geometry import _impl

    pt = _impl.pixel_to_world_point(
        pixel_x,
        pixel_y,
        np.asarray(depth, dtype=np.float32),
        np.asarray(intrinsics, dtype=np.float64),
        pose_to_matrix(camera_pose),
    )
    return {"point": _impl.vec3(pt)}


@tool(
    name="geometry.transform_points",
    summary="Apply a rigid SE(3) transform to a set of 3D points.",
    tags=("perception",),
)
def transform_points(points: PointCloud, transform: Se3Pose) -> PointCloudResult:
    from gap_skills.tools.geometry import _impl

    pts = _impl.as_points(points)
    if len(pts) == 0:
        return {"points": _pc(pts)}
    out = _impl._transform_points(pts, pose_to_matrix(transform))
    return {"points": _pc(out)}


def _vec3(values: np.ndarray) -> Vec3:
    return {"x": float(values[0]), "y": float(values[1]), "z": float(values[2])}


def _rotation_quaternion(matrix: np.ndarray) -> Quaternion:
    """Stable rotation-matrix to scalar-first quaternion, without a heavy import."""
    m = np.asarray(matrix, dtype=np.float64)
    trace = float(np.trace(m))
    if trace > 0.0:
        scale = math.sqrt(trace + 1.0) * 2.0
        quat = np.array([
            0.25 * scale,
            (m[2, 1] - m[1, 2]) / scale,
            (m[0, 2] - m[2, 0]) / scale,
            (m[1, 0] - m[0, 1]) / scale,
        ])
    else:
        index = int(np.argmax(np.diag(m)))
        if index == 0:
            scale = math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
            quat = np.array([(m[2, 1] - m[1, 2]) / scale, 0.25 * scale,
                             (m[0, 1] + m[1, 0]) / scale, (m[0, 2] + m[2, 0]) / scale])
        elif index == 1:
            scale = math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
            quat = np.array([(m[0, 2] - m[2, 0]) / scale, (m[0, 1] + m[1, 0]) / scale,
                             0.25 * scale, (m[1, 2] + m[2, 1]) / scale])
        else:
            scale = math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
            quat = np.array([(m[1, 0] - m[0, 1]) / scale, (m[0, 2] + m[2, 0]) / scale,
                             (m[1, 2] + m[2, 1]) / scale, 0.25 * scale])
    quat /= np.linalg.norm(quat)
    if quat[0] < 0.0:
        quat = -quat
    return {"w": float(quat[0]), "x": float(quat[1]), "y": float(quat[2]), "z": float(quat[3])}


@tool(
    name="geometry.fit_planar_feature",
    summary="Fit a robust center, plane normal, in-plane axes, and radius to a loop/opening point cloud.",
    tags=("perception", "planning"),
)
def fit_planar_feature(
    points: PointCloud,
    normal_hint: Vec3 | None = None,
    trim_fraction: float = 0.0,
    fit_circle_center: bool = False,
) -> PlanarFeatureResult:
    """Fit a 6D feature frame to a ring, loop, rim, or planar opening.

    The pose's local Z is the fitted normal. Local X is the dominant in-plane
    direction and local Y completes a right-handed frame. ``normal_hint`` only
    resolves the unavoidable normal-sign ambiguity; it does not change the fit.
    Radius is the median in-plane radial distance. When ``fit_circle_center``
    is true, the centre is refined by a robust algebraic circle fit in the
    fitted plane. This is the appropriate mode for a known circular mesh
    feature: the median of a partially visible rim lies on the visible arc,
    not at the centre of its opening.
    Trimming is opt-in: blindly removing a fixed fraction of an almost-perfect
    ring can break its symmetry through floating-point tie ordering and bias the
    fitted centre.
    """
    from gap_skills.tools.geometry import _impl

    cloud = _impl.as_points(points).astype(np.float64)
    if len(cloud) < 8:
        raise ValueError(f"fit_planar_feature needs at least 8 points, got {len(cloud)}")
    center = np.median(cloud, axis=0)
    distances = np.linalg.norm(cloud - center, axis=1)
    trim = float(np.clip(trim_fraction, 0.0, 0.40))
    if trim > 0.0 and len(cloud) >= 20:
        cutoff = float(np.quantile(distances, 1.0 - trim))
        inliers = cloud[distances <= cutoff]
    else:
        inliers = cloud
    center = np.median(inliers, axis=0)
    centered = inliers - center
    eigenvalues, eigenvectors = np.linalg.eigh(centered.T @ centered / max(len(inliers) - 1, 1))
    order = np.argsort(eigenvalues)
    normal = eigenvectors[:, order[0]]
    x_axis = eigenvectors[:, order[-1]]
    if normal_hint is not None:
        hint = np.array([normal_hint["x"], normal_hint["y"], normal_hint["z"]], dtype=np.float64)
        if np.linalg.norm(hint) > 1e-9 and float(normal @ hint) < 0.0:
            normal = -normal
    elif normal[2] < 0.0:
        normal = -normal
    x_axis -= normal * float(x_axis @ normal)
    x_axis /= np.linalg.norm(x_axis)
    y_axis = np.cross(normal, x_axis)
    y_axis /= np.linalg.norm(y_axis)
    rotation = np.column_stack((x_axis, y_axis, normal))
    if fit_circle_center and len(inliers) >= 8:
        uv = centered @ rotation[:, :2]
        circle_inliers = np.ones(len(uv), dtype=bool)
        circle_center = np.zeros(2, dtype=np.float64)
        # Iteratively discard points whose radius is inconsistent with the
        # dominant circular rim. This rejects shaft/finger pixels while still
        # allowing a partially occluded arc to determine its geometric centre.
        for _ in range(3):
            sample = uv[circle_inliers]
            design = np.column_stack(
                (2.0 * sample[:, 0], 2.0 * sample[:, 1], np.ones(len(sample)))
            )
            circle_center = np.linalg.lstsq(
                design, np.sum(sample * sample, axis=1), rcond=None
            )[0][:2]
            radii = np.linalg.norm(uv - circle_center, axis=1)
            median_radius = float(np.median(radii[circle_inliers]))
            residual = np.abs(radii - median_radius)
            cutoff = max(0.0015, float(np.quantile(residual[circle_inliers], 0.80)))
            updated = residual <= cutoff
            if np.count_nonzero(updated) < 8 or np.array_equal(updated, circle_inliers):
                break
            circle_inliers = updated
        center = center + rotation[:, :2] @ circle_center
        centered = inliers - center
    radial = np.linalg.norm(centered @ rotation[:, :2], axis=1)
    radius = float(np.median(radial))
    # A segmented loop is an annulus, not an infinitesimal circle. Robust
    # radial quantiles expose the opening and outer rim without CAD knowledge;
    # five percent rejects the sparse centre/background contamination visible
    # in RGB-D masks while tracking the physical inner edge.
    radius_inner = float(np.quantile(radial, 0.05))
    radius_outer = float(np.quantile(radial, 0.95))
    scale = max(float(eigenvalues[order[-1]]), 1e-12)
    planarity = float(np.clip(1.0 - eigenvalues[order[0]] / scale, 0.0, 1.0))
    position = _vec3(center)
    return {
        "pose": {"position": position, "rotation": _rotation_quaternion(rotation)},
        "center": position,
        "normal": _vec3(normal),
        "radius": radius,
        "radius_inner": radius_inner,
        "radius_outer": radius_outer,
        "planarity": planarity,
        "inlier_count": int(len(inliers)),
    }


@tool(
    name="geometry.fit_linear_feature",
    summary="Fit a robust 3D axis and endpoints to a rod, peg, shaft, or hook protrusion point cloud.",
    tags=("perception", "planning"),
)
def fit_linear_feature(
    points: PointCloud,
    axis_hint: Vec3 | None = None,
    trim_fraction: float = 0.05,
) -> LinearFeatureResult:
    """Fit the principal line of an elongated feature.

    Endpoint sign is inherently ambiguous. ``axis_hint`` selects the endpoint
    in the hinted direction as ``endpoint_max``. Percentile endpoints suppress
    isolated depth spikes without assuming a particular world axis.
    """
    from gap_skills.tools.geometry import _impl

    cloud = _impl.as_points(points).astype(np.float64)
    if len(cloud) < 8:
        raise ValueError(f"fit_linear_feature needs at least 8 points, got {len(cloud)}")
    center = np.median(cloud, axis=0)
    centered = cloud - center
    eigenvalues, eigenvectors = np.linalg.eigh(centered.T @ centered / max(len(cloud) - 1, 1))
    axis = eigenvectors[:, int(np.argmax(eigenvalues))]
    if axis_hint is not None:
        hint = np.array([axis_hint["x"], axis_hint["y"], axis_hint["z"]], dtype=np.float64)
        if np.linalg.norm(hint) > 1e-9 and float(axis @ hint) < 0.0:
            axis = -axis
    elif axis[0] < 0.0:
        axis = -axis
    projection = centered @ axis
    trim = float(np.clip(trim_fraction, 0.0, 0.40))
    low, high = np.quantile(projection, [trim, 1.0 - trim])
    endpoint_min = center + float(low) * axis
    endpoint_max = center + float(high) * axis
    total = max(float(np.sum(np.maximum(eigenvalues, 0.0))), 1e-12)
    linearity = float(np.clip(float(np.max(eigenvalues)) / total, 0.0, 1.0))
    return {
        "center": _vec3(center),
        "axis": _vec3(axis),
        "endpoint_min": _vec3(endpoint_min),
        "endpoint_max": _vec3(endpoint_max),
        "length": float(high - low),
        "linearity": linearity,
        "inlier_count": int(len(cloud)),
    }


def _pose_matrix(pose: Se3Pose) -> np.ndarray:
    p, q = pose["position"], pose["rotation"]
    w, x, y, z = (float(q[k]) for k in ("w", "x", "y", "z"))
    rotation = np.array([
        [1 - 2 * (y*y + z*z), 2 * (x*y - z*w), 2 * (x*z + y*w)],
        [2 * (x*y + z*w), 1 - 2 * (x*x + z*z), 2 * (y*z - x*w)],
        [2 * (x*z - y*w), 2 * (y*z + x*w), 1 - 2 * (x*x + y*y)],
    ], dtype=np.float64)
    out = np.eye(4, dtype=np.float64)
    out[:3, :3], out[:3, 3] = rotation, [p["x"], p["y"], p["z"]]
    return out


def _matrix_pose(matrix: np.ndarray) -> Se3Pose:
    return {
        "position": _vec3(matrix[:3, 3]),
        "rotation": _rotation_quaternion(matrix[:3, :3]),
    }


@tool(
    name="geometry.cloud_to_attachment",
    summary="Approximate a world-frame held-object cloud with bounded TCP-frame collision spheres.",
    tags=("geometry", "planning"),
)
def cloud_to_attachment(
    points: PointCloud,
    tcp_pose: Se3Pose,
    voxel_size: float = 0.008,
    max_spheres: int = 64,
    margin: float = 0.002,
    fit_type: str = "morphit",
    surface_radius: float = 0.003,
) -> AttachmentResult:
    cloud = np.asarray(points["points"], dtype=np.float64).reshape(-1, 3)
    if len(cloud) < 4:
        raise ValueError("cloud_to_attachment needs at least four points")
    pitch = float(np.clip(voxel_size, 0.003, 0.030))
    tcp_world = np.linalg.inv(_pose_matrix(tcp_pose))
    local = (tcp_world @ np.c_[cloud, np.ones(len(cloud))].T).T[:, :3]
    method = str(fit_type).strip().lower()
    count = max(1, int(max_spheres))
    if method == "morphit":
        import trimesh
        from curobo._src.geom.sphere_fit import SphereFitType, fit_spheres_to_mesh

        mesh = trimesh.points.PointCloud(local).convex_hull
        result = fit_spheres_to_mesh(
            mesh,
            num_spheres=count,
            surface_radius=float(np.clip(surface_radius, 0.001, 0.010)),
            fit_type=SphereFitType.MORPHIT,
            iterations=200,
            compute_metrics=True,
        )
        centers = result.centers.detach().cpu().numpy().reshape(-1, 3)
        radii = result.radii.detach().cpu().numpy().reshape(-1)
        shrink = float(max(0.0, margin))
        radii = np.maximum(radii - shrink, 0.001)
        valid = np.isfinite(centers).all(axis=1) & np.isfinite(radii) & (radii > 0.0)
        if not np.any(valid):
            raise RuntimeError("CuRobo MORPHIT fitting returned no attachment spheres")
        return {"attached_object": {
            "frame": "tcp",
            "sphere_fit_type": "morphit",
            "sphere_radius_shrink_m": shrink,
            "spheres": [
                {"center": _vec3(center), "radius": float(radius)}
                for center, radius in zip(centers[valid], radii[valid])
            ],
        }}
    if method == "surface":
        radius = float(np.clip(surface_radius, 0.001, 0.010))
        # Deterministic farthest-point sampling preserves the measured surface
        # of thin/elongated objects without filling their interior with large
        # voxel spheres. This is intentionally planner-independent.
        chosen = [int(np.argmax(np.linalg.norm(local - np.mean(local, axis=0), axis=1)))]
        nearest = np.linalg.norm(local - local[chosen[0]], axis=1)
        while len(chosen) < min(count, len(local)):
            index = int(np.argmax(nearest))
            if nearest[index] <= radius * 0.5:
                break
            chosen.append(index)
            nearest = np.minimum(nearest, np.linalg.norm(local - local[index], axis=1))
        spheres = [(local[index], radius) for index in chosen]
        return {"attached_object": {"frame": "tcp", "sphere_fit_type": "surface", "spheres": [
            {"center": _vec3(center), "radius": sphere_radius}
            for center, sphere_radius in spheres
        ]}}
    if method != "voxel":
        raise ValueError("fit_type must be 'morphit', 'surface', or 'voxel'")
    cells = np.floor(local / pitch).astype(np.int64)
    _, inverse = np.unique(cells, axis=0, return_inverse=True)
    spheres = []
    for index in range(int(inverse.max()) + 1):
        sample = local[inverse == index]
        center = np.mean(sample, axis=0)
        radius = float(np.max(np.linalg.norm(sample - center, axis=1)) + margin)
        spheres.append((center, float(np.clip(radius, 0.003, np.sqrt(3) * pitch / 2))))
    if len(spheres) > count:
        keep = np.linspace(0, len(spheres) - 1, count).round().astype(int)
        spheres = [spheres[i] for i in keep]
    return {"attached_object": {"frame": "tcp", "sphere_fit_type": "voxel", "spheres": [
        {"center": _vec3(center), "radius": radius} for center, radius in spheres
    ]}}


@tool(
    name="geometry.compute_feature_mate",
    summary="Compute approach and mate TCP poses for loop-over-shaft or shaft-into-aperture relations.",
    tags=("geometry", "planning"),
)
def compute_feature_mate(
    held_feature_in_tcp: Se3Pose,
    fixture_pose: Se3Pose,
    fixture_axis: Vec3,
    relation: str,
    held_radius: float = 0.0,
    fixture_radius: float = 0.0,
    approach_margin: float = 0.05,
    seating_margin: float = 0.012,
    reference_tcp_pose: Se3Pose | None = None,
    attached_object: dict | None = None,
) -> FeatureMateResult:
    relation = str(relation)
    aliases = {
        "tip_through_aperture": "shaft_into_aperture",
        "insert_through": "shaft_into_aperture",
    }
    relation = aliases.get(relation, relation)
    if relation not in {"loop_over_shaft", "shaft_into_aperture"}:
        raise ValueError(f"unsupported feature relation {relation!r}")
    axis = np.array([fixture_axis[k] for k in ("x", "y", "z")], dtype=np.float64)
    norm = float(np.linalg.norm(axis))
    if norm < 1e-9:
        raise ValueError("fixture axis must be non-zero")
    axis /= norm
    clearance = float(held_radius - fixture_radius)
    if relation == "loop_over_shaft" and held_radius > 0 and clearance <= 0:
        raise ValueError("loop inner radius does not clear fixture shaft")
    if relation == "shaft_into_aperture" and held_radius > 0 and fixture_radius > 0:
        clearance = float(fixture_radius - held_radius)
        if clearance <= 0:
            raise ValueError("shaft radius does not clear aperture")

    fixture = _pose_matrix(fixture_pose)
    # Desired feature frame: local Z follows the directed fixture axis.
    reference = np.array([0.0, 0.0, 1.0])
    if abs(float(reference @ axis)) > 0.9:
        reference = np.array([0.0, 1.0, 0.0])
    x_axis = np.cross(reference, axis); x_axis /= np.linalg.norm(x_axis)
    y_axis = np.cross(axis, x_axis)
    if reference_tcp_pose is not None:
        # A loop over a shaft is invariant to roll about the shaft. Choose the
        # equivalent feature frame that requires the least TCP reorientation,
        # instead of coupling robot reachability to an arbitrary PCA axis.
        tcp_feature_rotation = _pose_matrix(held_feature_in_tcp)[:3, :3]
        reference_rotation = _pose_matrix(reference_tcp_pose)[:3, :3]
        best = None
        for angle in np.linspace(0.0, 2.0 * np.pi, 72, endpoint=False):
            candidate_x = np.cos(angle) * x_axis + np.sin(angle) * y_axis
            candidate_y = -np.sin(angle) * x_axis + np.cos(angle) * y_axis
            feature_rotation = np.column_stack((candidate_x, candidate_y, axis))
            tcp_rotation = feature_rotation @ tcp_feature_rotation.T
            score = float(np.trace(reference_rotation.T @ tcp_rotation))
            if best is None or score > best[0]:
                best = (score, candidate_x, candidate_y)
        _, x_axis, y_axis = best
    world_feature = np.eye(4)
    world_feature[:3, :3] = np.column_stack((x_axis, y_axis, axis))
    world_feature_engaged = world_feature.copy()
    penetration = 0.0
    if relation == "loop_over_shaft":
        # Move the loop beyond the observed free tip, then let gravity seat the
        # shaft against the loop's inner edge. Both distances come from the
        # perceived feature sizes; no object-class waypoint is involved.
        penetration = max(
            2.0 * float(held_radius),
            float(seating_margin),
        )
        radial_clearance = max(float(held_radius), float(fixture_radius))
        world_feature_engaged[:3, 3] = fixture[:3, 3] - penetration * axis
        world_feature_engaged[2, 3] += radial_clearance
        world_feature = world_feature_engaged.copy()
        world_feature[2, 3] -= 2.0 * radial_clearance
    else:
        world_feature[:3, 3] = fixture[:3, 3] + float(seating_margin) * axis
        world_feature_engaged = world_feature.copy()
    tcp_feature = _pose_matrix(held_feature_in_tcp)
    world_tcp_mate = world_feature @ np.linalg.inv(tcp_feature)
    world_tcp_engaged = world_feature_engaged @ np.linalg.inv(tcp_feature)
    world_feature_approach = world_feature_engaged.copy()
    approach_shift = float(approach_margin + seating_margin)
    spheres = list((attached_object or {}).get("spheres") or [])
    if relation == "loop_over_shaft" and spheres:
        # Keep the complete held body on the free side of the fixture before
        # insertion. Ring radius alone is insufficient for long tools whose
        # handle sweeps near the hook during reorientation.
        tcp_rotation = world_tcp_engaged[:3, :3]
        feature_tcp_position = tcp_feature[:3, 3]
        projected = []
        for sphere in spheres:
            center = np.array(
                [sphere["center"][key] for key in ("x", "y", "z")],
                dtype=np.float64,
            )
            offset_world = tcp_rotation @ (center - feature_tcp_position)
            projected.append(
                -float(offset_world @ axis) + float(sphere.get("radius", 0.0))
            )
        approach_shift = max(approach_shift, penetration + max(projected) + 0.005)
    world_feature_approach[:3, 3] += approach_shift * axis
    world_tcp_approach = world_feature_approach @ np.linalg.inv(tcp_feature)
    return {
        "mate_pose": _matrix_pose(world_tcp_mate),
        "approach_pose": _matrix_pose(world_tcp_approach),
        "engaged_pose": _matrix_pose(world_tcp_engaged),
        "approach_axis": _vec3(axis),
        "seating_distance": float(seating_margin),
        "minimum_clearance": float(max(clearance, 0.0)),
    }


# ---------------------------------------------------------------------------
# Filtering + OBB fitting
# ---------------------------------------------------------------------------


@tool(
    name="geometry.exclude_robot_points",
    summary="Remove points near the robot body via FK-based sphere exclusion "
            "(7-DOF Franka; other arms pass through unchanged).",
    tags=("perception",),
)
def exclude_robot_points(
    points: PointCloud,
    joint_positions: JointState,
    distance_threshold: float = 0.05,
) -> PointCloudResult:
    """Strip robot-body points from a perception cloud (HyRL RobotSegmenter
    concept, simplified FK + link spheres). Essential when the perceived
    object sits against the robot base — the segmentation mask bleeds onto
    robot pixels and the merged cloud yields a wildly oversized OBB."""
    import numpy as np

    from gap_skills.tools.geometry import _impl

    pts = _impl.as_points(points)
    joints = np.asarray(joint_positions["positions"], dtype=np.float64).reshape(-1)
    if joints.shape[0] != 7:
        return {"points": _pc(pts)}
    return {
        "points": _pc(
            _impl._exclude_robot_points(pts, joints, distance_threshold)
        )
    }


@tool(
    name="geometry.filter_noise",
    summary="Filter point-cloud noise with DBSCAN clustering (keeps all non-noise points).",
    tags=("perception",),
)
def filter_noise(
    points: PointCloud,
    eps: float = 0.005,
    min_samples: int = 10,
) -> PointCloudResult:
    """Mirrors HyRL filter_noise: keeps ALL non-noise points (labels != -1),
    not just the largest cluster. If everything is classified as noise the
    original cloud is returned unchanged."""
    from gap_skills.tools.geometry import _impl

    pts = _impl.as_points(points)
    return {"points": _pc(_impl.filter_noise(pts, eps, min_samples))}


@tool(
    name="geometry.compute_obb",
    summary="Fit an oriented bounding box to 3D points (HyRL contour-based min-width fit, upright in Z).",
    tags=("perception",),
)
def compute_obb(points: PointCloud) -> ObbResult:
    """Statistical outlier removal → XY rasterization → contour polygon →
    min-width rectangle search → 2nd/98th percentile extents. The returned
    OBB is upright (rotation only around world Z); ``extent`` holds
    HALF-extents per gap.types. Raises PerceptionFailed on < 4 points."""
    from gap_skills.tools.geometry import _impl

    return {"obb": _impl.compute_obb(_impl.as_points(points))}


@tool(
    name="geometry.filter_and_compute_obb",
    summary="DBSCAN-filter a point cloud then fit its oriented bounding box in one call.",
    tags=("perception",),
)
def filter_and_compute_obb(
    points: PointCloud,
    eps: float = 0.005,
    min_samples: int = 10,
) -> ObbResult:
    """Sequences geometry.filter_noise + geometry.compute_obb (the servicer
    offered this fusion to avoid two round trips; kept for workflow parity)."""
    from gap_skills.tools.geometry import _impl

    pts = _impl.as_points(points)
    filtered = _impl.filter_noise(pts, eps, min_samples)
    return {"obb": _impl.compute_obb(filtered)}


# ---------------------------------------------------------------------------
# Grasp-pose derivation
# ---------------------------------------------------------------------------


@tool(
    name="geometry.top_down_grasp_from_obb",
    summary="Compute a single world-aligned top-down grasp pose from an oriented bounding box.",
    tags=("planning",),
)
def top_down_grasp_from_obb(obb: OrientedBoundingBox, z_offset: float = 0.0) -> PoseResult:
    """Gripper points straight down world -Z above the OBB centre; Z lands on
    the world-frame top surface plus ``z_offset`` (negative = lower, typical
    -0.06 for bottles), clamped to 5 cm below the table plane."""
    from gap_skills.tools.geometry import _impl

    return {"pose": _impl.compute_top_down_grasp_world_aligned(obb, z_offset)}


@tool(
    name="geometry.top_down_grasp_candidates",
    summary="Fan out top-down grasp candidates (canonical primary+alt first, then 8 yaws x 3 depths, plus pitched side-grasps for flat boxes only).",
    tags=("planning",),
)
def top_down_grasp_candidates(
    obb: OrientedBoundingBox,
    z_offset: float = -0.04,
) -> GraspCandidatesResult:
    """poses[0]/poses[1] reproduce the legacy 2-pose RPC exactly; the rest are
    enriched candidates for a planner goalset. Default ``z_offset=-0.04``
    puts the fingertip 4 cm below the OBB top — with 0.0 the fingers close
    above the object (silent empty grip)."""
    from gap_skills.tools.geometry import _impl

    poses = _impl.top_down_grasp_candidates(obb, z_offset)
    return {"candidates": {"poses": poses}}


@tool(
    name="geometry.select_top_down_grasp",
    summary="Select the most top-down oriented grasp from candidates (gripper distance as tiebreaker).",
    tags=("planning",),
)
def select_top_down_grasp(
    grasp_poses: list[Se3Pose],
    gripper_position: Vec3 | None = None,
) -> PoseResult:
    from gap_skills.tools.geometry import _impl

    return {"pose": _impl.select_top_down_grasp(grasp_poses, gripper_position)}


@tool(
    name="geometry.front_grasp_from_obb",
    summary="Compute front-approach grasp + pre-grasp poses for a handle from its OBB (drawers, doors).",
    tags=("planning",),
)
def front_grasp_from_obb(
    obb: OrientedBoundingBox,
    approach_offset: float = 0.08,
    approach_hint: Vec3 | None = None,
    z_offset: float = 0.0,
) -> FrontGraspResult:
    """Derives approach direction and slide axis from the OBB orientation.
    ``approach_hint`` points from the handle toward the robot (default:
    OBB centre → origin, XY only). Raises PlanningFailed when the approach
    direction is near-vertical — use top_down_grasp_from_obb instead."""
    from gap_skills.tools.geometry import _impl

    out = _impl.front_grasp_from_obb(obb, approach_offset, approach_hint, z_offset)
    return {
        "grasp_pose": out["grasp_pose"],
        "pre_grasp_pose": out["pre_grasp_pose"],
        "approach_direction": out["approach_direction"],
        "slide_axis": out["slide_axis"],
    }


# ---------------------------------------------------------------------------
# World reconstruction
# ---------------------------------------------------------------------------


@tool(
    name="geometry.build_world_config",
    summary="Build a planner-agnostic multi-surface collision world from RGB-D camera frames.",
    tags=("planning",),
)
def build_world_config(
    cameras: list[CameraFrame],
    object_masks: list[ObjectMaskEntry] | None = None,
    voxel_size: float = 0.005,
    noise_eps: float = 0.01,
    noise_min_samples: int = 5,
    table_z_threshold: float = 0.0,
    mesh_alpha: float = 0.03,
    robot_joint_state: JointState | None = None,
    robot_distance_threshold: float = 0.15,
    robot_spheres: list[dict] | None = None,
    robot_sphere_margin: float = 0.015,
    robot_file: str = "franka.yml",
    target_obb: OrientedBoundingBox | None = None,
    target_obb_name: str = "target",
) -> WorldConfigResult:
    """Pipeline: depth → merged world cloud → voxel downsample → optional
    FK-based robot-point exclusion → DBSCAN significant-cluster filtering →
    table removal (when ``table_z_threshold`` != 0; typical -0.01) → iterative
    plane separation → collision slabs or residual alpha-shape meshes, with ``object_masks``
    (or a projected ``target_obb``)
    points excluded so planners can ignore the grasp target by name.

    ``robot_file`` is accepted for parity with the service request but the
    FK exclusion is Franka-only (simplified DH model); non-7-DOF joint
    states skip exclusion. Returns an empty WorldConfig if no geometry can
    be reconstructed."""
    from gap_skills.tools.geometry import _impl

    config, mesh_names = _impl.build_world_config(
        cameras,
        list(object_masks or []),
        voxel_size=voxel_size if voxel_size > 0 else 0.005,
        noise_eps=noise_eps if noise_eps > 0 else 0.01,
        noise_min_samples=noise_min_samples if noise_min_samples > 0 else 5,
        table_z_threshold=table_z_threshold,
        mesh_alpha=mesh_alpha if mesh_alpha > 0 else 0.03,
        robot_joint_state=robot_joint_state,
        robot_distance_threshold=(
            robot_distance_threshold if robot_distance_threshold > 0 else 0.15
        ),
        robot_spheres=robot_spheres,
        robot_sphere_margin=robot_sphere_margin,
        target_obb=target_obb,
        target_obb_name=target_obb_name,
    )
    return {"config": config, "mesh_names": mesh_names}


# ---------------------------------------------------------------------------
# Utility ops (merged from the geometry_utils skill, same as the servicer)
# ---------------------------------------------------------------------------


@tool(
    name="geometry.rotate_quat_z90",
    summary="Rotate a wxyz quaternion by 90 degrees around the world Z axis.",
    tags=("planning",),
)
def rotate_quat_z90(quat: Quaternion) -> QuatResult:
    s = math.sqrt(2.0) / 2.0
    zw, zx, zy, zz = s, 0.0, 0.0, s
    q = quat
    rw = q["w"] * zw - q["x"] * zx - q["y"] * zy - q["z"] * zz
    rx = q["w"] * zx + q["x"] * zw + q["y"] * zz - q["z"] * zy
    ry = q["w"] * zy - q["x"] * zz + q["y"] * zw + q["z"] * zx
    rz = q["w"] * zz + q["x"] * zy - q["y"] * zx + q["z"] * zw
    return {"quat": {"w": rw, "x": rx, "y": ry, "z": rz}}


@tool(
    name="geometry.compute_drop_position",
    summary="Compute a drop position above a container from its oriented bounding box.",
    tags=("planning",),
)
def compute_drop_position(
    container_obb: OrientedBoundingBox,
    clearance: float = 0.05,
    object_z_extent: float = 0.0,
) -> PositionResult:
    obb = container_obb
    clearance = clearance or 0.05
    obj_z = object_z_extent or 0.0
    c = obb["center"]
    e = obb["extent"]
    drop_z = c["z"] + e["z"] / 2.0 + obj_z + clearance
    return {"position": {"x": c["x"], "y": c["y"], "z": drop_z}}


@tool(
    name="geometry.compute_xy_distance",
    summary="Euclidean distance between two 3D points projected onto the XY plane.",
    tags=("perception",),
)
def compute_xy_distance(point_a: Vec3, point_b: Vec3) -> DistanceResult:
    a, b = point_a, point_b
    dist = math.sqrt((a["x"] - b["x"]) ** 2 + (a["y"] - b["y"]) ** 2)
    return {"distance": dist}


# ---------------------------------------------------------------------------
# Legacy canary tools (ported from the dev tree's geometry_iou / pose_distance tools)
# ---------------------------------------------------------------------------


@tool(
    name="geometry.iou",
    summary="Compute IoU of two axis-aligned 2D boxes [x1, y1, x2, y2]. Returns 0 if boxes don't overlap.",
    tags=("perception",),
)
def iou(box_a: list[float], box_b: list[float]) -> IouResult:
    """Pure-Python intersection-over-union for axis-aligned 2D boxes.

    Args:
        box_a: ``[x1, y1, x2, y2]`` corners of box A.
        box_b: ``[x1, y1, x2, y2]`` corners of box B.

    Returns:
        ``{"iou": float}`` in ``[0, 1]``. Zero when the boxes don't overlap
        or either has zero area.
    """
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    ix = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    iy = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = ix * iy
    area_a = max(0.0, (ax2 - ax1) * (ay2 - ay1))
    area_b = max(0.0, (bx2 - bx1) * (by2 - by1))
    union = area_a + area_b - inter
    return {"iou": inter / union if union > 0 else 0.0}


@tool(
    name="geometry.pose_distance",
    summary="Euclidean distance between two 3D positions [x, y, z].",
    tags=("perception",),
)
def pose_distance(a: list[float], b: list[float]) -> DistanceResult:
    """Returns the Euclidean distance between two ``[x, y, z]`` points."""
    if len(a) != 3 or len(b) != 3:
        raise ValueError(f"expected 3-vectors, got len(a)={len(a)}, len(b)={len(b)}")
    dx = a[0] - b[0]
    dy = a[1] - b[1]
    dz = a[2] - b[2]
    return {"distance": math.sqrt(dx * dx + dy * dy + dz * dz)}
