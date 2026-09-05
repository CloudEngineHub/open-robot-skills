"""Reconstruct a camera-only planner world that keeps the goal reachable.

Every selected RGB-D view contributes depth. The grasp target is excluded by
its perceived mask in the mask camera and by text segmentation in the other
views (so its returns do not become a static obstacle where the attached
object later rotates); the robot is excluded by text segmentation per view and
by its model-backed collision spheres when the connector provides them; a
perceived fixture mask can be excluded from the mask camera too.

Two intentional-contact zones can be carved out of the reconstruction so the
alpha-shape meshes do not bridge the free space the goal needs: a narrow
approach tube along a fixture axis ending at its tip, and a vertical corridor
through an aperture in a lid. Nearly zero-thickness depth-edge slivers and
unmistakable invalid-depth sheets are dropped because a collision checker
treats each one as exact occupied volume.
"""

from typing import Any, TypedDict

import numpy as np
from gap import NodeContext


class Output(TypedDict):
    world_config: dict[str, Any]
    mesh_names: list[str]


def _visible_robot_mask(ctx: NodeContext, camera: dict[str, Any]) -> np.ndarray | None:
    """Segment robot pixels so the depth reconstruction is not self-obstacle."""
    image = camera["rgb"]
    merged = None
    for query in ("robot arm", "robot gripper"):
        result = ctx.tool("sam3.segment_text", image=image, query=query, max_results=2)
        for mask, score in zip(result.get("masks") or [], result.get("scores") or [], strict=False):
            if float(score) < 0.10:
                continue
            candidate = np.asarray(mask) > 0
            merged = candidate if merged is None else merged | candidate
    return None if merged is None else merged.astype(np.uint8) * 255


def _visible_target_mask(
    ctx: NodeContext, camera: dict[str, Any], description: str
) -> np.ndarray | None:
    """Exclude the future attachment from every reconstructed camera view."""
    result = ctx.tool("sam3.segment_text", image=camera["rgb"], query=description, max_results=2)
    masks, scores = result.get("masks") or [], result.get("scores") or []
    if not masks or not scores or float(scores[0]) < 0.08:
        return None
    return (np.asarray(masks[0]) > 0).astype(np.uint8) * 255


def _origin_triangle_distance_2d(triangle: np.ndarray) -> float:
    """Distance from the origin to a projected 2D triangle."""
    a, b, c = triangle

    def cross(u: np.ndarray, v: np.ndarray) -> float:
        return float(u[0] * v[1] - u[1] * v[0])

    signs = [cross(b - a, -a), cross(c - b, -b), cross(a - c, -c)]
    if all(value >= -1.0e-9 for value in signs) or all(value <= 1.0e-9 for value in signs):
        return 0.0
    distances = []
    for start, end in ((a, b), (b, c), (c, a)):
        edge = end - start
        t = float(np.clip(-(start @ edge) / max(float(edge @ edge), 1.0e-12), 0.0, 1.0))
        distances.append(float(np.linalg.norm(start + t * edge)))
    return min(distances)


def _distance_to_segment_xy(point: np.ndarray, start: np.ndarray, end: np.ndarray) -> float:
    edge = end - start
    t = float(np.clip((point - start) @ edge / max(float(edge @ edge), 1.0e-12), 0.0, 1.0))
    return float(np.linalg.norm(point - (start + t * edge)))


def _point_in_triangle_xy(point: np.ndarray, triangle: np.ndarray) -> bool:
    """Whether point lies in a projected triangle, including its boundary."""
    a, b, c = triangle[:, :2]
    v0, v1, v2 = c - a, b - a, point - a
    dot00, dot01, dot02 = v0 @ v0, v0 @ v1, v0 @ v2
    dot11, dot12 = v1 @ v1, v1 @ v2
    denominator = float(dot00 * dot11 - dot01 * dot01)
    if abs(denominator) < 1.0e-12:
        return False
    u = float((dot11 * dot02 - dot01 * dot12) / denominator)
    v = float((dot00 * dot12 - dot01 * dot02) / denominator)
    return u >= -1.0e-8 and v >= -1.0e-8 and u + v <= 1.0 + 1.0e-8


def _mesh_arrays(mesh: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    raw_vertices, raw_faces = mesh.get("vertices"), mesh.get("faces")
    vertices = np.asarray([] if raw_vertices is None else raw_vertices, dtype=np.float64)
    faces = np.asarray([] if raw_faces is None else raw_faces, dtype=np.int64)
    return vertices.reshape(-1, 3), faces.reshape(-1, 3)


def _keep_faces(
    mesh: dict[str, Any], vertices: np.ndarray, faces: np.ndarray, keep: list[bool]
) -> dict[str, Any] | None:
    kept_faces = faces[np.asarray(keep, dtype=bool)]
    if not len(kept_faces):
        return None
    used = np.unique(kept_faces)
    remap = np.full(len(vertices), -1, dtype=np.int64)
    remap[used] = np.arange(len(used))
    updated = dict(mesh)
    updated["vertices"] = vertices[used].astype(np.float32)
    updated["faces"] = remap[kept_faces].astype(np.int32)
    return updated


def _carve_approach_tube(
    meshes: list[dict[str, Any]], tip: np.ndarray, outward: np.ndarray
) -> list[dict[str, Any]]:
    """Remove faces inside a narrow tube along ``outward`` ending at ``tip``."""
    reference = np.array([1.0, 0.0, 0.0])
    if abs(float(reference @ outward)) > 0.9:
        reference = np.array([0.0, 1.0, 0.0])
    basis_u = np.cross(outward, reference)
    basis_u /= np.linalg.norm(basis_u)
    basis_v = np.cross(outward, basis_u)
    carved = []
    for mesh in meshes:
        vertices, faces = _mesh_arrays(mesh)
        if not len(vertices) or not len(faces):
            carved.append(mesh)
            continue
        relative = vertices - tip
        axial = relative @ outward
        projected = np.column_stack((relative @ basis_u, relative @ basis_v))
        keep = []
        for face in faces:
            face_axial = axial[face]
            overlaps = float(face_axial.max()) >= -0.012 and float(face_axial.min()) <= 0.25
            # 55 mm admits the gripper collision envelope at the intentional
            # mate while retaining nearby fixture geometry.
            intersects = overlaps and _origin_triangle_distance_2d(projected[face]) <= 0.055
            keep.append(not intersects)
        updated = _keep_faces(mesh, vertices, faces, keep)
        if updated is not None:
            carved.append(updated)
    return carved


def _carve_corridor(
    meshes: list[dict[str, Any]], center: np.ndarray, radius: float, rim_z: float
) -> list[dict[str, Any]]:
    """Remove faces crossing a vertical corridor of ``radius`` through ``center``."""
    carved = []
    for mesh in meshes:
        vertices, faces = _mesh_arrays(mesh)
        if not len(vertices) or not len(faces):
            carved.append(mesh)
            continue
        keep = []
        for face in faces:
            triangle = vertices[face]
            z_overlap = (
                float(triangle[:, 2].max()) >= rim_z - 0.035
                and float(triangle[:, 2].min()) <= rim_z + 0.18
            )
            distances = [float(np.linalg.norm(vertex[:2] - center)) for vertex in triangle]
            for start, end in (
                (triangle[0, :2], triangle[1, :2]),
                (triangle[1, :2], triangle[2, :2]),
                (triangle[2, :2], triangle[0, :2]),
            ):
                distances.append(_distance_to_segment_xy(center, start, end))
            crosses = min(distances) <= radius or _point_in_triangle_xy(center, triangle)
            keep.append(not (z_overlap and crosses))
        updated = _keep_faces(mesh, vertices, faces, keep)
        if updated is not None:
            carved.append(updated)
    return carved


def _pack(meshes: list[dict[str, Any]]) -> dict[str, Any]:
    """One indexed mesh from many components, without bridging triangles."""
    all_vertices, all_faces, offset = [], [], 0
    for mesh in meshes:
        vertices = np.asarray(mesh["vertices"], dtype=np.float32).reshape(-1, 3)
        faces = np.asarray(mesh["faces"], dtype=np.int32).reshape(-1, 3)
        all_vertices.append(vertices)
        all_faces.append(faces + offset)
        offset += len(vertices)
    return {
        "name": "perceived_scene",
        "vertices": np.concatenate(all_vertices, axis=0),
        "faces": np.concatenate(all_faces, axis=0),
        "pose": {
            "position": {"x": 0.0, "y": 0.0, "z": 0.0},
            "rotation": {"w": 1.0, "x": 0.0, "y": 0.0, "z": 0.0},
        },
    }


def _vec(value: dict[str, float]) -> np.ndarray:
    return np.array([value[k] for k in ("x", "y", "z")], dtype=np.float64)


def run(
    ctx: NodeContext,
    observation: dict[str, Any],
    target_mask: Any,
    target_description: str = "",
    fixture_mask: Any = None,
    fixture_tip: dict[str, float] | None = None,
    fixture_axis: dict[str, float] | None = None,
    outward_sign: float = 1.0,
    corridor_center: dict[str, float] | None = None,
    corridor_radius: float = 0.0,
    corridor_rim_z: float = 0.0,
    camera_names: str = "",
    mask_camera_name: str = "overhead",
    voxel_size: float = 0.008,
    pack_meshes: bool = False,
    sheet_floor_z: float = 0.30,
    sheet_max_extent_m: float = 2.0,
) -> Output:
    """Build the world from the selected views and carve the goal's free space.

    ``camera_names`` is a comma-separated selection (empty = every camera).
    ``target_mask`` and ``fixture_mask`` apply to ``mask_camera_name``; the
    other selected views exclude the target by segmenting
    ``target_description`` when it is non-empty. The approach tube is carved
    when both ``fixture_tip`` and ``fixture_axis`` are given, along
    ``outward_sign * fixture_axis`` (use ``-1`` for an axis that points into
    the fixture). The corridor is carved when ``corridor_center`` is given
    with a positive ``corridor_radius``, between ``corridor_rim_z - 35 mm``
    and ``+180 mm``. ``sheet_floor_z``/``sheet_max_extent_m`` drop components
    lying entirely below or spanning more than the workcell (0 disables).
    """
    all_cameras = list(observation.get("cameras") or [])
    if isinstance(observation.get("cameras"), dict):
        all_cameras = list(observation["cameras"].values())
    selected = [name.strip() for name in str(camera_names).split(",") if name.strip()]
    cameras = [c for c in all_cameras if not selected or c.get("name") in selected]
    if not cameras:
        raise ValueError("collision reconstruction requires RGB-D cameras")

    object_masks: list[dict[str, Any]] = []
    for index, camera in enumerate(cameras):
        if camera.get("name") == mask_camera_name:
            object_masks.append(
                {"name": "grasp_target", "mask": target_mask, "camera_index": index}
            )
            if fixture_mask is not None:
                contact_mask = np.asarray(fixture_mask, dtype=np.uint8)
                # Some geometric fixture detectors return a full-frame mask as
                # a placeholder. Excluding that would erase the entire world.
                if (
                    contact_mask.shape == camera["depth"].shape
                    and float(np.mean(contact_mask > 0)) < 0.5
                ):
                    object_masks.append(
                        {"name": "contact_fixture", "mask": contact_mask, "camera_index": index}
                    )
        elif target_description:
            # Reconstruction happens before grasping. If the target is
            # removed only from one view, its returns in the others become a
            # static obstacle exactly where the attached object later rotates.
            target_view = _visible_target_mask(ctx, camera, target_description)
            if target_view is not None:
                object_masks.append(
                    {
                        "name": f"grasp_target_view_{index}",
                        "mask": target_view,
                        "camera_index": index,
                    }
                )
        robot_mask = _visible_robot_mask(ctx, camera)
        if robot_mask is not None:
            object_masks.append(
                {"name": f"robot_view_{index}", "mask": robot_mask, "camera_index": index}
            )

    # Capture robot geometry with the same observation. Unlike a segmentation
    # mask, this includes occluded and visually ambiguous links and remains
    # aligned with the arm pixels even after the robot subsequently moves.
    robot_spheres: list[dict[str, Any]] = []
    try:
        sphere_result = ctx.tool("motion.get_robot_collision_spheres", arm_id=-1)
        robot_spheres = list(sphere_result.get("spheres") or [])
    except Exception:
        # Portable to robots without a model-backed planner; their visual
        # masks still provide the established fallback.
        robot_spheres = []

    response = ctx.tool(
        "geometry.build_world_config",
        cameras=cameras,
        object_masks=object_masks,
        voxel_size=float(voxel_size),
        noise_eps=0.025,
        noise_min_samples=4,
        mesh_alpha=0.04,
        robot_spheres=robot_spheres,
        robot_sphere_margin=0.015,
    )
    config = response.get("config") or {"meshes": []}

    # Alpha-shape reconstruction can emit isolated, nearly zero-thickness
    # sheets at depth discontinuities. A collision checker treats each such
    # sheet as an exact obstacle; a 2 mm sliver detached from an otherwise
    # represented surface can invalidate a whole roadmap even though it is
    # not a closed occupied volume. Keep real thin objects and compact
    # clutter; reject only small, sparse sheet fragments.
    cleaned = []
    for mesh in config.get("meshes") or []:
        vertices, _ = _mesh_arrays(mesh)
        extent = np.ptp(vertices, axis=0) if len(vertices) else np.zeros(3)
        depth_edge_sliver = bool(
            0 < len(vertices) < 32 and float(extent.min()) < 0.004 and float(extent.max()) < 0.15
        )
        # Invalid/far depth from an oblique or wrist camera can back-project
        # into a giant sheet near z=0. It is not part of the workcell, but a
        # collision checker treats it as a real wall or floor.
        invalid_depth_sheet = bool(
            len(vertices)
            and (
                (sheet_floor_z > 0.0 and float(vertices[:, 2].max()) < sheet_floor_z)
                or (sheet_max_extent_m > 0.0 and float(extent.max()) > sheet_max_extent_m)
            )
        )
        if not depth_edge_sliver and not invalid_depth_sheet:
            cleaned.append(mesh)
    meshes = cleaned

    if fixture_tip is not None and fixture_axis is not None:
        # The fixture contact zone is intentionally occupied at the goal, and
        # RGB-D alpha shapes can also bridge the thin free space around a
        # shaft. Carve only a narrow perceived approach tube ending at the
        # feature; the rest of the fixture remains an obstacle.
        outward = float(outward_sign) * _vec(fixture_axis)
        outward /= max(float(np.linalg.norm(outward)), 1.0e-12)
        meshes = _carve_approach_tube(meshes, _vec(fixture_tip), outward)

    if corridor_center is not None and float(corridor_radius) > 0.0:
        # Alpha-shape reconstruction may bridge a real lid opening. Preserve
        # a vertical corridor of the caller's measured radius.
        center = np.array([corridor_center["x"], corridor_center["y"]], dtype=np.float64)
        meshes = _carve_corridor(meshes, center, float(corridor_radius), float(corridor_rim_z))

    if not meshes:
        raise ValueError("RGB-D collision reconstruction produced no scene mesh")
    if pack_meshes:
        # A planner's scene cache may admit only a few dozen meshes. RGB-D
        # clustering can legitimately return more disconnected components;
        # collision checking does not need each as a separate named obstacle.
        meshes = [_pack(meshes)]
    config["meshes"] = meshes
    mesh_names = [str(mesh["name"]) for mesh in meshes if mesh.get("name")]
    if not mesh_names:
        mesh_names = list(response.get("mesh_names") or [])
    return {"world_config": config, "mesh_names": mesh_names}
