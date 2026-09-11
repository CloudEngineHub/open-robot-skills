"""Localize one of several described objects and fit its planar functional feature.

The generic core of language-driven object-plus-feature perception from one
RGB-D view: every candidate row is segmented by text, the strongest detection
wins, its mask is back-projected and optionally completed down to the support
surface it rests on, and -- when the winner names a feature -- that feature is
segmented, centred and fitted as a planar loop/aperture frame whose normal
sign is resolved by the camera position.

Two ways to declare the candidates, and they are the same body:

- ``candidates``: a JSON list of ``{kind, object_description,
  feature_description, feature_type, feature_score_min}`` rows. The feature
  is fitted from its own mask; object-specific landmark rules stay in the
  graph, downstream of ``target_cloud``/``target_obb``.
- ``feature_profiles``: a list of profiles that additionally carry a
  ``strategy`` -- how the feature is located when its own mask is not enough
  -- and the constants that strategy needs. ``planar_cad_loop`` registers the
  parent's silhouette to a declared CAD and picks the bow the instruction
  refers to; ``landmark_offsets`` places the feature and the grasp at declared
  offsets from the parent's centroid; ``distal_tip`` takes the far end of the
  parent from a segmented reference part; ``distal_planar_loop`` binds a loop
  detection to the parent's long-axis corridor at scale-relative bounds.
  Every constant is the profile's, never this script's, and a profile's
  ``aliases`` extend how the instruction may name its kind.

Both forms bind a feature to the SELECTED parent instance rather than to the
globally highest-scoring mask, which is what makes a scene of two identical
objects workable. Precedence when both are given: ``feature_profiles``.
"""

import json
from functools import lru_cache
from pathlib import Path
from typing import Any, NotRequired, TypedDict

import numpy as np
from gap import NodeContext
from scipy.ndimage import binary_dilation
from scipy.spatial.transform import Rotation

#: The winning object detection must score at least this to be trusted.
OBJECT_SCORE_MIN = 0.12
#: Feature segmentation floor used when a candidate row does not state one.
DEFAULT_FEATURE_SCORE_MIN = 0.05

_STRATEGIES = {"semantic_mask", "planar_cad_loop", "landmark_offsets", "distal_tip", "distal_planar_loop"}


class Output(TypedDict):
    target_kind: str
    target_obb: dict[str, Any]
    target_mask: np.ndarray
    target_cloud: dict[str, Any]
    feature_center: dict[str, float]
    feature_pose: dict[str, Any]
    functional_feature: dict[str, Any]
    #: Only when the selected ``feature_profiles`` row names the mating relation
    #: (the ``candidates`` path returns no relation, as before).
    relation: NotRequired[str]


def _camera(observation: dict[str, Any], name: str) -> dict[str, Any]:
    cameras = observation.get("cameras") or []
    if isinstance(cameras, dict):
        cameras = list(cameras.values())
    for camera in cameras:
        if camera.get("name") == name:
            return camera
    raise ValueError(f"observation has no camera named {name!r}")


def _backproject(ctx: NodeContext, camera: dict[str, Any], mask: Any) -> dict[str, Any]:
    return ctx.tool(
        "geometry.mask_to_world_points",
        mask=np.asarray(mask, dtype=np.uint8),
        depth=camera["depth"],
        intrinsics=camera["intrinsics"],
        camera_pose=camera["pose"],
    )["points"]


def _complete_to_support(
    ctx: NodeContext, camera: dict[str, Any], mask: Any, cloud: dict[str, Any]
) -> dict[str, Any]:
    """Add a phantom bottom layer so a thin object's OBB reaches its support."""
    fg = np.asarray(mask) > 0
    ring = binary_dilation(fg, iterations=8) & ~binary_dilation(fg, iterations=2)
    support = _backproject(ctx, camera, ring.astype(np.uint8) * 255)
    pts = np.asarray(cloud["points"], dtype=np.float32).reshape(-1, 3)
    support_pts = np.asarray(support["points"], dtype=np.float32).reshape(-1, 3)
    if len(pts) < 10 or len(support_pts) < 20:
        return cloud
    support_z = float(np.median(support_pts[:, 2]))
    top_z = float(np.percentile(pts[:, 2], 75))
    if not 0.001 < top_z - support_z < 0.08:
        return cloud
    bottom = pts.copy()
    bottom[:, 2] = support_z + 0.001
    return {"points": np.concatenate([pts, bottom], axis=0).astype(np.float32)}


def _mask_center_world(ctx: NodeContext, camera: dict[str, Any], mask: Any) -> dict[str, float]:
    pts = np.asarray(_backproject(ctx, camera, mask)["points"], dtype=np.float64).reshape(-1, 3)
    if len(pts) < 8:
        raise ValueError("feature mask contains too few depth points")
    center = np.median(pts, axis=0)
    return {"x": float(center[0]), "y": float(center[1]), "z": float(center[2])}


def _best_mask(
    ctx: NodeContext,
    image: np.ndarray,
    query: str,
    threshold: float,
    camera: dict[str, Any] | None = None,
    reference_center: dict[str, float] | None = None,
) -> np.ndarray:
    """The best mask for *query* -- by score, or, given a reference, by proximity.

    With ``camera`` and ``reference_center`` the part is bound to the selected
    parent instance: in a scene of two identical objects the globally highest
    scoring ring may belong to the other one. Without them the call is the
    original three-result, best-score form.
    """
    if camera is None or reference_center is None:
        result = ctx.tool("sam3.segment_text", image=image, query=query, max_results=3)
        if not result.get("masks") or float(result["scores"][0]) < threshold:
            raise ValueError(f"SAM3 could not localize {query!r}")
        return result["masks"][0]
    result = ctx.tool("sam3.segment_text", image=image, query=query, max_results=6)
    candidates = [
        (mask, float(score))
        for mask, score in zip(result.get("masks") or [], result.get("scores") or [], strict=False)
        if float(score) >= threshold
    ]
    if not candidates:
        raise ValueError(f"SAM3 could not localize {query!r}")
    reference = np.array([reference_center[k] for k in ("x", "y", "z")], dtype=float)
    localized = []
    for mask, score in candidates:
        try:
            center = _mask_center_world(ctx, camera, mask)
        except ValueError:
            continue
        xyz = np.array([center[k] for k in ("x", "y", "z")], dtype=float)
        localized.append((float(np.linalg.norm(xyz - reference)), -score, mask))
    if not localized:
        raise ValueError(f"SAM3 localized {query!r} but no mask had usable depth")
    return min(localized, key=lambda item: item[:2])[2]


def _feature_fit(ctx: NodeContext, camera: dict[str, Any], mask: Any, normal_hint=None) -> dict[str, Any]:
    return ctx.tool(
        "geometry.fit_planar_feature",
        points=_backproject(ctx, camera, mask),
        normal_hint=normal_hint,
        fit_circle_center=True,
    )


def _point_pose(center: dict[str, float]) -> dict[str, Any]:
    """A point feature has no observed orientation; mark it with identity."""
    return {
        "position": dict(center),
        "rotation": {"w": 1.0, "x": 0.0, "y": 0.0, "z": 0.0},
    }


def _axis_pose(center: dict[str, float], axis) -> dict[str, Any]:
    """Pose whose local Z is a measured directed mating axis."""
    z_axis = np.asarray(axis, dtype=np.float64)
    z_axis /= max(float(np.linalg.norm(z_axis)), 1.0e-12)
    reference = np.array([0.0, 0.0, 1.0])
    if abs(float(reference @ z_axis)) > 0.90:
        reference = np.array([0.0, 1.0, 0.0])
    x_axis = np.cross(reference, z_axis)
    x_axis /= np.linalg.norm(x_axis)
    y_axis = np.cross(z_axis, x_axis)
    quat = Rotation.from_matrix(np.column_stack((x_axis, y_axis, z_axis))).as_quat()
    return {
        "position": dict(center),
        "rotation": {"w": float(quat[3]), "x": float(quat[0]), "y": float(quat[1]), "z": float(quat[2])},
    }


@lru_cache(maxsize=8)
def _planar_model(model_path: str) -> np.ndarray:
    """Deterministic surface samples of a flat object's CAD, in its own XY."""
    import trimesh  # noqa: PLC0415

    mesh = trimesh.load_mesh(Path(model_path), process=False)
    points, _ = trimesh.sample.sample_surface(mesh, 30000, seed=0)
    return np.asarray(points[:, :2], dtype=np.float64)


def _register_planar_model(points: np.ndarray, model_path: str) -> tuple[np.ndarray, np.ndarray, float]:
    """Register an observed silhouette to a declared flat-object CAD model.

    A semantic mask of an empty finger hole frequently covers only a coloured
    rim and can displace its reported centre by several centimetres. The full
    silhouette is far better constrained. Search yaw globally, then refine a
    planar rigid transform; the segmenter is used afterwards only to choose
    which of two symmetric bows the instruction refers to.
    """
    from scipy.spatial import cKDTree  # noqa: PLC0415

    observed = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    observed = observed[np.all(np.isfinite(observed), axis=1)]
    if len(observed) < 80:
        raise ValueError("planar CAD registration needs at least 80 points")
    observed_xy = observed[:, :2]
    model_xy = _planar_model(model_path)
    model_center, observed_center = model_xy.mean(axis=0), observed_xy.mean(axis=0)
    model_tree = cKDTree(model_xy)
    best = None
    for yaw in np.deg2rad(np.arange(0.0, 360.0, 4.0)):
        rotation = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]], dtype=np.float64)
        translation = observed_center - rotation @ model_center
        distances, _ = model_tree.query((observed_xy - translation) @ rotation, k=1)
        score = float(np.sqrt(np.mean(np.square(np.minimum(distances, 0.015)))))
        if best is None or score < best[0]:
            best = (score, rotation, translation)
    _, rotation, translation = best
    distances = np.zeros(len(observed_xy))
    keep = np.ones(len(observed_xy), dtype=bool)
    # Point-to-point refinement removes the coarse yaw quantisation while a
    # trimmed set keeps table pixels inside the openings from biasing it.
    for _ in range(30):
        world_model = model_xy @ rotation.T + translation
        distances, indices = cKDTree(world_model).query(observed_xy, k=1)
        keep = distances <= min(0.008, float(np.quantile(distances, 0.85)))
        if int(np.count_nonzero(keep)) < 60:
            break
        source, target = world_model[indices[keep]], observed_xy[keep]
        source_center, target_center = source.mean(axis=0), target.mean(axis=0)
        u, _, vt = np.linalg.svd((source - source_center).T @ (target - target_center))
        correction = vt.T @ u.T
        if np.linalg.det(correction) < 0.0:
            vt[-1] *= -1.0
            correction = vt.T @ u.T
        delta = target_center - correction @ source_center
        rotation = correction @ rotation
        translation = correction @ translation + delta
        if float(np.linalg.norm(delta)) < 2.0e-6 and abs(float(np.arctan2(correction[1, 0], correction[0, 0]))) < 2.0e-5:
            break
    return rotation, translation, float(np.sqrt(np.mean(np.square(distances[keep]))))


def _best_distal_loop_mask(ctx: NodeContext, camera: dict[str, Any], obb: dict[str, Any], profile: dict[str, Any]) -> np.ndarray:
    """Bind a semantic loop detection to the selected elongated parent.

    Another object's opening can score as the requested loop. A valid distal
    loop lies near the parent's horizontal long-axis corridor and away from
    its centroid, at bounds relative to the parent's own length.
    """
    q = obb["orientation"]
    rotation = Rotation.from_quat([q["x"], q["y"], q["z"], q["w"]]).as_matrix()
    extents = np.array([obb["extent"][k] for k in ("x", "y", "z")], dtype=float)
    horizontal = [i for i in range(3) if abs(float(rotation[2, i])) < 0.75]
    long_index = max(horizontal, key=lambda i: extents[i])
    long_axis = rotation[:, long_index].copy()
    long_axis[2] = 0.0
    long_axis /= max(float(np.linalg.norm(long_axis)), 1.0e-12)
    center = np.array([obb["center"][k] for k in ("x", "y", "z")], dtype=float)
    length = float(extents[long_index])
    min_axial = max(float(profile.get("minimum_axial", 0.035)), float(profile.get("minimum_axial_scale", 0.40)) * length)
    max_lateral = max(float(profile.get("maximum_lateral", 0.025)), float(profile.get("maximum_lateral_scale", 0.35)) * length)
    # An open jaw is often labelled as a ring, but its accidental circle fit is
    # much smaller than the closed end; the floor scales with the parent.
    min_radius = max(float(profile.get("minimum_radius", 0.004)), float(profile.get("minimum_radius_scale", 0.060)) * length)
    candidates = []
    for query in profile.get("feature_queries") or [profile["feature_query"]]:
        result = ctx.tool("sam3.segment_text", image=camera["rgb"], query=query, max_results=6)
        for mask, score in zip(result.get("masks") or [], result.get("scores") or [], strict=False):
            if float(score) < 0.05:
                continue
            try:
                feature = _mask_center_world(ctx, camera, mask)
                fit = _feature_fit(ctx, camera, mask, {"x": 0.0, "y": 0.0, "z": 1.0})
            except (ValueError, KeyError):
                continue
            radius = float(fit.get("radius_inner", fit.get("radius", 0.0)))
            delta = np.array([feature[k] for k in ("x", "y", "z")]) - center
            axial = abs(float(delta @ long_axis))
            lateral = float(np.linalg.norm(delta - (delta @ long_axis) * long_axis))
            if axial >= min_axial and lateral <= max_lateral and radius >= min_radius:
                candidates.append((-radius, lateral, -axial, -float(score), mask))
    if not candidates:
        raise ValueError("SAM3 could not localize a distal loop on the selected parent")
    return min(candidates, key=lambda item: item[:4])[4]


def _shift_grasp_toward_feature(obb: dict[str, Any], feature_center: dict[str, float], profile: dict[str, Any]) -> None:
    """Apply a caller-declared grasp displacement toward a perceived feature."""
    requested = max(0.0, float(profile.get("grasp_toward_feature_m", 0.0)))
    if requested <= 0.0:
        return
    axes = tuple(str(axis) for axis in profile.get("grasp_toward_feature_axes", ("x", "y", "z")))
    invalid = set(axes) - {"x", "y", "z"}
    if invalid:
        raise ValueError(f"invalid grasp displacement axes: {sorted(invalid)}")
    origin = np.array([obb["center"][axis] for axis in ("x", "y", "z")], dtype=np.float64)
    feature = np.array([feature_center[axis] for axis in ("x", "y", "z")], dtype=np.float64)
    direction = feature - origin
    for index, axis in enumerate(("x", "y", "z")):
        if axis not in axes:
            direction[index] = 0.0
    distance = float(np.linalg.norm(direction))
    if distance <= 1.0e-9:
        return
    separation = max(0.0, float(profile.get("minimum_grasp_feature_separation_m", 0.0)))
    shifted = origin + min(requested, max(0.0, distance - separation)) * direction / distance
    for index, axis in enumerate(("x", "y", "z")):
        if axis in axes:
            obb["center"][axis] = float(shifted[index])


def _parse_candidates(candidates: str) -> list[dict[str, Any]]:
    try:
        rows = json.loads(candidates) if isinstance(candidates, str) else list(candidates)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"candidates must be a JSON list of objects: {exc}") from exc
    if not isinstance(rows, list) or not rows:
        raise ValueError("candidates must be a non-empty JSON list")
    parsed = []
    for row in rows:
        if not isinstance(row, dict) or not str(row.get("object_description", "")).strip():
            raise ValueError(f"candidate row needs an object_description: {row!r}")
        description = str(row["object_description"]).strip()
        parsed.append(
            {
                "kind": str(row.get("kind") or description),
                "object_description": description,
                "feature_description": str(row.get("feature_description") or "").strip(),
                "feature_type": str(row.get("feature_type") or "").strip(),
                "feature_score_min": float(row.get("feature_score_min", DEFAULT_FEATURE_SCORE_MIN)),
                "strategy": "semantic_mask",
            }
        )
    return parsed


def _parse_profiles(profiles: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """A profile in the candidate row's vocabulary, its own keys kept."""
    parsed = []
    for profile in profiles:
        row = dict(profile)
        row.setdefault("kind", str(profile.get("kind") or profile.get("object_query")))
        row["object_description"] = str(profile.get("object_query") or profile.get("object_description") or "").strip()
        if not row["object_description"]:
            raise ValueError(f"feature profile needs an object_query: {profile!r}")
        row["feature_description"] = str(profile.get("feature_query") or profile.get("feature_description") or "").strip()
        row["feature_type"] = str(profile.get("feature_type") or "").strip()
        row["feature_score_min"] = float(profile.get("feature_threshold", profile.get("feature_score_min", DEFAULT_FEATURE_SCORE_MIN)))
        row["strategy"] = str(profile.get("strategy", "semantic_mask"))
        if row["strategy"] not in _STRATEGIES:
            raise ValueError(f"unsupported manipulation feature strategy {row['strategy']!r}")
        parsed.append(row)
    return parsed


def _mentions(instruction: str, row: dict[str, Any]) -> bool:
    names = row.get("aliases") or [row["kind"]]
    return any(str(name).lower() in instruction for name in names)


def run(
    ctx: NodeContext,
    observation: dict[str, Any],
    candidates: str = "",
    instruction: str = "",
    camera_name: str = "overhead",
    complete_to_support: bool = True,
    feature_profiles: list[dict[str, Any]] | None = None,
    selected_mask: np.ndarray | None = None,
    object_description: str | None = None,
    feature_description: str | None = None,
    feature_type: str | None = None,
    object_kind: str | None = None,
) -> Output:
    """Segment every candidate, keep the strongest, fit its named feature.

    ``candidates`` is a JSON list of ``{kind, object_description,
    feature_description, feature_type, feature_score_min}`` rows. When
    ``instruction`` mentions one or more candidate ``kind`` names, only those
    rows are tried. A row with an empty ``feature_description`` yields the
    OBB centre as a point feature for a downstream landmark rule to refine.

    ``feature_profiles`` is the same list with a ``strategy`` per row (see the
    module docstring); ``selected_mask`` short-circuits the object search with
    a mask an upstream node already chose, and ``object_description`` +
    ``object_kind`` + ``feature_description`` name one explicit object.
    """
    camera = _camera(observation, camera_name)
    if feature_profiles:
        rows = _parse_profiles(feature_profiles)
    elif candidates:
        rows = _parse_candidates(candidates)
    else:
        raise ValueError("either candidates or feature_profiles must declare the objects")
    instruction_lower = str(instruction).lower()
    mentioned = [row for row in rows if _mentions(instruction_lower, row)]
    if mentioned:
        rows = mentioned
    if object_description is not None:
        if object_kind is None or feature_description is None:
            raise ValueError("explicit object perception requires object_kind and feature_description")
        base = next((dict(row) for row in rows if row["kind"] == object_kind), {"strategy": "semantic_mask",
                                                                                "feature_score_min": DEFAULT_FEATURE_SCORE_MIN})
        base.update({"kind": object_kind, "object_description": object_description,
                     "feature_description": feature_description, "feature_type": base.get("feature_type", "")})
        rows = [base]
    detections = []
    if selected_mask is not None and np.asarray(selected_mask).size > 1:
        detections.append((1.0, rows[0], selected_mask))
    else:
        for row in rows:
            result = ctx.tool(
                "sam3.segment_text",
                image=camera["rgb"],
                query=row["object_description"],
                max_results=1,
            )
            if result.get("masks") and result.get("scores"):
                detections.append((float(result["scores"][0]), row, result["masks"][0]))
    if not detections:
        raise ValueError("SAM3 could not localize any of the described candidate objects")
    score, row, target_mask = max(detections, key=lambda item: item[0])
    if score < float(row.get("object_score_min", OBJECT_SCORE_MIN)):
        raise ValueError(f"best object detection was too weak ({row['kind']}: {score:.3f})")
    cloud = ctx.tool(
        "geometry.mask_to_world_points",
        mask=target_mask,
        depth=camera["depth"],
        intrinsics=camera["intrinsics"],
        camera_pose=camera["pose"],
    )["points"]
    if complete_to_support:
        cloud = _complete_to_support(ctx, camera, target_mask, cloud)
    obb = ctx.tool("geometry.filter_and_compute_obb", points=cloud)["obb"]
    feature_query = row["feature_description"]
    feature_kind = feature_type or row["feature_type"] or "loop"
    strategy = row["strategy"]
    feature_radius = 0.0
    extra: dict[str, Any] = {}

    if strategy == "semantic_mask":
        if feature_query:
            feature_mask = _best_mask(ctx, camera["rgb"], feature_query, row["feature_score_min"])
            feature_center = _mask_center_world(ctx, camera, feature_mask)
            # The camera sees the broad face of the feature. Its optical direction
            # resolves the otherwise unavoidable +/- plane-normal ambiguity.
            camera_position = camera["pose"]["position"]
            hint = {
                "x": float(camera_position["x"] - feature_center["x"]),
                "y": float(camera_position["y"] - feature_center["y"]),
                "z": float(camera_position["z"] - feature_center["z"]),
            }
            fit = _feature_fit(ctx, camera, feature_mask, hint)
            feature_pose = fit["pose"]
            feature_radius = float(fit.get("radius_inner", fit.get("radius", 0.0)))
        else:
            feature_center = {key: float(obb["center"][key]) for key in ("x", "y", "z")}
            feature_pose = _point_pose(feature_center)
    elif strategy == "planar_cad_loop":
        # An empty opening shows the support rather than the parent in depth:
        # the semantic mask gives XY and the segmented parent surface gives Z,
        # and the old silhouette guess survives only as a degraded fallback.
        parent_center = dict(obb["center"])
        try:
            feature_center = _mask_center_world(
                ctx, camera, _best_mask(ctx, camera["rgb"], feature_query, row["feature_score_min"], camera, parent_center)
            )
            feature_center["z"] = float(obb["center"]["z"])
        except (ValueError, KeyError):
            feature_center = {
                "x": float(obb["center"]["x"] + 0.43 * float(obb["extent"]["x"])),
                "y": float(obb["center"]["y"] + 0.85 * float(obb["extent"]["y"])),
                "z": float(obb["center"]["z"]),
            }
        # The parent's rigid planar frame from its silhouette; the segmenter
        # then only decides which of the symmetric bows the instruction means.
        tool_points = np.asarray(cloud["points"], dtype=np.float64).reshape(-1, 3)
        planar_rotation, origin_xy, rmse = _register_planar_model(tool_points, str(row["model_path"]))
        semantic_xy = np.array([feature_center["x"], feature_center["y"]], dtype=np.float64)
        bow_local = min(
            (np.asarray(landmark, dtype=np.float64) for landmark in row["feature_landmarks_xy"]),
            key=lambda local: float(np.linalg.norm(planar_rotation @ local + origin_xy - semantic_xy)),
        )
        feature_center["x"], feature_center["y"] = map(float, planar_rotation @ bow_local + origin_xy)
        feature_center["z"] = float(np.mean(tool_points[:, 2]))
        print(f"[perceive_object_feature] planar CAD registration: rmse={1000.0 * rmse:.2f}mm")
        # The grasp is the surface-cloud centroid -- on the solid bridge, and
        # independent of side, arm and identity -- not a point extrapolated
        # from a hole mask that may cover one coloured rim.
        if len(tool_points) < 20:
            raise ValueError("parent cloud contains too few points for a stable grasp")
        obb["center"]["x"], obb["center"]["y"] = map(float, np.mean(tool_points[:, :2], axis=0))
        rotation = np.eye(3)
        rotation[:2, :2] = planar_rotation
        quat = Rotation.from_matrix(rotation).as_quat()
        feature_pose = {"position": dict(feature_center),
                        "rotation": {"w": float(quat[3]), "x": float(quat[0]), "y": float(quat[1]), "z": float(quat[2])}}
        feature_radius = float(row.get("radius_inner", 0.0))
        extra["local_center"] = [float(bow_local[0]), float(bow_local[1]), 0.0]
    elif strategy == "landmark_offsets":
        # Declared model landmarks carried into the observed support frame
        # from the parent's centroid; no object-class branch here.
        points = np.asarray(cloud["points"], dtype=np.float64).reshape(-1, 3)
        origin = np.median(points, axis=0)
        hang = origin + np.asarray(row["feature_offset"], dtype=np.float64)
        grasp = origin + np.asarray(row["grasp_offset"], dtype=np.float64)
        feature_center = {"x": float(hang[0]), "y": float(hang[1]), "z": float(hang[2])}
        feature_pose = _point_pose(feature_center)
        obb["center"] = {"x": float(grasp[0]), "y": float(grasp[1]), "z": float(grasp[2])}
        for axis, maximum in (row.get("grasp_extent_max") or {}).items():
            obb["extent"][axis] = min(float(obb["extent"][axis]), float(maximum))
    elif strategy == "distal_tip":
        handle_mask = _best_mask(ctx, camera["rgb"], str(row["reference_query"]),
                                 float(row.get("reference_threshold", 0.05)), camera, dict(obb["center"]))
        handle_center = _mask_center_world(ctx, camera, handle_mask)
        points = np.asarray(cloud["points"], dtype=np.float64).reshape(-1, 3)
        handle = np.array([handle_center[k] for k in ("x", "y", "z")], dtype=np.float64)
        distances = np.linalg.norm(points - handle, axis=1)
        distal = points[distances >= np.percentile(distances, float(row.get("distal_percentile", 97.0)))]
        if len(distal) < 4:
            raise ValueError("parent cloud does not expose a distal endpoint")
        tip = np.median(distal, axis=0)
        feature_center = {"x": float(tip[0]), "y": float(tip[1]), "z": float(tip[2])}
        obb["center"] = handle_center
        grasp_extent = row.get("grasp_extent", {"x": 0.025, "y": 0.012, "z": 0.012})
        obb["extent"] = {axis: float(grasp_extent[axis]) for axis in ("x", "y", "z")}
        tool_axis = np.array([feature_center[k] - handle_center[k] for k in ("x", "y", "z")], dtype=np.float64)
        if float(np.linalg.norm(tool_axis)) < float(row.get("minimum_axis_length", 0.03)):
            raise ValueError("distal feature and reference part do not define a usable axis")
        feature_pose = _axis_pose(feature_center, tool_axis)
    else:  # distal_planar_loop
        feature_mask = _best_distal_loop_mask(ctx, camera, obb, {**row, "feature_query": feature_query})
        feature_center = _mask_center_world(ctx, camera, feature_mask)
        camera_position = camera["pose"]["position"]
        hint = {
            "x": float(camera_position["x"] - feature_center["x"]),
            "y": float(camera_position["y"] - feature_center["y"]),
            "z": float(camera_position["z"] - feature_center["z"]),
        }
        fit = _feature_fit(ctx, camera, feature_mask, hint)
        feature_pose = fit["pose"]
        feature_radius = float(fit.get("radius_inner", fit.get("radius", 0.0)))

    if strategy != "semantic_mask":
        _shift_grasp_toward_feature(obb, feature_center, row)
        if "relation" in row:
            extra["relation"] = str(row["relation"])
        if "registration_profile" in row:
            extra["registration_profile"] = dict(row["registration_profile"])
    fq = feature_pose["rotation"]
    feature_axis = Rotation.from_quat([fq["x"], fq["y"], fq["z"], fq["w"]]).as_matrix()[:, 2]
    functional_feature = {
        "kind": feature_kind,
        "description": feature_query,
        "pose": feature_pose,
        "axis": {
            "x": float(feature_axis[0]),
            "y": float(feature_axis[1]),
            "z": float(feature_axis[2]),
        },
        "confidence": float(score),
        "radius_inner": feature_radius,
        **{k: v for k, v in extra.items() if k in ("registration_profile", "local_center")},
    }
    out: dict[str, Any] = {
        "target_kind": row["kind"],
        "target_obb": obb,
        "target_mask": target_mask,
        "target_cloud": cloud,
        "feature_center": feature_center,
        "feature_pose": feature_pose,
        "functional_feature": functional_feature,
    }
    if "relation" in extra:
        out["relation"] = extra["relation"]
    return out  # type: ignore[return-value]
