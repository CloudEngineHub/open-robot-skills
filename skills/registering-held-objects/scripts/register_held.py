"""Express a held functional feature and collision cloud in the TCP frame."""

from typing import Any, TypedDict

import numpy as np
from gap import NodeContext
from scipy.spatial.transform import Rotation


class Output(TypedDict):
    feature_in_tcp: dict[str, Any]
    object_in_tcp: dict[str, Any]
    attached_object: dict[str, Any]
    registration_confidence: float


def _matrix(pose: dict[str, Any]) -> np.ndarray:
    p, q = pose["position"], pose["rotation"]
    out = np.eye(4)
    out[:3, :3] = Rotation.from_quat([q["x"], q["y"], q["z"], q["w"]]).as_matrix()
    out[:3, 3] = [p["x"], p["y"], p["z"]]
    return out


def _pose(matrix: np.ndarray) -> dict[str, Any]:
    q = Rotation.from_matrix(matrix[:3, :3]).as_quat()
    return {
        "position": {"x": float(matrix[0, 3]), "y": float(matrix[1, 3]), "z": float(matrix[2, 3])},
        "rotation": {"w": float(q[3]), "x": float(q[0]), "y": float(q[1]), "z": float(q[2])},
    }


def _points(cloud: dict[str, Any]) -> np.ndarray:
    return np.asarray(cloud["points"], dtype=np.float64).reshape(-1, 3)


def _transform_cloud(cloud: dict[str, Any], transform: np.ndarray) -> dict[str, Any]:
    points = _points(cloud)
    moved = dict(cloud)
    moved["points"] = (transform[:3, :3] @ points.T).T + transform[:3, 3]
    return moved


def _align_normal_preserving_roll(base: np.ndarray, observed: np.ndarray) -> np.ndarray:
    """Use an observed loop normal without inventing rotation within its plane.

    A circle determines its plane normal but has no observable in-plane roll.
    ``fit_planar_feature`` may therefore return an arbitrary roll that changes
    between cameras.  Apply only the minimum rotation needed to align normals
    and preserve the prior rigid feature frame around that normal.
    """
    old_normal = np.asarray(base[:3, 2], dtype=np.float64)
    new_normal = np.asarray(observed[:3, 2], dtype=np.float64)
    old_normal /= max(float(np.linalg.norm(old_normal)), 1.0e-12)
    new_normal /= max(float(np.linalg.norm(new_normal)), 1.0e-12)
    if float(old_normal @ new_normal) < 0.0:
        new_normal = -new_normal
    correction, _ = Rotation.align_vectors([new_normal], [old_normal])
    result = base.copy()
    result[:3, :3] = correction.as_matrix() @ base[:3, :3]
    return result


def _fit_loop_center(ctx: NodeContext, world_feature: np.ndarray, cloud: dict[str, Any]):
    """Fit the loop plane and circle centre; return (aligned frame, centre).

    The circle fit gives the in-plane centre and the plane normal.  An oblique
    view may expose primarily one face of a thick ring, so the coordinate along
    the normal is the robust midpoint of the observed thickness instead.
    """
    normal = world_feature[:3, 2]
    fit = ctx.tool(
        "geometry.fit_planar_feature",
        points=cloud,
        normal_hint={"x": float(normal[0]), "y": float(normal[1]), "z": float(normal[2])},
        fit_circle_center=True,
    )
    aligned = _align_normal_preserving_roll(world_feature, _matrix(fit["pose"]))
    observed_center = np.array(
        [fit["pose"]["position"][key] for key in ("x", "y", "z")], dtype=np.float64
    )
    normal = aligned[:3, 2].copy()
    normal /= max(float(np.linalg.norm(normal)), 1.0e-12)
    projected = _points(cloud) @ normal
    axial_center = 0.5 * (float(np.quantile(projected, 0.02)) + float(np.quantile(projected, 0.98)))
    observed_center += (axial_center - float(observed_center @ normal)) * normal
    return aligned, observed_center


def _directed_tip_frame(points: np.ndarray, marker_points: np.ndarray) -> np.ndarray:
    """Frame at the distal endpoint of an elongated object, z toward its marker."""
    center = np.median(points, axis=0)
    centered = points - center
    _, _, basis = np.linalg.svd(centered, full_matrices=False)
    axis = basis[0]
    if float(axis @ (np.median(marker_points, axis=0) - center)) < 0.0:
        axis = -axis
    along = centered @ axis
    tip = np.median(points[along >= np.percentile(along, 98)], axis=0)
    reference = np.array([0.0, 0.0, 1.0])
    if abs(float(reference @ axis)) > 0.9:
        reference = np.array([1.0, 0.0, 0.0])
    x_axis = np.cross(reference, axis)
    x_axis /= max(float(np.linalg.norm(x_axis)), 1.0e-12)
    y_axis = np.cross(axis, x_axis)
    frame = np.eye(4)
    frame[:3, :3] = np.column_stack((x_axis, y_axis, axis))
    frame[:3, 3] = tip
    return frame


def _observed_tip_frame(
    ctx: NodeContext, camera: dict[str, Any], points: np.ndarray, marker_description: str
) -> np.ndarray | None:
    """Re-derive a tip from the object cloud when its direction marker is visible."""
    if len(points) < 20:
        return None
    marker = ctx.tool(
        "sam3.segment_text", image=camera["rgb"], query=marker_description, max_results=2
    )
    if not marker.get("masks") or not marker.get("scores") or float(marker["scores"][0]) < 0.04:
        return None
    marker_cloud = ctx.tool(
        "geometry.mask_to_world_points",
        mask=np.asarray(marker["masks"][0], dtype=np.uint8),
        depth=camera["depth"],
        intrinsics=camera["intrinsics"],
        camera_pose=camera["pose"],
    )["points"]
    marker_points = _points(marker_cloud)
    if len(marker_points) < 6:
        return None
    return _directed_tip_frame(points, marker_points)


def run(
    ctx: NodeContext,
    cameras: list[dict[str, Any]],
    reference_cloud: dict[str, Any],
    functional_feature: dict[str, Any],
    object_description: str,
    prior_feature_in_tcp: dict[str, Any] | None = None,
    prior_object_in_tcp: dict[str, Any] | None = None,
    prior_attached_object: dict[str, Any] | None = None,
    attachment_fit_type: str = "morphit",
    attachment_source: str = "reference",
    grasp_pose: dict[str, Any] | None = None,
    direction_marker_description: str = "",
    camera_name_filter: str = "eye_in_hand",
) -> Output:
    if attachment_source not in {"reference", "observed"}:
        raise ValueError("attachment_source must be 'reference' or 'observed'")
    ee = ctx.tool("robot.get_ee_pose")["pose"]
    world_tcp = _matrix(ee)
    kind = functional_feature.get("kind")
    # The reference cloud and the pre-grasp feature were observed before the
    # object moved.  When the commanded grasp frame is known, carry both by the
    # rigid grasp transform so a later hand pose (after a lift) does not turn a
    # missed wrist detection into a false in-hand displacement.
    carry = np.eye(4)
    if grasp_pose is not None:
        carry = world_tcp @ np.linalg.inv(_matrix(grasp_pose))
    reference = (
        _transform_cloud(reference_cloud, carry) if grasp_pose is not None else reference_cloud
    )

    best = None
    for camera in cameras:
        if camera_name_filter and camera_name_filter not in camera.get("name", ""):
            continue
        if kind == "loop":
            # A loop is localized from its own description on every pass: the
            # circle fit below needs the ring, not the whole object.
            query = functional_feature.get("description") or object_description
        elif kind == "tip" or prior_feature_in_tcp is None:
            # Thin tips are unreliable language-segmentation targets. Segment
            # the complete held object and recover the distal endpoint in 3D.
            query = object_description
        else:
            query = functional_feature.get("description") or object_description
        result = ctx.tool("sam3.segment_text", image=camera["rgb"], query=query, max_results=2)
        if result.get("masks") and result.get("scores"):
            score = float(result["scores"][0])
            if best is None or score > best[0]:
                best = (score, camera, result["masks"][0])

    cloud = reference
    confidence = 0.25
    observed_points = None
    if best is not None and best[0] >= 0.05:
        confidence = best[0]
        camera, mask = best[1], best[2]
        observed = ctx.tool(
            "geometry.mask_to_world_points",
            mask=np.asarray(mask, dtype=np.uint8),
            depth=camera["depth"],
            intrinsics=camera["intrinsics"],
            camera_pose=camera["pose"],
        )["points"]
        points = _points(observed)
        if len(points) >= 8:
            accept_observed = True
            if prior_feature_in_tcp is None and kind != "loop":
                # Segmentation can merge a held tool with the arm. Reject a
                # grossly inconsistent 3D extent before it becomes an enormous
                # attached collision model and clearance waypoint.
                reference_size = float(np.linalg.norm(np.ptp(_points(reference), axis=0)))
                observed_size = float(np.linalg.norm(np.ptp(points, axis=0)))
                ratio = observed_size / max(reference_size, 1.0e-6)
                if not 0.55 <= ratio <= 1.80:
                    accept_observed = False
                    confidence = 0.25
            if accept_observed:
                cloud = observed
                observed_points = points
    reliable_registration = best is not None and best[0] >= 0.20
    tip_frame = None
    if kind == "tip" and direction_marker_description and observed_points is not None:
        tip_frame = _observed_tip_frame(ctx, best[1], observed_points, direction_marker_description)

    registration_correction = np.zeros(3, dtype=np.float64)
    if prior_feature_in_tcp is not None and reliable_registration:
        # Near a fixture, whole-object matching is easily contaminated by the
        # fixture. Localize the functional feature directly, correct its center,
        # and preserve the already-reached feature orientation in the hand.
        world_feature = world_tcp @ _matrix(prior_feature_in_tcp)
        predicted_center = world_feature[:3, 3].copy()
        if kind == "loop":
            world_feature, observed_center = _fit_loop_center(ctx, world_feature, cloud)
        elif tip_frame is not None:
            world_feature = _align_normal_preserving_roll(world_feature, tip_frame)
            observed_center = tip_frame[:3, 3]
        else:
            observed_center = np.median(_points(cloud), axis=0)
        # A rigidly held feature cannot jump far from the pose predicted by
        # its prior TCP transform. Reject masks on the gripper or background.
        if float(np.linalg.norm(observed_center - predicted_center)) <= 0.04:
            registration_correction = observed_center - predicted_center
            world_feature[:3, 3] += registration_correction
        else:
            confidence = 0.25
    elif prior_feature_in_tcp is not None:
        world_feature = world_tcp @ _matrix(prior_feature_in_tcp)
    else:
        world_feature = carry @ _matrix(functional_feature["pose"])
        if tip_frame is not None:
            world_feature = tip_frame
        elif reliable_registration and kind == "loop":
            world_feature, observed_center = _fit_loop_center(ctx, world_feature, cloud)
            world_feature[:3, 3] = observed_center
        elif reliable_registration:
            correction = np.median(_points(cloud), axis=0) - np.median(_points(reference), axis=0)
            if float(np.linalg.norm(correction)) <= 0.02:
                world_feature[:3, 3] += correction
            else:
                confidence = 0.25
    feature_tcp = _pose(np.linalg.inv(world_tcp) @ world_feature)
    for key in ("kind", "radius_inner", "radius_outer"):
        if key in functional_feature:
            feature_tcp[key] = functional_feature[key]

    # The wrist mask may isolate only the functional feature; the reference
    # cloud is the complete pre-grasp object. ``attachment_source`` chooses
    # which one bounds the collision model.
    attachment_cloud = cloud if attachment_source == "observed" else reference
    attachment = prior_attached_object
    if attachment is None:
        # MORPHIT is cuRobo's fitter and lives in the curobo bundle; the CPU
        # geometry bundle fits the surface and voxel kinds.
        fit_kind = str(attachment_fit_type).strip().lower()
        attachment = ctx.tool(
            "curobo.cloud_to_attachment"
            if fit_kind == "morphit"
            else "geometry.cloud_to_attachment",
            points=attachment_cloud,
            tcp_pose=ee,
            surface_radius=0.002,
            margin=0.002,
            max_spheres=64,
            **({} if fit_kind == "morphit" else {"fit_type": fit_kind}),
        )["attached_object"]
    if prior_object_in_tcp is not None:
        obj = world_tcp @ _matrix(prior_object_in_tcp)
        obj[:3, 3] += registration_correction
    else:
        obj = np.eye(4)
        obj[:3, 3] = np.median(_points(attachment_cloud), axis=0)
    return {
        "feature_in_tcp": feature_tcp,
        "object_in_tcp": _pose(np.linalg.inv(world_tcp) @ obj),
        "attached_object": attachment,
        "registration_confidence": float(confidence),
    }
