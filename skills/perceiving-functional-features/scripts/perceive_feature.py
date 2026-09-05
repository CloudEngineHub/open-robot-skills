"""Language-localize a functional part and fit its metric geometry."""

from typing import Any, TypedDict

import numpy as np
from gap import NodeContext


class Output(TypedDict):
    feature: dict[str, Any]
    feature_mask: np.ndarray
    feature_cloud: dict[str, Any]


def run(
    ctx: NodeContext,
    cameras: list[dict[str, Any]],
    parent_cloud: dict[str, Any],
    parent_mask: Any,
    feature_description: str,
    feature_type: str,
) -> Output:
    candidates = []
    for camera in cameras:
        result = ctx.tool(
            "sam3.segment_text", image=camera["rgb"], query=feature_description, max_results=3
        )
        for mask, score in zip(result.get("masks", []), result.get("scores", []), strict=False):
            if float(score) >= 0.05:
                candidates.append((float(score), camera, mask))
    if not candidates:
        raise ValueError(f"functional feature not found: {feature_description}")
    parent = np.asarray(parent_cloud["points"], dtype=float).reshape(-1, 3)
    if len(parent) < 8:
        raise ValueError("parent object cloud has too few valid depth points")
    lower = np.percentile(parent, 1, axis=0) - 0.02
    upper = np.percentile(parent, 99, axis=0) + 0.02
    selected = None
    for score, camera, mask in sorted(candidates, key=lambda item: item[0], reverse=True):
        cloud = ctx.tool(
            "geometry.mask_to_world_points",
            mask=np.asarray(mask, dtype=np.uint8),
            depth=camera["depth"],
            intrinsics=camera["intrinsics"],
            camera_pose=camera["pose"],
        )["points"]
        points = np.asarray(cloud["points"]).reshape(-1, 3)
        if len(points) < 8:
            continue
        inside = np.all((points >= lower) & (points <= upper), axis=1)
        if float(np.mean(inside)) < 0.6:
            continue
        points = points[inside]
        cloud = {"points": points.astype(np.float32)}
        selected = score, camera, mask, cloud, points
        break
    if selected is None:
        raise ValueError("functional feature candidates were not geometrically part of the parent")
    score, camera, mask, cloud, points = selected
    kind = str(feature_type)
    if kind in {"loop", "aperture", "surface", "region"}:
        cp = camera["pose"]["position"]
        center = np.median(points, axis=0)
        hint = {
            "x": float(cp["x"] - center[0]),
            "y": float(cp["y"] - center[1]),
            "z": float(cp["z"] - center[2]),
        }
        fit = ctx.tool(
            "geometry.fit_planar_feature",
            points=cloud,
            normal_hint=hint,
            fit_circle_center=kind in {"loop", "aperture"},
        )
        feature = {
            "kind": kind,
            "pose": fit["pose"],
            "axis": fit["normal"],
            "confidence": float(score),
            "fit_quality": float(fit["planarity"]),
        }
        if kind == "loop":
            feature["radius_inner"] = float(fit["radius"])
        elif kind == "aperture":
            feature["radius_outer"] = float(fit["radius"])
    elif kind in {"shaft", "tip"}:
        fit = ctx.tool("geometry.fit_linear_feature", points=cloud)
        position = fit["endpoint_max"] if kind == "tip" else fit["center"]
        feature = {
            "kind": kind,
            "pose": {"position": position, "rotation": {"w": 1.0, "x": 0.0, "y": 0.0, "z": 0.0}},
            "axis": fit["axis"],
            "confidence": float(score),
            "fit_quality": float(fit["linearity"]),
            "length": float(fit["length"]),
        }
    else:
        raise ValueError(f"unsupported feature_type {kind!r}")
    return {"feature": feature, "feature_mask": mask, "feature_cloud": cloud}
