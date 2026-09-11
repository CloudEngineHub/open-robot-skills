"""Language-localize a functional part and fit its metric geometry.

The default ``method`` (``semantic_fit``) is the original body: segment the
feature by description in every camera, keep the candidates that lie inside
the parent's cloud, and fit a planar or linear feature to the best one. A
graph that binds only the original inputs gets exactly those calls.

Three parent-relative strategies exist for a feature that has no depth of its
own -- an empty aperture, an occluded opening, a tip that only reads as the far
end of its parent. Each is a DECLARATION owned by the calling graph
(``method`` + ``feature_options``), never an object-name branch here:

- ``parent_landmark``: the parent's median plus a declared ``offset``, with a
  declared ``axis``.
- ``parent_distal_endpoint``: the parent points farthest from a reference (the
  median, or ``reference_position``), above ``distal_percentile``.
- ``parent_inferred_aperture``: the parent's rim along ``rim_axis`` between
  ``rim_percentiles``, at the parent's median in the other two axes.

Every feature carries ``method`` and ``uncertainty_m`` so a consumer can weigh
a declared landmark against a measured fit.
"""

from typing import Any, TypedDict

import numpy as np
from gap import NodeContext


class Output(TypedDict):
    feature: dict[str, Any]
    feature_mask: np.ndarray
    feature_cloud: dict[str, Any]


def _point_pose(position: np.ndarray) -> dict[str, Any]:
    return {
        "position": dict(zip(("x", "y", "z"), map(float, position), strict=True)),
        "rotation": {"w": 1.0, "x": 0.0, "y": 0.0, "z": 0.0},
    }


def _axis(values: Any) -> dict[str, float]:
    vector = np.asarray(values, dtype=float).reshape(3)
    vector /= max(float(np.linalg.norm(vector)), 1.0e-12)
    return dict(zip(("x", "y", "z"), map(float, vector), strict=True))


def _parent_relative(
    method: str,
    feature_type: str,
    parent: np.ndarray,
    parent_mask: Any,
    parent_cloud: dict[str, Any],
    options: dict[str, Any],
) -> Output:
    """The three declared strategies; ``method`` has already been checked."""
    kind = str(feature_type)
    confidence = float(options.get("confidence", 0.75))
    if method == "parent_landmark":
        position = np.median(parent, axis=0) + np.asarray(options.get("offset", [0.0, 0.0, 0.0]), dtype=float)
        feature = {
            "kind": kind, "pose": _point_pose(position),
            "axis": _axis(options.get("axis", [0.0, 0.0, 1.0])),
            "confidence": confidence, "method": method,
            "uncertainty_m": float(options.get("uncertainty_m", 0.01)),
        }
        return {"feature": feature, "feature_mask": parent_mask, "feature_cloud": parent_cloud}
    if method == "parent_distal_endpoint":
        reference = np.median(parent, axis=0)
        if "reference_position" in options:
            reference = np.asarray(options["reference_position"], dtype=float).reshape(3)
        distances = np.linalg.norm(parent - reference, axis=1)
        distal = parent[distances >= np.percentile(distances, float(options.get("distal_percentile", 97.0)))]
        if len(distal) < int(options.get("minimum_feature_points", 4)):
            raise ValueError("parent cloud does not expose a stable distal endpoint")
        position = np.median(distal, axis=0)
        feature = {
            "kind": kind, "pose": _point_pose(position), "axis": _axis(position - reference),
            "confidence": confidence, "method": method,
            "uncertainty_m": float(np.median(np.linalg.norm(distal - position, axis=1))),
        }
        return {"feature": feature, "feature_mask": parent_mask, "feature_cloud": {"points": distal.astype(np.float32)}}
    # parent_inferred_aperture
    axis_index = {"x": 0, "y": 1, "z": 2}[str(options.get("rim_axis", "z"))]
    lo, hi = options.get("rim_percentiles", [85.0, 90.0])
    lower, upper = np.percentile(parent[:, axis_index], [float(lo), float(hi)])
    rim = parent[(parent[:, axis_index] >= lower) & (parent[:, axis_index] <= upper)]
    if len(rim) < int(options.get("minimum_feature_points", 20)):
        raise ValueError("parent cloud does not expose a stable rim")
    position = np.median(parent, axis=0)
    position[axis_index] = np.median(rim[:, axis_index])
    feature = {
        "kind": kind, "pose": _point_pose(position),
        "axis": _axis(options.get("axis", [0.0, 0.0, -1.0])),
        "confidence": confidence, "method": method,
        "uncertainty_m": float(np.std(rim[:, axis_index])),
    }
    if "radius_inner" in options:
        feature["radius_inner"] = float(options["radius_inner"])
    return {"feature": feature, "feature_mask": parent_mask, "feature_cloud": {"points": rim.astype(np.float32)}}


def run(
    ctx: NodeContext,
    cameras: list[dict[str, Any]],
    parent_cloud: dict[str, Any],
    parent_mask: Any,
    feature_description: str,
    feature_type: str,
    method: str = "semantic_fit",
    feature_options: dict[str, Any] | None = None,
) -> Output:
    options = feature_options or {}
    if method not in {"semantic_fit", "parent_landmark", "parent_distal_endpoint", "parent_inferred_aperture"}:
        raise ValueError(f"unsupported functional-feature method {method!r}")
    parent = np.asarray(parent_cloud["points"], dtype=float).reshape(-1, 3)
    if method != "semantic_fit":
        # Checked before any segmentation: a declared strategy makes no camera
        # call, and the parent count it needs is its own.
        if len(parent) < int(options.get("minimum_parent_points", 8)):
            raise ValueError("parent object cloud has too few valid depth points")
        return _parent_relative(method, feature_type, parent, parent_mask, parent_cloud, options)

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
    if len(parent) < int(options.get("minimum_parent_points", 8)):
        raise ValueError("parent object cloud has too few valid depth points")
    margin = float(options.get("parent_margin_m", 0.02))
    lower = np.percentile(parent, 1, axis=0) - margin
    upper = np.percentile(parent, 99, axis=0) + margin
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
        if float(np.mean(inside)) < float(options.get("minimum_parent_overlap", 0.6)):
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
            "method": method,
            "uncertainty_m": float(options.get("uncertainty_m", 0.0)),
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
            "method": method,
            "uncertainty_m": float(options.get("uncertainty_m", 0.0)),
        }
    else:
        raise ValueError(f"unsupported feature_type {kind!r}")
    return {"feature": feature, "feature_mask": mask, "feature_cloud": cloud}
