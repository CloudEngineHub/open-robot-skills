"""Camera-only evidence that a released object passed below an aperture.

After insertion and release, the object is looked for in the named camera.
An opaque container hides a contained object, so not seeing it at all counts
as success; seeing it below the rim within the aperture's footprint counts
too; seeing it above the rim or away from the aperture routes ``not_placed``.
"""

from typing import Any, TypedDict

import numpy as np
from gap import NodeContext


class Output(TypedDict):
    route: str
    verified: bool
    evidence: str


def _camera(observation: dict[str, Any], name: str) -> dict[str, Any]:
    cameras = observation.get("cameras") or []
    if isinstance(cameras, dict):
        cameras = list(cameras.values())
    for camera in cameras:
        if camera.get("name") == name:
            return camera
    raise ValueError(f"observation has no camera named {name!r}")


def run(
    ctx: NodeContext,
    observation: dict[str, Any],
    object_description: str,
    aperture_center: dict[str, float],
    rim_z: float,
    score_min: float = 0.03,
    xy_tolerance_m: float = 0.10,
    depth_margin_m: float = 0.005,
    min_points: int = 8,
    camera_name: str = "overhead",
) -> Output:
    camera = _camera(observation, camera_name)
    result = ctx.tool(
        "sam3.segment_text", image=camera["rgb"], query=object_description, max_results=3
    )
    if not result.get("masks") or float((result.get("scores") or [0.0])[0]) < float(score_min):
        # An opaque container hides a contained object. This evidence is
        # meaningful only after insertion and release completed.
        return {
            "route": "verified",
            "verified": True,
            "evidence": f"{object_description} disappeared after aperture release",
        }
    cloud = ctx.tool(
        "geometry.mask_to_world_points",
        mask=np.asarray(result["masks"][0], dtype=np.uint8),
        depth=camera["depth"],
        intrinsics=camera["intrinsics"],
        camera_pose=camera["pose"],
    )["points"]
    points = np.asarray(cloud["points"], dtype=np.float64).reshape(-1, 3)
    if len(points) < int(min_points):
        return {
            "route": "not_placed",
            "verified": False,
            "evidence": (
                f"visible {object_description} has too few valid depth points "
                f"({len(points)} < {int(min_points)}) to confirm placement"
            ),
        }
    center = np.median(points, axis=0)
    aperture_xy = np.array([aperture_center["x"], aperture_center["y"]], dtype=np.float64)
    offset = float(np.linalg.norm(center[:2] - aperture_xy))
    if float(center[2]) >= float(rim_z) - float(depth_margin_m) or offset > float(xy_tolerance_m):
        return {
            "route": "not_placed",
            "verified": False,
            "evidence": (
                f"{object_description} remains visible outside or above the aperture "
                f"(centre z {center[2]:.3f} vs rim {float(rim_z):.3f}, offset {offset * 1000.0:.0f} mm)"
            ),
        }
    return {
        "route": "verified",
        "verified": True,
        "evidence": f"visible {object_description} centre is below the aperture",
    }
