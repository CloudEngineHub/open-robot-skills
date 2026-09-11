"""Reach a top-down pre-grasp with a planned joint trajectory and verify it.

``robot.go_to_pose`` may silently retain the previous joint solution when a
seed starts from an awkward elbow configuration. The final pre-grasp segment
is a short, obstacle-free top-down move, so a miss is recovered with the
calibrated Cartesian controller before the grasp is declared unreachable --
and only a miss: an arrival inside tolerance never enters the recovery path.

The verification tolerances come from the selected ``grasp_profile`` (or a
``grasp_profiles`` table keyed by ``target_kind``); without one they are the
defaults below. ``arm_id`` names the hand on a bimanual cell and is threaded
as an ABSENT keyword when unset, so a single-arm graph's recorded calls carry
no ``arm_id``.
"""

from typing import Any, TypedDict

import numpy as np
from gap import NodeContext
from scipy.spatial.transform import Rotation


class Output(TypedDict):
    final_pose: dict[str, Any]
    position_error_m: float
    approach_alignment: float
    closing_alignment: float
    recovery_used: bool


def _rotation(pose: dict[str, Any]) -> Rotation:
    q = pose["rotation"]
    return Rotation.from_quat([q["x"], q["y"], q["z"], q["w"]])


def _tracking_error(actual: dict[str, Any], wanted: dict[str, Any]) -> tuple[float, float, float]:
    """Return position, approach-axis, and closing-axis agreement."""
    position_error = np.linalg.norm(
        np.array([actual["position"][k] for k in ("x", "y", "z")])
        - np.array([wanted["position"][k] for k in ("x", "y", "z")])
    )
    wanted_rotation = _rotation(wanted)
    actual_rotation = _rotation(actual)
    approach_alignment = float(np.dot(
        wanted_rotation.apply([0.0, 0.0, 1.0]),
        actual_rotation.apply([0.0, 0.0, 1.0]),
    ))
    # A parallel-jaw grasp is invariant to reversing the closing axis.
    closing_alignment = abs(float(np.dot(
        wanted_rotation.apply([1.0, 0.0, 0.0]),
        actual_rotation.apply([1.0, 0.0, 0.0]),
    )))
    return float(position_error), approach_alignment, closing_alignment


def _profile(
    grasp_profile: dict[str, Any] | None, target_kind: str, grasp_profiles: list[dict[str, Any]]
) -> dict[str, Any]:
    """Resolve the selected profile; keyed lists support existing graphs."""
    if grasp_profile is not None:
        return dict(grasp_profile)
    if not target_kind and not grasp_profiles:
        return {}
    profiles = {str(item["source_kind"]): item for item in grasp_profiles}
    if target_kind not in profiles:
        raise ValueError(f"no grasp verification profile declared for profile key {target_kind!r}")
    return dict(profiles[target_kind])


def run(
    ctx: NodeContext,
    pose: dict[str, Any],
    arm_id: int | None = None,
    grasp_profile: dict[str, Any] | None = None,
    target_kind: str = "",
    grasp_profiles: list[dict[str, Any]] | None = None,
) -> Output:
    profile = _profile(grasp_profile, target_kind, grasp_profiles or [])
    on_arm: dict[str, Any] = {} if arm_id is None else {"arm_id": int(arm_id)}

    def where() -> tuple[dict[str, Any], float, float, float]:
        actual = ctx.tool("robot.get_ee_pose", **on_arm)["pose"]
        return (actual, *_tracking_error(actual, pose))

    # The simulator connector has calibrated IK/control for both arms. The
    # optional cuRobo backend exposes trajectories only for its primary chain,
    # so using it here would reject every second-arm pre-grasp even when the
    # same-side robot IK can reach it exactly.
    ctx.tool("robot.go_to_pose", pose=pose, **on_arm)
    actual, position_error, approach_alignment, closing_alignment = where()
    tolerance_deg = float(profile.get("angular_tolerance_deg", 12.0))
    angular_tolerance = float(np.cos(np.deg2rad(tolerance_deg)))
    position_tolerance = float(profile.get("position_tolerance_m", 0.015))

    def failed() -> bool:
        return (
            position_error > position_tolerance
            or approach_alignment < angular_tolerance
            or closing_alignment < angular_tolerance
        )

    recovery_used = False
    # Recovery is confined to a pose the plain path would reject: a small
    # residual translation or wrist yaw gets one calibrated Cartesian
    # correction before failure is declared; arrivals stay on their old path.
    recovery_threshold = float(profile.get("cartesian_recovery_threshold_m", 0.030))
    recover = bool(profile.get("cartesian_recovery_on_verification_failure", True))
    if position_error > recovery_threshold and recover:
        # A combined translation + wrist rotation can be locally infeasible
        # even though both components are reachable, especially near the
        # symmetry plane of a dual-arm workspace: translate first while
        # keeping the controller's current wrist solution, then ask for the
        # grasp frame.
        ctx.tool(
            "robot.go_to_pose_cartesian",
            pose={"position": dict(pose["position"]), "rotation": dict(actual["rotation"])},
            **on_arm,
        )
        actual, position_error, approach_alignment, closing_alignment = where()
        recovery_used = True
    if failed() and recover:
        ctx.tool("robot.go_to_pose_cartesian", pose=pose, **on_arm)
        actual, position_error, approach_alignment, closing_alignment = where()
        recovery_used = True
    # Tracking quantisation and the mirrored wrist solution can contribute a
    # few millimetres/degrees without changing the grasp line; twelve degrees
    # stays conservative for a top-down parallel-jaw grasp. The selected grasp
    # region declares how much yaw residual it tolerates, which also admits a
    # mirrored arm's equivalent solution without any object-class assumption.
    if failed():
        raise RuntimeError(
            "pre-grasp tracking error too large: "
            f"{position_error * 1000:.1f} mm, approach_dot={approach_alignment:.3f}, "
            f"closing_dot={closing_alignment:.3f}"
        )
    return {
        "final_pose": actual,
        "position_error_m": float(position_error),
        "approach_alignment": approach_alignment,
        "closing_alignment": closing_alignment,
        "recovery_used": recovery_used,
    }
