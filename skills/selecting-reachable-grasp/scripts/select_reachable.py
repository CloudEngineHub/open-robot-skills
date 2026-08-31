"""Walk a ranked candidate list and take the first the arm can actually reach.

The failure this exists for, in its own words from a study packet: *"gripper
closed on nothing while grasping ring_spanner_19"*. A proposer returns a
ranked list and the obvious thing to do is take entry 0 — but a proposer ranks
by the object's geometry, which is the only thing it knows, and entry 0 is
routinely the one the arm cannot hold at that orientation. Nothing downstream
notices: IK returns *something*, the servo drives to it, the jaws close on air,
and the episode is spent.

So the list is walked. Two gates, and they are different questions:

* **reachable** — does IK solve this pose, and does the solution land near
  where it was asked to? A solver that returns its best effort is not the same
  as a solver that succeeded, and a pose missed by 15 cm is not a grasp.
* **clear** — is the arm, at that configuration, clear of the scene? This
  needs ``sim.clearance``, which the ``full`` tier has and the others do not,
  so its absence is a skipped gate rather than a refusal: a caller at
  ``camera_only`` still gets reachability, and is told the corridor was not
  checked rather than being told it was.

Taking the first that passes, not the best: the list is already ranked by the
proposer, and re-ranking it here would silently overrule a strategy that knows
things about the object this script does not.
"""

from typing import Any, TypedDict

from gap import NodeContext
from gap_core.types import Se3Pose

_POSITION_TOLERANCE_M = 0.02
"""How near the IK solution must land to count as having solved the pose.

Twenty millimetres: wide enough not to reject a good solve on solver noise,
tight enough that a pose the arm cannot actually reach does not pass. A
backend that reports no error at all is trusted, because refusing every
candidate on a missing field would turn a missing diagnostic into a failure."""

_CLEARANCE_FLOOR_M = 0.0
"""Contact-or-worse. The gap is a conservative floor -- every shape is carried
as a capsule containing it -- so demanding daylight would reject configurations
that are merely close, and this is a reachability gate rather than a safety
margin."""


class Output(TypedDict):
    route: str
    pose: Se3Pose | None
    joint_config: Any
    index: int
    checked: int
    corridor_checked: bool
    reason: str


def _solution(result: Any) -> Any:
    """The joint vector out of whatever shape the IK backend returned."""
    if isinstance(result, dict):
        for key in ("joint_config", "joints", "q", "solution"):
            if result.get(key) is not None:
                return result[key]
        return None
    return result


def _missed_by(result: Any) -> float | None:
    if isinstance(result, dict):
        for key in ("position_error_m", "position_error", "error_m"):
            if isinstance(result.get(key), (int, float)):
                return float(result[key])
    return None


def run(
    ctx: NodeContext,
    poses: list[Se3Pose],
    arm_id: int = 0,
    object_name: str = "",
    check_clearance: bool = True,
) -> Output:
    """The first candidate in *poses* that IK solves and (where checkable) is clear.

    Args:
        poses: Candidates, best first, from a propose skill.
        arm_id: Which arm must reach them.
        object_name: The object being grasped, excluded from the clearance test
            — the whole point of a grasp is to approach it, so counting it as
            an obstacle would refuse every candidate.
        check_clearance: Run the corridor gate when ``sim.clearance`` answers.

    Returns:
        ``route`` is ``"selected"`` or ``"none_reachable"``. ``corridor_checked``
        says whether the second gate actually ran, so a caller can tell "clear"
        from "not looked at".
    """
    corridor_checked = False
    tried = 0
    for index, pose in enumerate(poses or []):
        tried += 1
        try:
            solved = ctx.tool("robot.solve_ik", pose=pose, arm_id=arm_id)
        except Exception:
            continue
        joints = _solution(solved)
        if joints is None:
            continue
        missed = _missed_by(solved)
        if missed is not None and missed > _POSITION_TOLERANCE_M:
            continue
        if check_clearance:
            try:
                reading = ctx.tool(
                    "sim.clearance",
                    group_a="arm:work" if arm_id == 0 else "arm:support",
                    group_b=object_name or "scene",
                    joint_q=list(joints),
                )
                corridor_checked = True
                gap = reading.get("gap") if isinstance(reading, dict) else None
                if isinstance(gap, (int, float)) and gap < _CLEARANCE_FLOOR_M:
                    continue
            except Exception:
                # Withheld at this tier, or the group names do not resolve here.
                # A gate that cannot run is skipped and said so, never failed.
                pass
        return {
            "route": "selected",
            "pose": pose,
            "joint_config": joints,
            "index": index,
            "checked": tried,
            "corridor_checked": corridor_checked,
            "reason": "",
        }

    return {
        "route": "none_reachable",
        "pose": None,
        "joint_config": None,
        "index": -1,
        "checked": tried,
        "corridor_checked": corridor_checked,
        "reason": (
            f"none of {tried} proposed grasps solved IK within "
            f"{1e3 * _POSITION_TOLERANCE_M:.0f} mm"
            + (" and cleared the scene" if corridor_checked else "")
            + ". The proposer's strategy does not fit this object from this arm — "
            "try another propose rung (a side grasp, or a narrower part) rather "
            "than re-tuning this one."
        ),
    }
