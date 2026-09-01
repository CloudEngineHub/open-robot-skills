"""Move a held object so one of its declared features lands on a target point.

The composition that a place needs and that nothing generic provided. A goal
says "``wrench_1.ring`` encloses ``wrench_hook_1.shank``", and the natural
reading — drive the hand to where the hook is — is wrong by the length of the
tool. ``robot.solve_ik`` refuses that pose, correctly, on every arm and every
hook of a task that is demonstrably solvable; the hand has to sit a whole
tool-length away so that the *ring* arrives, not the wrist.

So the hand pose is composed rather than guessed:

    feature_now = object_position + R(object_rotation) · feature_local
    move        = target - feature_now
    hand_goal   = hand_now + move

Three properties worth stating, because each was learned the expensive way:

* **Recomputed from live poses, every round.** The object shifts in the jaws
  during a carry; a move planned from where the feature was at pick time
  arrives off by the slip. Reading the object's pose fresh means one round's
  residual cannot compound into the next.
* **``feature_local`` is an asset constant, not a per-episode number.** It is
  the feature's centre in the object's own frame, which the packet's declared
  parts state (``wrench_1.ring — a loop at [0.17, 0, 0]``). It is the same on
  every episode of every layout, so a policy that reads it once is not
  fitting to a seed.
* **Translation only.** This puts the feature *at* the point; it does not turn
  the object so a ring's hole lines up with a rod. Orientation is the caller's,
  because what "aligned" means is the goal's own predicate — a loop over a rod
  needs an axis, a part into a cavity needs a face — and inventing one here
  would be answering a question this function cannot see.
"""

from typing import Any, TypedDict

from gap import NodeContext

__all__ = ["run"]


class Output(TypedDict):
    route: str
    hand_goal: dict
    move_m: dict
    feature_now: dict
    distance_m: float
    reason: str


def _rotate(quat: dict[str, float], v: tuple[float, float, float]) -> tuple[float, float, float]:
    """*v* by the wxyz quaternion — the object's own frame into the world's."""
    w, x, y, z = quat["w"], quat["x"], quat["y"], quat["z"]
    tx, ty, tz = 2.0 * (y * v[2] - z * v[1]), 2.0 * (z * v[0] - x * v[2]), 2.0 * (x * v[1] - y * v[0])
    return (
        v[0] + w * tx + (y * tz - z * ty),
        v[1] + w * ty + (z * tx - x * tz),
        v[2] + w * tz + (x * ty - y * tx),
    )


def _xyz(value: Any) -> tuple[float, float, float] | None:
    if isinstance(value, dict):
        if all(k in value for k in "xyz"):
            return (float(value["x"]), float(value["y"]), float(value["z"]))
        for key in ("position", "at", "point", "center"):
            if key in value:
                return _xyz(value[key])
        return None
    if isinstance(value, (list, tuple)) and len(value) >= 3:
        return (float(value[0]), float(value[1]), float(value[2]))
    return None


def run(
    ctx: NodeContext,
    target: Any,
    object_name: str,
    feature_local: Any,
    arm_id: int = 0,
    stop_short_axis: str = "",
    stop_short_m: float = 0.0,
) -> Output:
    """The hand pose that puts *object_name*'s feature on *target*.

    Args:
        target: Where the feature must end up — ``sim.query``'s ``at`` for a
            welded fixture, a region's centre, or a bare ``[x, y, z]``.
        object_name: The held object, whose live pose is read fresh.
        feature_local: The feature's centre in that object's own frame, from
            the packet's declared parts.
        arm_id: The hand holding it.
        stop_short_axis: ``"x"``/``"y"``/``"z"`` to stop short of the target on
            one world axis — for approaching a fixture mounted flush to a
            board, where closing the other two axes fully is safe and the third
            is the one that would drive the load into it.
        stop_short_m: How far short, on that axis.

    Returns:
        ``route`` is ``"ready"`` or ``"unknown"`` when a pose could not be read.
        ``hand_goal`` is a full pose (the hand's current rotation, unchanged);
        ``move_m`` is the translation it applies, and ``distance_m`` its length,
        which is the number to watch shrink across rounds.
    """
    goal_point = _xyz(target)
    local = _xyz(feature_local)
    if goal_point is None or local is None:
        return {
            "route": "unknown", "hand_goal": {}, "move_m": {}, "feature_now": {},
            "distance_m": -1.0,
            "reason": "target or feature_local is not a point; pass sim.query's `at` and the "
                      "feature centre the declared parts state",
        }
    try:
        obj = ctx.tool("sim.get_object_pose", object_name=object_name) or {}
        hand = ctx.tool("robot.get_ee_pose", arm_id=arm_id) or {}
    except Exception as exc:
        return {
            "route": "unknown", "hand_goal": {}, "move_m": {}, "feature_now": {},
            "distance_m": -1.0, "reason": f"could not read a live pose: {exc}",
        }
    obj_pose = obj.get("pose", obj)
    hand_pose = hand.get("pose", hand)
    obj_p, hand_p = _xyz(obj_pose), _xyz(hand_pose)
    if obj_p is None or hand_p is None:
        return {
            "route": "unknown", "hand_goal": {}, "move_m": {}, "feature_now": {},
            "distance_m": -1.0, "reason": "a pose read back without a position",
        }
    spun = _rotate(obj_pose.get("rotation", {"w": 1.0, "x": 0.0, "y": 0.0, "z": 0.0}), local)
    feature_now = (obj_p[0] + spun[0], obj_p[1] + spun[1], obj_p[2] + spun[2])
    move = [goal_point[i] - feature_now[i] for i in range(3)]
    if stop_short_axis in ("x", "y", "z") and stop_short_m:
        axis = "xyz".index(stop_short_axis)
        # Shrink the move on that axis by the margin, toward where it came from.
        move[axis] -= stop_short_m if move[axis] > 0 else -stop_short_m
    distance = sum(component * component for component in move) ** 0.5
    return {
        "route": "ready",
        "hand_goal": {
            "position": {
                "x": hand_p[0] + move[0], "y": hand_p[1] + move[1], "z": hand_p[2] + move[2],
            },
            "rotation": hand_pose.get("rotation", {"w": 1.0, "x": 0.0, "y": 0.0, "z": 0.0}),
        },
        "move_m": {"x": move[0], "y": move[1], "z": move[2]},
        "feature_now": {"x": feature_now[0], "y": feature_now[1], "z": feature_now[2]},
        "distance_m": distance,
        "reason": "",
    }
