"""Take the cable's far end with the support hand, before any crossing.

**The one structural difference between a graph that weaves and one that does
not.** The hand-written routine that solves this task opens with this move; a
graph without it weaves single-handed.

Why it matters is a property of the material, not of the plan. A cable held at
one end and pulled sideways *bends*; a free cable **translates**. Without an
anchor the first crossing hauls the whole cable toward one station and the next
has nothing left to grasp -- the routine records 26 of 42 segments piled at a
single station. Every later crossing then works on a cable that has already
moved out from under the model that was measured.

Two legs and a close: enter, take, pinch. The pinch is a contact grasp at the
same half-millimetre engagement the work hand uses -- no latch, because the hold
the score sees has to be the hold the fingers earn.

Failure is fatal here in a way a missed look was not. A weave without an anchor
is not a degraded weave, it is a different and much worse task, so this routes
``unanchored`` to abort rather than carrying on and reporting the result as a
policy failure.

The grasp itself comes from ``cable.plan_support``: a privileged support-hand
planner that reads the true rod state and returns joint-space legs, the hand
that owns them (``arm_id``), the width it should close to and the IK error of
its worst waypoint. This script gates on that error, streams the legs, closes
and dwells.
"""

import json
from typing import Any, TypedDict

from gap import NodeContext

#: Refuse a grasp the solver could not reach. The routine's own limit: past this
#: the chain is in a different basin and what streams is a whip with the cable
#: near the jaws.
PLAN_ERROR_LIMIT = 0.05

#: Control steps to dwell after the close, for the jaws' own travel.
SETTLE_STEPS = 24


class Output(TypedDict):
    route: str
    segment: int
    worst_error_m: float
    q_support: str
    quat: str


def run(ctx: NodeContext, approach: float = 0.055) -> Output:
    """Plan, stream and close the support hand's anchor grasp.

    ``approach`` is the height above the grasp the enter leg starts from [m].
    """
    plan: dict[str, Any] = ctx.tool("cable.plan_support", approach=float(approach))
    worst = float(plan.get("worst_grasp_error", 0.0))
    stages = plan.get("stages") or []
    arm_id = int(plan.get("arm_id", 1))
    arm_name = str(plan.get("arm_name", ""))
    print(
        f"[anchor] arm={arm_name} segment={plan.get('segment')} "
        f"worst_ik={worst * 1000:.1f}mm legs={[s.get('stage') for s in stages]}",
        flush=True,
    )
    if not stages or worst > PLAN_ERROR_LIMIT:
        print(f"[anchor] REFUSED ({worst * 1000:.0f}mm)", flush=True)
        return {
            "route": "unanchored",
            "segment": -1,
            "worst_error_m": worst,
            "q_support": "null",
            "quat": "null",
        }

    for stage in stages:
        ctx.tool(
            "robot.execute_trajectory",
            trajectory=stage["trajectory"],
            arm_id=arm_id,
            max_steps_per_waypoint=15,
        )
    # Close on the far end, then dwell for the jaws' own travel: a finger
    # command is a target, not a state, and the first crossing must not start
    # while the pads are still moving. The close width and the hand are the
    # plan's own, so the script never states a number that depends on which
    # gripper arrived.
    ctx.tool("robot.set_grip", width_m=float(plan.get("close", 0.0045)), arm_id=arm_id)
    ctx.tool("robot.wait_steps", steps=SETTLE_STEPS)
    print(f"[anchor] held segment {plan.get('segment')}", flush=True)
    # The solved chain and the grasp orientation travel with the exit: every
    # later pay-out seeds its IK from them, and re-deriving them per crossing
    # would let the support chain drift into a different basin mid-weave.
    return {
        "route": "anchored",
        "segment": int(plan.get("segment", -1)),
        "worst_error_m": worst,
        "q_support": json.dumps([float(v) for v in plan.get("q_end", [])]),
        "quat": json.dumps([float(v) for v in plan.get("quat", [])]),
    }
