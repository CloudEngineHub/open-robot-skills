"""Close on the wall -- LOOSELY. The pads are a pin joint here, not a clamp: they
must carry a few newtons of pull along the wall and are allowed to let the wall
turn between them. A ramped close to 1.5 mm past the wall's width does that; the
routine's 2 mm clamp is what made the crate a plate in a friction clutch."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gap import NodeContext
import tipcommon as T


def run(ctx: NodeContext, arms: list, squeeze_m: float = 0.0015) -> dict:
    before = T.crate_state(ctx)
    res = T.grip_both(ctx, arms, object_width_m=T.WALL_T, squeeze_m=float(squeeze_m), ramp=40, settle=30)
    after = T.crate_state(ctx)
    gaps = {a["name"]: ctx.tool("robot.get_gripper", arm_id=a["arm_id"]) for a in arms}
    T.log("pinch", res=res, gaps=gaps, crate_moved_mm=1000 * sum((after["c"][i] - before["c"][i]) ** 2 for i in range(3)) ** 0.5)
    return {"arms": arms, "grip": res, "gaps": gaps}
