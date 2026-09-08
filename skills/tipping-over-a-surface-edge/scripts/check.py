"""Where did the crate come to rest? Inverted is done; standing means the pull
stopped short of balance and it rocked back, which the graph retries from a
re-grasp; anything else is a fall the graph did not plan."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gap import NodeContext
import tipcommon as T


def run(ctx: NodeContext, arms: list) -> dict:
    ctx.tool("robot.wait_steps", steps=60)
    st = T.crate_state(ctx)
    th = st["theta"]
    if th >= 160.0 or th <= -160.0:
        route = "inverted"
    elif 60.0 <= th <= 120.0:
        route = "standing"
    else:
        route = "other"
    T.log("check", theta=th, route=route, c=st["c"])
    return {"arms": arms, "theta": th, "route": route}
