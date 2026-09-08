"""Let go and get out. The inner pad is inside the crate and the only way out is
through the mouth, which is also where the crate is about to fall; so the jaws
crack open, both hands back out along the crate's own mouth direction, and then
sweep outboard past the crate's ends, up, and toward the robot, ahead of the fall."""
import math, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gap import NodeContext
import tipcommon as T

EXIT = 0.06
ESCAPE = (-0.05, 0.10, 0.06)


def run(ctx: NodeContext, arms: list, theta_cmd: float, phi: float = -1.0) -> dict:
    st = T.crate_state(ctx)
    phi = float(phi) if phi > 0 else T.PHI
    th = math.radians(max(st["theta"], theta_cmd))
    m = {"x": -math.sin(th), "z": math.cos(th)}   # the crate's mouth direction: the only way out
    pins = {}
    for arm in arms:
        p = ctx.tool("robot.get_ee_pose", arm_id=arm["arm_id"])["pose"]["position"]
        pins[arm["name"]] = [p["x"], p["y"], p["z"]]
    T.grip_both(ctx, arms, width_m=0.03, ramp=3, settle=0)
    exit_t = {n: [p[0] + m["x"] * EXIT, p[1], p[2] + m["z"] * EXIT] for n, p in pins.items()}
    T.move_pair(ctx, arms, exit_t, phi, n=8, settle=0)
    T.grip_both(ctx, arms, preset="open", ramp=4, settle=0)
    esc = {}
    for arm in arms:
        e = exit_t[arm["name"]]
        esc[arm["name"]] = [e[0] + ESCAPE[0], e[1] + arm["sign"] * ESCAPE[1], e[2] + ESCAPE[2]]
    try:
        T.move_pair(ctx, arms, esc, phi, n=12, settle=0)
    except RuntimeError as exc:
        T.log("escape_locked_failed", error=str(exc))
        tracks = {}
        for arm in arms:
            rot, _ = T.frame(ctx, arm, phi)
            pj = ctx.tool("motion.plan_joint", pose=T.pose_dict(esc[arm["name"]], rot), arm_id=arm["arm_id"], orientation="axis", num_waypoints=12)
            tracks[str(arm["arm_id"])] = [w["positions"] for w in pj["trajectory"]["waypoints"]]
        ctx.tool("robot.stream_dual", tracks=tracks, tolerance=0.02, settle_steps=0)
    ctx.tool("robot.wait_steps", steps=90)
    after = T.crate_state(ctx)
    T.log("release", theta_before=st["theta"], theta_after=after["theta"], c_after=after["c"])
    return {"arms": arms, "theta": after["theta"]}
