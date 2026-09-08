"""Both hands from home to a hover over their end wall, then straight down onto
it. The hover is 70 mm up at the manoeuvre's one wrist roll (tipcommon.PHI): a
top-down hover at this x is outside the arm's envelope, a rolled one is not, and
the roll costs nothing because the pads stay parallel to the wall whatever the
wrist does about y."""
import math, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gap import NodeContext
import tipcommon as T

HOVER = 0.07
PHI = None  # resolved to tipcommon.PHI at run time


def run(ctx: NodeContext, arms: list, scene: dict) -> dict:
    PHI = T.PHI
    T.grip_both(ctx, arms, preset="open", ramp=10, settle=10)
    st = T.crate_state(ctx)
    pins = {a["name"]: T.pin_world(st, a["sign"]) for a in arms}
    # Transit: pick each wrist's half-turn by which one plans the hover cleanly.
    tracks = {}
    for arm in arms:
        pin = pins[arm["name"]]
        hover = [pin[0], pin[1], pin[2] + HOVER]
        best = None
        for key in ("rotation", "rotation_flipped"):
            arm["rot_key"] = key
            rot, _ = T.frame(ctx, arm, PHI)
            pj = ctx.tool("motion.plan_joint", pose=T.pose_dict(hover, rot), arm_id=arm["arm_id"], orientation="lock", num_waypoints=40)
            perr, rerr = float(pj["position_error_m"]), float(pj["rotation_error_rad"])
            T.log("hover_plan", arm=arm["name"], key=key, err_m=perr, rot_err_deg=math.degrees(rerr))
            if best is None or (perr + 0.05 * rerr) < best[0]:
                best = (perr + 0.05 * rerr, key, pj, perr, rerr)
        arm["rot_key"] = best[1]
        if best[3] > 0.008 or best[4] > math.radians(6):
            raise RuntimeError(f"{arm['name']} cannot reach its hover: {best[3]*1000:.1f} mm, {math.degrees(best[4]):.1f} deg")
        tracks[str(arm["arm_id"])] = [w["positions"] for w in best[2]["trajectory"]["waypoints"]]
    ctx.tool("robot.stream_dual", tracks=tracks, tolerance=0.01, settle_steps=30)
    # Descend onto the wall: straight line, wrist locked, tight gate.
    T.move_pair(ctx, arms, pins, PHI, n=30, settle=20, tol=0.004)
    after = T.crate_state(ctx)
    moved = math.dist(after["c"], st["c"])
    T.log("approach_done", pins=pins, crate_moved_mm=moved * 1000, theta=after["theta"])
    return {"arms": arms, "pins": pins, "crate_moved_mm": moved * 1000}
