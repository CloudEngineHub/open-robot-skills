"""Lift the pinch points straight up until the crate hangs. The crate is held
120 mm off its own centre, so it hangs heavy on the near side: its near bottom
corner stays on the mat, sliding away from the robot as the crate stands up
around the pins -- the mat and gravity do the rotating, the arms only lift. The
wrist does NOT turn here: the crate turns inside the pads (that is the pin joint)
and a wrist turning against a crate resting on the mat is what slides the wall out.
The lift ends when the corner leaves the mat: from there the crate hangs at the
attitude its own centre of mass sets (67 degrees with the pinch 60 mm off
centre, 78 at 120), and no amount of lifting changes that -- the pull does."""
import math, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gap import NodeContext
import tipcommon as T

H_END = 0.325        # pinch height above the mat to command [m]: the hang height plus the arm's ~13 mm sag under load
CHORDS = 10
CORNER_CLEAR = 0.004


def run(ctx: NodeContext, arms: list, scene: dict) -> dict:
    st = T.crate_state(ctx)
    pins = {a["name"]: T.pin_world(st, a["sign"]) for a in arms}
    z_table = float(scene["z_table"])
    h0 = pins[arms[0]["name"]][2] - z_table
    hanging = 0
    for k in range(1, CHORDS + 1):
        h = h0 + (H_END - h0) * k / CHORDS
        targets = {n: [p[0], p[1], z_table + h] for n, p in pins.items()}
        T.move_pair(ctx, arms, targets, T.PHI, n=14, settle=3)
        now = T.crate_state(ctx)
        clear = min(now["floor_corner"][2], now["rim_corner"][2]) - z_table
        T.log("lift_chord", k=k, h=h, theta_geom=T.theta_on_corner(h), theta=now["theta"], corner=now["floor_corner"], corner_clear_mm=clear * 1000, c=now["c"], slip=T.slip(ctx, arms, now))
        hanging = hanging + 1 if clear > CORNER_CLEAR else 0
        if hanging >= 2:
            break
    ctx.tool("robot.wait_steps", steps=30)
    final = T.crate_state(ctx)
    return {"arms": arms, "theta": final["theta"], "hanging": bool(hanging >= 2), "c": final["c"]}
