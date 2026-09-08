"""Pull the crate over on the mat, closed on what the crate is actually doing.

Each chord reads the crate, takes whichever near bottom corner is on the mat as
the pivot (both, once it stands on its long wall), and rotates each hand a few
degrees about that pivot -- from where the hand actually is, so the pinch offsets
and the arm's sag under load drop out. A small downward bias keeps the crate's
weight on the mat: the pads are a pin near the crate's centre of mass, and a
crate hanging from them keeps its corner unloaded, so a bare pull only drags it.
Pressed onto the mat, the edge holds, and the pull turns the crate about it --
the mat does the rotating. The wrists turn WITH the crate, by what the crate has
actually turned plus one step of lead: measured against a wrist held still, that
doubles the rotation per chord and cuts the edge's sliding from 14 cm to 1.7,
because the pads never have to slip. Measured against a wrist turned a fixed step
per chord, it is what stops the wrist running 20 degrees ahead of a stalled crate
and sliding the wall 20-40 mm out of the pads. The roll is capped where the arm's
envelope ends at the x the pull reaches.

Balance, measured by planting the standing crate and letting go: 106 degrees
falls back, 107 goes over. The pull stops on the MEASURED attitude a little past
that, so the fall starts slowly and the hands have time to leave."""
import math, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gap import NodeContext
import tipcommon as T

STEP = 3.0          # deg of crate rotation asked per chord
BIAS = 0.008        # m the hands are commanded below the rotated point: the press
BIAS_STEP = 0.004   # more press each time four chords make no net progress
BIAS_MAX = 0.024
ON_MAT = 0.003      # m; a corner this close to the mat counts as grounded
MAX_CHORDS = 40
ROLL_LEAD = 3.0     # deg the wrist may run ahead of the measured crate attitude
ROLL_CAP = 80.0


def run(ctx: NodeContext, arms: list, scene: dict, theta_release: float = 108.5, phi: float = -1.0) -> dict:
    z_table = float(scene["z_table"])
    phi0 = float(phi) if phi > 0 else T.PHI
    roll = phi0
    bias = BIAS
    now = T.crate_state(ctx)
    theta_start = now["theta"]
    chords = 0
    history = [now["theta"]]
    stalls = 0
    while now["theta"] < theta_release and chords < MAX_CHORDS:
        fz = now["floor_corner"][2] - z_table
        rz = now["rim_corner"][2] - z_table
        if fz < ON_MAT and rz < ON_MAT:
            pivot = now["rim_corner"]          # standing on its long wall: it will go over the rim edge
        else:
            pivot = now["floor_corner"] if fz <= rz else now["rim_corner"]
        d = math.radians(STEP)
        targets = {}
        for a in arms:
            e = T.ee(ctx, a)
            vx, vz = e[0] - pivot[0], e[2] - pivot[2]
            rx, rz_ = vx * math.cos(d) - vz * math.sin(d), vx * math.sin(d) + vz * math.cos(d)
            targets[a["name"]] = [pivot[0] + rx, e[1], pivot[2] + rz_ - bias]
        roll = min(ROLL_CAP, max(phi0, phi0 + (now["theta"] - theta_start) + ROLL_LEAD))
        T.move_pair(ctx, arms, targets, roll, n=12, settle=3)
        now = T.crate_state(ctx)
        chords += 1
        track = {a["name"]: round(1000 * math.dist(T.ee(ctx, a), targets[a["name"]]), 1) for a in arms}
        T.log("pull_chord", k=chords, pivot=pivot, roll=roll, bias=bias, theta=now["theta"], floor_clear_mm=fz * 1000, rim_clear_mm=rz * 1000, c=now["c"], track_err_mm=track, slip=T.slip(ctx, arms, now))
        history.append(now["theta"])
        # four chords without a degree of net progress: the edge is sliding or the
        # pads are; press harder, and give up only when pressing harder does nothing
        if len(history) >= 5 and history[-1] - history[-5] < 1.0:
            stalls += 1
            bias = min(BIAS_MAX, bias + BIAS_STEP)
            history = [now["theta"]]
            T.log("pull_stall", theta=now["theta"], bias=bias, stalls=stalls)
            if stalls >= 4:
                break
    return {"arms": arms, "theta": now["theta"], "theta_cmd": now["theta"], "phi": roll, "rim_corner": now["rim_corner"], "chords": chords}
