"""Shared geometry for the tip-by-table crate flip.

The crate is 370 x 525 x 75 mm, long axis along world y, one 12.6 mm end wall
facing each arm. Everything here is stated in the crate's own frame and turned
into world targets from the crate's LIVE pose, read through sim.get_object_pose,
so a stage never aims at where the crate used to be.

Attitude convention: theta is the rotation about world -y that carries the
crate's +x (its far long edge) upward and its +z (open mouth) toward the robot.
theta = 0 flat and mouth-up, 90 standing on its near long wall with the mouth
facing the robot, 180 inverted.
"""
import json, math, os, time

HX, HZ = 0.185, 0.0375          # crate half-extents in x and z [m]
WALL_Y = 0.5 * (0.2499 + 0.26249999)   # end-wall mid-plane in the crate frame [m]
WALL_T = 0.26249999 - 0.2499           # end-wall thickness [m]
D = 0.12                        # pinch offset toward the far edge, in the crate frame [m]
"""Set by two things measured on this cell. A crate hanging from the pins settles
where its centre of mass is under them, atan(D / 0.0255) -- 67 degrees at 60 mm,
78 at 120 -- and every degree past that has to come from a pull that rotates the
crate about its grounded corner, which drags the pinch 5 mm toward the robot per
degree. At 60 mm the standing crate's rim edge ends at x = 0.62 and the flip lands
off the bench; at 120 mm it ends at 0.71 and lands with 40 mm to spare. Larger
still and the top of the lift (z = 1.07 at roll 50) leaves the arm's envelope."""
PHI = 50.0                      # wrist roll about world y at the grasp and through the lift [deg]
"""Held through the lift, and then made to FOLLOW the crate during the pull.
Measured: lifting 50 mm with the roll still, the wall slid under 1 mm in the pads
at every squeeze from 1.5 to 6 mm; a 30 degree roll applied in place slid it
45-50 mm and dropped the crate. So the wrist never turns against a crate that
cannot follow. During the pull the crate turns about a grounded edge, and a wrist
that turns WITH it (never more than a step ahead of the measured attitude) doubles
the rotation per chord and cuts the edge's sliding by ten -- the pads do not have
to slip. 50 degrees is inside the envelope at the top of the lift (z = 1.07) with
the pinch 120 mm off centre; 35 is not."""
R_PIN = HX + D                  # pinch point to the near long edge, along crate x [m]
RHO = math.hypot(R_PIN, HZ)     # pinch point to a near bottom corner [m]
BETA = math.degrees(math.atan2(HZ, R_PIN))

LOG_PATH = os.environ.get("CRATE_TIP_LOG", "")


def log(event, **kw):
    if not LOG_PATH:
        return
    rec = {"t": round(time.time(), 3), "event": event, **kw}
    with open(LOG_PATH, "a") as f:
        f.write(json.dumps(rec, default=str) + "\n")


def quat_to_mat(q):
    w, x, y, z = q["w"], q["x"], q["y"], q["z"]
    return [
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ]


def mat_vec(R, v):
    return [sum(R[i][j] * v[j] for j in range(3)) for i in range(3)]


def crate_state(ctx):
    """Live crate pose: centre, rotation, attitude theta [deg], and the two near
    bottom corners (floor side and rim side) in world coordinates."""
    pose = ctx.tool("sim.get_object_pose", object_name="crate")["pose"]
    p = pose["position"]
    c = [p["x"], p["y"], p["z"]]
    R = quat_to_mat(pose["rotation"])
    up = mat_vec(R, [0.0, 0.0, 1.0])
    theta = math.degrees(math.atan2(-up[0], up[2]))
    floor_corner = [c[i] + v for i, v in enumerate(mat_vec(R, [-HX, 0.0, -HZ]))]
    rim_corner = [c[i] + v for i, v in enumerate(mat_vec(R, [-HX, 0.0, HZ]))]
    return {"c": c, "R": R, "theta": theta, "floor_corner": floor_corner, "rim_corner": rim_corner, "up": up}


def pin_world(state, sign):
    """Where the pinch point on this arm's end wall is now, from the live pose."""
    c, R = state["c"], state["R"]
    return [c[i] + v for i, v in enumerate(mat_vec(R, [D, sign * WALL_Y, 0.0]))]


def approach_for(phi_deg):
    """Hand approach direction for wrist roll phi about world y: 0 is straight
    down, 90 is horizontal pointing away from the robot (+x)."""
    ph = math.radians(phi_deg)
    return {"x": math.sin(ph), "y": 0.0, "z": -math.cos(ph)}


def frame(ctx, arm, phi_deg):
    fr = ctx.tool("robot.grasp_frame", approach=approach_for(phi_deg), close_heading_deg=90.0, arm_id=arm["arm_id"])
    return fr[arm.get("rot_key", "rotation")], fr


def pose_dict(pos, rot):
    return {"position": {"x": float(pos[0]), "y": float(pos[1]), "z": float(pos[2])}, "rotation": rot}


def plan_to(ctx, arm, pos, phi_deg, n=12, tol=0.006, allow_joint=True):
    """One arm's dense joint track to a pinch-point pose with the wrist locked.
    Linear first; a refused straight line falls back to an eased joint move for
    the short chord, which for a few centimetres is the same path."""
    rot, fr = frame(ctx, arm, phi_deg)
    end = pose_dict(pos, rot)
    plan = ctx.tool("motion.plan_linear", end=end, arm_id=arm["arm_id"], orientation="lock", tolerance=tol, num_waypoints=n)
    if plan.get("planned") and plan.get("trajectory", {}).get("waypoints"):
        wps = [w["positions"] for w in plan["trajectory"]["waypoints"]]
        log("plan_linear", arm=arm["name"], pos=pos, phi=phi_deg, err_m=plan.get("position_error_m"), n=len(wps))
        return wps
    log("plan_linear_refused", arm=arm["name"], pos=pos, phi=phi_deg, reason=plan.get("reason"))
    if not allow_joint:
        return None
    pj = ctx.tool("motion.plan_joint", pose=end, arm_id=arm["arm_id"], orientation="lock", num_waypoints=n)
    perr, rerr = float(pj.get("position_error_m", 1.0)), float(pj.get("rotation_error_rad", 9.0))
    log("plan_joint", arm=arm["name"], pos=pos, phi=phi_deg, err_m=perr, rot_err_deg=math.degrees(rerr))
    if perr > 0.012 or rerr > math.radians(8.0):
        return None
    return [w["positions"] for w in pj["trajectory"]["waypoints"]]


def move_pair(ctx, arms, targets, phi_deg, n=12, settle=4, tol=0.006):
    """Both arms to their pinch-point targets together, wrist roll phi."""
    tracks = {}
    for arm in arms:
        wps = plan_to(ctx, arm, targets[arm["name"]], phi_deg, n=n, tol=tol)
        if wps is None:
            raise RuntimeError(f"{arm['name']} cannot reach {targets[arm['name']]} at roll {phi_deg:.1f}")
        tracks[str(arm["arm_id"])] = wps
    ctx.tool("robot.stream_dual", tracks=tracks, tolerance=0.01, settle_steps=int(settle))


def grip_both(ctx, arms, ramp=24, settle=0, **kw):
    out = {}
    for arm in arms:
        out[arm["name"]] = ctx.tool("robot.set_grip", arm_id=arm["arm_id"], ramp_steps=int(ramp), settle_steps=int(settle), **kw)
    return out


def theta_on_corner(h_above_table):
    """Crate attitude when the pinch point is h above the mat and the floor-side
    near corner still rests on it (the low branch, below top-dead-centre)."""
    s = max(-1.0, min(1.0, h_above_table / RHO))
    return math.degrees(math.asin(s)) - BETA


def pin_about_floor_corner(corner, theta_deg, sign_y):
    """Pinch point when the crate rotates about its floor-side near corner."""
    t = math.radians(theta_deg)
    return [corner[0] + R_PIN * math.cos(t) - HZ * math.sin(t), sign_y, corner[2] + R_PIN * math.sin(t) + HZ * math.cos(t)]


def pin_about_rim_corner(corner, theta_deg, sign_y):
    """Pinch point when the crate rotates about its rim-side near corner."""
    t = math.radians(theta_deg)
    return [corner[0] + R_PIN * math.cos(t) + HZ * math.sin(t), sign_y, corner[2] + R_PIN * math.sin(t) - HZ * math.cos(t)]


def ee(ctx, arm):
    p = ctx.tool("robot.get_ee_pose", arm_id=arm["arm_id"])["pose"]["position"]
    return [p["x"], p["y"], p["z"]]


def slip(ctx, arms, st=None):
    """Hand minus nominal pinch point, in the crate's frame [mm]: dz runs up the
    wall toward the rim, dx along the wall toward the far edge. Growth here is the
    wall sliding in the pads; the pads' own reading is `grip` (closed fraction)."""
    st = st or crate_state(ctx)
    R = st["R"]; out = {}
    for a in arms:
        pin = pin_world(st, a["sign"]); e = ee(ctx, a)
        dv = [e[i] - pin[i] for i in range(3)]
        loc = [sum(R[j][i] * dv[j] for j in range(3)) for i in range(3)]
        g = ctx.tool("robot.get_gripper", arm_id=a["arm_id"])
        out[a["name"]] = {"dx_mm": round(loc[0] * 1000, 1), "dz_mm": round(loc[2] * 1000, 1), "grip": g.get("position")}
    return out


def roll_pair(ctx, arms, phi_deg, n=10, settle=2, max_jump=0.6):
    """Turn both wrists IN PLACE to roll `phi_deg`, smoothly. plan_linear(lock)
    would apply the new rotation as a step at its first waypoint (that step is
    what slid the wall 45 mm in the pads); plan_joint eases the joints between
    the two solutions instead. Refuses a solution that jumps a joint by more
    than `max_jump` rad -- that is the other wrist branch, a half-turn away."""
    tracks = {}
    for arm in arms:
        pos = ee(ctx, arm)
        rot, _ = frame(ctx, arm, phi_deg)
        pj = ctx.tool("motion.plan_joint", pose=pose_dict(pos, rot), arm_id=arm["arm_id"], orientation="lock", num_waypoints=n)
        perr, rerr = float(pj["position_error_m"]), float(pj["rotation_error_rad"])
        wps = [w["positions"] for w in pj["trajectory"]["waypoints"]]
        jump = max(abs(a - b) for a, b in zip(wps[0], wps[-1])) if len(wps) > 1 else 0.0
        log("roll_plan", arm=arm["name"], phi=phi_deg, err_m=perr, rot_err_deg=math.degrees(rerr), joint_jump=jump)
        if perr > 0.01 or rerr > math.radians(6.0) or jump > max_jump:
            raise RuntimeError(f"{arm['name']} cannot roll to {phi_deg:.1f}: {perr*1000:.1f} mm, {math.degrees(rerr):.1f} deg, jump {jump:.2f} rad")
        tracks[str(arm["arm_id"])] = wps
    ctx.tool("robot.stream_dual", tracks=tracks, tolerance=0.01, settle_steps=int(settle))
