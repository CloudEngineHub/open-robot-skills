"""Carry the run's free end into the port, after the last crossing.

**The SUPPORT hand does it, because it is already holding the end.** The two
arms cover complementary halves of the weave bench and the run's free end lies
in the support arm's -- see the reach table at the top of :func:`run`. The work
arm can reach the port and cannot reach the cable it would have to bring there,
which is what this node spent a release trying to do.

**Why this is a node and not another station.** A station is a place the cable
passes; the port is where it stops. The crossing template exists to get material
onto the far side of a post and leave it there under its own tension, and none of
that applies to an end being put into a box: there is no side to be on, nothing
to reverse around, and the thing that has to arrive is one specific end of the
rod rather than whichever span happens to be abreast.

**The end, not the nearest node.** Every other node in this graph asks the
centreline what is closest to a station. Here that would be wrong twice over --
the closest material to the port after a weave is usually the last crossing's
span, and plugging that in leaves the actual end lying on the bench. The rod's
own arc-length parameterisation names what is wanted (``end_b`` at s = 1.0), and
the fitted centreline reproduces it: the free end is the last node of the chain.

**Low, not high, and the difference is the arm rather than the port.** Full-pose
IK over this port, measured: mouth 1.0 mm, centre 1.3 mm, and 120 mm above it
48.9 mm -- out of reach. This arm cannot hold a vertical wrist high and far out
at once, which is the same limit the hand-camera lookdown ran into over the peg
line. So the carry stages over the mouth at ``carry_z`` -- a hand's depth rather
than a hover -- and drops the last millimetres into the channel.

An earlier version of this note said "approached along the bench, never from
above", which the legs below have never done: the last two waypoints are
``over`` and then ``inset`` at the same ``x`` and ``y``, which is a descent. What
the measurement forbids is a HIGH approach, and a hand's depth is not one. The
socket is open on top for exactly this reason: the terminal asset is an
open-topped channel by design.

The port's appearance (``block_query``, ``block_score``, ``block_min_px``), how
far short of its centre the carry aims (``port_inset``) and the carry height
(``carry_z``) are parameters with the weave bench's measured numbers as defaults.
The rod's centreline arrives as ``prior``: the ``rod`` output of the
centreline-perceiving skill, carried by the graph as JSON.
"""

import json
from typing import TypedDict

import numpy as np
from gap import NodeContext

#: What SAM3 is asked for. The blocks are the same accent colour as the spools
#: and a different shape, and "block" is what separates them -- measured, the
#: spool prompt returns 2 on ``port2`` and does not pick up either terminal.
BLOCK_QUERY = "orange block"
BLOCK_SCORE = 0.50
"""Detection floor. Measured on ``port2``'s own frame, the prompt separates
cleanly and there is no reason to sit near the noise:

    "orange block"   0.97, 0.96   then 0.26 and below
    "orange spool"   0.85, 0.84   -- the SPOOLS, at a fifth the area

Both terminals come back in the nineties and the third candidate is a quarter of
that, so half is a floor with nothing near it."""

BLOCK_MIN_PX = 3400
"""Smallest mask, in pixels, that can be a terminal.

The prompt is not perfectly exclusive -- ``"orange block"`` also returns the
spools further down its ranking -- and shape words are a weak separator between
two objects of the same colour. Area is a strong one here, and the populations
moved when the terminals stopped being slabs. Measured on ``port3``'s opening
frame, before and after:

    slabs      blocks 8,100 and 11,600 px   spools 2,450 and 2,480   floor 4,000
    fittings   port 4,766 px                spools 2,479 and 2,779   floor 3,400

Two things changed and only one of them matters. The port is a channel with a
10 mm slot down the middle, so it masks about 40 per cent less than the slab it
replaced -- 4,766 against a 4,000 floor is 19 per cent of headroom, which is not
enough of a floor to be one. And the ANCHOR has stopped appearing at all: its
rails are the mounting foot and are painted metal -- the terminal asset's own
colour for them -- so "orange block" no longer grounds it.

That second one is fine and worth saying why: this node wants the PORT, and it
finds it as the block furthest from the run's anchored end. One candidate makes
that trivially right rather than wrong. The disambiguation stays because a board
seen from another angle may still return both.

3,400 sits between 2,779 and 4,766 with 22 per cent of margin below and 40 above,
which is a floor with something on either side of it."""

PORT_INSET = 0.004
"""How far short of the port's centre the carry aims [m].

**Re-derived off the socket, because the block it was derived off is gone.** It
was 12 mm, and the reason given was: "The block is a 40 mm cube and the rod is
10 mm: aiming at the centre drives the end into the far wall and the contact
pushes it back out." Neither number is this scene's any more -- the terminal was
a 32 mm plate before it became a socket, and the rod has been 7 mm since the
weave family was thinned.

The port is now an open channel with a stop at the back. In its own frame the
channel runs from the mouth at ``+16 mm`` to the back wall at ``-12.8 mm``, and
the block is turned so that axis lies along world ``-x``. So aiming 12 mm short
of centre put the end 4 mm inside a 29 mm channel -- technically in the mouth,
and lying almost entirely outside the thing it was supposed to be plugged into,
which is most of why the port clause read as "nearly" for so long.

4 mm short of centre puts it 12 mm in with 17 mm still ahead of it before the
stop. Deep enough to be in the socket, clear enough that an overshoot is
absorbed by the channel rather than by the wall.
"""

#: Height above the bench the carry runs at [m].
#:
#: **It was 30 mm, and a spool flange stands 23 mm.** So the carry ran the rod's
#: tail seven millimetres over the tops of the very fixtures it had just been
#: seated against, and dragged it across them on the way to the port.
#:
#: That is the whole of why the last crossing never held. Measured over three
#: episodes with the carry skipped entirely, every seat reads **0.088** against a
#: 0.047 threshold -- placement was never the problem. Run the carry and seat2
#: reads 0.000, 0.015, 0.029: the crossing nearest the port, and the one the
#: tail sweeps over on its way there.
#:
#: Low was chosen because the arm cannot hold a vertical wrist high and far out
#: at once (see the module docstring), and that constraint is real -- but it
#: binds on the MOUTH pose, which is at bench level, not on the three staging
#: poses. Those are allowed to be refused and skipped; the mouth is not.
CARRY_Z = 0.070

#: Control steps of settle after the jaws open, so the rod is measured at rest.
SETTLE = 30


class Output(TypedDict):
    exit: str
    detail: str
    port: str


def _blocks(
    ctx: NodeContext, camera: str, query: str, score_floor: float, min_px: int
) -> list[list[float]]:
    """Every terminal block the camera can see, as world centroids."""
    obs = ctx.tool("robot.get_observation")
    frames = obs["cameras"] if isinstance(obs, dict) else obs.cameras
    # BOTH shapes, as the centreline perceiver does. An observation carries its
    # frames as a dict on some paths and a list of named blocks on others, and a
    # lookup that only handles the dict returns None on the list -- silently,
    # which is how this node reported "no block in view" for three runs while
    # SAM3 was never being called at all.
    block = None
    if isinstance(frames, dict):
        block = frames.get(camera)
    else:
        for fr in frames or ():
            if str(fr.get("name", "")) == camera:
                block = fr
                break
    if block is None:
        have = (
            sorted(frames)
            if isinstance(frames, dict)
            else [str(f.get("name", "?")) for f in (frames or ())]
        )
        print(f"[plug] no {camera!r} camera in the observation; have {have}", flush=True)
        return []
    # The same two-dict, two-key lookup the centreline perceiver does. A frame
    # carries its channels under ``images`` on some paths and inline on others,
    # and the depth key is ``depth`` or ``depth_data`` -- measured, reading only
    # ``images["depth"]`` returned None on the live observation and this node
    # reported "no block in view" while SAM3 was grounding both blocks at 0.97.
    images = block.get("images") if isinstance(block.get("images"), dict) else {}

    def _first(*keys):
        for src in (images, block):
            for key in keys:
                v = src.get(key)
                if v is not None:
                    return v
        return None

    rgb = np.asarray(_first("rgb"), dtype=np.uint8)
    depth = _first("depth", "depth_data")
    if depth is None:
        print("[plug] camera frame carries no depth", flush=True)
        return []
    depth = np.asarray(depth, dtype=np.float64)
    if depth.ndim == 3:
        depth = depth[..., 0]
    K = np.asarray(block["intrinsics"], dtype=np.float64).reshape(3, 3)
    pose = block["pose"]
    if isinstance(pose, dict):
        pos, rot = pose.get("position", {}), pose.get("rotation", {})
        px, py, pz = (float(pos[k]) for k in ("x", "y", "z"))
        w, x, y, z = (float(rot[k]) for k in ("w", "x", "y", "z"))
    else:
        flat = np.asarray(pose, dtype=np.float64).reshape(-1)
        px, py, pz = flat[:3]
        w, x, y, z = flat[3:7]
    R = np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = R, (px, py, pz)

    found = ctx.tool("sam3.segment_text", image=rgb, query=query, max_results=0)
    masks, scores = found.get("masks") or [], found.get("scores") or []
    _sc = [round(float(v), 2) for v in scores]
    _ar = [int(np.asarray(m).sum()) for m in masks[:6]]
    print(
        f"[plug] sam3 {query!r}: {len(_sc)} masks, top scores {_sc[:6]}, "
        f"top areas {_ar}  (need score>={score_floor}, area>={min_px})",
        flush=True,
    )
    out: list[list[float]] = []
    for mask, score in zip(masks, scores, strict=False):
        if float(score) < score_floor:
            continue
        v, u = np.nonzero(np.asarray(mask) > 0)
        if len(u) < min_px:
            continue
        d = depth[v, u]
        ok = np.isfinite(d) & (d > 1e-4)
        if ok.sum() < 8:
            continue
        u, v, d = u[ok], v[ok], d[ok]
        cam = np.stack(
            [(u - K[0, 2]) * d / K[0, 0], (v - K[1, 2]) * d / K[1, 1], d, np.ones_like(d)],
            axis=1,
        )
        world = (T @ cam.T).T[:, :3]
        out.append([float(c) for c in np.median(world, axis=0)])
    return out


def run(
    ctx: NodeContext,
    scene: str = "",
    prior: str = "",
    arm_id: int = 0,
    grip_open_m: float = 0.0475,
    grip_close_m: float = 0.003,
    block_query: str = BLOCK_QUERY,
    block_score: float = BLOCK_SCORE,
    block_min_px: int = BLOCK_MIN_PX,
    port_inset: float = PORT_INSET,
    carry_z: float = CARRY_Z,
) -> Output:
    """Find the port, then carry the held free end into its mouth and let go.

    ``scene`` is the fixture survey's JSON (``stations`` as ``[x, y]``);
    ``prior`` is the rod's last fitted centreline as JSON: a list of
    ``[x, y, z]`` world points, or a ``Centerline`` object whose ``points``
    are that list. ``arm_id`` is the WORK arm and is only used as
    the name of what this node refuses to plug with; ``grip_open_m`` is the
    width the delivering hand opens to. ``grip_close_m`` is accepted so a graph
    can bind the same grip inputs to every crossing skill; nothing closes here.
    """
    del grip_close_m
    seen = json.loads(scene or "{}")
    stations = [(float(a), float(b)) for a, b in (seen.get("stations") or [])]
    rod = json.loads(prior or "null")
    if isinstance(rod, dict):
        # The perception bundle's ``rod`` is the bare point list, but a graph
        # may bind a whole ``Centerline`` object; either reads the same.
        rod = rod.get("points")
    # ENTRY, unconditionally. Every other exit below prints too, because the
    # first version of this node printed only on the happy path and a silent
    # early return is indistinguishable in a log from a node that never ran.
    print(
        f"[plug] entered: {len(stations)} stations, {0 if not rod else len(rod)} rod nodes",
        flush=True,
    )
    if not rod or len(rod) < 2 or not stations:
        print("[plug] no centreline or no stations; nothing to terminate", flush=True)
        return {"exit": "unplugged", "detail": "no centreline or no stations", "port": "null"}
    curve = np.asarray(rod, dtype=np.float64).reshape(-1, 3)

    blocks = _blocks(ctx, "cable", block_query, float(block_score), int(block_min_px))
    if len(blocks) < 1:
        # The weave layouts have no terminals, so this is the normal path there
        # and not a failure -- a graph routes ``unplugged`` on without alarm.
        print("[plug] no block in view; this layout has no port", flush=True)
        return {"exit": "unplugged", "detail": "no block in view", "port": "null"}

    # WHICH BLOCK IS THE PORT: the one furthest from the rod's anchored end.
    # Both terminals are the same part and the same colour, so nothing about
    # their appearance separates them -- what does is that one of them is where
    # the run starts, and the run starts at the end the support hand is holding.
    anchored = curve[0] if curve[0][1] < curve[-1][1] else curve[-1]
    free = curve[-1] if curve[0][1] < curve[-1][1] else curve[0]
    port = max(blocks, key=lambda b: float(np.hypot(b[0] - anchored[0], b[1] - anchored[1])))
    print(
        f"[plug] {len(blocks)} block(s); port at "
        f"({port[0]:.3f}, {port[1]:.3f}) free end at ({free[0]:.3f}, {free[1]:.3f})",
        flush=True,
    )
    port_json = json.dumps([round(float(v), 4) for v in port])

    # ONE CALL, READ TWICE. ``cable.plan_support`` is asked here rather than
    # further down because two separate things below need it: which hand is the
    # support hand (so it can be told to let go) and what height this cell's
    # jaws have to be at to straddle a rod lying on the bench. Both are
    # arithmetic over the hand's own colliders; neither reads where the rod is.
    support = ctx.tool("cable.plan_support")
    support_arm = str(support.get("arm_name", ""))

    # THE SUPPORT ARM CARRIES THE END, NOT THE WORK ARM, and this is the one
    # thing about this task that was not a bug in a number.
    #
    # The two hands cover complementary halves of this bench. Measured with a
    # reach sweep over the ``port3`` bench at the hand height of 0.8035 m, at
    # the spawn line x = 0.42: the work arm reaches y in [-0.55, +0.21] and the
    # support arm y in [-0.22, +0.55]. The run's free end sits at y = +0.29 to
    # +0.35 -- **inside the support arm's half and outside the work arm's**.
    #
    # This node asked the WORK arm to fetch it. Full-pose IK on the exact
    # waypoints it commands, straight-down wrist:
    #
    #     take the free end (0.408, +0.294)   right  65.2 mm   left  1.3 mm
    #     out past the end  (0.408, +0.354)   right 200.6 mm   left  (staged)
    #     across            (0.588, +0.354)   right 230.5 mm   left  1.2 mm
    #     over the port     (0.588, +0.122)   right   6.8 mm   left  0.8 mm
    #     the mouth         (0.588, +0.122)   right   2.4 mm   left  1.5 mm
    #
    # The work arm can reach the PORT and cannot reach the CABLE it is supposed
    # to bring there. Sixty-five millimetres is not a tuning problem; it is the
    # arm being asked to work in the other arm's half of the bench. Every
    # refusal was swallowed by ``_go``, so the hand skipped the grasp, flew to
    # the mouth holding nothing, and the node reported a delivery.
    #
    # The support hand has been holding this exact end since before the first
    # crossing. So it delivers it: no release, no re-grasp, no second hand
    # crossing the weave. The hand that holds the end is the hand that plugs it,
    # which is also how a person does it.
    # AND IF THE SUPPORT HAND CANNOT BE IDENTIFIED, DO NOT GUESS.
    #
    # This read ``int(support.get("arm_id", arm_id))``, whose default is the
    # node's own ``arm_id`` -- the WORK arm. So every way ``plan_support`` can
    # come back empty (a bus timeout, a refused build, a cell with one arm)
    # silently reinstated exactly the configuration the table above was written
    # to rule out: the work arm sent for a cable 65 mm outside its envelope,
    # skipping the grasp and flying to the mouth holding nothing, then reporting
    # a delivery. A fallback that lands on a known-broken state is worse than no
    # fallback, because the log looks the same either way.
    if "arm_id" not in support:
        print(
            f"[plug] no support hand from cable.plan_support; refusing to plug with "
            f"the work arm ({arm_id}) -- it cannot reach the run's free end",
            flush=True,
        )
        return {"exit": "unplugged", "detail": "no support arm", "port": port_json}
    arm_id = int(support["arm_id"])
    # WHICH HAND, in the log. Two arms and a claim about which one delivers is
    # exactly the kind of thing that stays true in a comment long after it has
    # stopped being true in the code.
    print(
        f"[plug] the {support_arm or f'arm {arm_id}'} hand delivers -- it has held the "
        f"free end since before the first crossing",
        flush=True,
    )
    ee = ctx.tool("robot.get_ee_pose", arm_id=int(arm_id))
    p = (ee.get("pose", ee)).get("position", {})
    here = np.array([float(p.get(k, 0.0)) for k in ("x", "y", "z")])

    # THE HEIGHT COMES FROM THE HAND, NOT FROM THE FITTED CURVE, and this was
    # the whole of why the port clause read 0.000 on every episode.
    #
    # This node was the only one in the graph that took a working HEIGHT off the
    # perceived centreline -- ``surface = median(curve[:, 2])`` -- and the
    # centreline's z is not fit for that. Measured on ``port3`` episode 0: the
    # rod lies at z = 0.7535 and the plug commanded its grasp at ~0.840, so the
    # jaws closed **87 mm above the cable**, travelled to the mouth holding air
    # and reported a delivery. Across the whole episode the work hand's pads
    # never came within 78 mm of the rod, and when they were over it in xy they
    # sat 80 to 257 mm above.
    #
    # Every other node survives the same curve because it takes only x and y
    # from it: the crossing planner computes its own ``rest_z`` from the bench
    # and the hand's measured pad drop, which is why all three crossings seat at
    # 0.073 to 0.088 off the same fit that defeats this node.
    #
    # So do what they do. ``cable.plan_support``'s ``grasp_m`` carries exactly
    # that height -- ``max(surface_z + rod radius, surface_z + pad_tip_drop +
    # PAD_FLOOR_CLEAR)``, 0.7607 on this cell -- and it is arithmetic over the
    # hand's colliders and the bench, not a reading of where the rod is. The
    # median of the curve stays as the fallback, because a node that cannot get
    # a height at all should still do what it used to rather than nothing.
    grasp_ref = support.get("grasp_m") or []
    if len(grasp_ref) >= 3:
        surface = float(grasp_ref[2])
        print(
            f"[plug] grasp height {surface:.4f} m from the hand's own pad drop "
            f"(the fitted curve's median z was {float(np.median(curve[:, 2])):.4f})",
            flush=True,
        )
    else:
        surface = float(np.median(curve[:, 2])) + 0.004
        print(
            f"[plug] no support plan; falling back to the curve's median z {surface:.4f}",
            flush=True,
        )

    # WHERE THE END IS: in the jaws already, so there is nothing here to reach
    # to. The fitted end is still read above for the surface height and is not
    # a waypoint -- the carry stages off the HAND's own pose, because the hand
    # is what is moving and the fit is only an estimate of where it is.
    # Into the mouth ALONG the bench, from the station side.
    toward = np.array([port[0], port[1], surface])
    inset = toward - np.array([float(port_inset), 0.0, 0.0]) * np.sign(port[0] - here[0] or 1.0)
    # ROUTE AROUND THE SEATS, NOT ACROSS THEM.
    #
    # The port is further out in +x than every station, so the direct path from
    # the free end to the port sweeps over the crossings that were just seated
    # and drags them out from under their spools -- measured, both seats went
    # from held to wrong-side and the score fell 0.67 -> 0.33 while the end
    # itself landed correctly in the mouth.
    #
    # There is no shortage of cable: with the anchor at one end and two crossings
    # taken, ~225 mm of rod remains for a 96 mm span to the port. What fails is
    # the PATH, so the fix is the path. The carry goes out past the free end's
    # own y first, then across in x well clear of the station line, and only then
    # back down into the mouth -- three sides of a rectangle rather than the
    # diagonal that cuts the weave.
    # Staged off the hand's OWN pose rather than off the fitted end, because the
    # hand is what is moving and the fit is only an estimate of where it is.
    clear_y = max(float(here[1]), float(port[1]) + 0.06)
    # THE WRIST, ASKED FOR RATHER THAN WRITTEN DOWN. Every waypoint below wants
    # the hand pointing straight down with its jaws across world +x, and the
    # literal that says so -- {"w": 0, "x": 1, "y": 0, "z": 0} -- is only that
    # pose for a hand whose approach axis is its tool-local +z. This one's is
    # -z, so the literal was a half turn out and the arm was reaching the mouth
    # from the wrong side of its own last joint. ``robot.grasp_frame`` composes
    # it from the live hand's measured axes; it returns both wrist solutions and
    # this takes the primary, which is the one the support arm's own plan uses.
    down = ctx.tool("robot.grasp_frame", arm_id=int(arm_id))["rotation"]
    lift = np.array([here[0], here[1], surface + float(carry_z)])
    across = np.array([inset[0], clear_y, surface + float(carry_z)])
    over = np.array([inset[0], inset[1], surface + float(carry_z)])

    def _go(target, what: str) -> bool:
        """One waypoint. A refused pose is reported and skipped, not fatal.

        The carry is four poses and they are not equally load-bearing: the
        staging heights exist to keep the rod off the fixtures, and losing one of
        them costs tidiness. Losing the MOUTH pose costs the task. Aborting the
        whole leg because a staging pose missed threw away the ones that solved
        -- measured, the first version refused the entire carry on one
        ``RuntimeError`` and never reached the port at all.

        **Swallowing a refusal is only safe now that the poses are reachable.**
        This same tolerance is what hid the real defect for a release: the work
        arm refused the grasp by 65 mm, the refusal was printed and skipped, and
        the node carried on to a mouth it could reach with nothing in its jaws.
        A skipped staging pose is tidiness; a skipped GRASP was the task. There
        is no grasp leg any more -- the hand arrives holding the end -- which is
        what makes this safe rather than merely quiet.
        """
        # RETRY LOWER BEFORE GIVING UP, because height is what this arm refuses
        # for. The envelope narrows as the wrist goes up and out (the module
        # docstring has the sweep), so a staging pose refused at carry height is
        # very often reachable 20 mm down -- and 20 mm down still clears a 23 mm
        # flange by more than the 30 mm this carry used to run at.
        #
        # This matters more than it did. When the carry ran at 30 mm a skipped
        # staging pose was tidiness. At 70 mm, skipping ``across`` drops the
        # hand onto the diagonal from the free end to the port, which is the
        # path that drags the cable over the crossings -- the exact failure
        # ``carry_z`` was raised to stop. Silently taking it back is not an
        # option.
        last = ""
        for drop in (0.0, 0.020, 0.040):
            z = float(target[2]) - drop
            try:
                ctx.tool(
                    "robot.go_to_pose",
                    arm_id=int(arm_id),
                    max_steps=160,
                    pose={
                        "position": {"x": float(target[0]), "y": float(target[1]), "z": z},
                        "rotation": dict(down),
                    },
                )
                if drop:
                    print(f"[plug]   {what} at {drop * 1000:.0f} mm lower", flush=True)
                return True
            except Exception as exc:  # a refused pose is skipped, not fatal
                last = type(exc).__name__
        print(f"[plug]   {what} refused at every height: {last}", flush=True)
        return False

    # NO RELEASE AND NO RE-GRASP, and the history of this comment is worth
    # keeping because it went round twice before the geometry was looked at.
    #
    # It said "RELEASE THE FAR END FIRST", then "THE FAR END STAYS HELD, and
    # that was measured both ways":
    #
    #   anchor released, direct path     0.33   both seats lost
    #   anchor released, routed path     0.67   the near seat still pulled out
    #   anchor held,     routed path     what shipped
    #
    # Both were measured with the support hand at the anchor line, +0.17 --
    # the MIDDLE of a 0.82 m rod, because that constant was sized for the
    # family's original 0.50 m one. Holding the middle while another hand
    # carries the end is a sound thing to do and the table is a fair reading of
    # it. It is simply not the question any more.
    #
    # The anchor now puts the support hand 20 mm in from the rod's own free
    # tip, which is where it was always meant to be. So the hand that would have
    # been released is the hand holding the thing being delivered, and the hand
    # that would have done the releasing cannot reach it. Both halves of the old
    # argument dissolve: nothing lets go, nothing is picked up, and the run
    # stays pinned at its welded end throughout.
    #
    # What is left is a carry. Out to carry height, out in +x well clear of the
    # station line, then down the port's own x into the mouth -- three sides of a
    # rectangle rather than the diagonal, because the port is further out in +x
    # than every station and the direct path sweeps over the crossings just
    # seated. Measured when it did: both seats went from held to wrong-side and
    # the score fell 0.67 to 0.33 while the end itself landed correctly.
    try:
        _go(lift, "up to carry height")
        # ``across`` is the one staging pose that is load-bearing: it is what
        # keeps the run's tail off the crossings. If it is lost, say so in the
        # RESULT and not only in the log, so a run's traces carry the reason its
        # seats came back thin.
        routed = _go(across, "across, clear of the stations")
        if not routed:
            print(
                "[plug]   WARNING: carrying on the diagonal; expect the crossing "
                "nearest the port to be dragged",
                flush=True,
            )
        _go(over, "over the port")
        # The one that has to land.
        if not _go(inset, "mouth"):
            return {
                "exit": "unplugged",
                "detail": "the port mouth was unreachable",
                "port": port_json,
            }
        # Only now does anything open, and only this hand -- the SUPPORT hand,
        # by the id its own planner returned.
        ctx.tool("robot.set_grip", width_m=float(grip_open_m), arm_id=int(arm_id))
        ctx.tool("robot.wait_steps", steps=int(SETTLE))
        # Out of the way, so the verifier measures the rod and not the hand.
        # Back along -x rather than up: this arm is at full stretch over the
        # port and a vertical retreat from there is the pose the reach sweep
        # refuses.
        _go(np.array([inset[0] - 0.10, clear_y, surface + 0.12]), "retreat")
    except Exception as exc:  # a refused pose is a failed plug, not a crash
        print(f"[plug] refused: {type(exc).__name__}", flush=True)
        return {
            "exit": "unplugged",
            "detail": f"{type(exc).__name__} on the carry",
            "port": port_json,
        }

    # WHAT THIS LINE IS A CLAIM ABOUT. The hand, and only the hand.
    #
    # It used to read "carried the free end to (x, y)" with the MOUTH's
    # coordinates in it -- the waypoint that was commanded, not a measurement of
    # the rod. On an episode where the grasp had missed entirely it printed
    # exactly the same sentence: the free end finished at (0.461, 0.249), 178 mm
    # from the port, having never moved toward it, and the closest any body came
    # was 85 mm. The log said delivered, the score said 0.000, and the
    # disagreement took a privileged replay to see.
    #
    # WHERE THE HAND IS, NOT WHERE IT WAS SENT. An earlier version printed
    # ``inset``, the commanded waypoint. A refused mouth pose printed the same
    # line as a delivered one, which is the whole class of bug this node keeps
    # producing: a log that describes the plan rather than the outcome.
    landed = None
    try:
        _ee = ctx.tool("robot.get_ee_pose", arm_id=int(arm_id))
        _p = (_ee.get("pose", _ee)).get("position", {})
        landed = np.array([float(_p.get(k, 0.0)) for k in ("x", "y", "z")])
    except Exception:  # a pose readback is diagnostics, not the task
        pass
    if landed is None:
        print(
            f"[plug] hand was commanded to the mouth at ({inset[0]:.3f}, {inset[1]:.3f}); "
            f"its actual pose could not be read",
            flush=True,
        )
    else:
        miss = float(np.linalg.norm(landed[:2] - inset[:2]))
        print(
            f"[plug] hand is at ({landed[0]:.3f}, {landed[1]:.3f}), "
            f"{miss * 1000:.0f} mm from the mouth it was sent to; "
            f"whether the rod came with it is the port clause's to say",
            flush=True,
        )
    # The exit stays ``plugged`` -- the router knows two exits and inventing a
    # third routes nowhere -- but the DETAIL says whether the run got there the
    # routed way or on the diagonal, because those two produce the same port
    # clause and very different seat clauses.
    return {
        "exit": "plugged",
        "detail": (
            "free end carried into the port"
            if routed
            else "free end carried into the port ON THE DIAGONAL -- the "
            "clear-of-the-stations pose was refused at every height"
        ),
        "port": port_json,
    }
