"""Read the cable's centreline, from a third-person camera pitched at the bench.

**Fit fresh, track only when the fit is broken.** A cold fit carries no history,
so when the whole rod is in frame it is simply the better estimate; the tracker's
value is entirely in surviving frames where the rod is NOT all there. Which of
the two is right is therefore a property of the view, not a fixed choice, and it
changed when the camera moved -- see :data:`WHOLE_ROD` for the eight poses that
measured it; the move was pitching the bench camera 55 degrees off the vertical.

**Discrete, not continuous.** The cable's shape only changes when a crossing
moves it, so it is read once per station rather than every frame. Continuous
tracking would spend a detector call per control step to re-derive a shape that
did not change between them.

**The failure this guards is a broken skeleton, not drift.** When something does
stand between the lens and the rod, the mask arrives in two pieces and the chain
merge resolves them to one -- a fit that is confidently wrong and *short*. Length
against the prior is what detects that, and a short fit hands the frame to the
tracker, which asks each of the 42 known nodes what the frame says about it and
moves the unanswered ones by the displacement field their visible neighbours
generate. That is Motion Coherence Theory doing the job it was written for, in
the case it was written for.

**Two gates, for two different questions.** :data:`WHOLE_ROD` asks whether a
fresh fit saw the whole object. :data:`MIN_VISIBLE` asks whether a tracked update
saw enough of the model to be believed; below it the prior simply stands, because
a tracker that accepted every frame would walk the model onto whatever happened
to be visible.

**The prior is carried by the graph, not by this script.** ``prior`` is the
centreline the previous pass returned (or the survey's, on the first pass), as
JSON text; the ``rod`` this script returns is the next pass's prior. A node
that holds the frame or loses it returns the prior it was given, so the loop
never binds a worse model than it had. There is no memory tool and no privileged
twin: a graph that can fall back to the simulator is a graph whose perception is
never load-bearing.
"""

import json
from typing import Any, TypedDict

import numpy as np
from gap import NodeContext

#: Where the centreline is born AND maintained: a fixed third-person camera
#: standing 500 mm off the cable's own run and pitched 55 degrees above the
#: bench, 1280x960.
#:
#: **Not the hand camera, and that was measured rather than assumed.** At any
#: pose this arm can hold, an eye-in-hand frames a fifth of the cable, or none
#: of it at the wide tilt that fixes the framing. An A/B over three episodes
#: came out identical either way -- 0.667 mean -- so the move it costs buys
#: nothing. A camera that can simply be *placed* where the object is gets the
#: whole rod at 0.50 mm/px.
#:
#: **And placed off the vertical, which mattered far more than the placement.**
#: Overhead, this camera looked down the axis the gripper descends, so the hand
#: covered the rod exactly when the rod was moving: 28 of 42 nodes visible, and
#: a centreline fitted at 60 mm RMS. Pitched to 55 degrees it sees 41 of 42 and
#: fits at 6.7 mm.
INIT_CAMERA = "cable"

#: The same camera. Init and update differ in the *operation* -- a cold fit
#: needs the whole object, an update needs a prior -- not in where they look.
TRACK_CAMERA = "cable"

ROD_QUERY = "thin white cable"
ROD_SCORE = 0.20
CURVE_NODES = 42

#: Nodes that must have real correspondence before a TRACKED update is believed.
#:
#: **Six, and the arithmetic matters.** 42 nodes over a 500 mm rod is 12 mm of
#: model per node, so the wrist's 155 mm window holds about thirteen of them --
#: and the hand occupies a third of its own frame, so an honest close view
#: delivers eight to ten. Ten was therefore right at the boundary and rejected
#: real wrist updates: measured, a pass that reached the viewpoint to 3 mm fell
#: back to the bench view anyway. Six is ~70 mm of rod, which is still a
#: meaningful anchor and comfortably inside what the window gives.
#:
#: Deliberately NOT raised when the camera improved. This gate now guards only
#: the fallback, which by construction runs on the frames where the view is bad;
#: tightening it against the 41/42 the good frames deliver would be tuning a
#: threshold on a population that no longer reaches it.
MIN_VISIBLE = 6

#: Correspondence mass below which a node counts as unobserved. Mirrors the
#: tracker's own floor; stated here because this node routes on it.
VIS_FLOOR = 0.05


#: Fraction of the prior's arclength a fresh fit must reach to be taken over a
#: tracked update.
#:
#: **This node used to track unconditionally, and that is now the wrong default.**
#: The tracker exists to survive occlusion, and it was paying for itself against
#: an overhead camera that lost a third of the model whenever the arm was on the
#: rod. Pitching that camera off the vertical removed the occlusion, and with it
#: the reason. Measured over eight arm poses from the new viewpoint, scoring
#: both against the rod's own 42 bodies:
#:
#:   cold fit per frame   median 6.7 mm   p90 8.4 mm   max  9.3 mm
#:   tracked              median 9.0 mm   p90 9.7 mm   max  9.9 mm
#:   nodes visible        median 41/42    min 40/42
#:
#: The tracker is now the WORSE estimate, and monotonically so across the eight
#: passes (7.6, 8.6, 9.2, 9.6, 9.6, 9.9): with nothing hidden, every pass is the
#: prior being dragged along rather than a hidden stretch being carried, and the
#: error only compounds. A prior is a liability when the object is fully visible.
#:
#: It is still the right fallback, because the cold fit's own failure is sharp
#: rather than gradual: a skeleton broken by the arm merges to one fragment, and
#: that measured 338 mm of a 488 mm rod. 0.80 sits between the two populations
#: with room on both sides -- honest fits ran 481-518 mm, or 0.99-1.06 of the
#: truth, and the fragment was 0.69.
WHOLE_ROD = 0.80

#: Candidate masks considered before one is chosen, and the widest an accepted
#: one may be.
#:
#: **The top-scoring mask is not always the cable.** The bench is grey, the rod
#: is white and so are the two arms, and a text prompt that grounds "white rod"
#: will sometimes return a blob that has merged the rod with an arm beside it.
#: That blob is confidently scored and quietly wrong: measured on ``weave3``
#: with a 700 mm rod, the fits came back 714, 707, 694 mm on four episodes and
#: **963 and 983 mm** on the other two -- and every episode with an inflated
#: seed failed, while three of four with an honest one solved.
#:
#: A cable is separable from an arm by SHAPE, not by score. The fit already
#: measures mean radius as mask area over centreline length, which is exactly
#: the ratio that distinguishes them: this rod is 3.5 mm, and anything merged
#: with a gripper reads several times that. So several candidates are fitted and
#: the thinnest plausible one is taken, rather than the first.
MASK_CANDIDATES = 4
MAX_ROD_RADIUS = 0.012


class Output(TypedDict):
    route: str
    rod: str
    arclength_m: float
    source: str
    camera: str
    visible: int


def _points(text: str | None) -> list[list[float]]:
    """The centreline points carried by ``text``: a JSON list of ``[x, y, z]``
    triples, or a JSON ``Centerline`` object whose ``points`` are that list.
    Empty text, ``null`` and a curve of fewer than two points all read as no
    prior at all."""
    if not text:
        return []
    raw = json.loads(text)
    if isinstance(raw, dict):
        raw = raw.get("points")
    pts = [[float(v) for v in p] for p in (raw or [])]
    return pts if len(pts) >= 2 else []


def _encode(points: list[list[float]] | None) -> str:
    """The wire form of a centreline: JSON text of its points, 0.01 mm precision.
    Encoding the same points twice gives the same text, so a prior that is
    handed back unchanged is byte-identical to what the loop already carried."""
    if not points:
        return "null"
    return json.dumps([[round(float(v), 5) for v in p] for p in points])


def _camera(observation: Any, name: str):
    frames = observation["cameras"] if isinstance(observation, dict) else observation.cameras
    if isinstance(frames, dict):
        return frames.get(name)
    for frame in frames or ():
        if str(frame.get("name", "")) == name:
            return frame
    return None


def _frame(block):
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
        return None
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
        ],
        dtype=np.float64,
    )
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = R, (px, py, pz)
    return rgb, np.asarray(depth, dtype=np.float64), K, T


def _masks(ctx: NodeContext, rgb, query: str, floor: float, limit: int) -> list[np.ndarray]:
    found = ctx.tool("sam3.segment_text", image=rgb, query=query, max_results=int(limit))
    masks, scores = found.get("masks") or [], found.get("scores") or []
    return [
        np.asarray(m) for m, sc in zip(masks, scores, strict=False) if float(sc) >= float(floor)
    ]


def _length(curve) -> float:
    pts = np.asarray(curve, dtype=np.float64).reshape(-1, 3)
    return float(np.sum(np.linalg.norm(np.diff(pts, axis=0), axis=1))) if len(pts) > 1 else 0.0


def _out(route, curve, source, camera, visible) -> Output:
    # Printed, not only returned. ``node_data/<node>/output.json`` holds the
    # LAST write per node, so a per-pass fact -- which camera carried this
    # update, how much of the model was in frame -- is invisible in a trace of a
    # loop. Three claims about this node were wrong before it said so out loud.
    length = f"{_length(curve) * 1000:.0f}mm" if curve else "-"
    print(
        f"[track] source={source} camera={camera or '-'} visible={visible} len={length}",
        flush=True,
    )
    return {
        "route": route,
        "rod": _encode(curve),
        "arclength_m": round(_length(curve), 4) if curve else 0.0,
        "source": source,
        "camera": camera,
        "visible": int(visible),
    }


def run(
    ctx: NodeContext,
    prior: str = "",
    scene: str = "",
    camera: str = "",
    index: int = 0,
    init_camera: str = INIT_CAMERA,
    track_camera: str = TRACK_CAMERA,
    rod_query: str = ROD_QUERY,
    rod_score: float = ROD_SCORE,
    whole_rod: float = WHOLE_ROD,
    min_visible: int = MIN_VISIBLE,
    max_rod_radius: float = MAX_ROD_RADIUS,
    curve_nodes: int = CURVE_NODES,
    mask_candidates: int = MASK_CANDIDATES,
) -> Output:
    """One read of the cable's centreline.

    ``prior`` is the centreline the loop carries (JSON text; empty on a cold
    start), ``scene`` the opening survey (JSON, its ``rod`` is the seed), ``camera``
    the view the look chose (empty: the bench camera), ``index`` the pass number.
    The returned ``rod`` is the next pass's ``prior``.
    """
    prior_pts = _points(prior)
    observation = ctx.tool("robot.get_observation")

    # ---------------------------------------------------------- cold start
    #
    # SEEDED FROM THE OPENING SURVEY, not re-fitted here. The survey already
    # fits the whole rod from the bench view with both arms at home; by the
    # time this node first runs, the look has parked the hand over station 0
    # and the arm is *in the bench camera's way*. Measured: re-fitting at this
    # point read 381 mm of a 500 mm rod, because the hand was standing on the
    # rest of it.
    #
    # The model has to be born from an unobstructed view of the whole object.
    # That view exists exactly once per episode, before the first move -- so
    # the first pass (index 0) takes the survey's curve whatever the loop
    # carries, and a pass with no prior at all takes it too.
    if int(index) == 0 or not prior_pts:
        seeded = _points(json.dumps((json.loads(scene or "{}") or {}).get("rod")))
        if seeded:
            return _out("found", seeded, "seeded", init_camera, len(seeded))
    if not prior_pts:
        block = _camera(observation, init_camera)
        got = _frame(block) if block is not None else None
        if got is None:
            return _out("lost", None, "no-camera", init_camera, 0)
        rgb, depth, K, T = got
        candidates = _masks(ctx, rgb, rod_query, rod_score, mask_candidates)
        if not candidates:
            return _out("lost", None, "no-mask", init_camera, 0)
        # Fit every candidate and keep the thinnest one that is still a curve.
        # This is the model the whole episode is built on, so it is worth four
        # fits once rather than a wrong prior for the rest of the run.
        best, best_r, pts = None, None, []
        for cand in candidates:
            trial = ctx.tool(
                "curve.fit_centerline",
                mask=cand,
                depth=depth,
                intrinsics=K,
                camera_pose=T,
                nodes=int(curve_nodes),
            )
            tp = trial.get("points") or []
            r = float(trial.get("radius_m") or 0.0)
            if len(tp) < 2 or r <= 0.0 or r > float(max_rod_radius):
                continue
            if best_r is None or r < best_r:
                best, best_r, pts = trial, r, tp
        if best is None:
            return _out("lost", None, "no-fit", init_camera, 0)
        print(
            f"[track]   seed: {len(candidates)} candidate mask(s), took "
            f"radius {best_r * 1000:.1f} mm",
            flush=True,
        )
        curve = [[float(v) for v in p] for p in pts]
        return _out("found", curve, "initialised", init_camera, len(curve))

    # ------------------------------------------------------------- update
    # The look chose a camera; prefer it, and fall back to the bench view when
    # the hand did not reach its viewpoint.
    #
    # A COLD FIT IS TRIED FIRST and taken when it returns a whole rod -- see
    # WHOLE_ROD. The tracker is the fallback, not the default, which is the
    # opposite of what this node used to do and is a consequence of the camera
    # moving off the vertical.
    # Each camera once: the same view retried with the same frame gives the
    # same answer and costs a detector call, a fit and a track to say so.
    order: list[str] = []
    for c in (str(camera or ""), str(track_camera or ""), str(init_camera or "")):
        if c and c not in order:
            order.append(c)
    frames = observation["cameras"] if isinstance(observation, dict) else observation.cameras
    have = (
        sorted(frames)
        if isinstance(frames, dict)
        else [str(f.get("name", "?")) for f in (frames or ())]
    )
    print(f"[track]   camera_in={camera!r} order={order} have={have}", flush=True)
    for name in order:
        block = _camera(observation, name)
        got = _frame(block) if block is not None else None
        if got is None:
            print(f"[track]   {name}: not in observation (or no depth)", flush=True)
            continue
        rgb, depth, K, T = got
        masks = _masks(ctx, rgb, rod_query, rod_score, mask_candidates)
        if not masks:
            print(f"[track]   {name}: no rod mask", flush=True)
            continue
        mask = masks[0]

        # A fresh fit, and its length is the test of whether to believe it. The
        # cold fit's failure mode is not drift -- it is a skeleton broken in two
        # by the arm, which the chain merge resolves to ONE fragment. A fragment
        # is short, and short is measurable against a rod whose length is known
        # from the prior. Anything full-length is a fit of the whole object with
        # no accumulated history in it, which is strictly the better estimate.
        want = _length(prior_pts)
        fresh = ctx.tool(
            "curve.fit_centerline",
            mask=mask,
            depth=depth,
            intrinsics=K,
            camera_pose=T,
            nodes=int(curve_nodes),
        )
        fpts = fresh.get("points") or []
        if len(fpts) >= 2 and _length(fpts) >= float(whole_rod) * want:
            curve = [[float(v) for v in p] for p in fpts]
            return _out("found", curve, "refit", name, len(curve))
        print(
            f"[track]   {name}: cold fit {_length(fpts) * 1000:.0f}mm of "
            f"{want * 1000:.0f}mm expected; tracking instead",
            flush=True,
        )

        upd = ctx.tool(
            "curve.track_centerline",
            prior=prior_pts,
            mask=mask,
            depth=depth,
            intrinsics=K,
            camera_pose=T,
        )
        vis = np.asarray(upd.get("visibility") or [], dtype=np.float64)
        seen = int((vis > VIS_FLOOR).sum())
        pts = upd.get("points") or []
        if len(pts) < 2 or seen < int(min_visible):
            print(
                f"[track]   {name}: refused, {seen}/{len(vis)} nodes visible "
                f"(need {int(min_visible)})",
                flush=True,
            )
            # Too little of the model in frame to trust the update. The prior is
            # still the best estimate -- a tracker that accepted every frame
            # would walk the model onto whatever happened to be visible.
            continue
        curve = [[float(v) for v in p] for p in pts]
        return _out("found", curve, "tracked", name, seen)

    # Nothing in this pass improved on the prior; hand it back unchanged so the
    # loop keeps the model it had.
    return _out("found", prior_pts, "held", "", 0)
