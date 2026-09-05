"""The weave's layout, from a camera.

The three numbers a routing plan needs -- where the stations are, which side of
each the rod must finish on, and where the rod seats against each -- earned
from one overhead RGB-D frame through a colour threshold, a text-prompted
segmenter and a depth back-projection, rather than read out of a simulator's
scene record.

**Measured on ``weave3``, 2026-08-26**, against the layout the bench declares:

===============================  ==================  ==========
Quantity                         Perceived           Error
===============================  ==================  ==========
station 0 (x, y)                 (0.5200, -0.1298)   0.2 mm
station 1 (x, y)                 (0.5200, -0.0600)   0.0 mm
station 2 (x, y)                 (0.5200, +0.0098)   0.2 mm
spool radius                     12.2 - 12.4 mm      ~0.3 mm
rod length along y               510 mm              +10 mm
===============================  ==================  ==========

Sub-millimetre from 16x16-pixel blobs because a centroid over ~189 mask pixels
is a sub-pixel estimate; the 1.75 mm/px of this rig is the *sampling* pitch, not
the error floor. The rod's +10 mm is mask dilation at the two tips, which is why
:func:`run` measures the rod's line and radius rather than trusting its extent.

**What a camera can and cannot earn.** One key has an honest camera answer,
one is language, and one cannot be earned at all:

``stations``
    Honest. The spools' own masks, back-projected and reduced to a centre.

``sides``
    **Language, not vision.** Which side of a post the rod must finish on is the
    instruction's ("alternating sides"), not a fact about the bench: both sides
    of a spool look identical to a camera, and at the opening frame the rod has
    not been anywhere. Derived here from the alternation the language states and
    a starting side, and that starting side is a convention this file cannot
    measure -- see :data:`FIRST_SIDE`.

``seats_x``
    **Cannot be earned, and this is the one that matters.** A scoring box has
    a centre of its own, 24 mm from the station on the bench this was measured
    on. What a camera can derive is the *physical* seat -- shaft radius plus rod
    radius, 9.75 mm -- which is where the rod actually rests against the spool.
    Those are different places, and the routing planner documents the
    difference as load-bearing: aiming at the box's inner edge "was worth
    ~20 mm of margin and the rod spent all of it". 9.75 mm is inside the scored
    box but near that inner edge. This body returns the physical seat, because
    that is what is true; whether the margin survives is exactly what running
    it measures.

A privileged reader of the same layout must agree on the *keys*, never on
where the values came from.
"""

import json
import os
import sys
from typing import Any, TypedDict

import numpy as np
from gap import NodeContext

# The runtime loads each node script standalone, so a sibling is not importable
# by package path. Put this script's own directory on the path and import it by
# name -- the same thing the runtime does for the entry module.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fixtures_cv  # noqa: E402

REQUIRED_TOOLS = ("sam3.segment_text", "curve.fit_centerline")
"""What this body needs beyond the runtime's own.

``robot.get_observation`` is the runtime's and is always present. The
back-projection is done here in numpy rather than through
``geometry.mask_to_world_points`` on purpose: it is eight lines of intrinsics
arithmetic, and depending on a second bundle to do it would make this body
un-runnable wherever that bundle is missing for a reason that has nothing to do
with perception.
"""

WHY_ABSENT = (
    "a required tool bundle (sam3, curve) is not registered, so the vision survey cannot "
    "run. It is not a task failure and it is not a perception result: install the bundles "
    "(gap skills install sam3 / curve) and put their registry on the skills path."
)

CAMERA = "overhead"
"""The view the stations are measured in.

The corner camera sees the same bench at a three-quarter angle, which is worse
for this specific job in a way worth stating: a spool 25 mm across is resolved
by its *flange disc*, and a disc viewed obliquely projects to an ellipse whose
centroid is not its centre. The overhead view looks straight down the spools'
own axis, so the centroid is the axis.
"""

POST_QUERY = "orange spool"
"""The prompt the text detector is asked for a station, when the colour path
finds nothing.

The colour word is load-bearing. Measured on this render: ``spool`` and
``peg`` return **zero** detections, ``orange spool`` returns exactly three at
0.906; ``cable``, ``wire`` and ``stick`` return zero, ``white rod`` returns one
at 0.836. The detector is keying on the palette rather than on what the object
is, which is a property of a flat-shaded synthetic render rather than of SAM3
-- the same brittleness a rigid-object port hit, where a hook went undetected
until the scene was repainted. Stated here rather than buried because it is the
first thing that breaks when the bench is re-lit or re-coloured.
"""

ROD_QUERIES: tuple[str, ...] = ("thin white cable", "white rod")
"""Every prompt tried for the rod, with the candidates pooled.

**Because one prompt is a calibration and calibrations expire.** This was one
prompt, changed once already: it was ``"white rod"`` until the cable was thinned
from 5 mm radius to 3.5 mm, at which point ``"thin white cable"`` measured 0.96
against ``"white rod"``'s 0.72 on the frame that was failing, and the constant
moved.

Measured again here, episode 0 of four variants as the bench now ships them --
the overhead camera, the 3.5 mm rod, the 0.20 floor:

    layout      "thin white cable"    "white rod"     wins
    weave3            0.135              0.295        rod
    weave4            0.809              0.064        cable
    weave5            0.664              0.027        cable
    board_l2          0.303              0.578        rod

**Neither prompt finds the rod on all four, and the split is two each way.**
Where one wins the other is under the 0.20 floor and returns nothing at all.
Which word wins is a property of one layout at one radius under one light, and
the bench ships thirteen layouts. Picking either alone loses episodes the other
would have found -- and the loss is total, not graceful: ``not_found`` here
routes straight to an abort, so the episode ends at ~180 steps having perceived
its three spools perfectly. That is what ``weave3`` did before both were asked;
with both, it seats the whole weave from the camera alone.

The other prompts tried on weave3, none of them close: ``"white cable"`` 0.085,
``"white string"`` 0.086, ``"cable"`` 0.043, ``"wire"`` 0.042, ``"string"``
0.009, ``"thin rod"`` 0.005.

So both are asked, and the candidates are pooled. Nothing downstream needs them
separated: the rod fit already ranks candidates by SHAPE rather than by score --
mask area over centreline length, the ratio that tells a cable from a cable
welded to a gripper -- for exactly the reason that score cannot separate them.
A prompt that returns nothing costs one detector call and contributes no
candidates; a prompt that returns the rod contributes it, and the fit decides.
"""

POST_SCORE = 0.30

POST_AREA_MIN = 100
"""Smallest mask, in pixels, a detection may have and still be a station.

**A floor, not a ceiling, and getting that backwards cost two runs.** The first
version of this filter rejected detections that were too BIG, on a measurement
of 2,450 px per spool taken from a different camera pose. From where the cable
camera now stands the spools mask 172 to 200 px, and what the terminated layouts
add is not a big blob but a small one: a sliver of a terminal block, 27 to 44 px,
which sailed under a ceiling and was planned as a station at (0.406, -0.410) --
250 mm off the end of the board.

The populations are still cleanly separated, just the other way round. 100 px
sits between them with room on both sides, and it is a floor on a *detection*
rather than a bound on a known fixture, so a scene with different hardware still
works.
"""

POST_RADIUS_CEIL = 0.018
"""**Retired as a gate; kept as a reading.** Largest in-plane radius a detection
had to have to be a station [m].

It was the discriminator, sized between an F1 spool's 12.4 mm flange and a
terminal block's 20-plus. That worked for exactly as long as the bench staged
one fixture. An industrial cleat's cap measures 19 mm and is a station, so the
number now sits between two populations it can no longer separate.

What replaced it is height, which separates both boards: 23 mm for a spool
flange, 43 for a cleat cap, 10 for a terminal plate. The constant stays because
the measurement is still worth having in the log -- it is how the terminals were
first told apart, and a detection whose radius is wildly outside either
population is still worth a look."""

ROD_SCORE = 0.20
"""Detection floors. Low, deliberately: on this render the true detections score
0.9 and 0.84 while the false ones do not appear at all, so a high floor buys
nothing and a low one keeps a dimmer re-lit scene working."""

FIRST_SIDE = -1
"""Which side the first station in +y order wants.

A convention, and the honest limit of what language plus a camera gives you. The
instruction says the crossings alternate; it does not say which side the first
one takes, and at the opening frame nothing in the scene does either -- the rod
lies straight on the robot's side of every station. Wrong here and every
crossing is mirrored, which the verify would catch and no amount of looking
would.
"""

ROD_RADIUS_FLOOR = 0.002
ROD_RADIUS_CEIL = 0.020
"""Bounds on the fitted rod radius [m]. A half-thickness outside these is a
mask that caught the bench or the arm, not a rod, and is refused rather than
propagated into a seat offset."""


class Output(TypedDict):
    route: str
    scene: str
    stations: int
    rod: str
    body: str


def _missing(ctx: NodeContext) -> list[str]:
    """Which required tools this registry does not carry.

    A body that cannot run must fail as a missing dependency and never as a
    perception result. The bundle's ``gap.allowed_tools`` already declares
    these names, so a registry check catches the same absence before the
    graph runs; this probe stays only so an ad-hoc context without a registry
    (a test double) and a live one fail the same way.
    """
    registry = getattr(ctx, "_tool_registry", None)
    if registry is None:
        return []
    return [name for name in REQUIRED_TOOLS if name not in registry]


def _camera(observation: Any, name: str) -> dict[str, Any]:
    """One camera's frame, whichever shape the runtime publishes them in.

    ``observation["cameras"]`` is a dict keyed by camera name on one runtime
    version and a list of name-carrying frames on another. Both are handled
    because the difference is a runtime version, not a decision this body gets
    to make, and pinning one spelling here would make the body fail on a
    checkout whose only sin is a newer gap.
    """
    frames = observation["cameras"] if isinstance(observation, dict) else observation.cameras
    if isinstance(frames, dict):
        if name in frames:
            return frames[name]
        raise ValueError(f"no camera {name!r} in the observation; have {sorted(frames)}")
    for frame in frames or ():
        if str(frame.get("name", "")) == name:
            return frame
    have = [str(f.get("name", "?")) for f in (frames or ())]
    raise ValueError(f"no camera {name!r} in the observation; have {have}")


def _frame(
    block: dict[str, Any], name: str = CAMERA
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """``(rgb, depth, K, world_from_camera)`` out of one camera block."""
    images = block.get("images") if isinstance(block.get("images"), dict) else {}

    def _first(*keys):
        # `or`-chaining these would ask numpy for the truth value of an array.
        for src in (images, block):
            for key in keys:
                v = src.get(key)
                if v is not None:
                    return v
        return None

    rgb = np.asarray(_first("rgb"), dtype=np.uint8)
    depth = _first("depth", "depth_data")
    if depth is None:
        raise ValueError(
            f"camera {name!r} carries no depth. A weave is measured in metres off the bench, "
            "so a colour-only rig cannot run this body."
        )
    depth = np.asarray(depth, dtype=np.float64)
    K = np.asarray(block["intrinsics"], dtype=np.float64).reshape(3, 3)
    # The runtime hands a camera pose as a typed Se3Pose -- ``position`` and
    # ``rotation`` dicts, wxyz -- not as the flat 7-vector an env may publish
    # underneath. Both spellings are read because a body written against one and
    # run on the other fails with "float() argument must be ... not 'dict'",
    # which names the symptom and not the seam.
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
    return rgb, depth, K, T


def _to_world(mask: np.ndarray, depth: np.ndarray, K: np.ndarray, T: np.ndarray) -> np.ndarray:
    """Masked pixels to world metres, in the renderer's +z-forward/+y-down basis."""
    v, u = np.nonzero(np.asarray(mask) > 0)
    if len(u) == 0:
        return np.zeros((0, 3))
    z = depth[v, u]
    ok = np.isfinite(z) & (z > 1.0e-4)
    u, v, z = u[ok], v[ok], z[ok]
    if len(z) == 0:
        return np.zeros((0, 3))
    cam = np.stack(
        [(u - K[0, 2]) * z / K[0, 0], (v - K[1, 2]) * z / K[1, 1], z, np.ones_like(z)], axis=1
    )
    return (T @ cam.T).T[:, :3]


def _detections(ctx: NodeContext, rgb: np.ndarray, query: str, floor: float) -> list[np.ndarray]:
    """Masks for one text prompt, best first, above ``floor``."""
    result = ctx.tool("sam3.segment_text", image=rgb, query=query, max_results=0)
    masks = result.get("masks") or []
    scores = [float(s) for s in (result.get("scores") or [])]
    order = sorted(range(len(scores)), key=lambda i: -scores[i])
    return [np.asarray(masks[i]) for i in order if scores[i] >= floor]


def _hsv(text: str, default: tuple[int, int, int]) -> tuple[int, int, int]:
    """``"h,s,v"`` as three ints; empty text keeps the default."""
    parts = [p.strip() for p in str(text or "").split(",") if p.strip()]
    if len(parts) != 3:
        return default
    return (int(parts[0]), int(parts[1]), int(parts[2]))


def _prompts(text: str) -> tuple[str, ...]:
    """Comma-separated prompts, in the order given, duplicates dropped."""
    out: list[str] = []
    for p in str(text or "").split(","):
        q = p.strip()
        if q and q not in out:
            out.append(q)
    return tuple(out) or ROD_QUERIES


def _encode(points: np.ndarray) -> str:
    """The wire form of a centreline: JSON text of its points, 0.01 mm precision
    -- the same encoding the cable-tracking skill reads and writes, so this
    survey's rod can seed its loop prior byte-for-byte."""
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    if len(pts) < 2:
        return "null"
    return json.dumps([[round(float(v), 5) for v in p] for p in pts])


def run(
    ctx: NodeContext,
    instruction: str = "",
    camera: str = CAMERA,
    post_query: str = POST_QUERY,
    rod_queries: str = ",".join(ROD_QUERIES),
    first_side: int = FIRST_SIDE,
    curve_nodes: int = 42,
    post_score: float = POST_SCORE,
    post_area_min: int = POST_AREA_MIN,
    rod_score: float = ROD_SCORE,
    spool_hsv_lo: str = ",".join(str(v) for v in fixtures_cv.HSV_LO),
    spool_hsv_hi: str = ",".join(str(v) for v in fixtures_cv.HSV_HI),
    spool_min_area: int = fixtures_cv.MIN_AREA,
    fixture_min_height_m: float = fixtures_cv.MIN_HEIGHT,
) -> Output:
    """Survey the routing fixtures and the rod from one overhead RGB-D frame.

    ``rod_queries`` is a comma-separated list of prompts, all asked and pooled;
    ``spool_hsv_lo``/``spool_hsv_hi`` are ``"h,s,v"`` bounds (OpenCV's 0-180
    hue) for the colour path; ``fixture_min_height_m`` separates a station from
    a plate of the same colour.
    """
    absent = _missing(ctx)
    if absent:
        raise RuntimeError(f"vision survey needs {list(absent)}: {WHY_ABSENT}")

    hsv_lo = _hsv(spool_hsv_lo, fixtures_cv.HSV_LO)
    hsv_hi = _hsv(spool_hsv_hi, fixtures_cv.HSV_HI)

    observation: Any = ctx.tool("robot.get_observation")
    # The instruction comes off the OBSERVATION, not from a subgraph input: a
    # subgraph's declared inputs bind to an upstream subgraph's outputs, and the
    # perception slot has nothing upstream of it. The connector already
    # publishes the episode's instruction beside the camera blocks, which is
    # the same place a real cell would carry the command it was given.
    if not str(instruction).strip():
        instruction = str(observation.get("instruction") or "")
    rgb, depth, K, T = _frame(_camera(observation, camera), camera)

    # ---------------------------------------------------------------- posts
    #
    # CLASSICAL FIRST, the network only if it fails. A spool is rigid, one known
    # colour and one known size, bolted to the bench: everything a text-prompted
    # detector is good at solving is already known about it, and re-running a
    # network every frame to rediscover it costs 13-35 s and can return four
    # spools on one frame and three on the next. Scored on the same frames
    # against the peg bodies' own positions:
    #
    #     SAM3, as it stood          3 found   2.8 mm from truth
    #     OpenCV + height gate       3 found   2.9 mm from truth
    #     ... + the bias correction  3 found   0.9 mm from truth
    #
    # The classical path is not a compromise here -- it is more accurate, once
    # the surface-to-axis bias both paths share is taken out, and it is
    # deterministic. See ``fixtures_cv`` for why height, not colour or area, is
    # what separates a spool from a terminal plate.
    posts = fixtures_cv.find_spools(
        rgb,
        depth,
        K,
        T,
        hsv_lo=hsv_lo,
        hsv_hi=hsv_hi,
        min_area=int(spool_min_area),
        min_height=float(fixture_min_height_m),
    )
    for px, py, pr in posts:
        print(f"[posts]   opencv  r={pr * 1000:5.1f} mm  at ({px:.3f}, {py:+.3f})", flush=True)
    if not posts:
        # The colour gate found nothing at all -- a re-lit bench, a repaint, a
        # fixture this file has not seen. Fall back rather than abort: the
        # prompt is slower and noisier but it is not colour-bound.
        print("[posts]   opencv found nothing; falling back to the text detector", flush=True)
        for mask in _detections(ctx, rgb, post_query, float(post_score)):
            # AREA, because it is the strongest separator and it is measured in
            # the image rather than through depth -- a back-projected radius
            # shrinks when a fragment is seen edge-on, which is how a terminal
            # block slipped past the radius test at 9.6 mm.
            area_px = int(np.count_nonzero(np.asarray(mask) > 0))
            if area_px < int(post_area_min):
                print(
                    f"[posts]   ignoring a {area_px} px detection -- too small for a spool",
                    flush=True,
                )
                continue
            pts = _to_world(mask, depth, K, T)
            if len(pts) < 8:
                continue
            centre = np.median(pts, axis=0)
            # The 90th percentile rather than the max: a flange disc's mask
            # carries a few pixels of the bench at its rim, and a max reads
            # those as radius.
            radius = float(np.percentile(np.linalg.norm(pts[:, :2] - centre[:2], axis=1), 90))
            # NOT EVERY ORANGE THING IS A STATION. The terminated layouts put a
            # block at each end of the run, in the same accent colour as the
            # stations -- on ``port3`` the prompt returned FIVE detections for
            # three pegs, and the planner tried to weave around the terminals.
            #
            # SEPARATED BY HEIGHT, not by radius, and that changed when the
            # bench grew a second fixture. The ceiling here was 18 mm, sized
            # between an F1 spool's 12.4 mm flange and a terminal's 20-plus.
            # An industrial cleat's cap measures **19 mm** -- so this fallback
            # would have rejected every station on ``panel_l2`` as "too big for
            # a spool" and aborted the episode, while the classical path in
            # front of it found all five.
            #
            # Height is what actually separates the two populations and it does
            # so for both boards: a spool's flange stands 23 mm off the bench, a
            # cleat's cap 43, and a terminal plate 10. It is the same gate
            # ``fixtures_cv`` uses, against a bench plane measured from the same
            # depth image, so the two paths now disagree about nothing.
            height = float(centre[2]) - fixtures_cv.bench_z(depth, K, T)
            if height < float(fixture_min_height_m):
                print(
                    f"[posts]   ignoring a detection {height * 1000:.0f} mm off the bench "
                    f"at ({centre[0]:.3f}, {centre[1]:+.3f}) -- a terminal, not a station",
                    flush=True,
                )
                continue
            print(
                f"[posts]   accepted {area_px:6d} px  r={radius * 1000:5.1f} mm  at "
                f"({centre[0]:.3f}, {centre[1]:+.3f})",
                flush=True,
            )
            posts.append((float(centre[0]), float(centre[1]), radius))
    if not posts:
        return {"route": "not_found", "scene": "{}", "stations": 0, "rod": "null", "body": "vision"}
    # Ordered along +y, which is the order the layout table is written in and the
    # order the pass ordering below re-derives. Not the detector's score order:
    # that is arbitrary and would permute the sides.
    posts.sort(key=lambda p: p[1])
    stations = [[p[0], p[1]] for p in posts]

    # ------------------------------------------------------------------ rod
    #
    # An ORDERED CENTRELINE, not a bounding box and not a principal axis. The
    # rod is the one thing in this scene that bends, and the axis of a curve
    # through four posts is a chord passing through none of the places a hand
    # goes. ``curve.fit_centerline`` thins the mask, recovers a traversal order
    # through breaks and crossings (TrackDLO's assignment), back-projects and
    # fits a spline in arc length -- the same s in [0, 1] a rod's own asset
    # record publishes its features against.
    #
    # THE LONGEST CABLE-SHAPED CANDIDATE, NOT THE FIRST. This fit seeds the
    # tracker and every plan for the rest of the episode, so taking the
    # top-scoring mask on faith is the single most expensive guess in the graph.
    #
    # The bench is grey, the rod is white, and so are both arms. A prompt for a
    # white rod will sometimes return a blob that has merged the cable with an
    # arm beside it -- confidently scored, and wrong in a way nothing downstream
    # can detect. Measured on ``weave3`` with a 700 mm rod: four episodes seeded
    # at 714, 707, 718 and 694 mm, and two at **963 and 983 mm**. Both inflated
    # ones failed; three of the four honest ones solved.
    #
    # Score cannot separate them and shape can. The fit reports mean radius as
    # mask area over centreline length, which is precisely the ratio that tells
    # a 3.5 mm cable from a cable welded to a gripper, so every candidate is
    # fitted and the radius ceiling decides what is cable-shaped at all.
    def L_of(p):
        return float(np.sum(np.linalg.norm(np.diff(p, axis=0), axis=1)))

    fit, curve, best_len = None, np.zeros((0, 3)), None
    fallback = None  # the longest over-thick candidate, if nothing else fits
    _queries = _prompts(rod_queries)
    _cands = [m for q in _queries for m in _detections(ctx, rgb, q, float(rod_score))]
    print(
        f"[posts]   rod: {len(_cands)} candidate mask(s) above {float(rod_score)} "
        f"over {len(_queries)} prompt(s) {list(_queries)}",
        flush=True,
    )
    for mask in _cands:
        if int(np.count_nonzero(mask)) < 64:
            continue
        trial = ctx.tool(
            "curve.fit_centerline",
            mask=mask,
            depth=depth,
            intrinsics=K,
            camera_pose=T,
            nodes=int(curve_nodes),
        )
        pts = np.asarray(trial.get("points") or [], dtype=np.float64).reshape(-1, 3)
        r = float(trial.get("radius_m") or 0.0)
        if len(pts) < 2 or r <= 0.0:
            continue
        if r > ROD_RADIUS_CEIL:
            # Too thick to be the cable on its own merits, but remembered in
            # case nothing thinner turns up -- see the fallback note below.
            if fallback is None or L_of(pts) > fallback[2]:
                fallback = (trial, pts, L_of(pts))
            continue
        # LONGEST plausible, not thinnest. Radius alone ranks backwards and
        # fails in both directions: a mask merged with an arm traces a longer
        # path, so area-over-length reads SMALLER, and a mask that is only a
        # fragment reads smaller still. Thinness therefore prefers exactly the
        # two things it was meant to reject -- measured on port3, it chose a
        # 148 mm fragment of a 700 mm rod. The radius ceiling already rejects
        # anything too thick; among what survives, the longest saw most of it.
        L = L_of(pts)
        if best_len is None or L > best_len:
            fit, curve, best_len = trial, pts, L
        # A FALLBACK, because this node's failure is the whole episode's.
        #
        # `not_found` here routes straight to an abort, so a filter with no way
        # through is not a filter -- it is an episode killer. Measured: with the
        # thinner cable some draws had every candidate over the radius ceiling,
        # the rod fit returned nothing, and the run ended at ~150 steps having
        # perceived its three spools perfectly. Keeping the longest candidate
        # regardless means a marginal fit is used and reported rather than the
        # episode being thrown away; the ceiling still decides the RANKING, it
        # just no longer decides whether there is an answer at all.
    if fit is None and fallback is not None:
        fit, curve, best_len = fallback
        print(
            f"[posts]   rod: no candidate under the radius ceiling; taking the "
            f"longest anyway ({best_len * 1000:.0f} mm)",
            flush=True,
        )
    if fit is None or len(curve) < 2:
        print("[posts]   rod: no usable candidate at all", flush=True)
        return {
            "route": "not_found",
            "scene": "{}",
            "stations": len(stations),
            "rod": "null",
            "body": "vision",
        }
    print(
        f"[posts]   rod: longest cable-shaped candidate, arclength {best_len * 1000:.0f} mm",
        flush=True,
    )
    # The fitted radius, measured as mask area over centreline length -- both in
    # metres, and neither assuming which way the rod points. The axis-aligned
    # estimate this replaces read 4.14 mm for a 5 mm rod on the arc layout,
    # because there the spread it halved was mostly the curve.
    rod_radius = float(
        np.clip(float(fit.get("radius_m") or 0.0), ROD_RADIUS_FLOOR, ROD_RADIUS_CEIL)
    )

    # ---------------------------------------------------------------- sides
    # The alternation is the instruction's; the phase is first_side's.
    alternating = "alternat" in str(instruction).lower()
    sides = [int(first_side) * (1 if k % 2 == 0 else -1) for k in range(len(stations))]

    # ---------------------------------------------------------------- seats
    # The PHYSICAL seat: the post's own radius plus the rod's, on the named
    # side. See the module docstring on why this differs from a scoring box's
    # centre and which is honest.
    seats_x = [
        float(posts[k][0] + sides[k] * (posts[k][2] + rod_radius)) for k in range(len(posts))
    ]

    rod_text = _encode(curve)
    scene = {
        "stations": stations,
        "sides": sides,
        "seats_x": seats_x,
        "rod": json.loads(rod_text),
        # Carried for the trace, not for the planner: a run that scores badly
        # should be answerable from what the body saw without re-running it.
        "measured": {
            "post_radius_m": [round(p[2], 5) for p in posts],
            "rod_radius_m": round(rod_radius, 5),
            "rod_arclength_m": round(float(fit.get("arclength_m") or 0.0), 4),
            "rod_nodes": int(fit.get("nodes") or 0),
            "rod_ordered": bool(fit.get("ordered")),
            "alternation_in_language": bool(alternating),
        },
    }
    print(
        f"[posts] {len(stations)} spools  sides={sides}  "
        f"seats_x={[round(v, 4) for v in seats_x]}  rod_nodes={len(curve)}",
        flush=True,
    )
    return {
        "route": "found",
        "scene": json.dumps(scene),
        "stations": len(stations),
        # The survey's centreline on its own, so a graph can seed the
        # cable-tracking loop's prior from it without unpacking ``scene``.
        "rod": rod_text,
        "body": "vision",
    }
