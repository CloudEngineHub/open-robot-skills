"""Did the weave hold, judged from a camera rather than from an evaluator?

A ground-truth graph can ask its simulator whether the task succeeded, and
that is the right answer for a ground-truth graph and an impossible one for a
policy that is supposed to know when it is finished. A real cell has no
evaluator to ask.

**What a routing goal scores, and what this reproduces.** The goal this was
written against is a ``worst_of`` over two clauses: every station's seat holds
rod, *and* the hand is clear of the rod at rest. Both are visible. The seat
clause is ``containment cable seatN >= 0.047`` -- a fraction of the rod's
segments inside a box -- and what a camera can measure instead is whether rod
material lies on the required side of each post, within the box's own reach.
The release clause is ``tcp-distance > 0.05`` with ``speed <= 0.05``, and the
first half of that is a distance between two things this body already
localises.

**This is a judgement, not a score.** It deliberately does not try to reproduce
``containment``'s exact fraction: matching an evaluator's arithmetic from
outside is how a graph ends up agreeing with a number it cannot see and
disagreeing with the thing the number was about. What it answers is the
question a policy actually needs -- *is every station seated, and am I off the
rod* -- and it reports its own margins so a disagreement with the evaluator is
diagnosable rather than mysterious.

A privileged verdict and this one agree on the *keys* and the *exits*, never
on where the verdict came from.
"""

import json
from typing import Any, TypedDict

import numpy as np
from gap import NodeContext

REQUIRED_TOOLS = ("sam3.segment_text", "curve.fit_centerline", "curve.track_centerline")

WHY_ABSENT = (
    "a required tool bundle (sam3, curve) is not registered, so the vision verify cannot run. "
    "Install the bundles (gap skills install sam3 / curve) and put their registry on the "
    "skills path."
)

CAMERA = "cable"
"""The rod is read from the cable camera, not the bench overhead.

By the time this node runs the hand has retreated, so the view is clear and the
final shape can simply be measured -- at 0.50 mm/px against the overhead's 1.75,
which is the difference between resolving a 10 mm rod and inferring one. The
stations are NOT re-detected here; they come from ``scene``. Re-detecting them
was a second, independent chance to mis-order the posts, and the verdict has to
be about the same stations the plan was built on.
"""
ROD_QUERY = "thin white cable"
"""The same prompt the survey uses, and the same colour-word caveat applies --
see ``perceiving-routing-fixtures``. Kept as a separate constant rather than
imported across bundles: a graph's scripts are loaded individually and a
cross-import between two of them is a coupling the runtime does not promise.

There is no post prompt here; the stations come from ``scene``."""

ROD_SCORE = 0.20

SEAT_REACH = 0.045
"""How far from a post, along its named side, rod material still counts as
seated [m].

The scored box's own outer edge. The goal draws each seat as
``(-0.045 .. -0.003)`` in x about its post, so 45 mm is where the box stops
rather than a tolerance invented here. Rod beyond it is not in the box, whatever
it looks like.
"""

SEAT_INNER = 0.003
"""How close to the post rod material stops counting [m]. The scored box's inner
edge: the goal draws each seat as ``0.003 .. 0.045`` out from its post, so
material pressed against the post itself is not in the box."""

SEAT_BAND = 0.035
"""Half-width, along the run, of the strip abreast of a post inside which rod
material is judged for that post [m]. The scored box's own extent along y."""

SEAT_FRACTION = 0.047
"""Fraction of the centreline that must lie in a seat box, which is the
goal's own ``(>= (containment cable seatN) 0.047)`` verbatim.

**This node used to answer a different question and get it wrong.** It counted
six raw MASK PIXELS in the band and called that seated -- six of several hundred
back-projected points, about 1.2%, against a clause asking for 4.7% of the rod.
Worse, a few hundred pixels of a thin rod are a couple of millimetres of it, so
the test could pass on a stray fleck of mask. Measured: on ``weave3`` episode 0
it reported ``seated=[True, True, True] -> woven`` while the evaluator scored
0.67, which is the one disagreement in twelve episodes and the reason this was
rewritten.

Counting NODES of the fitted centreline instead makes the two comparable: 42
nodes is the rod's own discretisation, so 0.047 of them is ~2 nodes or ~24 mm of
rod, and the clause means here what it means in the goal. The module
docstring's warning against reproducing an evaluator's arithmetic still stands
for the *geometry* -- what is copied here is the threshold, not a reimplementation
of ``containment`` over a representation this node cannot see."""

CURVE_NODES = 42

WHOLE_ROD = 0.80
"""Fraction of the prior's arclength a fresh fit must reach to be used here.

The same number, for the same reason, as the cable-tracking skill's
``WHOLE_ROD`` -- stated again rather than imported, because a graph's node
scripts are loaded individually and a cross-import between two of them is a
coupling the runtime does not promise."""

SAMPLE_STEP = 0.002
"""Spacing the fitted centreline is resampled to before the seat test [m].

**The fraction has to be measured on the CURVE, not on its 42 knots.** 42 nodes
over a 500 mm rod is 12 mm apart, and a seat box is only 70 mm of y, so a
station gets about six samples -- coarse enough that a box can fall between two
of them and be reported as holding no rod at all. Measured on ``weave_arc``:
``why=['no-rod', 'seated', 'wrong-side', 'seated']`` against an evaluator that
scored 3 of 4, the ``no-rod`` station being an artefact of the knot spacing
rather than anything about the cable.

Resampling the same polyline every 2 mm changes no geometry -- the curve is
already a piecewise-linear object and this only reads it more finely -- but it
turns the count into a faithful estimate of what fraction of the cable lies in
the box, which is what :data:`SEAT_FRACTION` is a threshold on. The fraction is
scale-free, so the threshold does not move with the sample count.
"""

CLEAR_M = 0.05
"""How far the hand must be off the nearest rod point [m]. The goal's own
``tcp-distance`` clause."""


class Output(TypedDict):
    exit: str
    success: bool
    seated: list[bool]
    worst_margin_m: float
    tcp_clear_m: float
    detail: str


def _missing(ctx: NodeContext) -> list[str]:
    # The bundle's ``gap.allowed_tools`` already declares these names, so a
    # registry check catches the same absence before the graph runs; this
    # probe stays only so a context without a registry (a test double) and a
    # live one fail the same way.
    registry = getattr(ctx, "_tool_registry", None)
    if registry is None:
        return []
    return [name for name in REQUIRED_TOOLS if name not in registry]


def _camera(observation: Any, name: str) -> dict[str, Any]:
    frames = observation["cameras"] if isinstance(observation, dict) else observation.cameras
    if isinstance(frames, dict):
        if name in frames:
            return frames[name]
        raise ValueError(f"no camera {name!r}; have {sorted(frames)}")
    for frame in frames or ():
        if str(frame.get("name", "")) == name:
            return frame
    raise ValueError(f"no camera {name!r}; have {[f.get('name') for f in (frames or ())]}")


def _frame(block: dict[str, Any], name: str = CAMERA):
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
        raise ValueError(f"camera {name!r} carries no depth; a route is verified in metres")
    depth = np.asarray(depth, dtype=np.float64)
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
    return rgb, depth, K, T


def _points(text: str | None) -> list[list[float]]:
    """The centreline points carried by ``text``: a JSON list of ``[x, y, z]``
    triples, or a JSON ``Centerline`` object whose ``points`` are that list."""
    if not text:
        return []
    raw = json.loads(text)
    if isinstance(raw, dict):
        raw = raw.get("points")
    pts = [[float(v) for v in p] for p in (raw or [])]
    return pts if len(pts) >= 2 else []


def _arclength(points) -> float:
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    return float(np.sum(np.linalg.norm(np.diff(pts, axis=0), axis=1))) if len(pts) > 1 else 0.0


def _densify(points: np.ndarray, step: float = SAMPLE_STEP) -> np.ndarray:
    """The same polyline, resampled to roughly uniform ``step`` spacing."""
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    if len(pts) < 2:
        return pts
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    if s[-1] <= 0.0:
        return pts
    want = np.arange(0.0, s[-1], max(step, 1.0e-4))
    return np.stack([np.interp(want, s, pts[:, k]) for k in range(3)], axis=1)


def _masks(ctx: NodeContext, rgb, query: str, floor: float) -> list[np.ndarray]:
    out = ctx.tool("sam3.segment_text", image=rgb, query=query, max_results=0)
    masks = out.get("masks") or []
    scores = [float(s) for s in (out.get("scores") or [])]
    order = sorted(range(len(scores)), key=lambda i: -scores[i])
    return [np.asarray(masks[i]) for i in order if scores[i] >= floor]


def run(
    ctx: NodeContext,
    scene: str = "",
    prior: str = "",
    camera: str = CAMERA,
    rod_query: str = ROD_QUERY,
    rod_score: float = ROD_SCORE,
    seat_reach: float = SEAT_REACH,
    seat_inner: float = SEAT_INNER,
    seat_band: float = SEAT_BAND,
    seat_fraction: float = SEAT_FRACTION,
    clear_m: float = CLEAR_M,
    whole_rod: float = WHOLE_ROD,
    curve_nodes: int = CURVE_NODES,
    sample_step: float = SAMPLE_STEP,
) -> Output:
    """Judge the finished route from one frame of the cable camera.

    ``scene`` is the survey JSON (its ``stations`` and ``sides`` are the plan's);
    ``prior`` is the centreline the loop carries (JSON text, empty for none),
    used only to carry an occluded end through when the cold fit comes back
    short. The router field is ``exit``.
    """
    absent = _missing(ctx)
    if absent:
        raise RuntimeError(f"vision verify needs {absent}: {WHY_ABSENT}")

    seen = json.loads(scene or "{}")
    sides = [int(v) for v in seen.get("sides", [])]
    # The stations the PLAN was built on, not a fresh detection. See CAMERA.
    posts = [(float(x), float(y)) for x, y in (seen.get("stations") or [])]

    observation: Any = ctx.tool("robot.get_observation")
    rgb, depth, K, T = _frame(_camera(observation, camera), camera)

    # The rod's final shape. The WORK hand has retreated by now, but the SUPPORT
    # hand has not -- it is still holding the far end, and it masks the rod there.
    #
    # A cold fit therefore comes back truncated at that end, and a truncated fit
    # does not report that it is truncated: it reports a station with no cable at
    # it. Measured on ``weave_arc``, station 0 -- the most -y station, the one
    # nearest the support grip -- came back ``no-rod`` in every single episode,
    # including ones the evaluator scored 3 of 4.
    #
    # So the same rule the cable-tracking skill uses applies here, for the same
    # reason and against the same threshold: take the fresh fit when it is a
    # whole rod, and otherwise track the prior onto this frame, which carries
    # the occluded end through instead of dropping it.
    prior_pts = _points(prior)
    rod = np.zeros((0, 3))
    for mask in _masks(ctx, rgb, rod_query, float(rod_score)):
        fit = ctx.tool(
            "curve.fit_centerline",
            mask=mask,
            depth=depth,
            intrinsics=K,
            camera_pose=T,
            nodes=int(curve_nodes),
        )
        pts = np.asarray(fit.get("points") or [], dtype=np.float64).reshape(-1, 3)
        want = _arclength(prior_pts) if prior_pts else 0.0
        if len(pts) >= 2 and (not want or _arclength(pts) >= float(whole_rod) * want):
            rod = _densify(pts, float(sample_step))
            break
        if prior_pts:
            upd = ctx.tool(
                "curve.track_centerline",
                prior=prior_pts,
                mask=mask,
                depth=depth,
                intrinsics=K,
                camera_pose=T,
            )
            tpts = np.asarray(upd.get("points") or [], dtype=np.float64).reshape(-1, 3)
            if len(tpts) >= 2:
                print(
                    f"[verify] cold fit {_arclength(pts) * 1000:.0f}mm of "
                    f"{want * 1000:.0f}mm expected; tracked instead",
                    flush=True,
                )
                rod = _densify(tpts, float(sample_step))
                break
        if len(pts) >= 2:
            rod = _densify(pts, float(sample_step))
            break
    if not posts or len(rod) < 2 or len(sides) < len(posts):
        return {
            "exit": "short",
            "success": False,
            "seated": [],
            "worst_margin_m": 0.0,
            "tcp_clear_m": 0.0,
            "detail": f"{len(posts)} stations, {len(rod)} rod nodes, {len(sides)} sides",
        }

    # The goal's clause is a FRACTION of the rod, so the threshold is one too.
    # Taken over the densified samples, which is why it means what it says.
    need = max(1, int(np.ceil(float(seat_fraction) * len(rod))))

    seated: list[bool] = []
    margins: list[float] = []
    reasons: list[str] = []
    for k, (sx, sy) in enumerate(posts):
        side = sides[k] if k < len(sides) else 0
        # Rod material abreast of this post, on the side the crossing owed.
        near = rod[np.abs(rod[:, 1] - sy) <= float(seat_band)]
        offset = (near[:, 0] - sx) * side if len(near) else np.zeros(0)
        inside = (
            near[(offset >= float(seat_inner)) & (offset <= float(seat_reach))]
            if len(near)
            else np.zeros((0, 3))
        )
        ok = len(inside) >= need
        seated.append(bool(ok))
        # Three outcomes, and the exit alone cannot tell them apart: seated, on
        # the wrong side, or no rod near this station at all. The margin carries
        # the first two -- signed onto the required side, so negative means
        # wrong-side -- and NaN marks the third. Worth the extra case: a previous
        # revision of this file read a wrong-side margin as a near-miss and a
        # whole build was spent pressing harder on a cable that was on the far
        # side of the post it should have crossed.
        margins.append(float(np.max(offset)) if len(offset) else float("nan"))
        if ok:
            reasons.append("seated")
        elif not len(offset):
            reasons.append("no-rod")
        elif float(np.max(offset)) < float(seat_inner):
            reasons.append("wrong-side")
        else:
            reasons.append("outside-box")

    ee = ctx.tool("robot.get_ee_pose")
    pos = ee.get("pose", ee).get("position", {})
    tcp = np.array([float(pos.get(k, 0.0)) for k in ("x", "y", "z")])
    clear = float(np.min(np.linalg.norm(rod - tcp, axis=1)))

    success = all(seated) and clear > float(clear_m)
    finite = [m for m in margins if np.isfinite(m)]
    worst = min(finite) if finite else float("nan")
    print(
        f"[verify] seated={seated} why={reasons} "
        f"worst_margin={worst * 1000:.1f}mm need={need}/{len(rod)} nodes "
        f"hand_clear={clear * 1000:.0f}mm -> {'woven' if success else 'short'}",
        flush=True,
    )
    return {
        "exit": "woven" if success else "short",
        "success": bool(success),
        "seated": seated,
        "worst_margin_m": round(float(worst), 4) if np.isfinite(worst) else 0.0,
        "tcp_clear_m": round(clear, 4),
        "detail": f"{sum(seated)}/{len(seated)} seated ({need} of {len(rod)} nodes each): "
        f"{', '.join(reasons)}; hand {clear * 1000:.0f} mm off the rod",
    }
