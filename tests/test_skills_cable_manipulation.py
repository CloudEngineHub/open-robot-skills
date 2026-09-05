"""FakeContext tests for the three cable MANIPULATION bundles.

- seating-a-cable-crossing: the seat fraction is the densified share of the
  fitted curve inside the station's box on the required side, the plan's own
  ``station`` (not the pass index) picks the box, and the place/settle/look
  call order;
- anchoring-a-free-end: the IK-error gate refuses before anything moves, and
  the happy path streams every leg then closes on the plan's own width and
  hand before dwelling;
- plugging-a-cable-end: an empty prior refuses before any tool call, a
  support planner that names no hand refuses rather than guessing the work
  arm, the port is the block furthest from the anchored end, the carry opens
  the SUPPORT hand, and an unreachable mouth is a failed plug.

The planner behind ``cable.plan_support`` is not exercisable on a CPU: every
test cans its answer.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
from gap.testing import FakeContext

ROOT = Path(__file__).resolve().parents[1] / "skills"

BUNDLES = {
    "seating-a-cable-crossing": ("seat_station", ["robot.wait_steps"]),
    "anchoring-a-free-end": (
        "anchor_far_end",
        ["cable.plan_support", "robot.set_grip", "robot.wait_steps"],
    ),
    "plugging-a-cable-end": (
        "plug_port",
        ["cable.plan_support", "robot.grasp_frame", "robot.set_grip", "robot.wait_steps"],
    ),
}


def _load(bundle: str, script: str):
    path = ROOT / bundle / "scripts" / f"{script}.py"
    spec = importlib.util.spec_from_file_location(f"cable_manipulation_{script}", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def seat():
    return _load("seating-a-cable-crossing", "seat_station")


@pytest.fixture(scope="module")
def anchor():
    return _load("anchoring-a-free-end", "anchor_far_end")


@pytest.fixture(scope="module")
def plug():
    return _load("plugging-a-cable-end", "plug_port")


def _names(ctx: FakeContext) -> list[str]:
    return [c.tool for c in ctx.calls]


# ---------------------------------------------------------------------------
# Canned scene: a bench camera looking straight down from 1 m above the bench.
# ---------------------------------------------------------------------------

CAM_XY = (0.5, 0.2)
CAM_HEIGHT = 1.0
BENCH_Z = 0.75
FOCAL = 200.0
IMAGE = 128
CENTRE = IMAGE / 2


def _camera_frame() -> dict:
    """A frame whose pose maps pixel (u, v) at depth 1 m to the bench."""
    return {
        "name": "cable",
        "rgb": np.zeros((IMAGE, IMAGE, 3), dtype=np.uint8),
        "depth": np.full((IMAGE, IMAGE), CAM_HEIGHT, dtype=np.float64),
        "intrinsics": [[FOCAL, 0.0, CENTRE], [0.0, FOCAL, CENTRE], [0.0, 0.0, 1.0]],
        # A half turn about x: camera +z looks down world -z, camera +y is world -y.
        "pose": {
            "position": {"x": CAM_XY[0], "y": CAM_XY[1], "z": BENCH_Z + CAM_HEIGHT},
            "rotation": {"w": 0.0, "x": 1.0, "y": 0.0, "z": 0.0},
        },
    }


def _observation() -> dict:
    return {"cameras": {"cable": _camera_frame()}}


def _rect_mask(u0: int, u1: int, v0: int, v1: int) -> np.ndarray:
    mask = np.zeros((IMAGE, IMAGE), dtype=np.uint8)
    mask[v0:v1, u0:u1] = 1
    return mask


def _rect_world(u0: int, u1: int, v0: int, v1: int) -> tuple[float, float, float]:
    """Where the camera above puts the median pixel of that rectangle."""
    u_med = (u0 + u1 - 1) / 2.0
    v_med = (v0 + v1 - 1) / 2.0
    return (
        CAM_XY[0] + (u_med - CENTRE) * CAM_HEIGHT / FOCAL,
        CAM_XY[1] - (v_med - CENTRE) * CAM_HEIGHT / FOCAL,
        BENCH_Z,
    )


# ---------------------------------------------------------------------------
# seating-a-cable-crossing
# ---------------------------------------------------------------------------

STATIONS = [[0.30, 0.00], [0.50, 0.20], [0.70, 0.40]]
SIDES = [-1.0, 1.0, -1.0]
SCENE = json.dumps({"stations": STATIONS, "sides": SIDES})
PLACE_LEG = {"stage": "place", "trajectory": {"waypoints": [{"positions": [0.1] * 6}]}}


def _straight_rod(x: float, y0: float, y1: float, knots: int = 21) -> list[list[float]]:
    """A 200 mm rod lying along y at a fixed x, as a polyline of knots."""
    return [[x, y0 + (y1 - y0) * i / (knots - 1), BENCH_Z] for i in range(knots)]


def _densified_share(points, station, side, inner=0.003, outer=0.045, half_y=0.035):
    """The seat fraction by its definition: resample the curve every 2 mm along
    its arclength, keep the samples abreast of the station, count those whose
    offset on the required side lies inside [inner, outer]."""
    pts = np.asarray(points, dtype=np.float64)
    s = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))])
    samples = [np.interp(t, s, pts[:, 0:2].T[0]) for t in np.arange(0.0, s[-1], 0.002)]
    ys = [np.interp(t, s, pts[:, 1]) for t in np.arange(0.0, s[-1], 0.002)]
    inside = 0
    for x, y in zip(samples, ys, strict=True):
        if abs(y - station[1]) > half_y:
            continue
        off = (x - station[0]) * side
        if inner <= off <= outer:
            inside += 1
    return inside / len(samples)


def _seat_ctx(points, *, masks=None):
    mask = _rect_mask(60, 68, 0, IMAGE)
    return FakeContext(
        tool_responses={
            "robot.execute_trajectory": {"ok": True},
            "robot.wait_steps": {"waited": 30},
            "robot.get_observation": _observation(),
            "sam3.segment_text": ({"masks": [mask], "scores": [0.9]} if masks is None else masks),
            "curve.fit_centerline": {"points": points, "ordered": True},
        }
    )


def test_seat_fraction_is_the_densified_share_on_the_required_side(seat):
    station, side = STATIONS[1], SIDES[1]
    # 20 mm out on the required (+x) side: inside the [3, 45] mm box.
    rod = _straight_rod(station[0] + 0.020, station[1] - 0.10, station[1] + 0.10)
    ctx = _seat_ctx(rod)
    plan = json.dumps({"stages": [PLACE_LEG], "station": 1})

    out = seat.run(ctx, plan=plan, scene=SCENE, index=0, arm_id=0)

    expected = _densified_share(rod, station, side)
    assert 0.30 < expected < 0.40, "the canned rod should cross a 70 mm box of a 200 mm run"
    assert out["route"] == "done"
    assert out["fraction"] == pytest.approx(expected, abs=1e-3)
    assert out["fraction"] >= 0.047
    assert out["seated"] is True
    assert out["detail"].startswith("seated")

    # Place, settle, then look -- in that order, on the work arm.
    assert _names(ctx) == [
        "robot.execute_trajectory",
        "robot.wait_steps",
        "robot.get_observation",
        "sam3.segment_text",
        "curve.fit_centerline",
    ]
    place = ctx.calls_to("robot.execute_trajectory")[0].kwargs
    assert place["trajectory"] == PLACE_LEG["trajectory"]
    assert place["arm_id"] == 0
    assert ctx.calls_to("robot.wait_steps")[0].kwargs == {"steps": 30}
    fit = ctx.calls_to("curve.fit_centerline")[0].kwargs
    assert fit["nodes"] == 42
    assert np.asarray(fit["camera_pose"]).shape == (4, 4)
    assert ctx.calls_to("sam3.segment_text")[0].kwargs["query"] == "thin white cable"


def test_seat_rod_on_the_wrong_side_counts_nothing(seat):
    station = STATIONS[1]
    rod = _straight_rod(station[0] - 0.020, station[1] - 0.10, station[1] + 0.10)
    ctx = _seat_ctx(rod)
    plan = json.dumps({"stages": [PLACE_LEG], "station": 1})

    out = seat.run(ctx, plan=plan, scene=SCENE, index=0, arm_id=0)

    assert out["fraction"] == 0.0
    assert out["seated"] is False
    assert out["route"] == "done"
    assert out["detail"] == "short at 0.000, no correction attempted"


def test_seat_uses_the_plans_station_not_the_pass_index(seat):
    station = STATIONS[1]
    rod = _straight_rod(station[0] + 0.020, station[1] - 0.10, station[1] + 0.10)

    # Pass 0 worked station 1: the plan says so, and the box measured is
    # station 1's, where the rod is.
    out = seat.run(
        _seat_ctx(rod),
        plan=json.dumps({"stages": [PLACE_LEG], "station": 1}),
        scene=SCENE,
        index=0,
        arm_id=0,
    )
    assert out["seated"] is True and out["fraction"] > 0.0

    # Without the plan's station the pass index falls through to station 0,
    # whose box the rod is nowhere near.
    out = seat.run(
        _seat_ctx(rod),
        plan=json.dumps({"stages": [PLACE_LEG]}),
        scene=SCENE,
        index=0,
        arm_id=0,
    )
    assert out["seated"] is False and out["fraction"] == 0.0


def test_seat_box_parameters_are_honoured(seat):
    station = STATIONS[1]
    # 20 mm out: inside the default box, outside a box that stops at 15 mm.
    rod = _straight_rod(station[0] + 0.020, station[1] - 0.10, station[1] + 0.10)
    plan = json.dumps({"stages": [PLACE_LEG], "station": 1})

    out = seat.run(_seat_ctx(rod), plan=plan, scene=SCENE, seat_outer=0.015)
    assert out["fraction"] == 0.0

    out = seat.run(_seat_ctx(rod), plan=plan, scene=SCENE, seat_fraction=0.5)
    assert out["seated"] is False and out["fraction"] > 0.0
    assert out["detail"].startswith("short at")


def test_seat_reports_when_it_cannot_measure(seat):
    plan = json.dumps({"stages": [PLACE_LEG], "station": 1})
    ctx = _seat_ctx([], masks={"masks": [], "scores": []})

    out = seat.run(ctx, plan=plan, scene=SCENE)

    assert out == {
        "route": "done",
        "seated": False,
        "fraction": 0.0,
        "detail": "could not measure the seat",
    }
    assert ctx.call_count("curve.fit_centerline") == 0


def test_seat_with_no_place_leg_touches_nothing(seat):
    ctx = _seat_ctx([])
    out = seat.run(ctx, plan=json.dumps({"stages": [{"stage": "move"}]}), scene=SCENE)
    assert out["detail"] == "no place leg"
    assert out["seated"] is False
    assert ctx.calls == []


# ---------------------------------------------------------------------------
# anchoring-a-free-end
# ---------------------------------------------------------------------------

LEGS = [
    {"stage": "enter", "trajectory": {"waypoints": [{"positions": [0.0] * 6}]}},
    {"stage": "take", "trajectory": {"waypoints": [{"positions": [0.2] * 6}]}},
]


def _support_plan(**over) -> dict:
    plan = {
        "arm_id": 1,
        "arm_name": "left",
        "stages": LEGS,
        "worst_grasp_error": 0.0012,
        "close": 0.0045,
        "segment": 40,
        "q_end": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
        "quat": [0.0, 1.0, 0.0, 0.0],
        "grasp_m": [0.4, 0.33, 0.7607],
    }
    plan.update(over)
    return plan


def test_anchor_refuses_a_grasp_past_the_error_limit_without_moving(anchor):
    ctx = FakeContext(
        tool_responses={
            "cable.plan_support": _support_plan(worst_grasp_error=0.08),
            "robot.execute_trajectory": {"ok": True},
            "robot.set_grip": {},
            "robot.wait_steps": {},
        }
    )
    out = anchor.run(ctx)

    assert out == {
        "route": "unanchored",
        "segment": -1,
        "worst_error_m": 0.08,
        "q_support": "null",
        "quat": "null",
    }
    assert _names(ctx) == ["cable.plan_support"]


def test_anchor_refuses_a_plan_with_no_legs(anchor):
    ctx = FakeContext(tool_responses={"cable.plan_support": _support_plan(stages=[])})
    out = anchor.run(ctx)
    assert out["route"] == "unanchored"
    assert _names(ctx) == ["cable.plan_support"]


def test_anchor_streams_every_leg_then_closes_on_the_plans_hand(anchor):
    ctx = FakeContext(
        tool_responses={
            "cable.plan_support": _support_plan(),
            "robot.execute_trajectory": {"ok": True},
            "robot.set_grip": {"arm_id": 1},
            "robot.wait_steps": {"waited": 24},
        }
    )
    out = anchor.run(ctx, approach=0.06)

    assert _names(ctx) == [
        "cable.plan_support",
        "robot.execute_trajectory",
        "robot.execute_trajectory",
        "robot.set_grip",
        "robot.wait_steps",
    ]
    assert ctx.calls_to("cable.plan_support")[0].kwargs == {"approach": 0.06}
    legs = ctx.calls_to("robot.execute_trajectory")
    assert [c.kwargs["trajectory"] for c in legs] == [leg["trajectory"] for leg in LEGS]
    assert all(c.kwargs["arm_id"] == 1 for c in legs)
    # The close is in metres on the plan's own hand -- never a preset, never
    # an arm name.
    assert ctx.calls_to("robot.set_grip")[0].kwargs == {"width_m": 0.0045, "arm_id": 1}
    assert ctx.calls_to("robot.wait_steps")[0].kwargs == {"steps": 24}

    assert out["route"] == "anchored"
    assert out["segment"] == 40
    assert out["worst_error_m"] == pytest.approx(0.0012)
    assert json.loads(out["q_support"]) == [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    assert json.loads(out["quat"]) == [0.0, 1.0, 0.0, 0.0]


# ---------------------------------------------------------------------------
# plugging-a-cable-end
# ---------------------------------------------------------------------------

PLUG_SCENE = json.dumps({"stations": [[0.30, 0.00], [0.50, 0.20]]})
#: The rod runs +y from its anchored end at y = 0.10 to its free end at 0.30.
ROD = [[0.40, 0.10, BENCH_Z], [0.40, 0.20, BENCH_Z], [0.40, 0.30, BENCH_Z]]
NEAR_BLOCK = (0, 60, 68, 128)  # near the anchored end
FAR_BLOCK = (68, 128, 68, 128)  # further out in +x: the port


def _plug_ctx(*, support, blocks=(NEAR_BLOCK, FAR_BLOCK), go_to_pose=None, ee=None):
    masks = [_rect_mask(*b) for b in blocks]
    ee_pose = ee or {"position": {"x": 0.41, "y": 0.33, "z": 0.77}}
    return FakeContext(
        tool_responses={
            "robot.get_observation": _observation(),
            "sam3.segment_text": {"masks": masks, "scores": [0.95] * len(masks)},
            "cable.plan_support": support,
            "robot.get_ee_pose": {"pose": ee_pose},
            "robot.grasp_frame": {
                "rotation": {"w": 0.0, "x": 1.0, "y": 0.0, "z": 0.0},
                "rotation_flipped": {"w": 1.0, "x": 0.0, "y": 0.0, "z": 0.0},
            },
            "robot.go_to_pose": go_to_pose if go_to_pose is not None else {"ok": True},
            "robot.set_grip": {"arm_id": 1},
            "robot.wait_steps": {"waited": 30},
        }
    )


@pytest.mark.parametrize("prior", ["", "null", json.dumps([[0.4, 0.1, BENCH_Z]])])
def test_plug_with_no_prior_refuses_before_any_tool_call(plug, prior):
    ctx = _plug_ctx(support=_support_plan())
    out = plug.run(ctx, scene=PLUG_SCENE, prior=prior, arm_id=0)
    assert out == {
        "exit": "unplugged",
        "detail": "no centreline or no stations",
        "port": "null",
    }
    assert ctx.calls == []


def test_plug_with_no_stations_refuses_before_any_tool_call(plug):
    ctx = _plug_ctx(support=_support_plan())
    out = plug.run(ctx, scene="{}", prior=json.dumps(ROD))
    assert out["exit"] == "unplugged"
    assert ctx.calls == []


def test_plug_refuses_when_the_support_planner_names_no_hand(plug):
    ctx = _plug_ctx(support={})
    out = plug.run(ctx, scene=PLUG_SCENE, prior=json.dumps(ROD), arm_id=0)

    assert out["exit"] == "unplugged"
    assert out["detail"] == "no support arm"
    # The port was found and is reported even though nothing moved.
    port = json.loads(out["port"])
    assert port == pytest.approx(_rect_world(*FAR_BLOCK), abs=1e-3)
    # Nothing was commanded with the work arm.
    assert _names(ctx) == ["robot.get_observation", "sam3.segment_text", "cable.plan_support"]


def test_plug_reads_a_centerline_object_prior_like_the_point_list(plug):
    """The perception bundles emit ``rod`` as the bare point list but accept a
    whole ``Centerline`` object on input; the plug reads both the same way."""
    as_list = _plug_ctx(support={})
    as_object = _plug_ctx(support={})
    out_list = plug.run(as_list, scene=PLUG_SCENE, prior=json.dumps(ROD), arm_id=0)
    out_object = plug.run(
        as_object,
        scene=PLUG_SCENE,
        prior=json.dumps({"points": ROD, "ordered": True, "arclength_m": 0.2}),
        arm_id=0,
    )
    assert out_object == out_list
    assert out_object["detail"] == "no support arm"
    assert _names(as_object) == _names(as_list)


def test_plug_carries_with_the_support_hand_and_opens_it(plug):
    ctx = _plug_ctx(support=_support_plan())
    out = plug.run(ctx, scene=PLUG_SCENE, prior=json.dumps(ROD), arm_id=0, grip_open_m=0.05)

    assert out["exit"] == "plugged"
    assert out["detail"] == "free end carried into the port"
    port = _rect_world(*FAR_BLOCK)
    assert json.loads(out["port"]) == pytest.approx(port, abs=1e-3)

    assert _names(ctx) == [
        "robot.get_observation",
        "sam3.segment_text",
        "cable.plan_support",
        "robot.get_ee_pose",
        "robot.grasp_frame",
        "robot.go_to_pose",  # lift
        "robot.go_to_pose",  # across
        "robot.go_to_pose",  # over
        "robot.go_to_pose",  # mouth
        "robot.set_grip",
        "robot.wait_steps",
        "robot.go_to_pose",  # retreat
        "robot.get_ee_pose",
    ]
    # Every waypoint goes to the SUPPORT hand with the wrist the hand composed.
    moves = ctx.calls_to("robot.go_to_pose")
    assert all(c.kwargs["arm_id"] == 1 for c in moves)
    assert all(
        c.kwargs["pose"]["rotation"] == {"w": 0.0, "x": 1.0, "y": 0.0, "z": 0.0} for c in moves
    )
    surface = 0.7607  # the planner's grasp height, not the curve's z
    lift, across, over, mouth = (c.kwargs["pose"]["position"] for c in moves[:4])
    assert lift == pytest.approx({"x": 0.41, "y": 0.33, "z": surface + 0.070})
    assert across["z"] == pytest.approx(surface + 0.070)
    assert across["y"] == pytest.approx(max(0.33, port[1] + 0.06))
    assert over["x"] == pytest.approx(mouth["x"])
    # The mouth: 4 mm short of the port's centre along x, at the working height.
    assert mouth == pytest.approx({"x": port[0] - 0.004, "y": port[1], "z": surface}, abs=1e-3)
    # The jaws that open are the support hand's, at the width asked for.
    assert ctx.calls_to("robot.set_grip")[0].kwargs == {"width_m": 0.05, "arm_id": 1}
    assert ctx.calls_to("robot.wait_steps")[0].kwargs == {"steps": 30}


def test_plug_treats_an_unreachable_mouth_as_a_failed_plug(plug):
    surface = 0.7607

    def _refuse_low(**kwargs):
        if kwargs["pose"]["position"]["z"] < surface + 0.05:
            raise RuntimeError("IK refused")
        return {"ok": True}

    ctx = _plug_ctx(support=_support_plan(), go_to_pose=_refuse_low)
    out = plug.run(ctx, scene=PLUG_SCENE, prior=json.dumps(ROD))

    assert out["exit"] == "unplugged"
    assert out["detail"] == "the port mouth was unreachable"
    # Three staging poses at carry height, then the mouth tried at three heights.
    assert ctx.call_count("robot.go_to_pose") == 6
    assert ctx.call_count("robot.set_grip") == 0


def test_plug_ignores_blocks_below_the_area_floor(plug):
    ctx = _plug_ctx(support=_support_plan(), blocks=((0, 40, 0, 40),))  # 1600 px
    out = plug.run(ctx, scene=PLUG_SCENE, prior=json.dumps(ROD))
    assert out == {"exit": "unplugged", "detail": "no block in view", "port": "null"}

    ctx = _plug_ctx(support=_support_plan(), blocks=((0, 40, 0, 40),))
    out = plug.run(ctx, scene=PLUG_SCENE, prior=json.dumps(ROD), block_min_px=1000)
    assert out["exit"] == "plugged"


# ---------------------------------------------------------------------------
# Loader-level
# ---------------------------------------------------------------------------


def test_cable_manipulation_bundles_discover_with_their_connector_needs(skills_registry):
    for bundle, (script, connector) in BUNDLES.items():
        info = skills_registry.get(bundle)
        assert info.kind == "skill"
        assert info.namespace == "skills"
        assert script in info.canonical_scripts, f"{bundle}: {script} not a canonical script"
        assert callable(info.canonical_scripts[script].module.run)
        assert info.meta.requires is not None
        assert sorted(info.meta.requires.connector) == sorted(connector)
        # Every connector requirement is also an allowed tool, and no bundle
        # tool is a connector requirement.
        assert set(connector) <= set(info.meta.allowed_tools)
        reads_sim = any(t.split(".")[0] in ("sim", "cable") for t in info.meta.allowed_tools)
        assert ("sim-only" in info.meta.tags) == reads_sim, bundle
