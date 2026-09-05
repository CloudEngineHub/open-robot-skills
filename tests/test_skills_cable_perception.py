"""CPU contracts for the three cable perception bundles.

``perceiving-deformable-linear-objects`` (fit-vs-track-vs-hold routing on a
graph-carried prior), ``perceiving-routing-fixtures`` (the OpenCV spool finder
on a synthetic frame, the ``not_found`` exit, and the ``rod`` output that seeds
the loop) and ``verifying-a-cable-route`` (``woven`` / ``short`` on a canned
centreline). Every model-backed tool is a :class:`gap.testing.FakeContext`
can; the only real numerics are the scripts' own.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
from gap.testing import FakeContext

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

IDENTITY = {
    "position": {"x": 0.0, "y": 0.0, "z": 0.0},
    "rotation": {"w": 1.0, "x": 0.0, "y": 0.0, "z": 0.0},
}


def _script(skills_registry, bundle, name):
    return skills_registry.get(bundle).canonical_scripts[name].module


def _line(length_m: float, n: int = 42, x: float = 0.5, z: float = 0.02) -> list[list[float]]:
    """A straight ``n``-node centreline of ``length_m`` along +y."""
    ys = np.linspace(-length_m / 2.0, length_m / 2.0, n)
    return [[x, float(y), z] for y in ys]


def _encode(points) -> str:
    """The bundles' own wire encoding of a centreline."""
    return json.dumps([[round(float(v), 5) for v in p] for p in points])


def _block(h: int = 8, w: int = 8, pose=None) -> dict:
    return {
        "rgb": np.zeros((h, w, 3), dtype=np.uint8),
        "depth": np.ones((h, w), dtype=np.float64),
        "intrinsics": np.eye(3),
        "pose": pose or IDENTITY,
    }


def _fit(points, radius_m: float = 0.0035) -> dict:
    pts = [[float(v) for v in p] for p in points]
    return {
        "points": pts,
        "arclength_m": float(np.sum(np.linalg.norm(np.diff(np.asarray(pts), axis=0), axis=1))),
        "radius_m": radius_m,
        "nodes": len(pts),
        "ordered": True,
    }


def _mask(h: int = 8, w: int = 8) -> np.ndarray:
    return np.ones((h, w), dtype=np.uint8)


# ---------------------------------------------------------------------------
# perceiving-deformable-linear-objects
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def perceive_rod(skills_registry):
    return _script(skills_registry, "perceiving-deformable-linear-objects", "perceive_rod")


def test_rod_bundle_declares_the_loop_contract(skills_registry):
    info = skills_registry.get("perceiving-deformable-linear-objects")
    assert info.kind == "skill"
    assert set(info.meta.exit_conditions) == {"found", "lost"}
    assert info.meta.required_inputs == {"prior": "string", "scene": "string"}
    assert set(info.meta.allowed_tools) == {
        "robot.get_observation",
        "sam3.segment_text",
        "curve.fit_centerline",
        "curve.track_centerline",
    }
    assert info.meta.requires is None or not info.meta.requires.connector
    inputs = info.canonical_scripts["perceive_rod"].schema.inputs
    assert {"prior", "scene", "camera", "index"} <= set(inputs)
    assert inputs["prior"].default == ""


def test_rod_empty_prior_seeds_from_scene_without_a_fit(perceive_rod):
    seed = _line(0.5)
    ctx = FakeContext(tool_responses={"robot.get_observation": {"cameras": {}}})
    out = perceive_rod.run(ctx, prior="", scene=json.dumps({"rod": seed}))
    assert out["route"] == "found"
    assert out["source"] == "seeded"
    assert json.loads(out["rod"]) == json.loads(_encode(seed))
    assert out["visible"] == 42
    assert ctx.call_count("curve.fit_centerline") == 0
    assert ctx.call_count("curve.track_centerline") == 0
    assert ctx.call_count("sam3.segment_text") == 0


def test_rod_first_pass_takes_the_survey_even_with_a_prior(perceive_rod):
    # The graph binds ``prior`` to the survey's rod on pass 0; the model is
    # still born from the survey rather than re-fitted with a hand in the way.
    seed = _line(0.5)
    ctx = FakeContext(tool_responses={"robot.get_observation": {"cameras": {}}})
    out = perceive_rod.run(ctx, prior=_encode(seed), scene=json.dumps({"rod": seed}), index=0)
    assert out["source"] == "seeded"
    assert ctx.call_count("sam3.segment_text") == 0


def test_rod_cold_start_fits_every_candidate_and_takes_the_thinnest(perceive_rod):
    thick = _line(0.55, x=0.4)
    thin = _line(0.50, x=0.5)
    ctx = FakeContext(
        tool_responses={
            "robot.get_observation": {"cameras": {"cable": _block()}},
            "sam3.segment_text": {"masks": [_mask(), _mask()], "scores": [0.9, 0.8]},
            "curve.fit_centerline": [_fit(thick, 0.004), _fit(thin, 0.002)],
        }
    )
    out = perceive_rod.run(ctx, prior="", scene="")
    assert out["route"] == "found"
    assert out["source"] == "initialised"
    assert out["camera"] == "cable"
    assert ctx.call_count("curve.fit_centerline") == 2
    assert ctx.call_count("curve.track_centerline") == 0
    assert json.loads(out["rod"]) == json.loads(_encode(thin))
    assert out["arclength_m"] == pytest.approx(0.5, abs=1e-4)
    assert ctx.calls_to("sam3.segment_text")[0].kwargs["query"] == "thin white cable"


def test_rod_whole_cold_fit_is_a_refit_and_never_tracks(perceive_rod):
    prior = _line(0.5)
    fresh = _line(0.48, x=0.51)
    ctx = FakeContext(
        tool_responses={
            "robot.get_observation": {"cameras": {"cable": _block()}},
            "sam3.segment_text": {"masks": [_mask()], "scores": [0.9]},
            "curve.fit_centerline": _fit(fresh),
        }
    )
    out = perceive_rod.run(ctx, prior=_encode(prior), scene="", index=1)
    assert out["route"] == "found"
    assert out["source"] == "refit"
    assert out["camera"] == "cable"
    assert ctx.call_count("curve.fit_centerline") == 1
    assert ctx.call_count("curve.track_centerline") == 0
    assert json.loads(out["rod"]) == json.loads(_encode(fresh))
    assert out["arclength_m"] == pytest.approx(0.48, abs=1e-4)


def test_rod_short_cold_fit_hands_the_frame_to_the_tracker(perceive_rod):
    prior = _line(0.5)
    fragment = _line(0.2)
    tracked = _line(0.5, x=0.52)
    visibility = [1.0] * 40 + [0.0] * 2
    ctx = FakeContext(
        tool_responses={
            "robot.get_observation": {"cameras": {"cable": _block()}},
            "sam3.segment_text": {"masks": [_mask()], "scores": [0.9]},
            "curve.fit_centerline": _fit(fragment),
            "curve.track_centerline": {"points": tracked, "visibility": visibility},
        }
    )
    out = perceive_rod.run(ctx, prior=_encode(prior), scene="", index=2)
    assert out["route"] == "found"
    assert out["source"] == "tracked"
    assert out["visible"] == 40
    assert ctx.call_count("curve.fit_centerline") == 1
    assert ctx.call_count("curve.track_centerline") == 1
    track = ctx.calls_to("curve.track_centerline")[0].kwargs
    assert np.asarray(track["prior"]).shape == (42, 3)
    assert json.loads(out["rod"]) == json.loads(_encode(tracked))


def test_rod_tracker_with_too_few_visible_nodes_holds_the_prior(perceive_rod):
    prior = _line(0.5)
    prior_text = _encode(prior)
    fragment = _line(0.2)
    ctx = FakeContext(
        tool_responses={
            "robot.get_observation": {"cameras": {"cable": _block()}},
            "sam3.segment_text": {"masks": [_mask()], "scores": [0.9]},
            "curve.fit_centerline": _fit(fragment),
            "curve.track_centerline": {
                "points": _line(0.5, x=0.9),
                "visibility": [1.0, 1.0] + [0.0] * 40,
            },
        }
    )
    out = perceive_rod.run(ctx, prior=prior_text, scene="", index=2)
    assert out["route"] == "found"
    assert out["source"] == "held"
    assert out["rod"] == prior_text
    assert out["arclength_m"] == pytest.approx(0.5, abs=1e-4)
    assert out["visible"] == 0
    assert ctx.call_count("curve.track_centerline") == 1


def test_rod_no_camera_on_a_cold_start_is_lost(perceive_rod):
    ctx = FakeContext(tool_responses={"robot.get_observation": {"cameras": {}}})
    out = perceive_rod.run(ctx, prior="", scene="")
    assert out["route"] == "lost"
    assert out["source"] == "no-camera"
    assert out["rod"] == "null"
    assert out["arclength_m"] == 0.0
    assert ctx.call_count("sam3.segment_text") == 0


def test_rod_accepts_a_centerline_object_as_prior(perceive_rod):
    prior = {"points": _line(0.5), "arclength_m": 0.5, "radius_m": 0.0035, "nodes": 42}
    fresh = _line(0.49)
    ctx = FakeContext(
        tool_responses={
            "robot.get_observation": {"cameras": {"cable": _block()}},
            "sam3.segment_text": {"masks": [_mask()], "scores": [0.9]},
            "curve.fit_centerline": _fit(fresh),
        }
    )
    out = perceive_rod.run(ctx, prior=json.dumps(prior), scene="", index=1)
    assert out["source"] == "refit"


# ---------------------------------------------------------------------------
# perceiving-routing-fixtures
# ---------------------------------------------------------------------------

# A 240x320 camera 1 m above the bench looking straight down: camera +z is
# world -z, camera +y (image down) is world -y.
_H, _W = 240, 320
_FX = 400.0
_K = np.array([[_FX, 0.0, _W / 2.0], [0.0, _FX, _H / 2.0], [0.0, 0.0, 1.0]])
_DOWN_POSE = {
    "position": {"x": 0.0, "y": 0.0, "z": 1.0},
    "rotation": {"w": 0.0, "x": 1.0, "y": 0.0, "z": 0.0},
}
_ORANGE = (255, 128, 0)  # HSV (15, 255, 255) on OpenCV's 0-180 hue
_SPOOL_UP = 0.023  # flange height off the bench [m]
_PLATE_UP = 0.010  # terminal plate height [m]
_DISC_PX = 7


def _disc(rgb, depth, u: int, v: int, height_m: float):
    vv, uu = np.mgrid[0:_H, 0:_W]
    inside = (uu - u) ** 2 + (vv - v) ** 2 <= _DISC_PX**2
    rgb[inside] = _ORANGE
    depth[inside] = 1.0 - height_m


def _bench_frame():
    """Grey bench, three orange spools at flange height, one orange plate."""
    rgb = np.full((_H, _W, 3), 128, dtype=np.uint8)
    depth = np.ones((_H, _W), dtype=np.float64)
    # Drawn out of +y order on purpose: the finder must sort them.
    _disc(rgb, depth, 180, 60, _SPOOL_UP)  # world (+0.049, +0.147)
    _disc(rgb, depth, 140, 180, _SPOOL_UP)  # world (-0.049, -0.147)
    _disc(rgb, depth, 160, 120, _SPOOL_UP)  # world (0, 0)
    _disc(rgb, depth, 240, 120, _PLATE_UP)  # a terminal plate, same colour
    return rgb, depth


def _world_xy(u: int, v: int, height_m: float) -> tuple[float, float]:
    z = 1.0 - height_m
    return ((u - _W / 2.0) * z / _FX, -(v - _H / 2.0) * z / _FX)


def _down_T() -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = np.diag([1.0, -1.0, -1.0])
    T[2, 3] = 1.0
    return T


@pytest.fixture(scope="module")
def perceive_fixtures(skills_registry):
    return _script(skills_registry, "perceiving-routing-fixtures", "perceive_weave_vision")


def test_fixtures_bundle_declares_the_survey_contract(skills_registry):
    info = skills_registry.get("perceiving-routing-fixtures")
    assert set(info.meta.exit_conditions) == {"found", "not_found"}
    assert info.meta.produces_outputs == {
        "scene": "string",
        "stations": "int",
        "rod": "string",
        "body": "string",
    }
    assert set(info.meta.allowed_tools) == {
        "robot.get_observation",
        "sam3.segment_text",
        "curve.fit_centerline",
    }
    assert (info.bundle_dir / "scripts" / "fixtures_cv.py").is_file()


def test_find_spools_orders_three_discs_along_y_and_drops_the_plate(perceive_fixtures):
    rgb, depth = _bench_frame()
    found = perceive_fixtures.fixtures_cv.find_spools(rgb, depth, _K, _down_T())
    assert len(found) == 3
    ys = [p[1] for p in found]
    assert ys == sorted(ys)
    expected = [
        _world_xy(140, 180, _SPOOL_UP),
        _world_xy(160, 120, _SPOOL_UP),
        _world_xy(180, 60, _SPOOL_UP),
    ]
    for (x, y, r), (ex, ey) in zip(found, expected, strict=True):
        assert x == pytest.approx(ex, abs=0.005)
        assert y == pytest.approx(ey, abs=0.005)
        assert 0.010 < r < 0.030


def test_find_spools_respects_the_height_gate_parameter(perceive_fixtures):
    rgb, depth = _bench_frame()
    lenient = perceive_fixtures.fixtures_cv.find_spools(rgb, depth, _K, _down_T(), min_height=0.005)
    assert len(lenient) == 4  # the plate now counts


def _survey_ctx(sam3):
    rgb, depth = _bench_frame()
    block = {"rgb": rgb, "depth": depth, "intrinsics": _K, "pose": _DOWN_POSE}
    return FakeContext(
        tool_responses={
            "robot.get_observation": {
                "cameras": {"overhead": block},
                "instruction": "weave the cable around the spools on alternating sides",
            },
            "sam3.segment_text": sam3,
        }
    )


def test_survey_with_no_rod_mask_is_not_found_after_seeing_the_spools(perceive_fixtures):
    ctx = _survey_ctx({"masks": [], "scores": []})
    out = perceive_fixtures.run(ctx)
    assert out["route"] == "not_found"
    assert out["stations"] == 3
    assert out["scene"] == "{}"
    assert out["rod"] == "null"
    assert out["body"] == "vision"
    # Both prompts are asked before giving up; nothing was fitted.
    assert ctx.call_count("sam3.segment_text") == 2
    assert [c.kwargs["query"] for c in ctx.calls_to("sam3.segment_text")] == [
        "thin white cable",
        "white rod",
    ]
    assert ctx.call_count("curve.fit_centerline") == 0


def test_survey_rod_output_is_the_rod_inside_scene(perceive_fixtures):
    rod_mask = np.zeros((_H, _W), dtype=np.uint8)
    rod_mask[100:104, 20:300] = 1

    def sam3(query, **_):
        if query == "thin white cable":
            return {"masks": [rod_mask], "scores": [0.9]}
        return {"masks": [], "scores": []}

    ctx = _survey_ctx(sam3)
    curve = _line(0.5, x=0.02)
    ctx._responses["curve.fit_centerline"] = _fit(curve, 0.0035)
    out = perceive_fixtures.run(ctx)
    assert out["route"] == "found"
    assert out["stations"] == 3
    scene = json.loads(out["scene"])
    assert json.loads(out["rod"]) == scene["rod"]
    assert out["rod"] == _encode(curve)
    assert scene["sides"] == [-1, 1, -1]
    assert len(scene["seats_x"]) == 3
    for k, (sx, _sy) in enumerate(scene["stations"]):
        # post_radius_m in the trace block is rounded to 5 decimals; the seat
        # itself is computed from the unrounded radius.
        seat = scene["sides"][k] * (scene["measured"]["post_radius_m"][k] + 0.0035)
        assert scene["seats_x"][k] == pytest.approx(sx + seat, abs=1e-5)
    assert scene["measured"]["alternation_in_language"] is True
    assert ctx.call_count("curve.fit_centerline") == 1
    assert ctx.calls_to("curve.fit_centerline")[0].kwargs["nodes"] == 42


# ---------------------------------------------------------------------------
# verifying-a-cable-route
# ---------------------------------------------------------------------------

_STATIONS = [[0.52, -0.13], [0.52, -0.06], [0.52, 0.01]]
_SIDES = [-1, 1, -1]


def _scene() -> str:
    return json.dumps({"stations": _STATIONS, "sides": _SIDES, "seats_x": [0.5, 0.54, 0.5]})


def _threaded(sides) -> list[list[float]]:
    """A centreline running along +y that sits 20 mm on ``sides[k]`` of
    station k while abreast of it, and on the post's own x in between."""
    knots = [[0.52, -0.30, 0.02]]
    for (sx, sy), side in zip(_STATIONS, sides, strict=True):
        x = sx + side * 0.02
        knots.append([x, sy - 0.03, 0.02])
        knots.append([x, sy + 0.03, 0.02])
    knots.append([0.52, 0.30, 0.02])
    return knots


def _verify_ctx(fit_points, prior_track=None, ee=(0.3, 0.0, 0.5)):
    responses = {
        "robot.get_observation": {"cameras": {"cable": _block()}},
        "sam3.segment_text": {"masks": [_mask()], "scores": [0.9]},
        "curve.fit_centerline": _fit(fit_points),
        "robot.get_ee_pose": {
            "pose": {
                "position": {"x": ee[0], "y": ee[1], "z": ee[2]},
                "rotation": {"w": 1.0, "x": 0.0, "y": 0.0, "z": 0.0},
            }
        },
    }
    if prior_track is not None:
        responses["curve.track_centerline"] = {"points": prior_track}
    return FakeContext(tool_responses=responses)


@pytest.fixture(scope="module")
def verify(skills_registry):
    return _script(skills_registry, "verifying-a-cable-route", "verify_weave_vision")


def test_verify_bundle_declares_the_exit_field(skills_registry):
    info = skills_registry.get("verifying-a-cable-route")
    assert set(info.meta.exit_conditions) == {"woven", "short"}
    assert info.meta.required_inputs == {"scene": "string", "prior": "string"}
    assert "robot.get_ee_pose" in info.meta.allowed_tools
    outputs = info.canonical_scripts["verify_weave_vision"].schema.outputs
    assert "exit" in outputs and "route" not in outputs


def test_verify_threaded_route_is_woven(verify):
    ctx = _verify_ctx(_threaded(_SIDES))
    out = verify.run(ctx, scene=_scene(), prior="")
    assert out["exit"] == "woven"
    assert out["success"] is True
    assert out["seated"] == [True, True, True]
    assert out["worst_margin_m"] == pytest.approx(0.02, abs=2e-3)
    assert out["tcp_clear_m"] > 0.05
    assert ctx.call_count("curve.track_centerline") == 0
    assert ctx.call_count("robot.get_ee_pose") == 1


def test_verify_one_station_on_the_wrong_side_is_short(verify):
    ctx = _verify_ctx(_threaded([-1, -1, -1]))
    out = verify.run(ctx, scene=_scene(), prior="")
    assert out["exit"] == "short"
    assert out["success"] is False
    assert out["seated"] == [True, False, True]
    assert out["worst_margin_m"] < 0.0  # signed onto the owed side: wrong-side
    assert "wrong-side" in out["detail"]


def test_verify_hand_on_the_rod_is_short(verify):
    # The hand hovers 10 mm over the rod's run between the anchor and station 0.
    ctx = _verify_ctx(_threaded(_SIDES), ee=(0.52, -0.2, 0.03))
    out = verify.run(ctx, scene=_scene(), prior="")
    assert out["exit"] == "short"
    assert out["seated"] == [True, True, True]
    assert out["tcp_clear_m"] < 0.05


def test_verify_short_cold_fit_tracks_the_prior(verify):
    prior = _threaded(_SIDES)
    fragment = _line(0.2)  # the support hand hides most of the rod
    ctx = _verify_ctx(fragment, prior_track=_threaded(_SIDES))
    out = verify.run(ctx, scene=_scene(), prior=_encode(prior))
    assert out["exit"] == "woven"
    assert ctx.call_count("curve.fit_centerline") == 1
    assert ctx.call_count("curve.track_centerline") == 1
    assert np.asarray(ctx.calls_to("curve.track_centerline")[0].kwargs["prior"]).shape[1] == 3


def test_verify_without_stations_is_short(verify):
    ctx = _verify_ctx(_threaded(_SIDES))
    out = verify.run(ctx, scene="{}", prior="")
    assert out["exit"] == "short"
    assert out["seated"] == []
    assert ctx.call_count("robot.get_ee_pose") == 0
