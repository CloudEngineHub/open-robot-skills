"""CPU-only contract tests for the feature-mating execution skill."""

from __future__ import annotations

import pytest
from gap.testing import FakeContext

_TRAJECTORY = {"planned": True, "trajectory": {"waypoints": [{"positions": [0.0]}]}}


def _pose(z, x=0.4, rotation=None):
    return {
        "position": {"x": x, "y": 0.0, "z": z},
        "rotation": rotation or {"w": 1.0, "x": 0.0, "y": 0.0, "z": 0.0},
    }


def _plan(*waypoints):
    return {
        "waypoints": list(waypoints),
        "world_config": {"meshes": []},
        "attached_object": {"frame": "tcp", "spheres": []},
    }


def _context(**overrides):
    responses = {
        "motion.plan_to_pose": _TRAJECTORY,
        "motion.plan_linear": _TRAJECTORY,
        "robot.execute_trajectory": None,
        "robot.go_to_pose_cartesian": None,
        "robot.move_cartesian_until_contact": None,
        "robot.open_gripper": None,
        "robot.wait_steps": None,
    }
    responses.update(overrides)
    return FakeContext(responses)


@pytest.fixture(scope="module")
def scripts(skills_registry):
    info = skills_registry.get("executing-feature-mating")
    return {name: script.module for name, script in info.canonical_scripts.items()}


def test_canonical_scripts_are_execute_and_release(scripts):
    assert set(scripts) == {"execute_placement_plan", "release_and_retract"}


def test_small_correction_is_servoed_then_reached_waypoint_is_skipped(scripts):
    execute = scripts["execute_placement_plan"]
    ctx = _context(
        **{
            "robot.get_ee_pose": [
                {"pose": _pose(0.505)},  # 5 mm off the first waypoint
                {"pose": _pose(0.4005)},  # 0.5 mm off the second: already there
            ]
        }
    )
    out = execute.run(
        ctx,
        _plan(
            {"pose": _pose(0.5), "mode": "planned_joint"},
            {"pose": _pose(0.4), "mode": "planned_linear"},
        ),
    )
    assert [call.tool for call in ctx.calls] == [
        "robot.get_ee_pose",
        "robot.go_to_pose_cartesian",
        "robot.get_ee_pose",
    ]
    assert ctx.calls[1].kwargs == {"pose": _pose(0.5)}
    assert ctx.call_count("motion.plan_to_pose") == 0
    assert ctx.call_count("robot.execute_trajectory") == 0
    assert out["final_pose"] == _pose(0.4)
    assert set(out) == {"final_pose", "waypoint_count", "waypoint_reports", "fallback_count", "registration_uncertainty_m"}
    assert out["fallback_count"] == 1  # the 5 mm correction was servoed, not planned


def test_reached_gate_is_millimetre_tight(scripts):
    execute = scripts["execute_placement_plan"]
    # 2 mm off is NOT "reached": the correction is still applied.
    ctx = _context(**{"robot.get_ee_pose": {"pose": _pose(0.502)}})
    execute.run(ctx, _plan({"pose": _pose(0.5), "mode": "planned_joint"}))
    assert [call.tool for call in ctx.calls] == ["robot.get_ee_pose", "robot.go_to_pose_cartesian"]
    # A 1 mm offset with a 3 degree twist is not reached either.
    twisted = {"w": 0.99966, "x": 0.0, "y": 0.0, "z": 0.02618}
    ctx = _context(**{"robot.get_ee_pose": {"pose": _pose(0.501, rotation=twisted)}})
    execute.run(ctx, _plan({"pose": _pose(0.5), "mode": "planned_joint"}))
    assert [call.tool for call in ctx.calls] == ["robot.get_ee_pose", "robot.go_to_pose_cartesian"]


def test_large_legs_are_planned_and_tracked_tightly(scripts):
    execute = scripts["execute_placement_plan"]
    ctx = _context(**{"robot.get_ee_pose": {"pose": _pose(0.8)}})
    out = execute.run(
        ctx,
        _plan(
            {"pose": _pose(0.5), "mode": "planned_joint"},
            {
                "pose": _pose(0.4),
                "mode": "planned_linear",
                "allow_goal_contact": True,
                "contact_margin": 0.002,
            },
        ),
    )
    assert [call.tool for call in ctx.calls] == [
        "robot.get_ee_pose",
        "motion.plan_to_pose",
        "robot.execute_trajectory",
        "robot.get_ee_pose",
        "motion.plan_linear",
        "robot.execute_trajectory",
    ]
    joint = ctx.calls[1].kwargs
    assert joint["pose"] == _pose(0.5)
    assert joint["world_config"] == {"meshes": []}
    assert joint["attached_object"] == {"frame": "tcp", "spheres": []}
    assert joint["allow_start_contact"] is False and joint["allow_goal_contact"] is False
    assert joint["contact_margin"] == pytest.approx(0.005)
    linear = ctx.calls[4].kwargs
    assert linear["end"] == _pose(0.4) and linear["orientation"] == "lock"
    assert linear["allow_goal_contact"] is True
    assert linear["contact_margin"] == pytest.approx(0.002)
    for call in ctx.calls_to("robot.execute_trajectory"):
        assert call.kwargs["tolerance"] == pytest.approx(0.002)
        assert call.kwargs["max_steps_per_waypoint"] == 60
    assert out["final_pose"] == _pose(0.4)


def test_planned_linear_never_uses_the_servo_shortcut(scripts):
    execute = scripts["execute_placement_plan"]
    ctx = _context(**{"robot.get_ee_pose": {"pose": _pose(0.505)}})
    execute.run(ctx, _plan({"pose": _pose(0.5), "mode": "planned_linear"}))
    assert [call.tool for call in ctx.calls] == [
        "robot.get_ee_pose",
        "motion.plan_linear",
        "robot.execute_trajectory",
    ]


def test_three_planner_refusals_raise_blocked(scripts):
    execute = scripts["execute_placement_plan"]
    ctx = _context(
        **{
            "robot.get_ee_pose": {"pose": _pose(0.8)},
            "motion.plan_to_pose": {"planned": False},
        }
    )
    with pytest.raises(RuntimeError, match="3 attempts"):
        execute.run(ctx, _plan({"pose": _pose(0.5), "mode": "planned_joint"}))
    assert ctx.call_count("motion.plan_to_pose") == 3
    assert ctx.call_count("robot.execute_trajectory") == 0

    ctx = _context(
        **{
            "robot.get_ee_pose": {"pose": _pose(0.8)},
            "motion.plan_to_pose": [
                {"planned": False},
                {"planned": True, "trajectory": {"waypoints": []}},
                _TRAJECTORY,
            ],
        }
    )
    execute.run(ctx, _plan({"pose": _pose(0.5), "mode": "planned_joint"}))
    assert ctx.call_count("motion.plan_to_pose") == 3
    assert ctx.call_count("robot.execute_trajectory") == 1


def test_cartesian_cross_and_contact_seat_bypass_the_planner(scripts):
    execute = scripts["execute_placement_plan"]
    ctx = _context()
    out = execute.run(
        ctx,
        _plan(
            {"pose": _pose(0.45), "mode": "cartesian_cross"},
            {"pose": _pose(0.40), "mode": "contact_seat"},
        ),
    )
    assert [(call.tool, call.kwargs) for call in ctx.calls] == [
        ("robot.go_to_pose_cartesian", {"pose": _pose(0.45)}),
        ("robot.move_cartesian_until_contact", {"pose": _pose(0.40)}),
    ]
    assert out["final_pose"] == _pose(0.40)


def test_rejects_empty_plan_and_unknown_mode(scripts):
    execute = scripts["execute_placement_plan"]
    with pytest.raises(ValueError, match="no waypoints"):
        execute.run(_context(), {"waypoints": []})
    with pytest.raises(ValueError, match="unknown placement waypoint mode"):
        execute.run(_context(), _plan({"pose": _pose(0.5), "mode": "contact_transition"}))


def _retreat_pose(ctx):
    (call,) = ctx.calls_to("robot.go_to_pose_cartesian")
    return call.kwargs["pose"]


def test_release_retreats_along_axis_by_attached_extent(scripts):
    release = scripts["release_and_retract"]
    ctx = _context()
    attached = {
        "frame": "tcp",
        "spheres": [
            {"center": {"x": 0.0, "y": 0.0, "z": 0.0}, "radius": 0.01},
            {"center": {"x": 0.12, "y": 0.0, "z": 0.0}, "radius": 0.02},  # extent 0.14
        ],
    }
    out = release.run(
        ctx,
        _pose(0.5),
        retreat_axis={"x": 0.0, "y": 2.0, "z": 0.0},
        attached_object=attached,
        relation="loop_over_shaft",
    )
    # The call sequence is the contract; the outputs may GROW (declared in
    # the script's Output) but every key must be one it declares.
    assert out["released"] is True
    assert set(out) == {"released", "retreat_pose", "release_report"}
    assert [call.tool for call in ctx.calls] == [
        "robot.open_gripper",
        "robot.go_to_pose_cartesian",
        "robot.wait_steps",
    ]
    assert ctx.calls[0].kwargs == {"settle_steps": 80}
    assert ctx.calls[2].kwargs == {"steps": 120}
    retreat = _retreat_pose(ctx)
    distance = 0.14 + 0.01
    assert retreat["position"]["x"] == pytest.approx(0.4)
    assert retreat["position"]["y"] == pytest.approx(distance)
    assert retreat["position"]["z"] == pytest.approx(0.5 + 0.5 * distance)
    assert retreat["rotation"] == _pose(0.5)["rotation"]


def test_release_axis_floor_and_inserted_relation_reverse(scripts):
    release = scripts["release_and_retract"]
    # A small attachment still retreats the 8 cm floor along the axis.
    ctx = _context()
    release.run(
        ctx,
        _pose(0.5),
        retreat_axis={"x": 1.0, "y": 0.0, "z": 0.0},
        attached_object={
            "frame": "tcp",
            "spheres": [{"center": {"x": 0.0, "y": 0.0, "z": 0.0}, "radius": 0.01}],
        },
    )
    retreat = _retreat_pose(ctx)
    assert retreat["position"]["x"] == pytest.approx(0.4 + 0.08)
    assert retreat["position"]["z"] == pytest.approx(0.5 + 0.04)
    # Features that went INTO the fixture back out against the axis.
    ctx = _context()
    release.run(
        ctx,
        _pose(0.5),
        retreat_axis={"x": 1.0, "y": 0.0, "z": 0.0},
        attached_object=None,
        relation="shaft_into_aperture",
    )
    retreat = _retreat_pose(ctx)
    assert retreat["position"]["x"] == pytest.approx(0.4 - 0.08)
    assert retreat["position"]["z"] == pytest.approx(0.5 + 0.04)
    # An explicit retract_m raises the floor along the axis.
    ctx = _context()
    release.run(ctx, _pose(0.5), retract_m=0.2, retreat_axis={"x": 1.0, "y": 0.0, "z": 0.0})
    retreat = _retreat_pose(ctx)
    assert retreat["position"]["x"] == pytest.approx(0.4 + 0.2)
    assert retreat["position"]["z"] == pytest.approx(0.5 + 0.1)


def test_release_without_axis_retracts_vertically_with_settle_knobs(scripts):
    release = scripts["release_and_retract"]
    ctx = _context()
    release.run(ctx, _pose(0.5), retract_m=0.16, open_settle_steps=100, settle_steps=200)
    assert [(call.tool, call.kwargs) for call in ctx.calls] == [
        ("robot.open_gripper", {"settle_steps": 100}),
        ("robot.go_to_pose_cartesian", {"pose": _pose(0.5 + 0.16)}),
        ("robot.wait_steps", {"steps": 200}),
    ]
    # Unset retract_m falls back to the 8 cm floor, still straight up.
    ctx = _context()
    release.run(ctx, _pose(0.5))
    assert ctx.calls[0].kwargs == {"settle_steps": 80}
    assert _retreat_pose(ctx) == _pose(0.58)
    assert ctx.calls[2].kwargs == {"steps": 120}
    # The attached extent guard still applies to a vertical retreat.
    ctx = _context()
    release.run(
        ctx,
        _pose(0.5),
        retract_m=0.05,
        attached_object={
            "frame": "tcp",
            "spheres": [{"center": {"x": 0.0, "y": 0.0, "z": -0.1}, "radius": 0.0}],
        },
    )
    assert _retreat_pose(ctx)["position"]["z"] == pytest.approx(0.5 + 0.11)
