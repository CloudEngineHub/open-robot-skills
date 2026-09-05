"""CPU-only contract tests for the held-object motion executors."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).parents[1] / "skills/executing-held-object-motion/scripts"


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


class Context:
    def __init__(self, plan_responses=None):
        self.calls = []
        self.plan_responses = list(plan_responses or [])

    def tool(self, name, **inputs):
        self.calls.append((name, inputs))
        if name.startswith("motion.plan_"):
            if self.plan_responses:
                return self.plan_responses.pop(0)
            return {
                "planned": True,
                "trajectory": {
                    "waypoints": [
                        {"positions": [0.0]},
                        {"positions": [1.0]},
                    ]
                },
            }
        return None


def _pose(x):
    return {
        "position": {"x": x, "y": 0.0, "z": 0.5},
        "rotation": {"w": 1.0, "x": 0.0, "y": 0.0, "z": 0.0},
    }


def _plan(*waypoints):
    return {
        "waypoints": list(waypoints),
        "world_config": {"meshes": []},
        "attached_object": {"frame": "tcp", "spheres": []},
    }


def test_executes_waypoints_in_order_without_gripper_calls():
    module = _load("execute_reorientation")
    ctx = Context()
    result = module.run(
        ctx,
        _plan(
            {"pose": _pose(0.1), "mode": "planned_joint"},
            {"pose": _pose(0.2), "mode": "planned_linear"},
        ),
    )

    assert [name for name, _ in ctx.calls] == [
        "motion.plan_to_pose",
        "robot.execute_trajectory",
        "motion.plan_linear",
        "robot.execute_trajectory",
    ]
    assert result["final_pose"] == _pose(0.2)
    assert all("gripper" not in name for name, _ in ctx.calls)
    for name, inputs in ctx.calls:
        if name.startswith("motion.plan_"):
            assert inputs["world_config"] == {"meshes": []}
            assert inputs["attached_object"] == {"frame": "tcp", "spheres": []}
            assert inputs["allow_start_contact"] is False
            assert inputs["allow_goal_contact"] is False
            assert inputs["contact_margin"] == pytest.approx(0.005)
        if name == "robot.execute_trajectory":
            assert inputs["max_steps_per_waypoint"] == 60
            assert "tolerance" not in inputs
    linear = ctx.calls[2][1]
    assert linear["end"] == _pose(0.2) and linear["orientation"] == "lock"


def test_contact_transition_is_a_plain_cartesian_move_and_flags_forward():
    module = _load("execute_reorientation")
    ctx = Context()
    module.run(
        ctx,
        _plan(
            {"pose": _pose(0.1), "mode": "contact_transition"},
            {
                "pose": _pose(0.2),
                "mode": "planned_joint",
                "use_attachment": False,
                "allow_start_contact": True,
                "contact_margin": 0.002,
            },
        ),
    )
    assert [name for name, _ in ctx.calls] == [
        "robot.go_to_pose_cartesian",
        "motion.plan_to_pose",
        "robot.execute_trajectory",
    ]
    assert ctx.calls[0][1] == {"pose": _pose(0.1)}
    planned = ctx.calls[1][1]
    assert planned["attached_object"] is None
    assert planned["allow_start_contact"] is True
    assert planned["contact_margin"] == pytest.approx(0.002)


def test_time_scale_resamples_the_joint_path_only():
    module = _load("execute_reorientation")
    ctx = Context()
    plan = _plan({"pose": _pose(0.1), "mode": "planned_joint"})
    plan["time_scale"] = 3.0
    module.run(ctx, plan)
    executed = ctx.calls[1][1]["trajectory"]["waypoints"]
    assert len(executed) == 4
    assert executed[0]["positions"] == [0.0] and executed[-1]["positions"] == [1.0]


def test_reorientation_retries_then_raises_on_refusal():
    module = _load("execute_reorientation")
    ctx = Context(plan_responses=[{"planned": False}] * 3)
    with pytest.raises(RuntimeError, match="reorientation failed at waypoint 0"):
        module.run(ctx, _plan({"pose": _pose(0.1), "mode": "planned_joint", "max_attempts": 3}))
    assert [name for name, _ in ctx.calls] == ["motion.plan_to_pose"] * 3

    ctx = Context(plan_responses=[{"planned": False}])
    with pytest.raises(RuntimeError):
        module.run(ctx, _plan({"pose": _pose(0.1), "mode": "planned_joint"}))
    assert [name for name, _ in ctx.calls] == ["motion.plan_to_pose"]


def test_reorientation_rejects_empty_plan_and_unknown_mode():
    module = _load("execute_reorientation")
    with pytest.raises(ValueError):
        module.run(Context(), {"waypoints": []})
    with pytest.raises(ValueError, match="unsupported"):
        module.run(Context(), _plan({"pose": _pose(0.1), "mode": "cartesian_cross"}))


def test_approach_only_executes_waypoint_zero_with_tight_tracking():
    module = _load("execute_approach_only")
    ctx = Context()
    result = module.run(
        ctx,
        _plan(
            {"pose": _pose(0.1), "mode": "planned_joint", "contact_margin": 0.003},
            {"pose": _pose(0.2), "mode": "planned_linear", "allow_goal_contact": True},
            {"pose": _pose(0.3), "mode": "contact_seat"},
        ),
    )
    assert [name for name, _ in ctx.calls] == ["motion.plan_to_pose", "robot.execute_trajectory"]
    planned = ctx.calls[0][1]
    assert planned["pose"] == _pose(0.1)
    assert planned["world_config"] == {"meshes": []}
    assert planned["attached_object"] == {"frame": "tcp", "spheres": []}
    assert planned["allow_start_contact"] is False and planned["allow_goal_contact"] is False
    assert planned["contact_margin"] == pytest.approx(0.003)
    executed = ctx.calls[1][1]
    assert executed["tolerance"] == pytest.approx(0.002)
    assert executed["max_steps_per_waypoint"] == 60
    assert result == {"approach_pose": _pose(0.1)}
    assert all("gripper" not in name for name, _ in ctx.calls)


def test_approach_only_retries_three_times_then_raises():
    module = _load("execute_approach_only")
    ctx = Context(
        plan_responses=[{"planned": False}, {"planned": True, "trajectory": {"waypoints": []}}]
    )
    result = module.run(ctx, _plan({"pose": _pose(0.1), "mode": "planned_joint"}))
    assert [name for name, _ in ctx.calls] == [
        "motion.plan_to_pose",
        "motion.plan_to_pose",
        "motion.plan_to_pose",
        "robot.execute_trajectory",
    ]
    assert result["approach_pose"] == _pose(0.1)

    ctx = Context(plan_responses=[{"planned": False}] * 3)
    with pytest.raises(RuntimeError, match="approach waypoint"):
        module.run(ctx, _plan({"pose": _pose(0.1), "mode": "planned_joint"}))
    assert [name for name, _ in ctx.calls] == ["motion.plan_to_pose"] * 3

    with pytest.raises(ValueError):
        module.run(Context(), {"waypoints": []})
