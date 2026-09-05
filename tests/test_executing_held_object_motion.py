from __future__ import annotations

import importlib.util
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "skills/executing-held-object-motion/scripts/execute_reorientation.py"


def _load():
    spec = importlib.util.spec_from_file_location("execute_reorientation", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


class Context:
    def __init__(self):
        self.calls = []

    def tool(self, name, **inputs):
        self.calls.append((name, inputs))
        if name.startswith("motion.plan_"):
            return {"planned": True, "trajectory": {"waypoints": [
                {"positions": [0.0]}, {"positions": [1.0]},
            ]}}
        return None


def _pose(x):
    return {
        "position": {"x": x, "y": 0.0, "z": 0.5},
        "rotation": {"w": 1.0, "x": 0.0, "y": 0.0, "z": 0.0},
    }


def test_executes_waypoints_in_order_without_gripper_calls():
    module = _load()
    ctx = Context()
    result = module.run(ctx, {"waypoints": [
        {"pose": _pose(0.1), "mode": "planned_joint"},
        {"pose": _pose(0.2), "mode": "planned_linear"},
    ], "world_config": {"meshes": []}, "attached_object": {"frame": "tcp", "spheres": []}})

    assert [name for name, _ in ctx.calls] == [
        "motion.plan_to_pose", "robot.execute_trajectory",
        "motion.plan_linear", "robot.execute_trajectory",
    ]
    assert result["final_pose"] == _pose(0.2)
    assert all("gripper" not in name for name, _ in ctx.calls)
    for name, inputs in ctx.calls:
        if name.startswith("motion.plan_"):
            assert inputs["world_config"] == {"meshes": []}
            assert inputs["attached_object"] == {"frame": "tcp", "spheres": []}
