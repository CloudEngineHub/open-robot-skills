"""CPU-only contract tests for the feature-mating execution skill."""

from gap.testing import FakeContext


def _pose(z):
    return {
        "position": {"x": 0.4, "y": 0.0, "z": z},
        "rotation": {"w": 1.0, "x": 0.0, "y": 0.0, "z": 0.0},
    }


def test_executes_in_order_then_releases(skills_registry):
    info = skills_registry.get("executing-feature-mating")
    execute = info.canonical_scripts["execute_waypoints"].module
    release = info.canonical_scripts["release_and_retract"].module
    ctx = FakeContext({
        "robot.get_ee_pose": {"pose": _pose(0.8)},
        "motion.plan_to_pose": {"planned": True, "trajectory": {"waypoints": [{"positions": [0.0]}]}},
        "motion.plan_linear": {"planned": True, "trajectory": {"waypoints": [{"positions": [0.0]}]}},
        "robot.execute_trajectory": None,
        "robot.open_gripper": None,
        "robot.wait_steps": None,
        "robot.go_to_pose_cartesian": None,
    })
    plan = {"waypoints": [
        {"pose": _pose(0.5), "mode": "planned_joint"},
        {"pose": _pose(0.4), "mode": "planned_linear"},
    ], "world_config": {"meshes": []}, "attached_object": {"frame": "tcp", "spheres": []}}
    out = execute.run(ctx, plan)
    assert out["waypoint_count"] == 2
    assert out["final_pose"] == _pose(0.4)
    assert [call.tool for call in ctx.calls] == [
        "robot.get_ee_pose", "motion.plan_to_pose", "robot.execute_trajectory",
        "robot.get_ee_pose", "motion.plan_linear", "robot.execute_trajectory",
    ]
    release.run(ctx, out["final_pose"], retract_m=0.08)
    assert [call.tool for call in ctx.calls[-3:]] == [
        "robot.open_gripper", "robot.go_to_pose_cartesian", "robot.wait_steps",
    ]
