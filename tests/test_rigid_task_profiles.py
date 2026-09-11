"""The profile-driven paths the tool-hanging fold added, CPU-only.

Ported from the delivery's own tests (`origin/rigid_tasks_skills`,
`tests/test_rigid_task_skills.py`) and adapted to where the fold put things:
`execute_waypoints` is `execute_placement_plan`, `execute_approach` is
`execute_approach_only`, `constructing-collision-worlds` is
`reconstructing-collision-worlds`, `destination_anchor_y` is
`correspondence_anchor`, and the recorded-call layers
(`verify_cartesian`, `measure_errors`) are asked for explicitly. The default
paths these scripts share with the pre-fold graphs are covered by the
existing per-bundle tests; this file covers only what a profile turns on.
"""

import importlib.util
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parents[1] / "skills"


def load(skill, script):
    path = ROOT / skill / "scripts" / script
    spec = importlib.util.spec_from_file_location(f"profiles_{skill}_{script[:-3]}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Context:
    def __init__(self, handlers):
        self.handlers, self.calls = handlers, []

    def tool(self, name, **kwargs):
        self.calls.append((name, kwargs))
        handler = self.handlers[name]
        return handler(**kwargs) if callable(handler) else handler

    def names(self):
        return [name for name, _ in self.calls]


def pose(x=0.0, y=0.0, z=0.0):
    return {"position": {"x": x, "y": y, "z": z}, "rotation": {"w": 1.0, "x": 0.0, "y": 0.0, "z": 0.0}}


def camera(name="overhead", size=2):
    return {"name": name, "rgb": np.zeros((size, size, 3), np.uint8), "depth": np.ones((size, size)),
            "intrinsics": np.eye(3), "pose": pose()}


# --- computing-feature-mating-poses ------------------------------------------------------------


def test_compute_mate_mating_profile_moves_poses_along_the_settling_axis():
    module = load("computing-feature-mating-poses", "compute_mate.py")
    result = {"mate_pose": pose(), "approach_pose": pose(x=0.08), "engaged_pose": pose(),
              "approach_axis": {"x": -1, "y": 0, "z": 0}, "seating_distance": 0.01, "minimum_clearance": 0.004}
    ctx = Context({"robot.get_ee_pose": {"pose": pose()}, "geometry.compute_feature_mate": result})
    fixture = {"pose": pose(), "axis": {"x": -1.0, "y": 0.0, "z": 0.0}, "radius_outer": 0.003, "usable_length": 0.06,
               "mating_profile": {"settling_axis": [0, 0, 1],
                                  "crossing_offsets_m": {"approach_pose": 0.008, "engaged_pose": 0.018},
                                  "seated_radial_offset_m": 0.0, "mate_settling_offset_m": 0.010}}
    held = {**pose(), "radius_inner": 0.013}
    out = module.run(ctx, held, fixture, "loop_over_shaft", {"frame": "tcp", "spheres": []})
    assert abs(out["approach_pose"]["position"]["z"] - 0.008) < 1e-9
    assert abs(out["engaged_pose"]["position"]["z"] - 0.018) < 1e-9
    assert abs(out["mate_pose"]["position"]["z"] - 0.010) < 1e-9
    # No arm was named, so none was passed.
    assert ctx.calls[0] == ("robot.get_ee_pose", {})


# --- perceiving-functional-features ------------------------------------------------------------


def test_perceive_feature_parent_strategies_make_no_camera_call():
    module = load("perceiving-functional-features", "perceive_feature.py")
    parent = {"points": np.array([[0, 0, 0], [0.1, 0, 0], [0, 0.1, 0], [0.1, 0.1, 0.1], [0.05, 0.05, 0.09],
                                  [0.04, 0.05, 0.09], [0.06, 0.05, 0.09], [0.05, 0.04, 0.09]], np.float32)}
    ctx = Context({})
    out = module.run(ctx, [], parent, np.ones((2, 2), np.uint8), "declared point", "tip",
                     method="parent_landmark", feature_options={"offset": [0.01, 0.02, 0.03], "axis": [1, 0, 0]})
    assert out["feature"]["method"] == "parent_landmark"
    assert out["feature"]["axis"] == {"x": 1.0, "y": 0.0, "z": 0.0}
    assert "uncertainty_m" in out["feature"]
    assert ctx.calls == []


def test_object_feature_profile_moves_grasp_toward_feature_with_clearance():
    module = load("perceiving-functional-features", "perceive_object_feature.py")
    obb = {"center": {"x": 0.0, "y": 0.0, "z": 0.4}}
    module._shift_grasp_toward_feature(obb, {"x": 0.10, "y": 0.0, "z": 0.5},
                                       {"grasp_toward_feature_m": 0.025, "grasp_toward_feature_axes": ["x", "y"],
                                        "minimum_grasp_feature_separation_m": 0.05})
    assert np.allclose([obb["center"]["x"], obb["center"]["y"], obb["center"]["z"]], [0.025, 0.0, 0.4])
    # The clearance bound, rather than an object branch, limits the shift when the feature is near.
    close = {"center": {"x": 0.0, "y": 0.0, "z": 0.4}}
    module._shift_grasp_toward_feature(close, {"x": 0.06, "y": 0.0, "z": 0.4},
                                       {"grasp_toward_feature_m": 0.025, "minimum_grasp_feature_separation_m": 0.05})
    assert abs(close["center"]["x"] - 0.01) < 1e-9


def test_object_feature_candidates_path_keeps_the_original_call_sequence():
    """Without profiles the body is the pre-fold one: one segment per row at max_results=1, then the feature."""
    module = load("perceiving-functional-features", "perceive_object_feature.py")
    mask = np.ones((4, 4), np.uint8)
    cloud = {"points": np.array([[0, 0, 0.5], [0.01, 0, 0.5], [0, 0.01, 0.5], [0.01, 0.01, 0.5]] * 3, np.float32)}
    ctx = Context({
        "sam3.segment_text": {"masks": [mask], "scores": [0.9]},
        "geometry.mask_to_world_points": {"points": cloud},
        "geometry.filter_and_compute_obb": {"obb": {"center": {"x": 0, "y": 0, "z": 0.5}, "extent": {"x": .1, "y": .1, "z": .1},
                                                    "orientation": {"w": 1, "x": 0, "y": 0, "z": 0}}},
        "geometry.fit_planar_feature": {"pose": pose(z=0.5), "normal": {"x": 0, "y": 0, "z": 1}, "radius": 0.01, "planarity": 1.0},
    })
    out = module.run(ctx, {"cameras": [camera(size=4)]},
                     '[{"kind": "widget", "object_description": "blue widget", "feature_description": "ring", "feature_type": "loop"}]',
                     complete_to_support=False)
    assert ctx.names() == ["sam3.segment_text", "geometry.mask_to_world_points", "geometry.filter_and_compute_obb",
                           "sam3.segment_text", "geometry.mask_to_world_points", "geometry.mask_to_world_points",
                           "geometry.fit_planar_feature"]
    assert ctx.calls[0][1]["max_results"] == 1 and ctx.calls[3][1]["max_results"] == 3
    assert out["target_kind"] == "widget" and "relation" not in out


def test_depth_support_anchor_failure_falls_back_to_declared_partition():
    module = load("perceiving-functional-features", "perceive_fixture_feature.py")
    # The narrow anchor window sees only board points; the positive partition
    # also holds the support foreground and recovers a valid feature.
    board = np.column_stack((np.full(240, 0.85), np.linspace(-0.12, 0.12, 240), np.full(240, 1.0)))
    support = np.column_stack((np.full(80, 0.76), np.linspace(0.075, 0.11, 80), np.full(80, 0.98)))
    points = np.vstack((board, support)).astype(np.float32)
    ctx = Context({"geometry.mask_to_world_points": {"points": {"points": points}},
                   "geometry.filter_and_compute_obb": {"obb": {"center": {}, "extent": {}, "orientation": {}}}})
    profile = {"source_kind": "generic", "strategy": "depth_support",
               "workspace": {"x_min": 0.72, "x_max": 0.90, "y_min": -0.12, "y_max": 0.12, "z_min": 0.90, "z_max": 1.30},
               "anchor_tolerance": 0.01, "anchor_fallback_partition": True, "partition_axis": "y", "partition_split": 0.0}
    out = module.run(ctx, {"cameras": [camera(size=8)]}, target_kind="generic",
                     correspondence_anchor=0.18, fixture_profiles=[profile])
    assert out["fixture_kind"] == "support"
    assert out["hook_tip"]["y"] > 0.0


# --- perceiving-relational-correspondences --------------------------------------------------------


def test_relational_next_source_uses_declared_semantics_workspace_and_arm_partition():
    module = load("perceiving-relational-correspondences", "select_next_source.py")
    low = np.ones((10, 10), np.uint8) * 255
    high = np.ones((10, 10), np.uint8) * 127

    def backproject(mask, **_):
        center = np.array([0.60, 0.04, 0.74]) if int(np.max(mask)) == 255 else np.array([0.80, -0.05, 0.95])
        return {"points": {"points": np.tile(center, (25, 1)).astype(np.float32)}}

    ctx = Context({"sam3.segment_text": {"masks": [low, high], "scores": [0.7, 0.95]},
                   "geometry.mask_to_world_points": backproject})
    out = module.run(ctx, {"cameras": [camera(size=10)]}, "place the widget",
                     [{"kind": "widget", "query": "blue item", "aliases": ["widget"]}],
                     {"x_max": 0.7, "z_max": 0.8, "minimum_mask_pixels": 50},
                     arm_partition={"split": 0.0, "positive_arm": 8, "negative_arm": 9})
    assert out["status"] == "found" and out["target_kind"] == "widget"
    assert out["arm_id"] == 8 and abs(out["correspondence_anchor"] - 0.04) < 1e-6
    assert out["source_id"].startswith("widget-")


# --- registering-held-objects ---------------------------------------------------------------------


def test_register_held_without_a_profile_reports_the_fallback_it_took():
    module = load("registering-held-objects", "register_held.py")
    cloud = {"points": np.array([[0, 0, 0], [0.01, 0, 0], [0, 0.01, 0], [0, 0, 0.01]], np.float32)}
    ctx = Context({"robot.get_ee_pose": {"pose": pose()},
                   "curobo.cloud_to_attachment": {"attached_object": {"frame": "tcp", "spheres": []}}})
    out = module.run(ctx, [], cloud, {"pose": pose()}, "held object")
    assert out["registration_confidence"] == 0.25
    assert out["attached_object"] == {"frame": "tcp", "spheres": []}  # untouched: no arm, no profile
    assert out["registration_method"] == "rigid_grasp_prior"
    assert out["fallback_used"] is True
    assert out["translation_uncertainty_m"] == 0.01


def test_overview_loop_recovery_associates_nearest_center_and_preserves_rotation():
    module = load("registering-held-objects", "register_held.py")
    masks = [np.ones((2, 2), np.uint8), np.ones((2, 2), np.uint8) * 2]

    def backproject(mask, **_):
        x = 0.01 if int(np.max(mask)) == 1 else 0.03
        return {"points": {"points": np.tile([x, 0, 0], (25, 1)).astype(np.float32)}}

    def fit(points, **_):
        measured = pose(x=float(np.asarray(points["points"])[0, 0]))
        measured["rotation"] = {"w": 0.0, "x": 1.0, "y": 0.0, "z": 0.0}
        return {"pose": measured}

    ctx = Context({"sam3.segment_text": {"masks": masks, "scores": [0.6, 0.9]},
                   "geometry.mask_to_world_points": backproject, "geometry.fit_planar_feature": fit})
    predicted = module._matrix(pose())
    observed, score = module._recover_loop_from_overview(ctx, [camera()], predicted, "loop", {})
    assert abs(observed[0, 3] - 0.01) < 1e-6 and score == 0.6
    assert np.allclose(observed[:3, :3], predicted[:3, :3])


# --- grasping-direct-ik ---------------------------------------------------------------------------


def test_direct_grasp_profiles_control_clearance_and_verification():
    align = load("grasping-direct-ik", "compute_align_pose.py")
    obb = {"center": {"x": 0, "y": 0, "z": 0.1}, "extent": {"x": 0.02, "y": 0.01, "z": 0.01},
           "orientation": {"w": 1, "x": 0, "y": 0, "z": 0}}
    out = align.run(Context({}), pose(z=0.11), obb, grasp_profile={"approach_clearance_m": 0.08})
    assert abs(out["align_pose"]["position"]["z"] - 0.19) < 1e-9
    execute = load("grasping-direct-ik", "execute_grasp_align.py")
    ctx = Context({"robot.go_to_pose": {}, "robot.get_ee_pose": {"pose": pose(z=0.19)}})
    checked = execute.run(ctx, pose(z=0.19), grasp_profile={"position_tolerance_m": 0.01, "angular_tolerance_deg": 15.0})
    assert checked["recovery_used"] is False and checked["position_error_m"] == 0.0
    assert all("arm_id" not in kwargs for _, kwargs in ctx.calls)


def test_direct_grasp_cartesian_recovery_only_on_failed_verification():
    execute = load("grasping-direct-ik", "execute_grasp_align.py")
    angle = np.deg2rad(21.0)
    bad = pose(z=0.19)
    bad["rotation"] = {"w": float(np.cos(angle / 2)), "x": 0.0, "y": 0.0, "z": float(np.sin(angle / 2))}
    reached = [bad, pose(z=0.19)]
    ctx = Context({"robot.go_to_pose": {}, "robot.go_to_pose_cartesian": {},
                   "robot.get_ee_pose": lambda **_: {"pose": reached.pop(0)}})
    checked = execute.run(ctx, pose(z=0.19), grasp_profile={"position_tolerance_m": 0.01, "angular_tolerance_deg": 20.0})
    assert checked["recovery_used"] is True
    assert ctx.names().count("robot.go_to_pose_cartesian") == 1


def test_direct_grasp_large_error_splits_translation_from_rotation():
    execute = load("grasping-direct-ik", "execute_grasp_align.py")
    angle = np.deg2rad(21.0)
    displaced = pose(x=0.08, z=0.19)
    displaced["rotation"] = {"w": float(np.cos(angle / 2)), "x": 0.0, "y": 0.0, "z": float(np.sin(angle / 2))}
    translated = pose(z=0.19)
    translated["rotation"] = dict(displaced["rotation"])
    reached = [displaced, translated, pose(z=0.19)]
    ctx = Context({"robot.go_to_pose": {}, "robot.go_to_pose_cartesian": {},
                   "robot.get_ee_pose": lambda **_: {"pose": reached.pop(0)}})
    checked = execute.run(ctx, pose(z=0.19), grasp_profile={"position_tolerance_m": 0.01, "angular_tolerance_deg": 20.0,
                                                            "cartesian_recovery_threshold_m": 0.03})
    corrections = [kwargs for name, kwargs in ctx.calls if name == "robot.go_to_pose_cartesian"]
    assert len(corrections) == 2
    assert corrections[0]["pose"]["position"] == pose(z=0.19)["position"]
    assert corrections[0]["pose"]["rotation"] == displaced["rotation"]
    assert checked["position_error_m"] == 0.0 and checked["recovery_used"] is True


# --- planning-held-object-motion ------------------------------------------------------------------


def test_held_motion_direct_cartesian_profile_needs_no_motion_planner():
    module = load("planning-held-object-motion", "plan_clearance_motion.py")
    ctx = Context({"robot.get_ee_pose": {"pose": pose(z=0.2)}})
    out = module.run(ctx, pose(), {"pose": pose(), "axis": {"x": 0, "y": 0, "z": 1}}, "loop_over_shaft", {},
                     {"frame": "tcp", "spheres": []}, approach_pose=pose(x=0.4, z=0.5),
                     motion_profile={"strategy": "direct_cartesian", "escape_distance_m": 0.06, "time_scale": 1.25,
                                     "lift_speed_scale": 0.6, "transit_speed_scale": 0.5})
    plan = out["reorientation_plan"]
    assert plan["time_scale"] == 1.25
    assert [w["cartesian"] for w in plan["waypoints"]] == [True, True]
    assert plan["waypoints"][0]["speed_scale"] == 0.6 and plan["waypoints"][1]["speed_scale"] == 0.5
    assert "motion.plan_joint" not in ctx.names()
    assert "arm_id" not in plan


def test_held_motion_unprofiled_plan_carries_no_profile_keys():
    module = load("planning-held-object-motion", "plan_clearance_motion.py")
    ctx = Context({"robot.get_ee_pose": {"pose": pose(z=0.2)},
                   "motion.plan_joint": {"planned": True, "position_error_m": 0.0, "rotation_error_rad": 0.0}})
    out = module.run(ctx, pose(z=0.1), {"pose": pose(x=0.4, z=0.3), "axis": {"x": 0.0, "y": 0.0, "z": 1.0}},
                     "shaft_into_aperture", {}, {"frame": "tcp", "spheres": [{"center": [0, 0, 0], "radius": 0.01}]})
    for waypoint in out["reorientation_plan"]["waypoints"]:
        assert not {"speed_scale", "use_world", "max_attempts", "cartesian_fallback"} & set(waypoint)
    assert out["reorientation_plan"]["time_scale"] == 2.0
    assert all("arm_id" not in kwargs for _, kwargs in ctx.calls)


# --- executing-feature-mating / executing-held-object-motion --------------------------------------


def test_feature_mating_executor_reports_contact_and_uncertainty():
    module = load("executing-feature-mating", "execute_placement_plan.py")
    target = pose(z=0.3)
    ctx = Context({"robot.describe_arm": {"solver": {"honours_roll": False}},
                   "robot.get_ee_pose": {"pose": target},
                   "robot.move_cartesian_until_contact": {"status": "stalled"}})
    plan = {"relation": "loop_over_shaft", "attached_object": {"arm_id": 1, "translation_uncertainty_m": 0.006},
            "waypoints": [{"pose": target, "mode": "planned_joint"}, {"pose": target, "mode": "contact_seat"}]}
    out = module.run(ctx, plan, relation="loop_over_shaft", verify_cartesian=True)
    assert out["registration_uncertainty_m"] == 0.006
    assert out["waypoint_reports"][0]["fallback"] == "already_reached"
    assert out["waypoint_reports"][1]["contact_status"] == "stalled"
    # The plan named arm 1, so every call carries it.
    assert all(kwargs.get("arm_id") == 1 for _, kwargs in ctx.calls)


def test_held_motion_executor_reports_declared_cartesian_transition():
    module = load("executing-held-object-motion", "execute_reorientation.py")
    target = pose(z=0.4)
    ctx = Context({"robot.go_to_pose_cartesian": {}, "robot.get_ee_pose": {"pose": target}})
    out = module.run(ctx, {"attached_object": {"arm_id": 1}, "waypoints": [{"pose": target, "mode": "contact_transition"}]},
                     measure_errors=True)
    assert out["fallback_count"] == 0
    assert out["waypoint_reports"][0]["mode"] == "contact_transition"
    assert out["waypoint_reports"][0]["position_error_m"] == 0.0
    assert ctx.names() == ["robot.go_to_pose_cartesian", "robot.get_ee_pose"]


# --- reconstructing-collision-worlds --------------------------------------------------------------


def test_collision_world_disabled_strategy_is_explicit_and_name_independent():
    module = load("reconstructing-collision-worlds", "build_collision_world.py")
    ctx = Context({})
    out = module.run(ctx, {"cameras": []}, np.ones((2, 2), np.uint8), target_kind="novel_object",
                     collision_profiles=[{"source_kind": "novel_object", "strategy": "disabled"}])
    assert out == {"world_config": {"meshes": []}, "mesh_names": [], "removed_mesh_names": [], "strategy": "disabled"}
    assert ctx.calls == []


def test_collision_world_profile_controls_reconstruction_and_keep_out():
    module = load("reconstructing-collision-worlds", "build_collision_world.py")
    vertices = np.column_stack((np.linspace(0, 0.2, 40), np.zeros(40), np.full(40, 0.5)))
    world = {"config": {"meshes": [{"name": "scene", "vertices": vertices.tolist(), "faces": [[0, 1, 2]]}]}}
    ctx = Context({"geometry.build_world_config": world})
    profile = {"source_kind": "novel_object", "strategy": "rgbd_mesh",
               "excluded_masks": {"target": True, "cross_view_target": False, "robot": False},
               "fixture_keep_out": {"boxes": [{"name": "wall", "center": {"x": 0.5, "y": 0, "z": 0.5},
                                               "size": {"x": 0.1, "y": 0.1, "z": 0.1}}]},
               "approach_corridor": {"enabled": False}, "obstacle_filter": {}}
    out = module.run(ctx, {"cameras": [camera()]}, np.ones((2, 2), np.uint8), target_kind="novel_object",
                     collision_profiles=[profile], robot_spheres=[])
    assert out["strategy"] == "rgbd_mesh"
    assert [mesh["name"] for mesh in out["world_config"]["meshes"]] == ["scene", "wall"]
    assert ctx.names() == ["geometry.build_world_config"]  # robot masks off, spheres supplied: no sam3, no planner
