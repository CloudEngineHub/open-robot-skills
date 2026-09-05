"""CPU-only contracts for the reusable rigid-task skills."""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).parents[1] / "skills"


def load(skill, script):
    path = ROOT / skill / "scripts" / script
    spec = importlib.util.spec_from_file_location(f"test_{skill}", path)
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

    def calls_to(self, name):
        return [kwargs for called, kwargs in self.calls if called == name]


def pose(x=0.0, y=0.0, z=0.0):
    return {
        "position": {"x": x, "y": y, "z": z},
        "rotation": {"w": 1.0, "x": 0.0, "y": 0.0, "z": 0.0},
    }


def rotated_pose(rotation, x=0.0, y=0.0, z=0.0):
    q = rotation.as_quat()
    return {
        "position": {"x": x, "y": y, "z": z},
        "rotation": {"w": float(q[3]), "x": float(q[0]), "y": float(q[1]), "z": float(q[2])},
    }


def matrix(pose_dict):
    q = pose_dict["rotation"]
    p = pose_dict["position"]
    out = np.eye(4)
    out[:3, :3] = Rotation.from_quat([q["x"], q["y"], q["z"], q["w"]]).as_matrix()
    out[:3, 3] = [p["x"], p["y"], p["z"]]
    return out


def wrist_camera(name="eye_in_hand_0"):
    return {
        "name": name,
        "rgb": np.zeros((8, 8, 3), np.uint8),
        "depth": np.ones((8, 8)),
        "intrinsics": np.eye(3),
        "pose": pose(z=0.3),
    }


def blob_cloud(center=(0.0, 0.0, 0.05), size=0.1, count=30):
    """A deterministic elongated cloud of a known 3D extent around ``center``."""
    t = np.linspace(-0.5, 0.5, count)
    points = np.column_stack([size * t, 0.2 * size * np.cos(6.0 * t), 0.1 * size * np.sin(9.0 * t)])
    return {"points": (points + np.asarray(center, dtype=np.float64)).astype(np.float32)}


def test_compute_mate_delegates_typed_relation():
    module = load("computing-feature-mating-poses", "compute_mate.py")
    expected = {
        "mate_pose": pose(),
        "approach_pose": pose(z=0.1),
        "approach_axis": {"x": 0, "y": 0, "z": 1},
        "seating_distance": 0.01,
        "minimum_clearance": 0.004,
    }
    ctx = Context(
        {"robot.get_ee_pose": {"pose": pose()}, "geometry.compute_feature_mate": expected}
    )
    fixture = {"pose": pose(), "axis": {"x": 0.0, "y": 0.0, "z": 1.0}, "radius_outer": 0.003}
    out = module.run(ctx, pose(), fixture, "loop_over_shaft", {"frame": "tcp", "spheres": []})
    assert out == expected
    assert ctx.calls[1][1]["relation"] == "loop_over_shaft"
    assert ctx.calls[1][1]["reference_tcp_pose"] == pose()


def _mate_result():
    return {
        "mate_pose": pose(x=0.4, z=0.1),
        "approach_pose": pose(x=0.4, z=0.15),
        "engaged_pose": pose(x=0.4, z=0.1),
        "approach_axis": {"x": 0, "y": 0, "z": 1},
        "seating_distance": 0.0,
        "minimum_clearance": 0.004,
    }


def test_compute_mate_loop_over_shaft_uses_held_inner_and_fixture_outer_radius():
    module = load("computing-feature-mating-poses", "compute_mate.py")
    ctx = Context(
        {
            "robot.get_ee_pose": {"pose": pose()},
            "geometry.compute_feature_mate": lambda **_: _mate_result(),
        }
    )
    held = {**pose(z=0.1), "radius_inner": 0.012, "radius_outer": 0.02}
    fixture = {
        "pose": pose(x=0.4, z=0.1),
        "axis": {"x": 0.0, "y": 0.0, "z": 1.0},
        "radius_outer": 0.004,
        "radius_inner": 0.0,
    }
    module.run(ctx, held, fixture, "loop_over_shaft", {"frame": "tcp", "spheres": []})
    call = ctx.calls_to("geometry.compute_feature_mate")[0]
    assert call["held_radius"] == pytest.approx(0.012)
    assert call["fixture_radius"] == pytest.approx(0.004)
    assert call["approach_margin"] == pytest.approx(3.0 * 0.012)


def test_compute_mate_shaft_into_aperture_uses_the_mirrored_pair_and_approach():
    module = load("computing-feature-mating-poses", "compute_mate.py")
    ctx = Context(
        {
            "robot.get_ee_pose": {"pose": pose()},
            "geometry.compute_feature_mate": lambda **_: _mate_result(),
        }
    )
    held = {**pose(z=0.1), "radius_inner": 0.0, "radius_outer": 0.005}
    fixture = {
        "pose": pose(x=0.4, z=0.1),
        "axis": {"x": 0.0, "y": 0.0, "z": 1.0},
        "radius_inner": 0.01,
        "radius_outer": 0.03,
    }
    out = module.run(ctx, held, fixture, "shaft_into_aperture", {"frame": "tcp", "spheres": []})
    call = ctx.calls_to("geometry.compute_feature_mate")[0]
    assert call["held_radius"] == pytest.approx(0.005)
    assert call["fixture_radius"] == pytest.approx(0.01)
    assert "approach_margin" not in call
    # The axis points into the aperture: the approach is mirrored onto the
    # free side of the mate rather than left along the axis.
    assert out["approach_pose"]["position"]["z"] == pytest.approx(0.05)
    assert out["mate_pose"]["position"]["z"] == pytest.approx(0.1)


def _loop_mate_result():
    return {
        "mate_pose": pose(x=0.38, z=0.2),
        "approach_pose": pose(x=0.45, z=0.2),
        "engaged_pose": pose(x=0.39, z=0.2),
        "approach_axis": {"x": 1, "y": 0, "z": 0},
        "seating_distance": 0.0,
        "minimum_clearance": 0.004,
    }


def test_crossing_lift_raises_only_the_approach_and_engaged_poses():
    module = load("computing-feature-mating-poses", "compute_mate.py")
    held = {**pose(z=0.1), "radius_inner": 0.012}
    fixture = {
        "pose": pose(x=0.4, z=0.3),
        "axis": {"x": 1.0, "y": 0.0, "z": 0.0},
        "radius_outer": 0.004,
        "usable_length": 0.06,
    }

    def run(lift):
        ctx = Context(
            {
                "robot.get_ee_pose": {"pose": pose()},
                "geometry.compute_feature_mate": lambda **_: _loop_mate_result(),
            }
        )
        return module.run(
            ctx,
            held,
            fixture,
            "loop_over_shaft",
            {"frame": "tcp", "spheres": []},
            crossing_lift_m=lift,
        )

    plain, lifted = run(0.0), run(0.008)
    for name in ("approach_pose", "engaged_pose"):
        assert (lifted[name]["position"]["z"] - plain[name]["position"]["z"]) == pytest.approx(
            0.008
        )
        for key in ("x", "y"):
            assert lifted[name]["position"][key] == pytest.approx(plain[name]["position"][key])
    assert lifted["mate_pose"] == plain["mate_pose"]


def _sorting_camera(size: int = 240) -> dict:
    return {
        "name": "overhead",
        "rgb": np.zeros((size, size, 3), np.uint8),
        "depth": np.ones((size, size)),
        "intrinsics": np.eye(3),
        "pose": pose(),
    }


# The source container the script localizes first: a detector box that is
# both large enough and labelled as a container, or it refuses to guess.
_SOURCE_BIN = {
    "detections": [
        {"score": 0.9, "label": "source bin", "box": {"x1": 10, "y1": 10, "x2": 160, "y2": 160}}
    ]
}


def test_sorting_pair_has_clean_finished_exit():
    module = load("perceiving-sorting-pairs", "select_pair.py")
    layout = json.dumps([{"label": "wrench", "obb": {"center": {"x": 2}}}])
    observation = {"cameras": [_sorting_camera()]}
    # Every label already attempted: finished before any tool is consulted.
    silent = Context({})
    out = module.run(silent, observation, "sort", layout, "source bin", exclude_label_1="wrench")
    assert out["status"] == "finished" and silent.calls == []
    # The model reports nothing left: finished with the same empty shape.
    ctx = Context({"grounding-dino.detect": _SOURCE_BIN, "vlm.query": {"text": "DONE"}})
    out = module.run(ctx, observation, "sort", layout, "source bin")
    assert out["status"] == "finished"
    assert out["target_name"] == "" and out["target_label"] == ""
    assert out["target_mask"].shape == (1, 1)
    assert out["target_cloud"]["points"].shape == (0, 3)
    assert ctx.calls_to("sam3.segment_box") == []


def test_sorting_pair_uses_semantic_association_not_index():
    module = load("perceiving-sorting-pairs", "select_pair.py")
    mask = np.zeros((240, 240), np.uint8)
    mask[30:90, 20:80] = 1
    cloud = {"points": np.array([[0, 0, 0], [0.01, 0, 0], [0, 0.01, 0]], np.float32)}
    ctx = Context(
        {
            "grounding-dino.detect": _SOURCE_BIN,
            "vlm.query": {
                "text": "TARGET: adjustable wrench; LABEL: wrench; BOX: 20,30,80,90; PIXEL: 50,60"
            },
            "sam3.segment_box": {"masks": [mask], "scores": [0.8]},
            "geometry.mask_to_world_points": {"points": cloud},
            "geometry.filter_and_compute_obb": {"obb": {"center": {"x": 9}}},
        }
    )
    regions = [
        {"label": "pliers", "obb": {"center": {"x": 1}}},
        {"label": "wrench", "obb": {"center": {"x": 2}}},
    ]
    out = module.run(
        ctx, {"cameras": [_sorting_camera()]}, "sort tools", json.dumps(regions), "source bin"
    )
    assert out["status"] == "found"
    assert out["target_name"] == "adjustable wrench" and out["target_label"] == "wrench"
    # The destination is the region whose label the model named, not region 0.
    assert out["destination_obb"]["center"]["x"] == 2
    assert out["target_obb"] == {"center": {"x": 9}}
    segment = ctx.calls_to("sam3.segment_box")[0]
    # Box and pixel come back in full-image coordinates; the crop's margin
    # around this source box is clipped at the image origin, so they are
    # the model's own numbers.
    assert segment["box"] == {"x1": 20.0, "y1": 30.0, "x2": 80.0, "y2": 90.0}
    assert (segment["pixel_x"], segment["pixel_y"]) == (50.0, 60.0)
    assert segment["use_point"] is True


def test_functional_feature_fits_planar_loop():
    module = load("perceiving-functional-features", "perceive_feature.py")
    mask = np.ones((2, 2), np.uint8) * 255
    camera = {
        "rgb": np.zeros((2, 2, 3), np.uint8),
        "depth": np.ones((2, 2)),
        "intrinsics": np.eye(3),
        "pose": {
            "position": {"x": 0, "y": 0, "z": 1},
            "rotation": {"w": 1, "x": 0, "y": 0, "z": 0},
        },
    }
    cloud = {
        "points": np.column_stack([np.cos(np.arange(8)), np.sin(np.arange(8)), np.zeros(8)]).astype(
            np.float32
        )
    }
    fit = {"pose": pose(), "normal": {"x": 0, "y": 0, "z": 1}, "radius": 0.01, "planarity": 1.0}
    ctx = Context(
        {
            "sam3.segment_text": {"masks": [mask], "scores": [0.9]},
            "geometry.mask_to_world_points": {"points": cloud},
            "geometry.fit_planar_feature": fit,
        }
    )
    out = module.run(ctx, [camera], cloud, mask, "ring", "loop")
    assert out["feature"]["kind"] == "loop" and out["feature"]["radius_inner"] == 0.01


def test_register_held_fallback_still_builds_attachment():
    module = load("registering-held-objects", "register_held.py")
    cloud = {"points": np.array([[0, 0, 0], [0.01, 0, 0], [0, 0.01, 0], [0, 0, 0.01]], np.float32)}
    ctx = Context(
        {
            "robot.get_ee_pose": {"pose": pose()},
            "geometry.cloud_to_attachment": {"attached_object": {"frame": "tcp", "spheres": []}},
        }
    )
    out = module.run(ctx, [], cloud, {"pose": pose()}, "held object")
    assert out["registration_confidence"] == 0.25
    assert out["attached_object"]["frame"] == "tcp"


def _register_context(observed_cloud, attachment_calls, score=0.9):
    def attach(**kwargs):
        attachment_calls.append(kwargs)
        return {"attached_object": {"frame": "tcp", "spheres": []}}

    return Context(
        {
            "robot.get_ee_pose": {"pose": pose()},
            "sam3.segment_text": {"masks": [np.ones((8, 8), np.uint8)], "scores": [score]},
            "geometry.mask_to_world_points": {"points": observed_cloud},
            "geometry.cloud_to_attachment": attach,
        }
    )


def test_register_held_attachment_source_selects_the_fitted_cloud():
    module = load("registering-held-objects", "register_held.py")
    reference = blob_cloud()
    observed = blob_cloud(center=(0.005, 0.0, 0.05))
    feature = {"pose": pose(z=0.05), "kind": "tip"}
    calls = []
    out = module.run(
        _register_context(observed, calls),
        [wrist_camera()],
        reference,
        feature,
        "tool",
        attachment_source="observed",
    )
    assert len(calls) == 1 and calls[0]["points"] is observed
    assert calls[0]["fit_type"] == "morphit" and calls[0]["max_spheres"] == 64
    assert calls[0]["surface_radius"] == pytest.approx(0.002) and calls[0][
        "margin"
    ] == pytest.approx(0.002)
    # The accepted wrist cloud also corrects the feature by the observed shift.
    assert out["feature_in_tcp"]["position"]["x"] == pytest.approx(0.005)
    assert out["registration_confidence"] == pytest.approx(0.9)
    calls = []
    module.run(
        _register_context(observed, calls),
        [wrist_camera()],
        reference,
        feature,
        "tool",
        attachment_source="reference",
    )
    assert len(calls) == 1 and calls[0]["points"] is reference


def test_register_held_extent_gate_rejects_an_outlier_cloud():
    module = load("registering-held-objects", "register_held.py")
    reference = blob_cloud(size=0.1)
    outlier = blob_cloud(size=1.0)  # the mask merged the held tool with the arm
    feature = {"pose": pose(z=0.05), "kind": "tip"}
    calls = []
    out = module.run(
        _register_context(outlier, calls),
        [wrist_camera()],
        reference,
        feature,
        "tool",
        attachment_source="observed",
    )
    # Rejected: the grasp-time transform is retained at low confidence and the
    # attachment is fitted from the reference cloud, never from the outlier.
    assert out["registration_confidence"] == 0.25
    assert calls[0]["points"] is reference
    assert out["feature_in_tcp"]["position"]["z"] == pytest.approx(0.05)
    assert out["feature_in_tcp"]["position"]["x"] == pytest.approx(0.0)


def test_register_held_grasp_pose_carries_the_reference_by_the_rigid_grasp_transform():
    module = load("registering-held-objects", "register_held.py")
    reference = blob_cloud(center=(0.5, 0.0, 0.02))
    feature = {"pose": pose(x=0.5, z=0.1), "kind": "tip"}
    grasp = pose(x=0.5, z=0.15)
    calls = []

    def attach(**kwargs):
        calls.append(kwargs)
        return {"attached_object": {"frame": "tcp", "spheres": []}}

    # The hand has lifted 20 cm since the grasp and the wrist sees nothing.
    ctx = Context(
        {
            "robot.get_ee_pose": {"pose": pose(x=0.5, z=0.35)},
            "sam3.segment_text": {"masks": [], "scores": []},
            "geometry.cloud_to_attachment": attach,
        }
    )
    out = module.run(ctx, [wrist_camera()], reference, feature, "tool", grasp_pose=grasp)
    assert out["feature_in_tcp"]["position"]["z"] == pytest.approx(-0.05)
    carried = np.asarray(calls[0]["points"]["points"])
    assert np.allclose(carried, np.asarray(reference["points"], dtype=np.float64) + [0.0, 0.0, 0.2])
    assert out["object_in_tcp"]["position"]["z"] == pytest.approx(0.02 - 0.15)


def test_register_held_directed_tip_replaces_the_fallback_feature():
    module = load("registering-held-objects", "register_held.py")
    n = 100
    z = np.linspace(0.0, 0.12, n)
    body = {
        "points": np.column_stack([1e-4 * np.cos(np.arange(n)), 1e-4 * np.sin(np.arange(n)), z])
    }
    marker = {"points": np.column_stack([np.zeros(8), np.zeros(8), np.linspace(0.11, 0.12, 8)])}

    def segment(query, **_):
        return (
            {"masks": [np.ones((8, 8), np.uint8)], "scores": [0.8]}
            if query == "syringe"
            else {"masks": [2 * np.ones((8, 8), np.uint8)], "scores": [0.7]}
        )

    def to_points(mask, **_):
        return {"points": body if int(np.max(mask)) == 1 else marker}

    ctx = Context(
        {
            "robot.get_ee_pose": {"pose": pose()},
            "sam3.segment_text": segment,
            "geometry.mask_to_world_points": to_points,
            "geometry.cloud_to_attachment": {"attached_object": {"frame": "tcp", "spheres": []}},
        }
    )
    out = module.run(
        ctx,
        [wrist_camera()],
        blob_cloud(center=(0, 0, 0.06), size=0.12),
        {"pose": pose(z=0.05), "kind": "tip"},
        "syringe",
        direction_marker_description="red needle cap of syringe",
        attachment_source="observed",
    )
    tip = out["feature_in_tcp"]["position"]
    assert abs(tip["z"] - 0.12) < 0.004 and abs(tip["x"]) < 1e-3 and abs(tip["y"]) < 1e-3
    axis = Rotation.from_quat(
        [out["feature_in_tcp"]["rotation"][k] for k in ("x", "y", "z", "w")]
    ).apply([0, 0, 1])
    assert abs(axis[2]) > 0.999
    assert [call["query"] for call in ctx.calls_to("sam3.segment_text")] == [
        "syringe",
        "red needle cap of syringe",
    ]


def test_align_normal_preserving_roll_applies_only_the_tilt():
    module = load("registering-held-objects", "register_held.py")
    base = np.eye(4)
    base[:3, :3] = Rotation.from_euler("z", 30, degrees=True).as_matrix()
    observed = np.eye(4)
    observed[:3, :3] = (
        Rotation.from_euler("x", 20, degrees=True) * Rotation.from_euler("z", 77, degrees=True)
    ).as_matrix()
    result = module._align_normal_preserving_roll(base, observed)
    assert np.allclose(result[:3, 2], observed[:3, 2])
    relative = Rotation.from_matrix(result[:3, :3] @ base[:3, :3].T)
    assert np.degrees(relative.magnitude()) == pytest.approx(20.0, abs=1e-6)
    expected = (
        Rotation.from_euler("x", 20, degrees=True) * Rotation.from_euler("z", 30, degrees=True)
    ).as_matrix()
    assert np.allclose(result[:3, :3], expected)
    assert not np.allclose(result[:3, :3], observed[:3, :3])


def test_register_held_loop_fit_preserves_the_prior_roll():
    module = load("registering-held-objects", "register_held.py")
    prior_roll = Rotation.from_euler("z", 30, degrees=True)
    tilt = Rotation.from_euler("x", 20, degrees=True)
    prior = rotated_pose(prior_roll, z=0.1)
    angles = np.linspace(0, 2 * np.pi, 40, endpoint=False)
    ring = tilt.apply(np.column_stack([0.02 * np.cos(angles), 0.02 * np.sin(angles), np.zeros(40)]))
    ring += [0.0, 0.0, 0.1]
    fit = {
        "pose": rotated_pose(tilt * Rotation.from_euler("z", 77, degrees=True), z=0.1),
        "normal": {"x": 0.0, "y": 0.0, "z": 1.0},
        "radius": 0.02,
        "planarity": 1.0,
    }
    ctx = Context(
        {
            "robot.get_ee_pose": {"pose": pose()},
            "sam3.segment_text": {"masks": [np.ones((8, 8), np.uint8)], "scores": [0.9]},
            "geometry.mask_to_world_points": {"points": {"points": ring}},
            "geometry.fit_planar_feature": fit,
        }
    )
    out = module.run(
        ctx,
        [wrist_camera()],
        blob_cloud(),
        {"pose": pose(z=0.1), "kind": "loop", "description": "ring"},
        "wrench",
        prior_feature_in_tcp=prior,
        prior_object_in_tcp=pose(z=0.05),
        prior_attached_object={"frame": "tcp", "spheres": []},
    )
    result = matrix(out["feature_in_tcp"])
    assert np.allclose(result[:3, :3], (tilt * prior_roll).as_matrix(), atol=1e-9)
    assert np.allclose(result[:3, 3], [0.0, 0.0, 0.1], atol=1e-9)
    assert out["registration_confidence"] == pytest.approx(0.9)
    assert ctx.calls_to("sam3.segment_text")[0]["query"] == "ring"


def test_held_motion_selects_minimum_rotation_and_separates_phases():
    module = load("planning-held-object-motion", "plan_clearance_motion.py")
    ctx = Context(
        {
            "robot.get_ee_pose": {"pose": pose(z=0.2)},
            "motion.plan_joint": {
                "planned": True,
                "position_error_m": 0.0,
                "rotation_error_rad": 0.0,
            },
        }
    )
    feature = pose(z=0.1)
    fixture = {"pose": pose(x=0.4, z=0.3), "axis": {"x": 0.0, "y": 0.0, "z": 1.0}}
    out = module.run(
        ctx,
        feature,
        fixture,
        "shaft_into_aperture",
        {},
        {"frame": "tcp", "spheres": [{"center": [0, 0, 0], "radius": 0.01}]},
    )
    waypoints = out["reorientation_plan"]["waypoints"]
    assert [item["cartesian"] for item in waypoints] == [True, False, False]
    assert abs(waypoints[0]["pose"]["position"]["z"] - 0.24) < 1e-9
    # Eight symmetry candidates are checked, but zero extra roll wins.
    assert len([call for call in ctx.calls if call[0] == "motion.plan_joint"]) == 8
    assert abs(waypoints[1]["pose"]["rotation"]["w"] - 1.0) < 1e-8


def test_held_motion_direct_strategy_hands_the_approach_to_the_executor():
    module = load("planning-held-object-motion", "plan_clearance_motion.py")
    ctx = Context(
        {"robot.get_ee_pose": {"pose": pose(z=0.2)}, "motion.plan_joint": {"planned": True}}
    )
    fixture = {"pose": pose(x=0.4, z=0.3), "axis": {"x": 0.0, "y": 0.0, "z": 1.0}}
    approach = pose(x=0.5, z=0.6)
    out = module.run(
        ctx,
        pose(z=0.1),
        fixture,
        "loop_over_shaft",
        {},
        {"frame": "tcp", "spheres": [{"center": [0.0, 0.12, 0.0], "radius": 0.01}]},
        approach_pose=approach,
        strategy="direct",
    )
    waypoints = out["reorientation_plan"]["waypoints"]
    assert [w["mode"] for w in waypoints] == ["contact_transition", "planned_joint"]
    assert [w["cartesian"] for w in waypoints] == [True, False]
    assert abs(waypoints[0]["pose"]["position"]["z"] - 0.26) < 1e-9
    assert waypoints[1]["pose"] == approach
    assert waypoints[1]["allow_start_contact"] is True and waypoints[1]["max_attempts"] == 3
    # The executor plans the single joint leg; the planner is not probed here.
    assert ctx.calls_to("motion.plan_joint") == []
    assert out["reorientation_plan"]["time_scale"] == 2.0
    with pytest.raises(ValueError):
        module.run(
            ctx,
            pose(z=0.1),
            fixture,
            "loop_over_shaft",
            {},
            {"frame": "tcp", "spheres": []},
            strategy="direct",
        )


def test_held_motion_stage_outward_shifts_the_staged_waypoint_in_the_support_plane():
    module = load("planning-held-object-motion", "plan_clearance_motion.py")
    fixture = {"pose": pose(x=0.4, z=0.3), "axis": {"x": 0.0, "y": 0.0, "z": 1.0}}

    def run(outward):
        ctx = Context(
            {
                "robot.get_ee_pose": {"pose": pose(x=0.2, z=0.5)},
                "motion.plan_joint": {
                    "planned": True,
                    "position_error_m": 0.0,
                    "rotation_error_rad": 0.0,
                },
            }
        )
        return module.run(
            ctx,
            pose(z=0.1),
            fixture,
            "shaft_into_aperture",
            {},
            {"frame": "tcp", "spheres": []},
            stage_outward_m=outward,
        )

    plain = run(0.0)["reorientation_plan"]["waypoints"][2]["pose"]["position"]
    shifted = run(0.03)["reorientation_plan"]["waypoints"][2]["pose"]["position"]
    assert shifted["x"] - plain["x"] == pytest.approx(-0.03)
    assert shifted["y"] == pytest.approx(plain["y"]) and shifted["z"] == pytest.approx(plain["z"])


def test_held_motion_engagement_is_linear_only_at_fixture():
    module = load("planning-held-object-motion", "plan_linear_engagement.py")
    ctx = Context({"robot.get_ee_pose": {"pose": pose(z=0.4)}})
    fixture = {"pose": pose(x=0.4, z=0.3), "axis": {"x": 0.0, "y": 0.0, "z": 1.0}}
    out = module.run(
        ctx, pose(z=0.1), fixture, "shaft_into_aperture", {}, {"frame": "tcp", "spheres": []}
    )
    waypoints = out["placement_plan"]["waypoints"]
    assert [item["cartesian"] for item in waypoints] == [False, True]
    assert waypoints[-1]["allow_goal_contact"] is True


def test_linear_engagement_reobserves_the_tip_from_the_wrist():
    module = load("planning-held-object-motion", "plan_linear_engagement.py")
    n = 100
    z = np.linspace(0.0, 0.12, n)
    body = {
        "points": np.column_stack([1e-4 * np.cos(np.arange(n)), 1e-4 * np.sin(np.arange(n)), z])
    }
    marker = {"points": np.column_stack([np.zeros(8), np.zeros(8), np.linspace(0.11, 0.12, 8)])}

    def segment(query, **_):
        return (
            {"masks": [np.ones((8, 8), np.uint8)], "scores": [0.8]}
            if query == "syringe"
            else {"masks": [2 * np.ones((8, 8), np.uint8)], "scores": [0.7]}
        )

    def to_points(mask, **_):
        return {"points": body if int(np.max(mask)) == 1 else marker}

    ctx = Context(
        {
            "robot.get_ee_pose": {"pose": pose()},
            "sam3.segment_text": segment,
            "geometry.mask_to_world_points": to_points,
        }
    )
    fixture = {"pose": pose(x=0.4, z=0.3), "axis": {"x": 0.0, "y": 0.0, "z": 1.0}}
    observation = {"cameras": [wrist_camera("overhead"), wrist_camera("eye_in_hand_0")]}
    out = module.run(
        ctx,
        pose(z=0.1),
        fixture,
        "shaft_into_aperture",
        {},
        {"frame": "tcp", "spheres": []},
        observation=observation,
        object_description="syringe",
        direction_marker_description="red needle cap of syringe",
    )
    precontact = out["placement_plan"]["waypoints"][0]["pose"]["position"]
    # The carried estimate (z=0.10) is replaced by the observed endpoint (~0.12).
    assert abs(precontact["z"] - (0.28 - 0.12)) < 0.004
    assert abs(precontact["z"] - 0.18) > 0.01
    assert len(ctx.calls_to("sam3.segment_text")) == 2  # the overview camera is skipped


def test_loop_over_shaft_relation_produces_crossing_then_seating_plan():
    module = load("planning-held-object-motion", "plan_feature_engagement.py")
    out = module.run(
        Context({}),
        pose(z=0.3),
        pose(z=0.2),
        pose(z=0.1),
        "loop_over_shaft",
        {},
        {"frame": "tcp", "spheres": []},
    )
    waypoints = out["placement_plan"]["waypoints"]
    assert [w["mode"] for w in waypoints] == ["planned_joint", "cartesian_cross", "contact_seat"]
    out = module.run(
        Context({}),
        pose(z=0.3),
        pose(z=0.2),
        pose(z=0.1),
        "feature_to_fixture",
        {},
        {"frame": "tcp", "spheres": []},
    )
    assert [w["mode"] for w in out["placement_plan"]["waypoints"]] == [
        "planned_joint",
        "contact_seat",
        "contact_seat",
    ]
    out = module.run(
        Context({}),
        pose(z=0.3),
        pose(z=0.2),
        pose(z=0.1),
        "shaft_into_aperture",
        {},
        {"frame": "tcp", "spheres": []},
    )
    assert all(w.get("allow_goal_contact") for w in out["placement_plan"]["waypoints"][1:])


def test_plan_from_feature_mate_is_gone():
    assert not (
        ROOT / "planning-held-object-motion" / "scripts" / "plan_from_feature_mate.py"
    ).exists()
