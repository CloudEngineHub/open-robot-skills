import numpy as np


def _script(skills_registry):
    return skills_registry.get("grasping-linear-feature").canonical_scripts["compute_grasp"].module


def _vec(value):
    return np.array([value["x"], value["y"], value["z"]], dtype=float)


def test_linear_feature_grasp_is_perpendicular_and_preserves_standoff(skills_registry):
    module = _script(skills_registry)
    center = {"x": 0.4, "y": -0.1, "z": 0.3}
    axis = {"x": 1.0, "y": 0.0, "z": 1.0}
    out = module.run(None, center, axis, standoff=0.15, surface_inset=0.02)

    approach = _vec(out["approach_axis"])
    unit_axis = _vec(axis) / np.linalg.norm(_vec(axis))
    grasp = _vec(out["grasp_pose"]["position"])
    pregrasp = _vec(out["pregrasp_pose"]["position"])

    assert np.isclose(np.linalg.norm(approach), 1.0)
    assert np.isclose(approach @ unit_axis, 0.0, atol=1e-7)
    assert approach[2] <= 0.0
    assert np.allclose(grasp, _vec(center) + 0.02 * approach)
    assert np.allclose(pregrasp, grasp - 0.15 * approach)


def test_vertical_feature_uses_horizontal_approach(skills_registry):
    module = _script(skills_registry)
    out = module.run(
        None,
        {"x": 0.0, "y": 0.0, "z": 0.5},
        {"x": 0.0, "y": 0.0, "z": 1.0},
    )
    assert np.allclose(_vec(out["approach_axis"]), [-1.0, 0.0, 0.0])


def test_zero_feature_axis_is_rejected(skills_registry):
    import pytest

    module = _script(skills_registry)
    with pytest.raises(ValueError, match="nonzero"):
        module.run(
            None,
            {"x": 0.0, "y": 0.0, "z": 0.5},
            {"x": 0.0, "y": 0.0, "z": 0.0},
        )
