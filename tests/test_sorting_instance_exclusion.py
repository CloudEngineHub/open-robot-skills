"""select_pair's per-instance exclusion: what the loop mode buys, and its edges.

Driven through a recording fake, so there is no simulator, no model and no
randomness -- the VLM's reply is scripted and the question is only what the
script does with it.
"""

from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "skills/perceiving-sorting-pairs/scripts/select_pair.py"


def _load():
    sys.path.insert(0, str(SCRIPT.parent))
    spec = importlib.util.spec_from_file_location("select_pair_under_test", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def select_pair():
    return _load()


def _region(label, cx, cy):
    return {
        "label": label,
        "image_position": "top-left",
        "obb": {
            "center": {"x": cx, "y": cy, "z": 0.8},
            "extent": {"x": 0.05, "y": 0.05, "z": 0.002},
            "orientation": {"w": 1.0, "x": 0.0, "y": 0.0, "z": 0.0},
        },
    }


def _ctx(replies, object_centre=(0.0, 0.0)):
    """A fake whose detector, VLM, segmenter and geometry are all scripted."""
    from gap.testing.fakes import FakeContext

    mask = np.ones((40, 40), dtype=np.uint8)
    cx, cy = object_centre
    return FakeContext(
        {
            "grounding-dino.detect": {
                "detections": [
                    {"box": {"x1": 0.0, "y1": 0.0, "x2": 400.0, "y2": 400.0},
                     "score": 0.9, "label": "source bin"}
                ]
            },
            "vlm.query": [{"text": reply} for reply in replies],
            "sam3.segment_box": {"masks": [mask], "scores": [0.9]},
            "geometry.mask_to_world_points": {"points": {"points": np.zeros((200, 3), "float32")}},
            "geometry.filter_and_compute_obb": {
                "obb": {
                    "center": {"x": cx, "y": cy, "z": 0.82},
                    "extent": {"x": 0.02, "y": 0.02, "z": 0.01},
                    "orientation": {"w": 1.0, "x": 0.0, "y": 0.0, "z": 0.0},
                }
            },
        }
    )


def _observation():
    return {
        "cameras": [
            {
                "name": "overhead",
                "rgb": np.zeros((480, 640, 3), dtype=np.uint8),
                "depth": np.ones((480, 640), dtype=np.float32),
                "intrinsics": {"fx": 500.0, "fy": 500.0, "cx": 320.0, "cy": 240.0},
                "pose": {"position": {"x": 0.0, "y": 0.0, "z": 1.5},
                         "rotation": {"w": 1.0, "x": 0.0, "y": 0.0, "z": 0.0}},
            }
        ],
        "arms": [{}],
    }


LAYOUT = json.dumps([_region("tape", 0.5, 0.5), _region("pliers", 0.7, 0.5)])
PICK = "TARGET: tape roll; LABEL: tape; BOX: 100,100,160,160; PIXEL: 130,130"


def _run(module, ctx, **kwargs):
    return module.run(
        ctx,
        observation=_observation(),
        instruction="sort the tools",
        layout_json=LAYOUT,
        source_description="the source bin",
        **kwargs,
    )


def test_label_mode_publishes_no_attempt_log(select_pair):
    """A caller that did not ask for the loop does not get a new output key."""
    out = _run(select_pair, _ctx([PICK]))
    assert out["status"] == "found"
    assert "attempted_json" not in out


def test_instance_mode_logs_the_spot_it_picked(select_pair):
    out = _run(select_pair, _ctx([PICK]), exclusion_mode="instance")
    assert out["status"] == "found"
    logged = json.loads(out["attempted_json"])
    assert len(logged) == 1
    assert logged[0]["label"] == "tape"
    assert logged[0]["attempts"] == 1


def test_a_spent_spot_is_re_asked_not_returned(select_pair):
    """The whole point: the same object twice must not be picked twice."""
    spent = json.dumps(
        [{"label": "tape", "px": 130.0, "py": 130.0, "rx": 30.0, "ry": 30.0,
          "attempts": 2, "reason": "tried"}]
    )
    ctx = _ctx([PICK, PICK, PICK])
    out = _run(select_pair, ctx, exclusion_mode="instance", attempted_json=spent)
    assert out["status"] == "finished"
    # Re-asked the full budget rather than returning the exhausted object.
    assert len(ctx.calls_to("vlm.query")) == 3


def test_the_model_is_told_which_spots_are_spent(select_pair):
    spent = json.dumps(
        [{"label": "tape", "px": 130.0, "py": 130.0, "rx": 30.0, "ry": 30.0,
          "attempts": 2, "reason": "tried"}]
    )
    ctx = _ctx([PICK, PICK, PICK])
    _run(select_pair, ctx, exclusion_mode="instance", attempted_json=spent)
    prompt = ctx.calls_to("vlm.query")[0].kwargs["prompt"]
    assert "already been attempted" in prompt


def test_an_object_already_in_its_region_is_not_unsorted(select_pair):
    """A looping graph re-reads the scene; a delivered object must not be re-picked."""
    ctx = _ctx([PICK, PICK, PICK], object_centre=(0.5, 0.5))  # inside the tape region
    out = _run(select_pair, ctx, exclusion_mode="instance", settled_margin_m=0.01)
    assert out["status"] == "finished"
    logged = json.loads(out["attempted_json"])
    assert logged and logged[0]["reason"] == "settled"


def test_the_settled_test_is_off_by_default(select_pair):
    """settled_margin_m defaults to zero, so the same scene returns the object."""
    ctx = _ctx([PICK], object_centre=(0.5, 0.5))
    out = _run(select_pair, ctx, exclusion_mode="instance")
    assert out["status"] == "found"


def test_a_newline_separated_reply_is_accepted(select_pair):
    """The reply that ended a level1 episode on the source branch.

    Well formed, newlines where the prompt asked for semicolons. Refusing it on
    punctuation is what a stricter regex did there, and a retry cannot undo it:
    the model runs at temperature zero and repeats itself exactly.
    """
    newline_reply = "TARGET: painters tape\nLABEL: tape\nBOX: 100,190,260,340\nPIXEL: 180,265"
    out = _run(select_pair, _ctx([newline_reply]))
    assert out["status"] == "found"
    assert out["target_name"] == "painters tape"
