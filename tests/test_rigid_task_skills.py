"""CPU-only contracts for the reusable rigid-task skills."""
import importlib.util
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).parents[1] / "skills"

def load(skill, script):
    path = ROOT / skill / "scripts" / script
    spec = importlib.util.spec_from_file_location(f"test_{skill}", path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module

class Context:
    def __init__(self, handlers): self.handlers, self.calls = handlers, []
    def tool(self, name, **kwargs):
        self.calls.append((name, kwargs)); handler=self.handlers[name]
        return handler(**kwargs) if callable(handler) else handler

def pose(x=0.0, y=0.0, z=0.0):
    return {"position":{"x":x,"y":y,"z":z},
            "rotation":{"w":1.0,"x":0.0,"y":0.0,"z":0.0}}

def test_compute_mate_delegates_typed_relation():
    module=load("computing-feature-mating-poses","compute_mate.py")
    expected={"mate_pose":pose(),"approach_pose":pose(z=0.1),"approach_axis":{"x":0,"y":0,"z":1},
              "seating_distance":0.01,"minimum_clearance":0.004}
    ctx=Context({"robot.get_ee_pose":{"pose":pose()},
                 "geometry.compute_feature_mate":expected})
    fixture={"pose":pose(),"axis":{"x":0.0,"y":0.0,"z":1.0},"radius_outer":0.003}
    out=module.run(ctx,pose(),fixture,"loop_over_shaft",{"frame":"tcp","spheres":[]})
    assert out==expected
    assert ctx.calls[1][1]["relation"]=="loop_over_shaft"
    assert ctx.calls[1][1]["reference_tcp_pose"]==pose()

def test_sorting_pair_has_clean_finished_exit():
    module=load("perceiving-sorting-pairs","select_pair.py")
    camera={"name":"overhead","rgb":np.zeros((8,8,3),np.uint8)}
    ctx=Context({"vlm.query":{"text":"DONE"}})
    assert module.run(ctx,{"cameras":[camera]},"sort",json.dumps([{"label":"wrench","obb":{}}]),"source") == {
        "status":"finished"}

def test_sorting_pair_uses_semantic_association_not_index():
    module=load("perceiving-sorting-pairs","select_pair.py")
    boxes=[{"detections":[{"score":.9,"box":{"x1":0,"y1":0,"x2":8,"y2":8}}]},
           {"detections":[{"score":.8,"box":{"x1":2,"y1":2,"x2":6,"y2":6}}]}]
    def detect(**_): return boxes.pop(0)
    cloud={"points":np.array([[0,0,0],[.01,0,0],[0,.01,0]],np.float32)}
    ctx=Context({"vlm.query":{"text":"TARGET: adjustable wrench; LABEL: wrench"},
                 "grounding-dino.detect":detect,
                 "sam3.segment_box":{"masks":[np.ones((8,8),np.uint8)]},
                 "geometry.mask_to_world_points":{"points":cloud},
                 "geometry.filter_and_compute_obb":{"obb":{"center":{}}}})
    camera={"name":"overhead","rgb":np.zeros((8,8,3),np.uint8),"depth":np.ones((8,8)),
            "intrinsics":np.eye(3),"pose":pose()}
    regions=[{"label":"pliers","obb":{"center":{"x":1}}},
             {"label":"wrench","obb":{"center":{"x":2}}}]
    out=module.run(ctx,{"cameras":[camera]},"sort tools",json.dumps(regions),"source bin")
    assert out["status"]=="found" and out["destination_obb"]["center"]["x"]==2

def test_functional_feature_fits_planar_loop():
    module=load("perceiving-functional-features","perceive_feature.py")
    mask=np.ones((2,2),np.uint8)*255
    camera={"rgb":np.zeros((2,2,3),np.uint8),"depth":np.ones((2,2)),"intrinsics":np.eye(3),
            "pose":{"position":{"x":0,"y":0,"z":1},"rotation":{"w":1,"x":0,"y":0,"z":0}}}
    cloud={"points":np.column_stack([np.cos(np.arange(8)),np.sin(np.arange(8)),np.zeros(8)]).astype(np.float32)}
    fit={"pose":pose(),"normal":{"x":0,"y":0,"z":1},"radius":0.01,"planarity":1.0}
    ctx=Context({"sam3.segment_text":{"masks":[mask],"scores":[0.9]},
                 "geometry.mask_to_world_points":{"points":cloud},"geometry.fit_planar_feature":fit})
    out=module.run(ctx,[camera],cloud,mask,"ring","loop")
    assert out["feature"]["kind"]=="loop" and out["feature"]["radius_inner"]==0.01

def test_register_held_fallback_still_builds_attachment():
    module=load("registering-held-objects","register_held.py")
    cloud={"points":np.array([[0,0,0],[.01,0,0],[0,.01,0],[0,0,.01]],np.float32)}
    ctx=Context({"robot.get_ee_pose":{"pose":pose()},
                 "geometry.cloud_to_attachment":{"attached_object":{"frame":"tcp","spheres":[]}}})
    out=module.run(ctx,[],cloud,{"pose":pose()},"held object")
    assert out["registration_confidence"]==0.25
    assert out["attached_object"]["frame"]=="tcp"

def test_held_motion_selects_minimum_rotation_and_separates_phases():
    module=load("planning-held-object-motion", "plan_clearance_motion.py")
    ctx=Context({"robot.get_ee_pose":{"pose":pose(z=.2)},
                 "motion.plan_joint":{"planned":True,"position_error_m":0.0,
                                      "rotation_error_rad":0.0}})
    feature=pose(z=.1)
    fixture={"pose":pose(x=.4,z=.3),"axis":{"x":0.0,"y":0.0,"z":1.0}}
    out=module.run(ctx,feature,fixture,"shaft_into_aperture",{},
                   {"frame":"tcp","spheres":[{"center":[0,0,0],"radius":.01}]})
    waypoints=out["reorientation_plan"]["waypoints"]
    assert [item["cartesian"] for item in waypoints] == [True,False,False]
    assert abs(waypoints[0]["pose"]["position"]["z"]-.24) < 1e-9
    # Eight symmetry candidates are checked, but zero extra roll wins.
    assert len([call for call in ctx.calls if call[0]=="motion.plan_joint"]) == 8
    assert abs(waypoints[1]["pose"]["rotation"]["w"]-1.0) < 1e-8

def test_held_motion_engagement_is_linear_only_at_fixture():
    module=load("planning-held-object-motion", "plan_linear_engagement.py")
    ctx=Context({"robot.get_ee_pose":{"pose":pose(z=.4)}})
    fixture={"pose":pose(x=.4,z=.3),"axis":{"x":0.0,"y":0.0,"z":1.0}}
    out=module.run(ctx,pose(z=.1),fixture,"shaft_into_aperture",{},
                   {"frame":"tcp","spheres":[]})
    waypoints=out["placement_plan"]["waypoints"]
    assert [item["cartesian"] for item in waypoints] == [False,True]
    assert waypoints[-1]["allow_goal_contact"] is True

def test_feature_mate_motion_plans_directly_to_approach():
    module=load("planning-held-object-motion", "plan_from_feature_mate.py")
    ctx=Context({"robot.get_ee_pose":{"pose":pose(z=.2)}})
    approach=pose(x=.5,z=.6)
    out=module.run(ctx,approach,{},
                   {"frame":"tcp","spheres":[{"center":[0.0,.12,0.0],"radius":.01}]})
    waypoints=out["reorientation_plan"]["waypoints"]
    assert [w["cartesian"] for w in waypoints] == [False]
    assert waypoints[0]["pose"] == approach
    assert "allow_start_contact" not in waypoints[0]

def test_loop_over_shaft_relation_produces_contact_seating_plan():
    module=load("planning-held-object-motion", "plan_feature_engagement.py")
    out=module.run(Context({}),pose(z=.3),pose(z=.2),pose(z=.1),"loop_over_shaft",{},
                   {"frame":"tcp","spheres":[]})
    waypoints=out["placement_plan"]["waypoints"]
    assert [w["mode"] for w in waypoints] == ["planned_joint","contact_seat","contact_seat"]
