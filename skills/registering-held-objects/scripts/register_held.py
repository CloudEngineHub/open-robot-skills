"""Express a held functional feature and collision cloud in the TCP frame."""
from typing import Any, TypedDict
import numpy as np
from gap import NodeContext
from scipy.spatial.transform import Rotation

class Output(TypedDict):
    feature_in_tcp: dict[str, Any]
    object_in_tcp: dict[str, Any]
    attached_object: dict[str, Any]
    registration_confidence: float

def _matrix(pose):
    p,q=pose["position"],pose["rotation"]; out=np.eye(4)
    out[:3,:3]=Rotation.from_quat([q["x"],q["y"],q["z"],q["w"]]).as_matrix(); out[:3,3]=[p["x"],p["y"],p["z"]]
    return out
def _pose(matrix):
    q=Rotation.from_matrix(matrix[:3,:3]).as_quat()
    return {"position":{"x":float(matrix[0,3]),"y":float(matrix[1,3]),"z":float(matrix[2,3])},
            "rotation":{"w":float(q[3]),"x":float(q[0]),"y":float(q[1]),"z":float(q[2])}}

def run(ctx: NodeContext, cameras: list[dict[str, Any]], reference_cloud: dict[str, Any],
        functional_feature: dict[str, Any], object_description: str,
        prior_feature_in_tcp: dict[str, Any] | None = None,
        prior_object_in_tcp: dict[str, Any] | None = None,
        prior_attached_object: dict[str, Any] | None = None,
        attachment_fit_type: str = "morphit") -> Output:
    ee=ctx.tool("robot.get_ee_pose")["pose"]; world_tcp=_matrix(ee)
    # The feature pose remains the geometric reference. Object re-detection
    # supplies a confidence signal and a current cloud when visible.
    best=None
    for camera in cameras:
        if "eye_in_hand" not in camera.get("name", ""): continue
        if prior_feature_in_tcp is not None and functional_feature.get("kind") == "tip":
            # Thin tips are unreliable language-segmentation targets. Segment
            # the complete held object and recover the distal endpoint in 3D.
            query = object_description
        else:
            query = (functional_feature.get("description")
                     if prior_feature_in_tcp is not None else object_description)
        result=ctx.tool("sam3.segment_text", image=camera["rgb"], query=query, max_results=2)
        if result.get("masks") and result.get("scores"):
            score=float(result["scores"][0])
            if best is None or score>best[0]: best=(score,camera,result["masks"][0])
    cloud=reference_cloud; confidence=0.25
    if best is not None and best[0]>=0.05:
        confidence=best[0]; camera,mask=best[1],best[2]
        observed=ctx.tool("geometry.mask_to_world_points", mask=np.asarray(mask,dtype=np.uint8),
                          depth=camera["depth"], intrinsics=camera["intrinsics"], camera_pose=camera["pose"])["points"]
        if len(np.asarray(observed["points"]).reshape(-1,3))>=8: cloud=observed
    reliable_registration = best is not None and best[0] >= 0.20
    registration_correction = np.zeros(3, dtype=np.float64)
    if prior_feature_in_tcp is not None and reliable_registration:
        # Near a fixture, whole-object matching is easily contaminated by the
        # fixture. Localize the functional feature directly, correct its center,
        # and preserve the already-reached feature orientation in the hand.
        world_feature = world_tcp @ _matrix(prior_feature_in_tcp)
        points = np.asarray(cloud["points"], dtype=np.float64).reshape(-1, 3)
        if functional_feature.get("kind") == "loop":
            normal=world_feature[:3,2]
            fit=ctx.tool(
                "geometry.fit_planar_feature", points=cloud,
                normal_hint={"x":float(normal[0]),"y":float(normal[1]),"z":float(normal[2])},
                fit_circle_center=True,
            )
            fitted_feature=_matrix(fit["pose"])
            world_feature[:3,:3]=fitted_feature[:3,:3]
            observed_center=np.array(
                [fit["pose"]["position"][key] for key in ("x","y","z")],
                dtype=np.float64,
            )
            normal=world_feature[:3,2]
            normal/=max(float(np.linalg.norm(normal)),1e-12)
            projected=points@normal
            axial_center=0.5*(float(np.quantile(projected,0.02))
                              +float(np.quantile(projected,0.98)))
            observed_center+=(axial_center-float(observed_center@normal))*normal
        else:
            observed_center=np.median(points,axis=0)
        predicted_center = world_feature[:3, 3]
        # A rigidly held feature cannot jump far from the pose predicted by
        # its prior TCP transform. Reject masks on the gripper or background.
        if float(np.linalg.norm(observed_center - predicted_center)) <= 0.04:
            registration_correction = observed_center - predicted_center
            world_feature[:3, 3] += registration_correction
        else:
            confidence = 0.25
    elif prior_feature_in_tcp is not None:
        world_feature = world_tcp @ _matrix(prior_feature_in_tcp)
    else:
        world_feature=_matrix(functional_feature["pose"])
        if reliable_registration:
            observed_center=np.median(np.asarray(cloud["points"],dtype=np.float64).reshape(-1,3),axis=0)
            reference_center=np.median(np.asarray(reference_cloud["points"],dtype=np.float64).reshape(-1,3),axis=0)
            correction=observed_center-reference_center
            if float(np.linalg.norm(correction))<=0.02: world_feature[:3,3]+=correction
            else: confidence=0.25
    feature_tcp=_pose(np.linalg.inv(world_tcp)@world_feature)
    for key in ("kind", "radius_inner", "radius_outer"):
        if key in functional_feature:
            feature_tcp[key] = functional_feature[key]
    attachment = prior_attached_object
    if attachment is None:
        attachment=ctx.tool(
            "geometry.cloud_to_attachment", points=cloud, tcp_pose=ee,
            fit_type=attachment_fit_type,
        )["attached_object"]
    if prior_object_in_tcp is not None:
        obj=world_tcp@_matrix(prior_object_in_tcp)
        obj[:3,3]+=registration_correction
    else:
        center=np.median(np.asarray(cloud["points"]).reshape(-1,3),axis=0)
        obj=np.eye(4); obj[:3,3]=center
    return {"feature_in_tcp":feature_tcp,"object_in_tcp":_pose(np.linalg.inv(world_tcp)@obj),
            "attached_object":attachment,"registration_confidence":float(confidence)}
