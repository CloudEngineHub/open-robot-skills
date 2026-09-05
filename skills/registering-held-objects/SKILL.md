---
name: registering-held-objects
description: Reobserve a grasped rigid object, estimate its functional feature in the TCP frame, and construct attached collision geometry.
license: Apache-2.0
compatibility: requires gap>=0.1
metadata: {category: perception, tags: [in-hand, registration, wrist-camera, collision]}
gap:
  allowed_tools: [robot.get_ee_pose, sam3.segment_text, geometry.mask_to_world_points, geometry.fit_planar_feature, geometry.cloud_to_attachment]
  required_inputs: {reference_cloud: PointCloud, functional_feature: FunctionalFeature, object_description: string}
  produces_outputs: {feature_in_tcp: Se3Pose, object_in_tcp: Se3Pose, attached_object: AttachedObject, registration_confidence: float}
  exit_conditions:
    registered: A reliable transform was measured.
    fallback: The grasp-time transform was retained.
    lost: The object is no longer localized.
  canonical_scripts:
    - register_held: scripts/register_held.py
  streaming: false
---

# registering-held-objects

`cameras` is a list of `CameraFrame` -- an `Observation`'s `cameras` field.
It is stated here rather than under `required_inputs` because the type
registry names no bare list of frames.

Use immediately after grasping or again at a pre-contact pose. For pre-contact
realignment, pass the prior feature-in-TCP and attachment: wrist views localize
the functional feature directly, fit a loop's geometric center and plane rather
than the centroid of its visible arc, and retain the collision model.
Grasp-time geometry remains an explicit
low-confidence fallback.

The default `attachment_fit_type="morphit"` fits 64 collision spheres to a
watertight convex hull of the observed object and contracts their radii by 2
mm. This is the standard CuRobo attachment representation for both transport
and constrained fixture motion. `surface` and `voxel` remain available only
for explicit fitting experiments; ordinary workflows should keep the default.
