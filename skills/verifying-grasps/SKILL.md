---
name: verifying-grasps
description: Lifts a just-grasped object a few centimetres, reobserves it from the wrist camera or the overhead camera, and confirms from metric depth that it clears the support surface and rides within reach of the hand, routing not_held otherwise. Use when a grasp closed on a small or thin object and the graph must know before carrying, registering, or inserting whether anything is actually in the gripper.
license: Apache-2.0
compatibility: requires gap>=0.1
metadata: {category: verification, tags: [grasp, verification, rgb-d, wrist-camera]}
gap:
  requires: {connector: [robot.describe_workspace]}
  allowed_tools:
    - robot.get_ee_pose
    - robot.go_to_pose_cartesian
    - robot.get_observation
    - robot.describe_workspace
    - sam3.segment_text
    - geometry.mask_to_world_points
  required_inputs:
    object_description: string
  produces_outputs:
    verified: bool
    observed_center: Vec3
  exit_conditions:
    verified: The object was seen after the lift, above the surface and near the hand.
    not_held: Nothing matching was seen, or what was seen stayed on the surface or far from the hand.
  canonical_scripts:
    - verify_grasp: scripts/verify_grasp.py
  streaming: false
---

# verifying-grasps

Use immediately after a grasp closes and before any carry. `verify_grasp`
raises the end effector by `lift_m` along world Z with a Cartesian move,
takes a fresh observation, and looks for `object_description` (then
`marker_description`, when given) first in the camera whose name contains
`wrist_camera_keyword` and then in `overhead_camera_name`. The first
segmentation at or above `score_min` is back-projected; the grasp is
`verified` only when the cloud has at least `min_points` valid depth points,
its median lies at least `min_above_table_m` above
`robot.describe_workspace().surface_z`, and within `max_hand_distance_m` of
the lifted hand. Every failed gate returns `route: not_held` with a
`reason`; only a rig without a wrist camera raises.

`score_min` is deliberately low: a small held object occupies only a few
dozen wrist pixels and true positives score in the hundredths. Confidence
admits a candidate; geometry decides.

## Boundaries

- The lift is the only motion; the object is neither released nor moved
  elsewhere.
- No gripper-width or force reading is consulted; those belong to the
  connector's own grasp check when it has one.
