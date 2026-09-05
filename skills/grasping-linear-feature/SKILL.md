---
name: grasping-linear-feature
description: Fit the full 3D axis of an elongated segmented feature and grasp it with a perpendicular, inclination-aware parallel-jaw pose. Use for handles, rods, shafts, tools, utensils, and other linear parts whose pitch or roll matters; use grasping-short-axis instead when an upright OBB and vertical approach are sufficient.
license: Apache-2.0
compatibility: requires gap>=0.1
metadata: {category: grasping, tags: [grasping, manipulation, linear-feature, inclined, point-cloud]}
gap:
  allowed_tools:
    - geometry.fit_linear_feature
    - robot.go_to_pose
    - robot.get_observation
    - robot.open_gripper
    - robot.close_gripper
  required_inputs:
    target_cloud: PointCloud
  produces_outputs:
    grasp_pose: Se3Pose
    pregrasp_pose: Se3Pose
    approach_axis: Vec3
    ee_pose_at_grasp: Se3Pose
  exit_conditions:
    grasped: The gripper closed after reaching the inclination-aware grasp pose.
    failed: Feature fitting or full-orientation motion failed.
  hard_rules:
    - Use geometry.fit_linear_feature on the segmented target_cloud; do not infer the 3D axis from an upright OBB.
    - Use the canonical compute_grasp script. It makes the approach perpendicular to the fitted axis and as close to world-down as possible.
    - Execute pregrasp before grasp so the wrist reaches its full orientation before entering the object.
    - Use a backend that honors the complete quaternion, such as newton_roll or pyroki. A roll-free backend defeats the axis alignment.
    - Begin with robot.open_gripper and capture ee_pose_at_grasp after descend and before close.
  canonical_scripts:
    - compute_grasp: scripts/axis_aware_grasp_pose.py
  examples:
    - title: Canonical full-3D linear-feature grasp
      path: examples/canonical_subgraph.json
  streaming: false
---

# Grasping a full-3D linear feature

Use this skill when a segmented graspable part may be inclined in 3D. The
point cloud is reduced to a robust principal axis by
`geometry.fit_linear_feature`. The canonical script then builds a gripper
frame with its local Y axis along the feature and its local Z approach axis
perpendicular to it. The approach is chosen as close to world-down as the
perpendicular constraint permits.

Recommended flow:

```text
open → fit_axis → compute_grasp → pregrasp → descend → observe → close → grasped
```

`surface_inset` moves the fingertip target from the feature center along the
approach direction; `standoff` places the pregrasp back along the same line.
Keep both configurable because gripper geometry and point-cloud completeness
vary across platforms.

This skill generates and executes a grasp pose; it does not decide which
semantic part to grasp or verify object identity. Upstream perception should
provide a point cloud containing only the selected linear feature.
