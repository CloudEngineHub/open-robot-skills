---
name: executing-held-object-motion
description: Execute an ordered, collision-aware pose sequence for an already-grasped rigid object without releasing it. Supports lifting from support, clear-space rotation, fixture transit, and orientation-locked local motion before insertion, mating, packing, or sorting.
compatibility: requires gap>=0.1
metadata: {category: motion, tags: [motion, held-object, execution, reorientation, insertion, packing]}
gap:
  requires: {connector: [motion.plan_linear, motion.plan_to_pose]}
  allowed_tools:
    - motion.plan_to_pose
    - motion.plan_linear
    - robot.execute_trajectory
    - robot.go_to_pose
    - robot.go_to_pose_cartesian
  exit_conditions:
    reoriented: The held-object waypoint sequence completed while the gripper remained closed.
    blocked: A waypoint motion failed before the requested orientation was reached.
  required_inputs:
    reorientation_plan: PoseSequence
  canonical_scripts:
    - execute_reorientation: scripts/execute_reorientation.py
  streaming: false
---

# executing-held-object-motion

Execute motion for a rigid object that is already held. The skill preserves the
grasp: it does not open or close the gripper. Its input is an ordered
`PoseSequence` containing clearance, transit, and orientation poses.

## When to use

- Stand an elongated object upright before putting it through an opening.
- Turn a tool from a table grasp into a hanging or fixture-mating orientation.
- Align a keyed object before insertion.
- Reorient an item before constrained packing or sorting.

Do not use it for an ordinary bin drop when `transporting-objects` can lift,
translate, descend, and release directly.

## Recommended subgraph

```text
execute_reorientation -> reoriented
```

`execute_reorientation` runs the canonical script with
`reorientation_plan = Ref("in.reorientation_plan")`. Each waypoint is a
typed record. It forwards the plan's `world_config` and `attached_object` to
every planned leg and never replaces a rejected collision-aware plan with an
unchecked move. Use `contact_transition` only for the initial support escape;
use `planned_joint` for clear-space rotation/transit and `planned_linear` for
an orientation-locked straight leg.

## Boundaries

- Upstream geometry or perception determines the desired object orientation
  and sufficient clearance. This skill executes that typed request.
- The sequence must describe end-effector poses, not object poses.
- The skill never releases the object and never queries cameras, simulation
  state, task goals, evaluators, or reward.
- Full 3-DOF orientation changes require a robot IK backend that honors wrist
  roll. A position/approach-axis-only backend cannot reliably stand a syringe
  or long tool upright.
