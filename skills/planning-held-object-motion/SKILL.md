---
name: planning-held-object-motion
description: "Plans a clearance-first motion sequence for an already-held rigid object: lift away from support, choose the smallest feasible symmetry-equivalent reorientation, translate above a fixture while holding orientation, then approach and engage linearly. Use for insertion, hanging, packing, racking, and constrained sorting; use an execution skill afterward."
compatibility: requires gap>=0.1
metadata: {category: planning, tags: [motion-planning, held-object, clearance, insertion, hanging, packing, sorting]}
gap:
  requires: {connector: [motion.plan_joint]}
  allowed_tools:
    - robot.get_ee_pose
    - motion.plan_joint
  required_inputs:
    held_feature_in_tcp: Se3Pose
    fixture_feature: FunctionalFeature
    relation: string
    world_config: WorldConfig
    attached_object: AttachedObject
  produces_outputs:
    reorientation_plan: PoseSequence
    placement_plan: PoseSequence
  exit_conditions:
    planned: Clearance-first carry and linear engagement plans were produced.
    blocked: No feasible symmetry-equivalent held-object orientation was found.
  canonical_scripts:
    - plan_clearance_motion: scripts/plan_clearance_motion.py
    - plan_from_feature_mate: scripts/plan_from_feature_mate.py
    - plan_feature_engagement: scripts/plan_feature_engagement.py
    - plan_linear_engagement: scripts/plan_linear_engagement.py
  streaming: false
---

# planning-held-object-motion

Turn a feature mate into safe phases for a held object:

```text
lift away from support
  -> rotate minimally in clear space
  -> translate above the fixture with orientation fixed
  -> approach and engage linearly
```

The skill plans; it does not move the robot or release the object. Follow it
with `executing-held-object-motion` to execute the carry plan and
`executing-feature-mating` to execute the engagement plan.

## Choosing direct or staged carry

Use `plan_from_feature_mate` when the object is already clear of its support,
the requested orientation change is small, and a single attached-object plan
to `approach_pose` is appropriate.

Use `plan_clearance_motion` when the object is elongated, must become upright,
or a large rotation near the support would sweep the object through the table.
It derives a lift from the farthest attached sphere, selects the smallest
reachable symmetry-equivalent rotation, rotates while clear, and translates
the aligned feature to a staging location on the free side of the fixture.
The staging location is derived from the fixture geometry and attached-object
extent; the caller does not supply task-specific staging coordinates.

If uncertain, choose staged carry for a large orientation change or long held
object. Do not add arbitrary midpoint waypoints and do not remove collision
geometry to make direct planning succeed.

`plan_feature_engagement` converts the typed relation and the approach,
engaged, and mate poses into execution semantics. `loop_over_shaft` uses
contact-controlled engagement and seating because crossing the hook is the
intended contact operation; aperture insertion uses goal-contact linear
waypoints. The task graph therefore does not decide ad hoc whether a contact
servo is needed.

## Recommended skill sequence

1. Use `registering-held-objects` to obtain `held_feature_in_tcp` and
   `attached_object`.
2. Use `computing-feature-mating-poses` when explicit approach/engaged/mate poses are
   needed.
3. Select direct or staged carry using the criteria above.
4. Execute the returned `reorientation_plan` with
   `executing-held-object-motion`.
5. Execute `placement_plan` with `executing-feature-mating`.

Waypoint modes are semantic: `contact_transition` leaves the initial support,
`planned_joint` is collision-aware free-space motion, `planned_linear` locks
orientation for a straight local leg, and `contact_seat` is the final
intentional-contact servo.

## Boundaries

- Upstream perception supplies the held feature in TCP coordinates and the
  fixture pose/axis. Registration supplies attached-object collision geometry.
- Inputs describe functional geometry and a relation rather than an object
  class or task name.
- Free roll about the fixture axis is treated as symmetry. Candidates are
  checked for reachability and the smallest feasible TCP rotation is selected.
- `support_normal` may be supplied when the escape direction is known; it
  defaults to world up for a horizontal support surface.
- Relationship-specific clearances may be supplied by the graph when fixture
  depth is observable or specified. Conservative geometric defaults are used
  otherwise.
- The final crossing is linear and may allow goal contact. All earlier motion
  remains collision-aware free-space motion.
