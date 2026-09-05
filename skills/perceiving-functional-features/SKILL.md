---
name: perceiving-functional-features
description: Locate a language-described functional part of a segmented rigid object or fixture and fit a typed loop, shaft, tip, aperture, surface, or region feature.
license: Apache-2.0
compatibility: requires gap>=0.1
metadata: {category: perception, tags: [perception, affordance, geometry, rgb-d]}
gap:
  allowed_tools: [sam3.segment_text, geometry.mask_to_world_points, geometry.fit_planar_feature, geometry.fit_linear_feature]
  required_inputs: {parent_cloud: PointCloud, parent_mask: Mask, feature_description: string, feature_type: string}
  produces_outputs: {feature: FunctionalFeature, feature_mask: Mask, feature_cloud: PointCloud}
  exit_conditions:
    found: One geometrically valid feature was fitted.
    ambiguous: Multiple materially different candidates remain.
    not_found: No valid feature was visible.
  canonical_scripts:
    - perceive_feature: scripts/perceive_feature.py
  streaming: false
---

# perceiving-functional-features

`cameras` is a list of `CameraFrame` -- an `Observation`'s `cameras` field.
It is stated here rather than under `required_inputs` because the type
registry names no bare list of frames.

Use after whole-object perception when manipulation depends on a loop, shaft,
tip, aperture, surface, or region. The caller supplies a natural-language part
description and the expected geometry family. The skill segments candidates,
fits the requested geometry, and validates the result against the parent
object's 3D bounds so a nearby object or background region is not accepted.

## When to use

- A manipulation target is a specific functional part rather than the entire
  object, such as a tool loop, insertion tip, peg, or aperture.
- Downstream mating or motion planning requires a metric center, axis, normal,
  or radius.

## Inputs

- `feature_description` names the visible part, for example `ring end of the
  tool` or `circular disposal opening`.
- `feature_type` selects one supported geometry family: loop, shaft, tip,
  aperture, surface, or region.
- `parent_cloud` and `parent_mask` constrain the search to the known object or
  fixture.

## Boundaries

- Use whole-object perception before this skill.
- Return `ambiguous` when materially different valid candidates remain; do not
  choose one using task-specific coordinates.
- This skill localizes geometry. It does not compute mating poses or execute
  motion.
