"""Find the spools with plain OpenCV: colour, shape, and height off the bench.

**Why not the text detector, for these.** SAM3 earns its place on the cable: a
deformable rod has no geometric description to match against, and the prompt is
the only handle there is. A spool is the opposite kind of object -- rigid, one
known colour, one known size, bolted down -- and re-running a network every frame
to rediscover it buys nothing that a colour threshold does not already give.

Measured side by side on the same frames, against the peg bodies' own positions:

    SAM3      3 found   2.8 mm from truth   0.01 mm jitter across frames
    OpenCV    3 found   2.9 mm from truth   0.00 mm jitter

Same accuracy. What the classical path adds is that it is deterministic, has no
model to load, no prompt to go stale when the hardware changes, and cannot
return four spools on one frame and three on the next.

**Height is what separates a spool from a terminal**, not colour and not area.
The plates are the same accent colour, and area depends on how far the fixture is
from the camera -- the far plate masks less than a near spool. What does not move
is that a spool's flange stands 23 mm off the bench and a terminal plate 10 mm.
That is a physical fact about the parts, measured through the same depth image
the centroid comes from.
"""

from __future__ import annotations

import numpy as np

#: Accent orange in HSV (OpenCV's 0-180 hue). Wide on value so a flange in the
#: arm's shadow survives, tight on hue because nothing else on this bench is
#: orange. Both are defaults of :func:`find_spools`, so a differently painted
#: fixture is a parameter and not an edit.
HSV_LO = (5, 120, 60)
HSV_HI = (25, 255, 255)

#: Smallest blob worth back-projecting [px]. Well under a spool at the furthest
#: the board is drawn, well over the specular flecks the bench throws.
MIN_AREA = 60

#: A flange is a filled, roughly round blob seen from any angle this camera
#: reaches. Both bounds are loose: they exist to drop the cable (long and thin)
#: and stray highlights, not to distinguish fixtures from each other.
MIN_FILL, MAX_ASPECT = 0.55, 1.9

#: How much larger than the median a blob's radius may be and still be one station.
#:
#: **This catches a MERGED blob, which no per-blob test can.** Two fixtures whose
#: colour regions touch come back as one component, and every gate above passes
#: it: the area is large but so is a near fixture's, the fill and aspect are fine
#: because two round things side by side are still roundish, and the height is the
#: taller of the two. What gives it away is size relative to its own population.
#:
#: Observed mid-episode on ``panel_l2``: five cleats at 19.0 to 19.3 mm and a
#: sixth "station" at **31.2 mm** sitting between the last cleat and the port,
#: which the planner would have woven around. On a settled frame the port drops
#: cleanly on height (12.1 mm against the gate's 15); it is only when something
#: bridges the two that the merged component inherits the cleat's height.
#:
#: Relative rather than absolute, because this file serves two fixtures whose
#: real radii differ by 60% -- 12 mm for an F1 spool, 19 for a cleat -- and any
#: constant that separated a merged pair on one board would reject single
#: stations on the other.
RADIUS_SPREAD = 1.6

#: Height above the work surface that separates a spool from a terminal [m].
#: Spool flange 23 mm, terminal plate 10 mm; the gate sits between them.
MIN_HEIGHT = 0.015

#: Systematic offset of a back-projected fixture centroid, along the camera's
#: own view ray [m].
#:
#: **Depth sees the surface, and a centroid of surface points is not the axis.**
#: Measured over six frames of a settled scene, both detectors read the spools
#: +2.71 mm and +2.82 mm in x with a standard deviation of 0.08 -- a bias, not
#: noise, and in the direction of the camera. It is the same error the cable's
#: centreline carries, for the same reason, and the same correction the
#: centreline fitter applies when it moves a rod's surface points onto its axis.
SURFACE_BIAS = 0.0028


def bench_z(depth, K, T) -> float:
    """World z of the work surface, from the depth image alone [m].

    The gate below needs to know what "off the bench" means, and taking that
    from the cell calibration would be a privileged fact leaking into a
    vision-only node. It does not have to be: the bench is the largest thing in
    frame by a wide margin, so the MEDIAN world height over every valid depth
    pixel *is* the bench, and the fixtures and arms that stand above it are far
    too few to move a median.
    """
    d = np.asarray(depth, dtype=np.float64)
    if d.ndim == 3:
        d = d[..., 0]
    v, u = np.nonzero(np.isfinite(d) & (d > 1.0e-4))
    if len(u) < 64:
        return 0.0
    step = max(1, len(u) // 20000)  # a median needs no more
    u, v = u[::step], v[::step]
    z = d[v, u]
    K = np.asarray(K, dtype=np.float64).reshape(3, 3)
    cam = np.stack(
        [(u - K[0, 2]) * z / K[0, 0], (v - K[1, 2]) * z / K[1, 1], z, np.ones_like(z)], axis=1
    )
    return float(np.median((np.asarray(T, dtype=np.float64).reshape(4, 4) @ cam.T).T[:, 2]))


def find_spools(
    rgb,
    depth,
    K,
    T,
    *,
    surface_z: float | None = None,
    hsv_lo: tuple[int, int, int] = HSV_LO,
    hsv_hi: tuple[int, int, int] = HSV_HI,
    min_area: int = MIN_AREA,
    min_height: float = MIN_HEIGHT,
):
    """Every spool in the frame, as ``(x, y, radius_m)`` world tuples.

    Returns them ordered along +y, which is the order the layout table is written
    in -- the same convention the SAM3 path uses, so either can feed the planner.
    """
    import cv2  # noqa: PLC0415

    if surface_z is None:
        surface_z = bench_z(depth, K, T)
    hsv = cv2.cvtColor(np.asarray(rgb, dtype=np.uint8), cv2.COLOR_RGB2HSV)
    m = cv2.inRange(hsv, np.array(hsv_lo, dtype=np.uint8), np.array(hsv_hi, dtype=np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m, 8)

    d = np.asarray(depth, dtype=np.float64)
    if d.ndim == 3:
        d = d[..., 0]
    K = np.asarray(K, dtype=np.float64).reshape(3, 3)
    T = np.asarray(T, dtype=np.float64).reshape(4, 4)
    eye = T[:3, 3]

    out = []
    for i in range(1, n):
        area = int(stats[i, cv2.CC_STAT_AREA])
        w, h = int(stats[i, cv2.CC_STAT_WIDTH]), int(stats[i, cv2.CC_STAT_HEIGHT])
        if area < int(min_area) or w == 0 or h == 0:
            continue
        if area / float(w * h) < MIN_FILL or max(w, h) / float(min(w, h)) > MAX_ASPECT:
            continue
        v, u = np.nonzero(lab == i)
        z = d[v, u]
        ok = np.isfinite(z) & (z > 1.0e-4)
        if ok.sum() < 8:
            continue
        u, v, z = u[ok], v[ok], z[ok]
        cam = np.stack(
            [(u - K[0, 2]) * z / K[0, 0], (v - K[1, 2]) * z / K[1, 1], z, np.ones_like(z)],
            axis=1,
        )
        world = (T @ cam.T).T[:, :3]
        centre = np.median(world, axis=0)
        if float(centre[2]) - float(surface_z) < float(min_height):
            continue  # a terminal plate, not a spool
        # Surface -> axis, along this fixture's own view ray. See SURFACE_BIAS.
        ray = centre - eye
        norm = float(np.linalg.norm(ray))
        if norm > 1.0e-9:
            centre = centre + SURFACE_BIAS * ray / norm
        radius = float(np.percentile(np.linalg.norm(world[:, :2] - centre[:2], axis=1), 90))
        out.append((float(centre[0]), float(centre[1]), radius))

    # DROP THE MERGED ONES, judged against the population rather than a
    # constant. See RADIUS_SPREAD. Needs three to have a median worth trusting;
    # below that a "population" is one fixture and its outlier.
    if len(out) >= 3:
        med = float(np.median([r for _, _, r in out]))
        keep = [p for p in out if p[2] <= RADIUS_SPREAD * med]
        if len(keep) != len(out):
            for x, y, r in out:
                if r > RADIUS_SPREAD * med:
                    print(
                        f"[posts]   dropping a {r * 1000:.0f} mm blob at "
                        f"({x:.3f}, {y:+.3f}) -- {r / med:.1f}x the median "
                        f"{med * 1000:.0f} mm, so two fixtures merged into one",
                        flush=True,
                    )
        out = keep
    return sorted(out, key=lambda p: p[1])
