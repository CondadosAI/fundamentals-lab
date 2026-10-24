"""Unit 3.3's opening animations, drawn from the unit's own runs.

- `matches`: the Barcelona pair, its SIFT points, every nearest-neighbour candidate, then
  the ones the ratio test keeps, coloured by the ORB reference (green right, red wrong).
- `follow`: photo A against a copy of itself that turns and shrinks; the matches keep
  landing on the same harbour while the copy moves.
"""

from __future__ import annotations

import cv2
import numpy as np

from fundamentals_lab.alignment.media import encode
from fundamentals_lab.config import MATCH_TOL_PX, RATIO
from fundamentals_lab.features import scene
from fundamentals_lab.features.figures import AMBER, BG, FIG_DIR, GREEN, RED, _dim, label

MEDIA_DIR = FIG_DIR / "media"


def _text(img, text, org, scale=0.5, colour=(245, 245, 245)):
    label(img, text, org, scale, colour)


def matches(width: int = 760):
    a_bgr, b_bgr = scene.pair_bgr()
    a, b = scene.pair()
    H, _ = scene.reference_homography()
    sift = scene.sift()
    ka, da = sift.compute(a, scene.sift_points(a))
    kb, db = sift.compute(b, scene.sift_points(b))
    kn = scene.knn(da, db)
    pa = np.array([ka[m.queryIdx].pt for m, _ in kn])
    pb = np.array([kb[m.trainIdx].pt for m, _ in kn])
    r = np.array([m.distance / n.distance for m, n in kn])
    ok = np.linalg.norm(scene.apply(H, pa) - pb, axis=1) <= MATCH_TOL_PX
    keep = r < RATIO
    gap = 8
    s = (width - gap) / (2 * a.shape[1])
    small = lambda im: cv2.resize(im, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)  # noqa: E731
    A, B = _dim(small(a_bgr), 0.8), _dim(small(b_bgr), 0.8)
    base = np.hstack([A, np.full((A.shape[0], gap, 3), BG, np.uint8), B])
    off = A.shape[1] + gap
    P = lambda p: (int(round(p[0] * s)), int(round(p[1] * s)))  # noqa: E731
    Q = lambda p: (int(round(p[0] * s)) + off, int(round(p[1] * s)))  # noqa: E731
    frames = []
    dots = base.copy()
    for k in ka:
        cv2.circle(dots, P(k.pt), 1, AMBER, -1)
    for k in kb:
        cv2.circle(dots, Q(k.pt), 1, AMBER, -1)
    f = dots.copy()
    _text(f, f"SIFT points: {len(ka)} and {len(kb)}", (10, 24))
    frames += [f] * 18
    rng = np.random.default_rng(0)
    cand = rng.permutation(len(pa))[:150]
    f = dots.copy()
    for i in cand:
        cv2.line(f, P(pa[i]), Q(pb[i]), AMBER, 1, cv2.LINE_AA)
    _text(f, "every point paired with its nearest descriptor", (10, 24))
    frames += [f] * 18
    f = base.copy()
    for i in np.nonzero(keep)[0]:
        cv2.line(f, P(pa[i]), Q(pb[i]), GREEN if ok[i] else RED, 1, cv2.LINE_AA)
    _text(
        f,
        f"ratio test keeps {int(keep.sum())}: "
        f"{int((keep & ok).sum())} right, {int((keep & ~ok).sum())} wrong",
        (10, 24),
    )
    frames += [f] * 30
    return frames, 10


def follow(width: int = 760, n: int = 60):
    a, _ = scene.pair()
    a_bgr = scene.pair_bgr()[0]
    sift = scene.sift()
    ka, da = sift.compute(a, scene.sift_points(a))
    gap = 8
    s = (width - gap) / (2 * a.shape[1])
    A = _dim(cv2.resize(a_bgr, None, fx=s, fy=s, interpolation=cv2.INTER_AREA), 0.8)
    h, w = A.shape[:2]
    frames = []
    for t in np.linspace(0, 1, n):
        deg = 10 + 80 * t
        zoom = 0.9 - 0.4 * t
        img, M = scene.rotate_scale(a, deg, zoom)
        col, _ = scene.rotate_scale(a_bgr, deg, zoom)
        kb, db = sift.compute(img, scene.sift_points(img))
        ms = [m for m, nn in scene.knn(da, db) if m.distance < RATIO * nn.distance]
        # the copy, fitted into a w x h box
        f2 = min(w / col.shape[1], h / col.shape[0])
        C = cv2.resize(col, None, fx=f2, fy=f2, interpolation=cv2.INTER_AREA)
        box = np.full((h, w, 3), BG, np.uint8)
        y0, x0 = (h - C.shape[0]) // 2, (w - C.shape[1]) // 2
        box[y0 : y0 + C.shape[0], x0 : x0 + C.shape[1]] = _dim(C, 0.8)
        frame = np.hstack([A, np.full((h, gap, 3), BG, np.uint8), box])
        rng = np.random.default_rng(0)
        for m in [ms[i] for i in sorted(rng.choice(len(ms), min(80, len(ms)), replace=False))]:
            p, q = ka[m.queryIdx].pt, kb[m.trainIdx].pt
            right = np.linalg.norm(scene.apply(M, [p])[0] - q) <= MATCH_TOL_PX
            cv2.line(
                frame,
                (int(p[0] * s), int(p[1] * s)),
                (int(q[0] * f2) + x0 + w + gap, int(q[1] * f2) + y0),
                GREEN if right else RED,
                1,
                cv2.LINE_AA,
            )
        _text(
            frame,
            f"copy turned {deg:.0f} deg, shrunk to {zoom:.2f}: {len(ms)} matches",
            (10, 22),
            0.5,
        )
        frames.append(frame)
    frames += [frames[-1]] * 12
    return frames, 10


def render_all() -> dict:
    out = {}
    for name, fn in (("sift-matches", matches), ("sift-follow", follow)):
        frames, fps = fn()
        out[name] = encode(frames, fps, MEDIA_DIR / f"{name}.webp")
    return out
