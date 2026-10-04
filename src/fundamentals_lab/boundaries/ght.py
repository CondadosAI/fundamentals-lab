"""Lesson 4: the generalized Hough transform, Ballard's R-table, in NumPy and in OpenCV."""

from __future__ import annotations

import cv2
import numpy as np

from fundamentals_lab.config import BOUNDARY_CANNY, GHT


def ring_template(blur: np.ndarray, circle, inner: float | None) -> np.ndarray:
    """A square crop around one transducer; with `inner`, the mesh inside it is flattened."""
    cx, cy, r = circle
    R = int(r) + 6
    t = blur[int(cy) - R : int(cy) + R, int(cx) - R : int(cx) + R].copy()
    if inner:
        yy, xx = np.mgrid[: 2 * R, : 2 * R]
        m = np.hypot(xx - R, yy - R) < inner * r
        t[m] = int(np.median(t[~m]))
    return t


# OpenCV's fastAtan2: a degree-7 polynomial, about 0.3 degrees from the true angle.
_P = [
    np.float32(v * 180 / np.pi)
    for v in (0.9997878412794807, -0.3258083974640975, 0.1555786518463281, -0.04432655554792128)
]


def fast_atan2(y: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Degrees in [0, 360), computed as cv::fastAtan2 does."""
    y = y.astype(np.float32)
    x = x.astype(np.float32)
    ax, ay = np.abs(x), np.abs(y)
    eps = np.float32(np.finfo(np.float64).eps)
    big = ax >= ay
    c = np.where(big, ay / (ax + eps), ax / (ay + eps)).astype(np.float32)
    c2 = c * c
    a = (((_P[3] * c2 + _P[2]) * c2 + _P[1]) * c2 + _P[0]) * c
    a = np.where(big, a, np.float32(90) - a)
    a = np.where(x < 0, np.float32(180) - a, a)
    a = np.where(y < 0, np.float32(360) - a, a)
    return a.astype(np.float32)


def _edges_and_bins(gray: np.ndarray, levels: int, parity: bool):
    edges = cv2.Canny(gray, *BOUNDARY_CANNY)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1)
    ys, xs = np.nonzero(edges)
    if parity:
        ang = fast_atan2(gy[ys, xs], gx[ys, xs])
        bins = np.rint(ang * np.float32(levels / 360)).astype(int)  # 0..levels inclusive
    else:
        ang = np.degrees(np.arctan2(gy[ys, xs], gx[ys, xs])) % 360
        bins = (ang * levels / 360).astype(int) % levels
    return xs, ys, bins


def reference_point(template: np.ndarray, parity: bool = True):
    """OpenCV uses the integer centre (w // 2, h // 2); the geometric one is half a pixel off."""
    h, w = template.shape
    return (w // 2, h // 2) if parity else ((w - 1) / 2, (h - 1) / 2)


def r_table(
    template: np.ndarray, levels: int = GHT["levels"], parity: bool = True
) -> tuple[dict, tuple]:
    """{orientation bin: [(dx, dy), ...]}, from each template edge pixel to the reference point."""
    xs, ys, bins = _edges_and_bins(template, levels, parity)
    ref = reference_point(template, parity)
    table: dict[int, list] = {}
    for x, y, b in zip(xs.tolist(), ys.tolist(), bins.tolist(), strict=False):
        table.setdefault(b, []).append((ref[0] - x, ref[1] - y))
    return table, ref


def vote(
    image: np.ndarray,
    table: dict,
    levels: int = GHT["levels"],
    dp: int = GHT["dp"],
    parity: bool = True,
) -> np.ndarray:
    """Every image edge pixel votes for the reference points its R-table row allows."""
    xs, ys, bins = _edges_and_bins(image, levels, parity)
    h, w = image.shape
    acc = np.zeros((h // dp + 3, w // dp + 3), np.int32)
    inv = np.float32(1 / dp)
    for b in np.unique(bins):
        rows = table.get(int(b))
        if not rows:
            continue
        sel = bins == b
        off = np.asarray(rows, np.float32)
        px = xs[sel].astype(np.float32)[:, None] + off[None, :, 0]
        py = ys[sel].astype(np.float32)[:, None] + off[None, :, 1]
        cx = np.rint(px * inv).astype(int).ravel()
        cy = np.rint(py * inv).astype(int).ravel()
        ok = (cx >= 0) & (cy >= 0) & (cx < acc.shape[1]) & (cy < acc.shape[0])
        np.add.at(acc, (cy[ok], cx[ok]), 1)
    return acc


def peaks(
    acc: np.ndarray, n: int, min_dist: int, dp: int = GHT["dp"]
) -> list[tuple[int, int, int]]:
    """Strongest n cells at least `min_dist` px apart, as (votes, x, y) in image pixels."""
    a = acc.copy()
    out = []
    for _ in range(n):
        y, x = np.unravel_index(int(a.argmax()), a.shape)
        v = int(a[y, x])
        if v <= 0:
            break
        out.append((v, int(x * dp), int(y * dp)))
        r = max(1, min_dist // dp)
        a[max(0, y - r) : y + r + 1, max(0, x - r) : x + r + 1] = 0
    return out


def opencv_ballard(image: np.ndarray, template: np.ndarray) -> list[tuple[int, int, int]]:
    g = cv2.createGeneralizedHoughBallard()
    g.setCannyLowThresh(BOUNDARY_CANNY[0])
    g.setCannyHighThresh(BOUNDARY_CANNY[1])
    g.setMinDist(GHT["min_dist"])
    g.setLevels(GHT["levels"])
    g.setDp(GHT["dp"])
    g.setVotesThreshold(GHT["votes_threshold"])
    g.setTemplate(template)
    pos, votes = g.detect(image)
    if pos is None:
        return []
    return [
        (int(v[0]), int(round(p[0])), int(round(p[1])))
        for p, v in zip(pos[0], votes[0], strict=False)
    ]
