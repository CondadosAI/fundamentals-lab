"""Circles on a pool table: M2's three photos, its 41 hand-labelled balls, and HoughCircles."""

from __future__ import annotations

import json

import cv2
import numpy as np

from fundamentals_lab.config import POOL_DIR, POOL_MEDIAN, POOL_MIN_DIST, POOL_PHOTOS, POOL_RADII


def labels() -> dict[str, list[tuple[int, int, int]]]:
    return {k: [tuple(v) for v in vs] for k, vs in json.loads((POOL_DIR / "labels.json").read_text()).items()}


def load(name: str) -> np.ndarray:
    img = cv2.imread(str(POOL_DIR / f"{name}.webp"))
    if img is None:
        raise FileNotFoundError(f"{name}.webp: run boundary-download first")
    return img


def grey(img: np.ndarray) -> np.ndarray:
    return cv2.medianBlur(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), POOL_MEDIAN)


def detect(img: np.ndarray, method: int, dp: float, param1: float, param2: float) -> list:
    c = cv2.HoughCircles(grey(img), method, dp, POOL_MIN_DIST, param1=param1, param2=param2,
                         minRadius=POOL_RADII[0], maxRadius=POOL_RADII[1])
    return [] if c is None else [tuple(map(float, v)) for v in c[0]]


def match(circles, truth) -> dict:
    """M2's rule: one detection per ball, closest pairs first, within the ball's radius (8 px min)."""
    pairs = sorted((float(np.hypot(x - lx, y - ly)), i, j)
                   for i, (lx, ly, _) in enumerate(truth) for j, (x, y, _) in enumerate(circles))
    hit_l, hit_c, rerr = set(), set(), []
    for d, i, j in pairs:
        if i in hit_l or j in hit_c or d >= max(truth[i][2], 8):
            continue
        hit_l.add(i)
        hit_c.add(j)
        rerr.append(abs(circles[j][2] - truth[i][2]))
    return {"found": len(hit_l), "balls": len(truth), "false": len(circles) - len(hit_c),
            "radius_errors": rerr,
            "missed": [list(truth[i]) for i in range(len(truth)) if i not in hit_l],
            "false_at": [[round(v, 1) for v in circles[j]] for j in range(len(circles)) if j not in hit_c]}


def run_all(method: int, dp: float, param1: float, param2: float, photos=POOL_PHOTOS) -> dict:
    L = labels()
    per, errs = {}, []
    for name in photos:
        m = match(detect(load(name), method, dp, param1, param2), L[name])
        errs += m.pop("radius_errors")
        per[name] = m
    found = sum(p["found"] for p in per.values())
    balls = sum(p["balls"] for p in per.values())
    false = sum(p["false"] for p in per.values())
    return {"found": found, "balls": balls, "false": false,
            "median_radius_error_px": round(float(np.median(errs)), 2) if errs else None,
            "per_photo": per}


def fixed_radius_accumulator(g: np.ndarray, r: int, canny=(50, 150)) -> np.ndarray:
    """The textbook circle Hough at one radius: every edge pixel votes for the centres
    r away along its gradient, both ways, at the nearest pixel."""
    e = cv2.Canny(g, *canny)
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1)
    ys, xs = np.nonzero(e)
    m = np.hypot(gx[ys, xs], gy[ys, xs]) + 1e-9
    ux, uy = gx[ys, xs] / m, gy[ys, xs] / m
    acc = np.zeros(g.shape, np.int32)
    for sgn in (1, -1):
        cx = np.rint(xs + sgn * r * ux).astype(int)
        cy = np.rint(ys + sgn * r * uy).astype(int)
        ok = (cx >= 0) & (cy >= 0) & (cx < g.shape[1]) & (cy < g.shape[0])
        np.add.at(acc, (cy[ok], cx[ok]), 1)
    return acc
