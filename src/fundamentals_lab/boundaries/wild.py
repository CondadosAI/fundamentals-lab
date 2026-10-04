"""Each lesson's measurement repeated on a second, real scene.

The cloister (Sindugab, Wikimedia Commons, CC0) for lessons 1 and 2, the stationery on a
bench (Brigitte Tohm, Unsplash via Commons, CC0) for lesson 3, and VisA's `candle` category
(Zou et al., ECCV 2022, Amazon, CC BY 4.0: four tea lights per photo) for lesson 4.
"""

from __future__ import annotations

import cv2
import numpy as np

from fundamentals_lab.boundaries import fitting, hough
from fundamentals_lab.config import DATA_DIR

CLOISTER = DATA_DIR / "wild" / "cloister.jpg"
DOCUMENT = DATA_DIR / "wild-document" / "stationery_bench_unsplash.jpg"
CANDLES = DATA_DIR / "visa-candle"
WIDTH = 900


def _load(path, width=WIDTH):
    img = cv2.imread(str(path))
    if img is None:
        raise FileNotFoundError(path)
    s = width / img.shape[1]
    return cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)


def _edges(bgr, sigma=1.4, canny=(50, 150)):
    g = cv2.GaussianBlur(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY), (0, 0), sigma)
    return g, cv2.Canny(g, *canny)


def cloister_fit() -> dict:
    """L1: LS against TLS on the cloister's strongest near-vertical edges."""
    bgr = _load(CLOISTER)
    _, e = _edges(bgr)
    acc = hough.accumulator(e)
    out = []
    ys, xs = np.nonzero(e)
    for v, ri, ti in hough.local_maxima(acc, 80):
        if not (ti <= 2 or ti >= 178):
            continue
        rho, t = hough.rho_of(ri, acc), np.radians(ti)
        d = np.abs(xs * np.cos(t) + ys * np.sin(t) - rho)
        P = np.c_[xs[d <= 3], ys[d <= 3]].astype(float)
        if len(P) < 80:
            continue
        ls = fitting.least_squares(P)["angle_deg"] % 180
        tls = fitting.total_least_squares(P)["angle_deg"] % 180
        out.append({"votes": v, "points": len(P), "ls": round(ls, 2), "tls": round(tls, 2)})
        if len(out) == 10:
            break
    ls_off = [abs(90 - o["ls"]) for o in out]
    tls_off = [abs(90 - o["tls"]) for o in out]
    return {
        "edges": out,
        "ls_off_vertical_median": round(float(np.median(ls_off)), 2),
        "tls_off_vertical_median": round(float(np.median(tls_off)), 2),
        "ls_off_vertical_max": round(float(np.max(ls_off)), 2),
        "tls_off_vertical_max": round(float(np.max(tls_off)), 2),
    }


def cloister_vanishing() -> dict:
    """L2: the strongest oblique lines of a one-point-perspective photo meet near one point."""
    bgr = _load(CLOISTER)
    _, e = _edges(bgr)
    acc = hough.accumulator(e)
    lm = hough.local_maxima(acc, 80)
    oblique = [
        (v, hough.rho_of(r, acc), np.radians(t))
        for v, r, t in lm
        if 15 <= t <= 75 or 105 <= t <= 165
    ][:12]
    pts = []
    for i in range(len(oblique)):
        for j in range(i + 1, len(oblique)):
            _, r1, t1 = oblique[i]
            _, r2, t2 = oblique[j]
            A = np.array([[np.cos(t1), np.sin(t1)], [np.cos(t2), np.sin(t2)]])
            if abs(np.linalg.det(A)) < 0.2:
                continue
            pts.append(np.linalg.solve(A, [r1, r2]))
    pts = np.array(pts)
    med = np.median(pts, axis=0)
    dist = np.hypot(*(pts - med).T)
    return {
        "lines_over_80": len(lm),
        "oblique_used": len(oblique),
        "intersections": len(pts),
        "median_point": [round(float(v), 1) for v in med],
        "frame": list(bgr.shape[1::-1]),
        "within_20px": round(float((dist <= 20).mean()), 3),
        "median_distance_px": round(float(np.median(dist)), 1),
    }


def document_segments() -> dict:
    """L3: the document's four sides as segments, and its tilt, from the probabilistic Hough."""
    bgr = _load(DOCUMENT)
    _, e = _edges(bgr)
    full = int((e > 0).sum()) * 180
    seg = cv2.HoughLinesP(e, 1, np.pi / 180, 90, minLineLength=80, maxLineGap=5)
    seg = np.empty((0, 4), int) if seg is None else seg.reshape(-1, 4)
    L = np.hypot(seg[:, 2] - seg[:, 0], seg[:, 3] - seg[:, 1])
    ang = np.degrees(np.arctan2(seg[:, 3] - seg[:, 1], seg[:, 2] - seg[:, 0])) % 180
    lines = hough.local_maxima(hough.accumulator(e), 90)
    return {
        "edge_pixels": int((e > 0).sum()),
        "votes_full": full,
        "segments": int(len(seg)),
        "lines_over_90": len(lines),
        "longest": [round(float(v), 1) for v in sorted(L)[::-1][:4]],
        "angles_of_longest": [round(float(a), 1) for a in ang[np.argsort(-L)][:4]],
    }


def candles() -> dict:
    """L4: four tea lights per photo, head-on and apart. How often does HoughCircles count 4?"""
    counts, radii = {}, []
    files = sorted(CANDLES.glob("*.JPG"))
    for f in files:
        img = _load(f)
        g = cv2.medianBlur(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 5)
        c = cv2.HoughCircles(g, cv2.HOUGH_GRADIENT_ALT, 1.5, 40, param1=300, param2=0.8,
                             minRadius=40, maxRadius=200)
        n = 0 if c is None else len(c[0])
        counts[str(n)] = counts.get(str(n), 0) + 1
        if c is not None:
            radii += c[0][:, 2].tolist()
    return {"photos": len(files), "circles_per_photo": counts,
            "radius_median": round(float(np.median(radii)), 1),
            "radius_p05_p95": [round(float(np.percentile(radii, 5)), 1), round(float(np.percentile(radii, 95)), 1)]}


def all_wild() -> dict:
    return {
        "cloister_fit": cloister_fit(),
        "cloister_vanishing": cloister_vanishing(),
        "document": document_segments(),
        "candles": candles(),
    }
