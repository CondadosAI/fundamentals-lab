"""Every number unit 3.2 publishes, one function per lesson.

Each returns a JSON-able dict; `cli/boundary_experiments.py` writes them all to
`output/boundary_numbers.json`. Worked examples are computed from the rounded values
the posts print, so a reader redoing the arithmetic gets the page's answer.
"""

from __future__ import annotations

import cv2
import numpy as np

from fundamentals_lab.boundaries import circles, fitting, frame, ght, hough
from fundamentals_lab.config import (
    BOUNDARY_FRAME,
    BOUNDARY_WIDTH,
    GHT,
    GHT_RING_INNER,
    HELDOUT_STEP,
    HOLE_PARAM2_SWEEP,
    HOUGH_THRESHOLD,
    LEFT_EDGE_BOX,
    PPH,
    RESTRICT_DEGREES,
    TOP_EDGE_BOX,
)


def _r(x, n=3):
    return None if x is None else round(float(x), n)


class Scene:
    """The frame, its edge map, the accumulator and the two transducers, computed once."""

    def __init__(self, name: str = BOUNDARY_FRAME):
        self.bgr = frame.load(name)
        self.blur = frame.blurred(self.bgr)
        self.edges = frame.edge_map(self.bgr)
        self.acc = hough.accumulator(self.edges)
        self.lines = hough.local_maxima(self.acc, HOUGH_THRESHOLD)
        self.transducers = circles.transducers(self.blur)


def _heldout_names() -> list[str]:
    return [p.name for p in frame.normal_frames()[::HELDOUT_STEP]]


def _spread(values) -> dict:
    v = np.asarray(values, float)
    return {
        "n": int(v.size),
        "median": _r(np.median(v)),
        "min": _r(v.min()),
        "max": _r(v.max()),
        "p05": _r(np.percentile(v, 5)),
        "p95": _r(np.percentile(v, 95)),
    }


# --- The scene ---------------------------------------------------------------------


def scene(s: Scene) -> dict:
    h, w = s.edges.shape
    return {
        "frame": f"pcb1/Data/Images/Normal/{BOUNDARY_FRAME}",
        "source_size": [1404, 1070],
        "working_size": [w, h],
        "width": BOUNDARY_WIDTH,
        "edge_pixels": int((s.edges > 0).sum()),
        "normal_frames": len(frame.normal_frames()),
        "opencv": cv2.__version__,
    }


# --- Lesson 1: fitting lines ----------------------------------------------------------


def _fold(angle_deg: float) -> float:
    """A line's angle in [0, 180)."""
    return float(angle_deg % 180)


def left_edge_points(edges: np.ndarray, band: float = 6.0):
    """Edge pixels of the board's left side, found by the accumulator, in any frame."""
    acc = hough.accumulator(edges)
    w = edges.shape[1]
    best = None
    for v, ri, ti in hough.local_maxima(acc, 60):
        rho = hough.rho_of(ri, acc)
        if (ti <= 3 or ti >= 177) and 0 < abs(rho) < w / 3:
            best = (rho, np.radians(ti))
            break
    if best is None:
        return None
    rho, t = best
    ys, xs = np.nonzero(edges)
    d = np.abs(xs * np.cos(t) + ys * np.sin(t) - rho)
    P = np.c_[xs[d <= band], ys[d <= band]].astype(float)
    if len(P) < 50:
        return None
    lo, hi = np.percentile(P[:, 1], [20, 80])
    P = P[(P[:, 1] >= lo) & (P[:, 1] <= hi)]
    return P if len(P) >= 50 else None


def fitting_lesson(s: Scene) -> dict:
    out = {}
    for name, box in (("left_edge", LEFT_EDGE_BOX), ("top_edge", TOP_EDGE_BOX)):
        P = frame.box_points(s.edges, box)
        ls = fitting.least_squares(P)
        tls = fitting.total_least_squares(P)
        vx, vy, *_ = cv2.fitLine(P.astype(np.float32), cv2.DIST_L2, 0, 0.01, 0.01).ravel()
        fl = float(np.degrees(np.arctan2(vy, vx)))
        fl = fl + 180 if fl < 0 else fl
        out[name] = {
            "box": list(box),
            "points": int(len(P)),
            "x_range": [int(P[:, 0].min()), int(P[:, 0].max())],
            "y_range": [int(P[:, 1].min()), int(P[:, 1].max())],
            "ls": {k: _r(v, 4) for k, v in ls.items()},
            "tls": {
                "angle_deg": _r(tls["angle_deg"], 4),
                "perpendicular_rms": _r(tls["perpendicular_rms"], 3),
                "centroid": [_r(v, 2) for v in tls["centroid"]],
            },
            "fitline_angle_deg": _r(fl, 4),
            "ls_minus_tls_deg": _r(((ls["angle_deg"] - tls["angle_deg"] + 90) % 180) - 90, 3),
            "ls_perpendicular_rss": _r(
                fitting.perpendicular_rss_of(P, ls["angle_deg"], P.mean(0)), 1
            ),
            "tls_perpendicular_rss": _r(tls["perpendicular_rss"], 1),
        }

    # Worked example: every edge pixel of the left side on four rows, 40 apart. The side
    # shows as two edges about 5 px apart, so each row gives two pixels.
    P = frame.box_points(s.edges, LEFT_EDGE_BOX)
    pts = []
    for y in (280, 320, 360, 400):
        pts += [(int(x), y) for x in sorted(P[P[:, 1] == y][:, 0])]
    W = np.array(pts, float)
    x, y = W[:, 0], W[:, 1]
    n = len(W)
    sx, sy, sxx, sxy = x.sum(), y.sum(), (x * x).sum(), (x * y).sum()
    m = (n * sxy - sx * sy) / (n * sxx - sx * sx)
    b = (sy - m * sx) / n
    tls = fitting.total_least_squares(W)
    out["worked_example"] = {
        "points": [list(map(int, p)) for p in pts],
        "n": n,
        "sum_x": _r(sx, 1),
        "sum_y": _r(sy, 1),
        "sum_xx": _r(sxx, 1),
        "sum_xy": _r(sxy, 1),
        "ls_m": _r(m, 4),
        "ls_b": _r(b, 2),
        "ls_angle_deg": _r(np.degrees(np.arctan(m)), 3),
        "denominator": _r(n * sxx - sx * sx, 1),
        "tls_angle_deg": _r(tls["angle_deg"], 3),
    }

    # One outlier: a pixel 40 px to the right of the left edge, halfway down.
    P = frame.box_points(s.edges, LEFT_EDGE_BOX)
    Po = np.vstack([P, [P[:, 0].mean() + 40, P[:, 1].mean()]])
    out["one_outlier"] = {
        "ls_angle_deg": _r(fitting.least_squares(Po)["angle_deg"], 3),
        "tls_angle_deg": _r(fitting.total_least_squares(Po)["angle_deg"], 3),
        "offset_px": 40,
    }

    # Held out: the board moves between frames, so the left edge is found again in
    # each one (the strongest near-vertical line in the left third) and its pixels are
    # taken within 6 px of it, middle 60% of their rows, as the box does on frame 0000.
    ls_err, tls_err, missing = [], [], 0
    for name in _heldout_names():
        e = frame.edge_map(frame.load(name))
        P = left_edge_points(e)
        if P is None:
            missing += 1
            continue
        ls_err.append(_fold(fitting.least_squares(P)["angle_deg"]))
        tls_err.append(_fold(fitting.total_least_squares(P)["angle_deg"]))
    out["heldout_left_edge"] = {
        "frames": len(ls_err),
        "not_found": missing,
        "ls_angle": _spread(ls_err),
        "tls_angle": _spread(tls_err),
        "ls_off_vertical_deg": _spread(np.abs(90 - np.asarray(ls_err))),
        "tls_off_vertical_deg": _spread(np.abs(90 - np.asarray(tls_err))),
    }
    return out


# --- Lesson 2: the Hough transform ----------------------------------------------------


def hough_lesson(s: Scene) -> dict:
    acc, lines = s.acc, s.lines
    n_edge = int((s.edges > 0).sum())
    cvl = cv2.HoughLinesWithAccumulator(s.edges, 1, np.pi / 180, HOUGH_THRESHOLD).reshape(-1, 3)
    half = (acc.shape[0] - 1) // 2
    cv_set = {(int(round(r)) + half, int(round(t / (np.pi / 180)))) for r, t, _ in cvl}
    ours = {(r, t) for _, r, t in lines}
    f64 = hough.local_maxima(
        hough.accumulator(s.edges, dtype=np.float64, exact_angle=True), HOUGH_THRESHOLD
    )
    f64_set = {(r, t) for _, r, t in f64}

    # Worked example: three pixels of the top edge, rho at a handful of theta.
    P = frame.box_points(s.edges, TOP_EDGE_BOX)
    top_row = int(np.median(P[:, 1]))
    xs = sorted(P[P[:, 1] == top_row][:, 0].astype(int).tolist())
    picks = [xs[0], xs[len(xs) // 2], xs[-1]]
    thetas = [0, 45, 89, 90, 91, 135]
    table = [
        {
            "x": x,
            "y": top_row,
            "rho": {
                str(t): _r(x * np.cos(np.radians(t)) + top_row * np.sin(np.radians(t)), 1)
                for t in thetas
            },
        }
        for x in picks
    ]

    # Bin size: the strongest peak at half and double the theta step.
    bins = {}
    for step in (0.5, 1.0, 2.0):
        a = hough.accumulator(s.edges, theta_deg=step)
        lm = hough.local_maxima(a, HOUGH_THRESHOLD)
        bins[str(step)] = {
            "cells": int(a.size),
            "top_votes": int(a.max()),
            "lines_over_threshold": len(lm),
        }

    # The board's top edge as a peak, and its neighbours: how one edge spreads.
    v, ri, ti = lines[0]
    nb = acc[ri - 3 : ri + 4, ti - 2 : ti + 3]

    holes = {str(p2): circles.holes(s.blur, p2) for p2 in HOLE_PARAM2_SWEEP}
    for d in holes.values():
        d.pop("circles")

    held_votes, held_lines, held_c = [], [], []
    for name in _heldout_names():
        sc = Scene(name)
        held_votes.append(sc.lines[0][0])
        held_lines.append(len(sc.lines))
        if sc.transducers is not None and len(sc.transducers) == 2:
            held_c.append(sc.transducers.ravel().tolist())
    held_c = np.array(held_c)
    return {
        "edge_pixels": n_edge,
        "n_theta": int(acc.shape[1]),
        "n_rho": int(acc.shape[0]),
        "cells": int(acc.size),
        "votes_cast": n_edge * int(acc.shape[1]),
        "threshold": HOUGH_THRESHOLD,
        "lines_over_threshold": len(lines),
        "top_lines": [
            {"votes": v, "rho": hough.rho_of(r, acc), "theta_deg": t} for v, r, t in lines[:8]
        ],
        "opencv_lines": int(len(cvl)),
        "same_set_as_opencv": cv_set == ours,
        "same_votes_as_opencv": bool(
            all(
                acc[r, t] == int(vv)
                for (r, t), vv in zip(
                    [(int(round(a)) + half, int(round(b / (np.pi / 180)))) for a, b, _ in cvl],
                    cvl[:, 2],
                )
            )
        ),
        "float64_lines_differing": len(cv_set ^ f64_set) // 2,
        "worked_example": {"row": top_row, "pixels": table, "thetas": thetas},
        "bin_size": bins,
        "top_peak_neighbourhood": {"centre": [hough.rho_of(ri, acc), ti], "votes": nb.tolist()},
        "transducers": [[_r(v, 1) for v in c] for c in s.transducers.tolist()],
        "holes_param2": holes,
        "heldout": {
            "frames": len(held_votes),
            "top_votes": _spread(held_votes),
            "lines_over_threshold": _spread(held_lines),
            "transducer_left_r": _spread(held_c[:, 2]) if len(held_c) else None,
            "transducer_right_r": _spread(held_c[:, 5]) if len(held_c) else None,
            "transducer_frames": int(len(held_c)),
        },
    }


# --- Lesson 3: restricted voting and the probabilistic Hough --------------------------


def probabilistic_lesson(s: Scene) -> dict:
    gx = cv2.Sobel(s.blur, cv2.CV_32F, 1, 0)
    gy = cv2.Sobel(s.blur, cv2.CV_32F, 0, 1)
    orient = (np.degrees(np.arctan2(gy, gx)) % 180).astype(np.float32)
    full_top = [(r, t) for _, r, t in s.lines[:10]]
    full_votes = int((s.edges > 0).sum()) * s.acc.shape[1]
    restricted = {}
    for k in RESTRICT_DEGREES:
        a, cast = hough.restricted_accumulator(s.edges, orient, k)
        lm = hough.local_maxima(a, HOUGH_THRESHOLD)
        top = [(r, t) for _, r, t in lm[:10]]
        restricted[str(k)] = {
            "votes_cast": cast,
            "fraction_of_full": _r(cast / full_votes, 4),
            "lines_over_threshold": len(lm),
            "top10_shared_with_full": len(set(top) & set(full_top)),
            "top_votes": int(a.max()),
        }

    def pph(thr=PPH["threshold"], minl=PPH["min_length"], gap=PPH["max_gap"]):
        seg = cv2.HoughLinesP(s.edges, 1, np.pi / 180, thr, minLineLength=minl, maxLineGap=gap)
        return np.empty((0, 4), int) if seg is None else seg.reshape(-1, 4)

    seg = pph()
    seg2 = pph()
    lengths = np.hypot(seg[:, 2] - seg[:, 0], seg[:, 3] - seg[:, 1])
    sweep = {}
    for minl in (40, 80, 160):
        for gap in (2, 5, 20):
            sg = pph(minl=minl, gap=gap)
            sweep[f"{minl}/{gap}"] = int(len(sg))
    return {
        "full_votes_cast": full_votes,
        "restricted": restricted,
        "pph": {
            **PPH,
            "segments": int(len(seg)),
            "deterministic": bool(np.array_equal(seg, seg2)),
            "longest_px": _r(lengths.max(), 1) if len(seg) else None,
            "median_length_px": _r(np.median(lengths), 1) if len(seg) else None,
            "first": seg[:5].tolist(),
        },
        "pph_sweep_minlen_gap": sweep,
    }


# --- Lesson 4: the generalized Hough transform -------------------------------------


def ght_lesson(s: Scene) -> dict:
    left = s.transducers[0]
    out = {
        "left_transducer": [_r(v, 1) for v in left],
        "right_transducer": [_r(v, 1) for v in s.transducers[1]],
    }
    for label, inner in (("full_crop", None), ("ring_only", GHT_RING_INNER)):
        tm = ght.ring_template(s.blur, left, inner)
        tab, ref = ght.r_table(tm)
        acc = ght.vote(s.blur, tab)
        pk = ght.peaks(acc, 4, GHT["min_dist"])
        cv = ght.opencv_ballard(s.blur, tm)
        out[label] = {
            "template_size": list(tm.shape[::-1]),
            "reference": list(ref),
            "r_table_bins": len(tab),
            "r_table_rows": int(sum(len(v) for v in tab.values())),
            "ours": pk,
            "opencv": cv[:4],
            "same_as_opencv": pk[: len(cv[:2])] == cv[:2],
        }
        if inner:
            tab_h, ref_h = ght.r_table(tm, parity=False)
            out[label]["half_pixel_reference"] = ght.peaks(
                ght.vote(s.blur, tab_h, parity=False), 2, GHT["min_dist"]
            )
            # three R-table rows to print in full
            some = sorted(tab)[::90][:4]
            out[label]["rows_sample"] = {str(b): tab[b][:3] for b in some}

    # Scale: resize the ring template and vote again; where does each transducer peak go?
    tm = ght.ring_template(s.blur, left, GHT_RING_INNER)
    scales = {}
    for f in (0.90, 0.95, 1.00, 1.05):
        t = cv2.resize(tm, None, fx=f, fy=f, interpolation=cv2.INTER_AREA)
        tab, _ = ght.r_table(t)
        pk = ght.peaks(ght.vote(s.blur, tab), 2, GHT["min_dist"])
        scales[f"{f:.2f}"] = pk
    out["scale"] = scales
    out["radius_ratio_right_over_left"] = _r(s.transducers[1][2] / left[2], 4)

    # Held out: the ring template from frame 0000, voted on every 10th frame.
    tab, _ = ght.r_table(tm)
    ratios = []
    for name in _heldout_names():
        b = frame.blurred(frame.load(name))
        pk = ght.peaks(ght.vote(b, tab), 2, GHT["min_dist"])
        if len(pk) == 2:
            ratios.append(pk[1][0] / pk[0][0])
    out["heldout_second_over_first"] = _spread(ratios)
    return out
