"""Every number unit 3.2 publishes, one function per lesson.

Lessons 1-3 measure on a comma10k highway frame and score every line against the
dataset's hand-painted lane mask; lesson 4 measures on OpenCV in Practice M2's three pool
photos and scores every circle against its 41 hand-labelled balls. Each function returns a
JSON-able dict; `cli/boundary_experiments.py` writes them to `output/boundary_numbers.json`.
Worked examples are computed from the values the posts print.
"""

from __future__ import annotations

import cv2
import numpy as np

from fundamentals_lab.boundaries import fitting, highway, hough, pool
from fundamentals_lab.config import (
    DAY_MIN_BRIGHTNESS,
    FLAT_THETA,
    HIGHWAY_FRAME,
    HOUGH_THRESHOLD,
    POOL_GRADIENT,
    POOL_GRADIENT_ALT,
    POOL_PHOTOS,
    PPH,
    RESTRICT_DEGREES,
    SCORE_TOL_PX,
)

#: OpenCV in Practice M2's colour-and-contour pipeline on the same photos and labels,
#: from fundamentals-lab `output/practice_m2_numbers.json` (stages.shadows.total).
M2_COLOUR = {"found": 29, "balls": 41, "false": 2}


def _r(x, n=3):
    return None if x is None else round(float(x), n)


def _spread(values) -> dict:
    v = np.asarray(values, float)
    if not v.size:
        return {"n": 0}
    return {
        "n": int(v.size),
        "median": _r(np.median(v)),
        "min": _r(v.min()),
        "max": _r(v.max()),
        "p05": _r(np.percentile(v, 5)),
        "p95": _r(np.percentile(v, 95)),
    }


def _fold(a: float) -> float:
    return float(a % 180)


def _gap(a: float, b: float) -> float:
    return abs(((a - b + 90) % 180) - 90)


def three_fits(P: np.ndarray) -> dict:
    """y on x, x on y, and total least squares, as line angles in [0, 180)."""
    yx = _fold(fitting.least_squares(P)["angle_deg"])
    m_xy = fitting.least_squares(P[:, ::-1])["m"]  # x = m y + b
    xy = _fold(np.degrees(np.arctan2(1.0, m_xy)))
    tls = _fold(fitting.total_least_squares(P)["angle_deg"])
    return {"y_on_x": yx, "x_on_y": xy, "tls": tls}


class Scene:
    """The highway frame, its mask, edge map and accumulator, computed once."""

    def __init__(self, name: str = HIGHWAY_FRAME):
        self.name = name
        self.bgr, self.lane = highway.load(name)
        self.edges = highway.edge_map(self.bgr)
        self.acc = hough.accumulator(self.edges)
        self.lines = hough.local_maxima(self.acc, HOUGH_THRESHOLD)

    def line_list(self, lines=None, acc=None):
        acc = self.acc if acc is None else acc
        lines = self.lines if lines is None else lines
        return [(hough.rho_of(r, acc), np.radians(t)) for _, r, t in lines]


_SAMPLE_CACHE: list | None = None


def _sample():
    """(name, is_day) for every sample frame except the scenario frame."""
    global _SAMPLE_CACHE
    if _SAMPLE_CACHE is None:
        out = []
        for p in highway.frames():
            if p.name == HIGHWAY_FRAME:
                continue
            img = cv2.imread(str(p))
            out.append(
                (p.name, float(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).mean()) >= DAY_MIN_BRIGHTNESS)
            )
        _SAMPLE_CACHE = out
    return _SAMPLE_CACHE


def worked_pixels(s: Scene) -> list[tuple[int, int]]:
    """Three edge pixels on the strongest cell's line: the top, middle and bottom one."""
    _, ri, ti = s.lines[0]
    rho, t = hough.rho_of(ri, s.acc), np.radians(ti)
    ys, xs = np.nonzero(s.edges)
    on = (
        np.rint(
            xs.astype(np.float32) * np.float32(np.cos(t))
            + ys.astype(np.float32) * np.float32(np.sin(t))
        )
        == rho
    )
    sx, sy = xs[on], ys[on]
    order = np.argsort(sy)
    return [(int(sx[i]), int(sy[i])) for i in (order[0], order[len(order) // 2], order[-1])]


# --- The scene ---------------------------------------------------------------------


def scene(s: Scene) -> dict:
    sample = _sample()
    return {
        "highway_frame": f"comma10k/imgs/{HIGHWAY_FRAME}",
        "size": list(s.bgr.shape[1::-1]),
        "road_trapezoid_px": highway.road_polygon(s.bgr.shape).tolist(),
        "edge_pixels": int((s.edges > 0).sum()),
        "edge_pixels_on_paint": _r(highway.near_lane(s.lane)[s.edges > 0].mean(), 3),
        "lane_pixels_in_road": int(highway.lane_in_road(s.lane).sum()),
        "sample_frames": len(sample),
        "sample_day": sum(d for _, d in sample),
        "score_tolerance_px": SCORE_TOL_PX,
        "pool_photos": list(POOL_PHOTOS),
        "pool_balls": sum(len(v) for v in pool.labels().values()),
        "opencv": cv2.__version__,
    }


# --- Lesson 1: fitting a line to one lane marking --------------------------------------


def fitting_lesson(s: Scene) -> dict:
    marks = highway.markings(s.lane)
    per = []
    for P in marks[:4]:
        f = three_fits(P)
        vx, vy, *_ = cv2.fitLine(P.astype(np.float32), cv2.DIST_L2, 0, 0.01, 0.01).ravel()
        spread = max(
            _gap(f["y_on_x"], f["x_on_y"]), _gap(f["y_on_x"], f["tls"]), _gap(f["x_on_y"], f["tls"])
        )
        per.append(
            {
                "pixels": int(len(P)),
                "x_range": [int(P[:, 0].min()), int(P[:, 0].max())],
                "y_range": [int(P[:, 1].min()), int(P[:, 1].max())],
                **{k: _r(v, 2) for k, v in f.items()},
                "fitline": _r(_fold(np.degrees(np.arctan2(vy, vx))), 2),
                "spread_deg": _r(spread, 2),
            }
        )
    # the marking the lesson works on: the widest disagreement among the four largest
    k = int(np.argmax([p["spread_deg"] for p in per]))
    P = marks[k]

    # Worked example: on four rows spread over the marking, its leftmost and rightmost pixel.
    ys = np.unique(P[:, 1])
    rows = [int(ys[int(i)]) for i in np.linspace(0, len(ys) - 1, 6)[1:5]]
    pts = []
    for y in rows:
        xs = P[P[:, 1] == y][:, 0]
        pts += [(int(xs.min()), y), (int(xs.max()), y)]
    W = np.array(pts, float)
    x, y = W[:, 0], W[:, 1]
    sxx = ((x - x.mean()) ** 2).sum()
    syy = ((y - y.mean()) ** 2).sum()
    sxy = ((x - x.mean()) * (y - y.mean())).sum()
    worked = {
        "points": [list(p) for p in pts],
        "mean": [_r(x.mean(), 3), _r(y.mean(), 3)],
        "Sxx": _r(sxx, 3),
        "Syy": _r(syy, 3),
        "Sxy": _r(sxy, 3),
        "y_on_x_slope": _r(sxy / sxx, 4),
        "x_on_y_slope": _r(sxy / syy, 4),
        **{k2: _r(v, 2) for k2, v in three_fits(W).items()},
    }

    # One outlier: a pixel 40 px to the side of the marking's centre, as a car's edge would be.
    c = P.mean(axis=0)
    outlier = {k2: _r(v, 2) for k2, v in three_fits(np.vstack([P, [c[0] + 40, c[1]]])).items()}

    # Held out: every marking of 300+ px on the other 299 frames.
    spreads, yx_tls, xy_tls, angles = [], [], [], []
    for name, _day in _sample():
        _, lane = highway.load(name)
        for Q in highway.markings(lane):
            f = three_fits(Q)
            yx_tls.append(_gap(f["y_on_x"], f["tls"]))
            xy_tls.append(_gap(f["x_on_y"], f["tls"]))
            spreads.append(_gap(f["y_on_x"], f["x_on_y"]))
            angles.append(f["tls"])
    angles = np.array(angles)
    steep = np.abs(90 - angles) < 20
    return {
        "markings": per,
        "worked_marking": k,
        "worked_example": worked,
        "one_outlier": outlier,
        "heldout": {
            "markings": len(spreads),
            "y_on_x_vs_x_on_y_deg": _spread(spreads),
            "y_on_x_vs_tls_deg": _spread(yx_tls),
            "x_on_y_vs_tls_deg": _spread(xy_tls),
            "share_over_1deg": _r(np.mean(np.array(spreads) > 1), 3),
            "within_20deg_of_vertical": int(steep.sum()),
            "y_on_x_vs_tls_near_vertical": _spread(np.array(yx_tls)[steep]),
        },
    }


# --- Lesson 2: the Hough transform on the highway ---------------------------------------


def _summary_lines(rows):
    prec = [r["on_paint"] / r["lines"] for r in rows if r["lines"]]
    return {
        "frames": len(rows),
        "lines": _spread([r["lines"] for r in rows]),
        "precision": _spread(prec),
        "paint_covered": _spread([r["paint_covered"] for r in rows]),
        "frames_with_no_line": sum(r["lines"] == 0 for r in rows),
    }


def hough_lesson(s: Scene) -> dict:
    acc, lines = s.acc, s.lines
    n_edge = int((s.edges > 0).sum())
    cvl = cv2.HoughLinesWithAccumulator(s.edges, 1, np.pi / 180, HOUGH_THRESHOLD).reshape(-1, 3)
    half = (acc.shape[0] - 1) // 2
    cv_keys = [(int(round(r)) + half, int(round(t / (np.pi / 180)))) for r, t, _ in cvl]
    ours = {(r, t) for _, r, t in lines}
    f64 = hough.local_maxima(
        hough.accumulator(s.edges, dtype=np.float64, exact_angle=True), HOUGH_THRESHOLD
    )

    v, ri, ti = lines[0]
    rho = hough.rho_of(ri, acc)
    thetas = [ti - 10, ti - 1, ti, ti + 1, ti + 10]
    table = [
        {
            "x": px,
            "y": py,
            "rho": {
                str(th): _r(px * np.cos(np.radians(th)) + py * np.sin(np.radians(th)), 1)
                for th in thetas
            },
        }
        for px, py in worked_pixels(s)
    ]

    bins = {}
    for step in (0.5, 1.0, 2.0):
        a = hough.accumulator(s.edges, theta_deg=step)
        lm = hough.local_maxima(a, HOUGH_THRESHOLD)
        sc = highway.score_lines(
            s.edges, s.lane, [(hough.rho_of(r, a), np.radians(tt * step)) for _, r, tt in lm]
        )
        bins[str(step)] = {"cells": int(a.size), "top_votes": int(a.max()), **sc}

    # Which painted marking each line belongs to: the one most of its edge pixels sit on.
    paint = highway.lane_in_road(s.lane).astype(np.uint8)
    k = 2 * SCORE_TOL_PX + 1
    _, lab = cv2.connectedComponents(cv2.dilate(paint, np.ones((k, k), np.uint8)))
    ys, xs = np.nonzero(s.edges)
    owner = {}
    for rr, tt in s.line_list():
        d = np.abs(xs * np.cos(tt) + ys * np.sin(tt) - rr) <= 1.5
        ids = lab[ys[d], xs[d]]
        ids = ids[ids > 0]
        key = int(np.bincount(ids).argmax()) if ids.size else 0
        owner[key] = owner.get(key, 0) + 1
    per_mark = {
        "painted_markings_hit": sum(1 for k2 in owner if k2),
        "lines_per_marking": sorted((v for k2, v in owner.items() if k2), reverse=True),
        "lines_off_paint": owner.get(0, 0),
    }

    held = {"day": [], "night": []}
    for name, day in _sample():
        sc = Scene(name)
        row = (
            highway.score_lines(sc.edges, sc.lane, sc.line_list())
            if sc.lines
            else {"lines": 0, "on_paint": 0, "paint_covered": 0.0}
        )
        held["day" if day else "night"].append(row)

    return {
        "edge_pixels": n_edge,
        "n_theta": int(acc.shape[1]),
        "n_rho": int(acc.shape[0]),
        "cells": int(acc.size),
        "votes_cast": n_edge * int(acc.shape[1]),
        "threshold": HOUGH_THRESHOLD,
        "top_lines": [
            {"votes": vv, "rho": hough.rho_of(r, acc), "theta_deg": tt} for vv, r, tt in lines[:8]
        ],
        **highway.score_lines(s.edges, s.lane, s.line_list()),
        "lines_by_marking": per_mark,
        "opencv_lines": int(len(cvl)),
        "same_set_as_opencv": set(cv_keys) == ours,
        "same_votes_as_opencv": bool(
            all(acc[k] == int(vv) for k, vv in zip(cv_keys, cvl[:, 2], strict=True))
        ),
        "float64_lines_differing": len(set(cv_keys) ^ {(r, t) for _, r, t in f64}) // 2,
        "worked_example": {"cell": [rho, ti], "votes": int(v), "pixels": table, "thetas": thetas},
        "bin_size": bins,
        "heldout": {k: _summary_lines(rows) for k, rows in held.items()},
    }


# --- Lesson 3: restricted voting and the probabilistic Hough ---------------------------


def _summary_segments(rows):
    prec = [r["on_paint"] / r["segments"] for r in rows if r["segments"]]
    return {
        "frames": len(rows),
        "segments": _spread([r["segments"] for r in rows]),
        "precision": _spread(prec),
        "paint_covered": _spread([r["paint_covered"] for r in rows]),
    }


def probabilistic_lesson(s: Scene) -> dict:
    blur = highway.blurred(s.bgr)
    gx = cv2.Sobel(blur, cv2.CV_32F, 1, 0)
    gy = cv2.Sobel(blur, cv2.CV_32F, 0, 1)
    orient = (np.degrees(np.arctan2(gy, gx)) % 180).astype(np.float32)
    full_votes = int((s.edges > 0).sum()) * s.acc.shape[1]
    full_set = {(r, t) for _, r, t in s.lines}
    restricted = {}
    for k in RESTRICT_DEGREES:
        a, cast = hough.restricted_accumulator(s.edges, orient, k)
        lm = hough.local_maxima(a, HOUGH_THRESHOLD)
        sc = highway.score_lines(
            s.edges, s.lane, [(hough.rho_of(r, a), np.radians(t)) for _, r, t in lm]
        )
        restricted[str(k)] = {
            "votes_cast": cast,
            "fraction_of_full": _r(cast / full_votes, 4),
            "top_votes": int(a.max()),
            "also_in_full_set": sum((r, t) in full_set for _, r, t in lm),
            **sc,
        }

    # The lane prior: drop every line within 20 degrees of horizontal, on its own and on
    # top of the gradient restriction.
    def lane_lines(acc, lines):
        return [
            (hough.rho_of(r, acc), np.radians(t))
            for _, r, t in lines
            if not FLAT_THETA[0] <= t <= FLAT_THETA[1]
        ]

    prior = {"full_vote": highway.score_lines(s.edges, s.lane, lane_lines(s.acc, s.lines))}
    a5, _ = hough.restricted_accumulator(s.edges, orient, 5)
    prior["restricted_5"] = highway.score_lines(
        s.edges, s.lane, lane_lines(a5, hough.local_maxima(a5, HOUGH_THRESHOLD))
    )

    def pph(thr=PPH["threshold"], minl=PPH["min_length"], gap=PPH["max_gap"], edges=None):
        e = s.edges if edges is None else edges
        seg = cv2.HoughLinesP(e, 1, np.pi / 180, thr, minLineLength=minl, maxLineGap=gap)
        return np.empty((0, 4), int) if seg is None else seg.reshape(-1, 4)

    seg, seg2 = pph(), pph()
    lengths = np.hypot(seg[:, 2] - seg[:, 0], seg[:, 3] - seg[:, 1])
    sweep = {
        f"{minl}/{gap}": highway.score_segments(s.lane, pph(minl=minl, gap=gap))
        for minl in (20, 40, 80)
        for gap in (5, 20, 60)
    }

    held = {"day": [], "night": []}
    held_lines = {"full": [], "prior": [], "restricted_5_prior": []}
    for name, day in _sample():
        sc = Scene(name)
        held["day" if day else "night"].append(highway.score_segments(sc.lane, pph(edges=sc.edges)))
        o = (
            np.degrees(
                np.arctan2(
                    cv2.Sobel(highway.blurred(sc.bgr), cv2.CV_32F, 0, 1),
                    cv2.Sobel(highway.blurred(sc.bgr), cv2.CV_32F, 1, 0),
                )
            )
            % 180
        ).astype(np.float32)
        ar, _ = hough.restricted_accumulator(sc.edges, o, 5)
        for key, L in (
            ("full", sc.line_list()),
            ("prior", lane_lines(sc.acc, sc.lines)),
            ("restricted_5_prior", lane_lines(ar, hough.local_maxima(ar, HOUGH_THRESHOLD))),
        ):
            held_lines[key].append(
                highway.score_lines(sc.edges, sc.lane, L)
                if L
                else {"lines": 0, "on_paint": 0, "paint_covered": 0.0}
            )

    return {
        "full_votes_cast": full_votes,
        "restricted": restricted,
        "lane_prior": prior,
        "heldout_lines": {k: _summary_lines(v) for k, v in held_lines.items()},
        "worked_example_orientation_deg": {
            f"{x},{y}": _r(orient[y, x], 2) for x, y in worked_pixels(s)
        },
        "pph": {
            **PPH,
            **highway.score_segments(s.lane, seg),
            "deterministic": bool(np.array_equal(seg, seg2)),
            "longest_px": _r(lengths.max(), 1) if len(seg) else None,
            "median_length_px": _r(np.median(lengths), 1) if len(seg) else None,
        },
        "pph_sweep_minlen_gap": sweep,
        "heldout": {k: _summary_segments(rows) for k, rows in held.items()},
    }


# --- Lesson 4: Hough circles on the pool table ------------------------------------------


def circles_lesson(s: Scene) -> dict:
    grad = {
        str(p2): pool.run_all(cv2.HOUGH_GRADIENT, POOL_GRADIENT["dp"], POOL_GRADIENT["param1"], p2)
        for p2 in POOL_GRADIENT["param2"]
    }
    alt = {
        str(p2): pool.run_all(
            cv2.HOUGH_GRADIENT_ALT, POOL_GRADIENT_ALT["dp"], POOL_GRADIENT_ALT["param1"], p2
        )
        for p2 in POOL_GRADIENT_ALT["param2"]
    }

    def chosen(method, cfg):
        """Choose param2 on one photo (most found minus false), score it on the other two."""

        def net(p2):
            r = pool.run_all(method, cfg["dp"], cfg["param1"], p2, photos=("pool",))
            return r["found"] - r["false"]

        best = max(cfg["param2"], key=net)
        rest = pool.run_all(
            method, cfg["dp"], cfg["param1"], best, photos=("pool-close", "pool-wide")
        )
        for v in rest["per_photo"].values():
            v.pop("missed")
            v.pop("false_at")
        return {"param2": best, "scored_on_other_two": rest}

    # The textbook accumulator at one radius, on the biggest ball in pool-close.
    L = pool.labels()
    bx, by, br = max(L["pool-close"], key=lambda b: b[2])
    img = pool.load("pool-close")
    g = pool.grey(img)
    acc = pool.fixed_radius_accumulator(g, int(br))
    y0, x0 = max(0, by - 3 * br), max(0, bx - 3 * br)
    win = acc[y0 : by + 3 * br, x0 : bx + 3 * br]
    py, px = np.unravel_index(int(win.argmax()), win.shape)
    px, py = px + x0, py + y0
    cv = pool.detect(
        img, cv2.HOUGH_GRADIENT_ALT, POOL_GRADIENT_ALT["dp"], POOL_GRADIENT_ALT["param1"], 0.8
    )
    near = min(cv, key=lambda c: np.hypot(c[0] - bx, c[1] - by)) if cv else None
    # The same votes in coarser cells: what HoughCircles' dp sets.
    cells = {}
    for cell in (1, 2, 4):
        h, w = acc.shape
        a = (
            acc[: h // cell * cell, : w // cell * cell]
            .reshape(h // cell, cell, w // cell, cell)
            .sum(axis=(1, 3))
        )
        y0c, x0c = (by - 120) // cell, (bx - 120) // cell
        sub = a[y0c : y0c + 240 // cell, x0c : x0c + 240 // cell]
        yy, xx = np.unravel_index(int(sub.argmax()), sub.shape)
        cx, cy = (x0c + xx) * cell + cell / 2, (y0c + yy) * cell + cell / 2
        cells[str(cell)] = {
            "peak_votes": int(sub.max()),
            "centre": [cx, cy],
            "to_label_px": _r(np.hypot(cx - bx, cy - by), 1),
        }
    textbook = {
        "label": [bx, by, br],
        "peak": [int(px), int(py)],
        "peak_votes": int(acc[py, px]),
        "cell_sizes": cells,
        "peak_to_label_px": _r(np.hypot(px - bx, py - by), 2),
        "opencv_alt": [_r(v, 1) for v in near] if near else None,
        "opencv_to_label_px": _r(np.hypot(near[0] - bx, near[1] - by), 2) if near else None,
        "edge_pixels": int((cv2.Canny(g, 50, 150) > 0).sum()),
        "radii_in_the_search": 60 - 8 + 1,
    }

    return {
        "gradient": grad,
        "gradient_alt": alt,
        "chosen_on_pool": {
            "gradient": chosen(cv2.HOUGH_GRADIENT, POOL_GRADIENT),
            "gradient_alt": chosen(cv2.HOUGH_GRADIENT_ALT, POOL_GRADIENT_ALT),
        },
        "m2_colour_pipeline": M2_COLOUR,
        "textbook_accumulator": textbook,
    }
