"""Lines on a highway: comma10k frames, the road region, and scoring against the lane mask."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from fundamentals_lab.config import (
    COMMA10K_DIR,
    HIGHWAY_CANNY,
    HIGHWAY_FRAME,
    HIGHWAY_SIGMA,
    LANE_BGR,
    ROAD_TRAPEZOID,
    SCORE_TOL_PX,
)


def frames() -> list[Path]:
    return sorted((COMMA10K_DIR / "imgs").glob("*.png"))


def load(name: str = HIGHWAY_FRAME) -> tuple[np.ndarray, np.ndarray]:
    """The frame (BGR) and its lane-marking mask (bool), both 1164x874."""
    img = cv2.imread(str(COMMA10K_DIR / "imgs" / name))
    mask = cv2.imread(str(COMMA10K_DIR / "masks" / name))
    if img is None or mask is None:
        raise FileNotFoundError(f"{name}: run boundary-download first")
    lane = np.all(np.abs(mask.astype(int) - LANE_BGR) < 30, axis=2)
    return img, lane


def road_polygon(shape) -> np.ndarray:
    h, w = shape[:2]
    return np.array([[fx * w, fy * h] for fx, fy in ROAD_TRAPEZOID], np.int32)


def road_mask(shape) -> np.ndarray:
    m = np.zeros(shape[:2], np.uint8)
    cv2.fillPoly(m, [road_polygon(shape)], 255)
    return m


def blurred(bgr: np.ndarray) -> np.ndarray:
    return cv2.GaussianBlur(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY), (0, 0), HIGHWAY_SIGMA)


def edge_map(bgr: np.ndarray, canny=HIGHWAY_CANNY) -> np.ndarray:
    """Canny inside the road trapezoid only."""
    return cv2.bitwise_and(cv2.Canny(blurred(bgr), *canny), road_mask(bgr.shape))


def near_lane(lane: np.ndarray, tol: int = SCORE_TOL_PX) -> np.ndarray:
    k = 2 * tol + 1
    return cv2.dilate(lane.astype(np.uint8), np.ones((k, k), np.uint8)) > 0


def lane_in_road(lane: np.ndarray) -> np.ndarray:
    return lane & (road_mask(lane.shape) > 0)


def score_lines(edges: np.ndarray, lane: np.ndarray, lines) -> dict:
    """lines: [(rho, theta_rad), ...]. Precision per line, and how much of the paint they cover."""
    near = near_lane(lane)
    paint = lane_in_road(lane)
    ys, xs = np.nonzero(edges)
    cover = np.zeros(lane.shape, np.uint8)
    on = 0
    for rho, t in lines:
        d = np.abs(xs * np.cos(t) + ys * np.sin(t) - rho) <= 1.5
        if d.any() and near[ys[d], xs[d]].mean() >= 0.5:
            on += 1
        c, s = np.cos(t), np.sin(t)
        cv2.line(cover, (int(c * rho - 3000 * s), int(s * rho + 3000 * c)),
                 (int(c * rho + 3000 * s), int(s * rho - 3000 * c)), 1, 2 * 3 + 1)
    recall = float((cover.astype(bool) & paint).sum() / max(1, paint.sum()))
    return {"lines": len(lines), "on_paint": on, "paint_covered": round(recall, 3)}


def score_segments(lane: np.ndarray, segs) -> dict:
    near = near_lane(lane)
    paint = lane_in_road(lane)
    cover = np.zeros(lane.shape, np.uint8)
    on = 0
    for x1, y1, x2, y2 in segs:
        n = int(np.hypot(x2 - x1, y2 - y1)) + 1
        px = np.rint(np.linspace(x1, x2, n)).astype(int)
        py = np.rint(np.linspace(y1, y2, n)).astype(int)
        on += near[py, px].mean() >= 0.5
        cv2.line(cover, (int(x1), int(y1)), (int(x2), int(y2)), 1, 7)
    recall = float((cover.astype(bool) & paint).sum() / max(1, paint.sum()))
    return {"segments": len(segs), "on_paint": int(on), "paint_covered": round(recall, 3)}


def markings(lane: np.ndarray, min_px: int = 300) -> list[np.ndarray]:
    """Each painted marking in the road rows as (N, 2) pixels (x, y), largest first."""
    paint = lane_in_road(lane).astype(np.uint8)
    n, lab, st, _ = cv2.connectedComponentsWithStats(paint)
    out = []
    for k in np.argsort(-st[1:, cv2.CC_STAT_AREA]) + 1:
        if st[k, cv2.CC_STAT_AREA] < min_px:
            break
        ys, xs = np.nonzero(lab == k)
        out.append(np.c_[xs, ys].astype(float))
    return out
