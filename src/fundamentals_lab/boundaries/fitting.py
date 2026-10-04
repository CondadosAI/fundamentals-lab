"""Lesson 1: a line through points, by vertical residuals and by perpendicular ones."""

from __future__ import annotations

import numpy as np


def least_squares(P: np.ndarray) -> dict:
    """y = m x + b, minimising vertical residuals."""
    A = np.c_[P[:, 0], np.ones(len(P))]
    (m, b), *_ = np.linalg.lstsq(A, P[:, 1], rcond=None)
    resid = P[:, 1] - (m * P[:, 0] + b)
    return {
        "m": float(m),
        "b": float(b),
        "angle_deg": float(np.degrees(np.arctan(m))),
        "cond": float(np.linalg.cond(A)),
        "vertical_rss": float((resid**2).sum()),
    }


def total_least_squares(P: np.ndarray) -> dict:
    """The line through the centroid along the first singular vector."""
    mu = P.mean(axis=0)
    _, s, vt = np.linalg.svd(P - mu, full_matrices=False)
    d = vt[0] if vt[0][1] >= 0 else -vt[0]
    n = np.array([-d[1], d[0]])
    resid = (P - mu) @ n
    return {
        "centroid": mu.tolist(),
        "direction": d.tolist(),
        "angle_deg": float(np.degrees(np.arctan2(d[1], d[0]))),
        "perpendicular_rss": float((resid**2).sum()),
        "perpendicular_rms": float(np.sqrt((resid**2).mean())),
        "singular_values": s.tolist(),
    }


def perpendicular_rss_of(P: np.ndarray, angle_deg: float, through: np.ndarray) -> float:
    t = np.radians(angle_deg)
    n = np.array([-np.sin(t), np.cos(t)])
    return float((((P - through) @ n) ** 2).sum())
