"""Lessons 2 and 3: the (rho, theta) accumulator, built to match OpenCV vote for vote.

OpenCV's HoughLines computes rho in float32, from a sine/cosine table whose angle is
accumulated by repeated float32 addition. Doing the same here reproduces its votes and
its set of lines exactly; doing it in float64 changes three of 312 lines on the board.
"""

from __future__ import annotations

import numpy as np


def trig_table(n_theta: int, theta: float, rho: float = 1.0, exact: bool = False):
    if exact:
        ang = np.arange(n_theta) * theta
        return (np.cos(ang) / rho).astype(np.float32), (np.sin(ang) / rho).astype(np.float32)
    cos, sin = np.empty(n_theta, np.float32), np.empty(n_theta, np.float32)
    ang = np.float32(0.0)
    for n in range(n_theta):
        cos[n] = np.float32(np.cos(np.float64(ang)) / rho)
        sin[n] = np.float32(np.sin(np.float64(ang)) / rho)
        ang = np.float32(ang + np.float32(theta))
    return cos, sin


def accumulator(
    edges: np.ndarray,
    theta_deg: float = 1.0,
    rho: float = 1.0,
    dtype=np.float32,
    exact_angle: bool = False,
) -> np.ndarray:
    """Votes, shape (n_rho, n_theta); row i is rho = i - (n_rho - 1) // 2."""
    h, w = edges.shape
    theta = np.radians(theta_deg)
    n_theta = int(round(np.pi / theta))
    n_rho = int(round(((w + h) * 2 + 1) / rho))
    cos, sin = trig_table(n_theta, theta, rho, exact=exact_angle)
    ys, xs = np.nonzero(edges)
    if dtype is np.float64:
        cos, sin = cos.astype(np.float64), sin.astype(np.float64)
        if exact_angle:
            ang = np.arange(n_theta) * theta
            cos, sin = np.cos(ang) / rho, np.sin(ang) / rho
    r = np.rint(xs.astype(dtype)[:, None] * cos[None, :] + ys.astype(dtype)[:, None] * sin[None, :])
    r = r.astype(np.int64) + (n_rho - 1) // 2
    acc = np.zeros((n_rho, n_theta), np.int32)
    np.add.at(acc, (r, np.broadcast_to(np.arange(n_theta), r.shape)), 1)
    return acc


def local_maxima(acc: np.ndarray, threshold: int) -> list[tuple[int, int, int]]:
    """OpenCV's rule: above threshold, > the lower neighbour and >= the upper, on both axes.

    Returned as (votes, rho_index, theta_index), strongest first.
    """
    A = np.pad(acc, 1)
    c = A[1:-1, 1:-1]
    m = (
        (c > threshold)
        & (c > A[:-2, 1:-1])
        & (c >= A[2:, 1:-1])
        & (c > A[1:-1, :-2])
        & (c >= A[1:-1, 2:])
    )
    ri, ti = np.nonzero(m)
    out = sorted(zip(acc[ri, ti].tolist(), ri.tolist(), ti.tolist()), key=lambda t: -t[0])
    return out


def restricted_accumulator(
    edges: np.ndarray, orientation_deg: np.ndarray, k: int, theta_deg: float = 1.0
) -> tuple[np.ndarray, int]:
    """Each edge pixel votes only for theta within ±k degrees of its gradient direction.

    `orientation_deg` is the gradient angle at every pixel, folded to [0, 180).
    Returns the accumulator and the number of votes cast.
    """
    h, w = edges.shape
    n_theta = int(round(180 / theta_deg))
    n_rho = (w + h) * 2 + 1
    cos, sin = trig_table(n_theta, np.radians(theta_deg))
    ys, xs = np.nonzero(edges)
    centre = np.rint(orientation_deg[ys, xs] / theta_deg).astype(np.int64)
    offs = np.arange(-k, k + 1)
    # Wrapping the index is enough: rho is computed at the wrapped angle itself, and
    # (rho, theta) and (-rho, theta + 180) are the same line.
    t = (centre[:, None] + offs[None, :]) % n_theta
    r = np.rint(
        xs.astype(np.float32)[:, None] * cos[t] + ys.astype(np.float32)[:, None] * sin[t]
    ).astype(np.int64)
    r = r + (n_rho - 1) // 2
    acc = np.zeros((n_rho, n_theta), np.int32)
    np.add.at(acc, (r, t), 1)
    return acc, int(r.size)


def rho_of(index: int, acc: np.ndarray) -> int:
    return int(index - (acc.shape[0] - 1) // 2)
