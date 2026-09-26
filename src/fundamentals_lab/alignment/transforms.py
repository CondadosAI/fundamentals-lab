"""The transforms of the unit, written out rather than called.

Every function takes points as (N, 2) arrays and names its direction in the
argument, not the docstring: `src` is where the points are now, `dst` is where
the returned matrix sends them. OpenCV's equivalents are used only to check these
(`experiments.parity`), never in their place.
"""

from __future__ import annotations

import numpy as np


def to_h(points: np.ndarray) -> np.ndarray:
    """(N, 2) -> (N, 3): append the 1."""
    points = np.asarray(points, dtype=np.float64)
    return np.hstack([points, np.ones((len(points), 1))])


def apply(M: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Apply a 2x2, 2x3 or 3x3 transform to (N, 2) points."""
    points = np.asarray(points, dtype=np.float64)
    if M.shape == (2, 2):
        return points @ M.T
    if M.shape == (2, 3):
        return points @ M[:, :2].T + M[:, 2]
    x = to_h(points) @ M.T
    return x[:, :2] / x[:, 2:3]


# --- Lesson 1: the 2x2 --------------------------------------------------------


def fit_linear(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """Least-squares 2x2 with dst ~ M @ src. No translation: the origin stays put."""
    X, *_ = np.linalg.lstsq(np.asarray(src, float), np.asarray(dst, float), rcond=None)
    return X.T


# --- Lesson 2: affine ---------------------------------------------------------


def fit_affine(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """2x3 affine by least squares; with exactly 3 points it is exact.

    Six unknowns, two equations per point: [x y 1] times each row of the 2x3.
    """
    A = to_h(src)
    X, *_ = np.linalg.lstsq(A, np.asarray(dst, float), rcond=None)
    return X.T


# --- Lesson 3: the DLT --------------------------------------------------------


def normalising_transform(points: np.ndarray) -> np.ndarray:
    """Hartley's conditioning: centroid to the origin, mean distance sqrt(2)."""
    points = np.asarray(points, dtype=np.float64)
    c = points.mean(axis=0)
    mean_dist = np.linalg.norm(points - c, axis=1).mean()
    s = np.sqrt(2) / mean_dist
    return np.array([[s, 0, -s * c[0]], [0, s, -s * c[1]], [0, 0, 1.0]])


def dlt_matrix(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """The 2N x 9 system A h = 0. Each correspondence contributes two rows.

    From dst ~ H src: x' (h3 . x) = h1 . x and y' (h3 . x) = h2 . x.
    """
    rows = []
    for (x, y), (u, v) in zip(src, dst, strict=True):
        rows.append([-x, -y, -1, 0, 0, 0, u * x, u * y, u])
        rows.append([0, 0, 0, -x, -y, -1, v * x, v * y, v])
    return np.asarray(rows, dtype=np.float64)


def dlt(src: np.ndarray, dst: np.ndarray, normalise: bool = True) -> np.ndarray:
    """Homography with dst ~ H src, from four or more correspondences.

    h is the right singular vector of A with the smallest singular value: with
    four points A is 8 x 9 and that vector spans its null space exactly; with more
    it is the least-squares answer to |A h| = min subject to |h| = 1.
    """
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    if len(src) < 4:
        raise ValueError(f"a homography needs 4 correspondences, got {len(src)}")
    if normalise:
        T_src, T_dst = normalising_transform(src), normalising_transform(dst)
        H_n = dlt(apply(T_src, src), apply(T_dst, dst), normalise=False)
        H = np.linalg.inv(T_dst) @ H_n @ T_src
    else:
        _, _, vt = np.linalg.svd(dlt_matrix(src, dst))
        H = vt[-1].reshape(3, 3)
    return H / H[2, 2]


def condition_number(src: np.ndarray, dst: np.ndarray, normalise: bool) -> float:
    """Ratio of the largest to the second-smallest singular value of A.

    The smallest is (near) zero by design, since that is the solution, so the
    number that governs how noise in A moves h is sigma_1 / sigma_8.
    """
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    if normalise:
        src = apply(normalising_transform(src), src)
        dst = apply(normalising_transform(dst), dst)
    s = np.linalg.svd(dlt_matrix(src, dst), compute_uv=False)
    return float(s[0] / s[-2])


def decompose_2x2(M: np.ndarray) -> dict:
    """det, and the SVD M = R(theta2) diag(s1, s2) R(theta1) read as angles."""
    U, s, Vt = np.linalg.svd(M)
    return {
        "det": float(np.linalg.det(M)),
        "singular_values": [float(v) for v in s],
        "rotate_first_deg": float(np.degrees(np.arctan2(Vt[1, 0], Vt[0, 0]))),
        "rotate_after_deg": float(np.degrees(np.arctan2(U[1, 0], U[0, 0]))),
        "reflects": bool(np.linalg.det(M) < 0),
    }
