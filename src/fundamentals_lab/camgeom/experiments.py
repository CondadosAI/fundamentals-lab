"""Every number unit 4.1's lessons 2 to 4 publish, on OpenCV's 13 chessboard photographs.

The board's square size is not published with these images, so everything runs in board
units (one unit = one square): K, the distortion and every pixel error are unaffected, and
no translation here is in metres.
"""

from __future__ import annotations

import itertools

import cv2
import numpy as np

from fundamentals_lab.formation.calibration import find_corners
from fundamentals_lab.formation.dataset import fetch_opencv_calibration

SEED = 20261004
WRONG_IDX = (5, 17, 23, 38, 50)
WRONG_OFFSETS = ((40, -25), (-35, 30), (30, 35), (-40, -20), (25, -40))


def _r(x, n=4):
    return None if x is None else round(float(x), n)


def _rms(obj, img, rvec, tvec, K, dist):
    p, _ = cv2.projectPoints(obj, rvec, tvec, K, dist)
    return float(np.sqrt(np.mean(np.sum((p.reshape(-1, 2) - img.reshape(-1, 2)) ** 2, 1))))


def _angle_deg(r1, r2):
    """Angle of the rotation that takes pose 1 to pose 2."""
    R1, _ = cv2.Rodrigues(r1)
    R2, _ = cv2.Rodrigues(r2)
    c = (np.trace(R1.T @ R2) - 1) / 2
    return float(np.degrees(np.arccos(np.clip(c, -1, 1))))


def _centre(rvec, tvec):
    R, _ = cv2.Rodrigues(rvec)
    return (-R.T @ np.asarray(tvec).reshape(3)).ravel()


class Boards:
    """The 13 views, their detected corners and the full calibration, computed once."""

    def __init__(self):
        paths = fetch_opencv_calibration()
        self.obj, self.img, self.size, self.used, _ = find_corners(paths)
        self.paths = [p for p in paths if p.name in self.used]
        self.rms, self.K, self.dist, self.rvecs, self.tvecs = cv2.calibrateCamera(
            self.obj, self.img, self.size, None, None
        )


def camera_matrix(b: Boards) -> dict:
    """Lesson 2: K, view 0's [R|t], P = K[R|t], and one corner pushed through it."""
    R, _ = cv2.Rodrigues(b.rvecs[0])
    t = b.tvecs[0].reshape(3)
    P = b.K @ np.hstack([R, t[:, None]])
    X = np.array([8.0, 0.0, 0.0])  # the board corner the 1.1 lessons work by hand
    cam = R @ X + t
    h = P @ np.append(X, 1)
    seen = b.img[0].reshape(-1, 2)[8]
    full, _ = cv2.projectPoints(X[None].astype(np.float32), b.rvecs[0], b.tvecs[0], b.K, b.dist)
    C = _centre(b.rvecs[0], b.tvecs[0])
    return {
        "K": [[_r(v, 4) for v in row] for row in b.K],
        "dist": [_r(v, 5) for v in b.dist.ravel()],
        "view": b.used[0],
        "R": [[_r(v, 4) for v in row] for row in R],
        "t": [_r(v, 4) for v in t],
        "P": [[_r(v, 3) for v in row] for row in P],
        "corner_board": X.tolist(),
        "corner_camera": [_r(v, 4) for v in cam],
        "corner_homogeneous": [_r(v, 3) for v in h],
        "corner_pinhole_px": [_r(v, 2) for v in h[:2] / h[2]],
        "corner_with_lens_px": [_r(v, 2) for v in full.ravel()],
        "corner_detected_px": [_r(v, 2) for v in seen],
        "camera_centre_board_units": [_r(v, 3) for v in C],
        "camera_distance_board_units": _r(np.linalg.norm(C), 3),
        "hfov_deg": _r(np.degrees(2 * np.arctan(b.size[0] / 2 / b.K[0, 0])), 2),
    }


def calibration(b: Boards) -> dict:
    """Lesson 3: the fit, what it scores on views it never saw, and what fewer views cost."""
    n = len(b.obj)
    # leave one view out: calibrate on the other 12, then place the held-out board by PnP
    # with that K and score it
    held = []
    for i in range(n):
        keep = [j for j in range(n) if j != i]
        _, K, d, _, _ = cv2.calibrateCamera(
            [b.obj[j] for j in keep], [b.img[j] for j in keep], b.size, None, None
        )
        _, rv, tv = cv2.solvePnP(b.obj[i], b.img[i], K, d)
        held.append(_rms(b.obj[i], b.img[i], rv, tv, K, d))
    in_sample = []
    for i in range(n):
        in_sample.append(_rms(b.obj[i], b.img[i], b.rvecs[i], b.tvecs[i], b.K, b.dist))
    # fewer views: every 3-view subset
    fx3 = []
    for combo in itertools.combinations(range(n), 3):
        try:
            _, K, _, _, _ = cv2.calibrateCamera(
                [b.obj[j] for j in combo], [b.img[j] for j in combo], b.size, None, None
            )
            fx3.append(float(K[0, 0]))
        except cv2.error:
            continue
    fx3 = np.array(fx3)
    # no lens model: the pinhole alone
    flags = cv2.CALIB_FIX_K1 | cv2.CALIB_FIX_K2 | cv2.CALIB_FIX_K3 | cv2.CALIB_ZERO_TANGENT_DIST
    rms0, K0, _, _, _ = cv2.calibrateCamera(b.obj, b.img, b.size, None, np.zeros(5), flags=flags)
    # the worked example: the four outer corners of the first view, detected minus predicted
    p0, _ = cv2.projectPoints(b.obj[0], b.rvecs[0], b.tvecs[0], b.K, b.dist)
    res = (b.img[0].reshape(-1, 2) - p0.reshape(-1, 2))[[0, 8, 45, 53]]
    worst = int(np.argmax(in_sample))
    return {
        "worked_residuals_px": [[_r(v, 3) for v in r] for r in res],
        "worked_rms_px": _r(np.sqrt(np.mean(np.sum(res**2, 1))), 3),
        "worst_view": {"name": b.used[worst], "rms_px": _r(in_sample[worst], 3)},
        "views": n,
        "corners_per_view": int(len(b.obj[0])),
        "rms_px": _r(b.rms, 4),
        "fx": _r(b.K[0, 0], 2),
        "fy": _r(b.K[1, 1], 2),
        "cx": _r(b.K[0, 2], 2),
        "cy": _r(b.K[1, 2], 2),
        "in_sample_view_rms_px": {
            "median": _r(np.median(in_sample), 3),
            "max": _r(max(in_sample), 3),
        },
        "held_out_view_rms_px": {"median": _r(np.median(held), 3), "max": _r(max(held), 3)},
        "three_view_fx": {
            "subsets": int(fx3.size),
            "median": _r(np.median(fx3), 1),
            "p05": _r(np.percentile(fx3, 5), 1),
            "p95": _r(np.percentile(fx3, 95), 1),
            "min": _r(fx3.min(), 1),
            "max": _r(fx3.max(), 1),
        },
        "no_distortion": {"rms_px": _r(rms0, 3), "fx": _r(K0[0, 0], 2)},
    }


def pnp(b: Boards) -> dict:
    """Lesson 4: pose from known points, against the calibration's own pose for each view."""
    rng = np.random.default_rng(SEED)
    obj0, img0 = b.obj[0], b.img[0]
    ok, rv, tv = cv2.solvePnP(obj0, img0, b.K, b.dist)
    full = {
        "rotation_vs_calibration_deg": _r(_angle_deg(rv, b.rvecs[0]), 5),
        "centre_vs_calibration_units": _r(
            np.linalg.norm(_centre(rv, tv) - _centre(b.rvecs[0], b.tvecs[0])), 5
        ),
        "rms_px": _r(_rms(obj0, img0, rv, tv, b.K, b.dist), 4),
        "centre_board_units": [_r(v, 3) for v in _centre(rv, tv)],
    }
    # four outer corners only, scored on all 54
    cols, rows = 9, 6
    four = [0, cols - 1, cols * (rows - 1), cols * rows - 1]
    _, rv4, tv4 = cv2.solvePnP(obj0[four], img0[four], b.K, b.dist, flags=cv2.SOLVEPNP_IPPE)
    four_pt = {
        "rms_all_54_px": _r(_rms(obj0, img0, rv4, tv4, b.K, b.dist), 3),
        "centre_shift_units": _r(np.linalg.norm(_centre(rv4, tv4) - _centre(rv, tv)), 3),
        "rotation_shift_deg": _r(_angle_deg(rv4, rv), 3),
    }
    # 1 px of noise on every corner, 200 times
    shifts = []
    for _ in range(200):
        noisy = img0 + rng.normal(0, 1.0, img0.shape).astype(np.float32)
        _, rvn, tvn = cv2.solvePnP(obj0, noisy, b.K, b.dist)
        shifts.append(np.linalg.norm(_centre(rvn, tvn) - _centre(rv, tv)))
    # five corners moved 40-50 px off, as a bad detection would; fixed so the page's cell
    # reproduces the same numbers
    wrong = img0.copy().reshape(-1, 2)
    idx = np.array(WRONG_IDX)
    wrong[idx] += np.array(WRONG_OFFSETS, np.float32)
    wrong = wrong.reshape(img0.shape).astype(np.float32)
    _, rvw, tvw = cv2.solvePnP(obj0, wrong, b.K, b.dist)
    _, rvr, tvr, inl = cv2.solvePnPRansac(obj0, wrong, b.K, b.dist, reprojectionError=3.0)
    outliers = {
        "wrong_corners": 5,
        "plain_centre_shift_units": _r(np.linalg.norm(_centre(rvw, tvw) - _centre(rv, tv)), 3),
        "plain_rms_on_good_px": _r(
            _rms(np.delete(obj0, idx, 0), np.delete(img0, idx, 0), rvw, tvw, b.K, b.dist), 3
        ),
        "ransac_inliers": int(len(inl)),
        "ransac_centre_shift_units": _r(np.linalg.norm(_centre(rvr, tvr) - _centre(rv, tv)), 4),
        "ransac_rms_on_good_px": _r(
            _rms(np.delete(obj0, idx, 0), np.delete(img0, idx, 0), rvr, tvr, b.K, b.dist), 3
        ),
    }
    dists = [float(np.linalg.norm(_centre(r, t))) for r, t in zip(b.rvecs, b.tvecs, strict=True)]
    rot, cen = [], []
    for o, im, r0, t0 in zip(b.obj, b.img, b.rvecs, b.tvecs, strict=True):
        _, r1, t1 = cv2.solvePnP(o, im, b.K, b.dist)
        rot.append(_angle_deg(r1, r0))
        cen.append(np.linalg.norm(_centre(r1, t1) - _centre(r0, t0)))
    return {
        "view": b.used[0],
        "all_54": full,
        "four_corners": four_pt,
        "noise_1px_200": {
            "centre_shift_median_units": _r(np.median(shifts), 4),
            "centre_shift_p95_units": _r(np.percentile(shifts, 95), 4),
            "camera_distance_units": _r(np.linalg.norm(_centre(rv, tv)), 3),
        },
        "outliers": outliers,
        "camera_distance_all_views_units": {"min": _r(min(dists), 2), "max": _r(max(dists), 2)},
        "all_views_vs_calibration": {
            "max_rotation_deg": _r(max(rot), 5),
            "max_centre_units": _r(max(cen), 5),
        },
    }


def corners_for_page(b: Boards) -> dict:
    """The 13 views' detected corners, rounded to 1/1000 px, for the page's calibration cell."""
    return {
        "views": b.used,
        "board": [9, 6],
        "image_size": list(b.size),
        "corners": [
            [[round(float(x), 3), round(float(y), 3)] for x, y in im.reshape(-1, 2)] for im in b.img
        ],
    }


def run_all() -> dict:
    b = Boards()
    return {
        "environment": {"opencv": cv2.__version__, "numpy": np.__version__, "seed": SEED},
        "board_units_note": "one unit = one chessboard square; the square's size is unpublished",
        "camera_matrix": camera_matrix(b),
        "calibration": calibration(b),
        "pnp": pnp(b),
    }, corners_for_page(b)
