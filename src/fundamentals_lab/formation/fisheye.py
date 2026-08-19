"""Fit both camera models to the same fisheye photographs and let them disagree.

Lesson 4's claim is that past a certain angle the pinhole model plus Brown-Conrady
distortion stops being a lens correction and starts being the wrong model. The way
to show that rather than assert it is to fit both to one set of images and compare
them with the same error metric.
"""

import cv2
import numpy as np

# The board in test_images/fish_1 has 6x8 interior corners, found in all 15 views.
# The repository's own checkerboard_sizes.txt says 5x7, which findChessboardCorners
# accepts on 4 of 15 images because a smaller grid fits inside a larger one. Taking
# the file at its word would calibrate on a third of the data and a wrong geometry.
FISHEYE_BOARD = (6, 8)
CRITERIA = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, 1e-8)


def board_points(board: tuple) -> np.ndarray:
    pts = np.zeros((board[0] * board[1], 3), np.float64)
    pts[:, :2] = np.mgrid[0 : board[0], 0 : board[1]].T.reshape(-1, 2)
    return pts


def find_corners(paths: list, board: tuple) -> tuple:
    objp = board_points(board)
    obj_points, img_points, used = [], [], []
    size = None
    for path in paths:
        gray = cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2GRAY)
        size = (gray.shape[1], gray.shape[0]) if size is None else size
        found, corners = cv2.findChessboardCorners(gray, board, None)
        if not found:
            continue
        refined = cv2.cornerSubPix(gray, corners, (5, 5), (-1, -1), CRITERIA)
        obj_points.append(objp)
        img_points.append(refined.astype(np.float64))
        used.append(path.name)
    return obj_points, img_points, size, used


def _rms(observed: np.ndarray, projected: np.ndarray) -> float:
    """One error metric, used for both models so the comparison means something."""
    d = observed.reshape(-1, 2) - projected.reshape(-1, 2)
    return float(np.sqrt(np.mean(np.sum(d**2, axis=1))))


def fit_pinhole(obj_points: list, img_points: list, size: tuple) -> dict:
    """The model unit 1.1 lessons 1-3 taught, applied to a lens it was never meant for."""
    objs = [o.astype(np.float32) for o in obj_points]
    imgs = [i.astype(np.float32).reshape(-1, 1, 2) for i in img_points]
    _, K, dist, rvecs, tvecs = cv2.calibrateCamera(objs, imgs, size, None, None)
    errors = []
    for i, obj in enumerate(objs):
        proj, _ = cv2.projectPoints(obj, rvecs[i], tvecs[i], K, dist)
        errors.append(_rms(imgs[i], proj))
    return {
        "rms_reprojection_error_px": _rms(
            np.vstack([i.reshape(-1, 2) for i in imgs]),
            np.vstack(
                [
                    cv2.projectPoints(objs[i], rvecs[i], tvecs[i], K, dist)[0].reshape(-1, 2)
                    for i in range(len(objs))
                ]
            ),
        ),
        "worst_view_rms_px": float(max(errors)),
        "fx": float(K[0, 0]),
        "cx": float(K[0, 2]),
        "cy": float(K[1, 2]),
        "dist": dist.ravel().tolist(),
    }


def fit_fisheye(obj_points: list, img_points: list, size: tuple) -> dict:
    """Kannala-Brandt, which needs to be told roughly where to start.

    Left to initialise K itself the solver lands 200 px from the corners on this
    lens -- it converges to a focal length it cannot recover from. Seeded with the
    equidistant-projection estimate f = max(w, h) / pi, the same data fits to well
    under a pixel. The guess is not a tuning knob; it is the difference between a
    fit and a failure, and it is the kind of thing a tutorial leaves out.
    """
    w, h = size
    f0 = max(w, h) / np.pi
    K = np.array([[f0, 0.0, w / 2.0], [0.0, f0, h / 2.0], [0.0, 0.0, 1.0]])
    D = np.zeros((4, 1))
    n = len(obj_points)
    objs = [o.reshape(1, -1, 3) for o in obj_points]
    imgs = [i.reshape(1, -1, 2) for i in img_points]
    flags = (
        cv2.CALIB_USE_INTRINSIC_GUESS
        | cv2.CALIB_RECOMPUTE_EXTRINSIC
        # OpenCV 5 moved these out of the cv2.fisheye namespace: cv2.fisheye.CALIB_*
        # no longer exists, so 4.x tutorials raise AttributeError here.
        | cv2.CALIB_FIX_SKEW
    )
    _, K, D, rvecs, tvecs = cv2.fisheye.calibrate(
        objs,
        imgs,
        size,
        K,
        D,
        [np.zeros((1, 1, 3)) for _ in range(n)],
        [np.zeros((1, 1, 3)) for _ in range(n)],
        flags,
        CRITERIA,
    )
    errors, all_obs, all_proj = [], [], []
    for i in range(n):
        proj, _ = cv2.fisheye.projectPoints(objs[i], rvecs[i], tvecs[i], K, D)
        errors.append(_rms(imgs[i], proj))
        all_obs.append(imgs[i].reshape(-1, 2))
        all_proj.append(proj.reshape(-1, 2))
    return {
        "rms_reprojection_error_px": _rms(np.vstack(all_obs), np.vstack(all_proj)),
        "worst_view_rms_px": float(max(errors)),
        "f0_guess_px": float(f0),
        "fx": float(K[0, 0]),
        "cx": float(K[0, 2]),
        "cy": float(K[1, 2]),
        "k": D.ravel().tolist(),
    }


def divergence_table(pinhole: dict, fisheye: dict) -> list:
    """Where each model puts a ray that arrives at incidence angle theta.

    Both models are asked the same question -- how far from the principal point
    does this ray land -- using each one's own fitted parameters.
    """
    k1, k2, _p1, _p2, k3 = pinhole["dist"][:5]
    f_pin, f_fish, kf = pinhole["fx"], fisheye["fx"], fisheye["k"]
    rows = []
    for deg in [5, 15, 30, 45, 60, 70, 80, 85, 89]:
        th = np.radians(deg)
        # Pinhole: the ray meets the ideal image plane at tan(theta), then distorts.
        r = np.tan(th)
        r_pin = f_pin * r * (1 + k1 * r**2 + k2 * r**4 + k3 * r**6)
        # Fisheye: the angle itself is the radial coordinate.
        th_d = th * (1 + kf[0] * th**2 + kf[1] * th**4 + kf[2] * th**6 + kf[3] * th**8)
        rows.append(
            {
                "theta_deg": deg,
                "pinhole_r_px": float(r_pin),
                "fisheye_r_px": float(f_fish * th_d),
            }
        )
    return rows
