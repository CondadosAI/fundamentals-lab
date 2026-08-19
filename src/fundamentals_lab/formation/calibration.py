"""Calibrate a camera from chessboard photographs, and report per-view honesty."""

import cv2
import numpy as np
from loguru import logger

from fundamentals_lab.config import FORMATION_BOARD, FORMATION_SQUARE_SIZE

# Sub-pixel refinement: without it the corner locations are integer pixels and
# the reprojection error is dominated by that quantisation rather than by the
# lens model being fitted.
CRITERIA = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-3)
WIN = (11, 11)


def board_points() -> np.ndarray:
    """The board's own coordinates: z = 0, one point per interior crossing."""
    pts = np.zeros((FORMATION_BOARD[0] * FORMATION_BOARD[1], 3), np.float32)
    pts[:, :2] = np.mgrid[0 : FORMATION_BOARD[0], 0 : FORMATION_BOARD[1]].T.reshape(-1, 2)
    return pts * FORMATION_SQUARE_SIZE


def find_corners(paths: list) -> tuple:
    """Locate the chessboard in each photograph.

    Returns (object_points, image_points, image_size, used, skipped). A view
    where the board is not fully visible is skipped rather than half-used, and
    the skip is reported: a silently dropped view changes the numbers.
    """
    objp = board_points()
    obj_points, img_points, used, skipped = [], [], [], []
    size = None
    for path in paths:
        img = cv2.imread(str(path))
        if img is None:
            raise RuntimeError(f"could not read {path}")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if size is None:
            size = (gray.shape[1], gray.shape[0])
        elif size != (gray.shape[1], gray.shape[0]):
            raise RuntimeError(f"{path.name} is {gray.shape[1]}x{gray.shape[0]}, expected {size}")
        found, corners = cv2.findChessboardCorners(gray, FORMATION_BOARD, None)
        if not found:
            skipped.append(path.name)
            logger.warning(f"{path.name}: board not found, skipping")
            continue
        refined = cv2.cornerSubPix(gray, corners, WIN, (-1, -1), CRITERIA)
        obj_points.append(objp)
        img_points.append(refined)
        used.append(path.name)
    return obj_points, img_points, size, used, skipped


def calibrate_camera(obj_points: list, img_points: list, size: tuple) -> dict:
    """Fit K and the Brown-Conrady coefficients, with a per-view error breakdown."""
    rms, K, dist, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, size, None, None)
    per_view = []
    for i, (objp, imgp) in enumerate(zip(obj_points, img_points, strict=True)):
        projected, _ = cv2.projectPoints(objp, rvecs[i], tvecs[i], K, dist)
        # RMS over the view's own points, matching how OpenCV aggregates them.
        err = np.sqrt(np.mean(np.sum((imgp - projected).reshape(-1, 2) ** 2, axis=1)))
        per_view.append(float(err))
    return {
        "rms_reprojection_error_px": float(rms),
        "K": K.tolist(),
        "fx": float(K[0, 0]),
        "fy": float(K[1, 1]),
        "cx": float(K[0, 2]),
        "cy": float(K[1, 2]),
        "dist": dist.ravel().tolist(),
        "k1": float(dist.ravel()[0]),
        "k2": float(dist.ravel()[1]),
        "p1": float(dist.ravel()[2]),
        "p2": float(dist.ravel()[3]),
        "k3": float(dist.ravel()[4]),
        "per_view_reprojection_error_px": per_view,
        "image_size": list(size),
    }
