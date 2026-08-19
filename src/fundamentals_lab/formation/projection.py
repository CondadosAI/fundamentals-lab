"""Lesson 1: predict where a scene point lands, then look at where it actually is.

The pinhole model is a prediction, not a definition, so it can be wrong and the
amount it is wrong by is measurable. This module projects the board corners of one
view with the two scalar equations alone -- no distortion terms -- and compares
that against the sub-pixel corner the detector found.
"""

import cv2
import numpy as np

from fundamentals_lab.formation.calibration import board_points


def _pose(obj_points: list, img_points: list, size: tuple) -> tuple:
    """Recover K, the distortion coefficients and every view's pose.

    Calibration is re-run here rather than read from the artifact because the
    per-view poses are not published in it. It is deterministic, so the intrinsics
    that come back are the same ones formation-calibrate wrote.
    """
    objs = [o.astype(np.float32) for o in obj_points]
    imgs = [i.astype(np.float32).reshape(-1, 1, 2) for i in img_points]
    return cv2.calibrateCamera(objs, imgs, size, None, None)


def project_view(obj_points: list, img_points: list, size: tuple, view: int) -> dict:
    """Project one view's corners with the pinhole equations and measure the error.

    Returns the worked example a lesson can print by hand, plus the residual split
    by how far the corner sits from the principal point -- which is where the
    pinhole model's error lives.
    """
    _, k, dist, rvecs, tvecs = _pose(obj_points, img_points, size)
    fx, fy = float(k[0, 0]), float(k[1, 1])
    cx, cy = float(k[0, 2]), float(k[1, 2])

    rot, _ = cv2.Rodrigues(rvecs[view])
    board = board_points()
    camera_frame = (rot @ board.T + tvecs[view]).T  # one row per corner, in metres of board unit

    # The lesson's two equations, written out rather than delegated to a matrix.
    pinhole = np.column_stack(
        [
            cx + fx * camera_frame[:, 0] / camera_frame[:, 2],
            cy + fy * camera_frame[:, 1] / camera_frame[:, 2],
        ]
    )
    measured = img_points[view].reshape(-1, 2)
    error = np.linalg.norm(pinhole - measured, axis=1)
    radius = np.linalg.norm(measured - np.array([cx, cy]), axis=1)

    # The full model, for the size of the gap the pinhole equations leave behind.
    full, _ = cv2.projectPoints(
        board.astype(np.float32), rvecs[view], tvecs[view], k, dist
    )
    full_error = np.linalg.norm(full.reshape(-1, 2) - measured, axis=1)

    # Split by the data rather than by a fixed radius: on a board that fills only
    # part of the frame there may be no corner past an arbitrary cutoff, and a
    # threshold that silently selects nothing is worse than no threshold.
    order = np.argsort(radius)
    third = max(1, len(order) // 3)
    near, far = order[:third], order[-third:]
    worst = int(np.argmax(error))
    return {
        "view_index": view,
        "fx": fx,
        "fy": fy,
        "cx": cx,
        "cy": cy,
        "worked_corner": {
            "board_point": board[worst].tolist(),
            "camera_frame_point": camera_frame[worst].tolist(),
            "pinhole_prediction_px": pinhole[worst].tolist(),
            "measured_px": measured[worst].tolist(),
            "error_px": float(error[worst]),
            "radius_px": float(radius[worst]),
        },
        "pinhole_only_error_px": {
            "mean": float(error.mean()),
            "max": float(error.max()),
            "innermost_third": {
                "mean_error_px": float(error[near].mean()),
                "mean_radius_px": float(radius[near].mean()),
                "corners": int(len(near)),
            },
            "outermost_third": {
                "mean_error_px": float(error[far].mean()),
                "mean_radius_px": float(radius[far].mean()),
                "corners": int(len(far)),
            },
        },
        "full_model_error_px": {
            "mean": float(full_error.mean()),
            "max": float(full_error.max()),
        },
    }
