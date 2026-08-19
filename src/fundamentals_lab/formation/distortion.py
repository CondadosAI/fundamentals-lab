"""What the measured lens does to a pixel, and what straightening it costs."""

import cv2
import numpy as np


def distort_points(points_px: np.ndarray, K: np.ndarray, dist: np.ndarray) -> np.ndarray:
    """Push ideal pixel positions through the Brown-Conrady model.

    OpenCV's model runs ideal -> distorted, which is the direction that answers
    "where does the lens actually put this point?". cv2.undistortPoints runs the
    other way and is the wrong tool for that question.
    """
    normalized = np.linalg.inv(K) @ np.vstack([points_px.T, np.ones(len(points_px))])
    xy = (normalized[:2] / normalized[2]).T.reshape(-1, 1, 2).astype(np.float64)
    # projectPoints with an identity pose applies distortion and then K.
    projected, _ = cv2.projectPoints(
        np.hstack([xy.reshape(-1, 2), np.ones((len(xy), 1))]),
        np.zeros(3),
        np.zeros(3),
        K,
        dist,
    )
    return projected.reshape(-1, 2)


def displacement_table(K: np.ndarray, dist: np.ndarray, size: tuple) -> dict:
    """How far the lens moves a point, at the four corners and along the diagonal."""
    w, h = size
    cx, cy = K[0, 2], K[1, 2]
    corners = np.array(
        [[0.0, 0.0], [w - 1.0, 0.0], [0.0, h - 1.0], [w - 1.0, h - 1.0]], dtype=np.float64
    )
    moved = distort_points(corners, K, dist)
    corner_shift = np.linalg.norm(moved - corners, axis=1)

    # Radial profile: sample the diagonal from the principal point outwards, so
    # the article can show the shift growing with r rather than asserting it.
    far = np.array([w - 1.0, h - 1.0])
    centre = np.array([cx, cy])
    ts = np.linspace(0.0, 1.0, 11)
    ray = np.array([centre + t * (far - centre) for t in ts])
    ray_moved = distort_points(ray, K, dist)
    radial = [
        {
            "r_px": float(np.linalg.norm(p - centre)),
            "shift_px": float(np.linalg.norm(q - p)),
        }
        for p, q in zip(ray, ray_moved, strict=True)
    ]
    return {
        "corner_shift_px": [float(v) for v in corner_shift],
        "max_corner_shift_px": float(corner_shift.max()),
        "radial_profile": radial,
    }


def hfov_deg(fx: float, width: int) -> float:
    """Horizontal field of view implied by a focal length, in degrees."""
    return float(2.0 * np.degrees(np.arctan((width / 2.0) / fx)))


def undistort_cost(K: np.ndarray, dist: np.ndarray, size: tuple) -> dict:
    """What straightening the image costs, measured on the map rather than asserted.

    getOptimalNewCameraMatrix also returns a "valid" ROI, and on this camera it
    returns the full frame while simultaneously choosing a shorter focal length --
    the two cannot both be true. So the cost is counted directly instead: build
    the remap and count how many output pixels have no source pixel to read.
    """
    w, h = size
    out = {"hfov_deg_before": hfov_deg(float(K[0, 0]), w)}

    for label, new_k in (
        ("same_k", K.copy()),
        ("alpha0", cv2.getOptimalNewCameraMatrix(K, dist, (w, h), 0)[0]),
        ("alpha1", cv2.getOptimalNewCameraMatrix(K, dist, (w, h), 1)[0]),
    ):
        map_x, map_y = cv2.initUndistortRectifyMap(K, dist, None, new_k, (w, h), cv2.CV_32FC1)
        outside = (map_x < 0) | (map_x > w - 1) | (map_y < 0) | (map_y > h - 1)
        out[label] = {
            "new_fx": float(new_k[0, 0]),
            "hfov_deg_after": hfov_deg(float(new_k[0, 0]), w),
            "empty_output_pct": float(100.0 * outside.mean()),
        }
    return out
