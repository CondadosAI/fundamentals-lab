"""The working frame and its edge map, built the way unit 3.1 builds them."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from fundamentals_lab.config import (
    BOUNDARY_CANNY,
    BOUNDARY_DIR,
    BOUNDARY_FRAME,
    BOUNDARY_SIGMA,
    BOUNDARY_WIDTH,
)


def normal_frames() -> list[Path]:
    return sorted((BOUNDARY_DIR / "Normal").glob("*.JPG"))


def load(name: str = BOUNDARY_FRAME) -> np.ndarray:
    """The frame at BOUNDARY_WIDTH, BGR uint8."""
    path = BOUNDARY_DIR / "Normal" / name
    img = cv2.imread(str(path))
    if img is None:
        raise FileNotFoundError(f"{path}: run boundary-download first")
    return working(img)


def working(img: np.ndarray) -> np.ndarray:
    s = BOUNDARY_WIDTH / img.shape[1]
    return cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)


def blurred(bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    return cv2.GaussianBlur(gray, (0, 0), BOUNDARY_SIGMA)


def edge_map(bgr: np.ndarray) -> np.ndarray:
    return cv2.Canny(blurred(bgr), *BOUNDARY_CANNY)


def box_points(edges: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    """Edge pixels inside (x0, y0, x1, y1) as an (N, 2) array of (x, y)."""
    x0, y0, x1, y1 = box
    ys, xs = np.nonzero(edges[y0:y1, x0:x1])
    return np.c_[xs + x0, ys + y0].astype(np.float64)
