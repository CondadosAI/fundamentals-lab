"""Warping the court to a bird's-eye view, two ways, and blending frames together.

Forward mapping pushes every source pixel to where H sends it. Backward mapping
asks, for every output pixel, where it came from. Only the second fills every
output pixel exactly once, which is why every library warps backwards.
"""

from __future__ import annotations

import cv2
import numpy as np

from fundamentals_lab.alignment import transforms
from fundamentals_lab.config import (
    COURT_LENGTH_M,
    COURT_WIDTH_M,
    TOPVIEW_MARGIN_M,
    TOPVIEW_PX_PER_M,
)


def court2top() -> tuple[np.ndarray, tuple[int, int]]:
    """Court metres -> top-view pixels, with the long axis running up the picture.

    Returns (H_court2top, (width, height)). The near baseline is at the bottom, as
    the camera sees it.
    """
    s, m = TOPVIEW_PX_PER_M, TOPVIEW_MARGIN_M
    width = round((COURT_WIDTH_M + 2 * m) * s)
    height = round((COURT_LENGTH_M + 2 * m) * s)
    # top-x = (y + m) s ; top-y = (L + m - x) s
    H = np.array([[0, s, m * s], [-s, 0, (COURT_LENGTH_M + m) * s], [0, 0, 1.0]])
    return H, (width, height)


def img2top(H_img2court: np.ndarray) -> tuple[np.ndarray, tuple[int, int]]:
    H_c2t, size = court2top()
    return H_c2t @ H_img2court, size


def backward(image: np.ndarray, H_src2dst: np.ndarray, size, interpolation: str = "bilinear"):
    """For each output pixel, pull from src = H^-1 dst. Written out, not cv2."""
    w, h = size
    H_dst2src = np.linalg.inv(H_src2dst)
    xs, ys = np.meshgrid(np.arange(w, dtype=np.float64), np.arange(h, dtype=np.float64))
    src = transforms.apply(H_dst2src, np.stack([xs.ravel(), ys.ravel()], axis=1))
    sx, sy = src[:, 0].reshape(h, w), src[:, 1].reshape(h, w)
    H_img, W_img = image.shape[:2]
    out = np.zeros((h, w, image.shape[2]), dtype=np.float64)
    if interpolation == "nearest":
        ix, iy = np.rint(sx).astype(int), np.rint(sy).astype(int)
        ok = (ix >= 0) & (ix < W_img) & (iy >= 0) & (iy < H_img)
        out[ok] = image[iy[ok], ix[ok]]
    else:
        x0, y0 = np.floor(sx).astype(int), np.floor(sy).astype(int)
        fx, fy = (sx - x0)[..., None], (sy - y0)[..., None]
        ok = (x0 >= 0) & (x0 < W_img - 1) & (y0 >= 0) & (y0 < H_img - 1)
        x0c, y0c = np.clip(x0, 0, W_img - 2), np.clip(y0, 0, H_img - 2)
        img = image.astype(np.float64)
        top = img[y0c, x0c] * (1 - fx) + img[y0c, x0c + 1] * fx
        bot = img[y0c + 1, x0c] * (1 - fx) + img[y0c + 1, x0c + 1] * fx
        val = top * (1 - fy) + bot * fy
        out[ok] = val[ok]
    return np.clip(np.rint(out), 0, 255).astype(np.uint8), ok


def forward(image: np.ndarray, H_src2dst: np.ndarray, size, region: np.ndarray | None = None):
    """Push every source pixel (optionally only those in `region`) to its nearest output pixel.

    Returns (output, hit) where `hit` counts how many source pixels landed on each
    output pixel: zeros are holes, values above one are pixels written twice.
    """
    w, h = size
    H_img, W_img = image.shape[:2]
    ys, xs = np.nonzero(region) if region is not None else np.mgrid[0:H_img, 0:W_img].reshape(2, -1)
    dst = transforms.apply(H_src2dst, np.stack([xs, ys], axis=1).astype(np.float64))
    dx, dy = np.rint(dst[:, 0]).astype(int), np.rint(dst[:, 1]).astype(int)
    ok = (dx >= 0) & (dx < w) & (dy >= 0) & (dy < h)
    out = np.zeros((h, w, image.shape[2]), np.uint8)
    hit = np.zeros((h, w), np.int32)
    out[dy[ok], dx[ok]] = image[ys[ok], xs[ok]]
    np.add.at(hit, (dy[ok], dx[ok]), 1)
    return out, hit


def court_mask_top(size) -> np.ndarray:
    """Top-view pixels that lie on the court itself (inside the outer lines)."""
    H_c2t, _ = court2top()
    corners = transforms.apply(
        H_c2t,
        np.array(
            [[0, 0], [COURT_LENGTH_M, 0], [COURT_LENGTH_M, COURT_WIDTH_M], [0, COURT_WIDTH_M]]
        ),
    )
    mask = np.zeros((size[1], size[0]), np.uint8)
    cv2.fillPoly(mask, [np.rint(corners).astype(np.int32)], 1)
    return mask.astype(bool)
