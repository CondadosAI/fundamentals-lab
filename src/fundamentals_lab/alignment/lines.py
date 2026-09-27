"""Painted lines as 3-vectors, fitted to paint pixels, and the corners where they meet.

A clicked corner is only as good as the click, and one of the four corners this
unit needs is off the frame. Fitting a line to thousands of paint pixels and
intersecting two fits gives each corner to a fraction of a pixel, including the
one nobody could click.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from fundamentals_lab.alignment import court, transforms
from fundamentals_lab.config import (
    ALIGN_SEED_CORNERS,
    LINE_BAND_PX,
    LINE_REFINE_PASSES,
    NET_POLYGON,
    TOPHAT_KERNEL_PX,
    TOPHAT_MIN,
    WHITE_MAX_SAT,
)


def paint_pixels(image: np.ndarray, exclude_net: bool = True) -> np.ndarray:
    """(N, 2) pixel coordinates of white paint, net region removed.

    Paint is a thin bright stripe, so it is found as a ridge rather than as a
    colour: a white top-hat (the image minus its morphological opening) keeps
    what is brighter than its surroundings over a width narrower than the kernel.
    A brightness threshold alone also admits the pale grey of the non-volley zone,
    which sits on one side of the line and drags the fit towards it.
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (TOPHAT_KERNEL_PX, TOPHAT_KERNEL_PX))
    ridge = cv2.morphologyEx(hsv[:, :, 2], cv2.MORPH_TOPHAT, kernel)
    mask = ((ridge > TOPHAT_MIN) & (hsv[:, :, 1] < WHITE_MAX_SAT)).astype(np.uint8) * 255
    if exclude_net:
        cv2.fillPoly(mask, [np.array(NET_POLYGON, np.int32)], 0)
    ys, xs = np.nonzero(mask)
    return np.stack([xs, ys], axis=1).astype(np.float64)


def band(pixels: np.ndarray, p: np.ndarray, q: np.ndarray, half_width: float = LINE_BAND_PX):
    """Pixels within `half_width` of segment p-q, and their position along it (0..1)."""
    d = q - p
    length = np.linalg.norm(d)
    d = d / length
    n = np.array([-d[1], d[0]])
    t = (pixels - p) @ d
    s = (pixels - p) @ n
    keep = (np.abs(s) < half_width) & (t > 0) & (t < length)
    return pixels[keep], t[keep] / length


@dataclass
class LineFit:
    line: np.ndarray  # (a, b, c) with a^2 + b^2 = 1: signed distance = a x + b y + c
    n_pixels: int
    rms_px: float


def fit_line(points: np.ndarray, passes: int = 4) -> LineFit:
    """Total least squares, trimmed: the normal is the least-variance direction.

    Ordinary least squares measures error vertically and fails on steep lines;
    TLS measures it perpendicular to the line. Each pass drops pixels further than
    2.5 robust standard deviations, which removes a shoe or a stray reflection
    without deciding in advance what an outlier looks like.
    """
    P = points
    for _ in range(passes):
        c = P.mean(axis=0)
        _, _, vt = np.linalg.svd(P - c, full_matrices=False)
        r = (P - c) @ vt[1]
        sigma = 1.4826 * np.median(np.abs(r))
        P = P[np.abs(r) < max(2.5 * sigma, 1.0)]
    c = P.mean(axis=0)
    _, _, vt = np.linalg.svd(P - c, full_matrices=False)
    normal = vt[1]
    r = (P - c) @ normal
    return LineFit(
        np.array([normal[0], normal[1], -normal @ c]), len(P), float(np.sqrt(np.mean(r**2)))
    )


def seed_homography(seed_corners=ALIGN_SEED_CORNERS) -> np.ndarray:
    """H_court2img from four approximate corners. Only used to know where to look."""
    return transforms.dlt(court.court_points(court.FIT_LANDMARKS), np.asarray(seed_corners, float))


@dataclass
class CourtFit:
    lines: dict[str, LineFit]
    H_court2img: np.ndarray  # from the four fitted near corners
    landmarks_img: dict[str, np.ndarray]


def fit_court(
    pixels: np.ndarray,
    names=court.NEAR_LINES,
    seed_corners=ALIGN_SEED_CORNERS,
    passes: int = LINE_REFINE_PASSES,
) -> CourtFit:
    """Fit the named lines, intersect them, refit H, look again. `passes` rounds."""
    H = seed_homography(seed_corners)
    for _ in range(passes + 1):
        fits = {}
        for name in names:
            p, q = transforms.apply(H, np.array(court.LINES[name]))
            pts, _ = band(pixels, p, q)
            fits[name] = fit_line(pts)
        corners = np.array([corner(fits, n) for n in court.FIT_LANDMARKS])
        H = transforms.dlt(court.court_points(court.FIT_LANDMARKS), corners)
    marks = {n: corner(fits, n) for n, (a, b) in court.LANDMARKS.items() if a in fits and b in fits}
    return CourtFit(fits, H, marks)


def corner(fits: dict[str, LineFit], name: str) -> np.ndarray:
    a, b = court.LANDMARKS[name]
    return court.dehomogenise(court.meet(fits[a].line, fits[b].line))


def sagitta(points: np.ndarray, line: np.ndarray) -> float:
    """How far a line bows: the quadratic term of residual-vs-position, at the ends.

    A straight stripe gives ~0; a stripe bent by the lens gives the pixel distance
    between its middle and the chord through its ends.
    """
    normal = line[:2]
    direction = np.array([-normal[1], normal[0]])
    t = points @ direction
    t = t - t.mean()
    r = points @ normal + line[2]
    coef = np.polyfit(t, r, 2)
    half = (t.max() - t.min()) / 2
    return float(abs(coef[0]) * half**2)
