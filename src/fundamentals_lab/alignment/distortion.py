"""How much the lens bends the court lines, and one parameter that straightens them.

A homography maps straight lines to straight lines. A real lens does not quite, so
before asking how well a 3x3 fits this court, measure how straight the court's
lines are in the picture. The correction is the one-parameter division model
fitted by the plumb-line method: choose k so that lines that are straight on the
ground come out straight in the image. The distortion centre is assumed to be the
frame centre, and no calibration of this camera exists to check that against.
"""

from __future__ import annotations

import numpy as np

from fundamentals_lab.alignment import court, lines, transforms

IMAGE_SIZE = (1920, 1080)


def undistort(points: np.ndarray, k: float, size=IMAGE_SIZE) -> np.ndarray:
    """Division model, radius normalised by the half-diagonal: x_u = c + (x_d - c)/(1 + k r^2)."""
    c = np.array([size[0] / 2, size[1] / 2])
    s = np.hypot(*c)
    q = (np.asarray(points, float) - c) / s
    r2 = (q**2).sum(axis=1, keepdims=True)
    return c + s * q / (1 + k * r2)


def line_bands(pixels: np.ndarray, H_court2img: np.ndarray, names) -> dict[str, np.ndarray]:
    out = {}
    for name in names:
        p, q = transforms.apply(H_court2img, np.array(court.LINES[name]))
        out[name], _ = lines.band(pixels, p, q)
    return out


def total_sagitta(bands: dict[str, np.ndarray], k: float) -> dict[str, float]:
    out = {}
    for name, pts in bands.items():
        u = undistort(pts, k)
        fit = lines.fit_line(u)
        out[name] = lines.sagitta(u, fit.line)
    return out


def plumb_line_k(bands: dict[str, np.ndarray], grid=np.linspace(-0.30, 0.30, 241)) -> float:
    """The k that minimises the pixel-weighted bow of the given lines."""
    weights = {n: len(p) for n, p in bands.items()}

    def cost(k):
        return sum(weights[n] * s for n, s in total_sagitta(bands, k).items())

    return float(grid[int(np.argmin([cost(k) for k in grid]))])
