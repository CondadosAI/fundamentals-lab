"""Canny's two contributions, kept separate so each can be measured alone.

Non-maximum suppression thins a ridge of gradient magnitude to a single pixel.
Hysteresis decides membership: a weak pixel is kept only if it is connected to a
strong one. Neither depends on the other, and the experiments switch each off
independently.
"""

import cv2
import numpy as np

from edge_lab.core.gradients import gradients, magnitude, orientation_deg


def non_max_suppression(mag: np.ndarray, ori: np.ndarray) -> np.ndarray:
    """Zero every pixel that is not a local maximum along the gradient direction.

    Orientation is quantised to the four directions an 8-neighbourhood can
    actually represent. The comparison uses the two neighbours *across* the
    edge, which is why the folded 0..180 orientation is the right input.
    """
    padded = np.pad(mag, 1, mode="constant", constant_values=0.0)
    # Neighbour pairs, indexed by the quantised direction.
    #   0   -> horizontal gradient, compare left / right
    #   45  -> compare upper-right / lower-left
    #   90  -> vertical gradient, compare up / down
    #   135 -> compare upper-left / lower-right
    h, w = mag.shape
    core = (slice(1, h + 1), slice(1, w + 1))

    def shifted(dy: int, dx: int) -> np.ndarray:
        return padded[1 + dy : 1 + dy + h, 1 + dx : 1 + dx + w]

    pairs = {
        0: (shifted(0, -1), shifted(0, 1)),
        45: (shifted(-1, 1), shifted(1, -1)),
        90: (shifted(-1, 0), shifted(1, 0)),
        135: (shifted(-1, -1), shifted(1, 1)),
    }
    bucket = np.zeros(mag.shape, dtype=np.int32)
    bucket[(ori >= 22.5) & (ori < 67.5)] = 45
    bucket[(ori >= 67.5) & (ori < 112.5)] = 90
    bucket[(ori >= 112.5) & (ori < 157.5)] = 135
    # Everything else (0..22.5 and 157.5..180) is the horizontal bucket, 0.

    out = np.zeros_like(mag)
    centre = padded[core]
    for direction, (a, b) in pairs.items():
        sel = bucket == direction
        keep = sel & (centre >= a) & (centre >= b)
        out[keep] = mag[keep]
    return out


def hysteresis(thin: np.ndarray, low: float, high: float) -> np.ndarray:
    """Keep weak edges only where they connect to a strong one.

    Implemented as a connected-component query rather than a flood fill: label
    everything above `low`, then keep whole labels that contain at least one
    pixel above `high`. Same answer, and it makes the "one long chain vs many
    fragments" property the experiments measure directly visible.
    """
    if high < low:
        raise ValueError("high must be >= low")
    weak = (thin >= low).astype(np.uint8)
    n_labels, labels = cv2.connectedComponents(weak, connectivity=8)
    if n_labels <= 1:
        return np.zeros(thin.shape, dtype=bool)
    strong_labels = np.unique(labels[thin >= high])
    strong_labels = strong_labels[strong_labels > 0]
    return np.isin(labels, strong_labels)


def canny_from_scratch(
    gray01: np.ndarray, sigma: float, low: float, high: float, operator: str = "sobel"
) -> dict:
    """The full pipeline, returning every intermediate stage.

    Thresholds are quoted on the 0..255 scale a reader sees in `cv2.Canny`
    documentation; the magnitude here lives on the 0..1 input scale, so they are
    converted once, at the boundary.
    """
    blurred = cv2.GaussianBlur(gray01, (0, 0), sigmaX=sigma, sigmaY=sigma)
    gx, gy = gradients(blurred, operator)
    mag = magnitude(gx, gy)
    ori = orientation_deg(gx, gy)
    thin = non_max_suppression(mag, ori)
    edges = hysteresis(thin, low / 255.0, high / 255.0)
    return {"magnitude": mag, "orientation": ori, "thin": thin, "edges": edges}


def opencv_canny(gray01: np.ndarray, sigma: float, low: float, high: float) -> np.ndarray:
    """OpenCV's own Canny, set up to answer the same question as ours.

    Two settings have to be matched or the comparison is meaningless.
    `cv2.Canny` does **no** smoothing of its own, so the caller must blur first;
    and it defaults to the L1 gradient norm |gx| + |gy|, which is larger than the
    L2 norm used here and therefore lets far more pixels past the same threshold.
    Left at their defaults, those two differences alone account for a 4x gap in
    edge-pixel count.
    """
    u8 = np.clip(gray01 * 255.0, 0, 255).astype(np.uint8)
    blurred = cv2.GaussianBlur(u8, (0, 0), sigmaX=sigma, sigmaY=sigma)
    return cv2.Canny(blurred, low, high, L2gradient=True) > 0
