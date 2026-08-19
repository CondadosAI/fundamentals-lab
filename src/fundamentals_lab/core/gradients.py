"""First-derivative operators: central differences, Prewitt, Sobel.

All three are separable and all three answer the same question — how fast does
brightness change here — but they weight the neighbouring rows differently, and
that weighting is the whole difference in how they behave on a noisy image.

`cv2.filter2D` computes correlation, not convolution. For these kernels that is
the intended behaviour: it makes a bright-to-the-right step produce a positive
x-response, which is the sign convention `cv2.Sobel` uses.
"""

import cv2
import numpy as np

# Central difference: (I[x+1] - I[x-1]) / 2. No smoothing at all.
_CENTRAL = np.array([[-0.5, 0.0, 0.5]], dtype=np.float32)
# Prewitt: a box average across the three rows.
_PREWITT = np.array([[-1, 0, 1]], dtype=np.float32) * np.array([[1], [1], [1]], np.float32)
# Sobel: a [1 2 1] triangular average across the rows. cv2.Sobel is used
# directly so the values match the rest of the CV world exactly.

OPERATORS = ("central", "prewitt", "sobel")


def gradients(gray01: np.ndarray, operator: str = "sobel") -> tuple:
    """(gx, gy) for one of the three operators, as float32."""
    if operator == "central":
        gx = cv2.filter2D(gray01, cv2.CV_32F, _CENTRAL)
        gy = cv2.filter2D(gray01, cv2.CV_32F, _CENTRAL.T)
    elif operator == "prewitt":
        gx = cv2.filter2D(gray01, cv2.CV_32F, _PREWITT)
        gy = cv2.filter2D(gray01, cv2.CV_32F, _PREWITT.T)
    elif operator == "sobel":
        gx = cv2.Sobel(gray01, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray01, cv2.CV_32F, 0, 1, ksize=3)
    else:
        raise ValueError(f"unknown operator {operator!r}; expected one of {OPERATORS}")
    return gx, gy


def magnitude(gx: np.ndarray, gy: np.ndarray) -> np.ndarray:
    return np.hypot(gx, gy).astype(np.float32)


def orientation_deg(gx: np.ndarray, gy: np.ndarray) -> np.ndarray:
    """Edge-normal direction in degrees, folded to 0..180.

    Folded because an edge and the same edge with light and dark swapped are the
    same edge; only the axis matters, not which side is bright.
    """
    return np.rad2deg(np.arctan2(gy, gx)) % 180.0


def top_fraction_mask(mag: np.ndarray, fraction: float) -> np.ndarray:
    """Keep the strongest `fraction` of pixels.

    Two operators cannot be compared at a shared numeric threshold — Sobel's
    kernel sums to 8 where the central difference sums to 1, so the same number
    means different things. Fixing the *count* of edge pixels removes that.
    """
    if not 0.0 < fraction < 1.0:
        raise ValueError("fraction must be in (0, 1)")
    cutoff = np.quantile(mag, 1.0 - fraction)
    return mag >= cutoff
