"""The structure tensor, and the three responses built on it.

Harris, Shi-Tomasi and Forstner all read the same 2x2 matrix of averaged
gradient outer products; they disagree only in how they turn its two
eigenvalues into one number. Computing all three from a shared tensor is what
makes that visible.
"""

import cv2
import numpy as np

from edge_lab.config import PATCH_HALF, PATCH_PRESETS


def eigenvalues_at(gray01: np.ndarray, x: int, y: int, half: int = PATCH_HALF) -> tuple:
    """(lambda1, lambda2), lambda1 >= lambda2, for the window centred on (x, y).

    The 1-pixel border of the window is dropped: Sobel's response there is
    computed against replicated pixels rather than image content, so including
    it would let the window's own edge contribute gradient energy.

    This must stay byte-identical to the recipe that produced the published
    numbers in the SIFT article — the parity check in the corners experiment is
    the only thing keeping the two articles' numbers in agreement.
    """
    patch = gray01[y - half : y + half + 1, x - half : x + half + 1]
    gx = cv2.Sobel(patch, cv2.CV_32F, 1, 0)[1:-1, 1:-1]
    gy = cv2.Sobel(patch, cv2.CV_32F, 0, 1)[1:-1, 1:-1]
    m = (
        np.array(
            [
                [np.sum(gx * gx), np.sum(gx * gy)],
                [np.sum(gx * gy), np.sum(gy * gy)],
            ]
        )
        / gx.size
    )
    lo, hi = np.linalg.eigvalsh(m)
    return float(hi), float(lo)


def preset_eigenvalues(gray01: np.ndarray) -> dict:
    """The three named spots: flat background, an edge, a corner."""
    return {
        name: {"xy": [x, y], "l1": round(l1, 3), "l2": round(l2, 3)}
        for name, (x, y) in PATCH_PRESETS.items()
        for l1, l2 in [eigenvalues_at(gray01, x, y)]
    }


def structure_tensor(gray01: np.ndarray, sigma: float = 1.5) -> tuple:
    """Per-pixel (Sxx, Sxy, Syy), Gaussian-weighted over the neighbourhood.

    A Gaussian window rather than a box: it makes the response rotation-covariant
    and stops a feature from flickering as a square window slides over it.
    """
    gx = cv2.Sobel(gray01, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray01, cv2.CV_32F, 0, 1, ksize=3)
    blur = lambda a: cv2.GaussianBlur(a, (0, 0), sigmaX=sigma, sigmaY=sigma)  # noqa: E731
    return blur(gx * gx), blur(gx * gy), blur(gy * gy)


def harris_response(gray01: np.ndarray, k: float = 0.04, sigma: float = 1.5) -> np.ndarray:
    """det(M) - k * trace(M)^2 — no eigendecomposition needed, which is the point."""
    sxx, sxy, syy = structure_tensor(gray01, sigma)
    det = sxx * syy - sxy * sxy
    trace = sxx + syy
    return det - k * trace * trace


def shi_tomasi_response(gray01: np.ndarray, sigma: float = 1.5) -> np.ndarray:
    """min(lambda1, lambda2), in closed form for a 2x2 symmetric matrix."""
    sxx, sxy, syy = structure_tensor(gray01, sigma)
    half_trace = 0.5 * (sxx + syy)
    root = np.sqrt(np.maximum(0.25 * (sxx - syy) ** 2 + sxy * sxy, 0.0))
    return half_trace - root


def forstner_response(gray01: np.ndarray, sigma: float = 1.5, eps: float = 1e-12) -> tuple:
    """(w, q): Forstner's precision weight and roundness.

    w = det(M) / trace(M) is how precisely the point can be located; q =
    4 det(M) / trace(M)^2 is how circular the error ellipse is, 1 for a perfect
    corner and 0 for an edge. Splitting strength from shape is what separates
    this from Harris, which folds both into one number.
    """
    sxx, sxy, syy = structure_tensor(gray01, sigma)
    det = sxx * syy - sxy * sxy
    trace = sxx + syy
    w = det / (trace + eps)
    q = 4.0 * det / (trace * trace + eps)
    return w, q
