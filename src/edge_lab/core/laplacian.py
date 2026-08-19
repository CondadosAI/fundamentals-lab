"""Second-derivative edges: the Laplacian, LoG, and zero-crossings.

Where a gradient peaks, the second derivative crosses zero. That trades one
problem for another: a crossing has a location but no strength, so a faint edge
and a strong one produce identical output, and the crossings always close into
loops — including loops around noise.
"""

import cv2
import numpy as np


def log_response(gray01: np.ndarray, sigma: float) -> np.ndarray:
    """Laplacian of Gaussian, as Marr-Hildreth actually compute it: smooth first.

    Applying the Laplacian directly to an unsmoothed image is the failure this
    exists to fix — a second derivative doubles the noise exponent, so a single
    stray pixel produces a crossing pair on its own.
    """
    blurred = cv2.GaussianBlur(gray01, (0, 0), sigmaX=sigma, sigmaY=sigma)
    return cv2.Laplacian(blurred, cv2.CV_32F, ksize=3)


def crossing_floor(resp: np.ndarray, rel_threshold: float = 0.25) -> float:
    """The steepness a sign change must have to count, in units of resp's std.

    Kept separate from `zero_crossings` because a threshold derived from the
    response being tested cannot compare two images: a noisy response has a
    larger standard deviation, so an adaptive floor silently raises the bar on
    exactly the image that needs more crossings rejected, and the comparison
    stops measuring noise and starts measuring the threshold.
    """
    return rel_threshold * float(resp.std())


def zero_crossings(resp: np.ndarray, floor: float | None = None) -> np.ndarray:
    """Binary map of sign changes in `resp`, ignoring crossings that are flat.

    A sign change alone is not enough: in a smooth region the response wanders
    across zero with no edge underneath. Pass an explicit `floor` when comparing
    two images; the default derives one from `resp` itself, which is only valid
    for a single image in isolation.
    """
    if floor is None:
        floor = crossing_floor(resp)
    cross = np.zeros(resp.shape, dtype=bool)
    # Horizontal neighbours, then vertical. A crossing is marked on the left /
    # upper pixel of the pair, which keeps the result one pixel wide.
    for axis in (1, 0):
        a = resp
        b = np.roll(resp, -1, axis=axis)
        sign_change = (np.sign(a) * np.sign(b)) < 0
        steep = np.abs(a - b) >= floor
        hit = sign_change & steep
        # The rolled-in edge wraps around; it is not a real neighbour pair.
        if axis == 1:
            hit[:, -1] = False
        else:
            hit[-1, :] = False
        cross |= hit
    return cross


_NEIGHBOUR_KERNEL = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=np.float32)


def contour_stats(mask: np.ndarray) -> dict:
    """Count the contours in a binary edge map and how many of them close.

    A contour is a connected component (8-connectivity). It is *open* if any of
    its pixels has exactly one neighbour — that is a loose end. A component with
    no loose ends is a closed loop, which is the property zero-crossings have
    and thresholded gradients do not.
    """
    u8 = mask.astype(np.uint8)
    n_labels, labels = cv2.connectedComponents(u8, connectivity=8)
    # filter2D has no uint8 -> int path, and the count must not wrap, so the
    # neighbour tally is done in float and rounded back.
    counts = cv2.filter2D(
        u8.astype(np.float32), cv2.CV_32F, _NEIGHBOUR_KERNEL, borderType=cv2.BORDER_CONSTANT
    )
    neighbours = np.rint(counts).astype(np.int32)
    endpoints = (u8 == 1) & (neighbours == 1)

    n_components = n_labels - 1  # label 0 is background
    if n_components == 0:
        return {"n_components": 0, "n_closed": 0, "closed_fraction": 0.0}
    with_endpoint = np.unique(labels[endpoints])
    n_open = int(np.count_nonzero(with_endpoint > 0))
    n_closed = n_components - n_open
    return {
        "n_components": int(n_components),
        "n_closed": int(n_closed),
        "closed_fraction": round(n_closed / n_components, 4),
    }
