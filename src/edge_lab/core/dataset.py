"""The canonical frame every lesson in the unit works on.

The image is Middlebury templeRing view 0, put through the same preprocessing
the SIFT article's interactive lab uses: upright rotation, a CLAHE lift on the
L channel, and a resize to 360x480. Reproducing that pipeline exactly is what
lets the corner eigenvalues computed here match the ones already published,
which is the parity check the corners experiment runs.
"""

import cv2
import numpy as np

from edge_lab.config import (
    CANONICAL_SIZE,
    CLAHE_CLIP,
    CLAHE_TILE,
    DATA_DIR,
    DATASET,
    VIEW,
)


def frame_paths() -> list:
    """Every PNG of the dataset, in view order."""
    return sorted((DATA_DIR / DATASET).glob("*.png"))


def load_view(index: int = VIEW) -> np.ndarray:
    """One raw BGR view. Raises if the dataset was never downloaded."""
    paths = frame_paths()
    if not paths:
        raise FileNotFoundError(
            f"no images under {DATA_DIR / DATASET} — run `uv run edge-download` first"
        )
    img = cv2.imread(str(paths[index]), cv2.IMREAD_COLOR)
    if img is None:
        raise OSError(f"could not decode {paths[index]}")
    return img


def canonical_bgr(index: int = VIEW) -> np.ndarray:
    """The preprocessed frame, BGR, 360x480."""
    up = cv2.rotate(load_view(index), cv2.ROTATE_90_COUNTERCLOCKWISE)
    lab = cv2.cvtColor(up, cv2.COLOR_BGR2LAB)
    lc, ac, bc = cv2.split(lab)
    lc = cv2.createCLAHE(clipLimit=CLAHE_CLIP, tileGridSize=CLAHE_TILE).apply(lc)
    up = cv2.cvtColor(cv2.merge([lc, ac, bc]), cv2.COLOR_LAB2BGR)
    return cv2.resize(up, CANONICAL_SIZE, interpolation=cv2.INTER_AREA)


def canonical_gray(index: int = VIEW) -> np.ndarray:
    """The preprocessed frame as float32 intensities in 0..1."""
    return cv2.cvtColor(canonical_bgr(index), cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0


def add_noise(gray01: np.ndarray, sigma255: float, rng: np.random.Generator) -> np.ndarray:
    """Gaussian noise specified in 0..255 intensity units, applied in 0..1 space.

    Sigma is quoted on the 8-bit scale because that is the scale a reader thinks
    in when they say "noisy image"; the conversion happens here so no caller has
    to remember it.
    """
    if sigma255 <= 0:
        return gray01.copy()
    noisy = gray01 + rng.normal(0.0, sigma255 / 255.0, gray01.shape).astype(np.float32)
    return np.clip(noisy, 0.0, 1.0)
