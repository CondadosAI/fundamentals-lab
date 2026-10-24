"""The unit's photographs, the warps it applies to them, and the bookkeeping every lesson shares."""

from __future__ import annotations

from functools import cache

import cv2
import numpy as np

from fundamentals_lab.alignment.wild import PANO_DIR
from fundamentals_lab.config import (
    DATA_DIR,
    HARRIS,
    REFERENCE_ORB_FEATURES,
    REFERENCE_ORB_RATIO,
    REFERENCE_RANSAC_PX,
    SIFT_PAIR,
    SIFT_SEED,
    SIFT_WIDTH,
)

TEMPLE_DIR = DATA_DIR / "templeRing"


def _resize(img: np.ndarray, width: int = SIFT_WIDTH) -> np.ndarray:
    s = width / img.shape[1]
    return cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)


@cache
def pair_bgr() -> tuple[np.ndarray, np.ndarray]:
    return tuple(_resize(cv2.imread(str(PANO_DIR / n))) for n in SIFT_PAIR)


@cache
def pair() -> tuple[np.ndarray, np.ndarray]:
    a, b = pair_bgr()
    return cv2.cvtColor(a, cv2.COLOR_BGR2GRAY), cv2.cvtColor(b, cv2.COLOR_BGR2GRAY)


@cache
def temple(view: int) -> np.ndarray:
    return cv2.imread(str(TEMPLE_DIR / f"templeR{view:04d}.png"), cv2.IMREAD_GRAYSCALE)


def apply(H: np.ndarray, pts) -> np.ndarray:
    p = np.asarray(pts, float).reshape(-1, 2)
    q = np.c_[p, np.ones(len(p))] @ np.asarray(H, float).T
    return q[:, :2] / q[:, 2:]


def rotate_scale(img: np.ndarray, deg: float = 0.0, scale: float = 1.0):
    """Rotate about the centre and scale, on a canvas that keeps every pixel. Returns the
    warped image and the 3x3 map from the original's pixels to the warped one's."""
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), deg, scale)
    c, s = abs(M[0, 0]), abs(M[0, 1])
    nw, nh = int(round(h * s + w * c)), int(round(h * c + w * s))
    M[0, 2] += nw / 2 - w / 2
    M[1, 2] += nh / 2 - h / 2
    interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
    return cv2.warpAffine(img, M, (nw, nh), flags=interp), np.vstack([M, [0, 0, 1]])


def photometric(img: np.ndarray, *, gain=1.0, blur=0.0, noise=0.0, gamma=1.0, seed=SIFT_SEED):
    out = img.astype(np.float64)
    if blur:
        out = cv2.GaussianBlur(out, (0, 0), blur)
    if gamma != 1.0:
        out = 255 * (out / 255) ** gamma
    out = out * gain
    if noise:
        out = out + np.random.default_rng(seed).normal(0, noise, out.shape)
    return np.clip(np.round(out), 0, 255).astype(np.uint8)


def dedupe(kps) -> list[cv2.KeyPoint]:
    """One keypoint per (position, size): OpenCV returns a second orientation as a copy."""
    seen, out = set(), []
    for k in sorted(kps, key=lambda k: -k.response):
        key = (round(k.pt[0], 2), round(k.pt[1], 2), round(k.size, 2))
        if key not in seen:
            seen.add(key)
            out.append(k)
    return out


def sift() -> cv2.SIFT:
    return cv2.SIFT_create()


def sift_points(img: np.ndarray, n: int | None = None) -> list[cv2.KeyPoint]:
    kps = dedupe(sift().detect(img, None))
    return kps[:n] if n else kps


def harris_points(img: np.ndarray, n: int, size: float = 8.0) -> list[cv2.KeyPoint]:
    c = cv2.goodFeaturesToTrack(
        img,
        n,
        HARRIS["qualityLevel"],
        HARRIS["minDistance"],
        useHarrisDetector=True,
        k=HARRIS["k"],
        blockSize=HARRIS["blockSize"],
    )
    return [cv2.KeyPoint(float(x), float(y), size) for x, y in c.reshape(-1, 2)]


def knn(da: np.ndarray, db: np.ndarray):
    return cv2.BFMatcher(cv2.NORM_L2).knnMatch(da, db, k=2)


@cache
def reference_homography() -> tuple[np.ndarray, dict]:
    """A -> B from ORB + MAGSAC: the independent answer key for the real pair."""
    a, b = pair()
    cv2.setRNGSeed(SIFT_SEED)
    orb = cv2.ORB_create(nfeatures=REFERENCE_ORB_FEATURES)
    ka, da = orb.detectAndCompute(a, None)
    kb, db = orb.detectAndCompute(b, None)
    mm = [
        m
        for m, n in cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(da, db, k=2)
        if m.distance < REFERENCE_ORB_RATIO * n.distance
    ]
    pa = np.float32([ka[m.queryIdx].pt for m in mm])
    pb = np.float32([kb[m.trainIdx].pt for m in mm])
    H, mask = cv2.findHomography(pa, pb, cv2.USAC_MAGSAC, REFERENCE_RANSAC_PX)
    inl = mask.ravel().astype(bool)
    err = np.linalg.norm(apply(H, pa[inl]) - pb[inl], axis=1)
    info = {
        "orb_keypoints": [len(ka), len(kb)],
        "orb_ratio_matches": len(mm),
        "orb_inliers": int(inl.sum()),
        "orb_inlier_residual_px_median": round(float(np.median(err)), 3),
    }
    return H, info


def inside(pts: np.ndarray, shape, border: float = 0.0) -> np.ndarray:
    h, w = shape[:2]
    p = np.asarray(pts).reshape(-1, 2)
    return (
        (p[:, 0] >= border) & (p[:, 0] < w - border) & (p[:, 1] >= border) & (p[:, 1] < h - border)
    )
