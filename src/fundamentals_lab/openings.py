"""Opening animations for the single-post units: 2.1 (convolution), 3.3 (SIFT) and
4.1 (camera models, calibration and PnP).

Drawn from data the track already uses: a pickleball match frame (CC BY 3.0), the
Barcelona harbour pair (unit 3.5's panorama) and OpenCV's calibration boards (Apache-2.0).
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from loguru import logger

from fundamentals_lab.alignment.media import BG, CLIP, GREEN, RED, WHITE, encode, label
from fundamentals_lab.config import DATA_DIR, OUTPUT_DIR
from fundamentals_lab.formation.media import _boards

MEDIA_DIR = OUTPUT_DIR / "figures" / "openings" / "media"
AMBER = (80, 170, 250)
FONT = cv2.FONT_HERSHEY_SIMPLEX
PANORAMA = DATA_DIR / "wild-panorama"


def _match_frame(width: int) -> np.ndarray:
    cap = cv2.VideoCapture(str(CLIP))
    cap.set(cv2.CAP_PROP_POS_FRAMES, 60)
    _, f = cap.read()
    return cv2.resize(
        f, (width, round(f.shape[0] * width / f.shape[1])), interpolation=cv2.INTER_AREA
    )


# --- 2.1: one operation, different grids ------------------------------------------------

KERNELS = [
    ("original", np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]], np.float32), 0),
    ("blur", np.full((3, 3), 1 / 9, np.float32), 0),
    ("sharpen", np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], np.float32), 0),
    ("edges (Laplacian)", np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], np.float32), 128),
    ("vertical edges (Sobel x)", np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], np.float32), 128),
    ("emboss", np.array([[-2, -1, 0], [-1, 1, 1], [0, 1, 2]], np.float32), 0),
]


def _kernel_inset(k: np.ndarray, cell: int = 40) -> np.ndarray:
    img = np.full((3 * cell + 8, 3 * cell + 8, 3), 25, np.uint8)
    for r in range(3):
        for c in range(3):
            x, y = 4 + c * cell, 4 + r * cell
            cv2.rectangle(img, (x, y), (x + cell - 2, y + cell - 2), (90, 90, 90), 1)
            v = k[r, c]
            text = f"{v:.2f}".rstrip("0").rstrip(".") if abs(v - round(v)) > 1e-6 else str(int(v))
            (tw, _), _ = cv2.getTextSize(text, FONT, 0.45, 1)
            cv2.putText(
                img,
                text,
                (x + (cell - tw) // 2, y + cell // 2 + 5),
                FONT,
                0.45,
                WHITE,
                1,
                cv2.LINE_AA,
            )
    return img


def kernel_gallery(width: int = 600):
    """The same frame convolved with six 3x3 kernels, the kernel printed in the corner."""
    f = _match_frame(width)
    frames = []
    for name, k, offset in KERNELS:
        g = (
            f
            if name != "edges (Laplacian)" and "Sobel" not in name
            else cv2.cvtColor(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
        )
        out = cv2.filter2D(g.astype(np.float32), -1, k) + offset
        if name == "blur":
            for _ in range(3):  # a 3x3 box is subtle at this size; run it a few times
                out = cv2.filter2D(out, -1, k)
        img = np.clip(out, 0, 255).astype(np.uint8)
        inset = _kernel_inset(k)
        h, w = img.shape[:2]
        img[h - inset.shape[0] - 10 : h - 10, w - inset.shape[1] - 10 : w - 10] = inset
        label(img, name, (12, 32), 0.7)
        frames += [img] * 9
    return frames, 5


# --- 3.3: points that can be found again -------------------------------------------------


def sift_matches(height: int = 300):
    """SIFT keypoints on two harbour photos, then the matches that survive the ratio test."""
    a = cv2.imread(str(PANORAMA / "BarcelonaHarbour1.jpg"))
    b = cv2.imread(str(PANORAMA / "BarcelonaHarbour2.jpg"))
    s = height / a.shape[0]
    a = cv2.resize(a, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    b = cv2.resize(
        b, (round(b.shape[1] * height / b.shape[0]), height), interpolation=cv2.INTER_AREA
    )
    sift = cv2.SIFT_create()
    ka, da = sift.detectAndCompute(cv2.cvtColor(a, cv2.COLOR_BGR2GRAY), None)
    kb, db = sift.detectAndCompute(cv2.cvtColor(b, cv2.COLOR_BGR2GRAY), None)
    pairs = cv2.BFMatcher().knnMatch(da, db, k=2)
    good = [m for m, n in pairs if m.distance < 0.75 * n.distance]
    n_good = len(good)
    bad = [m for m, n in pairs if m.distance >= 0.75 * n.distance]
    rng = np.random.default_rng(0)
    good = list(rng.permutation(good))[:120]
    bad = list(rng.permutation(bad))[:120]
    gap = 8
    base = np.hstack([a, np.full((height, gap, 3), 40, np.uint8), b])
    off = a.shape[1] + gap

    def draw(img, ms, colour):
        for m in ms:
            p = tuple(np.rint(ka[m.queryIdx].pt).astype(int))
            q = tuple(np.rint(kb[m.trainIdx].pt).astype(int) + np.array([off, 0]))
            cv2.line(img, p, q, colour, 1, cv2.LINE_AA)

    frames = []
    dots = base.copy()
    for k in ka:
        cv2.circle(dots, tuple(np.rint(k.pt).astype(int)), 1, AMBER, -1)
    for k in kb:
        cv2.circle(dots, (int(k.pt[0]) + off, int(k.pt[1])), 1, AMBER, -1)
    step = dots.copy()
    label(step, f"1. {len(ka)} and {len(kb)} SIFT points found", (10, 28), 0.6)
    frames += [step] * 16
    shown = good + bad  # the same number of each, so the picture is not rigged either way
    order = rng.permutation(len(shown))
    for n in range(0, len(shown) + 1, 20):
        img = dots.copy()
        draw(img, [shown[i] for i in order[:n]], AMBER)
        label(img, "2. each point paired with its nearest descriptor", (10, 28), 0.6)
        frames.append(img)
    frames += [frames[-1].copy()] * 8
    judged = dots.copy()
    draw(judged, bad, RED)
    draw(judged, good, GREEN)
    label(judged, "3. the ratio test: green kept, red rejected", (10, 28), 0.6)
    frames += [judged] * 14
    kept = dots.copy()
    draw(kept, good, GREEN)
    label(kept, "4. the survivors: nearly all agree on one shift", (10, 28), 0.6)
    frames += [kept] * 16
    logger.info(f"sift-matches: {len(ka)}/{len(kb)} keypoints, {n_good} ratio-test matches")
    return frames, 8


# --- 4.1: where the camera was ------------------------------------------------------------


def pnp_pose(plot: int = 360):
    """Per board: its axes from PnP, and the camera's recovered position seen from above."""
    greys, _, K, dist, rvecs, tvecs = _boards()
    centres = []
    for r, t in zip(rvecs, tvecs, strict=True):
        R, _ = cv2.Rodrigues(r)
        centres.append((-R.T @ t).ravel())  # camera centre in board coordinates, in squares
    centres = np.array(centres)
    xs, zs = centres[:, 0], centres[:, 2]
    lo = np.array([min(xs.min(), -2), min(zs.min(), -2)])
    hi = np.array([max(xs.max(), 10), max(zs.max(), 2)])
    scale = (plot - 60) / max(hi - lo)

    def to_plot(x, z):
        return int(30 + (x - lo[0]) * scale), int(plot - 30 - (z - lo[1]) * scale)

    frames = []
    axes = np.float32([[0, 0, 0], [3, 0, 0], [0, 3, 0], [0, 0, -3]])
    for i, (g, r, t) in enumerate(zip(greys, rvecs, tvecs, strict=True)):
        img = cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)
        p, _ = cv2.projectPoints(axes, r, t, K, dist)
        p = np.rint(p.reshape(-1, 2)).astype(int)
        for q, colour in zip(p[1:], (RED, GREEN, (235, 140, 40)), strict=True):
            cv2.arrowedLine(img, tuple(p[0]), tuple(q), colour, 3, cv2.LINE_AA, tipLength=0.15)
        label(img, "the board's axes, from PnP", (10, 28), 0.6)
        img = cv2.resize(img, (round(img.shape[1] * plot / img.shape[0]), plot))
        top = np.full((plot, plot, 3), BG, np.uint8)
        a, b = to_plot(0, 0), to_plot(8, 0)
        cv2.line(top, a, b, WHITE, 3)
        for j in range(i + 1):
            cv2.circle(
                top,
                to_plot(xs[j], zs[j]),
                6 if j == i else 4,
                AMBER if j == i else (140, 140, 140),
                -1,
                cv2.LINE_AA,
            )
        cv2.putText(top, "board", (a[0], a[1] + 20), FONT, 0.45, WHITE, 1, cv2.LINE_AA)
        label(top, "camera positions, from above", (10, 28), 0.55)
        frames += [np.hstack([img, np.full((plot, 6, 3), 40, np.uint8), top])] * 3
    return frames, 3


TABLE = {
    "kernel-gallery": (kernel_gallery, 560, 45),
    "sift-matches": (sift_matches, 640, 45),
    "pnp-pose": (pnp_pose, 700, 55),
}


def render_all(only: tuple[str, ...] = ()) -> dict[str, Path]:
    out = {}
    for name, (fn, width, quality) in TABLE.items():
        if only and name not in only:
            continue
        frames, fps = fn()
        out[name] = encode(frames, fps, MEDIA_DIR / f"{name}.webp", width, quality)
        logger.info(f"{name}: {len(frames)} frames, {out[name].stat().st_size / 1e6:.2f} MB")
    return out
