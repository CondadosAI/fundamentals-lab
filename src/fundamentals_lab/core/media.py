"""Animated figures for unit 3.1: the real application each post opens with.

The unit measures on Middlebury templeRing; these show the same operators on real
footage, so a reader sees what an edge detector is for before the worked example.
Everything is drawn from data already used by the track: the pickleball match clip
(unit 3.5), the CC0 photograph of a card on a bench, and OpenCV's calibration boards.
Each function returns a list of BGR frames and a frame rate; `render_all` writes them
as animated WebP through `alignment.media.encode`.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from loguru import logger

from fundamentals_lab.alignment import transforms
from fundamentals_lab.alignment.media import (
    BG,
    CLIP,
    DOCUMENT,
    GREEN,
    RED,
    WHITE,
    document_corners,
    encode,
    label,
)
from fundamentals_lab.config import OUTPUT_DIR

MEDIA_DIR = OUTPUT_DIR / "figures" / "edges" / "media"
YELLOW = (40, 220, 250)

#: Canny on the clip: the same 5x5, sigma 1.4 smoothing the lesson uses, and a 1:3
#: low:high ratio. Thresholds are for 8-bit Sobel magnitudes at the display width.
CANNY_LOW, CANNY_HIGH = 40, 120


#: The clip opens on a spectator's hand in the corner of the frame; start after it.
CLIP_START = 45


def _clip(width: int, step: int = 3, n: int = 48):
    """Every `step`-th frame of the match clip from CLIP_START, resized to `width` px."""
    cap = cv2.VideoCapture(str(CLIP))
    cap.set(cv2.CAP_PROP_POS_FRAMES, CLIP_START)
    out, i = [], 0
    while len(out) < n:
        ok, f = cap.read()
        if not ok:
            break
        if i % step == 0:
            h = round(f.shape[0] * width / f.shape[1])
            out.append(cv2.resize(f, (width, h), interpolation=cv2.INTER_AREA))
        i += 1
    return out


def _canny(bgr):
    g = cv2.GaussianBlur(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY), (5, 5), 1.4)
    return cv2.Canny(g, CANNY_LOW, CANNY_HIGH)


# --- hub: the rally and its edges ---------------------------------------------------


def rally_edges(width: int = 560):
    """The rally on top, its Canny edges underneath, frame by frame.

    Canny runs at the output width: a one-pixel edge map computed large and scaled
    down turns grey and soft, and costs more bytes than the crisp one.
    """
    frames = []
    for f in _clip(width, n=32):
        e = cv2.cvtColor(_canny(f), cv2.COLOR_GRAY2BGR)
        top = f.copy()
        label(top, "the match", (10, 26), 0.6)
        label(e, "its edges (Canny)", (10, 26), 0.6)
        frames.append(np.vstack([top, np.full((4, width, 3), 40, np.uint8), e]))
    return frames, 8


# --- lesson 1: one row of pixels ----------------------------------------------------


def scanline():
    """A row sweeps down one frame; underneath, the brightness along that row."""
    f = _clip(760, n=1)[0]
    g = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
    h, w = g.shape
    plot_h = 190
    ys = np.linspace(0.42 * h, 0.96 * h, 44).astype(int)
    frames = []
    for y in list(ys) + [ys[-1]] * 8:
        img = f.copy()
        cv2.line(img, (0, int(y)), (w, int(y)), YELLOW, 2, cv2.LINE_AA)
        plot = np.full((plot_h, w, 3), BG, np.uint8)
        prof = g[int(y)].astype(float)
        pts = np.stack([np.arange(w), plot_h - 12 - prof / 255 * (plot_h - 44)], 1)
        cv2.polylines(plot, [np.rint(pts).astype(np.int32)], False, YELLOW, 2, cv2.LINE_AA)
        label(plot, "brightness along the yellow row", (12, 26), 0.6, box=False)
        frames.append(np.vstack([img, plot]))
    return frames, 10


# --- lesson 2: direction and strength -----------------------------------------------


def _wheel(size=86):
    """Legend: hue = gradient direction, the same mapping as the frames."""
    yy, xx = np.mgrid[-1 : 1 : size * 1j, -1 : 1 : size * 1j]
    ang = (np.degrees(np.arctan2(yy, xx)) % 180).astype(np.uint8)
    r = np.hypot(xx, yy)
    hsv = np.zeros((size, size, 3), np.uint8)
    hsv[..., 0], hsv[..., 1] = ang, 255
    hsv[..., 2] = np.where((r <= 1) & (r >= 0.45), 255, 0)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def gradient_direction():
    """Each pixel coloured by gradient direction, brightness by gradient strength."""
    wheel = _wheel()
    frames = []
    for f in _clip(760):
        g = cv2.GaussianBlur(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY), (5, 5), 1.0).astype(np.float32)
        gx = cv2.Sobel(g, cv2.CV_32F, 1, 0)
        gy = cv2.Sobel(g, cv2.CV_32F, 0, 1)
        mag, ang = cv2.cartToPolar(gx, gy, angleInDegrees=True)
        hsv = np.zeros((*g.shape, 3), np.uint8)
        # A direction and its opposite share a colour: an edge's two sides
        # differ only in which way brightness increases.
        hsv[..., 0] = (ang % 180).astype(np.uint8)
        hsv[..., 1] = 255
        hsv[..., 2] = np.clip(mag / 400.0 * 255, 0, 255).astype(np.uint8)
        img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        h, w = img.shape[:2]
        img[h - wheel.shape[0] - 12 : h - 12, w - wheel.shape[1] - 12 : w - 12] = np.maximum(
            img[h - wheel.shape[0] - 12 : h - 12, w - wheel.shape[1] - 12 : w - 12], wheel
        )
        label(img, "colour = direction, brightness = strength", (14, 34), 0.7)
        frames.append(img)
    return frames, 10


# --- lesson 3: the scale knob -------------------------------------------------------


def log_scale_sweep():
    """Zero-crossings of the Laplacian of Gaussian as the blur scale grows and shrinks."""
    f = _clip(760, n=6)[5]
    g = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY).astype(np.float32)
    sigmas = np.geomspace(1.0, 8.0, 20)
    frames = []
    for s in list(sigmas) + [sigmas[-1]] * 4 + list(sigmas[::-1]) + [sigmas[0]] * 4:
        L = cv2.Laplacian(cv2.GaussianBlur(g, (0, 0), s), cv2.CV_32F) * s * s
        a, b, c = L[:-1, :-1], L[1:, :-1], L[:-1, 1:]
        # A sign change with a real slope across it; flat regions cross zero on noise.
        zc = ((np.sign(a) != np.sign(b)) & (np.abs(a - b) > 4)) | (
            (np.sign(a) != np.sign(c)) & (np.abs(a - c) > 4)
        )
        img = np.full((*g.shape, 3), BG, np.uint8)
        img[:-1, :-1][zc] = WHITE
        label(img, f"blur scale sigma = {s:.1f} px", (14, 34), 0.8)
        frames.append(img)
    return frames, 12


# --- lesson 4: Canny in a document scanner --------------------------------------------


def canny_scanner():
    """Photo -> Canny -> the four straight sides -> their corners -> the flattened card."""
    img = cv2.imread(str(DOCUMENT))
    scale = 1200 / img.shape[1]
    img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    h, w = img.shape[:2]
    # The corners come from document_corners (unit 3.5's fit, Canny at 30/90 under the
    # tapped seed). The edge map shown is smoothed harder, as a scanner app tunes it
    # for a whole-page search: the bench's wood grain drops out, the card's sides stay.
    grey = cv2.GaussianBlur(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (0, 0), 2.5)
    edges = cv2.cvtColor(cv2.Canny(grey, 40, 120), cv2.COLOR_GRAY2BGR)
    corners = document_corners(img)
    poly = np.rint(corners).astype(np.int32)

    frames = []
    for _ in range(6):
        f = img.copy()
        label(f, "a card photographed at an angle", (24, 60), 1.4)
        frames.append(f)
    for k in range(6):
        f = cv2.addWeighted(img, 1 - (k + 1) / 6, edges, (k + 1) / 6, 0)
        label(f, "1. Canny: thin, connected edges", (24, 60), 1.4)
        frames.append(f)
    for _ in range(8):
        f = edges.copy()
        label(f, "1. Canny: thin, connected edges", (24, 60), 1.4)
        frames.append(f)
    for _ in range(10):
        f = edges.copy()
        cv2.polylines(f, [poly], True, GREEN, 4, cv2.LINE_AA)
        label(f, "2. the four straight sides", (24, 60), 1.4)
        frames.append(f)
    for _ in range(10):
        f = edges.copy()
        cv2.polylines(f, [poly], True, GREEN, 4, cv2.LINE_AA)
        for p in corners:
            cv2.circle(f, tuple(np.rint(p).astype(int)), 12, RED, -1, cv2.LINE_AA)
        label(f, "3. where the sides meet: the corners", (24, 60), 1.4)
        frames.append(f)

    top, right = np.linalg.norm(corners[1] - corners[0]), np.linalg.norm(corners[2] - corners[1])
    bottom, left = np.linalg.norm(corners[2] - corners[3]), np.linalg.norm(corners[3] - corners[0])
    th = 0.8 * h
    tw = th * (top + bottom) / (left + right)
    cx, cy = w / 2, h / 2
    dst = np.array(
        [
            [cx - tw / 2, cy - th / 2],
            [cx + tw / 2, cy - th / 2],
            [cx + tw / 2, cy + th / 2],
            [cx - tw / 2, cy + th / 2],
        ]
    )
    ease = lambda t: 0.5 - 0.5 * np.cos(np.pi * t)  # noqa: E731
    for k in range(18):
        cur = corners * (1 - ease(k / 17)) + dst * ease(k / 17)
        f = cv2.warpPerspective(img, transforms.dlt(corners, cur), (w, h), borderValue=BG)
        label(f, "4. the card, flattened", (24, 60), 1.4)
        frames.append(f)
    frames += [frames[-1]] * 10
    return frames, 10


# --- lesson 5: corners on calibration boards ------------------------------------------


def corner_boards(n_boards: int = 6):
    """Per board: the corner response, the Shi-Tomasi corners, the ordered grid."""
    from fundamentals_lab.formation.dataset import fetch_opencv_calibration

    frames = []
    for p in fetch_opencv_calibration()[:n_boards]:
        g = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        base = cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)

        R = cv2.cornerMinEigenVal(g, blockSize=5, ksize=3)
        heat = cv2.applyColorMap(
            np.clip(R / np.percentile(R, 99.7) * 255, 0, 255).astype(np.uint8), cv2.COLORMAP_INFERNO
        )
        f = cv2.addWeighted(base, 0.35, heat, 0.9, 0)
        label(f, "1. corner response (Shi-Tomasi)", (10, 28), 0.6)
        frames.append(f)

        pts = cv2.goodFeaturesToTrack(g, maxCorners=120, qualityLevel=0.05, minDistance=10)
        f = base.copy()
        for q in pts.reshape(-1, 2):
            cv2.circle(f, tuple(np.rint(q).astype(int)), 4, RED, -1, cv2.LINE_AA)
        label(f, f"2. its {len(pts)} strongest peaks", (10, 28), 0.6)
        frames.append(f)

        ok, cs = cv2.findChessboardCorners(g, (9, 6))
        f = base.copy()
        if ok:
            cs = cv2.cornerSubPix(
                g,
                cs,
                (11, 11),
                (-1, -1),
                (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.01),
            )
            cv2.drawChessboardCorners(f, (9, 6), cs, ok)
        label(f, "3. the 54 board corners, in order", (10, 28), 0.6)
        frames.append(f)
    return frames, 1.5


# --- all --------------------------------------------------------------------------------


#: name -> (function, output width px, libwebp quality). Edge maps are all fine detail
#: and compress poorly, so they get a narrower frame and a lower quality than video.
TABLE = {
    "rally-edges": (rally_edges, 560, 35),
    "scanline": (scanline, 600, 45),
    "gradient-direction": (gradient_direction, 600, 45),
    "log-scale-sweep": (log_scale_sweep, 600, 40),
    "canny-scanner": (canny_scanner, 560, 35),
    "corner-boards": (corner_boards, 560, 50),
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
