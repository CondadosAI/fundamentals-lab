"""Animated openings for unit 3.2: what boundary detection is for, on real footage.

The unit measures on the VisA pcb1 board; the opening shows the same operators on the
pickleball match clip the track already uses (pickleball4you, CC BY 3.0): the court's
painted lines come back as straight segments, frame after frame, while the players move.
"""

from __future__ import annotations

import cv2
import numpy as np

from fundamentals_lab.alignment import lines, plate
from fundamentals_lab.alignment.media import BG, CLIP, GREEN, encode, label
from fundamentals_lab.config import COURT_LENGTH_M, COURT_WIDTH_M, OUTPUT_DIR

MEDIA_DIR = OUTPUT_DIR / "figures" / "boundaries" / "media"
CLIP_START = 45  # the clip opens on a spectator's hand; start after it
#: Segments on the clip at the display width. Canny as in unit 3.1's openings.
CANNY = (40, 120)
PPH = {"threshold": 60, "min_length": 60, "max_gap": 8}


def _clip(width: int, step: int = 3, n: int = 40):
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


def court_mask(shape, margin_m: float = 0.6) -> np.ndarray:
    """The court plus a margin, from unit 3.5's fitted homography, at this frame's size.

    A line detector for court graphics looks only where the court is; the stands and
    the roof are full of straight things that are not court lines.
    """
    H = lines.fit_court(lines.paint_pixels(plate.load_plate())).H_court2img
    m = margin_m
    # court frame: x along the length, y across the width (alignment/court.py)
    L, W = COURT_LENGTH_M, COURT_WIDTH_M
    quad = np.array([[-m, -m], [L + m, -m], [L + m, W + m], [-m, W + m]], np.float64)
    pts = cv2.perspectiveTransform(quad[None], H)[0]
    pts *= shape[1] / 1920.0
    mask = np.zeros(shape[:2], np.uint8)
    cv2.fillPoly(mask, [np.rint(pts).astype(np.int32)], 255)
    return mask


def segments(bgr: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    g = cv2.GaussianBlur(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY), (5, 5), 1.4)
    e = cv2.Canny(g, *CANNY)
    if mask is not None:
        e = cv2.bitwise_and(e, mask)
    s = cv2.HoughLinesP(
        e,
        1,
        np.pi / 180,
        PPH["threshold"],
        minLineLength=PPH["min_length"],
        maxLineGap=PPH["max_gap"],
    )
    return np.empty((0, 4), int) if s is None else s.reshape(-1, 4)


def court_lines(width: int = 720):
    """The rally, with every straight segment the probabilistic Hough finds drawn on it."""
    frames = []
    clip = _clip(width)
    mask = court_mask(clip[0].shape)
    for f in clip:
        img = (f * 0.55 + np.array(BG) * 0.45).astype(np.uint8)
        segs = segments(f, mask)
        for x1, y1, x2, y2 in segs:
            cv2.line(img, (int(x1), int(y1)), (int(x2), int(y2)), GREEN, 3, cv2.LINE_AA)
        label(img, f"straight segments on the court: {len(segs)}", (12, 30), 0.7)
        frames.append(img)
    return frames, 10


def render_all() -> dict:
    frames, fps = court_lines()
    return {"court-lines": encode(frames, fps, MEDIA_DIR / "court-lines.webp")}
