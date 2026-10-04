"""Animated opening for unit 3.2: lane segments on a minute of real freeway driving.

comma2k19 (comma.ai, MIT; Schafer et al., arXiv 1812.05752), segment
`b0c9d2329ad1606b|2018-08-03--10-35-16/13`: daytime CA-280. Each frame goes through the
pipeline the lessons build: the road trapezoid, Canny, then HoughLinesP with a gap long
enough to join a dashed line, and the near-horizontal strokes dropped (lesson 3's prior).
"""

from __future__ import annotations

import cv2
import numpy as np

from fundamentals_lab.alignment.media import BG, GREEN, encode, label
from fundamentals_lab.boundaries import highway
from fundamentals_lab.config import COMMA2K19_DIR, COMMA2K19_SEGMENT, OUTPUT_DIR

MEDIA_DIR = OUTPUT_DIR / "figures" / "boundaries" / "media"
#: The video is darker and softer than the comma10k stills; these were chosen by scanning
#: the whole minute: 85% of frames keep at least two lane segments.
CANNY = (20, 60)
PPH = {"threshold": 20, "min_length": 25, "max_gap": 60}
SECONDS, FPS = 7, 10


def segment_path():
    return COMMA2K19_DIR / COMMA2K19_SEGMENT.replace("|", "_").replace("/", "_")


def lane_segments(bgr: np.ndarray) -> np.ndarray:
    e = cv2.bitwise_and(cv2.Canny(highway.blurred(bgr), *CANNY), highway.road_mask(bgr.shape))
    s = cv2.HoughLinesP(
        e,
        1,
        np.pi / 180,
        PPH["threshold"],
        minLineLength=PPH["min_length"],
        maxLineGap=PPH["max_gap"],
    )
    if s is None:
        return np.empty((0, 4), int)
    s = s.reshape(-1, 4)
    ang = np.abs(np.degrees(np.arctan2(s[:, 3] - s[:, 1], s[:, 2] - s[:, 0])))
    return s[(ang > 20) & (ang < 160)]


def highway_lanes(width: int = 720):
    cap = cv2.VideoCapture(str(segment_path()))
    frames, i = [], 0
    while len(frames) < SECONDS * FPS:
        ok, f = cap.read()
        if not ok:
            break
        if i % 2 == 0:  # 20 fps source
            v = (f * 0.6 + np.array(BG) * 0.4).astype(np.uint8)
            cv2.polylines(v, [highway.road_polygon(f.shape)], True, (180, 180, 180), 1, cv2.LINE_AA)
            segs = lane_segments(f)
            for x1, y1, x2, y2 in segs:
                cv2.line(v, (int(x1), int(y1)), (int(x2), int(y2)), GREEN, 4, cv2.LINE_AA)
            v = cv2.resize(
                v, (width, int(width * f.shape[0] / f.shape[1])), interpolation=cv2.INTER_AREA
            )
            label(v, f"lane segments: {len(segs)}", (12, 30), 0.7)
            frames.append(v)
        i += 1
    return frames, FPS


def render_all() -> dict:
    frames, fps = highway_lanes()
    return {"highway-lanes": encode(frames, fps, MEDIA_DIR / "highway-lanes.webp")}
