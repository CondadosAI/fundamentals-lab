"""Animated opening for unit 3.2: the car's lane on real freeway driving, ADAS-style.

comma2k19 (comma.ai, MIT; Schafer et al., arXiv 1812.05752), segment
`b0c9d2329ad1606b|2018-08-03--10-35-16/13`: daytime CA-280. Each frame goes through the
pipeline the lessons build: the road trapezoid, Canny, then HoughLinesP with a gap long
enough to join a dashed line, and the near-horizontal strokes dropped (lesson 3's prior). The
segments on each side of the car are then fitted with one total-least-squares line (lesson 1),
averaged over recent frames, extended up the road, and the lane between them is filled.
"""

from __future__ import annotations

import cv2
import numpy as np

from fundamentals_lab.alignment.media import BG, GREEN, encode, label
from fundamentals_lab.boundaries import highway, pool
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


def side_line(segs: np.ndarray, side: str, cx: float):
    """One lane boundary from the HoughLinesP segments on one side of the car: points sampled
    along every segment (so a long segment weighs more), fitted by total least squares
    (lesson 1), returned as x = a*y + b. Left boundaries lean one way, right ones the other."""
    pts = []
    for x1, y1, x2, y2 in segs:
        if y1 == y2:
            continue
        lean = (x2 - x1) / (y2 - y1)  # dx per row; negative on the left boundary
        mid = (x1 + x2) / 2
        if (side == "left" and lean < 0 and mid < cx) or (
            side == "right" and lean > 0 and mid > cx
        ):
            n = int(np.hypot(x2 - x1, y2 - y1)) + 1
            pts.append(np.c_[np.linspace(x1, x2, n), np.linspace(y1, y2, n)])
    if not pts:
        return None
    vx, vy, x0, y0 = cv2.fitLine(
        np.vstack(pts).astype(np.float32), cv2.DIST_L2, 0, 0.01, 0.01
    ).ravel()
    if abs(vy) < 0.2:
        return None
    a = vx / vy
    return np.array([a, x0 - a * y0])


def draw_lane(f: np.ndarray, segs: np.ndarray, L, R) -> np.ndarray:
    """The ADAS-style overlay: the segments as thin white strokes, each side's fitted line in
    green from the bottom of the road up to just short of where the two meet, the lane filled."""
    h = f.shape[0]
    v = (f * 0.7 + np.array(BG) * 0.3).astype(np.uint8)
    for x1, y1, x2, y2 in segs:
        cv2.line(v, (int(x1), int(y1)), (int(x2), int(y2)), (235, 235, 235), 1, cv2.LINE_AA)
    y_bot = int(0.78 * h)
    if L is not None and R is not None:
        # up the road to just short of where the two boundaries meet
        y_meet = (R[1] - L[1]) / (L[0] - R[0]) if L[0] != R[0] else 0.45 * h
        y_top = int(max(0.45 * h, y_meet + 0.02 * h))
        xs = lambda ln, y: int(ln[0] * y + ln[1])  # noqa: E731
        poly = np.array(
            [
                (xs(L, y_bot), y_bot),
                (xs(L, y_top), y_top),
                (xs(R, y_top), y_top),
                (xs(R, y_bot), y_bot),
            ],
            np.int32,
        )
        fill = v.copy()
        cv2.fillPoly(fill, [poly], (120, 200, 60))
        v = cv2.addWeighted(fill, 0.35, v, 0.65, 0)
        for ln in (L, R):
            cv2.line(v, (xs(ln, y_bot), y_bot), (xs(ln, y_top), y_top), GREEN, 6, cv2.LINE_AA)
    else:
        for ln in (L, R):
            if ln is not None:
                y_top = int(0.5 * h)
                cv2.line(
                    v,
                    (int(ln[0] * y_bot + ln[1]), y_bot),
                    (int(ln[0] * y_top + ln[1]), y_top),
                    GREEN,
                    6,
                    cv2.LINE_AA,
                )
    return v


def highway_lanes(width: int = 720, smooth: float = 0.35, hold: int = 10):
    """ADAS-style overlay: the two boundaries of the car's lane, each one line fitted to that
    side's HoughLinesP segments, averaged over recent frames, extended up the road and the lane
    between them filled. The thin white strokes are the segments themselves."""
    cap = cv2.VideoCapture(str(segment_path()))
    frames, i = [], 0
    state = {"left": None, "right": None}
    missing = {"left": 0, "right": 0}
    while len(frames) < SECONDS * FPS:
        ok, f = cap.read()
        if not ok:
            break
        h, w = f.shape[:2]
        segs = lane_segments(f)
        for side in ("left", "right"):
            line = side_line(segs, side, w / 2)
            if line is not None:
                # a boundary of the car's own lane meets the bottom of the road on its own side
                xb = line[0] * 0.78 * h + line[1]
                ok_side = 0 <= xb <= 0.42 * w if side == "left" else 0.58 * w <= xb <= w
                line = line if ok_side else None
            if line is None:
                missing[side] += 1
                if missing[side] > hold:
                    state[side] = None
            else:
                missing[side] = 0
                prev = state[side]
                state[side] = line if prev is None else (1 - smooth) * prev + smooth * line
        # 20 fps source; the first 0.6 s (under a bridge) only warms up the averaging
        if i % 2 == 0 and i >= 12:
            v = draw_lane(f, segs, state["left"], state["right"])
            v = cv2.resize(v, (width, int(width * h / w)), interpolation=cv2.INTER_AREA)
            frames.append(v)
        i += 1
    return frames, FPS


def radius_sweep(width: int = 720):
    """The circle vote at one radius after another on a pool table: each ball lights up when
    the radius reaches its size. OpenCV in Practice M2's CC0 photo, with its hand labels."""
    img = pool.load("pool-close")
    g = pool.grey(img)
    labels = pool.labels()["pool-close"]
    frames = []
    radii = list(range(12, 57)) + [56] * 6
    for r in radii:
        acc = pool.fixed_radius_accumulator(g, r).astype(np.float32)
        # votes summed over 5x5 px, as a coarser accumulator (HoughCircles' dp) would
        acc = cv2.boxFilter(acc, -1, (5, 5), normalize=False)
        need = 0.15 * 2 * np.pi * r  # a peak needs 15% of the circumference
        heat = cv2.applyColorMap(
            np.clip(acc * 255 / need, 0, 255).astype(np.uint8), cv2.COLORMAP_MAGMA
        )
        v = cv2.addWeighted((img * 0.55).astype(np.uint8), 0.7, heat, 0.6, 0)
        for x, y, br in labels:
            if abs(br - r) <= 3:
                cv2.circle(v, (x, y), br, (245, 245, 245), 1, cv2.LINE_AA)
        a = acc.copy()
        for _ in range(12):
            y, x = np.unravel_index(int(a.argmax()), a.shape)
            if a[y, x] < need:
                break
            cv2.circle(v, (int(x), int(y)), r, (80, 220, 120), 2, cv2.LINE_AA)
            q = max(2, r // 2)
            a[max(0, y - q) : y + q + 1, max(0, x - q) : x + q + 1] = 0
        v = cv2.resize(
            v, (width, int(width * v.shape[0] / v.shape[1])), interpolation=cv2.INTER_AREA
        )
        label(v, f"radius {r} px", (12, 30), 0.7)
        frames.append(v)
    return frames, 10


def circle_results(width: int = 720, height: int = 540):
    """The unit's finished circle finder on every photo it was measured on: each photo plain,
    then its HoughCircles circles appearing one by one, then the score. Green circles found a
    labelled ball, amber ones did not. Ends on one VisA candle photo."""
    from fundamentals_lab.boundaries import wild
    from fundamentals_lab.config import POOL_GRADIENT_ALT

    def fit(img):
        h, w = img.shape[:2]
        k = min(width / w, height / h)
        small = cv2.resize(img, (int(w * k), int(h * k)), interpolation=cv2.INTER_AREA)
        canvas = np.full((height, width, 3), BG, np.uint8)
        y0, x0 = (height - small.shape[0]) // 2, (width - small.shape[1]) // 2
        canvas[y0 : y0 + small.shape[0], x0 : x0 + small.shape[1]] = small
        return canvas, k, x0, y0

    scenes = []
    L = pool.labels()
    for name in ("pool", "pool-close", "pool-wide"):
        img = pool.load(name)
        c = pool.detect(
            img, cv2.HOUGH_GRADIENT_ALT, POOL_GRADIENT_ALT["dp"], POOL_GRADIENT_ALT["param1"], 0.8
        )
        m = pool.match(c, L[name])
        false = {tuple(round(v, 1) for v in f) for f in m["false_at"]}
        circles = [(x, y, r, tuple(round(v, 1) for v in (x, y, r)) not in false) for x, y, r in c]
        text = f"{m['found']} of {m['balls']} balls found, {m['false']} false"
        scenes.append((img, circles, text))
    f = sorted(wild.CANDLES.glob("*.JPG"))[0]
    img = wild._load(f)
    g = cv2.medianBlur(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 5)
    c = cv2.HoughCircles(
        g, cv2.HOUGH_GRADIENT_ALT, 1.5, 40, param1=300, param2=0.8, minRadius=40, maxRadius=200
    )
    scenes.append(
        (
            img,
            [(x, y, r, True) for x, y, r in c[0]],
            f"{len(c[0])} circles; 4 on each of the 100 photos",
        )
    )

    frames = []
    for img, circles, text in scenes:
        base, k, x0, y0 = fit(img)
        frames += [base.copy() for _ in range(4)]
        circles = sorted(circles, key=lambda c: c[0])
        v = base.copy()
        for x, y, r, ok in circles:
            colour = GREEN if ok else (40, 180, 250)
            cv2.circle(
                v, (int(x0 + x * k), int(y0 + y * k)), max(3, int(r * k)), colour, 3, cv2.LINE_AA
            )
            frames.append(v.copy())
        label(v, text, (14, 32), 0.7)
        frames += [v.copy() for _ in range(14)]
    return frames, 10


def unit_results(width: int = 720):
    """The hub's opening: the lane finder on a minute of freeway, then the circle finder."""
    lanes, _ = highway_lanes(width)
    circles, fps = circle_results(width, lanes[0].shape[0])
    return lanes[:40] + circles, fps


def render_all() -> dict:
    frames, fps = highway_lanes()
    out = {"highway-lanes": encode(frames, fps, MEDIA_DIR / "highway-lanes.webp")}
    frames, fps = radius_sweep()
    out["radius-sweep"] = encode(frames, fps, MEDIA_DIR / "radius-sweep.webp")
    frames, fps = circle_results()
    out["circle-results"] = encode(frames, fps, MEDIA_DIR / "circle-results.webp", quality=40)
    frames, fps = unit_results()
    out["unit-results"] = encode(frames, fps, MEDIA_DIR / "unit-results.webp", quality=40)
    return out
