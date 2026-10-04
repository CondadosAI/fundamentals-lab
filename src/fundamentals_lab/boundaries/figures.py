"""Figures for unit 3.2, drawn from the fits and the votes themselves.

Lines are fitted lines or accumulator peaks on a comma10k highway frame, with the frame's
hand-painted lane mask shown where it is the reference; circles are HoughCircles detections
on M2's pool photos, with the 41 hand-labelled balls. PNG here; the site copies them as WebP.
Cover backgrounds (1600x900, content weighted right, no text) go to output/covers/.
"""

from __future__ import annotations

import cv2
import matplotlib
import numpy as np
from loguru import logger

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from fundamentals_lab.boundaries import highway, hough, pool, wild  # noqa: E402
from fundamentals_lab.boundaries.experiments import Scene, three_fits  # noqa: E402
from fundamentals_lab.config import (  # noqa: E402
    FLAT_THETA,
    HOUGH_THRESHOLD,
    OUTPUT_DIR,
    POOL_GRADIENT_ALT,
    PPH,
)

FIG_DIR = OUTPUT_DIR / "figures" / "boundaries"
COVER_DIR = OUTPUT_DIR / "covers"
BG = (24, 18, 14)
BG_HEX = "#0e1218"
GREEN = (80, 220, 120)
RED = (70, 70, 235)
AMBER = (40, 180, 250)
MAGENTA = (220, 80, 220)
WHITE = (245, 245, 245)


def _save(img, name: str, d=FIG_DIR):
    d.mkdir(parents=True, exist_ok=True)
    p = d / f"{name}.png"
    cv2.imwrite(str(p), img)
    logger.info(f"wrote {p}")
    return p


def _lossless(img, name: str):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    p = FIG_DIR / f"{name}.webp"
    cv2.imwrite(str(p), img, [cv2.IMWRITE_WEBP_QUALITY, 101])
    assert np.array_equal(cv2.imread(str(p)), img), "lossless WebP must round-trip"
    logger.info(f"wrote {p}")
    return p


def _dim(bgr, k=0.55):
    return (bgr * k + np.array(BG) * (1 - k)).astype(np.uint8)


def _line_through(img, rho, theta, colour, thick=2):
    a, b = np.cos(theta), np.sin(theta)
    x0, y0 = a * rho, b * rho
    L = 4000
    cv2.line(
        img,
        (int(x0 - L * b), int(y0 + L * a)),
        (int(x0 + L * b), int(y0 - L * a)),
        colour,
        thick,
        cv2.LINE_AA,
    )


def _paint(img, lane, alpha=0.6):
    out = img.copy()
    out[lane] = (out[lane] * (1 - alpha) + np.array(MAGENTA) * alpha).astype(np.uint8)
    return out


def _trapezoid(img):
    cv2.polylines(img, [highway.road_polygon(img.shape)], True, (180, 180, 180), 1, cv2.LINE_AA)


# --- the frames the page's cells load ------------------------------------------------


def page_frames(s: Scene):
    _lossless(s.bgr, "highway")
    # lesson 1's cells read the painted pixels of two markings: the dash and the solid line
    marks = highway.markings(s.lane)
    for name, P in (("dash-mask", marks[1]), ("line-mask", marks[0])):
        m = np.zeros(s.lane.shape, np.uint8)
        m[P[:, 1].astype(int), P[:, 0].astype(int)] = 255
        _lossless(cv2.cvtColor(m, cv2.COLOR_GRAY2BGR), name)


# --- hub ------------------------------------------------------------------------------


def hub(s: Scene):
    """The highway with its HoughLinesP segments, and a pool photo with its circles."""
    a = _dim(s.bgr)
    _trapezoid(a)
    seg = cv2.HoughLinesP(
        s.edges,
        1,
        np.pi / 180,
        PPH["threshold"],
        minLineLength=PPH["min_length"],
        maxLineGap=PPH["max_gap"],
    )
    for x1, y1, x2, y2 in seg.reshape(-1, 4):
        cv2.line(a, (x1, y1), (x2, y2), GREEN, 4, cv2.LINE_AA)
    img = pool.load("pool")
    b = _dim(img, 0.75)
    for x, y, r in pool.detect(
        img, cv2.HOUGH_GRADIENT_ALT, POOL_GRADIENT_ALT["dp"], POOL_GRADIENT_ALT["param1"], 0.8
    ):
        cv2.circle(b, (int(x), int(y)), int(r), GREEN, 3, cv2.LINE_AA)
    b = cv2.resize(b, (int(b.shape[1] * a.shape[0] / b.shape[0]), a.shape[0]))
    return _save(
        np.hstack([a, np.full((a.shape[0], 10, 3), BG, np.uint8), b]), "hub-lines-and-circles"
    )


# --- lesson 1 ----------------------------------------------------------------------


def lesson1(s: Scene):
    """The dash the lesson fits, magnified: its painted pixels and three fitted lines."""
    P = highway.markings(s.lane)[1]
    x0, y0 = int(P[:, 0].min()) - 25, int(P[:, 1].min()) - 25
    x1, y1 = int(P[:, 0].max()) + 25, int(P[:, 1].max()) + 25
    k = 6
    crop = cv2.resize(
        _dim(s.bgr[y0:y1, x0:x1], 0.8), None, fx=k, fy=k, interpolation=cv2.INTER_NEAREST
    )
    for x, y in P.astype(int):
        cv2.rectangle(
            crop,
            ((x - x0) * k, (y - y0) * k),
            ((x - x0) * k + k - 1, (y - y0) * k + k - 1),
            MAGENTA,
            -1,
        )
    f = three_fits(P)
    c = P.mean(axis=0)
    for key, colour in (("y_on_x", RED), ("x_on_y", AMBER), ("tls", GREEN)):
        t = np.radians(f[key])
        d = np.array([np.cos(t), np.sin(t)]) * 200
        p, q = c - d, c + d
        cv2.line(
            crop,
            (int((p[0] - x0 + 0.5) * k), int((p[1] - y0 + 0.5) * k)),
            (int((q[0] - x0 + 0.5) * k), int((q[1] - y0 + 0.5) * k)),
            colour,
            3,
            cv2.LINE_AA,
        )
    _save(crop, "l1-dash-three-fits")

    # The lesson's result: a total-least-squares line through each painted marking, drawn
    # across the road the lessons search in.
    v = _dim(s.bgr, 0.8)
    road = highway.road_mask(s.lane.shape) > 0
    paint = np.zeros_like(v)
    paint[highway.lane_in_road(s.lane)] = MAGENTA
    v = np.where(paint.any(axis=2, keepdims=True), cv2.addWeighted(v, 0.4, paint, 0.6, 0), v)
    lines = np.zeros_like(v)
    for M in highway.markings(s.lane):
        t = np.radians(three_fits(M)["tls"])
        c = M.mean(axis=0)
        d = np.array([np.cos(t), np.sin(t)]) * 3000
        p, q = (c - d).astype(int), (c + d).astype(int)
        cv2.line(lines, tuple(map(int, p)), tuple(map(int, q)), GREEN, 5, cv2.LINE_AA)
    on = lines.any(axis=2) & road
    v[on] = lines[on]
    cv2.polylines(v, [highway.road_polygon(s.lane.shape)], True, WHITE, 1, cv2.LINE_AA)
    return _save(v, "l1-lanes-fitted")


# --- lesson 2 ----------------------------------------------------------------------


def lesson2(s: Scene):
    acc = s.acc.astype(float)
    half = (acc.shape[0] - 1) // 2
    fig, ax = plt.subplots(figsize=(9, 5), dpi=110, facecolor=BG_HEX)
    ax.set_facecolor(BG_HEX)
    ax.imshow(np.log1p(acc), aspect="auto", cmap="magma", extent=[0, 180, half, -half])
    for _, ri, ti in s.lines:
        ax.plot(ti, hough.rho_of(ri, s.acc), "o", mfc="none", mec="#7CFC9A", ms=8, mew=1.3)
    ax.set_xlabel("θ (degrees)", color="#cbd5e1")
    ax.set_ylabel("ρ (pixels)", color="#cbd5e1")
    ax.set_ylim(1100, -600)
    ax.tick_params(colors="#94a3b8")
    for sp in ax.spines.values():
        sp.set_color("#334155")
    fig.tight_layout()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / "l2-accumulator.png", facecolor=BG_HEX)
    plt.close(fig)
    img = _paint(_dim(s.bgr), s.lane, 0.5)
    _trapezoid(img)
    for rr, tt in s.line_list():
        _line_through(img, rr, tt, GREEN, 1)
    return _save(img, "l2-lines")


# --- lesson 3 ----------------------------------------------------------------------


def lesson3(s: Scene):
    import json

    from fundamentals_lab.config import BOUNDARY_NUMBERS_JSON

    name = json.loads(BOUNDARY_NUMBERS_JSON.read_text())["probabilistic"]["showcase"]["frame"]
    hard = Scene(name)
    _lesson3_panels(hard)
    _lesson3_segments(s)


def _lesson3_panels(s: Scene):
    blur = highway.blurred(s.bgr)
    gx = cv2.Sobel(blur, cv2.CV_32F, 1, 0)
    gy = cv2.Sobel(blur, cv2.CV_32F, 0, 1)
    orient = (np.degrees(np.arctan2(gy, gx)) % 180).astype(np.float32)
    a5, _ = hough.restricted_accumulator(s.edges, orient, 5)
    lm5 = [
        (v, r, t)
        for v, r, t in hough.local_maxima(a5, HOUGH_THRESHOLD)
        if not FLAT_THETA[0] <= t <= FLAT_THETA[1]
    ]
    panels = []
    for acc, lines in ((s.acc, s.lines), (a5, lm5)):
        img = _dim(s.bgr)
        _trapezoid(img)
        for _, ri, ti in lines:
            _line_through(img, hough.rho_of(ri, acc), np.radians(ti), GREEN, 1)
        panels.append(img)
    _save(
        np.hstack([panels[0], np.full((s.bgr.shape[0], 10, 3), BG, np.uint8), panels[1]]),
        "l3-full-vs-restricted",
    )


def _lesson3_segments(s: Scene):
    img = _paint(_dim(s.bgr), s.lane, 0.45)
    _trapezoid(img)
    seg = cv2.HoughLinesP(
        s.edges,
        1,
        np.pi / 180,
        PPH["threshold"],
        minLineLength=PPH["min_length"],
        maxLineGap=PPH["max_gap"],
    )
    for x1, y1, x2, y2 in seg.reshape(-1, 4):
        cv2.line(img, (x1, y1), (x2, y2), GREEN, 4, cv2.LINE_AA)
        cv2.circle(img, (x1, y1), 6, AMBER, -1, cv2.LINE_AA)
        cv2.circle(img, (x2, y2), 6, AMBER, -1, cv2.LINE_AA)
    return _save(img, "l3-segments")


# --- lesson 4 ----------------------------------------------------------------------


def lesson4():
    L = pool.labels()
    tiles = []
    for name in ("pool", "pool-close", "pool-wide"):
        img = pool.load(name)
        v = _dim(img, 0.8)
        for lx, ly, lr in L[name]:
            cv2.circle(v, (lx, ly), lr, WHITE, 1, cv2.LINE_AA)
        for x, y, r in pool.detect(
            img, cv2.HOUGH_GRADIENT_ALT, POOL_GRADIENT_ALT["dp"], POOL_GRADIENT_ALT["param1"], 0.8
        ):
            cv2.circle(v, (int(x), int(y)), int(r), GREEN, 3, cv2.LINE_AA)
        tiles.append(v)
        _save(v, f"l4-{name}")
    # the textbook accumulator at the big ball's radius, over pool-close
    img = pool.load("pool-close")
    acc = pool.fixed_radius_accumulator(pool.grey(img), 52).astype(float)
    acc = cv2.GaussianBlur(acc, (0, 0), 1.0)
    heat = cv2.applyColorMap(
        (255 * np.log1p(acc) / np.log1p(acc.max())).astype(np.uint8), cv2.COLORMAP_MAGMA
    )
    _save(cv2.addWeighted(_dim(img, 0.5), 0.5, heat, 0.8, 0), "l4-accumulator-r52")


# --- in the wild -----------------------------------------------------------------


def wild_figures():
    bgr = wild._load(wild.CLOISTER)
    _, e = wild._edges(bgr)
    acc = hough.accumulator(e)
    lm = hough.local_maxima(acc, 80)
    img = _dim(bgr)
    ob = [
        (hough.rho_of(r, acc), np.radians(t)) for _, r, t in lm if 15 <= t <= 75 or 105 <= t <= 165
    ][:12]
    for rho, t in ob:
        _line_through(img, rho, t, GREEN, 2)
    vp = wild.cloister_vanishing()["median_point"]
    cv2.circle(img, (int(vp[0]), int(vp[1])), 9, AMBER, -1, cv2.LINE_AA)
    _save(img, "wild-cloister-vanishing")

    bgr = wild._load(wild.DOCUMENT)
    _, e = wild._edges(bgr)
    seg = cv2.HoughLinesP(e, 1, np.pi / 180, 90, minLineLength=80, maxLineGap=5).reshape(-1, 4)
    img = _dim(bgr)
    for x1, y1, x2, y2 in seg:
        cv2.line(img, (x1, y1), (x2, y2), GREEN, 3, cv2.LINE_AA)
    _save(img, "wild-document-segments")

    f = sorted(wild.CANDLES.glob("*.JPG"))[0]
    img = wild._load(f)
    g = cv2.medianBlur(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 5)
    c = cv2.HoughCircles(
        g, cv2.HOUGH_GRADIENT_ALT, 1.5, 40, param1=300, param2=0.8, minRadius=40, maxRadius=200
    )
    v = _dim(img, 0.8)
    for x, y, r in c[0]:
        cv2.circle(v, (int(x), int(y)), int(r), GREEN, 4, cv2.LINE_AA)
    _save(v, "wild-candles")


# --- lab data ----------------------------------------------------------------------


def lab_data(s: Scene):
    """Lossless WebPs the labs read. R: Canny edges; G: gradient direction [0, 360) halved;
    B: gradient direction folded to [0, 180). Directions only at edge pixels."""

    def pack(gray, edges):
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1)
        a360 = np.degrees(np.arctan2(gy, gx)) % 360
        a180 = np.degrees(np.arctan2(gy, gx)) % 180
        img = np.zeros((*edges.shape, 3), np.uint8)
        img[..., 2] = edges
        img[..., 1] = np.clip(np.floor(a360 / 2), 0, 179).astype(np.uint8)
        img[..., 0] = np.clip(np.rint(a180) % 180, 0, 179).astype(np.uint8)
        img[edges == 0] = 0
        return img

    def pack16(gray, edges):
        """CircleLab: R edges; G, B the high and low bytes of the direction in [0, 360) as a
        16-bit fraction of a turn. At r = 52 px a 2-degree step would move a vote by 1.8 px."""
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1)
        a = np.rint((np.degrees(np.arctan2(gy, gx)) % 360) * 65536 / 360).astype(np.int64) % 65536
        img = np.zeros((*edges.shape, 3), np.uint8)
        img[..., 2] = edges
        img[..., 1] = (a >> 8).astype(np.uint8)
        img[..., 0] = (a & 255).astype(np.uint8)
        img[edges == 0] = 0
        return img

    _lossless(pack(highway.blurred(s.bgr), s.edges), "lab-highway")
    # the frame's whole hand-painted lane mask, for the notebook's on-paint scores
    _lossless(cv2.merge([s.lane.astype(np.uint8) * 255] * 3), "highway-lane")
    g = pool.grey(pool.load("pool-close"))
    _lossless(pack16(g, cv2.Canny(g, 50, 150)), "lab-pool-close")


# --- covers ------------------------------------------------------------------------


def covers():
    def place(content):
        canvas = np.full((900, 1600, 3), BG, np.uint8)
        w = int(content.shape[1] * 900 / content.shape[0])
        c = cv2.resize(content, (w, 900), interpolation=cv2.INTER_AREA)
        if w > 1000:
            c = c[:, (w - 1000) // 2 : (w - 1000) // 2 + 1000]
            w = 1000
        canvas[:, 1600 - w :] = c
        return canvas

    rd = lambda n: cv2.imread(str(FIG_DIR / f"{n}.png"))  # noqa: E731

    def band(img, top, bottom, width=1000):
        """A horizontal band of a figure, rows top..bottom as fractions, at a given width."""
        h, w = img.shape[:2]
        crop = img[int(top * h) : int(bottom * h)]
        return cv2.resize(
            crop, (width, int(crop.shape[0] * width / w)), interpolation=cv2.INTER_AREA
        )

    # Every cover shows the most complete result of its post; the hub shows both of the
    # unit's finished systems, lane segments above and circles below.
    road = band(rd("l3-segments"), 0.33, 0.80)
    balls = band(rd("l4-pool"), 0.03, 0.72)
    gap = np.full((900 - road.shape[0] - balls.shape[0], 1000, 3), BG, np.uint8)
    _save(place(np.vstack([road, gap, balls])), "boundary-detection", COVER_DIR)
    _save(place(rd("l1-lanes-fitted")), "line-fitting-least-squares", COVER_DIR)
    _save(place(rd("wild-cloister-vanishing")), "hough-transform", COVER_DIR)
    _save(place(rd("l3-segments")), "probabilistic-hough-transform", COVER_DIR)
    _save(place(rd("l4-pool")), "hough-circle-transform", COVER_DIR)


def render_all():
    s = Scene()
    page_frames(s)
    hub(s)
    lesson1(s)
    lesson2(s)
    lesson3(s)
    lesson4()
    wild_figures()
    lab_data(s)
    covers()
