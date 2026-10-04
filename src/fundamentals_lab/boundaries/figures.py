"""Figures for unit 3.2, drawn from the fits themselves.

Every line drawn is a fitted line or an accumulator peak, every circle a HoughCircles
detection, every vote count the accumulator's. PNG here; the site copies them as WebP.
Cover backgrounds (1600x900, content weighted right, no text) go to output/covers/.
"""

from __future__ import annotations

import cv2
import matplotlib
import numpy as np
from loguru import logger

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from fundamentals_lab.boundaries import fitting, frame, ght, hough, wild  # noqa: E402
from fundamentals_lab.boundaries.experiments import Scene  # noqa: E402
from fundamentals_lab.config import (  # noqa: E402
    GHT,
    GHT_RING_INNER,
    HOUGH_THRESHOLD,
    LEFT_EDGE_BOX,
    OUTPUT_DIR,
    PPH,
)

FIG_DIR = OUTPUT_DIR / "figures" / "boundaries"
COVER_DIR = OUTPUT_DIR / "covers"
BG = (24, 18, 14)
BG_HEX = "#0e1218"
GREEN = (80, 220, 120)
RED = (70, 70, 235)
AMBER = (40, 180, 250)
CYAN = (230, 200, 60)
WHITE = (245, 245, 245)


def _save(img, name: str, d=FIG_DIR):
    d.mkdir(parents=True, exist_ok=True)
    p = d / f"{name}.png"
    cv2.imwrite(str(p), img)
    logger.info(f"wrote {p}")
    return p


def _dim(bgr, k=0.55):
    return (bgr * k + np.array(BG) * (1 - k)).astype(np.uint8)


def _line_through(img, rho, theta, colour, thick=2):
    a, b = np.cos(theta), np.sin(theta)
    x0, y0 = a * rho, b * rho
    L = 3000
    p1 = (int(x0 - L * b), int(y0 + L * a))
    p2 = (int(x0 + L * b), int(y0 - L * a))
    cv2.line(img, p1, p2, colour, thick, cv2.LINE_AA)


def working_frame(s: Scene):
    """The exact pixels the unit measures, for the site (lossless)."""
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    p = FIG_DIR / "pcb1.webp"
    cv2.imwrite(str(p), s.bgr, [cv2.IMWRITE_WEBP_QUALITY, 101])
    back = cv2.imread(str(p))
    assert np.array_equal(back, s.bgr), "lossless WebP must round-trip"
    logger.info(f"wrote {p}")


def hub(s: Scene):
    """The board with what each lesson finds on it: lines, segments, circles."""
    img = _dim(s.bgr)
    for _, ri, ti in s.lines[:6]:
        _line_through(img, hough.rho_of(ri, s.acc), np.radians(ti), (120, 120, 120), 1)
    seg = cv2.HoughLinesP(
        s.edges,
        1,
        np.pi / 180,
        PPH["threshold"],
        minLineLength=PPH["min_length"],
        maxLineGap=PPH["max_gap"],
    )
    for x1, y1, x2, y2 in seg.reshape(-1, 4):
        cv2.line(img, (x1, y1), (x2, y2), GREEN, 3, cv2.LINE_AA)
    for x, y, r in s.transducers:
        cv2.circle(img, (int(x), int(y)), int(r), AMBER, 3, cv2.LINE_AA)
    return _save(img, "hub-board")


def lesson1(s: Scene):
    """The left edge, magnified: its pixels, the LS line and the TLS line."""
    x0, y0, x1, y1 = 96, 250, 150, 460
    P = frame.box_points(s.edges, LEFT_EDGE_BOX)
    ls = fitting.least_squares(P)
    tls = fitting.total_least_squares(P)
    k = 4
    crop = cv2.resize(
        _dim(s.bgr[y0:y1, x0:x1], 0.7), None, fx=k, fy=k, interpolation=cv2.INTER_NEAREST
    )
    for x, y in P.astype(int):
        cv2.rectangle(
            crop,
            ((x - x0) * k, (y - y0) * k),
            ((x - x0) * k + k - 1, (y - y0) * k + k - 1),
            WHITE,
            -1,
        )

    def draw(m=None, b=None, c=None, d=None, colour=None):
        ys = np.array([y0, y1], float)
        xs = (ys - b) / m if m is not None else c[0] + (ys - c[1]) * d[0] / d[1]
        p = [
            (int((x - x0 + 0.5) * k), int((y - y0 + 0.5) * k)) for x, y in zip(xs, ys, strict=True)
        ]
        cv2.line(crop, p[0], p[1], colour, 3, cv2.LINE_AA)

    draw(m=ls["m"], b=ls["b"], colour=RED)
    draw(c=tls["centroid"], d=tls["direction"], colour=GREEN)
    return _save(crop, "l1-left-edge")


def lesson2(s: Scene):
    """The accumulator (log votes) and the board with its eight strongest lines."""
    acc = s.acc.astype(float)
    half = (acc.shape[0] - 1) // 2
    fig, ax = plt.subplots(figsize=(9, 5), dpi=110, facecolor=BG_HEX)
    ax.set_facecolor(BG_HEX)
    ax.imshow(np.log1p(acc), aspect="auto", cmap="magma", extent=[0, 180, half, -half])
    for _, ri, ti in s.lines[:8]:
        ax.plot(ti, hough.rho_of(ri, s.acc), "o", mfc="none", mec="#7CFC9A", ms=9, mew=1.5)
    ax.set_xlabel("θ (degrees)", color="#cbd5e1")
    ax.set_ylabel("ρ (pixels)", color="#cbd5e1")
    ax.set_ylim(900, -900)
    ax.tick_params(colors="#94a3b8")
    for sp in ax.spines.values():
        sp.set_color("#334155")
    fig.tight_layout()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    p = FIG_DIR / "l2-accumulator.png"
    fig.savefig(p, facecolor=BG_HEX)
    plt.close(fig)
    logger.info(f"wrote {p}")

    img = _dim(s.bgr)
    for _, ri, ti in s.lines[:8]:
        _line_through(img, hough.rho_of(ri, s.acc), np.radians(ti), GREEN, 2)
    return _save(img, "l2-top-lines")


def lesson3(s: Scene):
    """Full against gradient-restricted (±5°): every line over the threshold, drawn."""
    gx = cv2.Sobel(s.blur, cv2.CV_32F, 1, 0)
    gy = cv2.Sobel(s.blur, cv2.CV_32F, 0, 1)
    orient = (np.degrees(np.arctan2(gy, gx)) % 180).astype(np.float32)
    a5, _ = hough.restricted_accumulator(s.edges, orient, 5)
    panels = []
    for acc, lines in ((s.acc, s.lines), (a5, hough.local_maxima(a5, HOUGH_THRESHOLD))):
        img = _dim(s.bgr)
        for _, ri, ti in lines:
            _line_through(img, hough.rho_of(ri, acc), np.radians(ti), GREEN, 1)
        panels.append(img)
    _save(
        np.hstack([panels[0], np.full((s.bgr.shape[0], 8, 3), BG, np.uint8), panels[1]]),
        "l3-full-vs-restricted",
    )
    img = _dim(s.bgr)
    seg = cv2.HoughLinesP(
        s.edges,
        1,
        np.pi / 180,
        PPH["threshold"],
        minLineLength=PPH["min_length"],
        maxLineGap=PPH["max_gap"],
    )
    for x1, y1, x2, y2 in seg.reshape(-1, 4):
        cv2.line(img, (x1, y1), (x2, y2), GREEN, 3, cv2.LINE_AA)
        cv2.circle(img, (x1, y1), 4, AMBER, -1, cv2.LINE_AA)
        cv2.circle(img, (x2, y2), 4, AMBER, -1, cv2.LINE_AA)
    return _save(img, "l3-segments")


def lesson4(s: Scene):
    """The ring template with some R-table vectors, and the vote it casts over the board."""
    tm = ght.ring_template(s.blur, s.transducers[0], GHT_RING_INNER)
    tab, ref = ght.r_table(tm)
    t = cv2.cvtColor(tm, cv2.COLOR_GRAY2BGR)
    t = _dim(t, 0.6)
    e = cv2.Canny(tm, 50, 150)
    t[e > 0] = WHITE
    rng = np.random.default_rng(0)
    for b in rng.choice(sorted(tab), 14, replace=False):
        dx, dy = tab[b][0]
        px, py = ref[0] - dx, ref[1] - dy
        cv2.arrowedLine(
            t, (int(px), int(py)), (int(ref[0]), int(ref[1])), AMBER, 1, cv2.LINE_AA, tipLength=0.05
        )
    cv2.circle(t, (int(ref[0]), int(ref[1])), 5, GREEN, -1)
    _save(cv2.resize(t, None, fx=2, fy=2, interpolation=cv2.INTER_NEAREST), "l4-r-table")
    acc = ght.vote(s.blur, tab).astype(float)
    acc = cv2.resize(
        np.log1p(acc), (s.bgr.shape[1], s.bgr.shape[0]), interpolation=cv2.INTER_NEAREST
    )
    heat = cv2.applyColorMap((255 * acc / acc.max()).astype(np.uint8), cv2.COLORMAP_MAGMA)
    img = cv2.addWeighted(_dim(s.bgr, 0.5), 0.5, heat, 0.8, 0)
    for _, x, y in ght.peaks(ght.vote(s.blur, tab), 2, GHT["min_dist"]):
        cv2.circle(img, (x, y), 14, GREEN, 2, cv2.LINE_AA)
    return _save(img, "l4-votes")


def wild_figures():
    """The cloister's vanishing lines, the document's segments, the pool table's vote."""
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

    bgr = wild._load(wild.POOL)
    g, _ = wild._edges(bgr)
    c = cv2.HoughCircles(
        g, cv2.HOUGH_GRADIENT, 1, 18, param1=150, param2=22, minRadius=8, maxRadius=24
    )[0]
    rr = np.median(c[:, 2])
    k = int(np.argmin(np.abs(c[:, 2] - rr)))
    tab, _ = ght.r_table(ght.ring_template(g, c[k], None))
    img = _dim(bgr)
    for x, y, r in c:
        cv2.circle(img, (int(x), int(y)), int(r), (150, 150, 150), 1, cv2.LINE_AA)
    cv2.circle(img, (int(c[k][0]), int(c[k][1])), int(c[k][2]), AMBER, 3, cv2.LINE_AA)
    for _, x, y in ght.peaks(ght.vote(g, tab), 4, int(rr))[1:]:
        cv2.drawMarker(img, (x, y), GREEN, cv2.MARKER_TILTED_CROSS, 18, 2)
    _save(img, "wild-pool-ght")


def covers(s: Scene):
    """1600x900 backgrounds: content on the right, nothing on the left, no text."""

    def place(content):
        canvas = np.full((900, 1600, 3), (24, 18, 14), np.uint8)
        h = 900
        w = int(content.shape[1] * h / content.shape[0])
        c = cv2.resize(content, (w, h), interpolation=cv2.INTER_AREA)
        if w > 1000:
            c = c[:, (w - 1000) // 2 : (w - 1000) // 2 + 1000]
            w = 1000
        canvas[:, 1600 - w :] = c
        return canvas

    board = s.bgr
    hubimg = cv2.imread(str(FIG_DIR / "hub-board.png"))
    _save(place(hubimg), "boundary-detection", COVER_DIR)
    _save(
        place(cv2.imread(str(FIG_DIR / "l1-left-edge.png"))),
        "line-fitting-least-squares",
        COVER_DIR,
    )
    _save(place(cv2.imread(str(FIG_DIR / "l2-top-lines.png"))), "hough-transform", COVER_DIR)
    _save(
        place(cv2.imread(str(FIG_DIR / "l3-segments.png"))),
        "probabilistic-hough-transform",
        COVER_DIR,
    )
    _save(
        place(cv2.imread(str(FIG_DIR / "l4-votes.png"))), "generalized-hough-transform", COVER_DIR
    )
    del board


def lab_data(s: Scene):
    """The board for the labs, as one lossless PNG.

    R: the unit's Canny edge map (255 or 0). G: gradient direction in [0, 360) halved,
    for the generalized Hough's R-table. B: gradient direction folded to [0, 180), for
    gradient-restricted voting. Same Sobel on the same blur as experiments.py.
    """
    gx = cv2.Sobel(s.blur, cv2.CV_32F, 1, 0)
    gy = cv2.Sobel(s.blur, cv2.CV_32F, 0, 1)
    a360 = ght.fast_atan2(gy, gx)
    a180 = np.degrees(np.arctan2(gy, gx)) % 180
    img = np.zeros((*s.edges.shape, 3), np.uint8)  # BGR order on disk
    img[..., 2] = s.edges
    img[..., 1] = np.clip(np.floor(a360 / 2), 0, 179).astype(np.uint8)
    img[..., 0] = np.clip(np.rint(a180) % 180, 0, 179).astype(np.uint8)
    return _save(img, "lab-board")


def render_all():
    s = Scene()
    working_frame(s)
    hub(s)
    lesson1(s)
    lesson2(s)
    lesson3(s)
    lesson4(s)
    wild_figures()
    covers(s)
    lab_data(s)
