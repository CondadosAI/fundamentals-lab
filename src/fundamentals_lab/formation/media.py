"""Animated figures for unit 1.1: the real application each post opens with.

Drawn from data the track already uses: OpenCV's calibration boards (lessons 1 and 3),
a CC0 circular fisheye of a courtyard in Kashan (the hub and lesson 4) and one frame of
the pickleball match (lesson 2). Each function returns BGR frames and a frame rate;
`render_all` writes them as animated WebP through `alignment.media.encode`.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from loguru import logger

from fundamentals_lab.alignment.media import BG, CLIP, GREEN, RED, encode, label
from fundamentals_lab.config import FORMATION_BOARD, OUTPUT_DIR
from fundamentals_lab.formation.dataset import fetch_opencv_calibration

MEDIA_DIR = OUTPUT_DIR / "figures" / "formation" / "media"
AMBER = (80, 170, 250)
EASE = lambda t: 0.5 - 0.5 * np.cos(np.pi * t)  # noqa: E731


# --- the calibration boards ---------------------------------------------------------


def _boards():
    """Calibrate on the 13 boards; return greys, K, distortion and per-board poses."""
    cols, rows = FORMATION_BOARD
    obj = np.array([[x, y, 0] for y in range(rows) for x in range(cols)], np.float32)
    greys, img_pts = [], []
    for p in fetch_opencv_calibration():
        g = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        ok, c = cv2.findChessboardCorners(g, (cols, rows))
        if not ok:
            continue
        crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.01)
        greys.append(g)
        img_pts.append(cv2.cornerSubPix(g, c, (11, 11), (-1, -1), crit))
    _, K, dist, rvecs, tvecs = cv2.calibrateCamera(
        [obj] * len(greys), img_pts, greys[0].shape[::-1], None, None
    )
    return greys, img_pts, K, dist, rvecs, tvecs


def ar_cube():
    """A cube standing on each board, drawn through the calibrated pinhole model."""
    greys, _, K, dist, rvecs, tvecs = _boards()
    cube = np.float32(
        [[3, 1, 0], [6, 1, 0], [6, 4, 0], [3, 4, 0], [3, 1, -3], [6, 1, -3], [6, 4, -3], [3, 4, -3]]
    )
    frames = []
    for g, r, t in zip(greys, rvecs, tvecs, strict=True):
        f = cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)
        p, _ = cv2.projectPoints(cube, r, t, K, dist)
        p = np.rint(p.reshape(-1, 2)).astype(np.int32)
        overlay = f.copy()
        cv2.fillPoly(overlay, [p[:4]], GREEN)
        f = cv2.addWeighted(overlay, 0.55, f, 0.45, 0)
        for i in range(4):
            cv2.line(f, tuple(p[i]), tuple(p[i + 4]), AMBER, 3, cv2.LINE_AA)
        cv2.polylines(f, [p[4:]], True, RED, 3, cv2.LINE_AA)
        label(f, "a 3-D cube, placed by the pinhole model", (10, 28), 0.6)
        frames.append(f)
    return frames, 2


def hub_pipeline():
    """The hub's pipeline on board view 0, as six panels titled with the hub's step numbers:
    the corners, the pinhole prediction without the bend, the bend itself, the prediction
    with it, the straightened photo and a cube drawn in 3-D."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from fundamentals_lab.formation.calibration import board_points

    greys, img_pts, K, dist, rvecs, tvecs = _boards()
    g, seen, r, t = greys[0], img_pts[0].reshape(-1, 2), rvecs[0], tvecs[0]
    base = cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)
    dim = (base * 0.6).astype(np.uint8)
    board = board_points().astype(np.float32)

    def dots(img, pts, colour, rad=4):
        for x, y in pts:
            cv2.circle(img, (round(float(x) * 4), round(float(y) * 4)), rad * 4, colour, -1,
                       cv2.LINE_AA, shift=2)

    p1 = dim.copy()
    dots(p1, seen, GREEN)

    pin, _ = cv2.projectPoints(board, r, t, K, None)
    pin = pin.reshape(-1, 2)
    p2 = dim.copy()
    for a, b in zip(seen, pin, strict=True):
        cv2.line(p2, tuple(np.rint(a).astype(int)), tuple(np.rint(b).astype(int)), AMBER, 2)
    dots(p2, seen, GREEN, 3)
    dots(p2, pin, RED, 3)

    # the bend: where the lens moves points on a grid, drawn at true size
    p3 = dim.copy()
    h, w = g.shape
    xs, ys = np.meshgrid(np.arange(20, w, 40), np.arange(20, h, 40))
    grid = np.stack([xs.ravel(), ys.ravel()], 1).astype(np.float32)
    rays = cv2.undistortPoints(grid[:, None], K, None).reshape(-1, 2)
    rays3 = np.c_[rays, np.ones(len(rays))].astype(np.float32)
    bent, _ = cv2.projectPoints(rays3, np.zeros(3), np.zeros(3), K, dist)
    for a, b in zip(grid, bent.reshape(-1, 2), strict=True):
        cv2.arrowedLine(p3, tuple(np.rint(a).astype(int)), tuple(np.rint(b).astype(int)),
                        AMBER, 2, cv2.LINE_AA, tipLength=0.25)

    full, _ = cv2.projectPoints(board, r, t, K, dist)
    p4 = dim.copy()
    dots(p4, seen, GREEN, 4)
    dots(p4, full.reshape(-1, 2), RED, 2)

    p5 = cv2.undistort(base, K, dist)

    p6 = base.copy()
    cube = np.float32(
        [[3, 1, 0], [6, 1, 0], [6, 4, 0], [3, 4, 0], [3, 1, -3], [6, 1, -3], [6, 4, -3], [3, 4, -3]]
    )
    c, _ = cv2.projectPoints(cube, r, t, K, dist)
    c = np.rint(c.reshape(-1, 2)).astype(np.int32)
    overlay = p6.copy()
    cv2.fillPoly(overlay, [c[:4]], GREEN)
    p6 = cv2.addWeighted(overlay, 0.55, p6, 0.45, 0)
    for i in range(4):
        cv2.line(p6, tuple(c[i]), tuple(c[i + 4]), AMBER, 3, cv2.LINE_AA)
    cv2.polylines(p6, [c[4:]], True, RED, 3, cv2.LINE_AA)

    panels = [
        (p1, "1  Points on the board"),
        (p2, "2, 4  Without the bend"),
        (p3, "3  The bend"),
        (p4, "2–4  With the bend"),
        (p5, "5  Use: straightened"),
        (p6, "5  Use: a cube in 3-D"),
    ]
    bg = "#0e1218"
    fig, axes = plt.subplots(3, 2, figsize=(8, 10.3), dpi=110, facecolor=bg)
    for ax, (img, title) in zip(axes.ravel(), panels, strict=True):
        ax.set_facecolor(bg)
        ax.imshow(img[..., ::-1], aspect="auto")
        ax.set_title(title, color="#e2e8f0", fontsize=19, loc="left", pad=6)
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_color("#334155")
    fig.tight_layout(pad=0.6, h_pad=1.2, w_pad=0.8)
    out = OUTPUT_DIR / "figures" / "formation"
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / "hub-pipeline.png", facecolor=bg)
    plt.close(fig)
    logger.info(f"wrote {out / 'hub-pipeline.png'}")
    return out / "hub-pipeline.png"


def undistort_morph(step: int = 64):
    """One board eased from the photo as taken to the corrected one, and back.

    Over it, a grid that is straight in the corrected picture: drawn on the photo as
    taken it curves, most at the corners, by as much as the lens bends the image. The
    whole frame is kept (alpha = 1), so the corrected picture's own border bows too.
    """
    greys, _, K, dist, _, _ = _boards()
    g = greys[0]
    h, w = g.shape
    newK, _ = cv2.getOptimalNewCameraMatrix(K, dist, (w, h), 1, (w, h))
    mapx, mapy = cv2.initUndistortRectifyMap(K, dist, None, newK, (w, h), cv2.CV_32FC1)
    idx, idy = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    base = cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)

    # Grid lines straight in the corrected image, sampled densely along each line.
    lines = []
    for x in range(0, w + 1, step):
        lines.append(np.stack([np.full(60, x), np.linspace(0, h, 60)], 1))
    for y in range(0, h + 1, step):
        lines.append(np.stack([np.linspace(0, w, 80), np.full(80, y)], 1))
    straight = [ln.astype(np.float64) for ln in lines]
    # Where those points were in the photo as taken: back through K, forward through the lens.
    curved = []
    for ln in straight:
        rays = cv2.undistortPoints(ln.reshape(-1, 1, 2), newK, None).reshape(-1, 2)
        pts3 = np.concatenate([rays, np.ones((len(rays), 1))], 1)
        p, _ = cv2.projectPoints(pts3, np.zeros(3), np.zeros(3), K, dist)
        curved.append(p.reshape(-1, 2))
    shift = max(
        np.linalg.norm(c - s_, axis=1).max() for c, s_ in zip(curved, straight, strict=True)
    )

    ts = [0.0] * 10 + [EASE(i / 19) for i in range(20)] + [1.0] * 12
    ts += [1 - EASE(i / 19) for i in range(20)]
    frames = []
    for t in ts:
        t = float(t)
        mx = (idx + t * (mapx - idx)).astype(np.float32)
        my = (idy + t * (mapy - idy)).astype(np.float32)
        f = cv2.remap(base, mx, my, cv2.INTER_LINEAR, borderValue=BG)
        for c, s_ in zip(curved, straight, strict=True):
            pts = (1 - t) * c + t * s_
            cv2.polylines(f, [np.rint(pts).astype(np.int32)], False, RED, 1, cv2.LINE_AA)
        label(f, "as taken" if t < 0.5 else "corrected", (10, 28), 0.6)
        frames.append(f)
    logger.info(f"undistort-morph: largest grid shift {shift:.1f} px")
    return frames, 10


# --- the fisheye --------------------------------------------------------------------


class Fisheye:
    """The Kashan circular fisheye, as an equisolid lens: r = 2 f sin(theta / 2).

    The EF 8-15 mm at 8 mm puts 180 degrees across the image circle, so f follows from
    the circle's radius. The circle is found in the image, not assumed.
    """

    def __init__(self, size: int = 1600):
        from fundamentals_lab.imagedata.wild import load

        self.img = cv2.resize(
            load("fisheye-kashan.jpg"), (size, size), interpolation=cv2.INTER_AREA
        )
        mask = (cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY) > 12).astype(np.uint8)
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        (self.cx, self.cy), radius = cv2.minEnclosingCircle(max(cnts, key=cv2.contourArea))
        self.f = radius / (2 * np.sin(np.pi / 4))

    def _rays(self, hfov, yaw, pitch, W, H):
        fx = (W / 2) / np.tan(np.radians(hfov) / 2)
        xs, ys = np.meshgrid(np.arange(W) - W / 2, np.arange(H) - H / 2)
        d = np.stack([xs, ys, np.full_like(xs, fx)], -1).astype(np.float64)
        d /= np.linalg.norm(d, axis=-1, keepdims=True)
        cy, sy = np.cos(np.radians(yaw)), np.sin(np.radians(yaw))
        cp, sp = np.cos(np.radians(pitch)), np.sin(np.radians(pitch))
        Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
        Rx = np.array([[1, 0, 0], [0, cp, -sp], [0, sp, cp]])
        return d @ (Ry @ Rx).T

    def _to_image(self, d):
        th = np.arccos(np.clip(d[..., 2], -1, 1))
        ph = np.arctan2(d[..., 1], d[..., 0])
        r = 2 * self.f * np.sin(th / 2)
        return self.cx + r * np.cos(ph), self.cy + r * np.sin(ph)

    def view(self, hfov, yaw=0.0, pitch=0.0, W=600, H=400):
        """What a rectilinear (pinhole) camera pointed that way would have seen."""
        mx, my = self._to_image(self._rays(hfov, yaw, pitch, W, H))
        return cv2.remap(
            self.img, mx.astype(np.float32), my.astype(np.float32), cv2.INTER_LINEAR, borderValue=BG
        )

    def footprint(self, hfov, yaw, pitch, W=600, H=400, n=40):
        """The border of that view, traced onto the fisheye image."""
        d = self._rays(hfov, yaw, pitch, W, H)
        edge = np.concatenate(
            [d[0, :: W // n], d[:: H // n, -1], d[-1, :: -(W // n)], d[:: -(H // n), 0]]
        )
        x, y = self._to_image(edge)
        return np.stack([x, y], -1)


def lens_switch():
    """One fisheye photograph re-rendered as a phone's wide, normal and telephoto cameras."""
    fe = Fisheye()
    stops = [
        (110.0, "wide (0.5x): 110 deg across"),
        (70.0, "normal (1x): 70 deg"),
        (25.0, "telephoto (3x): 25 deg"),
    ]
    frames = []
    for k, (hfov, text) in enumerate(stops):
        for _ in range(12):
            f = fe.view(hfov)
            label(f, text, (12, 32), 0.7)
            frames.append(f)
        if k + 1 < len(stops):
            nxt = stops[k + 1][0]
            for i in range(10):
                frames.append(fe.view(hfov + (nxt - hfov) * EASE((i + 1) / 10)))
    raw = cv2.resize(fe.img, (400, 400), interpolation=cv2.INTER_AREA)
    canvas = np.full((400, 600, 3), BG, np.uint8)
    canvas[:, 100:500] = raw
    label(canvas, "the photo as taken: fisheye, 180 deg", (12, 32), 0.7)
    frames += [canvas] * 14
    return frames, 10


def fisheye_pan():
    """A virtual camera sweeping across the fisheye; its outline left, its view right."""
    fe = Fisheye()
    small = 400
    scale = small / fe.img.shape[1]
    base = cv2.resize(fe.img, (small, small), interpolation=cv2.INTER_AREA)
    path = [(-40 + 80 * EASE(i / 29), 10.0) for i in range(30)]
    path += [(40.0, 10 - 40 * EASE(i / 14)) for i in range(15)]
    path += [(40 - 40 * EASE(i / 14), -30 + 40 * EASE(i / 14)) for i in range(15)]
    frames = []
    for yaw, pitch in path:
        left = base.copy()
        fp = np.rint(fe.footprint(80, yaw, pitch) * scale).astype(np.int32)
        cv2.polylines(left, [fp], True, AMBER, 2, cv2.LINE_AA)
        right = fe.view(80, yaw, pitch, W=600, H=400)
        label(left, "fisheye", (10, 28), 0.6)
        label(right, "what the outlined camera sees", (10, 28), 0.6)
        frames.append(np.hstack([left, np.full((400, 6, 3), 40, np.uint8), right]))
    return frames, 10


# --- the match frame ----------------------------------------------------------------


def focus_sweep(width: int = 640):
    """A band of sharp focus sweeping the court, blur growing with distance from it.

    This is how a phone's miniature filter fakes depth of field: by image row, not by
    distance. On a floor seen from above the two are close, which is why the trick works.
    """
    cap = cv2.VideoCapture(str(CLIP))
    cap.set(cv2.CAP_PROP_POS_FRAMES, 60)
    ok, f = cap.read()
    f = cv2.resize(f, (width, round(f.shape[0] * width / f.shape[1])), interpolation=cv2.INTER_AREA)
    h = f.shape[0]
    sig = [0.0, 1.2, 2.5, 4.0, 6.0]
    levels = [f.astype(np.float32)] + [
        cv2.GaussianBlur(f, (0, 0), s).astype(np.float32) for s in sig[1:]
    ]
    centres = [0.30 * h + (0.92 - 0.30) * h * EASE(i / 27) for i in range(28)]
    centres = centres + [centres[-1]] * 6 + centres[::-1] + [centres[0]] * 6
    frames = []
    for c in centres:
        lvl = np.clip(np.abs(np.arange(h) - c) / h * 12, 0, len(levels) - 1)
        lo = np.floor(lvl).astype(int)
        hi = np.minimum(lo + 1, len(levels) - 1)
        t = (lvl - lo)[:, None, None]
        out = np.empty_like(levels[0])
        for y in range(h):
            out[y] = levels[lo[y]][y] * (1 - t[y]) + levels[hi[y]][y] * t[y]
        img = out.astype(np.uint8)
        for x0 in (0, img.shape[1] - 14):
            cv2.rectangle(img, (x0, int(c) - 2), (x0 + 14, int(c) + 2), AMBER, -1)
        label(img, "sharp only between the orange marks", (12, 30), 0.6)
        frames.append(img)
    return frames, 12


# --- all ----------------------------------------------------------------------------

#: name -> (function, output width px, libwebp quality)
TABLE = {
    "lens-switch": (lens_switch, 600, 50),
    "ar-cube": (ar_cube, 560, 50),
    "focus-sweep": (focus_sweep, 600, 50),
    "undistort-morph": (undistort_morph, 560, 50),
    "fisheye-pan": (fisheye_pan, 700, 50),
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
