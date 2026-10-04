"""Figures and cover backgrounds for unit 4.1 (lessons 2 to 4 and the hub), all drawn from
the calibration and PnP fits on OpenCV's boards."""

from __future__ import annotations

import cv2
import matplotlib
import numpy as np
from loguru import logger

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from fundamentals_lab.camgeom.experiments import Boards, _centre  # noqa: E402
from fundamentals_lab.config import OUTPUT_DIR  # noqa: E402

FIG_DIR = OUTPUT_DIR / "figures" / "camgeom"
COVER_DIR = OUTPUT_DIR / "covers"
GREEN, RED, AMBER, BLUE, WHITE = (
    (80, 220, 120),
    (70, 70, 235),
    (40, 180, 250),
    (235, 140, 40),
    (245, 245, 245),
)
BG = (24, 14, 11)
BG_HEX = "#0e1218"


def _photo(b: Boards, i: int = 0) -> np.ndarray:
    return cv2.imread(str(b.paths[i]))


def _dots(img, pts, colour, rad):
    for x, y in pts:
        cv2.circle(
            img,
            (round(float(x) * 4), round(float(y) * 4)),
            rad * 4,
            colour,
            -1,
            cv2.LINE_AA,
            shift=2,
        )


def plan(b: Boards, size=(640, 480)) -> np.ndarray:
    """The board in its own units: 9 x 6 inner corners, origin and axes."""
    w, h = size
    img = np.full((h, w, 3), 36, np.uint8)
    s, ox, oy = 52, 90, 70
    for j in range(7):
        for i in range(10):
            if (i + j) % 2 == 0:
                cv2.rectangle(
                    img,
                    (ox + (i - 1) * s, oy + (j - 1) * s),
                    (ox + i * s, oy + j * s),
                    (70, 70, 70),
                    -1,
                )
    for x in range(9):
        for y in range(6):
            cv2.circle(img, (ox + x * s, oy + y * s), 6, GREEN, -1, cv2.LINE_AA)
    cv2.arrowedLine(img, (ox, oy), (ox + 3 * s, oy), RED, 4, cv2.LINE_AA, tipLength=0.15)
    cv2.arrowedLine(img, (ox, oy), (ox, oy + 3 * s), GREEN, 4, cv2.LINE_AA, tipLength=0.15)
    cv2.putText(img, "X", (ox + 3 * s + 8, oy + 8), cv2.FONT_HERSHEY_SIMPLEX, 0.9, RED, 2)
    cv2.putText(img, "Y", (ox - 8, oy + 3 * s + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, GREEN, 2)
    cv2.putText(
        img, "(8, 0, 0)", (ox + 8 * s - 40, oy - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.6, WHITE, 2
    )
    cv2.circle(img, (ox + 8 * s, oy), 12, AMBER, 3, cv2.LINE_AA)
    return img


def from_above(b: Boards, highlight: int = 0, size=(640, 480)) -> np.ndarray:
    """Every camera centre in the board's frame, seen from above, in board squares."""
    w, h = size
    img = np.full((h, w, 3), 36, np.uint8)
    C = np.array([_centre(r, t) for r, t in zip(b.rvecs, b.tvecs, strict=True)])
    xs, zs = C[:, 0], C[:, 2]
    lo = np.array([min(xs.min(), -2) - 1, min(zs.min(), -2) - 1])
    hi = np.array([max(xs.max(), 10) + 1, max(zs.max(), 2) + 1])
    scale = min((w - 80) / (hi[0] - lo[0]), (h - 80) / (hi[1] - lo[1]))

    def P(x, z):
        return int(40 + (x - lo[0]) * scale), int(h - 40 - (z - lo[1]) * scale)

    cv2.line(img, P(0, 0), P(8, 0), WHITE, 4)
    cv2.putText(
        img, "the board", (P(0, 0)[0], P(0, 0)[1] + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, WHITE, 2
    )
    for i, (x, z) in enumerate(zip(xs, zs, strict=True)):
        cv2.circle(
            img,
            P(x, z),
            11 if i == highlight else 7,
            AMBER if i == highlight else (150, 150, 150),
            -1,
            cv2.LINE_AA,
        )
    return img


def projected(b: Boards) -> np.ndarray:
    img = (_photo(b) * 0.6).astype(np.uint8)
    seen = b.img[0].reshape(-1, 2)
    p, _ = cv2.projectPoints(b.obj[0], b.rvecs[0], b.tvecs[0], b.K, b.dist)
    _dots(img, seen, GREEN, 5)
    _dots(img, p.reshape(-1, 2), RED, 2)
    return img


def residuals(b: Boards, gain: float = 50) -> np.ndarray:
    img = (_photo(b) * 0.5).astype(np.uint8)
    seen = b.img[0].reshape(-1, 2)
    p, _ = cv2.projectPoints(b.obj[0], b.rvecs[0], b.tvecs[0], b.K, b.dist)
    for a, q in zip(seen, p.reshape(-1, 2), strict=True):
        end = a + gain * (q - a)
        cv2.arrowedLine(
            img,
            tuple(np.rint(a).astype(int)),
            tuple(np.rint(end).astype(int)),
            AMBER,
            2,
            cv2.LINE_AA,
            tipLength=0.3,
        )
    return img


def axes(b: Boards, i: int = 0) -> np.ndarray:
    img = _photo(b, i)
    ax = np.float32([[0, 0, 0], [3, 0, 0], [0, 3, 0], [0, 0, -3]])
    p, _ = cv2.projectPoints(ax, b.rvecs[i], b.tvecs[i], b.K, b.dist)
    p = np.rint(p.reshape(-1, 2)).astype(int)
    for q, colour in zip(p[1:], (RED, GREEN, BLUE), strict=True):
        cv2.arrowedLine(img, tuple(p[0]), tuple(q), colour, 4, cv2.LINE_AA, tipLength=0.15)
    return img


def cube(b: Boards, i: int = 0) -> np.ndarray:
    img = _photo(b, i)
    c3 = np.float32(
        [[3, 1, 0], [6, 1, 0], [6, 4, 0], [3, 4, 0], [3, 1, -3], [6, 1, -3], [6, 4, -3], [3, 4, -3]]
    )
    c, _ = cv2.projectPoints(c3, b.rvecs[i], b.tvecs[i], b.K, b.dist)
    c = np.rint(c.reshape(-1, 2)).astype(np.int32)
    over = img.copy()
    cv2.fillPoly(over, [c[:4]], GREEN)
    img = cv2.addWeighted(over, 0.55, img, 0.45, 0)
    for k in range(4):
        cv2.line(img, tuple(c[k]), tuple(c[k + 4]), AMBER, 3, cv2.LINE_AA)
    cv2.polylines(img, [c[4:]], True, RED, 3, cv2.LINE_AA)
    return img


def _grid(panels, name):
    fig, axs = plt.subplots(3, 2, figsize=(8, 10.3), dpi=110, facecolor=BG_HEX)
    for ax, (img, title) in zip(axs.ravel(), panels, strict=True):
        ax.set_facecolor(BG_HEX)
        ax.imshow(img[..., ::-1], aspect="auto")
        ax.set_title(title, color="#e2e8f0", fontsize=19, loc="left", pad=6)
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_color("#334155")
    fig.tight_layout(pad=0.6, h_pad=1.2, w_pad=0.8)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    p = FIG_DIR / f"{name}.png"
    fig.savefig(p, facecolor=BG_HEX)
    plt.close(fig)
    logger.info(f"wrote {p}")


def _save(img, name, folder=FIG_DIR):
    folder.mkdir(parents=True, exist_ok=True)
    p = folder / f"{name}.png"
    cv2.imwrite(str(p), img)
    logger.info(f"wrote {p}")


def cover_panel(img: np.ndarray, right_fraction: float = 0.62, size=(1600, 900)) -> np.ndarray:
    W, H = size
    canvas = np.full((H, W, 3), BG, np.uint8)
    pw = round(W * right_fraction)
    ih, iw = img.shape[:2]
    scale = max(pw / iw, H / ih)
    r = cv2.resize(img, (round(iw * scale), round(ih * scale)), interpolation=cv2.INTER_CUBIC)
    y0, x0 = (r.shape[0] - H) // 2, (r.shape[1] - pw) // 2
    canvas[:, W - pw :] = r[y0 : y0 + H, x0 : x0 + pw]
    fade = np.linspace(0, 1, 120)[None, :, None]
    canvas[:, W - pw : W - pw + 120] = (
        canvas[:, W - pw : W - pw + 120] * fade + np.array(BG) * (1 - fade)
    ).astype(np.uint8)
    return canvas


def render_all() -> None:
    b = Boards()
    _grid(
        [
            (plan(b), "1  Points on the board"),
            (from_above(b), "2  Where the camera stands"),
            (projected(b), "3–5  Into pixels"),
            (residuals(b), "6  Calibrate: misses ×50"),
            (axes(b), "6  PnP: the pose"),
            (cube(b), "7  Use: a cube in 3-D"),
        ],
        "hub-pipeline",
    )
    _save(projected(b), "l2-projected")
    _save(residuals(b), "l3-residuals")
    _save(from_above(b), "l4-from-above")
    side = np.hstack([axes(b), np.full((480, 8, 3), 40, np.uint8), from_above(b)])
    _save(side, "l4-axes-and-cameras")
    # the hub: the finished result, the pose drawn as axes with a cube standing on the board
    both = cube(b)
    ax = np.float32([[0, 0, 0], [3, 0, 0], [0, 3, 0], [0, 0, -3]])
    pa, _ = cv2.projectPoints(ax, b.rvecs[0], b.tvecs[0], b.K, b.dist)
    pa = np.rint(pa.reshape(-1, 2)).astype(int)
    for q, colour in zip(pa[1:], (RED, GREEN, BLUE), strict=True):
        cv2.arrowedLine(both, tuple(pa[0]), tuple(q), colour, 4, cv2.LINE_AA, tipLength=0.15)
    _save(cover_panel(both), "camera-geometry", COVER_DIR)
    _save(cover_panel(cube(b)), "camera-models-pinhole-distortion-calibration-pnp", COVER_DIR)
    _save(cover_panel(projected(b)), "what-is-camera-calibration", COVER_DIR)
    _save(cover_panel(axes(b)), "what-is-pnp-camera-pose", COVER_DIR)
