"""Animated figures for unit 3.5: every practical example the posts open with.

Each function returns a list of BGR frames and a frame rate; `render_all` writes them
as animated WebP through ffmpeg (libwebp). Everything is drawn from real data: the
match clip and its homography, the players' ankle keypoints (from CondadosAI/sportcv,
exported to `output/alignment_players_clip.csv`), the OpenCV calibration boards, the
Barcelona pair and a CC0 photograph of a sheet of paper.
"""

from __future__ import annotations

import csv
import shutil
import subprocess
import tempfile
from pathlib import Path

import cv2
import numpy as np
from loguru import logger

from fundamentals_lab.alignment import court, transforms, warp
from fundamentals_lab.alignment.experiments import Scene
from fundamentals_lab.config import COURT_LENGTH_M as L
from fundamentals_lab.config import COURT_WIDTH_M as W
from fundamentals_lab.config import DATA_DIR, OUTPUT_DIR

MEDIA_DIR = OUTPUT_DIR / "figures" / "alignment" / "media"
CLIP = DATA_DIR / "pickleball" / "clip_44850_10s.mp4"
CLIP_FIRST_FRAME = 44851  # clip index 0 is match frame 44851 (checked against frame 45000)
DOCUMENT = DATA_DIR / "wild-document" / "stationery_bench_unsplash.jpg"
#: The card's corners on the 1200 px-wide image, tapped by eye to ~10 px the way a
#: scanner app asks for them: top-left, top-right, bottom-right, bottom-left. Each side
#: is then refitted to Canny edges and the corners recomputed as intersections.
DOCUMENT_SEED = ((348.0, 386.0), (624.0, 202.0), (878.0, 536.0), (612.0, 740.0))
PLAYER_COLOURS = {
    "p1": (80, 80, 240),
    "p2": (60, 200, 250),
    "p3": (120, 220, 90),
    "p4": (240, 170, 80),
}
BG = (24, 18, 14)
COURT_BLUE = (120, 70, 35)
WHITE = (245, 245, 245)
GREEN = (80, 220, 120)
RED = (70, 70, 235)


# --- helpers -----------------------------------------------------------------------


def label(img, text, org, scale=0.7, colour=WHITE, box=True):
    font = cv2.FONT_HERSHEY_SIMPLEX
    (w, h), base = cv2.getTextSize(text, font, scale, 2)
    x, y = org
    if box:
        cv2.rectangle(img, (x - 6, y - h - 8), (x + w + 6, y + base + 4), (20, 20, 20), -1)
    cv2.putText(img, text, (x, y), font, scale, colour, 2, cv2.LINE_AA)


def encode(frames, fps, path: Path, width: int | None = None, quality: int = 50) -> Path:
    """Write frames as an animated WebP (libwebp, lossy), looping forever."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp())
    try:
        for i, f in enumerate(frames):
            cv2.imwrite(str(tmp / f"{i:05d}.png"), f)
        vf = f"scale={width}:-2:flags=lanczos" if width else "scale=trunc(iw/2)*2:trunc(ih/2)*2"
        subprocess.run(
            [
                "ffmpeg",
                "-loglevel",
                "error",
                "-y",
                "-framerate",
                str(fps),
                "-i",
                str(tmp / "%05d.png"),
                "-vf",
                vf,
                "-c:v",
                "libwebp",
                "-q:v",
                str(quality),
                "-loop",
                "0",
                str(path),
            ],
            check=True,
        )
    finally:
        shutil.rmtree(tmp)
    logger.info(f"wrote {path} ({path.stat().st_size / 1e3:.0f} kB, {len(frames)} frames)")
    return path


def court_diagram(px_per_m: float, margin: float = 1.0):
    """Top view of the court, near baseline at the bottom. Returns (image, court->pixel)."""
    w = round((W + 2 * margin) * px_per_m)
    h = round((L + 2 * margin) * px_per_m)
    img = np.full((h, w, 3), COURT_BLUE, np.uint8)
    H = np.array(
        [[0, px_per_m, margin * px_per_m], [-px_per_m, 0, (L + margin) * px_per_m], [0, 0, 1.0]]
    )
    for p, q in court.LINES.values():
        a, b = transforms.apply(H, np.array([p, q]))
        cv2.line(
            img, tuple(np.rint(a).astype(int)), tuple(np.rint(b).astype(int)), WHITE, 2, cv2.LINE_AA
        )
    a, b = transforms.apply(H, np.array([[L / 2, -0.3], [L / 2, W + 0.3]]))
    cv2.line(
        img,
        tuple(np.rint(a).astype(int)),
        tuple(np.rint(b).astype(int)),
        (30, 30, 30),
        3,
        cv2.LINE_AA,
    )
    return img, H


def clip_frames(step: int = 3, max_frames: int = 180):
    cap = cv2.VideoCapture(str(CLIP))
    i = 0
    while i < max_frames:
        ok, f = cap.read()
        if not ok:
            break
        if i % step == 0:
            yield CLIP_FIRST_FRAME + i, f
        i += 1


def players_track():
    """Ankle midpoints per player, linearly interpolated to every frame of the clip."""
    with open(OUTPUT_DIR / "alignment_players_clip.csv") as fh:
        rows = list(csv.DictReader(fh))
    tracks: dict[str, dict[int, tuple[float, float]]] = {}
    for r in rows:
        if float(r["ankle_score"]) < 0.1:
            continue
        tracks.setdefault(r["player"], {})[int(r["frame"])] = (
            float(r["ankle_x"]),
            float(r["ankle_y"]),
        )
    out = {}
    for p, pts in tracks.items():
        fr = np.array(sorted(pts))
        xy = np.array([pts[f] for f in fr])
        out[p] = (fr, xy)
    return out


def track_at(track, frame, max_gap=30):
    fr, xy = track
    if frame < fr[0] or frame > fr[-1]:
        return None
    j = np.searchsorted(fr, frame)
    if j < len(fr) and fr[j] == frame:
        return xy[j]
    if j == 0 or fr[j] - fr[j - 1] > max_gap:
        return None
    t = (frame - fr[j - 1]) / (fr[j] - fr[j - 1])
    return xy[j - 1] * (1 - t) + xy[j] * t


# --- 1. the sports minimap ---------------------------------------------------------------


def minimap(s: Scene):
    tracks = players_track()
    # 3 m around the court: players stand well behind the baselines to receive
    diag, H_c2d = court_diagram(30, margin=3.0)
    frames, trails = [], {p: [] for p in tracks}
    for frame, f in clip_frames():
        left = f.copy()
        right = diag.copy()
        for p, tr in tracks.items():
            xy = track_at(tr, frame)
            if xy is None:
                continue
            c = transforms.apply(s.H_img2court, xy[None])[0]
            d = transforms.apply(H_c2d, c[None])[0]
            trails[p] = (trails[p] + [d])[-15:]
            cv2.circle(left, tuple(np.rint(xy).astype(int)), 14, PLAYER_COLOURS[p], 4, cv2.LINE_AA)
            for a, b in zip(trails[p][:-1], trails[p][1:], strict=False):
                cv2.line(
                    right,
                    tuple(np.rint(a).astype(int)),
                    tuple(np.rint(b).astype(int)),
                    PLAYER_COLOURS[p],
                    2,
                    cv2.LINE_AA,
                )
            cv2.circle(right, tuple(np.rint(d).astype(int)), 8, PLAYER_COLOURS[p], -1, cv2.LINE_AA)
            cv2.circle(right, tuple(np.rint(d).astype(int)), 8, WHITE, 1, cv2.LINE_AA)
        left = cv2.resize(left, (960, 540), interpolation=cv2.INTER_AREA)
        label(left, "camera", (16, 36))
        rh = 540
        right = cv2.resize(
            right, (round(right.shape[1] * rh / right.shape[0]), rh), interpolation=cv2.INTER_AREA
        )
        label(right, "from above", (10, 30), 0.6)
        frames.append(np.hstack([left, np.full((540, 8, 3), 255, np.uint8), right]))
    return frames, 10


# --- 2. a virtual advert on the court -------------------------------------------------


def logo(w=600, h=200):
    img = np.full((h, w, 3), (140, 60, 70), np.uint8)
    cv2.rectangle(img, (6, 6), (w - 7, h - 7), WHITE, 6)
    font = cv2.FONT_HERSHEY_DUPLEX
    text = "condados.ai"
    (tw, th), _ = cv2.getTextSize(text, font, 2.6, 5)
    cv2.putText(img, text, ((w - tw) // 2, (h + th) // 2), font, 2.6, WHITE, 5, cv2.LINE_AA)
    return img


def virtual_ad(s: Scene):
    lg = logo()
    lh, lw = lg.shape[:2]
    # a 3 m x 1 m patch in the near-right service box, long side across the court
    x0, x1, y0, y1 = 1.4, 2.4, 3.35, 5.8
    court_quad = np.array(
        [[x1, y0], [x1, y1], [x0, y1], [x0, y0]]
    )  # top-left, top-right, br, bl in logo order
    img_quad = transforms.apply(s.H_court2img, court_quad).astype(np.float32)
    Hl = cv2.getPerspectiveTransform(np.float32([[0, 0], [lw, 0], [lw, lh], [0, lh]]), img_quad)
    shape = s.plate.shape[:2][::-1]
    warped = cv2.warpPerspective(lg, Hl, shape, flags=cv2.INTER_LINEAR)
    alpha = (
        cv2.warpPerspective(
            np.full((lh, lw), 255, np.uint8), Hl, shape, flags=cv2.INTER_LINEAR
        ).astype(np.float32)
        / 255
        * 0.85
    )
    plate = s.plate.astype(np.int16)
    frames = []
    for _, f in clip_frames():
        fg = (np.abs(f.astype(np.int16) - plate).max(axis=2) > 40).astype(np.uint8)
        fg = cv2.dilate(
            cv2.morphologyEx(fg, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)),
            np.ones((7, 7), np.uint8),
        )
        a = alpha * (1 - fg)
        out = (f * (1 - a[..., None]) + warped * a[..., None]).astype(np.uint8)
        frames.append(cv2.resize(out, (960, 540), interpolation=cv2.INTER_AREA))
    return frames, 10


# --- 3. the document scanner ---------------------------------------------------------------


def document_corners(img):
    """Refine the tapped corners: fit each side to edge pixels, intersect neighbours."""
    from fundamentals_lab.alignment import lines

    grey = cv2.GaussianBlur(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (5, 5), 1.2)
    edges = cv2.Canny(grey, 30, 90)
    ys, xs = np.nonzero(edges)
    pix = np.stack([xs, ys], 1).astype(np.float64)
    seed = np.array(DOCUMENT_SEED)
    sides = []
    for k in range(4):
        p, q = seed[k], seed[(k + 1) % 4]
        d = q - p
        # keep the middle 80% of the side, away from the corners
        pts, t = lines.band(pix, p, q, half_width=10.0)
        pts = pts[(t > 0.1) & (t < 0.9)]
        sides.append(lines.fit_line(pts).line if len(pts) > 20 else court.homogeneous_line(p, q))
        del d
    return np.array(
        [court.dehomogenise(court.meet(sides[(k - 1) % 4], sides[k])) for k in range(4)]
    )


def document_scan():
    img = cv2.imread(str(DOCUMENT))
    scale = 1200 / img.shape[1]
    img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    src = document_corners(img)
    h, w = img.shape[:2]
    # target: a rectangle whose sides are the mean lengths of opposite sides in the
    # photo (a demo's estimate of the card's shape, not a measurement of it), centred
    top, right = np.linalg.norm(src[1] - src[0]), np.linalg.norm(src[2] - src[1])
    bottom, left = np.linalg.norm(src[2] - src[3]), np.linalg.norm(src[3] - src[0])
    aspect = (top + bottom) / (left + right)
    th = 0.8 * h
    tw = th * aspect
    cx, cy = w / 2, h / 2
    dst = np.array(
        [
            [cx - tw / 2, cy - th / 2],
            [cx + tw / 2, cy - th / 2],
            [cx + tw / 2, cy + th / 2],
            [cx - tw / 2, cy + th / 2],
        ]
    )
    frames = []
    ease = lambda t: 0.5 - 0.5 * np.cos(np.pi * t)  # noqa: E731
    for _ in range(8):
        f = img.copy()
        cv2.polylines(f, [np.rint(src).astype(np.int32)], True, GREEN, 3, cv2.LINE_AA)
        for p in src:
            cv2.circle(f, tuple(np.rint(p).astype(int)), 9, GREEN, -1, cv2.LINE_AA)
        label(f, "1. find the four corners", (24, 60), 1.4)
        frames.append(f)
    for k in range(36):
        t = ease(k / 35)
        cur = src * (1 - t) + dst * t
        Hk = transforms.dlt(src, cur)
        f = cv2.warpPerspective(img, Hk, (w, h), borderValue=BG)
        cv2.polylines(f, [np.rint(cur).astype(np.int32)], True, GREEN, 3, cv2.LINE_AA)
        label(f, "2. one homography moves them to a rectangle", (24, 60), 1.4)
        frames.append(f)
    Hf = transforms.dlt(src, dst)
    final = cv2.warpPerspective(img, Hf, (w, h), borderValue=BG)
    for _ in range(12):
        f = final.copy()
        label(f, "3. the sheet, flattened", (24, 60), 1.4)
        frames.append(f)
    return frames, 12


# --- 4. the panorama -------------------------------------------------------------------------


def panorama_build():
    from fundamentals_lab.alignment.wild import PANO_DIR, _pyramid_blend

    a = cv2.imread(str(PANO_DIR / "BarcelonaHarbour1.jpg"))
    b = cv2.imread(str(PANO_DIR / "BarcelonaHarbour2.jpg"))
    sc = 800 / a.shape[1]
    a = cv2.resize(a, None, fx=sc, fy=sc, interpolation=cv2.INTER_AREA)
    b = cv2.resize(b, None, fx=sc, fy=sc, interpolation=cv2.INTER_AREA)
    sift = cv2.SIFT_create()
    ka, da = sift.detectAndCompute(cv2.cvtColor(a, cv2.COLOR_BGR2GRAY), None)
    kb, db = sift.detectAndCompute(cv2.cvtColor(b, cv2.COLOR_BGR2GRAY), None)
    good = [m for m, n in cv2.BFMatcher().knnMatch(db, da, k=2) if m.distance < 0.75 * n.distance]
    src = np.float64([kb[m.queryIdx].pt for m in good])
    dst = np.float64([ka[m.trainIdx].pt for m in good])
    H, inl = cv2.findHomography(src, dst, cv2.RANSAC, 2.0)
    hb, wb = b.shape[:2]
    ha, wa = a.shape[:2]
    corners_b = np.array([[0, 0], [wb, 0], [wb, hb], [0, hb]], float)
    end = transforms.apply(H, corners_b)
    ymin = int(np.floor(min(0, end[:, 1].min()))) - 10
    ymax = int(np.ceil(max(ha, end[:, 1].max()))) + 10
    xmax = int(np.ceil(max(wa + wb * 0.6, end[:, 0].max()))) + 10
    T = np.array([[1, 0, 0], [0, 1, -ymin], [0, 0, 1.0]])
    size = (xmax, ymax - ymin)
    return a, b, H, T, size, corners_b, end, _pyramid_blend


def panorama_anim():
    a, b, H, T, size, corners_b, end, blend = panorama_build()
    wa = a.shape[1]
    start = corners_b + [wa * 0.6, 0]  # b laid beside a, untransformed
    A = cv2.warpPerspective(a, T, size)
    ma = cv2.warpPerspective(np.ones(a.shape[:2], np.uint8), T, size) > 0
    frames = []
    ease = lambda t: 0.5 - 0.5 * np.cos(np.pi * t)  # noqa: E731
    for k in range(40):
        t = ease(min(1.0, k / 32))
        cur = start * (1 - t) + end * t
        Hk = T @ transforms.dlt(corners_b, cur)
        B = cv2.warpPerspective(b, Hk, size)
        mb = cv2.warpPerspective(np.ones(b.shape[:2], np.uint8), Hk, size) > 0
        f = A.copy()
        f[mb] = B[mb]
        f[~(ma | mb)] = BG
        cv2.polylines(
            f, [np.rint(transforms.apply(T, cur)).astype(np.int32)], True, GREEN, 2, cv2.LINE_AA
        )
        label(f, "the second photo, pulled onto the first by one homography", (18, 50), 1.2)
        frames.append(f)
    Hfull = T @ H
    B = cv2.warpPerspective(b, Hfull, size)
    mb = cv2.warpPerspective(np.ones(b.shape[:2], np.uint8), Hfull, size) > 0
    both = ma & mb
    xs = np.where(both.any(axis=0))[0]
    seam = int((xs.min() + xs.max()) / 2)
    cut = (ma & ((np.arange(size[0])[None, :] < seam) | ~mb)).astype(np.float32)
    hard = (A * cut[..., None] + B * (1 - cut[..., None])).astype(np.uint8)
    multi = blend(A, B, cut)
    hard[~(ma | mb)] = BG
    multi[~(ma | mb)] = BG
    for _ in range(10):
        f = hard.copy()
        label(f, "hard cut: the exposure step shows", (18, 50), 1.2)
        frames.append(f)
    for k in range(10):
        t = k / 9
        f = (hard * (1 - t) + multi * t).astype(np.uint8)
        label(f, "multiband blend: the seam disappears", (18, 50), 1.2)
        frames.append(f)
    for _ in range(10):
        f = multi.copy()
        label(f, "multiband blend: the seam disappears", (18, 50), 1.2)
        frames.append(f)
    return frames, 12


# --- 5. camera calibration boards --------------------------------------------------------------


def calibration():
    from fundamentals_lab.formation.dataset import fetch_opencv_calibration

    paths = fetch_opencv_calibration()
    frames = []
    cols, rows = 9, 6
    board = np.array([[x, y] for y in range(rows) for x in range(cols)], float)
    for p in paths:
        g = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        ok, corners = cv2.findChessboardCorners(g, (cols, rows))
        if not ok:
            continue
        corners = cv2.cornerSubPix(
            g,
            corners,
            (11, 11),
            (-1, -1),
            (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.01),
        )
        img_pts = corners.reshape(-1, 2).astype(np.float64)
        H = transforms.dlt(board, img_pts)
        f = cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)
        for x in np.arange(-1, cols + 1):
            a, b = transforms.apply(H, np.array([[x, -1.0], [x, rows]]))
            cv2.line(
                f,
                tuple(np.rint(a).astype(int)),
                tuple(np.rint(b).astype(int)),
                GREEN,
                1,
                cv2.LINE_AA,
            )
        for y in np.arange(-1, rows + 1):
            a, b = transforms.apply(H, np.array([[-1.0, y], [cols, y]]))
            cv2.line(
                f,
                tuple(np.rint(a).astype(int)),
                tuple(np.rint(b).astype(int)),
                GREEN,
                1,
                cv2.LINE_AA,
            )
        for q in img_pts:
            cv2.circle(f, tuple(np.rint(q).astype(int)), 3, RED, -1, cv2.LINE_AA)
        label(f, "board plane -> image: one homography per photo", (10, 26), 0.5)
        frames.append(f)
    return frames, 1.5


# --- 6. vanishing point ---------------------------------------------------------------------


def vanishing_point(s: Scene):
    left, right, top, bottom = 120, 480, 80, 20
    h, w = s.frame.shape[:2]
    base = np.full((h + top + bottom, w + left + right, 3), 36, np.uint8)
    base[top : top + h, left : left + w] = s.frame
    cv2.rectangle(base, (left, top), (left + w - 1, top + h - 1), WHITE, 2)
    off = np.array([left, top], float)
    L_ = {n: f.line for n, f in s.fit.lines.items()}
    vp = court.dehomogenise(court.meet(L_["side_left"], L_["side_right"]))
    lm = s.fit.landmarks_img
    starts = {"side_left": lm["NBL"], "side_right": lm["NBR"], "near_centre": lm["NBC"]}
    ends = {"side_left": lm["NKL"], "side_right": lm["NKR"], "near_centre": lm["NKC"]}
    frames = []
    for k in range(40):
        t = min(1.0, k / 30)
        f = base.copy()
        for n in starts:
            a = starts[n] + off
            e = ends[n] + (vp - ends[n]) * t + off
            cv2.line(
                f,
                tuple(np.rint(a).astype(int)),
                tuple(np.rint(e).astype(int)),
                GREEN,
                4,
                cv2.LINE_AA,
            )
        if t >= 1:
            cv2.circle(f, tuple(np.rint(vp + off).astype(int)), 20, WHITE, 5, cv2.LINE_AA)
            label(f, "parallel on the court, meeting off the frame", (40, 60), 1.4)
        frames.append(cv2.resize(f, None, fx=0.4, fy=0.4, interpolation=cv2.INTER_AREA))
    frames += [frames[-1]] * 10
    return frames, 12


# --- 7. rotate, scale, shear ---------------------------------------------------------------


def _diagram_through(M3, size=520, px_per_m=17):
    img = np.full((size, size, 3), BG, np.uint8)
    c = np.array([size / 2, size / 2])
    centre = np.array([L / 2, W / 2])

    def to_px(pts):
        q = transforms.apply(M3, (np.asarray(pts, float) - centre)[:, ::-1] * [1, -1])
        return q * px_per_m + c

    for p, q in court.LINES.values():
        a, b = to_px([p, q])
        cv2.line(
            img, tuple(np.rint(a).astype(int)), tuple(np.rint(b).astype(int)), WHITE, 2, cv2.LINE_AA
        )
    for side in ("side_left", "side_right"):
        a, b = to_px(court.LINES[side])
        cv2.line(
            img, tuple(np.rint(a).astype(int)), tuple(np.rint(b).astype(int)), GREEN, 3, cv2.LINE_AA
        )
    return img


def linear_morph():
    r = np.deg2rad(30)
    keys = [
        ("identity", np.eye(2)),
        ("rotate 30 deg", np.array([[np.cos(r), -np.sin(r)], [np.sin(r), np.cos(r)]])),
        ("scale x1.3", 1.3 * np.array([[np.cos(r), -np.sin(r)], [np.sin(r), np.cos(r)]])),
        (
            "shear",
            np.array([[1, 0.7], [0, 1]])
            @ (1.3 * np.array([[np.cos(r), -np.sin(r)], [np.sin(r), np.cos(r)]])),
        ),
        ("identity", np.eye(2)),
    ]
    frames = []
    for (n0, A0), (n1, A1) in zip(keys[:-1], keys[1:], strict=True):
        for k in range(22):
            t = 0.5 - 0.5 * np.cos(np.pi * min(1, k / 16))
            A = A0 * (1 - t) + A1 * t
            M3 = np.eye(3)
            M3[:2, :2] = A
            f = _diagram_through(M3)
            label(f, f"{n1 if t > 0.5 else n0}", (14, 34), 0.8)
            label(
                f,
                f"[{A[0, 0]:+.2f} {A[0, 1]:+.2f}; {A[1, 0]:+.2f} {A[1, 1]:+.2f}]"
                f"  det {np.linalg.det(A):.2f}",
                (14, 500),
                0.55,
            )
            frames.append(f)
    return frames, 12


# --- 8. affine to perspective ---------------------------------------------------------------


def affine_to_perspective():
    frames = []
    for k in range(60):
        t = 0.5 - 0.5 * np.cos(np.pi * min(1, (k % 60) / 40))
        g = 0.0
        h = 0.085 * t
        M3 = np.array([[1, 0.25, 0], [0, 0.8, 0], [g, h, 1.0]])
        f = _diagram_through(M3, px_per_m=17)
        label(
            f,
            "bottom row (0, 0, 1): parallels stay parallel"
            if t < 0.05
            else f"bottom row (0, {h:.3f}, 1): parallels meet",
            (14, 34),
            0.6,
        )
        frames.append(f)
    frames += frames[::-1][:20]
    return frames, 12


# --- 9. forward versus backward mapping --------------------------------------------------------


def forward_vs_backward(s: Scene):
    H_i2t, size = warp.img2top(s.H_img2court)
    tw, th = size
    back_full, seen = warp.backward(s.frame, H_i2t, size)
    fh, fw = s.frame.shape[:2]
    ys, xs = np.mgrid[0:fh, 0:fw]
    dst = transforms.apply(H_i2t, np.stack([xs.ravel(), ys.ravel()], 1).astype(np.float64))
    dx, dy = np.rint(dst[:, 0]).astype(int), np.rint(dst[:, 1]).astype(int)
    ok = (dx >= 0) & (dx < tw) & (dy >= 0) & (dy < th)
    src_rows = ys.ravel()
    frames = []
    fwd = np.zeros((th, tw, 3), np.uint8)
    n = 30
    for k in range(1, n + 1):
        rmax = int(fh * k / n)
        sel = ok & (src_rows < rmax)
        fwd[:] = 0
        fwd[dy[sel], dx[sel]] = s.frame.reshape(-1, 3)[sel]
        f1 = fwd.copy()
        f1[seen & (f1.sum(2) == 0) & (k == n)] = (255, 0, 255)
        b = np.zeros_like(back_full)
        b[: int(th * k / n)] = back_full[: int(th * k / n)]
        pane = np.hstack([f1, np.full((th, 10, 3), 255, np.uint8), b])
        label(pane, "forward: push", (10, 30), 0.7)
        label(pane, "backward: pull", (tw + 20, 30), 0.7)
        frames.append(pane)
    frames += [frames[-1]] * 18
    return frames, 12


# --- run everything -------------------------------------------------------------------------


def render_all() -> dict[str, Path]:
    s = Scene()
    jobs = {
        "minimap": (lambda: minimap(s), 760),
        "virtual-ad": (lambda: virtual_ad(s), 640),
        "document-scan": (document_scan, 640),
        "panorama": (panorama_anim, 720),
        "calibration": (calibration, 560),
        "vanishing-point": (lambda: vanishing_point(s), 640),
        "linear-morph": (linear_morph, 420),
        "affine-to-perspective": (affine_to_perspective, 420),
        "forward-vs-backward": (lambda: forward_vs_backward(s), 520),
    }
    out = {}
    for name, (fn, width) in jobs.items():
        frames, fps = fn()
        out[name] = encode(frames, fps, MEDIA_DIR / f"{name}.webp", width)
    return out
