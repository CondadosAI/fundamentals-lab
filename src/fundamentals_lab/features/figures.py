"""Figures, lab data and cover backgrounds for unit 3.3, drawn from the runs themselves.

Points are the detectors' own output on the Barcelona pair at 960 px, matches are scored
against the ORB reference homography, and every chart is drawn from the arrays the
experiments measure. PNG here; the site copies them as WebP. Cover backgrounds
(1600x900, content weighted right, no text) go to output/covers/. Lab data (JSON) goes
to output/figures/sift/labs/.
"""

from __future__ import annotations

import json

import cv2
import matplotlib
import numpy as np
from loguru import logger

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from fundamentals_lab.config import (  # noqa: E402
    MATCH_TOL_PX,
    OUTPUT_DIR,
    RATIO,
    REPEAT_TOL_PX,
    TEMPLE_STEP_DEG,
)
from fundamentals_lab.features import experiments as X  # noqa: E402
from fundamentals_lab.features import scene  # noqa: E402
from fundamentals_lab.features import sift_scratch as S  # noqa: E402

FIG_DIR = OUTPUT_DIR / "figures" / "sift"
LAB_DIR = FIG_DIR / "labs"
COVER_DIR = OUTPUT_DIR / "covers"
BG = (24, 18, 14)
BG_HEX = "#0e1218"
GREEN = (80, 220, 120)
RED = (70, 70, 235)
AMBER = (40, 180, 250)
BLUE = (250, 165, 96)
MAGENTA = (220, 80, 220)
WHITE = (245, 245, 245)
GREY = (150, 150, 150)
FONT = cv2.FONT_HERSHEY_SIMPLEX

HUB = "sift-what-makes-a-point-matchable"
SLUGS = {
    1: "what-makes-a-good-keypoint",
    2: "scale-space-difference-of-gaussians",
    3: "sift-keypoint-detection",
    4: "sift-descriptor",
    5: "lowe-ratio-test",
}


def _save(img, name: str, d=FIG_DIR):
    d.mkdir(parents=True, exist_ok=True)
    p = d / f"{name}.png"
    cv2.imwrite(str(p), img)
    logger.info(f"wrote {p}")
    return p


def _dim(bgr, k=0.55):
    return (bgr * k + np.array(BG) * (1 - k)).astype(np.uint8)


def _bgr(gray):
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def label(img, text, org, scale=0.6, colour=WHITE):
    """Text on a dark box (an outline drawn thicker would not line up: OpenCV widens each
    glyph's advance with the stroke)."""
    thick = 2 if img.shape[1] > 1200 else 1
    (w, h), base = cv2.getTextSize(text, FONT, scale, thick)
    x, y = org
    cv2.rectangle(img, (x - 5, y - h - 6), (x + w + 5, y + base + 3), (20, 20, 20), -1)
    cv2.putText(img, text, (x, y), FONT, scale, colour, thick, cv2.LINE_AA)


def _pt(p, s=1.0, off=(0, 0)):
    return (int(round(p[0] * s + off[0])), int(round(p[1] * s + off[1])))


def _side_by_side(a, b, gap=10):
    h = max(a.shape[0], b.shape[0])
    pad = lambda x: np.vstack([x, np.full((h - x.shape[0], x.shape[1], 3), BG, np.uint8)])  # noqa: E731
    return np.hstack([pad(a), np.full((h, gap, 3), BG, np.uint8), pad(b)]), a.shape[1] + gap


def _matches_canvas(a_bgr, b_bgr, pa, pb, ok, n_wrong=None):
    canvas, off = _side_by_side(_dim(a_bgr, 0.8), _dim(b_bgr, 0.8))
    idx = np.arange(len(pa))
    wrong = idx[~ok] if n_wrong is None else idx[~ok][:n_wrong]
    for i in idx[ok]:
        cv2.line(canvas, _pt(pa[i]), _pt(pb[i], off=(off, 0)), GREEN, 1, cv2.LINE_AA)
    for i in wrong:
        cv2.line(canvas, _pt(pa[i]), _pt(pb[i], off=(off, 0)), RED, 1, cv2.LINE_AA)
    return canvas


def _real_matches():
    a, b = scene.pair()
    H, _ = scene.reference_homography()
    sift = scene.sift()
    ka, da = sift.compute(a, scene.sift_points(a))
    kb, db = sift.compute(b, scene.sift_points(b))
    kn = scene.knn(da, db)
    pa = np.array([ka[m.queryIdx].pt for m, _ in kn])
    pb = np.array([kb[m.trainIdx].pt for m, _ in kn])
    r = np.array([m.distance / n.distance for m, n in kn])
    ok = np.linalg.norm(scene.apply(H, pa) - pb, axis=1) <= MATCH_TOL_PX
    over = scene.inside(scene.apply(H, pa), b.shape)
    return pa, pb, r, ok, over, ka, kb


# --- hub ---------------------------------------------------------------------------


def hub_matches():
    a_bgr, b_bgr = scene.pair_bgr()
    pa, pb, r, ok, _over, _ka, _kb = _real_matches()
    keep = r < RATIO
    img = _matches_canvas(a_bgr, b_bgr, pa[keep], pb[keep], ok[keep])
    label(img, f"{int((keep & ok).sum())} correct", (16, 44), 1.2, GREEN)
    label(img, f"{int((keep & ~ok).sum())} wrong", (16, 88), 1.2, RED)
    _save(img, "hub-matches")


def _panel(img, title, w=640):
    s = w / img.shape[1]
    out = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    bar = np.full((40, w, 3), BG, np.uint8)
    label(bar, title, (10, 27), 0.62)
    return np.vstack([bar, out])


def hub_pipeline():
    """Six steps on a crop of photo A, 3 rows x 2 columns so it reads at 390 px."""
    a_bgr, b_bgr = scene.pair_bgr()
    a, _b = scene.pair()
    x0, y0, w, h = 300, 260, 360, 270  # the harbour front: cranes, hulls, windows
    crop = lambda img: img[y0 : y0 + h, x0 : x0 + w]  # noqa: E731
    run = S.detect(a)
    pyr = run.pyramid
    # 1. blur at several sizes
    blur = [
        cv2.resize(crop(cv2.GaussianBlur(a, (0, 0), s)), (w // 2, h // 2)) for s in (1, 3, 6, 12)
    ]
    p1 = np.vstack([np.hstack(blur[:2]), np.hstack(blur[2:])])
    p1 = _bgr(p1)
    # 2. DoG with extrema
    d = pyr.dog[1][2]
    d = d[y0 : y0 + h, x0 : x0 + w]
    v = np.clip(d / (np.abs(d).max() * 0.35), -1, 1)
    p2 = np.zeros((h, w, 3), np.uint8)
    p2[..., 2] = (np.clip(v, 0, 1) * 255).astype(np.uint8)
    p2[..., 0] = (np.clip(-v, 0, 1) * 255).astype(np.uint8)
    for e in S.raw_extrema(pyr):
        if e.octave == 1:
            x, y = e.c - x0, e.r - y0
            if 0 <= x < w and 0 <= y < h:
                cv2.circle(p2, (x, y), 2, WHITE, -1, cv2.LINE_AA)
    # 3. kept vs dropped
    p3 = _dim(crop(a_bgr), 0.6).copy()
    colours = {"drift": GREY, "contrast": BLUE, "edge": MAGENTA, "kept": GREEN}
    for e in S.raw_extrema(pyr):
        stage, f = S.refine(pyr, e)
        x, y = f.pt[0] - x0, f.pt[1] - y0
        if 0 <= x < w and 0 <= y < h:
            cv2.circle(p3, _pt((x, y)), 2, colours[stage], -1, cv2.LINE_AA)
    # 4. orientation
    p4 = _dim(crop(a_bgr), 0.6).copy()
    for k in run.keypoints():
        x, y = k.pt[0] - x0, k.pt[1] - y0
        if 0 <= x < w and 0 <= y < h and k.size > 4:
            rr = k.size / 2
            cv2.circle(p4, _pt((x, y)), int(rr), AMBER, 1, cv2.LINE_AA)
            t = np.radians(360 - k.angle)
            cv2.line(
                p4,
                _pt((x, y)),
                _pt((x + rr * np.cos(t), y - rr * np.sin(t))),
                AMBER,
                1,
                cv2.LINE_AA,
            )
    # 5. descriptor grid on the strongest keypoint in the crop
    p5 = _descriptor_view(a, run, (x0, y0, w, h))
    # 6. matches
    pa, pb, r, ok, _over, _ka, _kb = _real_matches()
    keep = r < RATIO
    p6 = _matches_canvas(a_bgr, b_bgr, pa[keep], pb[keep], ok[keep])
    titles = [
        "1. blur at many sizes",
        "2. subtract, keep 3-D extrema",
        "3. refine, drop weak and edge-like",
        "4. give each point a direction",
        "5. 128 numbers around it",
        "6. match, keep unambiguous pairs",
    ]
    panels = [_panel(p, t) for p, t in zip((p1, p2, p3, p4, p5, p6), titles, strict=True)]
    hmax = max(p.shape[0] for p in panels)
    panels = [
        np.vstack([p, np.full((hmax - p.shape[0], p.shape[1], 3), BG, np.uint8)]) for p in panels
    ]
    gap = np.full((hmax, 12, 3), BG, np.uint8)
    rows = [np.hstack([panels[i], gap, panels[i + 1]]) for i in (0, 2, 4)]
    vgap = np.full((12, rows[0].shape[1], 3), BG, np.uint8)
    _save(np.vstack([rows[0], vgap, rows[1], vgap, rows[2]]), "hub-pipeline")


def _descriptor_view(a, run, box=None, zoom=None):
    """The strongest keypoint (inside `box` if given) with its rotated 4x4 grid and the
    8-bin histogram of each cell, drawn as arrows."""
    cands = run.kept
    if box:
        x0, y0, w, h = box
        cands = [
            e
            for e in run.kept
            if x0 + 40 < e.pt[0] < x0 + w - 40 and y0 + 40 < e.pt[1] < y0 + h - 40 and e.size > 6
        ]
    e = max(cands, key=lambda e: abs(e.response))
    img = run.pyramid.gauss[e.octave][e.layer]
    v = S.descriptor(img, e.c + e.xc, e.r + e.xr, e.angle, e.scale).reshape(4, 4, 8)
    f = 2.0 ** (e.octave - 1)
    cell = S.DESC_SCL * e.scale * f  # cell side in image pixels
    half = 2 * cell * 1.5
    cx, cy = e.pt
    Z = zoom or 300 / (2 * half)
    x0, y0 = int(cx - half), int(cy - half)
    side = int(2 * half)
    patch = a[max(y0, 0) : y0 + side, max(x0, 0) : x0 + side]
    big = cv2.resize(_bgr(patch), None, fx=Z, fy=Z, interpolation=cv2.INTER_NEAREST)
    big = _dim(big, 0.7)
    th = np.radians(360 - e.angle)  # the keypoint's direction, image y up
    ux, uy = np.array([np.cos(th), -np.sin(th)]), np.array([np.sin(th), np.cos(th)])
    c0 = np.array([cx - x0, cy - y0]) * Z
    for t in np.linspace(-2, 2, 5):
        p, q = c0 + (t * ux - 2 * uy) * cell * Z, c0 + (t * ux + 2 * uy) * cell * Z
        cv2.line(big, _pt(p), _pt(q), GREY, 1, cv2.LINE_AA)
        p, q = c0 + (-2 * ux + t * uy) * cell * Z, c0 + (2 * ux + t * uy) * cell * Z
        cv2.line(big, _pt(p), _pt(q), GREY, 1, cv2.LINE_AA)
    vmax = v.max()
    for ri in range(4):
        for ci in range(4):
            centre = c0 + ((ci - 1.5) * ux + (ri - 1.5) * uy) * cell * Z
            for o in range(8):
                ang = th - np.radians(o * 45)
                L = v[ri, ci, o] / vmax * cell * Z * 0.48
                d = np.array([np.cos(ang), -np.sin(ang)])
                cv2.arrowedLine(
                    big, _pt(centre), _pt(centre + d * L), AMBER, 1, cv2.LINE_AA, tipLength=0.2
                )
    cv2.arrowedLine(
        big, _pt(c0), _pt(c0 + ux * cell * Z * 2.2), GREEN, 2, cv2.LINE_AA, tipLength=0.15
    )
    return big


def hub_patch_vs_sift():
    """The same keypoint in photo A and in photo B rotated 20°: the raw patch changes, the
    descriptor barely does."""
    a, b = scene.pair()
    H, _ = scene.reference_homography()
    bw, M = scene.rotate_scale(b, 20)
    HM = M @ H
    sift = scene.sift()
    ka, da = sift.compute(a, scene.sift_points(a))
    kb, db = sift.compute(bw, scene.sift_points(bw))
    ms = [m for m, n in scene.knn(da, db) if m.distance < RATIO * n.distance]
    best = None
    for m in ms:
        p, q = ka[m.queryIdx], kb[m.trainIdx]
        on = np.linalg.norm(scene.apply(HM, [p.pt])[0] - q.pt) < 1 and 6 < p.size < 14
        if on and (best is None or m.distance < best[2]):
            best = (p, q, m.distance, da[m.queryIdx], db[m.trainIdx])
    p, q, dist, dva, dvb = best
    side = 33

    def cut(img, pt):
        h = side // 2
        x, y = int(round(pt[0])), int(round(pt[1]))
        return cv2.resize(
            _bgr(img[y - h : y + h + 1, x - h : x + h + 1]),
            (198, 198),
            interpolation=cv2.INTER_NEAREST,
        )

    pa, pb = cut(a, p.pt), cut(bw, q.pt)
    fig, axes = plt.subplots(
        2, 2, figsize=(8, 5.4), dpi=110, facecolor=BG_HEX, gridspec_kw={"width_ratios": [1, 2.3]}
    )
    for ax in axes.ravel():
        ax.set_facecolor(BG_HEX)
        for s in ax.spines.values():
            s.set_color("#445")
        ax.tick_params(colors="#bbb", labelsize=8)
    axes[0, 0].imshow(cv2.cvtColor(pa, cv2.COLOR_BGR2RGB))
    axes[1, 0].imshow(cv2.cvtColor(pb, cv2.COLOR_BGR2RGB))
    for ax, t in ((axes[0, 0], "photo A"), (axes[1, 0], "photo B, rotated 20°")):
        ax.set_xticks([]), ax.set_yticks([])
        ax.set_title(t, color="#ddd", fontsize=10)
    axes[0, 1].bar(range(128), dva, color="#60a5fa", width=1.0)
    axes[1, 1].bar(range(128), dvb, color="#a3e635", width=1.0)
    axes[0, 1].set_title("its 128 numbers", color="#ddd", fontsize=10)
    axes[1, 1].set_title(f"its 128 numbers (distance {dist:.0f})", color="#ddd", fontsize=10)
    for ax in axes[:, 1]:
        ax.set_xlim(-1, 128)
        ax.set_ylim(0, 160)
    fig.tight_layout()
    p_ = FIG_DIR / "hub-patch-vs-sift.png"
    fig.savefig(p_, facecolor=BG_HEX)
    plt.close(fig)
    logger.info(f"wrote {p_}")


# --- lesson 1 ----------------------------------------------------------------------


def lesson1():
    """Repeated points, Harris and SIFT, on photo A and its half-size copy."""
    a, _ = scene.pair()
    a_bgr = scene.pair_bgr()[0]
    small, H = scene.rotate_scale(a, 0, 0.5)
    rows = []
    for det in ("harris", "sift"):
        kA = scene.harris_points(a, 300) if det == "harris" else scene.sift_points(a, 300)
        kB = scene.harris_points(small, 300) if det == "harris" else scene.sift_points(small, 300)
        q = scene.apply(H, [k.pt for k in kA])
        pb = np.array([k.pt for k in kB])
        d = np.linalg.norm(q[:, None] - pb[None], axis=2).min(1)
        hit = d <= REPEAT_TOL_PX
        left = _dim(a_bgr, 0.6).copy()
        for k, h_ in zip(kA, hit, strict=True):
            col = GREEN if h_ else RED
            if det == "sift":
                cv2.circle(left, _pt(k.pt), max(int(k.size / 2), 2), col, 1, cv2.LINE_AA)
            else:
                cv2.circle(left, _pt(k.pt), 3, col, -1, cv2.LINE_AA)
        label(
            left,
            f"{'Harris' if det == 'harris' else 'SIFT'}: "
            f"{int(hit.sum())} of 300 found again at half size",
            (12, 30),
            0.65,
        )
        rows.append(left)
    _save(
        np.vstack([rows[0], np.full((12, rows[0].shape[1], 3), BG, np.uint8), rows[1]]),
        "l1-repeatability",
    )
    # what the size buys: SIFT circles on a crop of A and on the same crop of the small copy
    x0, y0, w, h = 330, 280, 300, 220
    A = _dim(a_bgr[y0 : y0 + h, x0 : x0 + w], 0.7).copy()
    B = cv2.resize(scene.pair_bgr()[0], None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
    B = _dim(B[y0 // 2 : (y0 + h) // 2, x0 // 2 : (x0 + w) // 2], 0.7)
    B = cv2.resize(B, (w, h), interpolation=cv2.INTER_NEAREST).copy()
    for k in scene.sift_points(a):
        x, y = k.pt[0] - x0, k.pt[1] - y0
        if 0 <= x < w and 0 <= y < h and k.size > 5:
            cv2.circle(A, _pt((x, y)), int(k.size / 2), AMBER, 1, cv2.LINE_AA)
    for k in scene.sift_points(small):
        x, y = k.pt[0] * 2 - x0, k.pt[1] * 2 - y0
        if 0 <= x < w and 0 <= y < h and k.size * 2 > 5:
            cv2.circle(B, _pt((x, y)), int(k.size), AMBER, 1, cv2.LINE_AA)
    s = 2
    A = cv2.resize(A, None, fx=s, fy=s, interpolation=cv2.INTER_LINEAR)
    B = cv2.resize(B, None, fx=s, fy=s, interpolation=cv2.INTER_LINEAR)
    label(A, "photo A", (12, 30), 0.8)
    label(B, "half-size copy, drawn at A's scale", (12, 30), 0.8)
    img, _ = _side_by_side(A, B)
    _save(img, "l1-sizes")


# --- lesson 2 ----------------------------------------------------------------------


def lesson2():
    a, _ = scene.pair()
    pyr = S.build_pyramid(a)
    tiles = []
    for o in (0, 2, 4):
        row = []
        for i in (1, 2, 3):
            d = pyr.dog[o][i]
            v = np.clip(d / (np.percentile(np.abs(d), 98) + 1e-6), -1, 1)
            t = np.zeros((*d.shape, 3), np.uint8)
            t[..., 2] = (np.clip(v, 0, 1) * 255).astype(np.uint8)
            t[..., 0] = (np.clip(-v, 0, 1) * 255).astype(np.uint8)
            t[..., 1] = (np.abs(v) * 60).astype(np.uint8)
            t = cv2.resize(
                t, (320, 241), interpolation=cv2.INTER_NEAREST if o >= 2 else cv2.INTER_AREA
            )
            label(t, f"octave {o}, level {i} ({d.shape[1]}x{d.shape[0]})", (8, 22), 0.45)
            row.append(t)
        tiles.append(
            np.hstack(
                [
                    row[0],
                    np.full((241, 6, 3), BG, np.uint8),
                    row[1],
                    np.full((241, 6, 3), BG, np.uint8),
                    row[2],
                ]
            )
        )
    g = np.full((6, tiles[0].shape[1], 3), BG, np.uint8)
    _save(np.vstack([tiles[0], g, tiles[1], g, tiles[2]]), "l2-dog-pyramid")
    # keypoints at their size: the post's result
    a_bgr = scene.pair_bgr()[0]
    img = _dim(a_bgr, 0.6).copy()
    kps = scene.sift_points(a)
    med = np.median([k.size for k in kps])
    for k in sorted(kps, key=lambda k: k.size):
        col = (200, 200, 90) if k.size < med else GREEN
        cv2.circle(img, _pt(k.pt), max(int(k.size / 2), 1), col, 1, cv2.LINE_AA)
    _save(img, "l2-keypoints-at-scale")
    # disc responses
    fig, ax = plt.subplots(figsize=(8, 4), dpi=110, facecolor=BG_HEX)
    ax.set_facecolor(BG_HEX)
    sig = np.geomspace(1, 30, 150)
    for r, col in ((4, "#60a5fa"), (8, "#a3e635"), (16, "#fbbf24")):
        img = np.zeros((257, 257))
        cv2.circle(img, (128, 128), r, 1.0, -1, lineType=cv2.LINE_AA)
        resp = [
            abs(
                s**2
                * cv2.Laplacian(cv2.GaussianBlur(img, (0, 0), s), cv2.CV_64F, ksize=1)[128, 128]
            )
            for s in sig
        ]
        ax.plot(sig, resp, color=col, lw=2, label=f"disc of radius {r} px")
        ax.axvline(sig[int(np.argmax(resp))], color=col, lw=0.8, ls="--")
    ax.set_xscale("log")
    ax.set_xlabel("σ (px)", color="#ccc")
    ax.set_ylabel("|σ² ∇²G| at the centre", color="#ccc")
    ax.tick_params(colors="#bbb")
    for s in ax.spines.values():
        s.set_color("#445")
    ax.legend(facecolor=BG_HEX, edgecolor="#445", labelcolor="#ddd")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "l2-disc-scale.png", facecolor=BG_HEX)
    plt.close(fig)


# --- lesson 3 ----------------------------------------------------------------------


def lesson3():
    a, _ = scene.pair()
    a_bgr = scene.pair_bgr()[0]
    pyr = S.build_pyramid(a)
    colours = {"drift": GREY, "contrast": BLUE, "edge": MAGENTA, "kept": GREEN}
    img = _dim(a_bgr, 0.5).copy()
    counts = dict.fromkeys(colours, 0)
    for e in S.raw_extrema(pyr):
        stage, f = S.refine(pyr, e)
        counts[stage] += 1
        cv2.circle(img, _pt(f.pt if stage != "drift" else e.pt), 2, colours[stage], -1, cv2.LINE_AA)
    y = 28
    for k, name in (
        ("kept", "kept"),
        ("contrast", "too weak (contrast)"),
        ("edge", "edge-like"),
        ("drift", "fit did not settle"),
    ):
        label(img, f"{name}: {counts[k]}", (12, y), 0.62, colours[k])
        y += 26
    _save(img, "l3-stages")
    # the edge test, one passed and one dropped, on crops
    num = json.loads((OUTPUT_DIR / "sift_numbers.json").read_text())["detector"]["edge_test"]
    tiles = []
    for key, col, txt in (("kept", GREEN, "kept"), ("dropped", MAGENTA, "dropped")):
        ex = num[key]
        x, y = ex["pt"]
        h = 24
        c = a_bgr[int(y) - h : int(y) + h, int(x) - h : int(x) + h]
        c = cv2.resize(c, (240, 240), interpolation=cv2.INTER_NEAREST).copy()
        cv2.circle(c, (120, 120), 6, col, 2, cv2.LINE_AA)
        label(c, f"{txt}: tr^2/det = {ex['tr2_over_det']:.1f}", (8, 230), 0.5, col)
        tiles.append(c)
    _save(_side_by_side(*tiles)[0], "l3-edge-test")
    # orientation histogram of the strongest keypoint
    st = json.loads((OUTPUT_DIR / "sift_numbers.json").read_text())["detector"]["strongest"]
    fig, ax = plt.subplots(figsize=(8, 3.2), dpi=110, facecolor=BG_HEX)
    ax.set_facecolor(BG_HEX)
    hst = np.array(st["histogram"])
    cols = ["#a3e635" if i == int(np.argmax(hst)) else "#60a5fa" for i in range(36)]
    ax.bar(np.arange(36) * 10, hst, width=9, color=cols, align="edge")
    ax.axhline(hst.max() * S.ORI_PEAK, color="#fbbf24", lw=1, ls="--")
    ax.set_xlim(0, 360)
    ax.set_xticks(range(0, 361, 90))
    ax.set_xlabel("gradient direction (degrees)", color="#ccc")
    ax.tick_params(colors="#bbb")
    for s in ax.spines.values():
        s.set_color("#445")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "l3-orientation-histogram.png", facecolor=BG_HEX)
    plt.close(fig)


# --- lesson 4 ----------------------------------------------------------------------


def lesson4():
    a, _ = scene.pair()
    run = S.detect(a)
    _save(_descriptor_view(a, run, (300, 260, 360, 270), zoom=None), "l4-descriptor-grid")
    # oriented vs upright at 45 degrees
    img45, H = scene.rotate_scale(a, 45)
    a_bgr = scene.pair_bgr()[0]
    b_bgr = scene.rotate_scale(a_bgr, 45)[0]
    out = []
    for upright in (False, True):
        kA, dA = X._scratch_set(a, upright)
        kB, dB = X._scratch_set(img45, upright)
        ms = X._ratio_matches(dA, dB)
        pa = np.array([kA[m.queryIdx].pt for m in ms]).reshape(-1, 2)
        pb = np.array([kB[m.trainIdx].pt for m in ms]).reshape(-1, 2)
        ok = np.linalg.norm(scene.apply(H, pa) - pb, axis=1) <= MATCH_TOL_PX
        sel = np.arange(len(pa))
        if len(sel) > 300:
            sel = np.random.default_rng(0).choice(sel, 300, replace=False)
        c = _matches_canvas(a_bgr, b_bgr, pa[sel], pb[sel], ok[sel])
        label(
            c,
            f"{'upright' if upright else 'oriented'}: {int(ok.sum())} of {len(ok)} correct",
            (16, 44),
            1.2,
        )
        out.append(c)
    _save(
        np.vstack([out[0], np.full((12, out[0].shape[1], 3), BG, np.uint8), out[1]]),
        "l4-upright-vs-oriented",
    )


# --- lesson 5 ----------------------------------------------------------------------


def lesson5():
    pa, pb, r, ok, over, _ka, _kb = _real_matches()
    fig, ax = plt.subplots(figsize=(8, 3.6), dpi=110, facecolor=BG_HEX)
    ax.set_facecolor(BG_HEX)
    bins = np.linspace(0, 1, 41)
    ax.hist(r[over & ok], bins, color="#a3e635", alpha=0.85, label="correct (by the ORB reference)")
    ax.hist(r[over & ~ok], bins, color="#f87171", alpha=0.7, label="wrong")
    ax.axvline(RATIO, color="#fbbf24", lw=1.5, ls="--")
    ax.text(RATIO + 0.01, ax.get_ylim()[1] * 0.9, "0.75", color="#fbbf24")
    ax.set_xlabel("ratio d₁ / d₂ (nearest / second nearest)", color="#ccc")
    ax.set_ylabel("matches", color="#ccc")
    ax.tick_params(colors="#bbb")
    for s in ax.spines.values():
        s.set_color("#445")
    ax.legend(facecolor=BG_HEX, edgecolor="#445", labelcolor="#ddd")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "l5-ratio-histogram.png", facecolor=BG_HEX)
    plt.close(fig)
    a_bgr, b_bgr = scene.pair_bgr()
    keep = r < RATIO
    rej = over & ~keep
    sel = np.nonzero(rej)[0]
    sel = np.random.default_rng(0).choice(sel, min(200, len(sel)), replace=False)
    canvas, off = _side_by_side(_dim(a_bgr, 0.8), _dim(b_bgr, 0.8))
    for i in sel:
        cv2.line(canvas, _pt(pa[i]), _pt(pb[i], off=(off, 0)), RED, 1, cv2.LINE_AA)
    for i in np.nonzero(keep & ok)[0]:
        cv2.line(canvas, _pt(pa[i]), _pt(pb[i], off=(off, 0)), GREEN, 1, cv2.LINE_AA)
    label(canvas, f"kept by the ratio test: {int(keep.sum())}", (16, 44), 1.2, GREEN)
    label(canvas, "rejected (200 of them drawn)", (16, 88), 1.2, RED)
    _save(canvas, "l5-kept-rejected")


def wild_viewpoint():
    num = json.loads((OUTPUT_DIR / "sift_numbers.json").read_text())["wild"]["viewpoint"]
    deg = [v["degrees"] for v in num]
    fig, ax = plt.subplots(figsize=(8, 3.8), dpi=110, facecolor=BG_HEX)
    ax.set_facecolor(BG_HEX)
    ax.plot(
        deg, [v["matches"] for v in num], "o-", color="#fbbf24", lw=2, label="ratio-test matches"
    )
    ax.plot(
        deg,
        [v["inliers"] for v in num],
        "o-",
        color="#a3e635",
        lw=2,
        label="RANSAC inliers (fundamental matrix)",
    )
    ax.set_xlabel(
        f"viewpoint change around the temple (degrees, {TEMPLE_STEP_DEG:.2f}° per photo)",
        color="#ccc",
    )
    ax.set_ylabel("matches with photo 1", color="#ccc")
    ax.tick_params(colors="#bbb")
    for s in ax.spines.values():
        s.set_color("#445")
    ax.legend(facecolor=BG_HEX, edgecolor="#445", labelcolor="#ddd")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "wild-viewpoint.png", facecolor=BG_HEX)
    plt.close(fig)


# --- lab data ----------------------------------------------------------------------


def lab_data():
    LAB_DIR.mkdir(parents=True, exist_ok=True)
    a, b = scene.pair()
    a_bgr, b_bgr = scene.pair_bgr()
    for name, img in (("barcelona-a", a_bgr), ("barcelona-b", b_bgr)):
        cv2.imwrite(str(LAB_DIR / f"{name}.webp"), img, [cv2.IMWRITE_WEBP_QUALITY, 82])
    # The reference homography, for the page's cells (a RANSAC run inside a cell would give a
    # slightly different answer key on OpenCV 4.11 and 5.0).
    H, info = scene.reference_homography()
    (LAB_DIR / "reference-homography.json").write_text(
        json.dumps(
            {"H_a_to_b": [[round(float(v), 10) for v in row] for row in H], **info}, indent=1
        )
    )
    # RatioTestLab: every query that lands in the overlap, with its two distances.
    pa, pb, r, ok, over, _ka, _kb = _real_matches()
    d = json.loads((OUTPUT_DIR / "sift_numbers.json").read_text())["matching"]["lab_matches"]
    (LAB_DIR / "ratio-matches.json").write_text(
        json.dumps(
            {
                "w": a.shape[1],
                "h": a.shape[0],
                "m": [[*m["a"], *m["b"], m["d1"], m["d2"], int(m["ok"])] for m in d],
            },
            separators=(",", ":"),
        )
    )
    # SiftStagesLab: every raw extremum, with the two numbers the tests threshold.
    pyr = S.build_pyramid(a)
    rows = []
    for e in S.raw_extrema(pyr):
        stage, f = S.refine(pyr, e)
        if stage == "drift":
            rows.append([round(e.pt[0], 1), round(e.pt[1], 1), -1, -1])
            continue
        _g, Hs = S._derivatives(pyr.dog[f.octave], f.layer, f.r, f.c)
        tr, det = Hs[0, 0] + Hs[1, 1], Hs[0, 0] * Hs[1, 1] - Hs[0, 1] ** 2
        score = tr * tr / det if det > 0 else 999.0
        rows.append(
            [
                round(f.pt[0], 1),
                round(f.pt[1], 1),
                round(abs(f.response) * S.LAYERS, 8),
                round(min(score, 999.0), 2),
            ]
        )
    (LAB_DIR / "sift-stages.json").write_text(
        json.dumps(
            {"w": a.shape[1], "h": a.shape[0], "contrast": S.CONTRAST, "edge": S.EDGE, "p": rows},
            separators=(",", ":"),
        )
    )
    # KeypointWarpLab: for a grid of rotations and zooms, which of A's strongest 300 points
    # each detector finds again, and how many matches survive with and without orientation.
    warps = []
    sift = scene.sift()
    kA = scene.sift_points(a, 300)
    hA = scene.harris_points(a, 300)
    for deg in (0, 15, 30, 45, 60, 90):
        for s in (1.0, 0.71, 0.5):
            img, H = scene.rotate_scale(a, deg, s)
            res = {"deg": deg, "s": s}
            for det, src, fn in (
                ("sift", kA, lambda im: scene.sift_points(im, 300)),
                ("harris", hA, lambda im: scene.harris_points(im, 300)),
            ):
                rep = X.repeatability(src, fn(img), H, a.shape, img.shape)
                q = scene.apply(H, [k.pt for k in src])
                pb_ = np.array([k.pt for k in fn(img)])
                dd = np.linalg.norm(q[:, None] - pb_[None], axis=2).min(1)
                inside = scene.inside(q, img.shape, 8)
                res[det] = {
                    "rate": rep["rate"],
                    "hit": [
                        int(h) if i else -1
                        for h, i in zip(dd <= REPEAT_TOL_PX, inside, strict=True)
                    ],
                }
            k1, d1 = sift.compute(a, scene.sift_points(a))
            k2, d2 = sift.compute(img, scene.sift_points(img))
            res["matches"] = X._scored(k1, k2, X._ratio_matches(d1, d2), H)["correct"]
            warps.append(res)
    pts = {
        "sift": [[round(k.pt[0], 1), round(k.pt[1], 1), round(k.size, 1)] for k in kA],
        "harris": [[round(k.pt[0], 1), round(k.pt[1], 1)] for k in hA],
    }
    (LAB_DIR / "keypoint-warps.json").write_text(
        json.dumps(
            {"w": a.shape[1], "h": a.shape[0], "points": pts, "warps": warps}, separators=(",", ":")
        )
    )
    logger.info(f"wrote lab data to {LAB_DIR}")


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
    pair = rd("hub-matches")
    h = pair.shape[0]
    _save(place(pair[:, : pair.shape[1]]), HUB, COVER_DIR)
    # covers carry no text: crop off the figures' labels
    _save(place(rd("l1-sizes")[70:]), SLUGS[1], COVER_DIR)
    _save(place(rd("l2-keypoints-at-scale")), SLUGS[2], COVER_DIR)
    _save(place(rd("l3-stages")[130:]), SLUGS[3], COVER_DIR)
    _save(place(rd("l4-descriptor-grid")), SLUGS[4], COVER_DIR)
    _save(place(rd("l5-kept-rejected")[:h]), SLUGS[5], COVER_DIR)


def render_all():
    hub_matches()
    hub_pipeline()
    hub_patch_vs_sift()
    lesson1()
    lesson2()
    lesson3()
    lesson4()
    lesson5()
    wild_viewpoint()
    lab_data()
    covers()
