"""Figures and cover backgrounds for units 3.5 and 4.1-L1, drawn from the fit itself.

Every overlay is computed, not traced: the lines drawn are the fitted 3-vectors,
the court drawn is the model pushed through the named transform.
"""

from __future__ import annotations

import json

import cv2
import numpy as np
from loguru import logger

from fundamentals_lab.alignment import court, distortion, transforms, warp
from fundamentals_lab.alignment.experiments import Scene
from fundamentals_lab.config import ALIGN_PLATE_FRAMES, OUTPUT_DIR

FIG_DIR = OUTPUT_DIR / "figures" / "alignment"
COVER_DIR = OUTPUT_DIR / "covers"
GREEN, RED, AMBER, CYAN, WHITE = (
    (80, 220, 120),
    (70, 70, 235),
    (40, 190, 250),
    (230, 200, 60),
    (245, 245, 245),
)
BG = (24, 14, 11)  # BGR of #0b0e18, the site's dark cover background

# Court model segments to draw: every painted line.
SEGMENTS = list(court.LINES.values())


def _save(img, name, folder=FIG_DIR):
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{name}.png"
    cv2.imwrite(str(path), img)
    logger.info(f"wrote {path}")
    return path


def _draw_court(img, H_court2img, colour, thickness=3, offset=(0, 0), n=60):
    """Push each court line through H, densely sampled so a curve would show as one."""
    for p, q in SEGMENTS:
        t = np.linspace(0, 1, n)[:, None]
        pts = transforms.apply(H_court2img, np.array(p) * (1 - t) + np.array(q) * t) + offset
        cv2.polylines(
            img, [np.rint(pts * 4).astype(np.int32)], False, colour, thickness, cv2.LINE_AA, shift=2
        )


def _line_across(img, line, colour, thickness=2, offset=(0, 0)):
    """Draw the infinite line a x + b y + c = 0 across the whole canvas."""
    h, w = img.shape[:2]
    a, b, c = line
    c = c - a * offset[0] - b * offset[1]  # shift into canvas coordinates
    pts = []
    for x in (0, w - 1):
        if abs(b) > 1e-9:
            pts.append((x, -(a * x + c) / b))
    for y in (0, h - 1):
        if abs(a) > 1e-9:
            pts.append((-(b * y + c) / a, y))
    pts = [p for p in pts if -1 <= p[0] <= w and -1 <= p[1] <= h]
    if len(pts) >= 2:
        p, q = np.array(pts[0]), np.array(pts[-1])
        cv2.line(
            img,
            tuple(np.rint(p * 4).astype(int)),
            tuple(np.rint(q * 4).astype(int)),
            colour,
            thickness,
            cv2.LINE_AA,
            shift=2,
        )


def extended_canvas(s: Scene) -> np.ndarray:
    """4.1-L1: the frame on a canvas wide enough to hold the off-frame corner and the VP."""
    left, right, top, bottom = 120, 480, 80, 20
    h, w = s.frame.shape[:2]
    canvas = np.full((h + top + bottom, w + left + right, 3), 36, np.uint8)
    canvas[top : top + h, left : left + w] = s.frame
    off = (left, top)
    L = {n: f.line for n, f in s.fit.lines.items()}
    for name in ("near_base", "near_kitchen"):
        _line_across(canvas, L[name], AMBER, 3, off)
    for name in ("side_left", "side_right", "near_centre"):
        _line_across(canvas, L[name], GREEN, 3, off)
    vp = court.dehomogenise(court.meet(L["side_left"], L["side_right"])) + off
    vp_across = court.meet(L["near_base"], L["near_kitchen"])
    horizon = np.cross(court.meet(L["side_left"], L["side_right"]), vp_across)
    _line_across(canvas, horizon / np.linalg.norm(horizon[:2]), WHITE, 2, off)
    cv2.rectangle(canvas, (left, top), (left + w - 1, top + h - 1), WHITE, 2)
    nbl = s.fit.landmarks_img["NBL"] + off
    for p, col in ((vp, WHITE), (nbl, RED)):
        cv2.circle(canvas, tuple(np.rint(p).astype(int)), 14, col, 4, cv2.LINE_AA)
    return canvas


def best_linear_overlay(s: Scene) -> np.ndarray:
    """L1: the best any 2x2 can do (a parallelogram) against the court as seen."""
    img = s.frame.copy()
    c_court, c_img = s.fit_court.mean(0), s.fit_img.mean(0)
    M = transforms.fit_linear(s.fit_court - c_court, s.fit_img - c_img)
    A = np.hstack([M, (c_img - M @ c_court)[:, None]])  # the same map, written as 2x3
    H_lin = np.vstack([A, [0, 0, 1]])
    _draw_court(img, s.H_court2img, GREEN, 3)
    _draw_court(img, H_lin, RED, 3)
    return img


def affine_vs_homography(s: Scene) -> np.ndarray:
    """L2: the court pushed through the 3-point affine (red) and the 4-point homography (green)."""
    img = s.frame.copy()
    three = [0, 1, 2]
    A3 = transforms.fit_affine(s.fit_court[three], s.fit_img[three])
    _draw_court(img, np.vstack([A3, [0, 0, 1]]), RED, 3)
    _draw_court(img, s.H_court2img, GREEN, 3)
    for n in court.FIT_LANDMARKS[:3]:
        cv2.circle(
            img, tuple(np.rint(s.fit.landmarks_img[n]).astype(int)), 12, WHITE, 3, cv2.LINE_AA
        )
    return img


def scale_map(s: Scene) -> np.ndarray:
    """L3: how many centimetres of court each pixel covers, across the court (log colour)."""
    img = s.frame.copy()
    h, w = img.shape[:2]
    ys, xs = np.mgrid[0:h:4, 0:w:4]
    pts = np.stack([xs.ravel(), ys.ravel()], 1).astype(np.float64)
    c = transforms.apply(s.H_img2court, pts)
    cx = transforms.apply(s.H_img2court, pts + [0, 1])
    cm = np.linalg.norm(cx - c, axis=1) * 100
    on_court = (c[:, 0] >= 0) & (c[:, 0] <= 13.41) & (c[:, 1] >= 0) & (c[:, 1] <= 6.10)
    val = np.clip((np.log10(np.clip(cm, 0.3, 30)) - np.log10(0.3)) / 2, 0, 1)
    cmap = cv2.applyColorMap(
        (val * 255).astype(np.uint8).reshape(-1, 1), cv2.COLORMAP_TURBO
    ).reshape(-1, 3)
    overlay = np.zeros((ys.shape[0], ys.shape[1], 3), np.uint8)
    overlay.reshape(-1, 3)[on_court] = cmap[on_court]
    overlay = cv2.resize(overlay, (w, h), interpolation=cv2.INTER_NEAREST)
    mask = (
        cv2.resize(
            on_court.reshape(ys.shape).astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST
        )
        > 0
    )
    img[mask] = (0.45 * img[mask] + 0.55 * overlay[mask]).astype(np.uint8)
    _draw_court(img, s.H_court2img, WHITE, 2)
    return img


def topviews(s: Scene) -> dict[str, np.ndarray]:
    """L5: backward warp, forward warp with its holes, nearest vs bilinear, and the median plate."""
    H_i2t, size = warp.img2top(s.H_img2court)
    out = {}
    out["topview_backward"], seen = warp.backward(s.frame, H_i2t, size)
    fwd, hit = warp.forward(s.frame, H_i2t, size, np.ones(s.frame.shape[:2], bool))
    fwd[(hit == 0) & seen] = (255, 0, 255)  # holes in magenta, only where the camera sees
    fwd[~seen] = 0
    out["topview_forward"] = fwd
    out["topview_nearest"], _ = warp.backward(s.frame, H_i2t, size, "nearest")
    out["topview_plate"] = cv2.warpPerspective(s.plate, H_i2t, size, flags=cv2.INTER_LINEAR)
    return out


def lens_profile(s: Scene) -> dict:
    """The near baseline's bow before and after the plumb-line k, for a hand-drawn chart."""
    k = json.loads((OUTPUT_DIR / "alignment_numbers.json").read_text())["lens"]["k_division_model"]
    bands = distortion.line_bands(s.pixels, s.H_court2img, ["near_base", "side_right"])
    out = {"k": k}
    for name, pts in bands.items():
        prof = {}
        for tag, kk in (("before", 0.0), ("after", k)):
            u = distortion.undistort(pts, kk)
            from fundamentals_lab.alignment import lines

            fit = lines.fit_line(u)
            d = np.array([-fit.line[1], fit.line[0]])
            t = u @ d
            r = u @ fit.line[:2] + fit.line[2]
            bins = np.linspace(t.min(), t.max(), 13)
            mids, med = [], []
            for i in range(12):
                m = (t >= bins[i]) & (t < bins[i + 1])
                if m.sum() > 30:
                    mids.append(float((bins[i] + bins[i + 1]) / 2 - t.min()))
                    med.append(float(np.median(r[m])))
            prof[tag] = {
                "t_px": [round(v, 1) for v in mids],
                "offset_px": [round(v, 2) for v in med],
            }
        out[name] = prof
    return out


def cover_panel(img: np.ndarray, right_fraction: float = 0.62, size=(1600, 900)) -> np.ndarray:
    """A cover background: dark canvas, the image cropped into the right `right_fraction`."""
    W, H = size
    canvas = np.full((H, W, 3), BG, np.uint8)
    pw = round(W * right_fraction)
    ih, iw = img.shape[:2]
    scale = max(pw / iw, H / ih)
    r = cv2.resize(img, (round(iw * scale), round(ih * scale)), interpolation=cv2.INTER_AREA)
    y0 = (r.shape[0] - H) // 2
    x0 = (r.shape[1] - pw) // 2
    canvas[:, W - pw :] = r[y0 : y0 + H, x0 : x0 + pw]
    # soft fade into the text side
    fade = np.linspace(0, 1, 120)[None, :, None]
    canvas[:, W - pw : W - pw + 120] = (
        canvas[:, W - pw : W - pw + 120] * fade + np.array(BG) * (1 - fade)
    ).astype(np.uint8)
    return canvas


def render_all() -> None:
    s = Scene()
    ext = extended_canvas(s)
    lin = best_linear_overlay(s)
    aff = affine_vs_homography(s)
    sm = scale_map(s)
    tv = topviews(s)
    _save(ext, "homogeneous_extended")
    _save(lin, "best_linear_overlay")
    _save(aff, "affine_vs_homography")
    _save(sm, "scale_map")
    for name, img in tv.items():
        _save(img, name)
    # nearest vs bilinear: a far-court crop of the top view, enlarged 4x without smoothing
    y0, y1, x0, x1 = 40, 160, 150, 300
    crop_n = cv2.resize(
        tv["topview_nearest"][y0:y1, x0:x1], None, fx=4, fy=4, interpolation=cv2.INTER_NEAREST
    )
    crop_b = cv2.resize(
        tv["topview_backward"][y0:y1, x0:x1], None, fx=4, fy=4, interpolation=cv2.INTER_NEAREST
    )
    _save(
        np.hstack([crop_n, np.full((crop_n.shape[0], 12, 3), 255, np.uint8), crop_b]),
        "interp_nearest_bilinear",
    )
    pair = np.hstack(
        [
            cv2.resize(s.frame, (1459, 820)),
            np.full((820, 16, 3), 255, np.uint8),
            tv["topview_backward"],
        ]
    )
    _save(pair, "hub_frame_and_topview")
    small = [
        cv2.resize(cv2.imread(str(p)), (960, 540), interpolation=cv2.INTER_AREA)
        for p in [
            OUTPUT_DIR.parent / "data" / "pickleball" / f"frame_{i:06d}.png"
            for i in ALIGN_PLATE_FRAMES
        ]
    ]
    for i, img in zip(ALIGN_PLATE_FRAMES, small, strict=True):
        _save(img, f"frame960_{i:06d}", FIG_DIR / "frames960")
    (FIG_DIR / "lens_profile.json").write_text(json.dumps(lens_profile(s), indent=1))
    covers = {
        "homogeneous-coordinates": ext[:, 200:],
        "image-alignment-and-stitching": pair,
        "2d-linear-transforms": lin,
        "affine-and-projective-transforms": aff,
        "computing-a-homography-dlt": sm,
        "image-warping-and-blending": np.rot90(tv["topview_plate"], 1).copy(),
    }
    for slug, img in covers.items():
        _save(cover_panel(img), slug, COVER_DIR)
