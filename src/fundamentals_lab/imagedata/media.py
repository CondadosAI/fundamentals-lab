"""Animated figures for unit 1.3: the real application each post opens with.

Most frames come from the unit's own exposure of the HDRPS Luxo Double Checker scene
(`scene.load`, research and non-commercial publication, cleared for the track); the
colour-selection one uses a frame of the pickleball match (CC BY 3.0). Per-patch colour
errors are read from `output/image_data_numbers.json`, not recomputed here.
"""

from __future__ import annotations

import json
from functools import cache
from pathlib import Path

import cv2
import numpy as np
from loguru import logger

from fundamentals_lab.alignment.media import BG, CLIP, WHITE, encode, label
from fundamentals_lab.config import IMAGEDATA_NUMBERS_JSON, OUTPUT_DIR
from fundamentals_lab.imagedata import colour, scene
from fundamentals_lab.sensing import charts

MEDIA_DIR = OUTPUT_DIR / "figures" / "imagedata" / "media"
AMBER = (80, 170, 250)
#: Top-left of a 320 px square on the net in match frame 60 (1920 x 1080): mesh only.
NET_PATCH = (900, 330)
EASE = lambda t: 0.5 - 0.5 * np.cos(np.pi * t)  # noqa: E731
FONT = cv2.FONT_HERSHEY_SIMPLEX


@cache
def _frame():
    return scene.load()


def _bgr() -> np.ndarray:
    return _frame().bgr8


def _fit(img: np.ndarray, width: int) -> np.ndarray:
    h = round(img.shape[0] * width / img.shape[1])
    return cv2.resize(img, (width, h), interpolation=cv2.INTER_AREA)


def _patch(number: int, chart: str = "bright"):
    return next(p for p in charts.patches(chart) if p.number == number)


# --- hub: from photograph to numbers ------------------------------------------------------


def pixel_zoom(out_h: int = 400, cells_w: int = 8, cells_h: int = 5):
    """Zoom into the chart until pixels are squares, then print each one's three numbers."""
    img = _bgr()
    h, w = img.shape[:2]
    (ax, ay), (bx, by) = _patch(14).centre, _patch(15).centre
    cx, cy = int((ax + bx) / 2), int((ay + by) / 2)
    aspect = 1.5
    out_w = int(out_h * aspect)
    spans = np.geomspace(h * 0.95, cells_h, 34)

    def crop(span):
        s = max(int(round(span)), cells_h)
        sw = int(round(s * aspect))
        x0 = int(np.clip(cx - sw / 2, 0, w - sw))
        y0 = int(np.clip(cy - s / 2, 0, h - s))
        interp = cv2.INTER_NEAREST if s < 200 else cv2.INTER_AREA
        return img[y0 : y0 + s, x0 : x0 + sw], cv2.resize(
            img[y0 : y0 + s, x0 : x0 + sw], (out_w, out_h), interpolation=interp
        )

    frames = []
    for s in spans:
        _, view = crop(s)
        label(view, "a photograph", (12, 32), 0.7)
        frames.append(view)
    block, view = crop(cells_h)
    cell_w, cell_h = out_w / block.shape[1], out_h / block.shape[0]
    numbers = view.copy()
    for r in range(block.shape[0]):
        for c in range(block.shape[1]):
            b, g, rr = (int(v) for v in block[r, c])
            ink = (20, 20, 20) if (0.299 * rr + 0.587 * g + 0.114 * b) > 130 else WHITE
            x, y = int(c * cell_w + 8), int(r * cell_h + 22)
            for k, (name, v) in enumerate((("R", rr), ("G", g), ("B", b))):
                cv2.putText(
                    numbers, f"{name} {v}", (x, y + k * 20), FONT, 0.45, ink, 1, cv2.LINE_AA
                )
    for k in range(8):
        frames.append(cv2.addWeighted(view, 1 - (k + 1) / 8, numbers, (k + 1) / 8, 0))
    done = numbers.copy()
    label(done, "what the computer holds: three numbers per pixel", (12, out_h - 16), 0.6)
    frames += [done] * 24
    return frames, 10


# --- lesson 1: the array, in the wrong order ---------------------------------------------


def channel_swap(width: int = 640):
    """The same array shown with its channels in the right order and in the wrong one."""
    right = _fit(_bgr(), width)
    wrong = right[:, :, ::-1].copy()
    ts = (
        [0.0] * 12
        + [EASE(i / 11) for i in range(12)]
        + [1.0] * 14
        + [1 - EASE(i / 11) for i in range(12)]
    )
    frames = []
    for t in ts:
        img = cv2.addWeighted(right, 1 - t, wrong, t, 0)
        label(
            img,
            "read as B, G, R: correct" if t < 0.5 else "same numbers read as R, G, B",
            (12, 32),
            0.7,
        )
        frames.append(img)
    return frames, 10


# --- lesson 2: sampling without blurring first --------------------------------------------


def aliasing(out: int = 300, size: int = 320):
    """A patch of the net, decimated by 1 to 8, without and with a blur first.

    The net's mesh is finer than most things in a frame, so skipping pixels folds it into
    coarse false stripes (moire); blurring first removes what the coarser grid cannot hold.
    """
    cap = cv2.VideoCapture(str(CLIP))
    cap.set(cv2.CAP_PROP_POS_FRAMES, 60)
    _, f = cap.read()
    x0, y0 = NET_PATCH
    crop = f[y0 : y0 + size, x0 : x0 + size]
    frames = []
    factors = list(range(1, 9)) + list(range(7, 1, -1))
    for k in factors:
        naive = crop[::k, ::k]
        blurred = cv2.GaussianBlur(crop, (0, 0), 0.5 * k) if k > 1 else crop
        smooth = blurred[::k, ::k]
        left = cv2.resize(naive, (out, out), interpolation=cv2.INTER_NEAREST)
        right = cv2.resize(smooth, (out, out), interpolation=cv2.INTER_NEAREST)
        label(left, f"keep 1 pixel in {k}" if k > 1 else "full resolution", (8, 26), 0.55)
        label(
            right, "blur, then keep 1 in " + str(k) if k > 1 else "full resolution", (8, 26), 0.55
        )
        img = np.hstack([left, np.full((out, 6, 3), 40, np.uint8), right])
        frames += [img] * 4
    return frames, 6


# --- lesson 3: fewer bits --------------------------------------------------------------------


def bit_depth(out_w: int = 600, out_h: int = 400):
    """The pool of lamplight on the table, requantised from 8 bits down to 3 and back.

    A smooth gradient is where missing levels show, as bands with visible edges.
    """
    img = _bgr()
    h, w = img.shape[:2]
    crop = img[int(0.66 * h) : int(0.98 * h), int(0.36 * w) : int(0.72 * w)]
    base = cv2.resize(crop, (out_w, out_h), interpolation=cv2.INTER_AREA)
    frames = []
    for bits in [8, 7, 6, 5, 4, 3, 3, 4, 5, 6, 7, 8]:
        step = 2 ** (8 - bits)
        view = (
            ((base // step) * step + step // 2).clip(0, 255).astype(np.uint8)
            if bits < 8
            else base.copy()
        )
        label(view, f"{bits} bits: {2**bits} levels per channel", (12, 32), 0.7)
        frames += [view] * 6
    return frames, 6


# --- lesson 4: picking by colour --------------------------------------------------------------


def hue_select(width: int = 640):
    """A window of hues sweeping the colour wheel, on a match frame: everything else greyed."""
    cap = cv2.VideoCapture(str(CLIP))
    cap.set(cv2.CAP_PROP_POS_FRAMES, 60)
    _, f = cap.read()
    f = _fit(f, width)
    hsv = cv2.cvtColor(f, cv2.COLOR_BGR2HSV)
    grey = cv2.cvtColor(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
    grey = (grey * 0.55).astype(np.uint8)
    frames = []
    for centre in np.linspace(0, 180, 48, endpoint=False):
        d = np.abs(((hsv[..., 0].astype(int) - centre + 90) % 180) - 90)
        keep = (d <= 12) & (hsv[..., 1] > 90) & (hsv[..., 2] > 40)
        img = np.where(keep[..., None], f, grey)
        swatch = cv2.cvtColor(np.uint8([[[centre, 220, 230]]]), cv2.COLOR_HSV2BGR)[0, 0]
        cv2.rectangle(img, (width - 70, 14), (width - 14, 50), tuple(int(v) for v in swatch), -1)
        label(img, f"keep hue {2 * centre:.0f} deg +- 24", (12, 32), 0.65)
        frames.append(img)
    return frames, 8


# --- lesson 5: how far off is each colour ----------------------------------------------------


def _lab_to_srgb8(lab: np.ndarray) -> np.ndarray:
    """CIELAB relative to a D65 white, back to an 8-bit sRGB swatch (BGR)."""
    L, a, b = lab
    fy = (L + 16) / 116
    fx, fz = fy + a / 500, fy - b / 200
    inv = lambda t: t**3 if t**3 > 216 / 24389 else (116 * t - 16) / (24389 / 27)  # noqa: E731
    xyz = np.array([inv(fx) * 0.95047, inv(fy), inv(fz) * 1.08883])
    lin = np.linalg.solve(colour.SRGB_FROM_XYZ_D65, xyz).clip(0, 1)
    enc = np.where(lin <= 0.0031308, 12.92 * lin, 1.055 * lin ** (1 / 2.4) - 0.055)
    return (enc[::-1] * 255).round().clip(0, 255).astype(np.uint8)


def delta_e_patches(width: int = 640):
    """Each usable patch: the file's colour beside the metered one, and the error between."""
    acc = json.loads(Path(IMAGEDATA_NUMBERS_JSON).read_text())["l4_colour"]["accuracy"]
    per_patch = acc["paths"]["naive_srgb"]["per_patch"]
    frame = _frame()
    samples = colour.patch_samples(frame)
    metered = colour._metered()
    white = metered[colour.WHITE_PATCH]
    white_xyz = colour.xyY_to_XYZ(white["x"], white["y"], white["luminance"])

    # Both swatches are shown relative to their own white patch, as the error is computed:
    # the file's colours through the sRGB assumption, the meter's in its own units.
    def naive_xyz(srgb8):
        return colour.SRGB_FROM_XYZ_D65 @ colour._srgb_to_linear(srgb8)

    file_white = naive_xyz(samples[colour.WHITE_PATCH]["srgb8"])
    photo = _fit(_bgr(), width)
    s = width / _bgr().shape[1]
    frames = []
    for n in acc["usable_patches"]:
        m = metered[n]
        truth = colour.to_lab(colour.xyY_to_XYZ(m["x"], m["y"], m["luminance"]), white_xyz)
        ref = _lab_to_srgb8(truth)
        got = _lab_to_srgb8(colour.to_lab(naive_xyz(samples[n]["srgb8"]), file_white))
        img = photo.copy()
        x0, y0, x1, y1 = (np.array(_patch(n).box) * s).astype(int)
        cv2.rectangle(img, (x0 - 5, y0 - 5), (x1 + 5, y1 + 5), AMBER, 2)
        sw = 120
        panel = np.full((img.shape[0], 2 * sw + 60, 3), BG, np.uint8)
        top = img.shape[0] // 2 - sw // 2
        panel[top : top + sw, 20 : 20 + sw] = got
        panel[top : top + sw, 40 + sw : 40 + 2 * sw] = ref
        cv2.putText(panel, "in the file", (20, top - 10), FONT, 0.5, WHITE, 1, cv2.LINE_AA)
        cv2.putText(panel, "measured", (40 + sw, top - 10), FONT, 0.5, WHITE, 1, cv2.LINE_AA)
        de = float(per_patch[str(n)])
        cv2.putText(
            panel,
            f"patch {n}: dE = {de:.1f}",
            (20, top + sw + 36),
            FONT,
            0.65,
            WHITE,
            1,
            cv2.LINE_AA,
        )
        frames += [np.hstack([img, panel])] * 2
    return frames, 1.5


# --- lesson 6: what compression costs -----------------------------------------------------------


def jpeg_sweep(out_w: int = 600, out_h: int = 400):
    """A crop of the chart saved as JPEG from quality 95 down to 5, with the file size."""
    img = _bgr()
    (cx, cy) = _patch(15).centre
    cw, ch = 150, 100
    x0, y0 = int(cx - cw / 2), int(cy - ch / 2)
    crop = np.ascontiguousarray(img[y0 : y0 + ch, x0 : x0 + cw])
    frames = []
    qualities = [95, 80, 60, 40, 25, 15, 10, 5]
    for q in qualities + qualities[-2::-1]:
        ok, buf = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, q])
        dec = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        view = cv2.resize(dec, (out_w, out_h), interpolation=cv2.INTER_NEAREST)
        label(view, f"JPEG quality {q}: {len(buf) / 1024:.1f} KB", (12, 32), 0.7)
        frames += [view] * 5
    return frames, 6


#: name -> (function, output width px, libwebp quality)
TABLE = {
    "pixel-zoom": (pixel_zoom, 600, 55),
    "channel-swap": (channel_swap, 600, 50),
    "aliasing": (aliasing, 606, 55),
    "bit-depth": (bit_depth, 600, 55),
    "hue-select": (hue_select, 600, 50),
    "delta-e-patches": (delta_e_patches, 700, 55),
    "jpeg-sweep": (jpeg_sweep, 600, 60),
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
