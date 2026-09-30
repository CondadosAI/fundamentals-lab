"""Animated figures for unit 1.2: the real application each post opens with.

All of it comes from the eighteen raw exposures of the HDR Photographic Survey's Luxo
Double Checker scene (Mark Fairchild, research and non-commercial publication; cleared
for the Fundamentals track), downloaded by `sensing-download`. The thresholds printed
on frames are the unit's own measurements, read from `output/sensing_numbers.json`.
"""

from __future__ import annotations

import json
from functools import cache
from pathlib import Path

import cv2
import numpy as np
import rawpy
from loguru import logger

from fundamentals_lab.alignment.media import BG, RED, WHITE, encode, label
from fundamentals_lab.config import OUTPUT_DIR, SCENE_SLUG, SENSING_NUMBERS_JSON
from fundamentals_lab.sensing import charts, hdr, hdrps, noise

MEDIA_DIR = OUTPUT_DIR / "figures" / "sensing" / "media"
BLUE = (235, 140, 40)
AMBER = (80, 170, 250)
EASE = lambda t: 0.5 - 0.5 * np.cos(np.pi * t)  # noqa: E731

#: Pixels whose signal is under this many dark-noise sigmas are drawn as lost in noise.
NOISE_SIGMAS = 3.0


@cache
def _frames():
    paths = sorted((hdrps.scene_dir(SCENE_SLUG) / "raw").glob("*.NEF"))
    if not paths:
        raise FileNotFoundError("run `uv run sensing-download` first")
    return sorted((hdrps.load_raw(p) for p in paths), key=lambda f: f.shutter or 0.0)


@cache
def _developed(index: int, width: int = 640) -> np.ndarray:
    """One exposure through the camera's own white balance, no auto brightening."""
    with rawpy.imread(str(_frames()[index].path)) as raw:
        rgb = raw.postprocess(use_camera_wb=True, output_bps=8, no_auto_bright=True, half_size=True)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    return cv2.resize(
        bgr, (width, round(bgr.shape[0] * width / bgr.shape[1])), interpolation=cv2.INTER_AREA
    )


def _shutter(s: float) -> str:
    return f"1/{round(1 / s)} s" if s < 1 else f"{s:g} s"


def _numbers():
    return json.loads(Path(SENSING_NUMBERS_JSON).read_text())["noise"]["channels"]["G1"]


#: A green photosite reads as clipped at 99% of the unit's measured clip level. Saturated
#: green photosites in these frames read 3869-3876 DN, a few counts under the measured
#: level (3880), so an exact comparison would never fire.
CLIP_FRACTION = 0.99


# --- hub: the bracket -----------------------------------------------------------------


def exposure_sweep():
    """All eighteen exposures in order, from 1/800 s to 30 s, developed the same way."""
    frames = []
    for i, f in enumerate(_frames()):
        img = _developed(i).copy()
        label(img, f"exposure {i + 1} of 18: {_shutter(f.shutter)}", (12, 32), 0.7)
        frames += [img] * 4
    return frames, 8


# --- lesson 1: a pixel is a counter ---------------------------------------------------


def counts_grid(patch_number: int = 22, n: int = 5):
    """A 5x5 block of raw green counts on one grey patch, as the exposure doubles."""
    frames_raw = _frames()
    box = next(p for p in charts.patches("bright") if p.number == patch_number).box
    cx, cy = (box[0] + box[2]) // 4, (box[1] + box[3]) // 4  # G1 plane is half resolution
    clip = CLIP_FRACTION * _numbers()["clip_level_dn"]
    frames = []
    for i, f in enumerate(frames_raw):
        g1 = noise.channel(f.counts, "G1").astype(np.int32) - int(f.black_level[1])
        block = g1[cy - n // 2 : cy + n // 2 + 1, cx - n // 2 : cx + n // 2 + 1]
        if block.mean() < 8 or block.max() >= clip - f.black_level[1]:
            continue
        photo = _developed(i).copy()
        s = photo.shape[1] / f.counts.shape[1]
        x0, y0, x1, y1 = (np.array(box) * s).astype(int)
        cv2.rectangle(photo, (x0 - 4, y0 - 4), (x1 + 4, y1 + 4), AMBER, 2)
        label(photo, _shutter(f.shutter), (12, 32), 0.7)
        cell, pad = 64, 14
        grid = np.full((photo.shape[0], cell * n + 2 * pad, 3), BG, np.uint8)
        top = (photo.shape[0] - cell * n) // 2
        for r in range(n):
            for c in range(n):
                x, y = pad + c * cell, top + r * cell
                cv2.rectangle(grid, (x, y), (x + cell - 2, y + cell - 2), (60, 60, 60), 1)
                cv2.putText(
                    grid,
                    str(block[r, c]),
                    (x + 6, y + 38),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    WHITE,
                    1,
                    cv2.LINE_AA,
                )
        label(grid, f"raw counts, mean {block.mean():.0f}", (pad, top - 14), 0.55, box=False)
        img = np.hstack([photo, grid])
        frames += [img] * 3
    return frames, 2


# --- lesson 2: clipped and lost ---------------------------------------------------------


def clip_and_noise():
    """Each exposure with clipped pixels in red and pixels lost in noise in blue."""
    nums = _numbers()
    clip, sigma = CLIP_FRACTION * nums["clip_level_dn"], nums["dark_corner_sigma_dn"]
    frames = []
    for i, f in enumerate(_frames()):
        img = _developed(i).copy()
        g1 = noise.channel(f.counts, "G1").astype(np.float32)
        signal = g1 - f.black_level[1]
        h, w = img.shape[:2]
        clipped = (
            cv2.resize((g1 >= clip).astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST) > 0
        )
        lost = (
            cv2.resize(
                (signal < NOISE_SIGMAS * sigma).astype(np.uint8),
                (w, h),
                interpolation=cv2.INTER_NEAREST,
            )
            > 0
        )
        img[clipped] = (0.35 * img[clipped] + 0.65 * np.array(RED)).astype(np.uint8)
        img[lost] = (0.35 * img[lost] + 0.65 * np.array(BLUE)).astype(np.uint8)
        label(
            img,
            f"{_shutter(f.shutter)}: {clipped.mean():.0%} clipped, {lost.mean():.0%} lost in noise",
            (12, 32),
            0.65,
        )
        frames += [img] * 3
    return frames, 6


# --- lesson 3: the curve between sensor and file ---------------------------------------


def _curve_inset(mix: float, size: int = 150) -> np.ndarray:
    """Output against input: the straight line (raw) and the sRGB curve, blended by `mix`."""
    img = np.full((size, size, 3), 30, np.uint8)
    x = np.linspace(0, 1, 100)
    srgb = np.where(x <= 0.0031308, 12.92 * x, 1.055 * x ** (1 / 2.4) - 0.055)
    for ys, colour in ((x, (130, 130, 130)), (srgb, (130, 130, 130))):
        pts = np.stack([8 + x * (size - 16), size - 8 - ys * (size - 16)], 1)
        cv2.polylines(img, [np.rint(pts).astype(np.int32)], False, colour, 1, cv2.LINE_AA)
    cur = (1 - mix) * x + mix * srgb
    pts = np.stack([8 + x * (size - 16), size - 8 - cur * (size - 16)], 1)
    cv2.polylines(img, [np.rint(pts).astype(np.int32)], False, AMBER, 2, cv2.LINE_AA)
    return img


def response_curve(index: int = 9):
    """One exposure eased from linear (proportional to light) to the developed file."""
    with rawpy.imread(str(_frames()[index].path)) as raw:
        lin = raw.postprocess(
            use_camera_wb=True, output_bps=8, no_auto_bright=True, half_size=True, gamma=(1, 1)
        )
        dev = raw.postprocess(use_camera_wb=True, output_bps=8, no_auto_bright=True, half_size=True)

    def size(im):
        return cv2.resize(
            cv2.cvtColor(im, cv2.COLOR_RGB2BGR),
            (640, round(im.shape[0] * 640 / im.shape[1])),
            interpolation=cv2.INTER_AREA,
        )

    lin, dev = size(lin), size(dev)
    ts = (
        [0.0] * 10
        + [EASE(i / 19) for i in range(20)]
        + [1.0] * 10
        + [1 - EASE(i / 19) for i in range(20)]
    )
    frames = []
    for t in ts:
        img = cv2.addWeighted(lin, 1 - t, dev, t, 0)
        inset = _curve_inset(t)
        img[
            img.shape[0] - inset.shape[0] - 10 : img.shape[0] - 10,
            img.shape[1] - inset.shape[1] - 10 : img.shape[1] - 10,
        ] = inset
        label(
            img,
            "raw: proportional to light" if t < 0.5 else "developed: what a JPEG holds",
            (12, 32),
            0.7,
        )
        frames.append(img)
    return frames, 10


# --- lesson 4: merge ---------------------------------------------------------------------


def hdr_merge():
    """The bracket flicks past, then settles on the merge of all eighteen, tone-mapped."""
    from fundamentals_lab.cli.sensing_figures import _tone_map

    fr = _frames()
    exposures = [(f.counts, f.shutter) for f in fr]
    planes = {}
    for name in ("R", "G1", "B"):
        planes[name] = hdr.merge(exposures, name, noise.clip_level(fr[-1].counts, name)).plane
    with rawpy.imread(str(fr[0].path)) as raw:
        wb = [float(v) for v in raw.camera_whitebalance[:3]]
    merged = _tone_map(np.dstack([planes["B"] * wb[2], planes["G1"] * wb[1], planes["R"] * wb[0]]))
    merged = cv2.resize(merged, _developed(0).shape[1::-1], interpolation=cv2.INTER_AREA)
    frames = []
    for i, f in enumerate(fr):
        img = _developed(i).copy()
        label(img, f"{_shutter(f.shutter)}: no single exposure holds it all", (12, 32), 0.65)
        frames += [img] * 2
    for k in range(8):
        frames.append(
            cv2.addWeighted(_developed(len(fr) - 1), 1 - (k + 1) / 8, merged, (k + 1) / 8, 0)
        )
    done = merged.copy()
    label(done, "all 18 merged, then tone-mapped for the screen", (12, 32), 0.65)
    frames += [done] * 24
    return frames, 8


# --- lesson 5: the mosaic ------------------------------------------------------------------


def bayer_zoom(out: int = 400):
    """Zoom into the lit chart until single photosites show, as the mosaic, then demosaiced."""
    from fundamentals_lab.config import CHART_REFERENCE_FRAME

    f = _frames()[CHART_REFERENCE_FRAME["bright"]]
    by_number = {p.number: p for p in charts.patches("bright")}
    # The border between two coloured patches, so the mosaic has three colours to show.
    (ax, ay), (bx, by) = by_number[14].centre, by_number[15].centre
    cx, cy = int((ax + bx) / 2), int((ay + by) / 2)
    raw = f.counts.astype(np.float32) - float(np.mean(f.black_level))
    h, w = raw.shape
    final = 24
    region = raw[cy - final : cy + final, cx - final : cx + final]
    gain = 1.0 / max(np.percentile(region, 99), 1.0)
    mosaic = np.zeros((h, w, 3), np.float32)
    channel_index = {"B": 0, "G": 1, "R": 2}
    for k, ch in enumerate(f.cfa_pattern):
        dy, dx = divmod(k, 2)
        mosaic[dy::2, dx::2, channel_index[ch]] = raw[dy::2, dx::2]
    mosaic = (np.clip(mosaic * gain, 0, 1) ** (1 / 2.2) * 255).astype(np.uint8)
    with rawpy.imread(str(f.path)) as r:
        rgb = r.postprocess(use_camera_wb=True, output_bps=8, no_auto_bright=True, gamma=(1, 1))
    demosaiced = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR).astype(np.float32)
    ref = max(np.percentile(demosaiced[cy - final : cy + final, cx - final : cx + final], 99), 1.0)
    demosaiced = (np.clip(demosaiced / ref, 0, 1) ** (1 / 2.2) * 255).astype(np.uint8)
    aspect = 1.5
    spans = np.geomspace(h * 0.95, final, 30)

    def crop(img, span):
        s = int(span)
        sw = int(s * aspect)
        x0 = int(np.clip(cx - sw / 2, 0, w - sw))
        y0 = int(np.clip(cy - s / 2, 0, h - s))
        interp = cv2.INTER_NEAREST if s < 200 else cv2.INTER_AREA
        return cv2.resize(
            img[y0 : y0 + s, x0 : x0 + sw], (int(out * aspect), out), interpolation=interp
        )

    frames = []
    for s in spans:
        img = crop(demosaiced, s)
        label(img, "the photograph", (12, 32), 0.7)
        frames.append(img)
    last_photo, last_mosaic = crop(demosaiced, spans[-1]), crop(mosaic, spans[-1])
    for k in range(10):
        frames.append(cv2.addWeighted(last_photo, 1 - (k + 1) / 10, last_mosaic, (k + 1) / 10, 0))
    img = last_mosaic.copy()
    label(img, "what the sensor recorded: one colour per photosite", (12, 32), 0.6)
    frames += [img] * 16
    for k in range(10):
        frames.append(cv2.addWeighted(last_mosaic, 1 - (k + 1) / 10, last_photo, (k + 1) / 10, 0))
    img = last_photo.copy()
    label(img, "demosaiced: the two missing colours interpolated", (12, 32), 0.6)
    frames += [img] * 16
    return frames, 10


#: name -> (function, output width px, libwebp quality)
TABLE = {
    "exposure-sweep": (exposure_sweep, 600, 50),
    "counts-grid": (counts_grid, 700, 55),
    "clip-and-noise": (clip_and_noise, 560, 42),
    "response-curve": (response_curve, 600, 50),
    "hdr-merge": (hdr_merge, 600, 50),
    "bayer-zoom": (bayer_zoom, 600, 50),
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
