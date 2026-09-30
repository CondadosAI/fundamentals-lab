"""OpenCV in Practice, module M1: fix a badly exposed photo.

The scene is the HDR Photographic Survey's Luxo Double Checker bracket (Mark Fairchild;
research and non-commercial publication, cleared for the educational tracks), the same
eighteen raw exposures unit 1.2 measures. Four of them are developed the same way
(camera white balance, no auto-brightening) and published as the module's data:

    reference.webp  5 s     the exposure the fixes are scored against
    dark.webp       1.3 s   about two stops under
    darker.webp     1/2 s   about three stops under
    bright.webp     20 s    two stops over, with the lamp and its pool of light clipped

Every number the posts print is computed here from the published WebP files, decoded
the way the page decodes them, so a reader who runs a cell gets the page's number.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
from loguru import logger

from fundamentals_lab.config import OUTPUT_DIR

#: Developed frames, by index into the 18-exposure bracket sorted by shutter time.
FRAMES = {"reference": 12, "dark": 10, "darker": 9, "bright": 14}
WIDTH = 960
#: WebP quality of the published files. High enough that the numbers barely move;
#: they are computed on the decoded files regardless.
QUALITY = 90
#: A pixel counts as clipped when its grey value reaches this, as lesson 2's cell counts it.
CLIP = 250

ASSET_DIR = OUTPUT_DIR / "practice" / "m1"
NUMBERS_JSON = OUTPUT_DIR / "practice_m1_numbers.json"


def build_assets(out_dir: Path = ASSET_DIR) -> dict[str, Path]:
    """Develop the four frames and write them as the module's WebP files."""
    from fundamentals_lab.sensing.media import _developed, _frames, _shutter

    out_dir.mkdir(parents=True, exist_ok=True)
    frames = _frames()
    paths = {}
    for name, index in FRAMES.items():
        img = _developed(index, WIDTH)
        path = out_dir / f"{name}.webp"
        cv2.imwrite(str(path), img, [cv2.IMWRITE_WEBP_QUALITY, QUALITY])
        paths[name] = path
        logger.info(f"{name}: frame {index}, {_shutter(frames[index].shutter)}, {img.shape}")
    return paths


def load(name: str, asset_dir: Path = ASSET_DIR) -> np.ndarray:
    img = cv2.imread(str(asset_dir / f"{name}.webp"))
    if img is None:
        raise FileNotFoundError(f"{name}.webp: run `uv run practice-m1 --assets` first")
    return img


def mae(a: np.ndarray, b: np.ndarray) -> float:
    """Mean absolute difference over every pixel and channel, on the 0–255 scale."""
    return float(np.abs(a.astype(np.int16) - b.astype(np.int16)).mean())


def gamma_lut(g: float) -> np.ndarray:
    # Written the way lesson 4's cell writes it, so the two give the same table.
    return ((np.arange(256) / 255) ** g * 255).round().astype(np.uint8)


def fixes(img: np.ndarray) -> dict[str, tuple[np.ndarray, float | None]]:
    """The fixes the module teaches, each as a function of its one parameter."""
    return {
        "add": lambda b: cv2.add(img, np.full_like(img, int(b))),
        "multiply": lambda a: cv2.convertScaleAbs(img, alpha=a, beta=0),
        "gamma": lambda g: cv2.LUT(img, gamma_lut(g)),
    }


def on_luma(img: np.ndarray, op) -> np.ndarray:
    """Apply a greyscale operation to the brightness channel only, keeping colour."""
    ycc = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    ycc[..., 0] = op(ycc[..., 0])
    return cv2.cvtColor(ycc, cv2.COLOR_YCrCb2BGR)


GRIDS = {
    "add": [float(b) for b in range(0, 151, 5)],
    "multiply": [round(a / 10, 1) for a in range(2, 61)],
    "gamma": [round(g / 100, 2) for g in range(20, 301, 5)],
}


def best(img: np.ndarray, ref: np.ndarray, name: str) -> dict:
    fn = fixes(img)[name]
    scored = [(mae(fn(p), ref), p) for p in GRIDS[name]]
    err, p = min(scored)
    return {"param": p, "mae": round(err, 1)}


def stats(img: np.ndarray) -> dict:
    grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return {
        "shape": list(img.shape),
        "mean_grey": round(float(grey.mean()), 1),
        "clipped_pct": round(100 * float((grey >= CLIP).mean()), 1),
        "dark_pct": round(100 * float((grey <= 5).mean()), 1),
    }


def experiments() -> dict:
    ref = load("reference")
    out: dict = {"clip_level": CLIP, "frames": {}, "fixes": {}}
    for name in FRAMES:
        out["frames"][name] = stats(load(name))
    for name in ("dark", "darker", "bright"):
        img = load(name)
        row = {"none": round(mae(img, ref), 1)}
        for fix in ("add", "multiply", "gamma"):
            row[fix] = best(img, ref, fix)
        row["equalize"] = round(mae(on_luma(img, cv2.equalizeHist), ref), 1)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        row["clahe"] = round(mae(on_luma(img, clahe.apply), ref), 1)
        out["fixes"][name] = row
    # What cannot come back: pixels clipped in the bright frame that the reference
    # still resolves, which is detail no fix of the bright frame can recover.
    grey = lambda name: cv2.cvtColor(load(name), cv2.COLOR_BGR2GRAY)  # noqa: E731
    lost = (grey("bright") >= CLIP) & (grey("reference") < CLIP)
    out["unrecoverable_pct"] = round(100 * float(lost.mean()), 1)
    NUMBERS_JSON.write_text(json.dumps(out, indent=1) + "\n")
    logger.info(f"wrote {NUMBERS_JSON}")
    return out


# --- media: the opening image of every post, and the hub's animation ------------------

MEDIA_DIR = OUTPUT_DIR / "practice" / "m1" / "media"


def _half(img: np.ndarray, width: int = 480) -> np.ndarray:
    return cv2.resize(
        img, (width, round(img.shape[0] * width / img.shape[1])), interpolation=cv2.INTER_AREA
    )


def _pair(left: np.ndarray, right: np.ndarray, a: str, b: str) -> np.ndarray:
    from fundamentals_lab.alignment.media import label

    lh, rh = _half(left), _half(right)
    if rh.shape[0] != lh.shape[0]:
        rh = cv2.resize(rh, (lh.shape[1], lh.shape[0]), interpolation=cv2.INTER_AREA)
    lh, rh = lh.copy(), rh.copy()
    label(lh, a, (12, 30), 0.6)
    label(rh, b, (12, 30), 0.6)
    return np.hstack([lh, np.full((lh.shape[0], 6, 3), 40, np.uint8), rh])


def _pixel_grid(img: np.ndarray, centre=(320, 480), n=(6, 8), cell=56) -> np.ndarray:
    """The grey values of a small patch, drawn as a grid of numbers."""
    grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    r0, c0 = centre[0] - n[0] // 2, centre[1] - n[1] // 2
    out = np.full((n[0] * cell, n[1] * cell, 3), 255, np.uint8)
    for i in range(n[0]):
        for j in range(n[1]):
            v = int(grey[r0 + i, c0 + j])
            y, x = i * cell, j * cell
            out[y : y + cell, x : x + cell] = v
            ink = (0, 0, 0) if v > 110 else (255, 255, 255)
            cv2.putText(
                out,
                str(v),
                (x + 8, y + cell // 2 + 7),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                ink,
                1,
                cv2.LINE_AA,
            )
            cv2.rectangle(out, (x, y), (x + cell - 1, y + cell - 1), (150, 150, 150), 1)
    return out


def _histogram(grey: np.ndarray, size=(480, 320)) -> np.ndarray:
    hist = cv2.calcHist([grey], [0], None, [256], [0, 256]).ravel()
    w, h = size
    out = np.full((h, w, 3), 255, np.uint8)
    top = hist.max()
    for v, c in enumerate(hist):
        x = int(v * w / 256)
        cv2.rectangle(out, (x, h - 1), (x + 1, h - 1 - int((h - 40) * c / top)), (70, 70, 70), -1)
    return out


def build_media(out_dir: Path = MEDIA_DIR) -> dict[str, Path]:
    from fundamentals_lab.alignment.media import encode

    out_dir.mkdir(parents=True, exist_ok=True)
    dark, darker, ref = load("dark"), load("darker"), load("reference")
    n = experiments()["fixes"]
    fixed = cv2.convertScaleAbs(dark, alpha=2.4)
    stills = {
        "pixels": _pair(
            dark, _pixel_grid(dark), "dark.webp", "grey values around row 320, column 480"
        ),
        "histogram": _pair(
            dark,
            _histogram(cv2.cvtColor(dark, cv2.COLOR_BGR2GRAY)),
            "dark.webp",
            "its histogram, 0 to 255",
        ),
        "multiply": _pair(
            dark,
            fixed,
            f"as shot: distance {n['dark']['none']}",
            f"times 2.4: distance {n['dark']['multiply']['mae']}",
        ),
        "gamma": _pair(
            darker,
            cv2.LUT(darker, gamma_lut(0.40)),
            f"as shot: distance {n['darker']['none']}",
            f"gamma 0.40: distance {n['darker']['gamma']['mae']}",
        ),
        "equalize": _pair(
            dark,
            on_luma(dark, cv2.equalizeHist),
            f"as shot: distance {n['dark']['none']}",
            f"equalizeHist: distance {n['dark']['equalize']}",
        ),
        "save": _pair(
            fixed, fixed[150:450, 0:330], "fixed, 960 x 639", "cropped: rows 150-449, columns 0-329"
        ),
    }
    paths = {}
    for name, img in stills.items():
        p = out_dir / f"{name}.webp"
        cv2.imwrite(str(p), img, [cv2.IMWRITE_WEBP_QUALITY, 82])
        paths[name] = p
    # Hub: the multiplier climbing from 1.0 to 2.4, the distance to the reference falling,
    # then the result beside the reference.
    frames = []
    for a in [round(1.0 + 0.1 * k, 1) for k in range(15)]:
        f = cv2.convertScaleAbs(dark, alpha=a)
        img = _pair(
            f, ref, f"times {a:.1f}: distance {mae(f, ref):.1f}", "reference (5 s exposure)"
        )
        frames += [img] * (8 if a in (1.0, 2.4) else 2)
    frames += [frames[-1]] * 10
    paths["fix"] = encode(frames, 6, out_dir / "fix.webp", 966, 60)
    for p in paths.values():
        logger.info(f"{p.name}: {p.stat().st_size / 1e3:.0f} KB")
    return paths


# --- covers: backgrounds for the title card (scripts/gen-thumbnails.mjs in the site) ---

COVER_DIR = OUTPUT_DIR / "covers"
COVER_BG = (24, 14, 11)  # the site's #0b0e18 in BGR


def _cover_bg(img: np.ndarray, w: int = 1600, h: int = 900, right: float = 0.58) -> np.ndarray:
    """Place an image in the right part of a dark 1600x900 canvas, fading into the text area."""
    canvas = np.full((h, w, 3), COVER_BG, np.uint8)
    tw = int(w * right)
    th = round(img.shape[0] * tw / img.shape[1])
    if th > h:
        th, tw = h, round(img.shape[1] * h / img.shape[0])
    fit = cv2.resize(img, (tw, th), interpolation=cv2.INTER_AREA)
    y, x = (h - th) // 2, w - tw
    canvas[y : y + th, x : x + tw] = fit
    band = tw // 3
    fade = np.linspace(0.0, 1.0, band)[None, :, None]
    region = canvas[y : y + th, x : x + band].astype(float)
    canvas[y : y + th, x : x + band] = (region * fade + np.array(COVER_BG) * (1 - fade)).astype(
        np.uint8
    )
    return canvas


def build_covers(out_dir: Path = COVER_DIR) -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    dark, darker = load("dark"), load("darker")
    fixed = cv2.convertScaleAbs(dark, alpha=2.4)
    grey = cv2.cvtColor(dark, cv2.COLOR_BGR2GRAY)
    backgrounds = {
        "fix-badly-exposed-photo-opencv": np.hstack([dark[:, :480], fixed[:, 480:]]),
        "opencv-read-image-pixels": _pixel_grid(dark, n=(6, 9)),
        "opencv-image-histogram": _histogram(grey, (960, 540)),
        "opencv-brightness-contrast": fixed,
        "opencv-gamma-correction": cv2.LUT(darker, gamma_lut(0.40)),
        "opencv-histogram-equalization": on_luma(dark, cv2.equalizeHist),
        "opencv-crop-resize-save": fixed[150:450, 0:330],
    }
    paths = {}
    for slug, img in backgrounds.items():
        p = out_dir / f"{slug}.png"
        cv2.imwrite(str(p), _cover_bg(img))
        paths[slug] = p
    logger.info(f"wrote {len(paths)} cover backgrounds to {out_dir}")
    return paths
