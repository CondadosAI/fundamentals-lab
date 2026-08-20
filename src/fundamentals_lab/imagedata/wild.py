"""Unit 1.3's "in the wild" half: the same measurements, on photographs.

The Luxo Double Checker scene is a bench setup in a dark room. It measures beautifully
-- there is a colorimeter behind it -- and it looks like nothing anybody photographs.
So every lesson repeats its measurement on a real picture, and the pair is the point:
the lab number says what the effect *is*, the photograph says what it is *worth*.

Every asset here is registered in `cv-assets/registry` with a verified licence and a
sha256 of the exact bytes. Nothing is redistributed from this repository; the fetcher
downloads from Wikimedia Commons and the articles publish derived figures only.
"""

import hashlib
import subprocess
import urllib.parse
import urllib.request
from pathlib import Path

import cv2
import numpy as np

from fundamentals_lab.config import DATA_DIR, IMAGEDATA_CANNY, JPEG_QUALITIES

WILD_DIR = DATA_DIR / "wild"
UA = "CondadosAI-fundamentals-lab/1.0 (https://condados.ai)"
COMMONS = "https://commons.wikimedia.org/wiki/Special:FilePath/"

# filename -> (Commons title, sha256 of the bytes we measured)
ASSETS = {
    "telephone-box.jpg": (
        "Red telephone box, St Paul's Cathedral, London, England, GB, IMG 5182 edit.jpg",
        "8cc6a477a54c1cda",
    ),
    "brick-wall.jpg": ("Surfaces brick wall with unstandard pattern.JPG", "42816f8a7a8599e8"),
    "twilight-sky.jpg": ("Endless expanse (potw2549a).jpg", "3af90571b26b50e5"),
    "souk-fruit.jpg": ("Marrakech souk fruit vendor.jpg", "e103021e703daad5"),
    "colorchecker-photo.jpg": ("ColorChecker100423.jpg", "d0f1d5872005bcb9"),
    "newspaper-1814.jpg": (
        "Departementaal Dagblad van de Zuiderzee. Extra Ordinaire Amsterdamsche Courant. "
        "Ao 1814 No. 7 30 maart 1814, RP-P-OB-87.141.jpg",
        "513f20a9174fb98e",
    ),
    "wagon-wheel.ogv": ("The wagon-wheel effect.ogv", "3b74d192ce875c26"),
    # Units 1.1, 1.2 and 3.1
    "cloister.jpg": (
        "The Cloister Mandapam, in One point perspective.jpg",
        "38757975e752e6b8",
    ),
    "fisheye-kashan.jpg": (
        "Fisheye lens Photography In Iran-Kashan City-Mostafa Meraji-2016-free Pictures-Canon EF 8-15mm lens 03.jpg",
        "20c08a411cb38461",
    ),
    "orion-astro.jpg": (
        "Orion Nebula (M42) – Brod, Dragash – Long-Exposure DSLR Astrophotography.jpg",
        "db8e8fac7d640e6a",
    ),
}


def fetch(name: str) -> Path:
    """Download one registered asset, and check its bytes against the registry."""
    title, prefix = ASSETS[name]
    target = WILD_DIR / name
    if not target.exists():
        WILD_DIR.mkdir(parents=True, exist_ok=True)
        url = COMMONS + urllib.parse.quote(title)
        request = urllib.request.Request(url, headers={"User-Agent": UA})
        with urllib.request.urlopen(request) as response, target.open("wb") as out:
            out.write(response.read())
    digest = hashlib.sha256(target.read_bytes()).hexdigest()
    if not digest.startswith(prefix):
        raise ValueError(
            f"{name}: sha256 starts {digest[:16]}, registry says {prefix}. "
            "The file at the source changed; re-verify the licence before using it."
        )
    return target


def load(name: str) -> np.ndarray:
    image = cv2.imread(str(fetch(name)), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"{name} did not decode")
    return image


# --- L1: the channel swap, on something unmistakable -------------------------
def channel_swap() -> dict:
    image = load("telephone-box.jpg")
    height, width = image.shape[:2]
    blue, _, red = cv2.split(image)
    differing = int((red != blue).sum())

    # The box itself: the reddest region of the frame, found rather than hand-drawn.
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = ((hsv[:, :, 0] < 8) | (hsv[:, :, 0] > 172)) & (hsv[:, :, 1] > 120) & (hsv[:, :, 2] > 60)
    box = image[mask]
    swapped = box[:, ::-1]
    return {
        "asset": "telephone-box.jpg",
        "shape": [height, width, 3],
        "file_bytes": fetch("telephone-box.jpg").stat().st_size,
        "array_bytes": int(image.nbytes),
        "megapixels": round(height * width / 1e6, 1),
        "pixels_where_red_differs_from_blue": differing,
        "fraction_of_frame": round(differing / (height * width), 4),
        "red_mask_pixels": int(mask.sum()),
        "red_mask_mean_bgr": [round(float(v), 1) for v in box.mean(axis=0)],
        "same_pixels_read_as_rgb": [round(float(v), 1) for v in swapped.mean(axis=0)],
        "mean_hue_before_deg": round(float(np.median(hsv[:, :, 0][mask])) * 2, 1),
        "mean_hue_after_deg": round(
            float(np.median(cv2.cvtColor(image[:, :, ::-1], cv2.COLOR_BGR2HSV)[:, :, 0][mask])) * 2, 1
        ),
    }


# --- L2: aliasing on a real periodic surface ---------------------------------
def brick_aliasing() -> dict:
    from fundamentals_lab.imagedata.sampling import _highpass, _peak, _fold

    image = load("brick-wall.jpg")
    grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float64)
    # Courses of brick run horizontally, so the period lives down a column band.
    height = grey.shape[0]
    column = grey[:, grey.shape[1] // 2 - 200 : grey.shape[1] // 2 + 200].mean(axis=1)
    ribbing = _highpass(column, sigma=14.0)
    base = _peak(ribbing, band_px=(8.0, 120.0))

    rows = []
    for factor in (4, 8, 16, 32, 64):
        decimated = ribbing[::factor]
        measured = _peak(decimated)
        predicted = _fold(base, factor)
        per_cycle = (1 / base) / factor
        rows.append({
            "factor": factor,
            "samples_per_original_cycle": round(per_cycle, 3),
            "above_nyquist": bool(per_cycle >= 2),
            "predicted_frequency": round(predicted, 5),
            "measured_frequency": round(measured, 5),
            "apparent_period_in_original_px": round(factor / measured, 1),
        })
    return {
        "asset": "brick-wall.jpg",
        "shape": list(image.shape[:2]),
        "course_period_px": round(1 / base, 2),
        "profile_samples": int(height),
        "factors": rows,
    }


# --- L3: banding on a real sky ------------------------------------------------
def sky_banding() -> dict:
    from fundamentals_lab.imagedata.quantize import requantize

    image = load("twilight-sky.jpg")
    height, width = image.shape[:2]
    # The upper third is sky and nothing else: no stars bright enough to matter, no
    # horizon, no observatory. That is the region a banding claim has to be made on.
    sky = image[: height // 3, :]
    green = sky[:, :, 1].astype(np.float64)

    # The source is already 8-bit, so anything above 8 would be a no-op; the sweep runs
    # down from what the file actually has.
    # Averaged across the width and then smoothed, because a photograph's own noise
    # dithers the gradient: counting code changes on raw pixels counts the noise, which
    # flips a code on almost every row. Smoothing first leaves the ramp, and the bands
    # counted below are bands a viewer would see rather than grain.
    rows = []
    profile = green.mean(axis=1)
    column = cv2.GaussianBlur(profile.reshape(-1, 1), (0, 0), sigmaX=1e-6, sigmaY=8.0).ravel()
    for bits in (8, 7, 6, 5, 4):
        step = 256 / 2**bits
        reduced = np.clip(np.rint(green / step) * step, 0, 255)
        # A contour is a row where the quantised column changes value: on a smooth
        # gradient those are the visible bands, and counting them is counting the bands.
        down = np.clip(np.rint(column / step) * step, 0, 255)
        contours = int((np.diff(down) != 0).sum())
        rows.append({
            "bits": bits,
            "step_code_values": round(step, 2),
            "distinct_values": int(len(np.unique(reduced))),
            "bands_down_the_sky": contours,
            "mean_band_height_px": round(len(down) / max(contours, 1), 1),
            "note": "bands counted on the width-averaged, smoothed profile, so grain is not counted as banding",
        })
    gradient = float(green[0].mean() - green[-1].mean())
    return {
        "asset": "twilight-sky.jpg",
        "shape": [height, width],
        "sky_region": [0, 0, width, height // 3],
        "sky_pixels": int(green.size),
        "gradient_top_to_bottom_code_values": round(gradient, 1),
        "distinct_values_in_source": int(len(np.unique(np.rint(green)))),
        "std_code_values": round(float(green.std()), 2),
        "rows": rows,
    }


# --- L4: a hue window on a real market stall ---------------------------------
def hue_threshold() -> dict:
    image = load("souk-fruit.jpg")
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hue, saturation, value = cv2.split(hsv)
    total = hue.size

    # Oranges sit near hue 10-25 in OpenCV's halved scale (20-50 degrees). Widening the
    # window is what shows the problem: lemons and pomegranates are the neighbours.
    rows = []
    for low, high in ((10, 20), (8, 25), (5, 30), (0, 40)):
        mask = (hue >= low) & (hue <= high) & (saturation > 90) & (value > 60)
        rows.append({
            "hue_window_8bit": [low, high],
            "hue_window_deg": [low * 2, high * 2],
            "selected_pixels": int(mask.sum()),
            "fraction_of_frame": round(float(mask.sum()) / total, 4),
        })

    # The histogram is the honest picture: if two fruits share a hue, no window separates
    # them, and the counts say by how much they overlap.
    strong = (saturation > 90) & (value > 60)
    histogram = cv2.calcHist([hue], [0], strong.astype(np.uint8), [180], [0, 180]).ravel()
    peaks = np.argsort(histogram)[::-1][:6]
    return {
        "asset": "souk-fruit.jpg",
        "shape": list(image.shape[:2]),
        "saturated_pixels": int(strong.sum()),
        "windows": rows,
        "top_hues_8bit": [int(p) for p in sorted(peaks)],
        "top_hues_deg": [int(p) * 2 for p in sorted(peaks)],
        "histogram_peak_counts": {int(p): int(histogram[p]) for p in sorted(peaks)},
    }


# --- L5: how neutral are the greys that are neutral by construction? ---------
def chart_neutrality() -> dict:
    """A ColorChecker's bottom row is neutral by manufacture.

    That makes it a reference needing no external table: whatever a* and b* come out as,
    the true answer is zero. Any departure is the illuminant, the camera, and the sRGB
    assumption between them -- exactly what unit 1.3's lesson 5 measures against a
    colorimeter, here with the chart supplying its own ground truth.
    """
    image = load("colorchecker-photo.jpg")
    height, width = image.shape[:2]
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2Lab).astype(np.float64)

    # Six patches across the bottom row of a 6x4 chart, sampled from patch centres with
    # a generous inset. The chart fills the frame, so the lattice is the frame.
    rows = []
    for index in range(6):
        cx = int(width * (index + 0.5) / 6)
        cy = int(height * 3.5 / 4)
        half = int(min(width / 6, height / 4) * 0.18)
        block = lab[cy - half : cy + half, cx - half : cx + half].reshape(-1, 3).mean(axis=0)
        lightness = block[0] * 100 / 255
        a_star, b_star = block[1] - 128, block[2] - 128
        rows.append({
            "patch": 19 + index,
            "L": round(float(lightness), 1),
            "a": round(float(a_star), 2),
            "b": round(float(b_star), 2),
            "chroma_from_neutral": round(float(np.hypot(a_star, b_star)), 2),
        })
    chroma = [r["chroma_from_neutral"] for r in rows]
    return {
        "asset": "colorchecker-photo.jpg",
        "shape": [height, width],
        "reference": "the chart's bottom row is neutral by manufacture: a* = b* = 0",
        "rows": rows,
        "mean_chroma": round(float(np.mean(chroma)), 2),
        "max_chroma": round(float(np.max(chroma)), 2),
        "note": "chroma here is the distance from neutral in the a*b* plane, which is the "
                "Delta-E this photograph can be held to without an external reference table",
    }


# --- L6: what compression does to dense text ---------------------------------
def text_compression() -> dict:
    image = load("newspaper-1814.jpg")
    # A column of body text, where the strokes are a couple of pixels wide.
    height, width = image.shape[:2]
    crop = image[height // 3 : height // 3 + 900, width // 4 : width // 4 + 900]
    low, high = IMAGEDATA_CANNY
    reference_edges = cv2.Canny(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY), low, high) > 0

    from fundamentals_lab.imagedata.formats import _edge_f1, _psnr

    # Paper: the flat regions between the letters. Ringing lives here -- the letters
    # themselves keep their edges, and what an encoder leaves behind is a halo on the
    # blank space around them. Found by local variance rather than by hand.
    grey = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY).astype(np.float64)
    local = cv2.GaussianBlur(grey**2, (0, 0), 3) - cv2.GaussianBlur(grey, (0, 0), 3) ** 2
    ink = (grey < 120).astype(np.uint8)
    halo = cv2.dilate(ink, np.ones((9, 9), np.uint8)).astype(bool)
    # Two kinds of blank paper: the halo just outside a letter, and paper far from any
    # ink at all. Both are flat in the original, so any difference between them after
    # encoding is the encoder reacting to the letter next door.
    near_ink = (grey > 150) & halo
    paper = (grey > 150) & ~halo & (local < np.percentile(local, 60))

    rows = []
    for quality in JPEG_QUALITIES:
        ok, buffer = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, quality])
        if not ok:
            continue
        decoded = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
        edges = cv2.Canny(cv2.cvtColor(decoded, cv2.COLOR_BGR2GRAY), low, high) > 0
        error = cv2.cvtColor(decoded, cv2.COLOR_BGR2GRAY).astype(np.float64) - grey
        rows.append({
            "quality": quality,
            "bytes": int(buffer.nbytes),
            "compression_ratio": round(crop.nbytes / buffer.nbytes, 1),
            "psnr_db": round(_psnr(crop, decoded), 2),
            "edges": _edge_f1(reference_edges, edges),
            "paper_rms_error": round(float(np.sqrt((error[paper] ** 2).mean())), 3),
            "ringing_rms_beside_letters": round(float(np.sqrt((error[near_ink] ** 2).mean())), 3),
        })
    return {
        "asset": "newspaper-1814.jpg",
        "shape": list(image.shape[:2]),
        "crop": [width // 4, height // 3, 900, 900],
        "uncompressed_bytes": int(crop.nbytes),
        "canny_thresholds": list(IMAGEDATA_CANNY),
        "paper_pixels": int(paper.sum()),
        "paper_pixels_beside_letters": int(near_ink.sum()),
        "rows": rows,
    }


# --- L2 extra: temporal aliasing ---------------------------------------------
def wagon_wheel(start_s: float = 58.0, frames: int = 120) -> dict:
    """The apparent rotation of a spoked wheel, frame to frame.

    A ring of pixels around the hub is unwrapped into a 1-D signal of angle; the shift
    that best lines up two consecutive rings is the apparent rotation. Circular
    cross-correlation gives a signed answer, so a wheel that appears to turn backwards
    reports a negative shift -- which is the whole phenomenon, as a number.

    The disc is found by brightness rather than by motion: it is white on black, and a
    motion centroid locks onto the presenter's hand instead. `start_s` picks a stretch
    where the wheel is already turning fast, which is where the reversal lives.
    """
    path = fetch("wagon-wheel.ogv")
    raw = subprocess.run(
        ["ffmpeg", "-v", "error", "-ss", str(start_s), "-i", str(path), "-vf", "scale=480:270",
         "-frames:v", str(frames), "-f", "rawvideo", "-pix_fmt", "gray", "-"],
        capture_output=True, check=True,
    ).stdout
    stack = np.frombuffer(raw, np.uint8).reshape(-1, 270, 480).astype(np.float64)
    if len(stack) < 8:
        raise ValueError("not enough frames decoded to measure a rotation")

    # The disc: the largest bright blob in the median frame.
    median = np.median(stack, axis=0).astype(np.uint8)
    _, bright = cv2.threshold(median, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(bright, 8)
    biggest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    cx, cy = (float(v) for v in centroids[biggest])
    disc_radius = float(np.sqrt(stats[biggest, cv2.CC_STAT_AREA] / np.pi))
    radius = 0.62 * disc_radius  # inside the rim, across the spokes

    angles = np.linspace(0, 2 * np.pi, 720, endpoint=False)

    def ring(frame):
        y = np.clip((cy + radius * np.sin(angles)).astype(int), 0, frame.shape[0] - 1)
        x = np.clip((cx + radius * np.cos(angles)).astype(int), 0, frame.shape[1] - 1)
        values = frame[y, x]
        return values - values.mean()

    rings = [ring(f) for f in stack]
    spectrum = np.abs(np.fft.rfft(rings[0]))
    spokes = int(np.argmax(spectrum[3:40]) + 3)  # spokes show up as that many cycles

    shifts = []
    for a, b in zip(rings, rings[1:]):
        correlation = np.fft.irfft(np.fft.rfft(b) * np.conj(np.fft.rfft(a)), 720)
        peak = int(np.argmax(correlation))
        step = peak - 720 if peak > 360 else peak
        shifts.append(step * 0.5)  # 720 samples over 360 degrees
    shifts = np.array(shifts, dtype=float)

    # One spoke looks like the next, so the honest bound is the angle between spokes:
    # any apparent rotation is only known modulo that.
    ambiguity = 360.0 / max(spokes, 1)
    wrapped = (shifts + ambiguity / 2) % ambiguity - ambiguity / 2
    return {
        "asset": "wagon-wheel.ogv",
        "start_s": start_s,
        "frames_analysed": int(len(stack)),
        "frame_rate": 24,
        "disc_centre_px": [round(cx, 1), round(cy, 1)],
        "disc_radius_px": round(disc_radius, 1),
        "ring_radius_px": round(radius, 1),
        "spokes_detected": spokes,
        "angle_between_spokes_deg": round(ambiguity, 1),
        "median_apparent_deg_per_frame": round(float(np.median(wrapped)), 2),
        "apparent_rev_per_s": round(float(np.median(wrapped)) * 24 / 360, 3),
        "frames_apparently_backward": int((wrapped < -0.5).sum()),
        "frames_apparently_forward": int((wrapped > 0.5).sum()),
        "note": (
            "apparent rotation is only ever known modulo the angle between spokes; a "
            "negative median means the wheel appears to turn against its true direction"
        ),
    }
