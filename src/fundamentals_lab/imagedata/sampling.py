"""L2 — where the samples sit, and what falls between them.

The lesson needs one thing the rest of the scene cannot give it: something periodic.
The table under the lamp is a ribbed surface, and its ribbing is fine enough that
decimating the frame drops below two samples per cycle -- so aliasing can be measured
here rather than asserted, and predicted before it is measured.

The *why* -- spectra, the sampling theorem, what a low-pass filter is -- belongs to
unit 2.3. This module produces the phenomenon and the arithmetic, and stops.
"""

import cv2
import numpy as np

from fundamentals_lab.config import (
    D2X_EFFECTIVE_PX,
    D2X_FOCAL_LENGTH_MM,
    D2X_SENSOR_MM,
    DECIMATION_FACTORS,
    GRAIN_CROP,
    GRAIN_STRIP,
    GRAIN_WINDOW,
)
from fundamentals_lab.imagedata.scene import Frame
from fundamentals_lab.sensing import charts

# The three ways to shrink an image, and what each one is: a bare sample, OpenCV's
# default, and a box prefilter. AREA is the reference because averaging over the
# footprint is what a smaller sensor would have measured.
METHODS = {
    "nearest": cv2.INTER_NEAREST,
    "bilinear (cv2 default)": cv2.INTER_LINEAR,
    "area": cv2.INTER_AREA,
}


def geometry() -> dict:
    """What one photosite covers, from the camera's published dimensions.

    Nikon's effective figures, not the array LibRaw returns: LibRaw exposes a masked
    border, so dividing the sensor width by 4312 would quietly shrink the pitch.
    """
    width_mm, height_mm = D2X_SENSOR_MM
    width_px, height_px = D2X_EFFECTIVE_PX
    pitch_um = width_mm / width_px * 1000
    pitch_v_um = height_mm / height_px * 1000
    angle_rad = (pitch_um / 1000) / D2X_FOCAL_LENGTH_MM
    return {
        "sensor_mm": list(D2X_SENSOR_MM),
        "effective_px": list(D2X_EFFECTIVE_PX),
        "focal_length_mm": D2X_FOCAL_LENGTH_MM,
        "pitch_um": round(pitch_um, 3),
        "pitch_vertical_um": round(pitch_v_um, 3),
        "pitch_agrees_within_um": round(abs(pitch_um - pitch_v_um), 4),
        "angular_footprint_rad": round(angle_rad, 8),
        "angular_footprint_arcmin": round(np.degrees(angle_rad) * 60, 3),
    }


def patch_footprint(frame: Frame) -> dict:
    """How many samples one chart patch gets, and how many its border gets.

    The point of the pair: a flat region is sampled thousands of times and its edge is
    sampled once, so every error a sampling grid makes lives at the edges.
    """
    patch = next(p for p in charts.patches("bright") if p.number == 19)
    x0, y0, x1, y1 = patch.box
    return {
        "patch": patch.key,
        "sampling_box_px": [x1 - x0, y1 - y0],
        "samples_in_box": int((x1 - x0) * (y1 - y0)),
        "note": "the box is the middle 40% of the patch; the patch itself is wider",
    }


# The illumination across this strip is far stronger than the ribbing on it, and it is
# smooth. Subtracting a Gaussian-smoothed copy removes everything coarser than a few
# tens of pixels and leaves the ribbing, so the peak search does not have to compete
# with the lamp. The band below is where a ribbing can plausibly live.
HIGHPASS_SIGMA_PX = 6.0
SEARCH_PERIOD_PX = (3.0, 40.0)


def _highpass(signal: np.ndarray, sigma: float = HIGHPASS_SIGMA_PX) -> np.ndarray:
    """Whatever is finer than `sigma`, with the lamp's gradient taken out."""
    smooth = cv2.GaussianBlur(
        signal.reshape(-1, 1).astype(np.float64), (0, 0), sigmaX=1e-6, sigmaY=sigma
    ).ravel()
    return signal - smooth


def _spectrum(signal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    windowed = (signal - signal.mean()) * np.hanning(len(signal))
    return np.abs(np.fft.rfft(windowed)), np.fft.rfftfreq(len(signal))


def _peak(signal: np.ndarray, band_px: tuple[float, float] | None = None) -> float:
    """The strongest frequency in a profile, in cycles per sample.

    `band_px` limits the search to a period range. The base measurement uses it -- a
    ribbing is not 300 pixels wide -- and the decimated measurements deliberately do
    not, because the whole question is where the frequency moved to.
    """
    spectrum, freqs = _spectrum(signal)
    mask = freqs > 0
    if band_px is not None:
        low, high = band_px
        mask &= (freqs >= 1.0 / high) & (freqs <= 1.0 / low)
    if not mask.any():
        raise ValueError("no frequency bins inside the requested band")
    candidates = np.where(mask)[0]
    return float(freqs[candidates[np.argmax(spectrum[candidates])]])


def _fold(frequency: float, factor: int) -> float:
    """Where a frequency reappears after keeping every `factor`-th sample.

    In the decimated grid the frequency is `frequency * factor` cycles per new sample.
    Anything past 0.5 has nowhere to go and reflects back -- that reflection is the
    alias, and it is a frequency the scene never contained.
    """
    scaled = (frequency * factor) % 1.0
    return scaled if scaled <= 0.5 else 1.0 - scaled


def period_drift(ribbing: np.ndarray, window: int = 192, step: int = 96) -> list[dict]:
    """How the ribbing's image period changes down the crop.

    It is a physical grating seen at a shallow angle, so the far end of the table is
    foreshortened and the near end is not. The drift is why the frequency measurement
    below runs on a window rather than on the whole strip -- and it is worth publishing
    on its own, because it is what perspective does to a texture.
    """
    x0, y0, _, _ = GRAIN_CROP
    out = []
    for start in range(0, len(ribbing) - window + 1, step):
        segment = ribbing[start : start + window]
        out.append(
            {
                "frame_row": int(y0 + start),
                "rows": [int(start), int(start + window)],
                "period_px": round(1 / _peak(segment, band_px=SEARCH_PERIOD_PX), 2),
            }
        )
    return out


def aliasing(frame: Frame) -> dict:
    """The ribbing, its period, and where each decimation puts it.

    Prediction first, measurement second, in that order in the output as well as in
    the lesson: the folded frequency is arithmetic, and the transform of the decimated
    window either lands on it or does not.

    The strip is high-passed once, before any decimation, so that what gets decimated
    is the ribbing alone. Decimating first and detrending afterwards would fold the
    lamp's gradient into the same band as the alias and make the two impossible to
    tell apart.
    """
    x0, y0, x1, y1 = GRAIN_CROP
    grey = cv2.cvtColor(frame.bgr8[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY).astype(np.float64)
    a, b = GRAIN_STRIP
    ribbing = _highpass(grey[:, a:b].mean(axis=1))

    start, length = GRAIN_WINDOW
    window = ribbing[start : start + length]
    base = _peak(window, band_px=SEARCH_PERIOD_PX)

    out = {
        "crop_xyxy": list(GRAIN_CROP),
        "strip_columns": list(GRAIN_STRIP),
        "highpass_sigma_px": HIGHPASS_SIGMA_PX,
        "search_band_period_px": list(SEARCH_PERIOD_PX),
        "drift": period_drift(ribbing),
        "window_rows_in_crop": [start, start + length],
        "window_rows_in_frame": [int(y0 + start), int(y0 + start + length)],
        "window_samples": int(length),
        "fft_bin_cycles_per_px": round(1.0 / length, 5),
        "base_frequency_cycles_per_px": round(base, 5),
        "base_period_px": round(1 / base, 3),
        "nyquist_needs_samples_per_cycle": 2.0,
        "factors": [],
    }
    for factor in DECIMATION_FACTORS:
        decimated = window[::factor]
        measured = _peak(decimated)
        predicted = _fold(base, factor)
        bin_size = 1.0 / len(decimated)
        samples_per_cycle = (1 / base) / factor
        out["factors"].append(
            {
                "factor": factor,
                "samples_per_original_cycle": round(samples_per_cycle, 3),
                "above_nyquist": bool(samples_per_cycle >= 2),
                "predicted_frequency_cycles_per_new_px": round(predicted, 5),
                "measured_frequency_cycles_per_new_px": round(measured, 5),
                "fft_bin_cycles_per_new_px": round(bin_size, 5),
                "difference_in_fft_bins": round(abs(measured - predicted) / bin_size, 2),
                "measured_period_new_px": round(1 / measured, 3),
                "apparent_period_in_original_px": round(factor / measured, 2),
            }
        )
    return out


def decimation_error(frame: Frame) -> dict:
    """What each shrink method costs, against the box-averaged reference.

    Reported on the ribbed crop, where the methods disagree, and on the chart, where
    they mostly do not -- the contrast between the two is the practical rule.
    """
    x0, y0, x1, y1 = GRAIN_CROP
    crop = frame.bgr8[y0:y1, x0:x1]
    height, width = crop.shape[:2]

    rows = []
    for factor in DECIMATION_FACTORS:
        size = (width // factor, height // factor)
        reference = cv2.resize(crop, size, interpolation=cv2.INTER_AREA).astype(np.float64)
        entry = {"factor": factor, "output_px": list(size)}
        for label, flag in METHODS.items():
            if flag == cv2.INTER_AREA:
                continue
            shrunk = cv2.resize(crop, size, interpolation=flag).astype(np.float64)
            error = shrunk - reference
            entry[label] = {
                "rms_vs_area": round(float(np.sqrt((error**2).mean())), 3),
                "max_abs_vs_area": int(np.abs(error).max()),
            }
        rows.append(entry)

    # Same three methods on the chart, where there is nothing above Nyquist to lose.
    patch = next(p for p in charts.patches("bright") if p.number == 19)
    px0, py0, px1, py1 = patch.box
    flat = frame.bgr8[py0:py1, px0:px1]
    flat_rows = {}
    for label, flag in METHODS.items():
        if flag == cv2.INTER_AREA:
            continue
        size = (max(1, flat.shape[1] // 4), max(1, flat.shape[0] // 4))
        shrunk = cv2.resize(flat, size, interpolation=flag).astype(np.float64)
        reference = cv2.resize(flat, size, interpolation=cv2.INTER_AREA).astype(np.float64)
        flat_rows[label] = round(float(np.sqrt(((shrunk - reference) ** 2).mean())), 3)

    return {
        "reference": "cv2.INTER_AREA — a box average over the footprint",
        "ribbed_crop": rows,
        "flat_patch_x4_rms_vs_area": flat_rows,
        "flat_patch": patch.key,
    }
