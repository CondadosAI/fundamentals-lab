"""L3 — how many levels a sample deserves.

The NEF is 12-bit, so it is the original every reduction here is measured against.
Two questions, and the second is the one unit 1.2 makes answerable: how large is the
error a quantizer introduces, and at what point does that error disappear underneath
the noise the sensor already had.
"""

import json

import numpy as np

from fundamentals_lab.config import REQUANT_BITS, SENSING_NUMBERS_JSON
from fundamentals_lab.imagedata.scene import Frame, plane
from fundamentals_lab.sensing import charts

SOURCE_BITS = 12  # white level 4095 on this camera, read from the file and checked


def requantize(values: np.ndarray, bits: int, source_bits: int = SOURCE_BITS) -> np.ndarray:
    """Round to `bits` levels and put it back on the original scale.

    Rounding rather than truncating, because that is what an honest converter does and
    it halves the error: truncation biases everything downwards by half a step.
    """
    step = 2 ** (source_bits - bits)
    top = 2**source_bits - 1
    return np.clip(np.rint(values / step) * step, 0, top)


def _sensing_numbers() -> dict:
    if not SENSING_NUMBERS_JSON.exists():
        raise FileNotFoundError(
            f"{SENSING_NUMBERS_JSON} missing — unit 1.3 spends unit 1.2's measurements, "
            "so run `uv run sensing-experiments` first"
        )
    return json.loads(SENSING_NUMBERS_JSON.read_text())


def error_vs_prediction(frame: Frame) -> dict:
    """Measured quantization error against the textbook Delta/sqrt(12).

    Reported twice on purpose. Over the whole plane the prediction should come out
    optimistic: most of this frame is dark, so most values sit in the first few steps
    rather than spread evenly inside them, and the uniform-distribution assumption the
    formula rests on is simply not true there. Over the lit chart's patches, which are
    mid-tones, it has a fair chance.
    """
    green = plane(frame, "G1").astype(np.float64)
    patches = [p for p in charts.patches("bright")]
    patch_pixels = np.concatenate(
        [p.sample_cfa(frame.mosaic, 0, 1).ravel() for p in patches]
    ).astype(np.float64)

    rows = []
    for bits in REQUANT_BITS:
        step = 2 ** (SOURCE_BITS - bits)
        predicted = step / np.sqrt(12)
        full_error = requantize(green, bits) - green
        patch_error = requantize(patch_pixels, bits) - patch_pixels
        rows.append(
            {
                "bits": bits,
                "levels": 2**bits,
                "step_dn": step,
                "predicted_rms_dn": round(float(predicted), 3),
                "measured_rms_whole_plane_dn": round(float(np.sqrt((full_error**2).mean())), 3),
                "measured_rms_chart_patches_dn": round(
                    float(np.sqrt((patch_error**2).mean())), 3
                ),
                "ratio_chart_to_predicted": round(
                    float(np.sqrt((patch_error**2).mean()) / predicted), 3
                ),
            }
        )
    return {
        "source_bits": SOURCE_BITS,
        "white_level": frame.white_level,
        "plane": "G1",
        "plane_pixels": int(green.size),
        "chart_pixels": int(patch_pixels.size),
        "prediction": "step / sqrt(12), the RMS error of a uniform quantizer",
        "rows": rows,
    }


def noise_floor_crossing() -> dict:
    """The bit depth at which quantization error meets the sensor's own noise.

    Unit 1.2 measured the dark-frame sigma of every channel. Setting Delta/sqrt(12)
    equal to that sigma and solving for the bit depth says where a converter stops
    recording the scene and starts recording noise -- and it is the honest answer to
    "how many bits does this camera deserve", which is not the same question as "how
    many bits does the file have".
    """
    channels = _sensing_numbers()["noise"]["channels"]
    full_scale = 2**SOURCE_BITS
    rows = []
    for name, measured in channels.items():
        sigma = float(measured["dark_corner_sigma_dn"])
        step = sigma * np.sqrt(12)
        bits = float(np.log2(full_scale / step))
        rows.append(
            {
                "channel": name,
                "dark_sigma_dn": sigma,
                "step_matching_noise_dn": round(float(step), 3),
                "bits_at_crossing": round(bits, 2),
                "clip_level_dn": measured["clip_level_dn"],
            }
        )
    crossings = [r["bits_at_crossing"] for r in rows]
    return {
        "full_scale_dn": full_scale,
        "source": "unit 1.2, output/sensing_numbers.json (dark_corner_sigma_dn)",
        "rows": rows,
        "range_bits": [round(min(crossings), 2), round(max(crossings), 2)],
        "spare_bits_in_a_12_bit_file": round(SOURCE_BITS - max(crossings), 2),
    }


def separability(frame: Frame) -> dict:
    """How many of the chart's patches survive as distinct values, per bit depth.

    This is the honest form of the banding question on this scene. A textbook banding
    figure wants a broad smooth gradient, and a dark room with one lamp in it does not
    contain one -- the only wide tonal ranges here are edges, and an edge is not a ramp.
    What the scene does have is 24 patches with metered luminances, so the question
    becomes the one bit depth actually answers: at what point do two tones the meter
    can tell apart stop being different numbers.

    The neutral row is the interesting half. Its patches are already close together at
    the dark end, so they collapse first, which is why shadows are where bit depth is
    spent.
    """
    patches = charts.patches("bright")
    green = {p.number: float(p.sample_cfa(frame.mosaic, 0, 1).mean()) for p in patches}
    neutral = [19, 20, 21, 22, 23, 24]

    rows = []
    for bits in (SOURCE_BITS, *REQUANT_BITS):
        step = 2 ** (SOURCE_BITS - bits)
        codes = {n: int(requantize(np.array([v]), bits)[0]) // step for n, v in green.items()}
        collapsed = []
        for a, b in zip(neutral, neutral[1:]):
            if codes[a] == codes[b]:
                collapsed.append(f"{a}+{b}")
        rows.append(
            {
                "bits": bits,
                "step_dn": step,
                "distinct_codes_all_24": len(set(codes.values())),
                "distinct_codes_neutral_row": len({codes[n] for n in neutral}),
                "collapsed_neutral_pairs": collapsed,
                "darkest_patch_code": codes[24],
                "darkest_patch_dn": round(green[24], 1),
            }
        )
    return {
        "chart": "bright",
        "plane": "G1 photosites inside each patch box",
        "neutral_row": neutral,
        "patch_means_dn": {str(n): round(v, 1) for n, v in sorted(green.items())},
        "rows": rows,
    }


def clipping(frame: Frame) -> dict:
    """How much of the frame has already hit the ceiling.

    Quantization loses precision everywhere; clipping loses the value entirely, and no
    bit depth brings it back. The two greens top out below 4095 while red and blue reach
    it, which is unit 1.2's finding, re-counted here on this frame rather than restated.

    Counted at one DN below the level 1.2 derived: that level is where the histogram
    piles up, and on this frame the greens sit exactly one count under it, so testing
    `>= level` would report zero clipped photosites in a channel that is plainly
    saturated.
    """
    channels = _sensing_numbers()["noise"]["channels"]
    rows = []
    for name in ("R", "G1", "G2", "B"):
        values = plane(frame, name)
        level = float(channels[name]["clip_level_dn"])
        at_ceiling = int((values >= level - 1).sum())
        rows.append(
            {
                "channel": name,
                "clip_level_dn": level,
                "counted_at_or_above_dn": level - 1,
                "photosites": int(values.size),
                "at_ceiling": at_ceiling,
                "fraction": round(at_ceiling / values.size, 5),
                "max_dn": int(values.max()),
            }
        )
    return {"frame": frame.name, "rows": rows}
