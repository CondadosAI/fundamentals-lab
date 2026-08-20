"""L1 — what `imread` hands you, measured rather than described.

Three claims live here and each one is a number a reader can print for themselves:
what the three representations cost in memory, what one named pixel reads in each,
and what integer arithmetic does at the top of the range.
"""

import cv2
import numpy as np

from fundamentals_lab.imagedata.scene import Frame, exr_facts, plane
from fundamentals_lab.sensing import charts

PROBE_PATCH = 19  # the lit chart's white patch: bright, neutral, and named in unit 1.2


def _wrap(value: int, add: np.uint8) -> np.uint8:
    """uint8 addition, allowed to overflow, because the overflow is what we measure."""
    with np.errstate(over="ignore"):
        return np.uint8(np.uint8(value) + add)


def _probe_pixel(frame: Frame) -> tuple[int, int]:
    """A fixed, named point to read in every representation: patch 19's centre."""
    patch = next(p for p in charts.patches("bright") if p.number == PROBE_PATCH)
    x, y = patch.centre
    # Even coordinates, so the mosaic reading lands on the same colour every time --
    # an odd row would read a green photosite and an even one red, and the "same
    # pixel in three files" comparison would quietly be comparing three things.
    return int(round(y)) & ~1, int(round(x)) & ~1


def representations(frame: Frame) -> dict:
    y, x = _probe_pixel(frame)
    rows = [
        {
            "name": "raw mosaic",
            "source": f"{frame.name}, LibRaw visible area",
            "shape": list(frame.mosaic.shape),
            "dtype": str(frame.mosaic.dtype),
            "channels": 1,
            "bits_used": 12,
            "bytes_in_memory": int(frame.mosaic.nbytes),
            "value_at_probe": [int(frame.mosaic[y, x])],
            "note": "one number per photosite, still mosaiced; the only measurement of the three",
        },
        {
            "name": "developed 16-bit",
            "source": "rawpy postprocess, output_bps=16",
            "shape": list(frame.rgb16.shape),
            "dtype": str(frame.rgb16.dtype),
            "channels": 3,
            "bits_used": 16,
            "bytes_in_memory": int(frame.rgb16.nbytes),
            "value_at_probe": [int(v) for v in frame.rgb16[y, x]],
            "note": "RGB order; two of every three numbers are interpolated",
        },
        {
            "name": "developed 8-bit",
            "source": "rawpy postprocess, output_bps=8",
            "shape": list(frame.rgb8.shape),
            "dtype": str(frame.rgb8.dtype),
            "channels": 3,
            "bits_used": 8,
            "bytes_in_memory": int(frame.rgb8.nbytes),
            "value_at_probe": [int(v) for v in frame.rgb8[y, x]],
            "note": "what imread would hand you; the tone curve is already baked in",
        },
    ]
    return {
        "probe_pixel_row_col": [y, x],
        "probe_patch": f"bright:{PROBE_PATCH}",
        "rows": rows,
        "exr": exr_facts(),
        "file_bytes_nef": frame.file_bytes,
        "megapixels": round(frame.rgb8.shape[0] * frame.rgb8.shape[1] / 1e6, 2),
    }


def channel_order(frame: Frame) -> dict:
    """The BGR/RGB swap, as the two numbers that differ.

    Stating "OpenCV is BGR" convinces nobody. Printing the same pixel read both ways,
    and the fraction of the frame where the two orderings disagree, does.
    """
    y, x = _probe_pixel(frame)
    rgb = frame.rgb8
    bgr = frame.bgr8
    differing = int((rgb[:, :, 0] != rgb[:, :, 2]).sum())
    return {
        "probe_rgb": [int(v) for v in rgb[y, x]],
        "probe_bgr": [int(v) for v in bgr[y, x]],
        "pixels_where_r_differs_from_b": differing,
        "fraction_of_frame": round(differing / (rgb.shape[0] * rgb.shape[1]), 4),
    }


def integer_arithmetic(frame: Frame) -> dict:
    """What "+60" does to a bright patch, three ways.

    NumPy wraps on uint8 overflow, OpenCV saturates, and a float pipeline does neither
    until you cast back. The three answers to the same instruction are the lesson, and
    the wrap is the one that turns a highlight black.
    """
    patch = next(p for p in charts.patches("bright") if p.number == PROBE_PATCH)
    block = patch.sample(frame.rgb8).astype(np.uint8)
    add = np.uint8(60)
    wrapped = (block + add).astype(np.uint8)  # NumPy: modulo 256
    saturated = cv2.add(block, np.full_like(block, add))  # OpenCV: clipped at 255
    floated = np.clip(block.astype(np.float32) + float(add), 0, 255)

    brightest = int(block.max())
    return {
        "patch": f"bright:{PROBE_PATCH}",
        "pixels": int(block.size),
        "added": int(add),
        "input_mean": round(float(block.mean()), 1),
        "input_max": brightest,
        "numpy_wrap_mean": round(float(wrapped.mean()), 1),
        "opencv_saturate_mean": round(float(saturated.mean()), 1),
        "float_then_clip_mean": round(float(floated.mean()), 1),
        "values_that_wrapped": int((block.astype(int) + int(add) > 255).sum()),
        "of_values": int(block.size),
        # The single number that makes it concrete: the brightest value in the patch,
        # put through each of the three routes.
        "brightest_value": {
            "before": brightest,
            # The overflow is the finding, so it is allowed rather than warned about.
            "numpy_wrap": int(_wrap(brightest, add)),
            "opencv_saturate": int(cv2.add(np.uint8([[brightest]]), np.uint8([[add]]))[0, 0]),
            "float_then_clip": int(min(255, brightest + int(add))),
        },
    }


def views_and_copies(frame: Frame) -> dict:
    """A slice shares memory; fancy indexing does not.

    The measurable version of the warning: write into each, then read the original
    back. Taken from the lit patch rather than the frame corner, because the corner of
    this photograph is black and a write into black proves nothing.

    Sampled from `bgr8` -- the array `imread` would have handed over -- and reported as
    the whole pixel rather than one channel. Reporting a single channel here would
    quietly depend on which order it was stored in, which is the bug two sections up.
    """
    patch = next(p for p in charts.patches("bright") if p.number == PROBE_PATCH)
    x0, y0, _, _ = patch.box
    region = frame.bgr8[y0 : y0 + 64, x0 : x0 + 64]

    sliced = region.copy()
    before_slice = [int(v) for v in sliced[0, 0]]
    view = sliced[0:8, 0:8]
    view[:] = 0
    after_slice = [int(v) for v in sliced[0, 0]]

    fancied = region.copy()
    before_fancy = [int(v) for v in fancied[0, 0]]
    fancy = fancied[[0, 1, 2]]
    fancy[:] = 0
    after_fancy = [int(v) for v in fancied[0, 0]]

    return {
        "sampled_at_row_col": [y0, x0],
        "channel_order": "BGR, as imread returns it",
        "slice": {
            "shares_memory": True,
            "before": before_slice,
            "after_writing_zero_to_the_view": after_slice,
        },
        "fancy_index": {
            "shares_memory": False,
            "before": before_fancy,
            "after_writing_zero_to_the_copy": after_fancy,
        },
    }


def mosaic_sampling(frame: Frame) -> dict:
    """How many photosites each colour actually gets.

    L1 states it; L2 spends it. Green is sampled twice as often as red or blue, so
    "the green channel" of a developed image is half measurement and half guess, while
    red and blue are three quarters guess.
    """
    total = frame.mosaic.size
    out = {}
    for colour in ("R", "G1", "G2", "B"):
        out[colour] = {
            "shape": list(plane(frame, colour).shape),
            "photosites": int(plane(frame, colour).size),
            "share": round(plane(frame, colour).size / total, 3),
        }
    out["green_share"] = round(
        (plane(frame, "G1").size + plane(frame, "G2").size) / total, 3
    )
    out["interpolated_fraction_of_developed_image"] = round(
        1 - (total / (frame.rgb8.shape[0] * frame.rgb8.shape[1] * 3)), 3
    )
    return out
