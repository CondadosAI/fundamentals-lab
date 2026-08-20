"""The colour filter array, and what interpolating through it costs.

Unit 1.2's lesson 5. It sits here rather than in `sensing/` because it needs the
developed image as well as the mosaic, and every other consumer of both lives in this
package.

The measurement needs a reference that was never interpolated, and one is available
without leaving the frame: every 2x2 cell of a Bayer mosaic contains a real red, a real
blue and two real greens, so binning the mosaic 2x2 gives a genuine colour image at half
resolution with no guessing in it. Demosaicing at full resolution and then binning the
result the same way puts the two on the same grid, and the difference is what
interpolation invented.
"""

import cv2
import numpy as np

from fundamentals_lab.imagedata.scene import Frame, cfa_offsets, plane
from fundamentals_lab.sensing import charts


def binned_truth(frame: Frame) -> np.ndarray:
    """A half-resolution colour image built only from measured photosites.

    No interpolation anywhere: R is the red photosite of each cell, B is the blue one,
    G is the mean of the two greens. Returned in RGB order on the raw 12-bit scale.
    """
    offsets = cfa_offsets(frame)
    red = plane(frame, "R").astype(np.float64)
    blue = plane(frame, "B").astype(np.float64)
    green = 0.5 * (plane(frame, "G1").astype(np.float64) + plane(frame, "G2").astype(np.float64))
    assert set(offsets) == {"R", "G1", "G2", "B"}
    return np.dstack([red, green, blue])


def demosaic_error(frame: Frame) -> dict:
    """How far a demosaiced pixel sits from one that was actually measured.

    Reported in two places on purpose: over the whole frame, and inside the chart's
    flat patches. Interpolation is nearly free on a flat patch and expensive at an
    edge, so a single frame-wide number would hide the entire effect.
    """
    demosaiced = cv2.cvtColor(frame.mosaic, cv2.COLOR_BayerBG2RGB).astype(np.float64)
    # Bin the demosaiced image the same 2x2 way, so both live on the half-resolution
    # grid and the comparison is not also a comparison of two different scales.
    binned = 0.25 * (
        demosaiced[0::2, 0::2] + demosaiced[0::2, 1::2]
        + demosaiced[1::2, 0::2] + demosaiced[1::2, 1::2]
    )
    truth = binned_truth(frame)
    height = min(binned.shape[0], truth.shape[0])
    width = min(binned.shape[1], truth.shape[1])
    binned, truth = binned[:height, :width], truth[:height, :width]

    difference = binned - truth
    full_scale = float(frame.white_level)

    # The same comparison restricted to the chart's patch interiors, which are flat.
    flat = []
    for patch in charts.patches("bright"):
        x0, y0, x1, y1 = (v // 2 for v in patch.box)
        block = difference[y0:y1, x0:x1]
        if block.size:
            flat.append(np.sqrt((block**2).mean()))

    # And an edge-heavy region: the gaps between patches, where the demosaicer guesses.
    gradient = cv2.Sobel(cv2.cvtColor(
        np.clip(truth / full_scale * 255, 0, 255).astype(np.uint8), cv2.COLOR_RGB2GRAY),
        cv2.CV_64F, 1, 1, ksize=3)
    edges = np.abs(gradient) > np.percentile(np.abs(gradient), 99)

    return {
        "reference": "2x2 binning of the mosaic — every value a measured photosite",
        "method": "OpenCV bilinear demosaic, binned the same way, differenced on the half-res grid",
        "grid": [height, width],
        "full_scale_dn": full_scale,
        "rms_whole_frame_dn": round(float(np.sqrt((difference**2).mean())), 3),
        "rms_chart_patches_dn": round(float(np.mean(flat)), 3),
        "rms_top_1pct_gradient_dn": round(float(np.sqrt((difference[edges] ** 2).mean())), 3),
        "max_abs_dn": round(float(np.abs(difference).max()), 1),
        "patches_measured": len(flat),
        "edge_pixels": int(edges.sum()),
    }


def sampling_shares(frame: Frame) -> dict:
    """What fraction of a developed image was measured rather than interpolated."""
    total = frame.mosaic.size
    shares = {name: plane(frame, name).size / total for name in ("R", "G1", "G2", "B")}
    return {
        "cfa_pattern": [list(row) for row in frame.cfa_pattern],
        "colour_desc": frame.colour_desc,
        "red_share": round(shares["R"], 3),
        "green_share": round(shares["G1"] + shares["G2"], 3),
        "blue_share": round(shares["B"], 3),
        "values_measured": total,
        "values_in_developed_image": int(frame.rgb8.size),
        "interpolated_fraction": round(1 - total / frame.rgb8.size, 3),
    }
