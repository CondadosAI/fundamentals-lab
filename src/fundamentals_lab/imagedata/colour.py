"""L4 — the axes, and how far they are from the colour that was in the room.

Unit 1.2 used the survey's colorimeter readings for luminance alone. Each reading also
carries a CIE 1931 chromaticity, and luminance plus chromaticity is a full XYZ -- which
turns 24 metered patches into a CIELAB reference measured with an instrument that never
saw our code. That is what makes this lesson checkable rather than merely illustrated.

Three paths from photosites to Lab are compared, in increasing order of how much they
know about the camera:

1. naive -- treat the developed 8-bit file as sRGB and convert;
2. published -- Fairchild's own D2x matrix on the linear raw, with one free scale;
3. fitted -- our own 3x3 on the linear raw, fitted on half the patches and scored on
   the half it never saw.

Only the third is a prediction. The first two are stated as what they are.
"""

import json

import cv2
import numpy as np

from fundamentals_lab.config import (
    D2X_PUBLISHED_FIT,
    D2X_XYZ_FROM_RGB_D65,
    D2X_XYZ_FROM_RGB_SCENE,
    SENSING_NUMBERS_JSON,
)
from fundamentals_lab.imagedata.scene import Frame
from fundamentals_lab.sensing import charts

WHITE_PATCH = 19
# Held out by patch number, chosen before any error was computed: every other patch in
# chart order, so the split covers colours and neutrals evenly rather than handing the
# fit all the saturated ones.
TEST_PATCHES = (2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24)

SRGB_FROM_XYZ_D65 = np.array(
    [
        [0.4124564, 0.3575761, 0.1804375],
        [0.2126729, 0.7151522, 0.0721750],
        [0.0193339, 0.1191920, 0.9503041],
    ]
)


def _metered() -> dict[int, dict]:
    points = json.loads(SENSING_NUMBERS_JSON.read_text())["metered_points"]
    return {p["patch"]: p for p in points if p["chart"] == "bright" and p["patch"]}


def xyY_to_XYZ(x: float, y: float, luminance: float) -> np.ndarray:
    """A colorimeter reading as a tristimulus vector.

    Y is the luminance it printed; X and Z follow from the chromaticity it printed
    beside it. No assumption about the display or the camera enters here.
    """
    return np.array([x * luminance / y, luminance, (1 - x - y) * luminance / y])


def to_lab(xyz: np.ndarray, white: np.ndarray) -> np.ndarray:
    """CIELAB, against a stated white.

    `white` is an argument rather than a constant because this scene is lit by a
    tungsten lamp: its neutral row meters near x=0.462, y=0.406, nowhere near D65. Lab
    against the wrong white would report a colour cast that is the illuminant, not the
    camera.
    """
    ratio = xyz / white
    delta = 6 / 29
    f = np.where(ratio > delta**3, np.cbrt(ratio), ratio / (3 * delta**2) + 4 / 29)
    return np.array([116 * f[1] - 16, 500 * (f[0] - f[1]), 200 * (f[1] - f[2])])


def delta_e(a: np.ndarray, b: np.ndarray) -> float:
    """CIE 1976 Delta-E*ab: the straight-line distance in Lab.

    The 1976 formula, not one of its successors, because it is the one Fairchild's
    published characterisation reports and the comparison has to be like for like.
    """
    return float(np.linalg.norm(a - b))


def patch_samples(frame: Frame) -> dict:
    """Every patch of the lit chart, in the two forms the paths need.

    `srgb8` is the developed file a reader would load. `linear_rgb` comes from the
    mosaic itself -- red photosites for red, both greens averaged for green -- so it
    is proportional to the light that arrived, with no tone curve on it.
    """
    out = {}
    for patch in charts.patches("bright"):
        srgb = patch.sample(frame.rgb8).reshape(-1, 3).mean(axis=0)
        red = patch.sample_cfa(frame.mosaic, 0, 0).mean()
        green1 = patch.sample_cfa(frame.mosaic, 0, 1).mean()
        green2 = patch.sample_cfa(frame.mosaic, 1, 0).mean()
        blue = patch.sample_cfa(frame.mosaic, 1, 1).mean()
        out[patch.number] = {
            "srgb8": srgb,
            "srgb8_max": float(patch.sample(frame.rgb8).max()),
            "linear_rgb": np.array([red, 0.5 * (green1 + green2), blue]),
        }
    return out


def _srgb_to_linear(code: np.ndarray) -> np.ndarray:
    value = code / 255.0
    return np.where(value <= 0.04045, value / 12.92, ((value + 0.055) / 1.055) ** 2.4)


def _fit_matrix(linear: np.ndarray, xyz: np.ndarray) -> np.ndarray:
    """Least-squares 3x3 from camera RGB to XYZ, no intercept.

    Same shape of model as the published one, so the comparison is between what the
    two matrices know, not between two different kinds of model.
    """
    solution, *_ = np.linalg.lstsq(linear, xyz, rcond=None)
    return solution.T


def summarise_later(errors: dict[int, float], subset: list[int]) -> dict:
    """Mean, median and worst of a set of per-patch errors."""
    values = [errors[n] for n in subset]
    return {
        "patches": len(values),
        "mean": round(float(np.mean(values)), 2),
        "median": round(float(np.median(values)), 2),
        "max": round(float(np.max(values)), 2),
        "worst_patch": int(subset[int(np.argmax(values))]),
    }


def accuracy(frame: Frame) -> dict:
    samples = patch_samples(frame)
    metered = _metered()

    usable = [
        n
        for n in sorted(samples)
        if n in metered and samples[n]["srgb8_max"] < 255 and metered[n]["y"]
    ]
    clipped = [n for n in sorted(samples) if samples[n]["srgb8_max"] >= 255]

    reference = {
        n: xyY_to_XYZ(metered[n]["x"], metered[n]["y"], metered[n]["luminance"])
        for n in usable
    }
    white_metered = xyY_to_XYZ(
        metered[WHITE_PATCH]["x"], metered[WHITE_PATCH]["y"], metered[WHITE_PATCH]["luminance"]
    )
    truth = {n: to_lab(reference[n], white_metered) for n in usable}

    # --- path 1: the developed file, assumed to be sRGB -----------------------
    naive_xyz = {n: SRGB_FROM_XYZ_D65 @ _srgb_to_linear(samples[n]["srgb8"]) for n in usable}
    naive_white = SRGB_FROM_XYZ_D65 @ _srgb_to_linear(samples[WHITE_PATCH]["srgb8"])
    naive = {n: delta_e(truth[n], to_lab(naive_xyz[n], naive_white)) for n in usable}

    # --- path 2: Fairchild's published matrix on the linear raw ---------------
    # Applied four ways rather than one. The document publishes two matrices -- the
    # scene fit and a D65-normalised version -- and says the second is for images "that
    # have been color balanced", so both are tried on raw and on white-balanced linear
    # RGB. One configuration landing badly would be a mistake on our side; four landing
    # in the same place is a result.
    raw = np.array([samples[n]["linear_rgb"] for n in usable])
    balance = np.array(json.loads(SENSING_NUMBERS_JSON.read_text())["hdr"]["white_balance_as_shot"])
    balance = balance / balance[1]
    published_variants = {}
    published = None
    for matrix_name, matrix in (
        ("scene", np.array(D2X_XYZ_FROM_RGB_SCENE)),
        ("d65", np.array(D2X_XYZ_FROM_RGB_D65)),
    ):
        for balance_name, rgb in (("raw", raw), ("white_balanced", raw * balance)):
            predicted = rgb @ matrix.T
            # One free parameter, and only one: the absolute scale. The published
            # matrix is normalised and the per-scene factor for this frame is not
            # published, so the scale is fitted here and said out loud.
            target = np.array([reference[n][1] for n in usable])
            scale = float((target * predicted[:, 1]).sum() / (predicted[:, 1] ** 2).sum())
            xyz = {n: scale * predicted[i] for i, n in enumerate(usable)}
            errors = {n: delta_e(truth[n], to_lab(xyz[n], xyz[WHITE_PATCH])) for n in usable}
            published_variants[f"{matrix_name}_{balance_name}"] = {
                "fitted_scale": round(scale, 6),
                **summarise_later(errors, usable),
            }
            if published is None:
                published = errors

    # --- path 3: our own matrix, fitted on half and scored on the other half ---
    train = [n for n in usable if n not in TEST_PATCHES]
    test = [n for n in usable if n in TEST_PATCHES]
    fitted_matrix = _fit_matrix(
        np.array([samples[n]["linear_rgb"] for n in train]),
        np.array([reference[n] for n in train]),
    )
    fitted_xyz = {n: fitted_matrix @ samples[n]["linear_rgb"] for n in usable}
    fitted_white = fitted_xyz[WHITE_PATCH]
    fitted = {n: delta_e(truth[n], to_lab(fitted_xyz[n], fitted_white)) for n in usable}

    return {
        "chart": "bright",
        "reference": "Konica Minolta CS-100 readings published with the survey: luminance and CIE 1931 x, y",
        "adapting_white": {
            "patch": WHITE_PATCH,
            "metered_xy": [metered[WHITE_PATCH]["x"], metered[WHITE_PATCH]["y"]],
            "metered_luminance_cd_m2": metered[WHITE_PATCH]["luminance"],
            "d65_xy_for_comparison": [0.3127, 0.3290],
        },
        "usable_patches": usable,
        "excluded_because_clipped": clipped,
        "paths": {
            "naive_srgb": {
                "what_it_knows": "nothing about the camera; assumes the developed file is sRGB",
                "free_parameters": 0,
                "all": summarise_later(naive, usable),
                "per_patch": {str(n): round(naive[n], 2) for n in usable},
            },
            "published_matrix": {
                "what_it_knows": "Fairchild's D2x RGB->XYZ characterisation, applied to the linear mosaic",
                "free_parameters": 1,
                "free_parameter": "absolute scale",
                "published_fit_residual_on_its_own_patches": D2X_PUBLISHED_FIT,
                "variants": published_variants,
                "per_patch_scene_raw": {str(n): round(published[n], 2) for n in usable},
            },
            "fitted_held_out": {
                "what_it_knows": "a 3x3 fitted here, on patches it is then not scored on",
                "free_parameters": 9,
                "train_patches": train,
                "test_patches": test,
                "train": summarise_later(fitted, train),
                "test": summarise_later(fitted, test),
                "per_patch": {str(n): round(fitted[n], 2) for n in usable},
            },
        },
        "neutral_row_naive_delta_e": {
            str(n): round(naive[n], 2) for n in (19, 20, 21, 22, 23, 24) if n in naive
        },
    }


def worked_patch(frame: Frame, number: int = 15) -> dict:
    """One patch, converted by hand into every space the lesson names.

    Patch 15 is the chart's red: saturated enough that hue is unambiguous, dark enough
    that nothing clips. The numbers here are what the lesson substitutes into its
    formulas, so they come out of the same sampling every other measurement uses.
    """
    patch = next(p for p in charts.patches("bright") if p.number == number)
    rgb = patch.sample(frame.rgb8).reshape(-1, 3).mean(axis=0)
    r, g, b = (float(v) for v in rgb)
    pixel = np.uint8([[[b, g, r]]])  # OpenCV wants BGR

    luma = 0.299 * r + 0.587 * g + 0.114 * b
    hsv = cv2.cvtColor(pixel, cv2.COLOR_BGR2HSV)[0, 0]
    lab = cv2.cvtColor(pixel, cv2.COLOR_BGR2Lab)[0, 0]
    grey = cv2.cvtColor(pixel, cv2.COLOR_BGR2GRAY)[0, 0]

    maximum, minimum = max(r, g, b) / 255, min(r, g, b) / 255
    chroma = maximum - minimum
    return {
        "patch": patch.key,
        "rgb": [round(r, 1), round(g, 1), round(b, 1)],
        "bgr_as_opencv_sees_it": [round(b, 1), round(g, 1), round(r, 1)],
        "luma_rec601": round(luma, 1),
        "plain_average_for_comparison": round((r + g + b) / 3, 1),
        "opencv_gray": int(grey),
        "hsv_opencv_8bit": [int(v) for v in hsv],
        "hue_degrees": round(float(hsv[0]) * 2, 1),
        "saturation_fraction": round(chroma / maximum, 3),
        "value_fraction": round(maximum, 3),
        "lab_opencv_8bit": [int(v) for v in lab],
        "note": "OpenCV stores 8-bit hue halved, in [0,179]; degrees are twice the stored value",
    }
