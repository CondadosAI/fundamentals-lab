"""Every number unit 1.2 publishes, computed from the eighteen raw exposures.

Writes `output/sensing_numbers.json`. Each section corresponds to one lesson: what a
count is, what its noise is, what the developed image did to it, and what the merged
bracket says the light was — checked, in that last case, against a colorimeter.
"""

import json
import platform
import sys
from dataclasses import asdict

import click
import cv2
import numpy as np
import rawpy
from loguru import logger

from fundamentals_lab.config import (
    CALIBRATION_ANCHOR,
    CHART_CORNERS,
    CHART_REFERENCE_FRAME,
    D2X_LUMINANCE_ROW,
    NEUTRAL_ROW,
    OUTPUT_DIR,
    SCENE,
    SCENE_SLUG,
)
from fundamentals_lab.sensing import charts, hdr, hdrps, noise, response

CHANNELS = ("R", "G1", "G2", "B")
DARK_CORNER = (100, 100, 900, 900)  # a patch of the unlit background, in the shortest frame


def _environment() -> dict:
    return {
        "python": sys.version.split()[0],
        "platform": f"{platform.system()} {platform.release()}",
        "numpy": np.__version__,
        "opencv": cv2.__version__,
        "rawpy": rawpy.__version__,
        "libraw": ".".join(str(v) for v in rawpy.libraw_version),
    }


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    rank = lambda values: np.argsort(np.argsort(values))  # noqa: E731
    return float(np.corrcoef(rank(a), rank(b))[0, 1])


@click.command()
@click.option("--slug", default=SCENE_SLUG)
def sensing_experiments(slug: str) -> None:
    """Run every measurement of unit 1.2 and write the artifact."""
    paths = sorted((hdrps.scene_dir(slug) / "raw").glob("*.NEF"))
    if len(paths) < 4:
        raise click.ClickException("run `uv run sensing-download` first")
    frames = [hdrps.load_raw(path) for path in paths]
    frames.sort(key=lambda frame: frame.shutter or 0.0)
    metered_points = hdrps.metered_points(slug)
    metered = {point.key: point.luminance for point in metered_points}
    all_patches = charts.all_patches()
    logger.info(f"{len(frames)} exposures, {len(metered_points)} metered points")

    # --- lesson 1: what the file says, and what the counts do ------------------
    longest = frames[-1]
    ceilings = {name: float(noise.channel(longest.counts, name).max()) for name in CHANNELS}
    frame_rows = []
    for frame in frames:
        plane = noise.channel(frame.counts, "G1").astype(np.float64)
        frame_rows.append(
            {
                "file": frame.path.name,
                "shutter_s": frame.shutter,
                "iso": frame.iso,
                "black_level": list(frame.black_level),
                "white_level_file": frame.white_level,
                "cfa": frame.cfa_pattern,
                "median_dn_g1": round(float(np.median(plane)), 2),
                "clipped_fraction_g1": round(float((plane >= ceilings["G1"] * 0.98).mean()), 5),
            }
        )

    # --- chart geometry, and the check that it is right ------------------------
    chart_checks = {}
    for chart, index in CHART_REFERENCE_FRAME.items():
        with rawpy.imread(str(frames[index].path)) as raw:
            developed = raw.postprocess(use_camera_wb=True, output_bps=8, no_auto_bright=True)
        grey = cv2.cvtColor(cv2.cvtColor(developed, cv2.COLOR_RGB2BGR), cv2.COLOR_BGR2GRAY)
        patches = charts.patches(chart)
        sampled = np.array([float(patch.sample(grey).mean()) for patch in patches])
        reference = np.array([metered[patch.key] for patch in patches])
        chart_checks[chart] = {
            "reference_frame": frames[index].path.name,
            "shutter_s": frames[index].shutter,
            "corners_px": CHART_CORNERS[chart],
            "spearman_pixel_vs_metered": round(_spearman(sampled, reference), 4),
            "metered_range_cd_m2": [float(reference.min()), float(reference.max())],
        }
        logger.info(f"{chart} chart: Spearman {chart_checks[chart]['spearman_pixel_vs_metered']}")

    # --- lesson 2: the photon transfer curve -----------------------------------
    repeats = [frame for frame in frames if frame.shutter == frames[-1].shutter]
    if len(repeats) < 3:
        raise click.ClickException("this scene no longer has three frames at one exposure")
    noise_section = {
        "repeat_frames": [frame.path.name for frame in repeats],
        "repeat_shutter_s": repeats[0].shutter,
        "estimator": "median absolute deviation of frame differences, Gaussian-corrected",
        "channels": {},
    }
    for name in CHANNELS:
        saturation = ceilings[name] * noise.CLIP_MARGIN
        points = noise.transfer_points([frame.counts for frame in repeats], name, saturation)
        fitted = noise.fit(points)
        cross = noise.fit(points, cross_check=True)
        dark = noise.dark_noise_bound(frames[0].counts, DARK_CORNER, name)
        patch_pts = noise.patch_points(
            [frame.counts for frame in repeats], all_patches, name, saturation
        )
        noise_section["channels"][name] = {
            "clip_level_dn": ceilings[name],
            "gain_dn_per_electron": round(fitted.gain_dn_per_electron, 4),
            "gain_stderr": round(fitted.gain_stderr, 4),
            "electrons_per_dn": round(fitted.electrons_per_dn, 3),
            "gain_cross_check_dn_per_electron": round(cross.gain_dn_per_electron, 4),
            "intercept_dn2": round(fitted.intercept_dn2, 1),
            "intercept_stderr_dn2": round(fitted.intercept_stderr, 1),
            "r_squared": round(fitted.r_squared, 4),
            "bins": fitted.points,
            "signal_range_dn": [round(v, 1) for v in fitted.signal_range_dn],
            "dark_corner_sigma_dn": round(dark, 3),
            "dark_corner_sigma_electrons": round(fitted.electrons(dark), 2),
            "full_well_electrons": round(fitted.full_well(ceilings[name])),
            "dynamic_range_stops_lower_bound": round(
                fitted.dynamic_range_stops(ceilings[name], dark), 2
            ),
            "unclipped_patches": len(patch_pts),
            "curve": [asdict(point) for point in points],
            "patch_points": [asdict(point) for point in patch_pts],
        }
    logger.info(
        "gain (G1) {:.4f} DN/e-, full well {} e-",
        noise_section["channels"]["G1"]["gain_dn_per_electron"],
        noise_section["channels"]["G1"]["full_well_electrons"],
    )

    # --- lesson 3: the response curve ------------------------------------------
    index = CHART_REFERENCE_FRAME["bright"]
    with rawpy.imread(str(frames[index].path)) as raw:
        developed = raw.postprocess(use_camera_wb=True, output_bps=8, no_auto_bright=True)
    grey = cv2.cvtColor(cv2.cvtColor(developed, cv2.COLOR_RGB2BGR), cv2.COLOR_BGR2GRAY)
    neutral = [p for p in charts.patches("bright") if p.number in NEUTRAL_ROW]
    points = response.response_points(frames[index].counts, grey, neutral)
    gamma = response.fit_gamma(points, ceilings["G1"])
    relative = np.array([point.linear_dn / ceilings["G1"] for point in points])
    srgb = response.srgb_encode(relative)
    response_section = {
        "frame": frames[index].path.name,
        "shutter_s": frames[index].shutter,
        "ceiling_dn": ceilings["G1"],
        "gamma": round(gamma.gamma, 3),
        "gamma_stderr": round(gamma.stderr, 3),
        "scale": round(gamma.scale, 3),
        "r_squared": round(gamma.r_squared, 4),
        "mean_abs_difference_from_srgb_code_values": round(
            float(np.mean(np.abs(np.array([p.code_value for p in points]) - srgb))), 2
        ),
        "ramp": [
            {
                "key": point.key,
                "linear_dn": point.linear_dn,
                "relative_to_saturation": round(float(rel), 4),
                "code_value": point.code_value,
                "gamma_fit_code": round(float(fit_code), 1),
                "srgb_code": round(float(srgb_code), 1),
            }
            for point, rel, fit_code, srgb_code in zip(
                points, relative, gamma.predict(relative), srgb, strict=True
            )
        ],
    }
    logger.info(f"response: gamma {gamma.gamma:.3f}, R2 {gamma.r_squared:.4f}")

    # --- lesson 4: merge, then check against the colorimeter -------------------
    with rawpy.imread(str(frames[0].path)) as raw:
        white_balance = tuple(float(v) for v in raw.camera_whitebalance[:3])
    exposures = [(frame.counts, frame.shutter) for frame in frames]
    radiances = {
        name: hdr.merge(exposures, name, ceilings[name] * noise.CLIP_MARGIN) for name in CHANNELS
    }
    luminances = hdr.luminance(radiances, all_patches, white_balance, D2X_LUMINANCE_ROW)
    scale, predictions = hdr.calibrate(luminances, metered, CALIBRATION_ANCHOR)
    errors = np.array([prediction.error_stops for prediction in predictions])
    neutral_keys = {f"{chart}:{number}" for chart in CHART_CORNERS for number in NEUTRAL_ROW}
    neutral_errors = np.array(
        [p.error_stops for p in predictions if p.key in neutral_keys]
    )
    green_only = hdr.sample(radiances["G1"], all_patches)
    _, green_predictions = hdr.calibrate(green_only, metered, CALIBRATION_ANCHOR)
    green_errors = np.array([p.error_stops for p in green_predictions])
    hdr_section = {
        "frames": len(exposures),
        "anchor_patch": CALIBRATION_ANCHOR,
        "anchor_metered_cd_m2": metered[CALIBRATION_ANCHOR],
        "scale_cd_m2_per_unit": scale,
        "white_balance_as_shot": white_balance,
        "luminance_row": list(D2X_LUMINANCE_ROW),
        "points": len(predictions),
        "median_abs_error_stops": round(float(np.median(np.abs(errors))), 3),
        "mean_abs_error_stops": round(float(np.mean(np.abs(errors))), 3),
        "max_abs_error_stops": round(float(np.max(np.abs(errors))), 3),
        "neutral_median_abs_error_stops": round(float(np.median(np.abs(neutral_errors))), 3),
        "neutral_points": len(neutral_errors),
        "green_only_median_abs_error_stops": round(float(np.median(np.abs(green_errors))), 3),
        "metered_range_cd_m2": [
            min(p.metered_cd_m2 for p in predictions),
            max(p.metered_cd_m2 for p in predictions),
        ],
        "predictions": [
            {
                "key": p.key,
                "metered_cd_m2": p.metered_cd_m2,
                "predicted_cd_m2": round(p.predicted_cd_m2, 4),
                "error_stops": round(p.error_stops, 3),
            }
            for p in sorted(predictions, key=lambda p: -p.metered_cd_m2)
        ],
    }
    logger.info(
        "merge: median |error| {:.3f} stops over {} points ({:.3f} on the neutral rows)",
        hdr_section["median_abs_error_stops"],
        hdr_section["points"],
        hdr_section["neutral_median_abs_error_stops"],
    )

    artifact = {
        "scene": SCENE,
        "source": {
            "survey": "Mark Fairchild's HDR Photographic Survey",
            "terms": "research and non-commercial publication; images not redistributed",
            "capture": hdrps.scene_metadata(slug),
        },
        "environment": _environment(),
        "frames": frame_rows,
        "charts": chart_checks,
        "noise": noise_section,
        "response": response_section,
        "hdr": hdr_section,
        "metered_points": [asdict(point) for point in metered_points],
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    target = OUTPUT_DIR / "sensing_numbers.json"
    target.write_text(json.dumps(artifact, indent=2) + "\n")
    logger.info(f"wrote {target}")


if __name__ == "__main__":
    sensing_experiments()
