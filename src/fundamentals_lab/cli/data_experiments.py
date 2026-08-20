"""Every number unit 1.3 publishes, in one artifact.

Writes `output/image_data_numbers.json`. One section per lesson: what the array is,
where the samples sit, how many levels they get, what colour they are, and what the
file did to all four. Nothing in the posts is quoted that is not in here.
"""

import json
import platform
import sys

import click
import cv2
import numpy as np
from loguru import logger

from fundamentals_lab.config import IMAGEDATA_NUMBERS_JSON, OUTPUT_DIR
from fundamentals_lab.imagedata import (
    arrays,
    colour,
    formats,
    mosaic,
    quantize,
    sampling,
    scene,
    wild,
    wild_units,
)


def _environment() -> dict:
    import PIL
    import rawpy

    return {
        "python": platform.python_version(),
        "platform": f"{platform.system()} {platform.release()}",
        "numpy": np.__version__,
        "opencv": cv2.__version__,
        "rawpy": rawpy.__version__,
        "libraw": ".".join(str(v) for v in rawpy.libraw_version),
        "pillow": PIL.__version__,
    }


@click.command()
@click.option("--frame", default=None, help="NEF to work on; defaults to the unit's frame.")
def data_experiments(frame: str | None) -> None:
    logger.info("loading the frame and developing it twice")
    exposure = scene.load(frame) if frame else scene.load()

    logger.info("L1 — the array")
    l1 = {
        "representations": arrays.representations(exposure),
        "channel_order": arrays.channel_order(exposure),
        "integer_arithmetic": arrays.integer_arithmetic(exposure),
        "views_and_copies": arrays.views_and_copies(exposure),
        "mosaic_sampling": arrays.mosaic_sampling(exposure),
    }

    logger.info("L2 — sampling")
    l2 = {
        "geometry": sampling.geometry(),
        "patch_footprint": sampling.patch_footprint(exposure),
        "aliasing": sampling.aliasing(exposure),
        "decimation_error": sampling.decimation_error(exposure),
    }
    base = l2["aliasing"]["base_period_px"]
    logger.info(f"  ribbing at {base} px per cycle")

    logger.info("L3 — quantization")
    l3 = {
        "error_vs_prediction": quantize.error_vs_prediction(exposure),
        "noise_floor_crossing": quantize.noise_floor_crossing(),
        "separability": quantize.separability(exposure),
        "clipping": quantize.clipping(exposure),
    }
    crossing = l3["noise_floor_crossing"]["range_bits"]
    logger.info(f"  quantization meets the noise floor at {crossing[0]}-{crossing[1]} bits")

    logger.info("unit 1.2 L5 — the colour filter array")
    cfa = {
        "sampling_shares": mosaic.sampling_shares(exposure),
        "demosaic_error": mosaic.demosaic_error(exposure),
    }
    error = cfa["demosaic_error"]
    logger.info(
        f"  demosaic rms {error['rms_chart_patches_dn']} DN on flat patches, "
        f"{error['rms_top_1pct_gradient_dn']} at edges"
    )

    logger.info("L4 — colour")
    l4 = {"accuracy": colour.accuracy(exposure), "worked_patch": colour.worked_patch(exposure)}
    paths = l4["accuracy"]["paths"]
    logger.info(
        f"  naive dE {paths['naive_srgb']['all']['mean']}, "
        f"fitted held-out dE {paths['fitted_held_out']['test']['mean']}"
    )

    logger.info("L5 — file formats (this one encodes the frame nine times, give it a minute)")
    l5 = {
        "sweep": formats.sweep(exposure),
        "chroma_subsampling": formats.chroma_subsampling(exposure),
        "decoders": formats.decoder_disagreement(exposure),
        "exif": formats.exif_orientation(exposure),
    }

    logger.info("in the wild — the same measurements, on photographs")
    in_the_wild = {
        "l1_channel_swap": wild.channel_swap(),
        "l2_brick_aliasing": wild.brick_aliasing(),
        # Two speeds, because the effect is a function of speed and one point would be
        # a coincidence rather than a demonstration.
        "l2_wagon_wheel": [wild.wagon_wheel(start_s=20.0), wild.wagon_wheel(start_s=45.0)],
        "l3_sky_banding": wild.sky_banding(),
        "l4_hue_threshold": wild.hue_threshold(),
        "l5_chart_neutrality": wild.chart_neutrality(),
        "l6_text_compression": wild.text_compression(),
    }
    logger.info(
        f"  sky bands at 6 bits: {in_the_wild['l3_sky_banding']['rows'][2]['bands_down_the_sky']}; "
        f"wheel apparent {in_the_wild['l2_wagon_wheel'][0]['apparent_rev_per_s']} rev/s"
    )

    logger.info("in the wild — units 1.1, 1.2 and 3.1")
    other_units = {
        "unit_1_1": {
            "l1_vanishing_point": wild_units.vanishing_point(),
            "l2_depth_of_field": wild_units.depth_of_field(),
            "l3_vignetting": wild_units.vignetting(),
            "l4_fisheye_image_circle": wild_units.fisheye_image_circle(),
        },
        "unit_1_2": {
            # One photograph, two questions: lesson 1 asks what a count is worth and
            # lesson 2 asks how much it varies, and both are answered on the same sky.
            "l1_l2_astro_noise": wild_units.astro_noise(),
            "l3_response_curve": wild_units.response_curve_from_chart(),
            "l4_single_exposure_range": wild_units.single_exposure_range(),
            "l5_colour_fringing": wild_units.colour_fringing(),
        },
        "unit_3_1": {
            "l1_edge_profile": wild_units.edge_profile(),
            "l2_gradient_orientations": wild_units.gradient_orientations(),
            "l3_zero_crossings": wild_units.zero_crossing_counts(),
            "l4_canny_hysteresis": wild_units.canny_hysteresis(),
            "l5_corner_repeatability": wild_units.corner_repeatability(),
        },
    }
    logger.info(
        f"  vanishing point residual {other_units['unit_1_1']['l1_vanishing_point']['median_distance_px']} px; "
        f"edges {other_units['unit_3_1']['l1_edge_profile']['median_width_px']} px wide"
    )

    payload = {
        "unit": "1.3 — the image as data",
        "scene": "Luxo Double Checker, Mark Fairchild's HDR Photographic Survey",
        "terms": "research and non-commercial publication; images downloaded, never redistributed",
        "frame": exposure.name,
        "environment": _environment(),
        "depends_on": "output/sensing_numbers.json (unit 1.2): gain, dark sigma, clip levels, metered points",
        "l1_array": l1,
        "l2_sampling": l2,
        "l3_quantization": l3,
        "cfa": cfa,
        "l4_colour": l4,
        "l5_formats": l5,
        "in_the_wild": in_the_wild,
        "in_the_wild_other_units": other_units,
        "in_the_wild_assets": "registered in CondadosAI/cv-assets with verified licences and sha256",
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    IMAGEDATA_NUMBERS_JSON.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n")
    logger.info(f"wrote {IMAGEDATA_NUMBERS_JSON}")
    return None


if __name__ == "__main__":
    sys.exit(data_experiments())
