"""Lesson 4's numbers: two models, one set of fisheye photographs."""

import click
from loguru import logger

from fundamentals_lab.formation import numbers
from fundamentals_lab.formation.dataset import fetch_fisheye
from fundamentals_lab.formation.fisheye import (
    FISHEYE_BOARD,
    divergence_table,
    find_corners,
    fit_fisheye,
    fit_pinhole,
)


@click.command()
def fisheye() -> None:
    """Fit pinhole and Kannala-Brandt to the same lens; extend formation_numbers.json."""
    paths = fetch_fisheye()

    obj_points, img_points, size, used = find_corners(paths, FISHEYE_BOARD)
    logger.info(f"board {FISHEYE_BOARD} found in {len(used)}/{len(paths)} views")
    result = {
        "views": len(used),
        "image_size": list(size),
        "pinhole": fit_pinhole(obj_points, img_points, size),
        "fisheye": fit_fisheye(obj_points, img_points, size),
    }
    result["board_inner_corners"] = list(FISHEYE_BOARD)
    result["source"] = (
        "jakarto3d/py-OCamCalib test_images/fish_1 (GPL-2.0, not redistributed)"
    )
    result["divergence"] = divergence_table(result["pinhole"], result["fisheye"])

    numbers.save("fisheye", result)

    p, f = result["pinhole"], result["fisheye"]
    logger.info(
        f"pinhole+Brown-Conrady RMS {p['rms_reprojection_error_px']:.3f} px "
        f"vs Kannala-Brandt RMS {f['rms_reprojection_error_px']:.3f} px "
        f"over {result['views']} views"
    )
    for row in result["divergence"]:
        logger.info(
            f"  theta {row['theta_deg']:>2}deg: pinhole {row['pinhole_r_px']:10.1f} px, "
            f"fisheye {row['fisheye_r_px']:7.1f} px"
        )
    logger.info(f"wrote {numbers.FORMATION_NUMBERS_JSON}")


if __name__ == "__main__":
    fisheye()
