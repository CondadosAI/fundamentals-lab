"""Lesson 1 and lesson 3's numbers: K, the distortion coefficients, the error."""

import click
from loguru import logger

from fundamentals_lab.config import (
    FORMATION_BOARD,
    FORMATION_NUMBERS_JSON,
    FORMATION_SQUARE_SIZE,
)
from fundamentals_lab.formation import numbers
from fundamentals_lab.formation.calibration import calibrate_camera, find_corners
from fundamentals_lab.formation.dataset import fetch_opencv_calibration


@click.command()
def calibrate() -> None:
    """Calibrate on OpenCV's chessboard set and write output/formation_numbers.json."""
    paths = fetch_opencv_calibration()
    obj_points, img_points, size, used, skipped = find_corners(paths)
    if len(used) < 3:
        raise click.ClickException(f"only {len(used)} usable views; calibration would be junk")
    result = calibrate_camera(obj_points, img_points, size)
    result |= {
        "source": "opencv/opencv samples/data (Apache-2.0)",
        "board_inner_corners": list(FORMATION_BOARD),
        "square_size_units": FORMATION_SQUARE_SIZE,
        "square_size_note": "physical size unpublished; board units. K and dist are unaffected.",
        "views_offered": len(paths),
        "views_used": used,
        "views_skipped": skipped,
    }
    numbers.save("calibration", result)
    logger.info(
        f"fx={result['fx']:.2f} fy={result['fy']:.2f} "
        f"cx={result['cx']:.2f} cy={result['cy']:.2f} "
        f"RMS={result['rms_reprojection_error_px']:.4f} px over {len(used)} views"
    )
    logger.info(f"k1={result['k1']:.5f} k2={result['k2']:.5f} k3={result['k3']:.5f}")
    logger.info(f"wrote {FORMATION_NUMBERS_JSON}")


if __name__ == "__main__":
    calibrate()
