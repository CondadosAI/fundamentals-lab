"""Lesson 3's numbers: how far the measured lens moves a pixel, and the cost of fixing it."""

import click
import numpy as np
from loguru import logger

from fundamentals_lab.config import FORMATION_NUMBERS_JSON
from fundamentals_lab.formation import numbers
from fundamentals_lab.formation.distortion import displacement_table, undistort_cost


@click.command()
def distort() -> None:
    """Measure distortion displacement and undistortion crop; extend formation_numbers.json."""
    if not FORMATION_NUMBERS_JSON.exists():
        raise click.ClickException("run if-calibrate first -- this reads its output")
    calib = numbers.load()["calibration"]
    K = np.array(calib["K"], dtype=np.float64)
    dist = np.array(calib["dist"], dtype=np.float64).reshape(1, -1)
    size = tuple(calib["image_size"])

    table = displacement_table(K, dist, size)
    cost = undistort_cost(K, dist, size)
    numbers.save("distortion", table | cost)

    logger.info(
        f"max corner shift {table['max_corner_shift_px']:.2f} px "
        f"on a {size[0]}x{size[1]} frame"
    )
    logger.info(f"horizontal field of view before undistortion: {cost['hfov_deg_before']:.2f} deg")
    for label in ("same_k", "alpha0", "alpha1"):
        c = cost[label]
        logger.info(
            f"  {label}: fx {c['new_fx']:.2f}, hfov {c['hfov_deg_after']:.2f} deg, "
            f"{c['empty_output_pct']:.2f}% of the output has no source pixel"
        )
    logger.info(f"wrote {FORMATION_NUMBERS_JSON}")


if __name__ == "__main__":
    distort()
