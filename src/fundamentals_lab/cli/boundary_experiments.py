"""Every number unit 3.2 publishes, in one artifact.

Writes `output/boundary_numbers.json`. Nothing in the posts is quoted that is not
in here.
"""

import json
import platform

import click
import cv2
import numpy as np
from loguru import logger

from fundamentals_lab.boundaries import experiments, wild
from fundamentals_lab.config import BOUNDARY_NUMBERS_JSON, OUTPUT_DIR

SECTIONS = {
    "scene": experiments.scene,
    "fitting": experiments.fitting_lesson,
    "hough": experiments.hough_lesson,
    "probabilistic": experiments.probabilistic_lesson,
    "ght": experiments.ght_lesson,
    "wild": lambda s: wild.all_wild(),
}


@click.command()
@click.option(
    "--only", multiple=True, type=click.Choice(list(SECTIONS)), help="Run these sections only."
)
def boundary_experiments(only: tuple[str, ...]) -> None:
    """Measure the board: fits, the accumulator, restricted voting, PPH and the R-table."""
    s = experiments.Scene()
    numbers = (
        json.loads(BOUNDARY_NUMBERS_JSON.read_text()) if BOUNDARY_NUMBERS_JSON.exists() else {}
    )
    for name in only or SECTIONS:
        logger.info(f"section {name}")
        numbers[name] = SECTIONS[name](s)
    numbers["environment"] = {
        "python": platform.python_version(),
        "opencv": cv2.__version__,
        "numpy": np.__version__,
        "platform": platform.platform(),
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    BOUNDARY_NUMBERS_JSON.write_text(json.dumps(numbers, indent=2) + "\n")
    logger.info(f"wrote {BOUNDARY_NUMBERS_JSON}")
