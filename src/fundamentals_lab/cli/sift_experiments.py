"""Every number unit 3.3 publishes, in one artifact.

Writes `output/sift_numbers.json`. Nothing in the posts is quoted that is not in here.
"""

import json
import platform

import click
import cv2
import numpy as np
from loguru import logger

from fundamentals_lab.config import OUTPUT_DIR, SIFT_NUMBERS_JSON
from fundamentals_lab.features import experiments

SECTIONS = {
    "hub": experiments.hub,
    "keypoints": experiments.keypoints_lesson,
    "scale_space": experiments.scale_space_lesson,
    "detector": experiments.detector_lesson,
    "descriptor": experiments.descriptor_lesson,
    "matching": experiments.matching_lesson,
    "wild": experiments.wild,
}


@click.command()
@click.option(
    "--only", multiple=True, type=click.Choice(list(SECTIONS)), help="Run these sections only."
)
def sift_experiments(only: tuple[str, ...]) -> None:
    """Measure SIFT on the Barcelona pair (and templeRing), stage by stage."""
    numbers = json.loads(SIFT_NUMBERS_JSON.read_text()) if SIFT_NUMBERS_JSON.exists() else {}
    for name in only or SECTIONS:
        logger.info(f"section {name}")
        numbers[name] = SECTIONS[name]()
    numbers["environment"] = {
        "python": platform.python_version(),
        "opencv": cv2.__version__,
        "numpy": np.__version__,
        "platform": platform.platform(),
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    SIFT_NUMBERS_JSON.write_text(json.dumps(numbers, indent=2, ensure_ascii=False) + "\n")
    logger.info(f"wrote {SIFT_NUMBERS_JSON}")
