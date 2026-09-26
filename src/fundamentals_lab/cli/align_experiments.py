"""Every number units 3.5 and 4.1-L1 publish, in one artifact.

Writes `output/alignment_numbers.json`. Nothing in the posts is quoted that is not
in here.
"""

import json
import platform

import click
import cv2
import numpy as np
from loguru import logger

from fundamentals_lab.alignment import experiments
from fundamentals_lab.config import ALIGNMENT_NUMBERS_JSON, OUTPUT_DIR

SECTIONS = (
    "scene",
    "homogeneous",
    "linear",
    "affine_vs_projective",
    "dlt",
    "lens",
    "warping",
    "stability",
)


@click.command()
@click.option("--only", multiple=True, type=click.Choice(SECTIONS), help="Run these sections only.")
def align_experiments(only: tuple[str, ...]) -> None:
    """Measure the court: transforms, the DLT, the lens, the warp, and their stability."""
    scene = experiments.Scene()
    numbers = (
        json.loads(ALIGNMENT_NUMBERS_JSON.read_text()) if ALIGNMENT_NUMBERS_JSON.exists() else {}
    )
    for name in only or SECTIONS:
        logger.info(f"section {name}")
        numbers[name] = getattr(experiments, name)(scene)
    numbers["environment"] = {
        "python": platform.python_version(),
        "platform": f"{platform.system()} {platform.release()}",
        "numpy": np.__version__,
        "opencv": cv2.__version__,
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ALIGNMENT_NUMBERS_JSON.write_text(json.dumps(numbers, indent=2) + "\n")
    logger.info(f"wrote {ALIGNMENT_NUMBERS_JSON}")


if __name__ == "__main__":
    align_experiments()
