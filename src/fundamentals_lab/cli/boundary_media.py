"""Render unit 3.2's opening animations to output/figures/boundaries/media/."""

import click
from loguru import logger

from fundamentals_lab.boundaries import media


@click.command()
def boundary_media() -> None:
    """The court lines found by the probabilistic Hough, frame by frame."""
    for name, path in media.render_all().items():
        logger.info(f"{name}: {path}")
