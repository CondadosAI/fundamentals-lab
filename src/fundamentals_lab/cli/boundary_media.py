"""Render unit 3.2's opening animation to output/figures/boundaries/media/."""

import click
from loguru import logger

from fundamentals_lab.boundaries import media


@click.command()
def boundary_media() -> None:
    """Lane segments on a minute of freeway (comma2k19), frame by frame."""
    for name, path in media.render_all().items():
        logger.info(f"{name}: {path}")
