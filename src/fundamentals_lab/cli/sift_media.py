"""Unit 3.3's opening animations, to output/figures/sift/media/ (needs ffmpeg + libwebp)."""

import click
from loguru import logger

from fundamentals_lab.features import media


@click.command()
def sift_media() -> None:
    """The Barcelona matches, and the matches following a turning, shrinking copy."""
    for name, path in media.render_all().items():
        logger.info(f"{name}: {path}")
