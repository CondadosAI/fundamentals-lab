"""Fetch unit 1.1's source images."""

import click
from loguru import logger

from fundamentals_lab.formation.dataset import fetch_fisheye, fetch_opencv_calibration


@click.command()
@click.option("--skip-fisheye", is_flag=True, help="Only the Apache-2.0 calibration set.")
def formation_download(skip_fisheye: bool) -> None:
    """Download the chessboard photographs, and the fisheye board lesson 4 needs."""
    paths = fetch_opencv_calibration()
    if not skip_fisheye:
        paths += fetch_fisheye()
    total = sum(p.stat().st_size for p in paths)
    logger.info(f"{len(paths)} files, {total / 1e6:.2f} MB")


if __name__ == "__main__":
    formation_download()
