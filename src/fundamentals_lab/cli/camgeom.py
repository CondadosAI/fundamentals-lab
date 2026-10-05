"""Unit 4.1 lessons 2 to 4: `uv run camgeom-experiments`."""

import json

import click
from loguru import logger

from fundamentals_lab.config import OUTPUT_DIR


@click.command()
def camgeom_experiments() -> None:
    """Write output/camera_geometry_numbers.json and the corners the page serves."""
    from fundamentals_lab.camgeom.experiments import run_all

    numbers, corners = run_all()
    out = OUTPUT_DIR / "camera_geometry_numbers.json"
    out.write_text(json.dumps(numbers, indent=2))
    logger.info(f"wrote {out}")
    c = OUTPUT_DIR / "camgeom_corners.json"
    c.write_text(json.dumps(corners, separators=(",", ":")))
    logger.info(f"wrote {c} ({c.stat().st_size} bytes)")


@click.command()
def camgeom_figures() -> None:
    """Draw the hub's pipeline panels, the lesson figures and the cover backgrounds."""
    from fundamentals_lab.camgeom.figures import render_all

    render_all()
