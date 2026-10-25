"""Render unit 3.3's figures, the labs' data and the cover backgrounds."""

import click

from fundamentals_lab.features import figures


@click.command()
def sift_figures() -> None:
    """Figures and lab data to output/figures/sift/, covers to output/covers/."""
    figures.render_all()
