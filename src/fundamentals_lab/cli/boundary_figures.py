"""Render unit 3.2's figures, the lossless working frame and the cover backgrounds."""

import click

from fundamentals_lab.boundaries import figures


@click.command()
def boundary_figures() -> None:
    """Figures to output/figures/boundaries/, covers to output/covers/."""
    figures.render_all()
