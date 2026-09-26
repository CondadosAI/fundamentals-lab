"""Figures and cover backgrounds for units 3.5 and 4.1-L1."""

import click

from fundamentals_lab.alignment import figures


@click.command()
def align_figures() -> None:
    """Render every overlay, top view and cover from the fitted court."""
    figures.render_all()


if __name__ == "__main__":
    align_figures()
