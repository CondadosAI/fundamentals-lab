"""Animated figures for unit 3.1 (needs ffmpeg with libwebp)."""

import click

from fundamentals_lab.core import media


@click.command()
@click.option("--only", multiple=True, help="Render these animations only (by name).")
def edge_media(only: tuple[str, ...]) -> None:
    """Render the real-application animation each edge-detection post opens with."""
    media.render_all(only)


if __name__ == "__main__":
    edge_media()
