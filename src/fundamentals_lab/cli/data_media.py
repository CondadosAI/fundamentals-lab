"""Animated figures for unit 1.3 (needs ffmpeg with libwebp and the HDRPS raws)."""

import click

from fundamentals_lab.imagedata import media


@click.command()
@click.option("--only", multiple=True, help="Render these animations only (by name).")
def data_media(only: tuple[str, ...]) -> None:
    """Render the real-application animation each image-as-data post opens with."""
    media.render_all(only)


if __name__ == "__main__":
    data_media()
