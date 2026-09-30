"""Animated figures for unit 1.2 (needs ffmpeg with libwebp and the HDRPS raws)."""

import click

from fundamentals_lab.sensing import media


@click.command()
@click.option("--only", multiple=True, help="Render these animations only (by name).")
def sensing_media(only: tuple[str, ...]) -> None:
    """Render the real-application animation each image-sensing post opens with."""
    media.render_all(only)


if __name__ == "__main__":
    sensing_media()
