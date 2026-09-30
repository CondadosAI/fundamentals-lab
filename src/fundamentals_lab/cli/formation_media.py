"""Animated figures for unit 1.1 (needs ffmpeg with libwebp)."""

import click

from fundamentals_lab.formation import media


@click.command()
@click.option("--only", multiple=True, help="Render these animations only (by name).")
def formation_media(only: tuple[str, ...]) -> None:
    """Render the real-application animation each image-formation post opens with."""
    media.render_all(only)


if __name__ == "__main__":
    formation_media()
