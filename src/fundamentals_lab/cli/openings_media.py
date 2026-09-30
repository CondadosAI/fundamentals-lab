"""Opening animations for units 2.1, 3.3 and 4.1 (needs ffmpeg with libwebp)."""

import click

from fundamentals_lab import openings


@click.command()
@click.option("--only", multiple=True, help="Render these animations only (by name).")
def openings_media(only: tuple[str, ...]) -> None:
    """Render the opening animations of the single-post units."""
    openings.render_all(only)


if __name__ == "__main__":
    openings_media()
