"""OpenCV in Practice M1: build the module's images and compute its numbers."""

import json

import click

from fundamentals_lab.practice import m1


@click.command()
@click.option(
    "--assets", is_flag=True, help="Develop the four frames first (needs --extra sensing)."
)
@click.option("--media", is_flag=True, help="Render the opening images and the hub animation.")
@click.option("--covers", is_flag=True, help="Render the title-card backgrounds (output/covers/).")
def practice_m1(assets: bool, media: bool, covers: bool) -> None:
    """Write output/practice/m1/*.webp (with --assets) and output/practice_m1_numbers.json."""
    if assets:
        m1.build_assets()
    print(json.dumps(m1.experiments(), indent=1))
    if media:
        m1.build_media()
    if covers:
        m1.build_covers()


if __name__ == "__main__":
    practice_m1()
