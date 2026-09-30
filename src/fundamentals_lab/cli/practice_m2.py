"""OpenCV in Practice M2: build the module's images and compute its numbers."""

import json

import click

from fundamentals_lab.practice import m2


@click.command()
@click.option("--assets", is_flag=True, help="Resize the three originals in data/pool first.")
@click.option("--media", is_flag=True, help="Render the opening images and the hub animation.")
@click.option("--covers", is_flag=True, help="Render the title-card backgrounds (output/covers/).")
def practice_m2(assets: bool, media: bool, covers: bool) -> None:
    """Write output/practice/m2/*.webp (with --assets) and output/practice_m2_numbers.json."""
    if assets:
        m2.build_assets()
    print(json.dumps(m2.experiments(), indent=1))
    if media:
        m2.build_media()
    if covers:
        m2.build_covers()


if __name__ == "__main__":
    practice_m2()
