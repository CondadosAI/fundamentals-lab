"""Fetch the scene unit 1.2 measures: assets first, then the RAW exposures."""

import click
from loguru import logger

from fundamentals_lab.config import SCENE, SCENE_SLUG
from fundamentals_lab.sensing import hdrps


@click.command()
@click.option("--scene", default=SCENE, help="Folder name inside the HDRPS RAW archive.")
@click.option("--slug", default=SCENE_SLUG, help="Name the survey uses in its file paths.")
@click.option("--assets-only", is_flag=True, help="Skip the 220 MB of RAW exposures.")
def sensing_download(scene: str, slug: str, assets_only: bool) -> None:
    """Download one HDR Photographic Survey scene.

    Downloaded, never redistributed: the survey's terms are research use and
    non-commercial publication, with the source acknowledged.
    """
    assets = hdrps.download_assets(slug)
    logger.info(f"assets: {', '.join(sorted(p.name for p in assets.values()))}")
    if assets_only:
        return
    paths = hdrps.download_raws(scene, slug)
    logger.info(f"{len(paths)} exposures ready for {scene}")


if __name__ == "__main__":
    sensing_download()
