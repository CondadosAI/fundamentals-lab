"""Fetch unit 3.2's frames and its three "in the wild" photographs.

The board: the 1,004 normal pcb1 frames of VisA. `--from-dir` copies them from an
extracted VisA tree (the folder holding `pcb1/`). Without it the VisA tar (1.93 GB) is
streamed from Amazon's bucket and only `pcb1/Data/Images/Normal/*` is kept.

The photographs: three CC0 files from Wikimedia Commons, registered in
CondadosAI/cv-assets, fetched and never redistributed.
"""

import shutil
import tarfile
import urllib.request
from pathlib import Path

import click
from loguru import logger

from fundamentals_lab.config import BOUNDARY_DIR, DATA_DIR, VISA_TAR_URL

PREFIX = "pcb1/Data/Images/Normal/"
UA = {"User-Agent": "fundamentals-lab/0.1 (+https://github.com/CondadosAI/fundamentals-lab)"}
#: Commons file name -> where boundaries/wild.py reads it. All CC0.
WILD = {
    "The_Cloister_Mandapam,_in_One_point_perspective.jpg": DATA_DIR / "wild" / "cloister.jpg",
    "Stationery_on_a_bench_(Unsplash).jpg": DATA_DIR
    / "wild-document"
    / "stationery_bench_unsplash.jpg",
    "Billiards_table_1.JPG": DATA_DIR / "pool" / "Billiards_table_1.JPG",
}


def fetch_wild() -> None:
    import json
    import urllib.parse

    for name, dst in WILD.items():
        if dst.exists():
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        api = (
            "https://commons.wikimedia.org/w/api.php?action=query&prop=imageinfo&iiprop=url"
            f"&format=json&titles=File:{urllib.parse.quote(name)}"
        )
        pages = json.load(urllib.request.urlopen(urllib.request.Request(api, headers=UA)))["query"][
            "pages"
        ]
        url = next(iter(pages.values()))["imageinfo"][0]["url"]
        logger.info(f"fetching {url}")
        dst.write_bytes(urllib.request.urlopen(urllib.request.Request(url, headers=UA)).read())


@click.command()
@click.option(
    "--from-dir",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    help="An extracted VisA folder containing pcb1/.",
)
def boundary_download(from_dir: Path | None) -> None:
    """Get the VisA pcb1 normal frames and the three in-the-wild photographs."""
    fetch_wild()
    dst = BOUNDARY_DIR / "Normal"
    dst.mkdir(parents=True, exist_ok=True)
    if from_dir:
        src = from_dir / PREFIX
        for p in sorted(src.glob("*.JPG")):
            if not (dst / p.name).exists():
                shutil.copy2(p, dst / p.name)
        logger.info(f"{len(list(dst.glob('*.JPG')))} frames in {dst}")
        return
    req = urllib.request.Request(VISA_TAR_URL, headers=UA)
    n = 0
    with urllib.request.urlopen(req) as resp, tarfile.open(fileobj=resp, mode="r|") as tar:
        for m in tar:
            if m.isfile() and PREFIX in m.name and m.name.endswith(".JPG"):
                out = dst / Path(m.name).name
                if not out.exists():
                    out.write_bytes(tar.extractfile(m).read())
                n += 1
            elif n and PREFIX not in m.name and "pcb1/" not in m.name:
                break  # the pcb1 normals are contiguous in the tar; stop once past them
    logger.info(f"{n} frames in {dst}")
