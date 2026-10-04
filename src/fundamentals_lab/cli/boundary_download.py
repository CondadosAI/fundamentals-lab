"""Fetch everything unit 3.2 measures on. Nothing here is committed to the repository.

- comma10k (comma.ai, MIT): the frame the lessons measure on and every 33rd frame of the
  listing (300 in all), each with its hand-painted mask, plus the dataset's LICENSE.
- comma2k19 (comma.ai, MIT): one minute of freeway for the opening animation, read out of
  the 2.4 GB compression-challenge archive on Hugging Face by HTTP range, not downloaded whole.
- OpenCV in Practice M2's three pool photos and its hand labels, as condados.ai serves them.
- VisA `candle` (Amazon, CC BY 4.0): 100 normal frames, from an extracted VisA folder.
- Two CC0 photographs from Wikimedia Commons: a cloister and a sheet of paper on a bench.
"""

import json
import shutil
import urllib.parse
import urllib.request
from pathlib import Path

import click
from loguru import logger

from fundamentals_lab.config import (
    COMMA2K19_DIR,
    COMMA2K19_SEGMENT,
    COMMA2K19_ZIP,
    COMMA10K_DIR,
    COMMA10K_RAW,
    COMMA10K_SAMPLE_EVERY,
    DATA_DIR,
    HIGHWAY_FRAME,
    POOL_DIR,
    POOL_PHOTOS,
    POOL_SITE,
)

UA = {"User-Agent": "fundamentals-lab/0.1 (+https://github.com/CondadosAI/fundamentals-lab)"}
#: Commons file name -> where boundaries/wild.py reads it. Both CC0.
WILD = {
    "The_Cloister_Mandapam,_in_One_point_perspective.jpg": DATA_DIR / "wild" / "cloister.jpg",
    "Stationery_on_a_bench_(Unsplash).jpg": DATA_DIR
    / "wild-document"
    / "stationery_bench_unsplash.jpg",
}
CANDLE_DIR = DATA_DIR / "visa-candle"


def _get(url: str) -> bytes:
    return urllib.request.urlopen(urllib.request.Request(url, headers=UA)).read()


def fetch_comma10k() -> None:
    for sub in ("imgs", "masks"):
        (COMMA10K_DIR / sub).mkdir(parents=True, exist_ok=True)
    listing = COMMA10K_DIR / "listing.txt"
    if not listing.exists():
        tree = json.loads(_get("https://api.github.com/repos/commaai/comma10k/git/trees/master"))
        sha = next(t["sha"] for t in tree["tree"] if t["path"] == "imgs")
        names = [
            t["path"]
            for t in json.loads(
                _get(f"https://api.github.com/repos/commaai/comma10k/git/trees/{sha}")
            )["tree"]
        ]
        listing.write_text("\n".join(names) + "\n")
    names = listing.read_text().split()
    pick = names[::COMMA10K_SAMPLE_EVERY]
    if HIGHWAY_FRAME not in pick:
        pick.append(HIGHWAY_FRAME)
    for n in pick:
        for sub in ("imgs", "masks"):
            dst = COMMA10K_DIR / sub / n
            if not dst.exists():
                dst.write_bytes(_get(f"{COMMA10K_RAW}/{sub}/{n}"))
    lic = COMMA10K_DIR / "LICENSE"
    if not lic.exists():
        lic.write_bytes(_get(f"{COMMA10K_RAW}/LICENSE"))
    logger.info(f"comma10k: {len(pick)} frames with masks in {COMMA10K_DIR}")


def fetch_comma2k19() -> None:
    from remotezip import RemoteZip

    dst = COMMA2K19_DIR / COMMA2K19_SEGMENT.replace("|", "_").replace("/", "_")
    if dst.exists():
        return
    COMMA2K19_DIR.mkdir(parents=True, exist_ok=True)
    with RemoteZip(COMMA2K19_ZIP, headers=UA) as z:
        name = next(i.filename for i in z.infolist() if i.filename.endswith(COMMA2K19_SEGMENT))
        dst.write_bytes(z.read(name))
    logger.info(f"comma2k19: {dst.name}")


def fetch_pool() -> None:
    POOL_DIR.mkdir(parents=True, exist_ok=True)
    for name in [f"{p}.webp" for p in POOL_PHOTOS] + ["labels.json"]:
        dst = POOL_DIR / name
        if not dst.exists():
            dst.write_bytes(_get(f"{POOL_SITE}/{name}"))
    logger.info(f"pool: {len(POOL_PHOTOS)} photos and labels in {POOL_DIR}")


def fetch_wild() -> None:
    for name, dst in WILD.items():
        if dst.exists():
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        api = (
            "https://commons.wikimedia.org/w/api.php?action=query&prop=imageinfo&iiprop=url"
            f"&format=json&titles=File:{urllib.parse.quote(name)}"
        )
        pages = json.loads(_get(api))["query"]["pages"]
        url = next(iter(pages.values()))["imageinfo"][0]["url"]
        logger.info(f"fetching {url}")
        dst.write_bytes(_get(url))


def copy_candles(visa_dir: Path) -> None:
    CANDLE_DIR.mkdir(parents=True, exist_ok=True)
    files = sorted((visa_dir / "candle" / "Data" / "Images" / "Normal").glob("*.JPG"))[::10][:100]
    for f in files:
        if not (CANDLE_DIR / f.name).exists():
            shutil.copy2(f, CANDLE_DIR / f.name)
    logger.info(f"candles: {len(list(CANDLE_DIR.glob('*.JPG')))} frames in {CANDLE_DIR}")


@click.command()
@click.option(
    "--visa-dir",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    help="An extracted VisA folder (holding candle/), for lesson 4's in-the-wild check.",
)
def boundary_download(visa_dir: Path | None) -> None:
    """Get the highway frames and masks, the freeway minute, the pool photos and the photographs."""
    fetch_comma10k()
    fetch_comma2k19()
    fetch_pool()
    fetch_wild()
    if visa_dir:
        copy_candles(visa_dir)
    elif not any(CANDLE_DIR.glob("*.JPG")):
        logger.warning("no VisA candle frames: pass --visa-dir (VisA_20220922.tar, extracted)")
