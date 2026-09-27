"""Fetch unit 3.5's frames and its "in the wild" photographs.

By default the pickleball frame and its median plate come from condados.ai (lossless
WebP, the same pixels the posts measure). `--from-youtube` fetches the 31 plate frames
at full resolution from the source video instead, and the plate is rebuilt from them.
"""

import json
import urllib.request

import click
from loguru import logger

from fundamentals_lab.alignment import plate
from fundamentals_lab.alignment.wild import OUTDOOR_DIR, PANO_DIR

# Wikimedia Commons, public domain ({{PD-self}}), by Pap3rinik: two source frames of
# the Hugin-stitched File:BarcelonaHarbour.jpg, taken from one viewpoint on Montjuïc.
PANORAMA = ("BarcelonaHarbour1.jpg", "BarcelonaHarbour2.jpg")
# "2024.08.30 MS4.0 Philip Wong vs Tristan Clark (Round Robin, match 6)" by
# pickleball4you, CC BY 3.0. A 70 s section; the camera is fixed from 380 to 635 s.
OUTDOOR = {"video_id": "K0qrASvix3Y", "start_s": 450.0, "end_s": 520.0, "step": 60}


def _commons_url(name: str) -> str:
    api = (
        "https://commons.wikimedia.org/w/api.php?action=query&prop=imageinfo&iiprop=url"
        f"&format=json&titles=File:{name}"
    )
    req = urllib.request.Request(api, headers={"User-Agent": "fundamentals-lab/0.1"})
    pages = json.load(urllib.request.urlopen(req))["query"]["pages"]
    return next(iter(pages.values()))["imageinfo"][0]["url"]


def fetch_panorama() -> None:
    PANO_DIR.mkdir(parents=True, exist_ok=True)
    for name in PANORAMA:
        dst = PANO_DIR / name
        if dst.exists():
            continue
        url = _commons_url(name)
        logger.info(f"fetching {url}")
        req = urllib.request.Request(url, headers={"User-Agent": "fundamentals-lab/0.1"})
        dst.write_bytes(urllib.request.urlopen(req).read())


def fetch_outdoor() -> None:
    import subprocess

    import cv2

    OUTDOOR_DIR.mkdir(parents=True, exist_ok=True)
    clip = OUTDOOR_DIR / f"{OUTDOOR['video_id']}_450-520.mp4"
    if not clip.exists():
        subprocess.run(
            [
                "yt-dlp",
                "-f",
                "137",
                "--download-sections",
                f"*{OUTDOOR['start_s']}-{OUTDOOR['end_s']}",
                "--force-keyframes-at-cuts",
                "-o",
                str(clip),
                f"https://www.youtube.com/watch?v={OUTDOOR['video_id']}",
            ],
            check=True,
        )
    cap = cv2.VideoCapture(str(clip))
    for i in range(0, 1801, OUTDOOR["step"]):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ok, frame = cap.read()
        if ok:
            cv2.imwrite(str(OUTDOOR_DIR / f"frame_{i:06d}.png"), frame)


@click.command()
@click.option(
    "--from-youtube", is_flag=True, help="Full-resolution plate frames from the source video."
)
@click.option("--skip-wild", is_flag=True, help="Only the main court frame and plate.")
def align_download(from_youtube: bool, skip_wild: bool) -> None:
    """Download the court frame, its median plate, and the in-the-wild photographs."""
    if from_youtube:
        plate.fetch_from_youtube()
        plate.build_plate()
    else:
        plate.fetch_from_site()
    if not skip_wild:
        fetch_panorama()
        fetch_outdoor()
    logger.info("done")


if __name__ == "__main__":
    align_download()
