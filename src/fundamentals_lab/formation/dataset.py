"""Fetch the photographs unit 1.1 measures on.

Neither source is redistributed. The OpenCV set could be -- it is Apache-2.0 --
but keeping both on the same footing means one rule to remember: this repository
holds code and numbers, and the images arrive at run time from where they live.
"""

import urllib.request

from loguru import logger

from fundamentals_lab.config import (
    FORMATION_CALIB_DIR,
    FORMATION_FISHEYE_DIR,
    FORMATION_FISHEYE_IMAGES,
    FORMATION_FISHEYE_RAW,
    FORMATION_LEFT_IMAGES,
    FORMATION_OPENCV_RAW,
)

# A raw.githubusercontent 404 arrives as a short HTML page, not an error, so the
# payload is checked rather than the status. Every real photograph here is >5 kB.
MIN_IMAGE_BYTES = 5_000


def _fetch(base_url: str, names: tuple, target, label: str) -> list:
    target.mkdir(parents=True, exist_ok=True)
    paths = []
    for name in names:
        dst = target / name
        if not dst.exists():
            url = f"{base_url}/{name}"
            logger.info(f"downloading {url}")
            urllib.request.urlretrieve(url, dst)  # noqa: S310 -- fixed https URL
        if dst.stat().st_size < MIN_IMAGE_BYTES:
            dst.unlink()
            raise RuntimeError(f"{name} came back too small to be a photograph")
        paths.append(dst)
    logger.info(f"{len(paths)} {label} images in {target}")
    return paths


def fetch_opencv_calibration() -> list:
    """OpenCV's own chessboard photographs (Apache-2.0) -- lessons 1 and 3."""
    return _fetch(FORMATION_OPENCV_RAW, FORMATION_LEFT_IMAGES, FORMATION_CALIB_DIR, "calibration")


def fetch_fisheye() -> list:
    """py-OCamCalib's fish_1 board views (GPL-2.0, never vendored) -- lesson 4."""
    paths = _fetch(
        FORMATION_FISHEYE_RAW, FORMATION_FISHEYE_IMAGES, FORMATION_FISHEYE_DIR, "fisheye"
    )
    return sorted(paths, key=lambda p: int(p.stem.split("_")[1]))
