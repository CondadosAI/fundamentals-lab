"""The frames, and the empty court made from them.

The camera does not move and the players do, so the per-pixel median of frames
spread over a minute is the court with nobody on it. Lines are measured on that
plate; the reader is shown frame 45000.
"""

from __future__ import annotations

import subprocess
import urllib.request
from pathlib import Path

import cv2
import numpy as np
from loguru import logger

from fundamentals_lab.config import (
    ALIGN_DIR,
    ALIGN_FPS,
    ALIGN_FRAME,
    ALIGN_PLATE_FRAMES,
    ALIGN_SITE_BASE,
    ALIGN_VIDEO_ID,
)

FRAME_PNG = "frame_{:06d}.png"
PLATE_PNG = "plate_median31.png"


def frame_path(index: int) -> Path:
    return ALIGN_DIR / FRAME_PNG.format(index)


def plate_path() -> Path:
    return ALIGN_DIR / PLATE_PNG


def fetch_from_site() -> list[Path]:
    """Frame 45000 and the plate, lossless WebP from condados.ai, decoded to PNG."""
    ALIGN_DIR.mkdir(parents=True, exist_ok=True)
    out = []
    for name, dst in (
        (f"frame_{ALIGN_FRAME:06d}.webp", frame_path(ALIGN_FRAME)),
        ("plate_median31.webp", plate_path()),
    ):
        if dst.exists():
            out.append(dst)
            continue
        tmp = ALIGN_DIR / name
        logger.info(f"fetching {ALIGN_SITE_BASE}/{name}")
        urllib.request.urlretrieve(f"{ALIGN_SITE_BASE}/{name}", tmp)
        cv2.imwrite(str(dst), cv2.imread(str(tmp)))
        out.append(dst)
    return out


def fetch_from_youtube() -> list[Path]:
    """All 31 plate frames at full resolution, straight from the source video.

    Needs yt-dlp (a recent one: older releases get HTTP 403 on the media while the
    metadata still resolves) and ffmpeg. Downloads one short section, not the
    65-minute match.
    """
    ALIGN_DIR.mkdir(parents=True, exist_ok=True)
    t0 = ALIGN_PLATE_FRAMES[0] / ALIGN_FPS - 1.0
    t1 = ALIGN_PLATE_FRAMES[-1] / ALIGN_FPS + 1.0
    clip = ALIGN_DIR / f"{ALIGN_VIDEO_ID}_section.mp4"
    if not clip.exists():
        subprocess.run(
            [
                "yt-dlp",
                "-f",
                "137",
                "--download-sections",
                f"*{t0:.2f}-{t1:.2f}",
                "--force-keyframes-at-cuts",
                "-o",
                str(clip),
                f"https://www.youtube.com/watch?v={ALIGN_VIDEO_ID}",
            ],
            check=True,
        )
    return extract_frames(clip, offset_s=t0)


def extract_frames(video: Path, offset_s: float = 0.0) -> list[Path]:
    """Write the plate frames out of `video`, whose first frame is at `offset_s`."""
    cap = cv2.VideoCapture(str(video))
    out = []
    for index in ALIGN_PLATE_FRAMES:
        local = round((index / ALIGN_FPS - offset_s) * ALIGN_FPS)
        cap.set(cv2.CAP_PROP_POS_FRAMES, local)
        ok, frame = cap.read()
        if not ok:
            raise RuntimeError(f"could not read frame {index} from {video}")
        cv2.imwrite(str(frame_path(index)), frame)
        out.append(frame_path(index))
    return out


def load_frame(index: int = ALIGN_FRAME) -> np.ndarray:
    frame = cv2.imread(str(frame_path(index)))
    if frame is None:
        raise FileNotFoundError(f"{frame_path(index)} missing; run `align-download` first")
    return frame


def build_plate() -> np.ndarray:
    stack = np.stack([load_frame(i) for i in ALIGN_PLATE_FRAMES])
    plate = np.median(stack, axis=0).astype(np.uint8)
    cv2.imwrite(str(plate_path()), plate)
    return plate


def load_plate() -> np.ndarray:
    if not plate_path().exists():
        return build_plate()
    return cv2.imread(str(plate_path()))
