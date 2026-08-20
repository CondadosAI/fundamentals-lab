"""The one frame this unit works on, in the representations it compares.

The survey's own rendered JPEG is 600x337 and its EXR is a merge on a different grid,
so neither can be compared pixel-for-pixel with the RAW. Every representation below is
therefore developed here from a single NEF, which puts the 12-bit mosaic, the 16-bit
development and the 8-bit development on one 2868x4312 grid.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from fundamentals_lab.config import (
    DEVELOP_KWARGS,
    IMAGEDATA_FRAME,
    SCENE_SLUG,
    SENSING_DIR,
)
from fundamentals_lab.sensing.hdrps import _sensing_import


def frame_path(name: str = IMAGEDATA_FRAME) -> Path:
    path = SENSING_DIR / SCENE_SLUG / "raw" / name
    if not path.exists():
        raise FileNotFoundError(
            f"{path} missing — run `uv run sensing-download` (unit 1.2's fetcher)"
        )
    return path


@dataclass(frozen=True)
class Frame:
    """One exposure, in the three forms this unit compares.

    `mosaic` is what the sensor wrote: one 12-bit number per photosite, still in the
    Bayer pattern. `rgb16` and `rgb8` are the same frame developed, so they carry three
    interpolated numbers per pixel. The mosaic is the only one of the three that is a
    measurement rather than a reconstruction, which is the sentence L1 is built on.
    """

    name: str
    mosaic: np.ndarray  # (H, W) uint16, 12-bit values
    rgb16: np.ndarray  # (H, W, 3) uint16, RGB order
    rgb8: np.ndarray  # (H, W, 3) uint8, RGB order
    white_level: int
    black_level: tuple[int, ...]
    cfa_pattern: tuple[tuple[int, ...], ...]
    colour_desc: str
    file_bytes: int

    @property
    def bgr8(self) -> np.ndarray:
        """The 8-bit development in OpenCV's channel order.

        rawpy hands back RGB and OpenCV expects BGR. Converting once, here, is why no
        lesson has to apologise for a red-and-blue-swapped figure.
        """
        return self.rgb8[:, :, ::-1].copy()


def load(name: str = IMAGEDATA_FRAME) -> Frame:
    """One exposure, read once and developed twice.

    Metadata is read *before* `postprocess` runs, and that ordering is load-bearing:
    postprocessing mutates LibRaw's internal state, and `raw_pattern` afterwards
    reports the processed layout rather than the one in the file. Reading it late
    turned RGGB's `[[0, 1], [3, 2]]` into `[[0, 1], [1, 2]]` -- which happens to
    resolve to the same four offsets, so nothing downstream broke and the only
    casualty would have been a wrong number printed in a post.
    """
    rawpy = _sensing_import("rawpy")
    path = frame_path(name)
    with rawpy.imread(str(path)) as raw:
        mosaic = raw.raw_image_visible.copy()
        white_level = int(raw.white_level)
        black_level = tuple(int(v) for v in raw.black_level_per_channel)
        cfa_pattern = tuple(tuple(int(v) for v in row) for row in raw.raw_pattern)
        colour_desc = raw.color_desc.decode()

        rgb16 = raw.postprocess(output_bps=16, **DEVELOP_KWARGS)
        rgb8 = raw.postprocess(output_bps=8, **DEVELOP_KWARGS)

    return Frame(
        name=name,
        mosaic=mosaic,
        rgb16=rgb16,
        rgb8=rgb8,
        white_level=white_level,
        black_level=black_level,
        cfa_pattern=cfa_pattern,
        colour_desc=colour_desc,
        file_bytes=path.stat().st_size,
    )


def exr_facts() -> dict:
    """What the survey's published EXR actually is.

    L1 prints this to show a linear float file beside the integer ones. It is not a
    reference for anything: the grid does not match the RAW, and the file is a
    Photoshop merge whose scale was fitted on this scene's own chart.
    """
    OpenEXR = _sensing_import("OpenEXR")
    path = SENSING_DIR / SCENE_SLUG / f"{SCENE_SLUG}.exr"
    if not path.exists():
        raise FileNotFoundError(f"{path} missing — run `uv run sensing-download` first")
    with OpenEXR.File(str(path)) as handle:
        header = handle.header()
        (name, channel), *_ = handle.channels().items()
        pixels = channel.pixels
        return {
            "file": path.name,
            "channels": name,
            "shape": list(pixels.shape),
            "dtype": str(pixels.dtype),
            "compression": str(header.get("compression")),
            "min": float(pixels.min()),
            "max": float(pixels.max()),
            "negative_pixels": int((pixels < 0).sum()),
            "file_bytes": path.stat().st_size,
            "note": (
                "a merge on its own grid, not a rendering of this exposure; "
                "reference for nothing in this unit"
            ),
        }


def cfa_offsets(frame: Frame) -> dict[str, tuple[int, int]]:
    """Where each colour sits in the 2x2 mosaic cell, as (dy, dx).

    LibRaw reports the pattern as indices into `colour_desc` -- for this camera
    "RGBG", so index 3 is the second green. Reading it from the file rather than
    assuming RGGB is what keeps this code honest on a different camera.
    """
    pattern, desc = frame.cfa_pattern, frame.colour_desc
    seen: dict[str, tuple[int, int]] = {}
    for dy, row in enumerate(pattern):
        for dx, index in enumerate(row):
            letter = desc[index]
            label = letter if letter != "G" else ("G1" if "G1" not in seen else "G2")
            seen[label] = (dy, dx)
    return seen


def plane(frame: Frame, colour: str) -> np.ndarray:
    """One colour's photosites, as their own array.

    Every second row and column, so the returned plane is half-size in each axis --
    which is the fact L2 spends: red is not sampled on the grid the reader thinks
    they have.
    """
    dy, dx = cfa_offsets(frame)[colour]
    return frame.mosaic[dy::2, dx::2]
