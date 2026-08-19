"""What the eighteen exposures declare about themselves, before any statistics.

This is lesson 1's table: the numbers a RAW file carries about the sensor that
made it — black level, saturation point, CFA layout — beside the exposure time
that separates one frame from the next. Nothing here is estimated.
"""

import json

import click
import numpy as np
from loguru import logger

from fundamentals_lab.config import OUTPUT_DIR, SCENE, SCENE_SLUG
from fundamentals_lab.sensing import hdrps


@click.command()
@click.option("--slug", default=SCENE_SLUG)
@click.option("--patch", default=64, help="Side of the centre block sampled per frame.")
def sensing_inspect(slug: str, patch: int) -> None:
    """Read every exposure of the scene and report what the files say."""
    paths = sorted((hdrps.scene_dir(slug) / "raw").glob("*.NEF"))
    if not paths:
        raise click.ClickException("no exposures found — run `uv run sensing-download` first")

    rows = []
    for path in paths:
        frame = hdrps.load_raw(path)
        counts = frame.counts
        half = patch // 2
        cy, cx = counts.shape[0] // 2, counts.shape[1] // 2
        block = counts[cy - half : cy + half, cx - half : cx + half].astype(np.float64)
        black = float(np.mean(frame.black_level))
        rows.append(
            {
                "file": path.name,
                "camera": frame.camera,
                "shutter_s": frame.shutter,
                "iso": frame.iso,
                "black_level": list(frame.black_level),
                "white_level": frame.white_level,
                "cfa": frame.cfa_pattern,
                "raw_shape": list(counts.shape),
                "centre_mean_dn": round(float(block.mean()), 2),
                "centre_std_dn": round(float(block.std(ddof=1)), 2),
                "centre_mean_above_black": round(float(block.mean() - black), 2),
                "saturated_fraction": round(
                    float((counts >= frame.white_level).mean()), 6
                ),
            }
        )
        logger.info(
            f"{path.name}: {frame.shutter}s ISO {frame.iso} "
            f"black {frame.black_level} white {frame.white_level} "
            f"centre {rows[-1]['centre_mean_dn']} ± {rows[-1]['centre_std_dn']} DN"
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    target = OUTPUT_DIR / "sensing_frames.json"
    target.write_text(
        json.dumps(
            {
                "scene": SCENE,
                "capture": hdrps.scene_metadata(slug),
                "patch_side_px": patch,
                "frames": rows,
            },
            indent=2,
        )
        + "\n"
    )
    logger.info(f"wrote {target}")


if __name__ == "__main__":
    sensing_inspect()
