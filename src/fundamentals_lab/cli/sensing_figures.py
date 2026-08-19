"""The photographs unit 1.2 publishes, rendered from the raw exposures.

Charts and curves in the posts are hand-authored SVG. What has to come from here is
anything made of pixels: the scene itself, the bracket that covers it, where the
patches are sampled, and what noise looks like when you subtract two frames.

Every image derives from Mark Fairchild's HDR Photographic Survey and carries its
attribution in the post that shows it.
"""

import click
import cv2
import numpy as np
import rawpy
from loguru import logger

from fundamentals_lab.config import (
    CHART_REFERENCE_FRAME,
    FIGURE_DIR,
    SCENE_SLUG,
)
from fundamentals_lab.sensing import charts, hdr, hdrps, noise

WEBP = [cv2.IMWRITE_WEBP_QUALITY, 82]


def _save(image: np.ndarray, name: str) -> None:
    path = FIGURE_DIR / name
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), image, WEBP)
    logger.info(f"wrote {path} ({path.stat().st_size / 1024:.0f} KB)")


def _tone_map(radiance: np.ndarray, strength: float = 20_000.0, gamma: float = 1.2) -> np.ndarray:
    """A plain log tone map, so a 17-stop scene fits in 8 bits.

    `strength` is how hard the log lifts the shadows. It is a choice, not a
    measurement: at 200,000 the shadowed chart is as bright as the lit one and the
    picture goes flat, at 2,000 the shadowed chart disappears again. Lesson 4 shows
    all three side by side for that reason.
    """
    reference = np.percentile(radiance[radiance > 0], 99.9)
    scaled = np.clip(radiance / reference, 0.0, None)
    display = np.log1p(strength * scaled) / np.log1p(strength)
    return np.clip(display ** (1.0 / gamma) * 255.0, 0, 255).astype(np.uint8)


@click.command()
@click.option("--slug", default=SCENE_SLUG)
@click.option("--width", default=1600, help="Long side of the published figures, in pixels.")
def sensing_figures(slug: str, width: int) -> None:
    """Render unit 1.2's figures into output/figures/."""
    paths = sorted((hdrps.scene_dir(slug) / "raw").glob("*.NEF"))
    if not paths:
        raise click.ClickException("run `uv run sensing-download` first")
    frames = sorted(
        (hdrps.load_raw(path) for path in paths), key=lambda frame: frame.shutter or 0.0
    )

    def resize(image: np.ndarray) -> np.ndarray:
        scale = width / image.shape[1]
        return cv2.resize(image, (width, int(image.shape[0] * scale)), interpolation=cv2.INTER_AREA)

    # 1. The scene: our own merge of all eighteen exposures, tone-mapped.
    exposures = [(frame.counts, frame.shutter) for frame in frames]
    channels = {}
    for name in ("R", "G1", "B"):
        ceiling = noise.clip_level(frames[-1].counts, name)
        channels[name] = hdr.merge(exposures, name, ceiling).plane
    with rawpy.imread(str(frames[0].path)) as raw:
        balance = [float(v) for v in raw.camera_whitebalance[:3]]
    stacked = np.dstack(
        [channels["B"] * balance[2], channels["G1"] * balance[1], channels["R"] * balance[0]]
    )
    _save(resize(_tone_map(stacked)), "scene-merged.webp")

    # 1b. The same radiance map, tone-mapped three ways, because the choice is visible.
    panels = []
    for strength, gamma in ((2_000.0, 1.0), (20_000.0, 1.2), (200_000.0, 1.4)):
        panel = _tone_map(stacked, strength, gamma)
        panel = cv2.resize(panel, (width // 3, int(panel.shape[0] * (width // 3) / panel.shape[1])))
        cv2.rectangle(panel, (0, panel.shape[0] - 30), (190, panel.shape[0]), (0, 0, 0), -1)
        cv2.putText(
            panel, f"log lift {strength:.0f}".replace("000", "k").replace("k0", "k"),
            (8, panel.shape[0] - 9), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1,
        )
        panels.append(panel)
    _save(np.hstack(panels), "tonemaps.webp")

    # 2. The bracket: six of the eighteen frames, developed identically.
    picks = [0, 3, 6, 9, 12, 17]
    tiles = []
    for index in picks:
        with rawpy.imread(str(frames[index].path)) as raw:
            rgb = raw.postprocess(use_camera_wb=True, output_bps=8, no_auto_bright=True)
        tile = cv2.resize(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), (width // 3, width // 3 * 2 // 3))
        label = f"{frames[index].shutter:g} s"
        cv2.rectangle(tile, (0, tile.shape[0] - 34), (150, tile.shape[0]), (0, 0, 0), -1)
        cv2.putText(
            tile, label, (10, tile.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2
        )
        tiles.append(tile)
    strip = np.vstack([np.hstack(tiles[:3]), np.hstack(tiles[3:])])
    _save(strip, "bracket.webp")

    # 3. Where the patches are sampled, drawn on the frame that shows each chart best.
    for chart, index in CHART_REFERENCE_FRAME.items():
        with rawpy.imread(str(frames[index].path)) as raw:
            rgb = raw.postprocess(use_camera_wb=True, output_bps=8, no_auto_bright=False)
        canvas = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        for patch in charts.patches(chart):
            x0, y0, x1, y1 = patch.box
            cv2.rectangle(canvas, (x0, y0), (x1, y1), (0, 255, 255), 3)
            cv2.putText(
                canvas, str(patch.number), (x0, y0 - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2,
            )
        boxes = np.array([patch.box for patch in charts.patches(chart)])
        pad = 90
        x0, y0 = max(int(boxes[:, 0].min()) - pad, 0), max(int(boxes[:, 1].min()) - pad, 0)
        x1, y1 = int(boxes[:, 2].max()) + pad, int(boxes[:, 3].max()) + pad
        _save(resize(canvas[y0:y1, x0:x1]), f"patches-{chart}.webp")

    # 4. What noise looks like: one 30 s frame, and the difference of two of them.
    repeats = [frame for frame in frames if frame.shutter == frames[-1].shutter]
    plane_a = noise.channel(repeats[0].counts, "G1").astype(np.float32)
    plane_b = noise.channel(repeats[1].counts, "G1").astype(np.float32)
    dim_boxes = np.array([patch.box for patch in charts.patches("dim")])
    x0, y0 = int(dim_boxes[:, 0].min()) // 2, int(dim_boxes[:, 1].min()) // 2
    x1, y1 = int(dim_boxes[:, 2].max()) // 2, int(dim_boxes[:, 3].max()) // 2
    crop_a = plane_a[y0:y1, x0:x1]
    difference = (plane_a - plane_b)[y0:y1, x0:x1]
    left = np.clip(crop_a / crop_a.max() * 255, 0, 255).astype(np.uint8)
    right = np.clip(difference * 12 + 128, 0, 255).astype(np.uint8)
    pair = np.hstack(
        [cv2.cvtColor(left, cv2.COLOR_GRAY2BGR), cv2.cvtColor(right, cv2.COLOR_GRAY2BGR)]
    )
    _save(resize(pair), "noise-difference.webp")

    logger.info("figures done")


if __name__ == "__main__":
    sensing_figures()
