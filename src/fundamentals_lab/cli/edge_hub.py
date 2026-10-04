"""The Edge Detection hub's pipeline, run on one photograph the site may serve.

`uv run edge-hub`. The unit's lessons measure on Middlebury templeRing, whose licence is
not stated, so the hub's per-step panels and the numbers its representation section
prints run on the CC0 cloister photograph instead (fetched by `boundary-download`): the
middle 960x720 of the photo at 960 px wide. The numbers go into `output/edge_numbers.json`
under `hub_cloister`; the figure into `output/figures/edges/hub-pipeline.png`.
"""

import json

import click
import cv2
import matplotlib
import numpy as np
from loguru import logger

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from fundamentals_lab.cli.experiments import (  # noqa: E402
    hysteresis_vs_single_threshold,
    threshold_dilemma,
)
from fundamentals_lab.config import DATA_DIR, OUTPUT_DIR  # noqa: E402
from fundamentals_lab.core import canny as canny_mod  # noqa: E402

CLOISTER = DATA_DIR / "wild" / "cloister.jpg"
WIDTH, CROP_TOP, CROP_H = 960, 260, 720
SIGMA, LOW, HIGH = 1.4, 50, 150
BG_HEX = "#0e1218"


def cloister_crop() -> np.ndarray:
    img = cv2.imread(str(CLOISTER))
    if img is None:
        raise FileNotFoundError(f"{CLOISTER}: run boundary-download first")
    img = cv2.resize(
        img, (WIDTH, round(img.shape[0] * WIDTH / img.shape[1])), interpolation=cv2.INTER_AREA
    )
    return img[CROP_TOP : CROP_TOP + CROP_H]


def hub_cloister() -> dict:
    """The threshold sweep and the hysteresis comparison of the lessons, on the cloister."""
    bgr = cloister_crop()
    grey = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    stages = canny_mod.canny_from_scratch(grey, sigma=SIGMA, low=LOW, high=HIGH)
    return {
        "asset": "cloister.jpg (Wikimedia Commons, CC0)",
        "crop": {"width": WIDTH, "top": CROP_TOP, "height": CROP_H},
        "edge_pixels": int(np.count_nonzero(stages["edges"])),
        "threshold_dilemma": threshold_dilemma(grey),
        "hysteresis_vs_single_threshold": hysteresis_vs_single_threshold(grey)["operating_points"][
            0
        ],
    }


def _direction_image(mag: np.ndarray, ori: np.ndarray) -> np.ndarray:
    """Hue = edge-normal direction (0..180 degrees), brightness = strength."""
    v = np.clip(mag / np.percentile(mag, 99.5), 0, 1)
    # float HSV hue runs 0..360; doubling the folded angle makes 0 and 180 the same colour
    hsv = np.dstack([2 * ori.astype(np.float32), np.ones_like(v), v]).astype(np.float32)
    return (cv2.cvtColor(hsv * np.array([1, 1, 255], np.float32), cv2.COLOR_HSV2BGR)).astype(
        np.uint8
    )


def hub_pipeline() -> None:
    bgr = cloister_crop()
    grey = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    blurred = cv2.GaussianBlur(grey, (0, 0), sigmaX=SIGMA, sigmaY=SIGMA)
    stages = canny_mod.canny_from_scratch(grey, sigma=SIGMA, low=LOW, high=HIGH)
    mag, ori, thin, edges = (stages[k] for k in ("magnitude", "orientation", "thin", "edges"))

    def u8(a, top=None):
        top = np.percentile(a, 99.5) if top is None else top
        return (np.clip(a / top, 0, 1) * 255).astype(np.uint8)

    k = np.ones((2, 2), np.uint8)  # one-pixel lines survive a third-of-the-page width
    panels = [
        (bgr, "1  Image"),
        ((blurred * 255).astype(np.uint8), "2–3  Grey, smoothed"),
        (u8(mag), "4  Measure: strength"),
        (_direction_image(mag, ori), "4  Measure: direction"),
        (cv2.dilate(u8(thin), k), "5  Thin to the peak"),
        (cv2.dilate(edges.astype(np.uint8) * 255, k), "6  Link: 0 or 255"),
    ]
    fig, axes = plt.subplots(3, 2, figsize=(8, 10.3), dpi=110, facecolor=BG_HEX)
    for ax, (img, title) in zip(axes.ravel(), panels, strict=True):
        ax.set_facecolor(BG_HEX)
        if img.ndim == 2:
            ax.imshow(img, cmap="gray", vmin=0, vmax=255, aspect="auto")
        else:
            ax.imshow(img[..., ::-1], aspect="auto")
        ax.set_title(title, color="#e2e8f0", fontsize=19, loc="left", pad=6)
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_color("#334155")
    fig.tight_layout(pad=0.6, h_pad=1.2, w_pad=0.8)
    out = OUTPUT_DIR / "figures" / "edges"
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / "hub-pipeline.png", facecolor=BG_HEX)
    plt.close(fig)
    logger.info(f"wrote {out / 'hub-pipeline.png'}")


@click.command()
def edge_hub() -> None:
    """Write the hub's cloister numbers into edge_numbers.json and draw its pipeline figure."""
    path = OUTPUT_DIR / "edge_numbers.json"
    numbers = json.loads(path.read_text())
    numbers["hub_cloister"] = hub_cloister()
    path.write_text(json.dumps(numbers, indent=2))
    logger.info(f"wrote hub_cloister into {path}")
    hub_pipeline()


if __name__ == "__main__":
    edge_hub()
