"""Cover backgrounds for the unit's six posts, rendered from real output.

The site's source policy is real project output whenever the article produced
some, and every lesson here did. These are backgrounds only: the title card
composites its own text over the left two-thirds and crops to the right, so
nothing is labelled and the content is weighted right.
"""

import click
import matplotlib
import numpy as np
from loguru import logger

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from fundamentals_lab.config import OUTPUT_DIR  # noqa: E402
from fundamentals_lab.core.canny import canny_from_scratch  # noqa: E402
from fundamentals_lab.core.corners import harris_response  # noqa: E402
from fundamentals_lab.core.dataset import canonical_gray  # noqa: E402
from fundamentals_lab.core.gradients import (  # noqa: E402
    OPERATORS,
    gradients,
    magnitude,
    top_fraction_mask,
)
from fundamentals_lab.core.laplacian import (  # noqa: E402
    crossing_floor,
    log_response,
    zero_crossings,
)

BG = "#0b0e18"
COVER_DIR = OUTPUT_DIR / "covers"
# 16:9 at a size the 1200x675 title card can crop into without upscaling.
W, H, DPI = 1600, 900, 100


def _figure():
    fig = plt.figure(figsize=(W / DPI, H / DPI), dpi=DPI, facecolor=BG)
    return fig


def _save(fig, name: str) -> None:
    COVER_DIR.mkdir(parents=True, exist_ok=True)
    path = COVER_DIR / f"{name}.png"
    fig.savefig(path, dpi=DPI, facecolor=BG, pad_inches=0)
    plt.close(fig)
    logger.info(f"wrote {path}")


def _tile(fig, panels: list, right_fraction: float = 0.62):
    """Lay panels across the right `right_fraction` of the canvas, flush."""
    n = len(panels)
    left = 1.0 - right_fraction
    for i, img in enumerate(panels):
        ax = fig.add_axes([left + i * right_fraction / n, 0.0, right_fraction / n, 1.0])
        ax.imshow(img, cmap="gray", aspect="auto")
        ax.set_axis_off()


@click.command()
def covers() -> None:
    """Render the six cover backgrounds into output/covers/."""
    gray = canonical_gray()
    stages = canny_from_scratch(gray, sigma=1.4, low=50, high=150)

    # Hub: the photograph with its edges burned in, which is what the unit is for.
    fig = _figure()
    ax = fig.add_axes([0.38, 0.0, 0.62, 1.0])
    rgb = np.repeat((gray * 0.55)[:, :, None], 3, axis=2)
    rgb[stages["edges"]] = [0.39, 0.40, 0.95]
    ax.imshow(rgb, aspect="auto")
    ax.set_axis_off()
    _save(fig, "edge-detection")

    # L1: one row of pixels and its two derivatives, as curves alone.
    fig = _figure()
    row = gray.shape[0] // 2
    profile = gray[row]
    first = np.gradient(profile)
    second = np.gradient(first)
    for i, (data, colour) in enumerate(
        [(profile, "#60a5fa"), (first, "#f59e0b"), (second, "#a78bfa")]
    ):
        ax = fig.add_axes([0.40, 0.70 - i * 0.30, 0.58, 0.24])
        ax.plot(data, lw=3, color=colour)
        ax.set_facecolor(BG)
        ax.set_axis_off()
    _save(fig, "what-is-an-edge")

    # L2: the same pixels through the three operators, at a matched budget.
    fig = _figure()
    panels = []
    for op in OPERATORS:
        gx, gy = gradients(gray, op)
        panels.append(top_fraction_mask(magnitude(gx, gy), 0.05))
    _tile(fig, panels)
    _save(fig, "image-gradients-sobel-prewitt")

    # L3: zero-crossings tightening as sigma rises.
    fig = _figure()
    _tile(
        fig,
        [
            zero_crossings(log_response(gray, s), crossing_floor(log_response(gray, s)))
            for s in (1.0, 2.0, 3.0)
        ],
    )
    _save(fig, "laplacian-log-zero-crossings")

    # L4: the pipeline, one panel per stage.
    fig = _figure()
    _tile(fig, [gray, stages["magnitude"], stages["thin"], stages["edges"]], right_fraction=0.68)
    _save(fig, "canny-edge-detector")

    # L5: the photograph with its strongest corners marked.
    fig = _figure()
    ax = fig.add_axes([0.38, 0.0, 0.62, 1.0])
    ax.imshow(gray * 0.5, cmap="gray", aspect="auto")
    r = harris_response(gray)
    ys, xs = np.nonzero(r > np.quantile(r, 0.9993))
    ax.scatter(xs, ys, s=210, facecolors="none", edgecolors="#34d399", linewidths=2.4)
    ax.set_xlim(0, gray.shape[1])
    ax.set_ylim(gray.shape[0], 0)
    ax.set_axis_off()
    _save(fig, "corner-detection-harris-shi-tomasi")


if __name__ == "__main__":
    covers()
