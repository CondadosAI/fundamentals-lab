"""One figure per lesson, all on the canonical frame."""

import click
import matplotlib
import numpy as np
from loguru import logger

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from edge_lab.config import FIGURE_DIR, LOG_SIGMAS, PATCH_PRESETS  # noqa: E402
from edge_lab.core.canny import canny_from_scratch  # noqa: E402
from edge_lab.core.dataset import canonical_gray  # noqa: E402
from edge_lab.core.gradients import (  # noqa: E402
    OPERATORS,
    gradients,
    magnitude,
    top_fraction_mask,
)
from edge_lab.core.laplacian import log_response, zero_crossings  # noqa: E402


def _save(fig, name: str) -> None:
    path = FIGURE_DIR / name
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info(f"wrote {path}")


@click.command()
def figures() -> None:
    """Render the unit's figures into output/figures/."""
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    gray = canonical_gray()

    # L1 — a scanline through an edge, with its two derivatives.
    row = gray.shape[0] // 2
    profile = gray[row]
    first = np.gradient(profile)
    second = np.gradient(first)
    fig, axes = plt.subplots(3, 1, figsize=(9, 6), sharex=True)
    for ax, (data, label) in zip(
        axes,
        [(profile, "intensity"), (first, "1st derivative"), (second, "2nd derivative")],
        strict=True,
    ):
        ax.plot(data, lw=1.2)
        ax.axhline(0, color="0.7", lw=0.8)
        ax.set_ylabel(label)
    axes[-1].set_xlabel(f"column (row {row} of the canonical frame)")
    _save(fig, "l1_scanline.png")

    # L2 — the three gradient operators at a matched edge-pixel budget.
    fig, axes = plt.subplots(1, len(OPERATORS) + 1, figsize=(4 * (len(OPERATORS) + 1), 5))
    axes[0].imshow(gray, cmap="gray")
    axes[0].set_title("canonical frame")
    for ax, op in zip(axes[1:], OPERATORS, strict=True):
        gx, gy = gradients(gray, op)
        ax.imshow(top_fraction_mask(magnitude(gx, gy), 0.05), cmap="gray")
        ax.set_title(f"{op} — top 5%")
    for ax in axes:
        ax.axis("off")
    _save(fig, "l2_operators.png")

    # L3 — LoG zero-crossings across sigma.
    fig, axes = plt.subplots(1, len(LOG_SIGMAS), figsize=(4 * len(LOG_SIGMAS), 5))
    for ax, sigma in zip(axes, LOG_SIGMAS, strict=True):
        ax.imshow(zero_crossings(log_response(gray, sigma)), cmap="gray")
        ax.set_title(f"LoG zero-crossings, σ = {sigma}")
        ax.axis("off")
    _save(fig, "l3_zero_crossings.png")

    # L4 — the Canny pipeline, stage by stage.
    stages = canny_from_scratch(gray, sigma=1.4, low=50, high=150)
    panels = [
        (gray, "input"),
        (stages["magnitude"], "gradient magnitude"),
        (stages["thin"], "after NMS"),
        (stages["edges"], "after hysteresis"),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(18, 5))
    for ax, (img, title) in zip(axes, panels, strict=True):
        ax.imshow(img, cmap="gray")
        ax.set_title(title)
        ax.axis("off")
    _save(fig, "l4_canny_stages.png")

    # L5 — where the three structure-tensor presets sit.
    fig, ax = plt.subplots(figsize=(5, 7))
    ax.imshow(gray, cmap="gray")
    for name, (x, y) in PATCH_PRESETS.items():
        ax.plot(x, y, "o", ms=9, mfc="none", mew=2)
        ax.annotate(name, (x, y), textcoords="offset points", xytext=(10, 6), fontsize=9)
    ax.axis("off")
    _save(fig, "l5_presets.png")


if __name__ == "__main__":
    figures()
