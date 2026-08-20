"""Cover backgrounds for unit 1.3, rendered from the unit's own measurements.

Backgrounds only: the title card composites its own text over the left two-thirds, so
nothing here is labelled and every image is weighted to the right.
"""

import json

import click
import matplotlib
import numpy as np
from loguru import logger

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from fundamentals_lab.config import IMAGEDATA_NUMBERS_JSON, OUTPUT_DIR  # noqa: E402

BG = "#0b0e18"
INK = "#e5e7eb"
ACCENT = "#6366f1"
WARM = "#f59e0b"
GOOD = "#34d399"
COVER_DIR = OUTPUT_DIR / "covers"
W, H, DPI = 1600, 900, 100


def _figure():
    return plt.figure(figsize=(W / DPI, H / DPI), dpi=DPI, facecolor=BG)


def _axes(fig, rect=(0.42, 0.14, 0.54, 0.74)):
    ax = fig.add_axes(rect, facecolor=BG)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color("#2c313f")
    ax.tick_params(colors="#6b7280", labelsize=9)
    ax.grid(color="#1b2030", linewidth=0.8)
    ax.set_axisbelow(True)
    return ax


def _save(fig, name: str) -> None:
    COVER_DIR.mkdir(parents=True, exist_ok=True)
    path = COVER_DIR / name
    fig.savefig(path, facecolor=BG, dpi=DPI)
    plt.close(fig)
    logger.info(f"wrote {path}")


@click.command()
def data_covers() -> None:
    """Render the covers for unit 1.3, plus unit 1.2's fifth lesson."""
    numbers = json.loads(IMAGEDATA_NUMBERS_JSON.read_text())

    # --- hub: the four losses, as a descending staircase of what survives ---------
    fig = _figure()
    ax = _axes(fig)
    stages = ["continuous", "sampled", "quantized", "3 numbers", "a file"]
    height = [1.0, 0.78, 0.6, 0.44, 0.24]
    ax.bar(stages, height, color=[ACCENT, ACCENT, ACCENT, WARM, WARM], width=0.62)
    ax.set_ylim(0, 1.1)
    ax.set_yticks([])
    ax.tick_params(axis="x", labelsize=11, colors="#9aa1ad")
    _save(fig, "the-image-as-data.png")

    # --- L1: the three representations, by size in memory ------------------------
    fig = _figure()
    ax = _axes(fig)
    rows = numbers["l1_array"]["representations"]["rows"]
    names = ["mosaic\n12-bit", "developed\n16-bit", "developed\n8-bit"]
    megabytes = [row["bytes_in_memory"] / 1e6 for row in rows]
    ax.barh(names, megabytes, color=[GOOD, ACCENT, ACCENT], height=0.55)
    ax.set_xlabel("MB in memory", color="#9aa1ad", fontsize=10)
    ax.tick_params(axis="y", labelsize=10, colors="#9aa1ad")
    _save(fig, "an-image-is-an-array.png")

    # --- L2: the ribbing, and the same ribbing sampled every 8th pixel ------------
    fig = _figure()
    ax = _axes(fig)
    period = numbers["l2_sampling"]["aliasing"]["base_period_px"]
    x = np.linspace(0, 96, 900)
    ax.plot(x, np.sin(2 * np.pi * x / period), color="#4b5563", linewidth=1.2)
    samples = np.arange(0, 97, 8)
    ax.plot(samples, np.sin(2 * np.pi * samples / period), color=WARM, linewidth=2.4,
            marker="o", markersize=5)
    ax.set_xlabel("pixels", color="#9aa1ad", fontsize=10)
    ax.set_yticks([])
    _save(fig, "sampling-the-pixel-grid.png")

    # --- L3: measured quantization error against the prediction -------------------
    fig = _figure()
    ax = _axes(fig)
    rows = numbers["l3_quantization"]["error_vs_prediction"]["rows"]
    bits = [row["bits"] for row in rows]
    ax.plot(bits, [row["predicted_rms_dn"] for row in rows], color=ACCENT, linewidth=2.2,
            marker="o", label="Δ/√12")
    ax.plot(bits, [row["measured_rms_chart_patches_dn"] for row in rows], color=WARM,
            linewidth=2.2, marker="s", linestyle="--", label="measured")
    ax.set_yscale("log")
    ax.invert_xaxis()
    ax.set_xlabel("bits", color="#9aa1ad", fontsize=10)
    ax.legend(facecolor=BG, edgecolor="#2c313f", labelcolor=INK, fontsize=9)
    _save(fig, "quantization-bit-depth-and-banding.png")

    # --- L4: Delta-E by route ----------------------------------------------------
    fig = _figure()
    ax = _axes(fig)
    paths = numbers["l4_colour"]["accuracy"]["paths"]
    labels = ["assume\nsRGB", "published\nmatrix", "fitted,\nheld out"]
    values = [
        paths["naive_srgb"]["all"]["mean"],
        paths["published_matrix"]["variants"]["scene_raw"]["mean"],
        paths["fitted_held_out"]["test"]["mean"],
    ]
    ax.bar(labels, values, color=[WARM, WARM, GOOD], width=0.55)
    ax.set_ylabel("ΔE*ab", color="#9aa1ad", fontsize=10)
    ax.tick_params(axis="x", labelsize=10, colors="#9aa1ad")
    _save(fig, "image-formation-and-color-spaces.png")

    # --- unit 1.2 L5: demosaicing error, flat against edge ------------------------
    fig = _figure()
    ax = _axes(fig)
    error = numbers["cfa"]["demosaic_error"]
    ax.bar(
        ["flat\npatches", "whole\nframe", "strongest\nedges"],
        [error["rms_chart_patches_dn"], error["rms_whole_frame_dn"], error["rms_top_1pct_gradient_dn"]],
        color=[GOOD, ACCENT, WARM], width=0.55,
    )
    ax.set_ylabel("RMS error (DN)", color="#9aa1ad", fontsize=10)
    ax.tick_params(axis="x", labelsize=10, colors="#9aa1ad")
    _save(fig, "sensing-colour-the-bayer-cfa.png")

    # --- L5 (1.3): per-patch colour error, naive against fitted --------------------
    fig = _figure()
    ax = _axes(fig)
    paths = numbers["l4_colour"]["accuracy"]["paths"]
    naive = paths["naive_srgb"]["per_patch"]
    fitted = paths["fitted_held_out"]["per_patch"]
    keys = sorted(naive, key=int)
    index = np.arange(len(keys))
    ax.bar(index - 0.2, [naive[k] for k in keys], width=0.4, color=WARM, label="assume sRGB")
    ax.bar(index + 0.2, [fitted[k] for k in keys], width=0.4, color=GOOD, label="fitted 3x3")
    ax.set_ylabel("ΔE*ab", color="#9aa1ad", fontsize=10)
    ax.set_xticks([])
    ax.legend(facecolor=BG, edgecolor="#2c313f", labelcolor=INK, fontsize=9)
    _save(fig, "how-close-is-your-colour.png")

    # --- L5: bytes against edge agreement, over the JPEG sweep -------------------
    fig = _figure()
    ax = _axes(fig)
    rows = [r for r in numbers["l5_formats"]["sweep"]["whole_frame"]["rows"]
            if r.get("label", "").startswith("JPEG")]
    ax.plot([r["bytes"] / 1e3 for r in rows], [r["edges"]["f1"] for r in rows],
            color=WARM, linewidth=2.4, marker="o", markersize=7)
    ax.set_xscale("log")
    ax.set_xlabel("KB", color="#9aa1ad", fontsize=10)
    ax.set_ylabel("edge agreement (F1)", color="#9aa1ad", fontsize=10)
    _save(fig, "image-file-formats-and-compression.png")


if __name__ == "__main__":
    data_covers()
