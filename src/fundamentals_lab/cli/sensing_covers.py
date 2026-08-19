"""Cover backgrounds for unit 1.2, rendered from the measurements themselves.

Backgrounds only: the title card composites its own text over the left two-thirds,
so nothing here is labelled and every image is weighted to the right.
"""

import json

import click
import cv2
import matplotlib
import numpy as np
from loguru import logger

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from fundamentals_lab.config import OUTPUT_DIR  # noqa: E402

BG = "#0b0e18"
INK = "#e5e7eb"
ACCENT = "#6366f1"
WARM = "#f59e0b"
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
def sensing_covers() -> None:
    """Render the five cover backgrounds of unit 1.2."""
    numbers = json.loads((OUTPUT_DIR / "sensing_numbers.json").read_text())

    # Hub: the merged scene itself, cropped to the right two-thirds.
    scene = cv2.imread(str(OUTPUT_DIR / "figures" / "scene-merged.webp"))
    if scene is None:
        raise click.ClickException("run `uv run sensing-figures` first")
    scaled = cv2.resize(scene, (int(H * scene.shape[1] / scene.shape[0]), H))
    canvas = np.full((H, W, 3), (24, 14, 11), dtype=np.uint8)
    crop = scaled[:, max(scaled.shape[1] - int(W * 0.62), 0) :][:, : int(W * 0.62)]
    canvas[:, W - crop.shape[1] :] = crop
    fade = np.linspace(0.0, 1.0, crop.shape[1] // 3)[None, :, None]
    left = W - crop.shape[1]
    canvas[:, left : left + fade.shape[1]] = (
        canvas[:, left : left + fade.shape[1]] * fade
        + np.full_like(canvas[:, left : left + fade.shape[1]], (24, 14, 11)) * (1 - fade)
    ).astype(np.uint8)
    COVER_DIR.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(COVER_DIR / "image-sensing.png"), canvas)
    logger.info(f"wrote {COVER_DIR / 'image-sensing.png'}")

    # L1: the exposure ladder — counts against shutter time, on log axes.
    frames = numbers["frames"]
    fig = _figure()
    ax = _axes(fig)
    shutter = [f["shutter_s"] for f in frames]
    counts = [f["median_dn_g1"] for f in frames]
    ax.loglog(shutter, counts, "o-", color=WARM, linewidth=2.5, markersize=9)
    ax.set_xlabel("exposure time (s)", color=INK)
    ax.set_ylabel("median count (DN)", color=INK)
    _save(fig, "from-photons-to-counts.png")

    # L2: the photon transfer curve, variance against signal.
    curve = numbers["noise"]["channels"]["G1"]["curve"]
    fig = _figure()
    ax = _axes(fig)
    means = [p["mean_dn"] for p in curve]
    variances = [p["variance_dn2"] for p in curve]
    ax.plot(means, variances, "o", color=ACCENT, markersize=9)
    gain = numbers["noise"]["channels"]["G1"]["gain_dn_per_electron"]
    intercept = numbers["noise"]["channels"]["G1"]["intercept_dn2"]
    line = np.linspace(min(means), max(means), 64)
    ax.plot(line, gain * line + intercept, color=INK, linewidth=2)
    ax.set_xlabel("signal (DN)", color=INK)
    ax.set_ylabel("noise variance (DN²)", color=INK)
    _save(fig, "sensor-noise-and-dynamic-range.png")

    # L3: the tone curve — code value against relative light.
    ramp = numbers["response"]["ramp"]
    fig = _figure()
    ax = _axes(fig)
    relative = np.array([p["relative_to_saturation"] for p in ramp])
    ax.plot(relative, [p["code_value"] for p in ramp], "o", color=WARM, markersize=11)
    grid = np.linspace(0.002, 1.0, 200)
    from fundamentals_lab.sensing.response import srgb_encode  # noqa: PLC0415

    ax.plot(grid, srgb_encode(grid), color=ACCENT, linewidth=2.5)
    ax.plot(grid, 255 * grid, color="#374151", linewidth=1.5, linestyle="--")
    ax.set_xscale("log")
    ax.set_xlabel("light, relative to saturation", color=INK)
    ax.set_ylabel("code value", color=INK)
    _save(fig, "the-camera-response-curve.png")

    # L4: predicted luminance against metered luminance, both on log axes.
    predictions = numbers["hdr"]["predictions"]
    fig = _figure()
    ax = _axes(fig)
    metered = [p["metered_cd_m2"] for p in predictions]
    predicted = [p["predicted_cd_m2"] for p in predictions]
    limits = [min(metered) * 0.6, max(metered) * 1.6]
    ax.plot(limits, limits, color="#374151", linewidth=1.5, linestyle="--")
    ax.loglog(metered, predicted, "o", color=WARM, markersize=9, alpha=0.9)
    ax.set_xlabel("metered luminance (cd/m²)", color=INK)
    ax.set_ylabel("merged bracket (cd/m²)", color=INK)
    _save(fig, "high-dynamic-range-imaging.png")


if __name__ == "__main__":
    sensing_covers()
