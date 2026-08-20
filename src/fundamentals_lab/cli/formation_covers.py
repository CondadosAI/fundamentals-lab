"""Cover backgrounds for unit 1.1, rendered from the unit's own measurements.

Backgrounds only: the title card composites its text over the left two-thirds, so
nothing here is labelled and every image is weighted to the right.

No third-party pixels. Everything below is drawn from numbers in
output/formation_numbers.json, which sidesteps the licence question entirely: the
fisheye photographs are GPL-2.0 and a crop of one must not be published, and even
the Apache-2.0 chessboard set would need its licence carried alongside. Plots of
our own measurements carry neither obligation.
"""

import click
import matplotlib
import numpy as np
from loguru import logger

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from fundamentals_lab.config import OUTPUT_DIR  # noqa: E402
from fundamentals_lab.formation import numbers  # noqa: E402

BG = "#0b0e18"
ACCENT = "#6366f1"
WARM = "#f59e0b"
GOOD = "#34d399"
BAD = "#ef4444"
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


def _distorted_grid(k, cx, cy, fx, fy, w, h, n=13, m=10):
    """The measured lens applied to a regular grid, in pixels."""
    k1, k2, p1, p2, k3 = k[:5]
    us = np.linspace(0, w - 1, n)
    vs = np.linspace(0, h - 1, m)
    grid_u, grid_v = np.meshgrid(us, vs)
    x = (grid_u - cx) / fx
    y = (grid_v - cy) / fy
    r2 = x**2 + y**2
    radial = 1 + k1 * r2 + k2 * r2**2 + k3 * r2**3
    xd = x * radial + 2 * p1 * x * y + p2 * (r2 + 2 * x**2)
    yd = y * radial + p1 * (r2 + 2 * y**2) + 2 * p2 * x * y
    return cx + fx * xd, cy + fy * yd


def cover_hub(d: dict) -> None:
    """The unit in one picture: the same grid, straight and as the lens draws it."""
    c = d["calibration"]
    w, h = c["image_size"]
    fig = _figure()
    ax = fig.add_axes((0.40, 0.10, 0.56, 0.80), facecolor=BG)
    ax.set_axis_off()
    du, dv = _distorted_grid(c["dist"], c["cx"], c["cy"], c["fx"], c["fy"], w, h)
    us = np.linspace(0, w - 1, du.shape[1])
    vs = np.linspace(0, h - 1, du.shape[0])
    for j in range(du.shape[0]):
        ax.plot(us, np.full_like(us, vs[j]), color=ACCENT, lw=0.8, alpha=0.28)
    for i in range(du.shape[1]):
        ax.plot(np.full_like(vs, us[i]), vs, color=ACCENT, lw=0.8, alpha=0.28)
    for j in range(du.shape[0]):
        ax.plot(du[j], dv[j], color=WARM, lw=1.8)
    for i in range(du.shape[1]):
        ax.plot(du[:, i], dv[:, i], color=WARM, lw=1.8)
    ax.set_xlim(-40, w + 40)
    ax.set_ylim(h + 40, -40)
    _save(fig, "how-a-camera-forms-an-image.png")


def cover_pinhole(d: dict) -> None:
    """Where the straight-line model misses, against how far out the point sits."""
    p = d["projection"]
    prof = d["distortion"]["radial_profile"]
    fig = _figure()
    ax = _axes(fig)
    r = [row["r_px"] for row in prof]
    shift = [row["shift_px"] for row in prof]
    ax.plot(r, shift, color=WARM, lw=3)
    ax.fill_between(r, 0, shift, color=WARM, alpha=0.12)
    ax.axhline(0, color=ACCENT, lw=2, ls="--", alpha=0.8)
    inner = p["pinhole_only_error_px"]["innermost_third"]
    outer = p["pinhole_only_error_px"]["outermost_third"]
    for row, colour in ((inner, GOOD), (outer, BAD)):
        ax.scatter([row["mean_radius_px"]], [row["mean_error_px"]], s=150, color=colour, zorder=5)
    ax.set_xlabel("distance from the principal point (px)", color="#6b7280", fontsize=10)
    ax.set_ylabel("displacement (px)", color="#6b7280", fontsize=10)
    _save(fig, "pinhole-and-perspective-projection.png")


def cover_dof(d: dict) -> None:
    """The in-focus band at both apertures, on a shared distance axis."""
    o = d["depth_of_field"]
    fig = _figure()
    ax = _axes(fig, rect=(0.42, 0.20, 0.54, 0.62))
    for i, (stop, colour) in enumerate((("4.0", BAD), ("22.0", GOOD))):
        row = o["apertures"][stop]
        ax.barh(
            i, row["far_m"] - row["near_m"], left=row["near_m"],
            height=0.45, color=colour, alpha=0.75,
        )
    ax.axvline(o["recovered_subject_distance_m"], color=ACCENT, lw=2, ls="--")
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["f/4", "f/22"], color="#9ca3af", fontsize=12)
    ax.set_xlim(1.2, 1.85)
    ax.set_xlabel("distance (m)", color="#6b7280", fontsize=10)
    _save(fig, "lenses-focal-length-depth-of-field.png")


def cover_distortion(d: dict) -> None:
    """How far the lens moves each point, drawn as the displacement itself.

    Deliberately not the warped grid: that is the hub's image, and two covers of
    the same picture read as a duplicate in a list of posts. This one shows the
    quantity lesson 3 measures — the vector from where the model says a point
    belongs to where the lens puts it — which is near zero in the middle and
    unmissable at the edges.
    """
    c = d["calibration"]
    w, h = c["image_size"]
    fig = _figure()
    ax = fig.add_axes((0.40, 0.10, 0.56, 0.80), facecolor=BG)
    ax.set_axis_off()
    du, dv = _distorted_grid(c["dist"], c["cx"], c["cy"], c["fx"], c["fy"], w, h, n=19, m=15)
    us = np.linspace(0, w - 1, du.shape[1])
    vs = np.linspace(0, h - 1, du.shape[0])
    grid_u, grid_v = np.meshgrid(us, vs)
    shift = np.hypot(du - grid_u, dv - grid_v)
    ax.quiver(
        grid_u, grid_v, du - grid_u, dv - grid_v,
        shift, angles="xy", scale_units="xy", scale=1,
        cmap="autumn_r", width=0.004, alpha=0.95,
    )
    ax.scatter([c["cx"]], [c["cy"]], s=90, color=ACCENT, zorder=5)
    ax.set_xlim(-70, w + 70)
    ax.set_ylim(h + 70, -70)
    _save(fig, "lens-distortion-vignetting-aberration.png")


def cover_fisheye(d: dict) -> None:
    """Both models answering the same question, one of them leaving the plot."""
    f = d["fisheye"]
    fig = _figure()
    ax = _axes(fig)
    ff, kf = f["fisheye"]["fx"], f["fisheye"]["k"]
    fp = f["pinhole"]["fx"]
    k1, k2, _p1, _p2, k3 = f["pinhole"]["dist"][:5]
    t = np.radians(np.linspace(0, 89, 400))
    th_d = t * (1 + kf[0] * t**2 + kf[1] * t**4 + kf[2] * t**6 + kf[3] * t**8)
    r = np.tan(t)
    pin = fp * r * (1 + k1 * r**2 + k2 * r**4 + k3 * r**6)
    ax.plot(np.degrees(t), ff * th_d, color=GOOD, lw=3.5)
    ax.plot(np.degrees(t), pin, color=BAD, lw=3.5)
    ax.axhline(646, color="#6b7280", lw=1.6, ls="--")
    ax.set_ylim(0, 1400)
    ax.set_xlim(0, 90)
    ax.set_xlabel("incidence angle θ (deg)", color="#6b7280", fontsize=10)
    ax.set_ylabel("radius on the sensor (px)", color="#6b7280", fontsize=10)
    _save(fig, "wide-angle-and-fisheye-models.png")


@click.command()
def formation_covers() -> None:
    """Render unit 1.1's five cover backgrounds from output/formation_numbers.json."""
    d = numbers.load()
    missing = {"calibration", "distortion", "depth_of_field", "fisheye", "projection"} - set(d)
    if missing:
        raise click.ClickException(
            f"missing sections {sorted(missing)} — run the measurement commands first"
        )
    cover_hub(d)
    cover_pinhole(d)
    cover_dof(d)
    cover_distortion(d)
    cover_fisheye(d)


if __name__ == "__main__":
    formation_covers()
