"""The photographs unit 1.3 publishes.

Curves, staircases and memory diagrams in the posts are hand-authored SVG. What has to
come from here is anything made of pixels: the grid itself at a zoom where you can read
the numbers, the ribbing that aliases, the patches losing levels, the chart the colour
lesson samples, and what an encoder leaves behind.

Every image derives from Mark Fairchild's HDR Photographic Survey and carries its
attribution in the post that shows it.
"""

import subprocess

import click
import cv2
import numpy as np
from loguru import logger
from PIL import Image

from fundamentals_lab.config import (
    CONTENT_CROP,
    GLOW_CROP,
    DECIMATION_FACTORS,
    FIGURE_DIR,
    GRAIN_CROP,
    GRAIN_WINDOW,
    IMAGEDATA_CANNY,
)
from fundamentals_lab.imagedata import quantize, scene
from fundamentals_lab.sensing import charts

# Every figure in this unit ships lossless, and that is a measured decision rather than
# the site's usual q82 default. Encoded at q82 these files are 7-18% of the size, but the
# maximum per-pixel error is 37 on the moire panels, 63 on the posterisation panels and
# 238 on the amplified difference map -- and those artifacts are the entire content of
# the figures. A lossy encoder would smooth away exactly what the reader is looking at.
# `l1_grid` needs no argument at all: lossless is 30 KB against q82's 38 KB, because flat
# tiles and text are what lossless is good at. Panel sizes are kept modest instead.
WEBP = [cv2.IMWRITE_WEBP_QUALITY, 82]
LOSSLESS = [cv2.IMWRITE_WEBP_QUALITY, 101]
INK = (235, 235, 235)
DIM = (150, 150, 150)


def _save(image: np.ndarray, name: str, params: list[int] | None = None) -> None:
    path = FIGURE_DIR / name
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), image, params or WEBP)
    logger.info(f"wrote {path} ({path.stat().st_size / 1024:.0f} KB)")


def _label(image: np.ndarray, text: str, origin: tuple[int, int], scale: float = 0.6) -> None:
    """Caption drawn twice: black underneath, so it reads on any background."""
    cv2.putText(image, text, origin, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(image, text, origin, cv2.FONT_HERSHEY_SIMPLEX, scale, INK, 1, cv2.LINE_AA)


def _panels(images: list[np.ndarray], captions: list[str], gap: int = 12) -> np.ndarray:
    """Lay panels out in a row on a dark background, each with a caption above it."""
    height = max(image.shape[0] for image in images)
    width = sum(image.shape[1] for image in images) + gap * (len(images) + 1)
    header = 30
    canvas = np.full((height + header + gap, width, 3), 18, np.uint8)
    x = gap
    for image, caption in zip(images, captions):
        canvas[header : header + image.shape[0], x : x + image.shape[1]] = image
        _label(canvas, caption, (x, header - 10))
        x += image.shape[1] + gap
    return canvas


def figure_grid(frame) -> None:
    """L1 — the array, at a zoom where the numbers are legible.

    A 16x10 neighbourhood of the probe pixel, one photosite per tile, each tile printed
    with the value the file holds. This is the picture behind "an image is a grid of
    integers": nothing here is a rendering, it is the numbers themselves.
    """
    patch = next(p for p in charts.patches("bright") if p.number == 19)
    cx, cy = (int(round(v)) for v in patch.centre)
    columns, rows, tile = 16, 10, 58
    block = frame.rgb8[cy : cy + rows, cx : cx + columns]

    canvas = np.full((rows * tile, columns * tile, 3), 18, np.uint8)
    for row in range(rows):
        for column in range(columns):
            r, g, b = (int(v) for v in block[row, column])
            y0, x0 = row * tile, column * tile
            canvas[y0 : y0 + tile - 2, x0 : x0 + tile - 2] = (b, g, r)
            ink = (20, 20, 20) if (0.299 * r + 0.587 * g + 0.114 * b) > 110 else (245, 245, 245)
            cv2.putText(
                canvas, str(g), (x0 + 8, y0 + tile // 2 + 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, ink, 1, cv2.LINE_AA,
            )
    banner = np.full((40, canvas.shape[1], 3), 18, np.uint8)
    _label(banner, f"green channel, row {cy}, column {cx}", (10, 26), 0.55)
    _save(np.vstack([banner, canvas]), "l1_grid.webp", LOSSLESS)


def figure_aliasing(frame) -> None:
    """L2 — the same ribbing sampled three ways.

    The reference is an area average, which is what a smaller sensor would have
    measured. Beside it, the same reduction done by keeping one pixel in every k. Each
    panel is blown back up with nearest-neighbour so the comparison is at one size and
    nothing is smoothed on the way to the page.
    """
    x0, y0, x1, y1 = GRAIN_CROP
    start, length = GRAIN_WINDOW
    crop = frame.bgr8[y0 + start : y0 + start + length * 2, x0 : x0 + 460]
    height, width = crop.shape[:2]

    def boost(image: np.ndarray) -> np.ndarray:
        return cv2.convertScaleAbs(image, alpha=2.4)

    panels, captions = [boost(crop)], ["full resolution"]
    for factor in DECIMATION_FACTORS:
        size = (width // factor, height // factor)
        nearest = cv2.resize(crop, size, interpolation=cv2.INTER_NEAREST)
        panels.append(boost(cv2.resize(nearest, (width, height), interpolation=cv2.INTER_NEAREST)))
        captions.append(f"every {factor}th pixel")
    _save(_panels(panels[:3], captions[:3]), "l2_aliasing.webp", LOSSLESS)

    area = cv2.resize(crop, (width // 8, height // 8), interpolation=cv2.INTER_AREA)
    nearest = cv2.resize(crop, (width // 8, height // 8), interpolation=cv2.INTER_NEAREST)
    up = lambda image: cv2.resize(image, (width, height), interpolation=cv2.INTER_NEAREST)
    _save(
        _panels(
            [boost(crop), boost(up(nearest)), boost(up(area))],
            ["full resolution", "every 8th pixel", "averaged, then sampled"],
        ),
        "l2_prefilter.webp",
        LOSSLESS,
    )


def figure_bit_depth(frame) -> None:
    """L3 — the lamp's shade as the levels are taken away.

    The chart's patches are flat, so requantising them changes their numbers without
    changing how they look. Posterisation needs gradation, and the shade's falloff is
    the only real gradation in this scene. Note what the figure honestly shows: 12, 8
    and 6 bits are hard to tell apart here, and only at 4 does the curve break into
    bands -- which is the same conclusion the noise-floor arithmetic reaches from the
    other direction.
    """
    x0, y0, x1, y1 = GLOW_CROP
    glow = frame.rgb16[y0:y1, x0:x1].astype(np.float64) / 16.0  # 16-bit -> the 12-bit scale

    panels, captions = [], []
    for bits in (12, 8, 6, 4):
        reduced = quantize.requantize(glow, bits) / 16.0
        panels.append(np.clip(reduced, 0, 255).astype(np.uint8)[:, :, ::-1])
        captions.append(f"{bits} bits")
    _save(_panels(panels, captions), "l3_bit_depth.webp", LOSSLESS)


def figure_chart(frame) -> None:
    """L4 — where the 24 patches are sampled, and what they are numbered."""
    patches = charts.patches("bright")
    x0 = min(p.box[0] for p in patches) - 40
    y0 = min(p.box[1] for p in patches) - 40
    x1 = max(p.box[2] for p in patches) + 40
    y1 = max(p.box[3] for p in patches) + 40
    canvas = frame.bgr8[y0:y1, x0:x1].copy()
    for patch in patches:
        px0, py0, px1, py1 = patch.box
        cv2.rectangle(canvas, (px0 - x0, py0 - y0), (px1 - x0, py1 - y0), DIM, 1)
        _label(canvas, str(patch.number), (px0 - x0 + 2, py0 - y0 - 5), 0.45)
    scale = 1100 / canvas.shape[1]
    _save(cv2.resize(canvas, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA), "l4_chart.webp")


def figure_compression(frame) -> None:
    """L5 — what q25 leaves behind, and where the edge detector notices."""
    x0, y0, x1, y1 = CONTENT_CROP
    crop = frame.bgr8[y0 + 940 : y0 + 1280, x0 + 1140 : x0 + 1580]
    ok, buffer = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 25])
    decoded = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
    difference = cv2.convertScaleAbs(cv2.absdiff(crop, decoded), alpha=8.0)

    low, high = IMAGEDATA_CANNY
    edges_before = cv2.Canny(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY), low, high)
    edges_after = cv2.Canny(cv2.cvtColor(decoded, cv2.COLOR_BGR2GRAY), low, high)
    overlay = np.zeros_like(crop)
    overlay[:, :, 1] = cv2.bitwise_and(edges_before, edges_after)  # survived: green
    overlay[:, :, 2] = cv2.subtract(edges_before, edges_after)  # gone: red

    _save(
        _panels(
            [crop, decoded, difference, overlay],
            ["source", "JPEG q25", "difference x8", "edges: kept / lost"],
        ),
        "l5_compression.webp",
        LOSSLESS,
    )


@click.command()
@click.option("--frame", default=None, help="NEF to work on; defaults to the unit's frame.")
def data_figures(frame: str | None) -> None:
    """Render unit 1.3's figures into output/figures/."""
    exposure = scene.load(frame) if frame else scene.load()
    figure_grid(exposure)
    figure_aliasing(exposure)
    figure_bit_depth(exposure)
    figure_chart(exposure)
    figure_compression(exposure)

    logger.info("in the wild")
    figure_wild_channels()
    figure_wild_brick()
    figure_wild_sky()
    figure_wild_hue()
    figure_wild_text()
    figure_wild_chart()
    figure_wild_wheel()

    logger.info("in the wild — units 1.1, 1.2 and 3.1")
    figure_wild_vanishing()
    figure_wild_dof()
    figure_wild_fisheye()
    figure_wild_astro()
    figure_wild_fringing()
    figure_wild_canny()
    figure_wild_corners()
    figure_wild_extras()




# =============================================================================
# "In the wild" — the same ideas, on photographs
# =============================================================================
def figure_wild_channels() -> None:
    """L1 — the same photograph read in the two channel orders."""
    from fundamentals_lab.imagedata import wild

    image = wild.load("telephone-box.jpg")
    scale = 520 / image.shape[0]
    small = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    _save(_panels([small, small[:, :, ::-1]], ["read as BGR (correct)", "read as RGB (wrong)"]),
          "wild_l1_channels.webp")


def figure_wild_brick() -> None:
    """L2 — a brick wall shrunk to a thumbnail, sampled and averaged."""
    from fundamentals_lab.imagedata import wild

    image = wild.load("brick-wall.jpg")
    crop = image[: 1600, : 1600]
    for factor, name in ((32, "wild_l2_brick.webp"),):
        size = (crop.shape[1] // factor, crop.shape[0] // factor)
        nearest = cv2.resize(crop, size, interpolation=cv2.INTER_NEAREST)
        area = cv2.resize(crop, size, interpolation=cv2.INTER_AREA)
        big = lambda x: cv2.resize(x, (420, 420), interpolation=cv2.INTER_NEAREST)
        _save(_panels([cv2.resize(crop, (420, 420), interpolation=cv2.INTER_AREA),
                       big(nearest), big(area)],
                      ["the wall", f"every {factor}th pixel", "averaged first"]), name)


def figure_wild_sky() -> None:
    """L3 — a real sky gradient losing its levels."""
    from fundamentals_lab.imagedata import wild
    from fundamentals_lab.imagedata.quantize import requantize

    image = wild.load("twilight-sky.jpg")
    sky = image[: image.shape[0] // 3, :]
    strip = cv2.resize(sky, (460, 300), interpolation=cv2.INTER_AREA).astype(np.float64)
    panels, captions = [], []
    for bits in (8, 6, 5, 4):
        step = 256 / 2**bits
        panels.append(np.clip(np.rint(strip / step) * step, 0, 255).astype(np.uint8))
        captions.append(f"{bits} bits")
    _save(_panels(panels, captions), "wild_l3_sky.webp")


def figure_wild_hue() -> None:
    """L4 — a hue window on a market stall, narrow and wide."""
    from fundamentals_lab.imagedata import wild

    image = wild.load("souk-fruit.jpg")
    small = cv2.resize(image, (620, int(620 * image.shape[0] / image.shape[1])),
                       interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    hue, saturation, value = cv2.split(hsv)
    panels, captions = [small], ["the stall"]
    for low, high in ((10, 20), (0, 40)):
        mask = (hue >= low) & (hue <= high) & (saturation > 90) & (value > 60)
        shown = small.copy()
        shown[~mask] = (shown[~mask] * 0.18).astype(np.uint8)
        panels.append(shown)
        captions.append(f"hue {low * 2}-{high * 2}°")
    _save(_panels(panels, captions), "wild_l4_hue.webp")


def figure_wild_text() -> None:
    """L6 — what an encoder leaves on the paper around the letters."""
    from fundamentals_lab.imagedata import wild

    image = wild.load("newspaper-1814.jpg")
    height, width = image.shape[:2]
    crop = image[height // 3 : height // 3 + 460, width // 4 : width // 4 + 460]
    ok, buffer = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 25])
    decoded = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
    difference = cv2.convertScaleAbs(cv2.absdiff(crop, decoded), alpha=10.0)
    _save(_panels([crop, decoded, difference],
                  ["the scan", "JPEG q25", "difference x10 — a halo on the paper"]),
          "wild_l6_text.webp")


def figure_wild_wheel() -> None:
    """L2 extra — the wagon-wheel effect, as an animation.

    Encoded lossy rather than lossless, and that is measured rather than assumed: this
    is video frames, not a 256-colour GIF, so the repo's lossless rule for animated GIFs
    does not apply and lossless comes out several times larger for no visible gain.
    """
    from fundamentals_lab.imagedata import wild

    path = wild.fetch("wagon-wheel.ogv")
    raw = subprocess.run(
        ["ffmpeg", "-v", "error", "-ss", "20", "-i", str(path), "-vf",
         "scale=420:236,crop=300:236:120:0", "-frames:v", "48", "-f", "rawvideo",
         "-pix_fmt", "bgr24", "-"],
        capture_output=True, check=True,
    ).stdout
    frames = np.frombuffer(raw, np.uint8).reshape(-1, 236, 300, 3)
    images = [Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)) for f in frames]
    target = FIGURE_DIR / "wild_l2_wheel.webp"
    images[0].save(target, save_all=True, append_images=images[1:], duration=42, loop=0,
                   format="WEBP", quality=80, method=4)
    logger.info(f"wrote {target} ({target.stat().st_size / 1024:.0f} KB, {len(images)} frames)")


def figure_wild_chart() -> None:
    """L5 — the chart's neutral row, which is neutral by manufacture."""
    from fundamentals_lab.imagedata import wild

    image = wild.load("colorchecker-photo.jpg")
    height, width = image.shape[:2]
    canvas = cv2.resize(image, (760, int(760 * height / width)), interpolation=cv2.INTER_AREA)
    ch, cw = canvas.shape[:2]
    for index in range(6):
        cx = int(cw * (index + 0.5) / 6)
        cy = int(ch * 3.5 / 4)
        half = int(min(cw / 6, ch / 4) * 0.18)
        cv2.rectangle(canvas, (cx - half, cy - half), (cx + half, cy + half), (245, 245, 245), 2)
        _label(canvas, str(19 + index), (cx - half, cy - half - 6), 0.5)
    _save(_panels([canvas], ["the row that should have no colour in it at all"]),
          "wild_l5_chart.webp")


# =============================================================================
# "In the wild" for units 1.1, 1.2 and 3.1
# =============================================================================
def figure_wild_vanishing() -> None:
    """1.1 L1 — the lines that were fitted, and the point they agree on."""
    from fundamentals_lab.imagedata import wild, wild_units

    image = wild.load("cloister.jpg")
    result = wild_units.vanishing_point()
    canvas = image.copy()
    grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    segments = cv2.HoughLinesP(
        cv2.Canny(grey, 40, 120), 1, np.pi / 360, threshold=50,
        minLineLength=int(0.06 * grey.shape[0]), maxLineGap=14,
    ).reshape(-1, 4)
    order = np.argsort(-np.hypot(segments[:, 2] - segments[:, 0], segments[:, 3] - segments[:, 1]))
    drawn = 0
    for x1, y1, x2, y2 in segments[order]:
        angle = np.degrees(np.arctan2(y2 - y1, x2 - x1)) % 180
        if not (12 < angle < 78 or 102 < angle < 168):
            continue
        cv2.line(canvas, (x1, y1), (x2, y2), (90, 200, 245), 3)
        drawn += 1
        if drawn >= 60:
            break
    vx, vy = (int(v) for v in result["vanishing_point_px"])
    cv2.circle(canvas, (vx, vy), 26, (60, 60, 250), 4)
    cv2.line(canvas, (vx - 40, vy), (vx + 40, vy), (60, 60, 250), 3)
    cv2.line(canvas, (vx, vy - 40), (vx, vy + 40), (60, 60, 250), 3)
    scale = 620 / canvas.shape[0]
    _save(cv2.resize(canvas, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA),
          "wild_11_vanishing.webp")


def figure_wild_dof() -> None:
    """1.1 L2 — the sharp band, and the blur either side of it."""
    from fundamentals_lab.imagedata import wild

    image = wild.load("souk-fruit.jpg")
    height, width = image.shape[:2]
    strip = lambda frac: cv2.resize(
        image[int(frac * height) : int((frac + 0.1) * height), width // 3 : width // 3 + 900],
        (400, 180), interpolation=cv2.INTER_AREA)
    _save(_panels([strip(0.05), strip(0.6), strip(0.85)],
                  ["the awning, far behind", "the fruit, in focus", "the front row, nearer"]),
          "wild_11_dof.webp")


def figure_wild_fisheye() -> None:
    """1.1 L4 — the image circle, against the radius the model predicts."""
    from fundamentals_lab.imagedata import wild, wild_units

    image = wild.load("fisheye-kashan.jpg")
    result = wild_units.fisheye_image_circle()
    canvas = image.copy()
    cx, cy = (int(v) for v in result["disc_centre_px"])
    cv2.circle(canvas, (cx, cy), int(result["predicted_radius_px"]), (60, 60, 250), 10)
    cv2.circle(canvas, (cx, cy), int(result["measured_radius_px_from_bbox"]), (90, 230, 120), 10)
    scale = 620 / canvas.shape[0]
    _save(cv2.resize(canvas, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA),
          "wild_11_fisheye.webp")


def figure_wild_astro() -> None:
    """1.2 L1/L2 — the emptiest sky in the frame, stretched until the noise shows."""
    from fundamentals_lab.imagedata import wild, wild_units

    image = wild.load("orion-astro.jpg")
    result = wild_units.astro_noise()
    x, y, tile, _ = result["background_tile_px"]
    patch = image[y : y + tile, x : x + tile]
    scale = 620 / image.shape[1]
    whole = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    cv2.rectangle(whole, (int(x * scale), int(y * scale)),
                  (int((x + tile) * scale), int((y + tile) * scale)), (90, 230, 120), 2)
    big = cv2.resize(patch, (300, 300), interpolation=cv2.INTER_NEAREST)
    _save(_panels([whole, cv2.convertScaleAbs(big, alpha=14.0)],
                  ["the frame", "the background tile, x14"]), "wild_12_astro.webp")


def figure_wild_fringing() -> None:
    """1.2 L5 — colour error that sits on edges and nowhere else."""
    from fundamentals_lab.imagedata import wild

    image = wild.load("telephone-box.jpg").astype(np.float64)
    height, width = image.shape[:2]
    crop = image[int(0.30 * height) : int(0.30 * height) + 500, int(0.42 * width) : int(0.42 * width) + 500]
    blue, _, red = cv2.split(crop)
    high_pass = lambda c: c - cv2.GaussianBlur(c, (0, 0), 3)
    difference = np.abs(high_pass(red) - high_pass(blue))
    _save(_panels([crop.astype(np.uint8),
                   cv2.applyColorMap(cv2.convertScaleAbs(difference, alpha=6.0), cv2.COLORMAP_INFERNO)],
                  ["the crop", "red minus blue, high-passed, x6"]), "wild_12_fringing.webp")


def figure_wild_canny() -> None:
    """3.1 L4 — one low threshold, one high, and both together."""
    from fundamentals_lab.imagedata import wild

    image = wild.load("cloister.jpg")
    grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    panels, captions = [], []
    for low, high, caption in ((60, 60, "threshold 60 alone"), (180, 180, "threshold 180 alone"),
                               (60, 180, "60 and 180 together")):
        edges = cv2.Canny(grey, low, high)
        panels.append(cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR))
        captions.append(caption)
    scale = 420 / grey.shape[0]
    panels = [cv2.resize(p, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) for p in panels]
    _save(_panels(panels, captions), "wild_31_canny.webp")


def figure_wild_corners() -> None:
    """3.1 L5 — the corners that come back after a rotation, and the ones that do not."""
    from fundamentals_lab.imagedata import wild

    image = wild.load("souk-fruit.jpg")
    grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    grey = cv2.resize(grey, (grey.shape[1] // 3, grey.shape[0] // 3), interpolation=cv2.INTER_AREA)
    small = cv2.resize(image, (grey.shape[1], grey.shape[0]), interpolation=cv2.INTER_AREA)
    corners = cv2.goodFeaturesToTrack(grey, maxCorners=600, qualityLevel=0.01, minDistance=8,
                                      useHarrisDetector=True, k=0.04).reshape(-1, 2)
    canvas = small.copy()
    for x, y in corners:
        cv2.circle(canvas, (int(x), int(y)), 5, (90, 230, 120), 2)
    scale = 620 / canvas.shape[1]
    _save(cv2.resize(canvas, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA),
          "wild_31_corners.webp")


def figure_wild_extras() -> None:
    """The remaining subjects, shown plainly beside what was measured on them."""
    from fundamentals_lab.imagedata import wild, wild_units

    # 1.1 L3 — vignetting: the wall, and the same wall with its own centre divided out,
    # which is what makes a 0.2-stop falloff visible at all.
    wall = wild.load("brick-wall.jpg")
    grey = cv2.cvtColor(wall, cv2.COLOR_BGR2GRAY).astype(np.float64)
    flat = cv2.GaussianBlur(grey, (0, 0), 120)
    relative = np.clip(128 * flat / flat.max(), 0, 255).astype(np.uint8)
    small = lambda img: cv2.resize(img, (420, int(420 * img.shape[0] / img.shape[1])),
                                   interpolation=cv2.INTER_AREA)
    _save(_panels([small(wall), small(cv2.applyColorMap(
        cv2.convertScaleAbs(relative, alpha=2.4, beta=-180), cv2.COLORMAP_INFERNO))],
        ["the wall", "its brightness alone, stretched"]), "wild_11_vignetting.webp")

    # 1.2 L3 — the neutral ramp the response curve is read from.
    chart = wild.load("colorchecker-photo.jpg")
    height, width = chart.shape[:2]
    row = chart[int(height * 0.75) - int(height * 0.11) : int(height * 0.75) + int(height * 0.11), :]
    _save(_panels([cv2.resize(row, (620, 150), interpolation=cv2.INTER_AREA)],
                  ["white to black in six steps — the ramp the curve is read from"]),
          "wild_12_ramp.webp")

    # 1.2 L4 — the whole twilight frame, which one exposure held.
    sky = wild.load("twilight-sky.jpg")
    _save(_panels([small(sky)], ["one exposure, and it did not need a second"]),
          "wild_12_range.webp")

    # 3.1 L1 — a single scanline across the sharpest edge in the cloister.
    cloister = wild.load("cloister.jpg")
    g = cv2.cvtColor(cloister, cv2.COLOR_BGR2GRAY).astype(np.float64)
    gradient = np.abs(cv2.Sobel(g, cv2.CV_64F, 1, 0, ksize=3)) / 8.0
    row_index = int(g.shape[0] * 0.55)
    column = int(np.argmax(gradient[row_index]))
    strip = cloister[row_index - 60 : row_index + 60, column - 60 : column + 60]
    plot = np.full((240, 360, 3), 18, np.uint8)
    profile = g[row_index, column - 30 : column + 30]
    lo, hi = profile.min(), profile.max()
    for i in range(len(profile) - 1):
        y0 = int(220 - 190 * (profile[i] - lo) / max(hi - lo, 1e-6))
        y1 = int(220 - 190 * (profile[i + 1] - lo) / max(hi - lo, 1e-6))
        cv2.line(plot, (10 + i * 6, y0), (10 + (i + 1) * 6, y1), (245, 200, 90), 2)
    _save(_panels([cv2.resize(strip, (240, 240), interpolation=cv2.INTER_NEAREST), plot],
                  ["one edge, magnified", "the row across it"]), "wild_31_profile.webp")

    # 3.1 L2 — where the gradients point.
    gx = cv2.Sobel(g, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_64F, 0, 1, ksize=3)
    magnitude = np.hypot(gx, gy)
    strong = magnitude > np.percentile(magnitude, 95)
    angles = (np.degrees(np.arctan2(gy[strong], gx[strong])) % 180).astype(int)
    counts = np.bincount(angles, minlength=180).astype(float)
    counts /= counts.max()
    chart_img = np.full((240, 380, 3), 18, np.uint8)
    for a in range(180):
        h = int(200 * counts[a])
        cv2.line(chart_img, (10 + a * 2, 225), (10 + a * 2, 225 - h), (245, 200, 90), 2)
    cv2.putText(chart_img, "0", (8, 238), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)
    cv2.putText(chart_img, "90", (176, 238), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)
    cv2.putText(chart_img, "180", (334, 238), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)
    _save(_panels([small(cloister), chart_img],
                  ["the cloister", "gradient orientation, 0-180 degrees"]), "wild_31_orient.webp")

    # 3.1 L3 — zero crossings on text, unsmoothed and smoothed.
    news = wild.load("newspaper-1814.jpg")
    nh, nw = news.shape[:2]
    crop = cv2.cvtColor(news[nh // 3 : nh // 3 + 320, nw // 4 : nw // 4 + 320], cv2.COLOR_BGR2GRAY)
    panels, captions = [cv2.cvtColor(crop, cv2.COLOR_GRAY2BGR)], ["the text"]
    for sigma in (0.0, 2.0):
        blurred = crop.astype(np.float64) if sigma == 0 else cv2.GaussianBlur(crop.astype(np.float64), (0, 0), sigma)
        sign = np.sign(cv2.Laplacian(blurred, cv2.CV_64F, ksize=3))
        crossings = np.zeros_like(crop)
        crossings[:, :-1][(sign[:, :-1] * sign[:, 1:]) < 0] = 255
        panels.append(cv2.cvtColor(crossings, cv2.COLOR_GRAY2BGR))
        captions.append(f"zero crossings, sigma {sigma:.0f}" if sigma else "zero crossings, no smoothing")
    _save(_panels(panels, captions), "wild_31_zero.webp")


if __name__ == "__main__":
    data_figures()
