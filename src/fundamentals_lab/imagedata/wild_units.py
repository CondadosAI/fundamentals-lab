"""The "in the wild" half for units 1.1, 1.2 and 3.1.

Same contract as `wild.py`: every asset is registered in `cv-assets` with a verified
licence and a sha256, fetched at run time, never redistributed. Each function repeats
one lesson's measurement on a photograph, so the lesson can put the controlled number
and the real one side by side.
"""

import cv2
import numpy as np

from fundamentals_lab.imagedata.wild import load, fetch


def _exif(name: str) -> dict:
    from fundamentals_lab.sensing.hdrps import _sensing_import

    exifread = _sensing_import("exifread")
    with fetch(name).open("rb") as handle:
        tags = exifread.process_file(handle, details=False)

    def ratio(key):
        if key not in tags:
            return None
        value = tags[key].values[0]
        return float(value.num) / float(value.den) if hasattr(value, "num") else float(value)

    return {
        "camera": str(tags.get("Image Model", "")),
        "f_number": ratio("EXIF FNumber"),
        "focal_length_mm": ratio("EXIF FocalLength"),
        "iso": int(str(tags["EXIF ISOSpeedRatings"])) if "EXIF ISOSpeedRatings" in tags else None,
        "exposure_s": str(tags.get("EXIF ExposureTime", "")),
        "subject_distance_m": ratio("EXIF SubjectDistance"),
    }


# =============================================================================
# Unit 1.1 — Image formation
# =============================================================================
def vanishing_point() -> dict:
    """Parallel lines in the world meet at one point in the image. Fitted, not asserted.

    Long segments are detected, the ones near-horizontal or near-vertical in the image
    are set aside (those are the frame's own axes), and the remaining receding lines are
    intersected in the least-squares sense. The residual says how well a single point
    explains them -- which is the pinhole model's central claim, tested on a photograph.
    """
    image = load("cloister.jpg")
    grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    height, width = grey.shape

    segments = cv2.HoughLinesP(
        cv2.Canny(grey, 40, 120), 1, np.pi / 360, threshold=50,
        minLineLength=int(0.06 * height), maxLineGap=14,
    )
    if segments is None:
        raise ValueError("no line segments found")
    segments = segments.reshape(-1, 4)

    # Longest first, and only the structural ones are kept: a cloister is full of short
    # contours -- railing bars, shadows, stonework -- and they point nowhere in
    # particular. The architecture is in the long lines.
    order = np.argsort(-np.hypot(segments[:, 2] - segments[:, 0], segments[:, 3] - segments[:, 1]))
    segments_sorted = segments[order]

    lines, kept = [], []
    for x1, y1, x2, y2 in segments_sorted:
        if len(lines) >= 60:
            break
        angle = np.degrees(np.arctan2(y2 - y1, x2 - x1)) % 180
        # Receding lines are the diagonal ones: the verticals and horizontals in this
        # frame are the building's own axes and meet only at infinity.
        if not (12 < angle < 78 or 102 < angle < 168):
            continue
        # A line as (a, b, c) with a*x + b*y + c = 0, normalised.
        a, b = float(y2 - y1), float(x1 - x2)
        c = -(a * x1 + b * y1)
        norm = np.hypot(a, b)
        lines.append([a / norm, b / norm, c / norm])
        kept.append([int(x1), int(y1), int(x2), int(y2)])
    lengths = [float(np.hypot(k[2] - k[0], k[3] - k[1])) for k in kept]
    lines = np.array(lines)

    # The point closest to every line, in the least-squares sense.
    if len(lines) < 8:
        raise ValueError(f"only {len(lines)} receding lines found; not enough to fit a point")
    A, rhs = lines[:, :2], -lines[:, 2]
    point, *_ = np.linalg.lstsq(A, rhs, rcond=None)
    distances = np.abs(lines[:, :2] @ point + lines[:, 2])

    # One robust pass: lines more than three median distances away are the ones that
    # were never going to the same place -- railings, shadows, a stray contour -- and
    # keeping them would let a handful of outliers set the answer.
    keep = distances < 3 * np.median(distances)
    A, rhs = lines[keep, :2], -lines[keep, 2]
    point, *_ = np.linalg.lstsq(A, rhs, rcond=None)
    distances = np.abs(lines[keep, :2] @ point + lines[keep, 2])
    lines_kept = int(keep.sum())

    return {
        "asset": "cloister.jpg",
        "shape": [height, width],
        "exif": _exif("cloister.jpg"),
        "segments_found": int(len(segments)),
        "shortest_line_used_px": round(min(lengths), 1),
        "longest_line_used_px": round(max(lengths), 1),
        "receding_lines_found": int(len(lines)),
        "receding_lines_after_outlier_pass": lines_kept,
        "vanishing_point_px": [round(float(point[0]), 1), round(float(point[1]), 1)],
        "inside_the_frame": bool(0 <= point[0] < width and 0 <= point[1] < height),
        "median_distance_px": round(float(np.median(distances)), 2),
        "mean_distance_px": round(float(distances.mean()), 2),
        "worst_distance_px": round(float(distances.max()), 1),
        "frame_diagonal_px": round(float(np.hypot(height, width)), 1),
    }


def depth_of_field() -> dict:
    """The depth of field this photograph was taken with, from its own EXIF.

    The camera recorded focal length, aperture and subject distance, which is everything
    the thin-lens depth-of-field formula needs. The prediction is then checked against
    the picture: local sharpness measured in horizontal bands should collapse away from
    the plane of focus.
    """
    image = load("souk-fruit.jpg")
    exif = _exif("souk-fruit.jpg")
    focal, f_number, distance = exif["focal_length_mm"], exif["f_number"], exif["subject_distance_m"]

    # Full-frame circle of confusion, the usual 0.030 mm. Stated rather than derived:
    # it is a convention about acceptable blur, not a property of the lens.
    coc_mm = 0.030
    s_mm = distance * 1000
    hyperfocal_mm = focal**2 / (f_number * coc_mm) + focal
    near_mm = s_mm * (hyperfocal_mm - focal) / (hyperfocal_mm + s_mm - 2 * focal)
    far_mm = s_mm * (hyperfocal_mm - focal) / (hyperfocal_mm - s_mm)

    grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float64)
    bands = 10
    step = grey.shape[0] // bands
    sharpness = [
        round(float(cv2.Laplacian(grey[i * step : (i + 1) * step], cv2.CV_64F).var()), 1)
        for i in range(bands)
    ]
    return {
        "asset": "souk-fruit.jpg",
        "shape": list(image.shape[:2]),
        "exif": exif,
        "circle_of_confusion_mm": coc_mm,
        "hyperfocal_m": round(hyperfocal_mm / 1000, 1),
        "near_limit_m": round(near_mm / 1000, 2),
        "far_limit_m": round(far_mm / 1000, 2) if far_mm > 0 else None,
        "depth_of_field_m": round((far_mm - near_mm) / 1000, 2) if far_mm > 0 else None,
        "sharpness_by_band_top_to_bottom": sharpness,
        "sharpest_band": int(np.argmax(sharpness)),
        "sharpness_ratio_best_to_worst": round(max(sharpness) / min(sharpness), 1),
    }


def vignetting() -> dict:
    """How much darker the corners are than the centre, on a flat, evenly-lit wall.

    A brick wall shot square-on is the closest thing to a flat field a photograph gets.
    Whatever falloff survives is the lens, and it is measured against the centre rather
    than against an assumption.
    """
    image = load("brick-wall.jpg")
    grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float64)
    height, width = grey.shape
    patch = min(height, width) // 12

    def mean_at(fy, fx):
        y, x = int(fy * height), int(fx * width)
        y = min(max(y, patch), height - patch)
        x = min(max(x, patch), width - patch)
        return float(grey[y - patch : y + patch, x - patch : x + patch].mean())

    centre = mean_at(0.5, 0.5)
    corners = {
        "top_left": mean_at(0.06, 0.06), "top_right": mean_at(0.06, 0.94),
        "bottom_left": mean_at(0.94, 0.06), "bottom_right": mean_at(0.94, 0.94),
    }
    worst = min(corners.values())
    return {
        "asset": "brick-wall.jpg",
        "shape": [height, width],
        "exif": _exif("brick-wall.jpg"),
        "centre_code_value": round(centre, 1),
        "corners": {k: round(v, 1) for k, v in corners.items()},
        "worst_corner_fraction_of_centre": round(worst / centre, 3),
        "falloff_stops": round(float(np.log2(centre / worst)), 2),
    }


def fisheye_image_circle() -> dict:
    """A circular fisheye puts a whole hemisphere inside a disc. How big should the disc be?

    The equidistant model says radius grows linearly with angle, r = f x theta. At the
    edge of a 180-degree field theta is pi/2, so the disc's radius should be f x pi/2 on
    the sensor -- a number that follows from the focal length alone and can be checked
    against the picture with a threshold.

    The photograph is a square crop of a full-frame body, so its height is the sensor's
    24 mm short side and that fixes the pixel pitch without needing the crop factor.
    """
    image = load("fisheye-kashan.jpg")
    exif = _exif("fisheye-kashan.jpg")
    grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    height, width = grey.shape

    # The disc against the unexposed corners: Otsu on a heavily blurred copy, so texture
    # inside the frame cannot break the region up.
    blurred = cv2.GaussianBlur(grey, (0, 0), 9)
    _, mask = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, 8)
    biggest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    area = float(stats[biggest, cv2.CC_STAT_AREA])
    cx, cy = (float(v) for v in centroids[biggest])
    radius_area = float(np.sqrt(area / np.pi))
    radius_bbox = float(
        (stats[biggest, cv2.CC_STAT_WIDTH] + stats[biggest, cv2.CC_STAT_HEIGHT]) / 4
    )

    sensor_short_mm = 24.0  # full frame; the square crop's height is the short side
    pitch_mm = sensor_short_mm / height
    focal = exif["focal_length_mm"]
    predicted_mm = focal * (np.pi / 2)
    predicted_px = predicted_mm / pitch_mm
    measured_px = radius_bbox

    return {
        "asset": "fisheye-kashan.jpg",
        "shape": [height, width],
        "exif": exif,
        "model": "equidistant, r = f x theta",
        "sensor_short_side_mm": sensor_short_mm,
        "pixel_pitch_um": round(pitch_mm * 1000, 3),
        "predicted_radius_mm": round(predicted_mm, 2),
        "predicted_radius_px": round(predicted_px, 1),
        "measured_radius_px_from_bbox": round(radius_bbox, 1),
        "measured_radius_px_from_area": round(radius_area, 1),
        "disc_centre_px": [round(cx, 1), round(cy, 1)],
        "agreement_percent": round(100 * measured_px / predicted_px, 1),
        "implied_degrees_per_100px": round(100 * pitch_mm / focal * 180 / np.pi, 2),
    }


# =============================================================================
# Unit 1.2 — Image sensing
# =============================================================================
def astro_noise() -> dict:
    """The noise floor of a real exposure, measured on the emptiest sky in the frame.

    A long-exposure astrophotograph is the honest place to ask what a count is worth:
    most of the frame is signal-free, so whatever varies there is the sensor and the sky
    rather than the subject.
    """
    image = load("orion-astro.jpg")
    grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float64)
    height, width = grey.shape

    # The emptiest tile in the frame, found rather than chosen: lowest local variance
    # among tiles that are also dark.
    tile = 128
    best, best_var = None, None
    for y in range(0, height - tile, tile):
        for x in range(0, width - tile, tile):
            block = grey[y : y + tile, x : x + tile]
            if block.mean() > 60:
                continue
            variance = float(block.var())
            if best_var is None or variance < best_var:
                best, best_var = (y, x), variance
    y, x = best
    background = grey[y : y + tile, x : x + tile]

    clipped_high = int((grey >= 255).sum())
    clipped_low = int((grey <= 0).sum())
    star = float(np.percentile(grey, 99.99))
    return {
        "asset": "orion-astro.jpg",
        "shape": [height, width],
        "background_tile_px": [int(x), int(y), tile, tile],
        "background_mean": round(float(background.mean()), 2),
        "background_sigma": round(float(background.std()), 2),
        "brightest_percentile_99_99": round(star, 1),
        "snr_of_the_subject": round((star - background.mean()) / max(background.std(), 1e-6), 1),
        "clipped_at_255": clipped_high,
        "clipped_at_0": clipped_low,
        "fraction_clipped_high": round(clipped_high / grey.size, 6),
        "usable_stops_in_the_file": round(
            float(np.log2(255 / max(background.std(), 1e-6))), 2
        ),
    }


def response_curve_from_chart() -> dict:
    """What assuming linearity costs, on a photograph of a neutral ramp.

    The chart's bottom row steps from white to black. Read the code values as if they
    were proportional to light, then read them again through the sRGB curve, and the two
    answers for "how much brighter is the white patch than the black one" differ by a
    factor that is the tone curve, made concrete without any reference table.
    """
    image = load("colorchecker-photo.jpg")
    height, width = image.shape[:2]
    grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float64)

    rows = []
    for index in range(6):
        cx = int(width * (index + 0.5) / 6)
        cy = int(height * 3.5 / 4)
        half = int(min(width / 6, height / 4) * 0.18)
        code = float(grey[cy - half : cy + half, cx - half : cx + half].mean())
        normalised = code / 255
        linear = normalised / 12.92 if normalised <= 0.04045 else ((normalised + 0.055) / 1.055) ** 2.4
        rows.append({
            "patch": 19 + index,
            "code_value": round(code, 1),
            "as_if_linear": round(normalised, 4),
            "srgb_linearised": round(linear, 5),
        })
    code_ratio = rows[0]["code_value"] / max(rows[-1]["code_value"], 1e-6)
    linear_ratio = rows[0]["srgb_linearised"] / max(rows[-1]["srgb_linearised"], 1e-9)
    return {
        "asset": "colorchecker-photo.jpg",
        "rows": rows,
        "white_over_black_if_linear": round(code_ratio, 1),
        "white_over_black_through_srgb": round(linear_ratio, 1),
        "factor_between_the_two_readings": round(linear_ratio / code_ratio, 1),
        "note": "the file is not proportional to light; reading it as if it were "
                "understates the range between the lightest and darkest patch",
    }


def single_exposure_range() -> dict:
    """What one exposure of a high-range scene throws away at both ends."""
    image = load("twilight-sky.jpg")
    grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    height, width = grey.shape
    total = grey.size
    high = int((grey >= 254).sum())
    low = int((grey <= 1).sum())
    histogram = cv2.calcHist([grey], [0], None, [256], [0, 256]).ravel()
    occupied = int((histogram > total * 1e-6).sum())
    return {
        "asset": "twilight-sky.jpg",
        "shape": [height, width],
        "pixels": total,
        "clipped_white": high,
        "clipped_black": low,
        "fraction_clipped_white": round(high / total, 6),
        "fraction_clipped_black": round(low / total, 5),
        "code_values_occupied": occupied,
        "code_values_unused": 256 - occupied,
        "stops_the_file_can_hold": 8.0,
    }


def colour_fringing() -> dict:
    """Colour error concentrated on high-contrast edges, in a real photograph.

    Where a demosaicer has to guess and a lens does not focus every wavelength in the
    same place, the two show up together as a red-blue difference that sits on edges and
    nowhere else. Measured against the same difference on flat areas, which is the
    control.
    """
    image = load("telephone-box.jpg").astype(np.float64)
    grey = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_BGR2GRAY)
    blue, green, red = cv2.split(image)

    # Chromatic difference after removing the colour the scene actually has: compare the
    # local *variation* of red against blue rather than their absolute values.
    high_pass = lambda c: c - cv2.GaussianBlur(c, (0, 0), 3)
    difference = np.abs(high_pass(red) - high_pass(blue))

    magnitude = np.hypot(cv2.Sobel(grey, cv2.CV_64F, 1, 0, 3), cv2.Sobel(grey, cv2.CV_64F, 0, 1, 3))
    strong = magnitude > np.percentile(magnitude, 99)
    flat = magnitude < np.percentile(magnitude, 50)
    return {
        "asset": "telephone-box.jpg",
        "shape": list(image.shape[:2]),
        "exif": _exif("telephone-box.jpg"),
        "edge_pixels": int(strong.sum()),
        "flat_pixels": int(flat.sum()),
        "chromatic_difference_on_edges": round(float(difference[strong].mean()), 3),
        "chromatic_difference_on_flat": round(float(difference[flat].mean()), 3),
        "ratio_edge_to_flat": round(float(difference[strong].mean() / difference[flat].mean()), 1),
        "worst_pixel": round(float(difference.max()), 1),
    }


# =============================================================================
# Unit 3.1 — Edge detection
# =============================================================================
def edge_profile() -> dict:
    """How wide a real edge is, measured across the sharpest one in the frame.

    Textbook edges are steps. Photographed edges are ramps, and the ramp has a width:
    the distance over which the profile climbs from 10% to 90% of its total rise. That
    number is what every operator downstream is actually working on.
    """
    image = load("cloister.jpg")
    grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float64)
    height, width = grey.shape

    # Sobel's 3x3 kernel carries a scale of 8 against a true derivative (four in the
    # differencing row, and it sums three rows). Dividing it out is what makes the width
    # below come out in pixels rather than in Sobel units.
    gradient = np.abs(cv2.Sobel(grey, cv2.CV_64F, 1, 0, ksize=3)) / 8.0
    widths = []
    rng = np.random.default_rng(20260819)
    rows = rng.choice(np.arange(height // 4, 3 * height // 4), size=400, replace=False)
    for row in rows:
        profile = grey[row]
        column = int(np.argmax(gradient[row]))
        if column < 12 or column > width - 13:
            continue
        window = profile[column - 10 : column + 11]
        low, high = window.min(), window.max()
        if high - low < 40:  # not a real edge, just texture
            continue
        # Equivalent width: the rise divided by the steepest slope across it. For a step
        # blurred by a Gaussian this is about 2.5 sigma, and unlike a 10-to-90 count on a
        # fixed window it cannot be capped by the window it was measured in.
        widths.append(float((high - low) / max(gradient[row, column], 1e-6)))
    widths = np.array(widths)
    return {
        "asset": "cloister.jpg",
        "shape": [height, width],
        "edges_measured": int(len(widths)),
        "estimator": "equivalent width: (rise) / (steepest slope), about 2.5 sigma for a blurred step",
        "median_width_px": round(float(np.median(widths)), 2),
        "p10_width_px": round(float(np.percentile(widths, 10)), 2),
        "p90_width_px": round(float(np.percentile(widths, 90)), 2),
        "fraction_wider_than_2px": round(float((widths > 2).mean()), 3),
        "implied_median_sigma_px": round(float(np.median(widths)) / 2.5, 2),
    }


def gradient_orientations() -> dict:
    """Where a photograph's edges point, as a histogram.

    Architecture is built plumb and level, so its gradients should pile up at two
    orientations ninety degrees apart. Anything else in the frame -- foliage, cloth,
    people -- spreads out. The concentration is the measurement.
    """
    image = load("cloister.jpg")
    grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float64)
    gx = cv2.Sobel(grey, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(grey, cv2.CV_64F, 0, 1, ksize=3)
    magnitude = np.hypot(gx, gy)
    strong = magnitude > np.percentile(magnitude, 95)
    angles = (np.degrees(np.arctan2(gy[strong], gx[strong])) % 180).astype(int)
    histogram = np.bincount(angles, minlength=180).astype(float)
    histogram /= histogram.sum()

    peak = int(np.argmax(histogram))
    within10 = lambda centre: float(
        sum(histogram[(centre + d) % 180] for d in range(-10, 11))
    )
    return {
        "asset": "cloister.jpg",
        "strong_gradient_pixels": int(strong.sum()),
        "peak_orientation_deg": peak,
        "share_within_10deg_of_peak": round(within10(peak), 3),
        "share_within_10deg_of_perpendicular": round(within10((peak + 90) % 180), 3),
        "share_in_those_two_bands": round(within10(peak) + within10((peak + 90) % 180), 3),
        "uniform_expectation_for_two_bands": round(2 * 21 / 180, 3),
    }


def zero_crossing_counts() -> dict:
    """How many zero-crossings a second derivative finds, and how many are edges.

    On dense text the true answer is roughly "two per stroke". The Laplacian finds far
    more than that until it is smoothed, and the count is the argument for why the
    second derivative is fragile and why Canny is built on the first.
    """
    image = load("newspaper-1814.jpg")
    height, width = image.shape[:2]
    crop = cv2.cvtColor(
        image[height // 3 : height // 3 + 700, width // 4 : width // 4 + 700], cv2.COLOR_BGR2GRAY
    ).astype(np.float64)

    rows = []
    for sigma in (0.0, 1.0, 2.0, 4.0):
        blurred = crop if sigma == 0 else cv2.GaussianBlur(crop, (0, 0), sigma)
        laplacian = cv2.Laplacian(blurred, cv2.CV_64F, ksize=3)
        sign = np.sign(laplacian)
        crossings = int(((sign[:, :-1] * sign[:, 1:]) < 0).sum() + ((sign[:-1] * sign[1:]) < 0).sum())
        # A crossing that sits where the image is flat is noise, not an edge.
        magnitude = np.abs(cv2.Sobel(blurred, cv2.CV_64F, 1, 1, ksize=3))
        strong = magnitude > np.percentile(magnitude, 90)
        meaningful = int(((sign[:, :-1] * sign[:, 1:]) < 0)[:, :-0 or None][strong[:, :-1]].sum())
        rows.append({
            "sigma": sigma,
            "zero_crossings": crossings,
            "on_strong_gradient": meaningful,
            "fraction_on_strong_gradient": round(meaningful / max(crossings, 1), 3),
        })
    return {"asset": "newspaper-1814.jpg", "crop_px": [700, 700], "rows": rows}


def canny_hysteresis() -> dict:
    """What the second threshold buys, on a photograph rather than a chart.

    Same low threshold, same image; the only change is whether a high threshold seeds
    the contours. The number that matters is not how many pixels survive but how long
    the surviving contours are, because a broken edge is worth much less than a whole one.
    """
    image = load("cloister.jpg")
    grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    def contour_stats(edges):
        contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        lengths = sorted((len(c) for c in contours), reverse=True)
        return {
            "edge_pixels": int((edges > 0).sum()),
            "contours": len(contours),
            "longest_contour_px": lengths[0] if lengths else 0,
            "median_contour_px": int(np.median(lengths)) if lengths else 0,
            "contours_over_100px": int(sum(1 for v in lengths if v > 100)),
        }

    single_low = cv2.Canny(grey, 60, 60)
    single_high = cv2.Canny(grey, 180, 180)
    hysteresis = cv2.Canny(grey, 60, 180)
    return {
        "asset": "cloister.jpg",
        "single_threshold_60": contour_stats(single_low),
        "single_threshold_180": contour_stats(single_high),
        "hysteresis_60_180": contour_stats(hysteresis),
    }


def corner_repeatability() -> dict:
    """Do the same corners come back when the picture is rotated?

    A detector that finds a thousand points is worth nothing if they are different
    points every time. The image is rotated by a known angle, the corners are detected
    again, mapped back, and counted as repeats when they land within three pixels.
    """
    image = load("souk-fruit.jpg")
    grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    grey = cv2.resize(grey, (grey.shape[1] // 3, grey.shape[0] // 3), interpolation=cv2.INTER_AREA)
    height, width = grey.shape

    detect = lambda g: cv2.goodFeaturesToTrack(
        g, maxCorners=600, qualityLevel=0.01, minDistance=8, useHarrisDetector=True, k=0.04
    ).reshape(-1, 2)

    base = detect(grey)
    rows = []
    for angle in (5, 15, 45):
        matrix = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1.0)
        rotated = cv2.warpAffine(grey, matrix, (width, height), flags=cv2.INTER_LINEAR)
        found = detect(rotated)
        # Map the rotated detections back into the original frame.
        inverse = cv2.invertAffineTransform(matrix)
        back = (np.hstack([found, np.ones((len(found), 1))]) @ inverse.T)
        # A corner counts as repeated if some detection lands within three pixels, and
        # only corners far enough from the border to survive the rotation are counted.
        margin = 0.15 * min(height, width)
        inside = base[
            (base[:, 0] > margin) & (base[:, 0] < width - margin)
            & (base[:, 1] > margin) & (base[:, 1] < height - margin)
        ]
        distances = np.linalg.norm(inside[:, None, :] - back[None, :, :], axis=2)
        repeated = int((distances.min(axis=1) < 3).sum())
        rows.append({
            "rotation_deg": angle,
            "corners_in_original_interior": int(len(inside)),
            "corners_after_rotation": int(len(found)),
            "repeated_within_3px": repeated,
            "repeatability": round(repeated / max(len(inside), 1), 3),
        })
    return {
        "asset": "souk-fruit.jpg",
        "analysed_at_px": [height, width],
        "corners_detected": int(len(base)),
        "rows": rows,
    }
