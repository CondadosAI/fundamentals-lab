"""Merge the bracket into one radiance map, then check it against a colorimeter.

A single exposure of this scene cannot hold it: the bulb clips at any exposure that
shows the shadowed chart, and the shadowed chart is buried at any exposure that keeps
the bulb. The eighteen frames each cover part of the range, and the merge is the
weighted average of what they agree on once each is divided by its own exposure time.

Because the frames are raw, and raw is linear in light, there is no response curve to
recover first — which is the fact lesson 3 spends its time on. Debevec and Malik's
method exists for the case where you only have JPEGs and the curve is unknown.

The check is what makes this a measurement: Fairchild metered 54 points in the room
with a Konica Minolta CS-100, so after fitting a single scale factor on one patch, the
other points are predictions with errors that can be reported in stops.
"""

from dataclasses import dataclass

import numpy as np

from fundamentals_lab.sensing.charts import Patch
from fundamentals_lab.sensing.noise import channel


@dataclass(frozen=True)
class Radiance:
    """A relative radiance map for one colour channel, in DN per second."""

    plane: np.ndarray
    channel: str
    frames: int
    covered_fraction: float  # pixels where at least one exposure was usable


def merge(
    frames: list[tuple[np.ndarray, float]],
    name: str = "G1",
    saturation: float | None = None,
    floor_dn: float = 24.0,
) -> Radiance:
    """Weighted average of every exposure, each divided by its own shutter time.

    The weight is a triangle over the usable part of the range: a count near the floor
    is mostly noise, a count near the ceiling is mostly a lie, and the middle of the
    range is where a photosite is telling the truth about the light.
    """
    if len(frames) < 2:
        raise ValueError("a merge of one exposure is just an exposure")
    planes = [(channel(raw, name).astype(np.float32), float(shutter)) for raw, shutter in frames]
    if saturation is None:
        saturation = max(float(plane.max()) for plane, _ in planes) * 0.98

    weight_sum = np.zeros_like(planes[0][0], dtype=np.float64)
    value_sum = np.zeros_like(weight_sum)
    middle = (floor_dn + saturation) / 2.0
    for plane, shutter in planes:
        usable = (plane >= floor_dn) & (plane < saturation)
        weight = np.where(usable, 1.0 - np.abs(plane - middle) / (middle - floor_dn), 0.0)
        np.clip(weight, 0.0, None, out=weight)
        weight_sum += weight
        value_sum += weight * (plane / shutter)

    covered = weight_sum > 0
    radiance = np.zeros_like(value_sum)
    radiance[covered] = value_sum[covered] / weight_sum[covered]
    return Radiance(
        plane=radiance,
        channel=name,
        frames=len(frames),
        covered_fraction=float(covered.mean()),
    )


@dataclass(frozen=True)
class Prediction:
    """One metered point, what the merge says about it, and the gap in stops."""

    key: str
    metered_cd_m2: float
    predicted_cd_m2: float

    @property
    def error_stops(self) -> float:
        return float(np.log2(self.predicted_cd_m2 / self.metered_cd_m2))


def sample(radiance: Radiance, patches: list[Patch]) -> dict[str, float]:
    """Median radiance inside each patch, in the merge's own units.

    The radiance plane is one colour channel, so it is half the width and half the
    height of the frame the patch boxes were measured in; the boxes are halved to
    match rather than the plane being resized back up.
    """
    values: dict[str, float] = {}
    for patch in patches:
        x0, y0, x1, y1 = patch.box
        block = radiance.plane[y0 // 2 : y1 // 2, x0 // 2 : x1 // 2]
        positive = block[block > 0]
        if positive.size:
            values[patch.key] = float(np.median(positive))
    return values


def calibrate(
    values: dict[str, float], metered: dict[str, float], anchor: str
) -> tuple[float, list[Prediction]]:
    """Fit one scale factor on `anchor`, then predict every other metered point.

    One degree of freedom, fitted on one patch, is the whole calibration: a camera
    measures ratios of light, and it takes an instrument to say what one of those
    ratios is worth in cd/m². Fairchild's own pipeline does the same thing with a
    single normalisation factor, and this keeps our check honest — every point except
    the anchor is a prediction, not a fit.
    """
    if anchor not in values or anchor not in metered:
        raise KeyError(f"anchor {anchor!r} missing from the merge or from the measurements")
    scale = metered[anchor] / values[anchor]
    predictions = [
        Prediction(key=key, metered_cd_m2=metered[key], predicted_cd_m2=value * scale)
        for key, value in sorted(values.items())
        if key in metered and metered[key] > 0
    ]
    return scale, predictions


def luminance(
    radiances: dict[str, Radiance],
    patches: list[Patch],
    white_balance: tuple[float, float, float],
    matrix_row: tuple[float, float, float],
) -> dict[str, float]:
    """Relative luminance per patch, from all four mosaic channels.

    One channel is enough to compare neutral patches, and wrong for coloured ones: a
    green photosite under a red patch reports what green light the patch reflected, not
    what a colorimeter would call its luminance. Combining the channels with the
    camera's own white balance and the published luminance row fixes that, and it is
    the one place where this unit borrows the survey's characterisation rather than
    measuring for itself.
    """
    red, green, blue = white_balance
    per_channel = {name: sample(radiance, patches) for name, radiance in radiances.items()}
    keys = set.intersection(*(set(values) for values in per_channel.values()))
    result: dict[str, float] = {}
    for key in keys:
        r = per_channel["R"][key] * red
        g = 0.5 * (per_channel["G1"][key] + per_channel["G2"][key]) * green
        b = per_channel["B"][key] * blue
        result[key] = matrix_row[0] * r + matrix_row[1] * g + matrix_row[2] * b
    return result
