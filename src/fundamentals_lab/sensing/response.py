"""The curve between the light that arrived and the number you finally see.

A raw count is proportional to the light — lesson 1 measures that, and lesson 4 relies
on it. The JPEG on your disk is not: somewhere in the camera, or in the raw converter,
a tone curve lifted the shadows and rolled off the highlights so the picture would look
right on a display. Everything downstream that assumes brightness is proportional to
light — HDR merging, photometric stereo, brightness constancy in optical flow — is
wrong on that file by exactly this curve.

Here it is measured on the neutral row of the lit chart: linear values from the raw
frame against the code values a standard development produced from the same pixels.
"""

from dataclasses import dataclass

import numpy as np

from fundamentals_lab.sensing.charts import Patch
from fundamentals_lab.sensing.noise import CFA_OFFSETS


@dataclass(frozen=True)
class ResponsePoint:
    """One patch, seen twice: as raw counts and as an 8-bit code value."""

    key: str
    linear_dn: float
    code_value: float

    def relative(self, white_dn: float) -> float:
        return self.linear_dn / white_dn


@dataclass(frozen=True)
class GammaFit:
    """A single exponent through the response points, with its residual.

    `scale` is fitted alongside the exponent because a developed image does not put
    the brightest patch at code 255 — it puts the sensor's saturation there, and the
    chart's white sits wherever the exposure left it.
    """

    gamma: float
    scale: float
    stderr: float
    r_squared: float
    points: int

    def predict(self, relative: np.ndarray) -> np.ndarray:
        return 255.0 * self.scale * np.clip(relative, 1e-9, 1.0) ** (1.0 / self.gamma)


def srgb_encode(relative: np.ndarray) -> np.ndarray:
    """The standard sRGB transfer function, for comparison with what the camera did."""
    relative = np.clip(relative, 0.0, 1.0)
    low = relative * 12.92
    high = 1.055 * np.power(relative, 1 / 2.4) - 0.055
    return 255.0 * np.where(relative <= 0.0031308, low, high)


def response_points(
    raw: np.ndarray,
    developed: np.ndarray,
    patches: list[Patch],
    black_level: float = 0.0,
    name: str = "G1",
) -> list[ResponsePoint]:
    """Pair each patch's raw counts with its code value in the developed image."""
    dy, dx = CFA_OFFSETS[name]
    points: list[ResponsePoint] = []
    for patch in patches:
        linear = float(np.median(patch.sample_cfa(raw, dy, dx).astype(np.float64))) - black_level
        code = float(np.median(patch.sample(developed).astype(np.float64)))
        points.append(ResponsePoint(patch.key, linear, code))
    return points


def fit_gamma(points: list[ResponsePoint], white_dn: float) -> GammaFit:
    """Fit code = 255 * scale * relative**(1/gamma) by least squares in log space.

    `white_dn` is the channel's saturation level, not the white patch: relative light
    means relative to what the sensor can hold, which is what the raw converter's own
    curve is defined against.

    A single exponent is an approximation and the residual says how rough it is:
    manufacturers stack an S-shaped contrast curve on top of the gamma, which is
    exactly what the departures at the two ends of the ramp are.
    """
    usable = [p for p in points if p.linear_dn > 0 and 1.0 < p.code_value < 254.0]
    if len(usable) < 3:
        raise ValueError(f"only {len(usable)} usable points — the exposure is wrong for this fit")
    relative = np.array([p.linear_dn / white_dn for p in usable])
    code = np.array([p.code_value / 255.0 for p in usable])
    x = np.log(relative)
    y = np.log(code)
    slope, intercept = np.polyfit(x, y, 1)
    predicted = slope * x + intercept
    residual = float(((y - predicted) ** 2).sum())
    total = float(((y - y.mean()) ** 2).sum())
    stderr = float(np.sqrt(residual / max(len(x) - 2, 1) / ((x - x.mean()) ** 2).sum()))
    return GammaFit(
        gamma=float(1.0 / slope),
        scale=float(np.exp(intercept)),
        stderr=float(stderr / slope**2),
        r_squared=float(1 - residual / total),
        points=len(usable),
    )
