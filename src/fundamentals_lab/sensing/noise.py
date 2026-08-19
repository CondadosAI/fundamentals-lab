"""How much of a raw count is signal and how much is noise — measured, not modelled.

The measurement is the photon transfer curve: for many places in the frame, plot the
variance of the count against its mean. Shot noise rises in proportion to the signal,
so the slope of that line is the sensor's gain in DN per electron; read noise does not
depend on the signal, so it would be the intercept.

Two things let this scene carry it. The bracket ends with **three frames at the same
30 s exposure** — the camera hit its shutter limit and Fairchild kept shooting — which
gives variance *in time*, the only kind that separates noise from the scene's own
texture. And the shadowed ColorChecker puts twenty-four uniform patches of known
luminance inside those frames.

Variance is estimated from the median absolute deviation of frame differences rather
than from the sample variance. Three frames give a variance with two degrees of
freedom, whose sample median sits about 30% below the true value, and whose sample
mean is dragged upwards by every pixel where the frames differ for a reason that is
not noise. The MAD estimator is robust to the second and carries an exact Gaussian
correction for the first; `estimator_cross_check` reports the other one beside it.
"""

from dataclasses import dataclass

import numpy as np

from fundamentals_lab.sensing.charts import Patch

# The Bayer offsets of an RGGB mosaic: name -> (row, column) inside the 2x2 cell.
CFA_OFFSETS = {"R": (0, 0), "G1": (0, 1), "G2": (1, 0), "B": (1, 1)}

# A channel counts as clipped this far below its own ceiling. The ceiling is measured
# per channel rather than read off the file, because these four channels do not stop
# at the same number: red and blue pile up at 4095 while the greens top out near 3875.
# That is the analog white-balance gain applied before the converter, and it makes
# "white level = 4095" true of the file and false of the green photosites.
CLIP_MARGIN = 0.98

# Half-normal median: for Gaussian noise, median(|x|) = 0.6745 sigma.
_MAD_TO_SIGMA = 1.0 / 0.6745
# Chi-squared with two degrees of freedom: median = 0.6931 * mean.
_CHI2_MEDIAN_BIAS = 0.6931


def channel(raw: np.ndarray, name: str = "G1") -> np.ndarray:
    """One colour channel of a Bayer frame, as its own half-resolution plane."""
    dy, dx = CFA_OFFSETS[name]
    return raw[dy::2, dx::2]


def clip_level(raw: np.ndarray, name: str = "G1") -> float:
    """Where this channel actually stops, measured on the frame that saturates most."""
    return float(channel(raw, name).max()) * CLIP_MARGIN


@dataclass(frozen=True)
class TransferPoint:
    """One point of the photon transfer curve, pooled over many pixels."""

    label: str
    mean_dn: float
    variance_dn2: float  # from the MAD of frame differences
    variance_cross_check_dn2: float  # from the median sample variance, bias-corrected
    pixels: int


@dataclass(frozen=True)
class TransferFit:
    """A straight line through the photon transfer curve, and what it implies.

    The intercept is where read noise would be, and on this scene it is not a
    measurement: the repeated frames are 30 s long, so the darkest signal they hold is
    a few hundred DN and the intercept sits far outside the data. It is kept with its
    standard error so a lesson can report the bound instead of inventing a number.
    """

    gain_dn_per_electron: float
    gain_stderr: float
    intercept_dn2: float
    intercept_stderr: float
    r_squared: float
    points: int
    signal_range_dn: tuple[float, float]

    @property
    def electrons_per_dn(self) -> float:
        return 1.0 / self.gain_dn_per_electron

    def full_well(self, ceiling_dn: float) -> float:
        return ceiling_dn / self.gain_dn_per_electron

    def electrons(self, noise_dn: float) -> float:
        return noise_dn / self.gain_dn_per_electron

    def dynamic_range_stops(self, ceiling_dn: float, noise_dn: float) -> float:
        """Stops between saturation and a noise floor the caller has to supply.

        The floor is an argument rather than this fit's intercept on purpose: the
        honest floor here is a bound, and the caller must say which bound it used.
        """
        return float(np.log2(ceiling_dn / noise_dn))


def _variance(samples: np.ndarray, selected: np.ndarray) -> tuple[float, float]:
    """Noise variance over the selected pixels, by both estimators."""
    differences = np.concatenate(
        [
            (samples[i] - samples[j])[selected]
            for i in range(len(samples))
            for j in range(i + 1, len(samples))
        ]
    )
    sigma_difference = float(np.median(np.abs(differences))) * _MAD_TO_SIGMA
    robust = sigma_difference**2 / 2.0
    sample_variance = samples.var(axis=0, ddof=1)[selected]
    return robust, float(np.median(sample_variance)) / _CHI2_MEDIAN_BIAS


def transfer_points(
    repeats: list[np.ndarray],
    name: str = "G1",
    saturation: float | None = None,
    bins: int = 28,
    min_pixels: int = 2000,
) -> list[TransferPoint]:
    """The transfer curve read off the whole frame, one point per signal level.

    The charts give twenty-four labelled points over two decades; the frame itself
    gives every level the scene happens to contain, including the dark corners no chart
    reaches. Both are the same measurement, and the lesson plots them together.
    """
    if len(repeats) < 3:
        raise ValueError("a variance from two frames is not worth publishing")
    planes = np.stack([channel(frame, name).astype(np.float64) for frame in repeats])
    if saturation is None:
        saturation = float(planes.max()) * CLIP_MARGIN
    mean = planes.mean(axis=0)
    usable = (planes.max(axis=0) < saturation) & (mean > 1)
    if usable.sum() < min_pixels * 2:
        raise ValueError("almost everything is clipped — wrong frames for this channel")
    edges = np.geomspace(float(mean[usable].min()), float(mean[usable].max()), bins + 1)
    points: list[TransferPoint] = []
    for index, (start, stop) in enumerate(zip(edges[:-1], edges[1:], strict=True)):
        selected = usable & (mean >= start) & (mean < stop)
        if selected.sum() < min_pixels:
            continue
        robust, cross_check = _variance(planes, selected)
        points.append(
            TransferPoint(
                label=f"bin{index:02d}",
                mean_dn=float(np.median(mean[selected])),
                variance_dn2=robust,
                variance_cross_check_dn2=cross_check,
                pixels=int(selected.sum()),
            )
        )
    return points


def patch_points(
    repeats: list[np.ndarray],
    patches: list[Patch],
    name: str = "G1",
    saturation: float | None = None,
) -> list[TransferPoint]:
    """The same measurement, one point per ColorChecker patch, so points carry names."""
    planes = [channel(frame, name) for frame in repeats]
    if saturation is None:
        saturation = float(np.stack(planes).max()) * CLIP_MARGIN
    dy, dx = CFA_OFFSETS[name]
    points: list[TransferPoint] = []
    for patch in patches:
        samples = np.stack(
            [patch.sample_cfa(frame, dy, dx).astype(np.float64) for frame in repeats]
        )
        selected = samples.max(axis=0) < saturation
        if selected.mean() < 0.99:  # any clipping at all disqualifies a patch
            continue
        robust, cross_check = _variance(samples, selected)
        points.append(
            TransferPoint(
                label=patch.key,
                mean_dn=float(np.median(samples.mean(axis=0)[selected])),
                variance_dn2=robust,
                variance_cross_check_dn2=cross_check,
                pixels=int(selected.sum()),
            )
        )
    return points


def fit(points: list[TransferPoint], cross_check: bool = False) -> TransferFit:
    """Least squares through variance against mean."""
    if len(points) < 5:
        raise ValueError(f"only {len(points)} usable points — nothing worth fitting")
    means = np.array([p.mean_dn for p in points])
    variances = np.array(
        [p.variance_cross_check_dn2 if cross_check else p.variance_dn2 for p in points]
    )
    design = np.vstack([means, np.ones_like(means)]).T
    (slope, intercept), *_ = np.linalg.lstsq(design, variances, rcond=None)
    if slope <= 0:
        raise ValueError("variance falls with signal — wrong frames or wrong channel")
    predicted = design @ [slope, intercept]
    residual = float(((variances - predicted) ** 2).sum())
    total = float(((variances - variances.mean()) ** 2).sum())
    covariance = (residual / max(len(points) - 2, 1)) * np.linalg.inv(design.T @ design)
    return TransferFit(
        gain_dn_per_electron=float(slope),
        gain_stderr=float(np.sqrt(covariance[0, 0])),
        intercept_dn2=float(intercept),
        intercept_stderr=float(np.sqrt(covariance[1, 1])),
        r_squared=float(1 - residual / total),
        points=len(points),
        signal_range_dn=(float(means.min()), float(means.max())),
    )


def dark_noise_bound(
    frame: np.ndarray, region: tuple[int, int, int, int], name: str = "G1"
) -> float:
    """Spatial standard deviation of a dark corner of the shortest exposure, in DN.

    An upper bound on read noise, not read noise: with a single frame there is no way
    to separate the electronics from fixed pattern, dark current, or whatever faint
    light reached that corner. The bound is worth having because the transfer fit
    cannot reach this far down.
    """
    x0, y0, x1, y1 = region
    plane = channel(frame[y0:y1, x0:x1], name).astype(np.float64)
    return float(plane.std(ddof=1))
