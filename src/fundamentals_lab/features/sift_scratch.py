"""SIFT from scratch, stage by stage, following OpenCV's `sift.dispatch.cpp`.

OpenCV exposes only the end of the pipeline (keypoints and descriptors), so lesson 3's
per-stage counts and lesson 4's ablations need every intermediate step in the open:
the Gaussian and difference-of-Gaussians pyramids, the 26-neighbour extrema, the
sub-pixel fit, the contrast and edge tests, the orientation histogram and the 4x4x8
descriptor. Constants and conventions are OpenCV's (image on a 0-255 scale, the base
image doubled, three layers per octave), so the result can be compared with
`cv2.SIFT_create()` keypoint for keypoint.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

SIGMA = 1.6
INIT_SIGMA = 0.5
LAYERS = 3  # nOctaveLayers
CONTRAST = 0.04  # contrastThreshold
EDGE = 10.0  # edgeThreshold
BORDER = 5  # SIFT_IMG_BORDER
MAX_STEPS = 5  # SIFT_MAX_INTERP_STEPS
ORI_BINS = 36
ORI_PEAK = 0.8
ORI_SIG = 1.5
ORI_RADIUS = 3 * ORI_SIG
DESC_D = 4
DESC_N = 8
DESC_SCL = 3.0
DESC_CLIP = 0.2


@dataclass
class Extremum:
    """One keypoint as the pipeline carries it, in the coordinates of its octave."""

    octave: int  # pyramid index; image coordinates are scaled by 2**(octave - 1)
    layer: int
    r: int
    c: int
    xi: float = 0.0
    xr: float = 0.0
    xc: float = 0.0
    response: float = 0.0
    angle: float = 0.0  # OpenCV's convention: 360 - gradient direction, degrees

    @property
    def scale(self) -> float:
        """Blur of the keypoint's level, in octave pixels."""
        return SIGMA * 2 ** ((self.layer + self.xi) / LAYERS)

    @property
    def pt(self) -> tuple[float, float]:
        f = 2.0 ** (self.octave - 1)
        return ((self.c + self.xc) * f, (self.r + self.xr) * f)

    @property
    def size(self) -> float:
        """OpenCV's `KeyPoint.size`: the diameter, in image pixels."""
        return self.scale * 2.0 ** (self.octave - 1) * 2

    def keypoint(self) -> cv2.KeyPoint:
        # OpenCV packs octave, layer and the sub-layer offset into one int, and reads them
        # back in compute(); without it the descriptor is taken from the wrong level.
        packed = ((self.octave - 1) & 255) | (self.layer << 8)
        packed |= int(round((self.xi + 0.5) * 255)) << 16
        return cv2.KeyPoint(*self.pt, self.size, self.angle, abs(self.response), packed)


@dataclass
class Pyramid:
    gauss: list[list[np.ndarray]]  # [octave][level], 6 levels per octave
    dog: list[list[np.ndarray]]  # [octave][level], 5 levels per octave
    sigmas: list[float] = field(default_factory=list)  # incremental blur per level


def n_octaves(shape: tuple[int, int]) -> int:
    """OpenCV's count: round(log2(min side of the doubled image) - 2) + 1."""
    return int(round(np.log2(min(shape[0], shape[1]) * 2) - 2)) + 1


def build_pyramid(gray: np.ndarray) -> Pyramid:
    img = gray.astype(np.float32)
    base = cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)
    diff = np.sqrt(max(SIGMA**2 - (2 * INIT_SIGMA) ** 2, 0.01))
    base = cv2.GaussianBlur(base, (0, 0), diff, diff)
    k = 2 ** (1 / LAYERS)
    sig = [SIGMA]
    for i in range(1, LAYERS + 3):
        prev = SIGMA * k ** (i - 1)
        sig.append(np.sqrt((prev * k) ** 2 - prev**2))
    gauss = []
    for o in range(n_octaves(gray.shape)):
        levels = []
        for i in range(LAYERS + 3):
            if o == 0 and i == 0:
                levels.append(base)
            elif i == 0:
                levels.append(gauss[o - 1][LAYERS][::2, ::2].copy())
            else:
                levels.append(cv2.GaussianBlur(levels[i - 1], (0, 0), sig[i], sig[i]))
        gauss.append(levels)
    dog = [[g[i + 1] - g[i] for i in range(LAYERS + 2)] for g in gauss]
    return Pyramid(gauss, dog, sig)


def raw_extrema(pyr: Pyramid) -> list[Extremum]:
    """Points that beat (or tie) all 26 neighbours in (x, y, scale), above a small floor."""
    floor = np.floor(0.5 * CONTRAST / LAYERS * 255)
    out = []
    for o, dogs in enumerate(pyr.dog):
        for i in range(1, LAYERS + 1):
            cube = np.stack(dogs[i - 1 : i + 2])
            mx = cv2.dilate(np.max(cube, axis=0), np.ones((3, 3), np.uint8))
            mn = cv2.erode(np.min(cube, axis=0), np.ones((3, 3), np.uint8))
            v = dogs[i]
            hit = ((v > floor) & (v >= mx)) | ((v < -floor) & (v <= mn))
            hit[:BORDER] = hit[-BORDER:] = False
            hit[:, :BORDER] = hit[:, -BORDER:] = False
            for r, c in zip(*np.nonzero(hit), strict=True):
                out.append(Extremum(o, i, int(r), int(c)))
    return out


def _derivatives(dogs, i, r, c):
    s = 1.0 / 255
    prv, cur, nxt = dogs[i - 1], dogs[i], dogs[i + 1]
    g = (
        0.5
        * s
        * np.array(
            [cur[r, c + 1] - cur[r, c - 1], cur[r + 1, c] - cur[r - 1, c], nxt[r, c] - prv[r, c]]
        )
    )
    v2 = 2 * cur[r, c]
    dxx = (cur[r, c + 1] + cur[r, c - 1] - v2) * s
    dyy = (cur[r + 1, c] + cur[r - 1, c] - v2) * s
    dss = (nxt[r, c] + prv[r, c] - v2) * s
    q = 0.25 * s
    dxy = (cur[r + 1, c + 1] - cur[r + 1, c - 1] - cur[r - 1, c + 1] + cur[r - 1, c - 1]) * q
    dxs = (nxt[r, c + 1] - nxt[r, c - 1] - prv[r, c + 1] + prv[r, c - 1]) * q
    dys = (nxt[r + 1, c] - nxt[r - 1, c] - prv[r + 1, c] + prv[r - 1, c]) * q
    H = np.array([[dxx, dxy, dxs], [dxy, dyy, dys], [dxs, dys, dss]], dtype=np.float64)
    return g.astype(np.float64), H


def refine(pyr: Pyramid, e: Extremum) -> tuple[str, Extremum]:
    """Fit a quadratic around the extremum and move it there.

    Returns ("kept", e) or the stage that dropped it: "drift" (the fit never settled
    inside the octave), "contrast" (|D| at the refined point too small) or "edge" (the
    principal curvatures too unequal: a ridge, not a blob).
    """
    dogs = pyr.dog[e.octave]
    rows, cols = dogs[0].shape
    i, r, c = e.layer, e.r, e.c
    for _ in range(MAX_STEPS):
        g, H = _derivatives(dogs, i, r, c)
        try:
            X = -np.linalg.solve(H, g)
        except np.linalg.LinAlgError:
            return "drift", e
        xc, xr, xi = X
        if abs(xi) < 0.5 and abs(xr) < 0.5 and abs(xc) < 0.5:
            break
        if max(abs(xi), abs(xr), abs(xc)) > 1e6:
            return "drift", e
        c += int(round(xc))
        r += int(round(xr))
        i += int(round(xi))
        if (
            i < 1
            or i > LAYERS
            or c < BORDER
            or c >= cols - BORDER
            or r < BORDER
            or r >= rows - BORDER
        ):
            return "drift", e
    else:
        return "drift", e
    g, H = _derivatives(dogs, i, r, c)
    contr = dogs[i][r, c] / 255 + 0.5 * float(g @ np.array([xc, xr, xi]))
    out = Extremum(e.octave, i, r, c, float(xi), float(xr), float(xc), float(contr))
    if abs(contr) * LAYERS < CONTRAST:
        return "contrast", out
    tr = H[0, 0] + H[1, 1]
    det = H[0, 0] * H[1, 1] - H[0, 1] ** 2
    if det <= 0 or tr * tr * EDGE >= (EDGE + 1) ** 2 * det:
        return "edge", out
    return "kept", out


def orientation_histogram(img: np.ndarray, r: int, c: int, scale: float) -> np.ndarray:
    """36-bin histogram of gradient directions around (r, c), Gaussian- and magnitude-weighted,
    then smoothed with OpenCV's (1 4 6 4 1)/16 kernel."""
    radius = int(round(ORI_RADIUS * scale))
    sigma = ORI_SIG * scale
    rows, cols = img.shape
    y0, y1 = max(r - radius, 1), min(r + radius, rows - 2)
    x0, x1 = max(c - radius, 1), min(c + radius, cols - 2)
    ys, xs = np.mgrid[y0 : y1 + 1, x0 : x1 + 1]
    dx = img[ys, xs + 1] - img[ys, xs - 1]
    dy = img[ys - 1, xs] - img[ys + 1, xs]
    w = np.exp(-((ys - r) ** 2 + (xs - c) ** 2) / (2 * sigma**2))
    ori = np.degrees(np.arctan2(dy, dx)) % 360
    b = np.round(ori * ORI_BINS / 360).astype(int) % ORI_BINS
    raw = np.bincount(b.ravel(), (w * np.hypot(dx, dy)).ravel(), ORI_BINS)
    return (
        (np.roll(raw, 2) + np.roll(raw, -2)) / 16
        + (np.roll(raw, 1) + np.roll(raw, -1)) * 4 / 16
        + raw * 6 / 16
    )


def orientations(hist: np.ndarray, peak: float = ORI_PEAK) -> list[float]:
    """Every local peak within `peak` of the tallest, parabola-interpolated, as OpenCV angles."""
    n = len(hist)
    thr = hist.max() * peak
    out = []
    for j in range(n):
        lft, rgt = hist[(j - 1) % n], hist[(j + 1) % n]
        if hist[j] > lft and hist[j] > rgt and hist[j] >= thr:
            b = j + 0.5 * (lft - rgt) / (lft - 2 * hist[j] + rgt)
            b = b % n
            a = 360.0 - 360.0 / n * b
            out.append(0.0 if abs(a - 360.0) < 1e-6 else a)
    return out


def descriptor(
    img: np.ndarray,
    x: float,
    y: float,
    angle: float,
    scale: float,
    *,
    normalize: bool = True,
    clip: bool = True,
) -> np.ndarray:
    """The 4x4x8 gradient histogram around (x, y) in octave pixels, rotated by `angle`.

    `normalize=False` returns the raw histogram; `clip=False` skips the 0.2 clamp.
    """
    d, n = DESC_D, DESC_N
    ori = 360.0 - angle if abs(360.0 - angle) > 1e-6 else 0.0
    cos_t, sin_t = np.cos(np.radians(ori)), np.sin(np.radians(ori))
    hist_w = DESC_SCL * scale
    radius = int(round(hist_w * np.sqrt(2) * (d + 1) * 0.5))
    rows, cols = img.shape
    radius = min(radius, int(np.hypot(rows, cols)))
    cos_t, sin_t = cos_t / hist_w, sin_t / hist_w
    pr, pc = int(round(y)), int(round(x))
    ii, jj = np.mgrid[-radius : radius + 1, -radius : radius + 1]
    c_rot = jj * cos_t - ii * sin_t
    r_rot = jj * sin_t + ii * cos_t
    rbin = r_rot + d / 2 - 0.5
    cbin = c_rot + d / 2 - 0.5
    rr, cc = pr + ii, pc + jj
    ok = (rbin > -1) & (rbin < d) & (cbin > -1) & (cbin < d)
    ok &= (rr > 0) & (rr < rows - 1) & (cc > 0) & (cc < cols - 1)
    rr, cc, rbin, cbin = rr[ok], cc[ok], rbin[ok], cbin[ok]
    c_rot, r_rot = c_rot[ok], r_rot[ok]
    dx = img[rr, cc + 1] - img[rr, cc - 1]
    dy = img[rr - 1, cc] - img[rr + 1, cc]
    w = np.exp(-(c_rot**2 + r_rot**2) / (d * d * 0.5))
    mag = np.hypot(dx, dy) * w
    obin = ((np.degrees(np.arctan2(dy, dx)) % 360) - ori) * n / 360
    hist = np.zeros((d + 2, d + 2, n + 2))
    r0, c0, o0 = np.floor(rbin).astype(int), np.floor(cbin).astype(int), np.floor(obin).astype(int)
    fr, fc, fo = rbin - r0, cbin - c0, obin - o0
    o0 = np.where(o0 < 0, o0 + n, o0)
    o0 = np.where(o0 >= n, o0 - n, o0)
    for dr, wr in ((0, 1 - fr), (1, fr)):
        for dc, wc in ((0, 1 - fc), (1, fc)):
            for do, wo in ((0, 1 - fo), (1, fo)):
                np.add.at(hist, (r0 + 1 + dr, c0 + 1 + dc, o0 + do), mag * wr * wc * wo)
    hist[:, :, 0] += hist[:, :, n]
    hist[:, :, 1] += hist[:, :, n + 1]
    v = hist[1 : d + 1, 1 : d + 1, :n].reshape(-1).astype(np.float64)
    if not normalize:
        return v.astype(np.float32)
    if clip:
        v = np.minimum(v, np.linalg.norm(v) * DESC_CLIP)
    v = v * 512.0 / max(np.linalg.norm(v), 1e-7)
    return np.clip(np.round(v), 0, 255).astype(np.float32)


@dataclass
class Run:
    """Everything one from-scratch pass produced, with the stage each extremum died at."""

    pyramid: Pyramid
    extrema: int
    dropped: dict[str, int]
    kept: list[Extremum]  # one entry per orientation, as OpenCV returns them

    def keypoints(self) -> list[cv2.KeyPoint]:
        return [e.keypoint() for e in self.kept]

    def descriptors(self, **kw) -> np.ndarray:
        return np.array(
            [
                descriptor(
                    self.pyramid.gauss[e.octave][e.layer],
                    e.c + e.xc,
                    e.r + e.xr,
                    e.angle,
                    e.scale,  # OpenCV passes KeyPoint.size * 0.5, in octave pixels
                    **kw,
                )
                for e in self.kept
            ]
        )


def detect(gray: np.ndarray) -> Run:
    """The whole detector: pyramid, extrema, refinement, tests, orientations."""
    pyr = build_pyramid(gray)
    ext = raw_extrema(pyr)
    dropped = {"drift": 0, "contrast": 0, "edge": 0}
    kept: list[Extremum] = []
    for e in ext:
        stage, f = refine(pyr, e)
        if stage != "kept":
            dropped[stage] += 1
            continue
        hist = orientation_histogram(pyr.gauss[f.octave][f.layer], f.r, f.c, f.scale)
        for a in orientations(hist):
            kept.append(Extremum(f.octave, f.layer, f.r, f.c, f.xi, f.xr, f.xc, f.response, a))
    return Run(pyr, len(ext), dropped, kept)
