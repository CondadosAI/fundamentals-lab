"""Held-out evidence: what a fitted image->court map gets wrong where it was not fitted.

A homography through four points reproduces those four exactly, so their residual
is zero by construction and proves nothing. Everything here is measured on paint
the fit never saw, and reported in centimetres on the court:

- ``NBC``, ``NKC``: where the near centre line meets the baseline and the kitchen line;
- the near centre line along its whole length, which must sit at y = W/2;
- the far part of the right sideline, beyond the net, which must sit at y = W - hw;
- the far baseline where it shows to the right of the net post, at x = L - hw.

The last two are the far field, where one pixel spans several centimetres. Paint
seen through the net mesh is not used (see ``config.NET_POLYGON``).
"""

from __future__ import annotations

import numpy as np

from fundamentals_lab.alignment import court, lines, transforms
from fundamentals_lab.config import COURT_KITCHEN_M as K
from fundamentals_lab.config import COURT_LENGTH_M as L
from fundamentals_lab.config import COURT_LINE_HALF_M as HW
from fundamentals_lab.config import COURT_WIDTH_M as W


def evidence_pixels(pixels: np.ndarray, H_court2img: np.ndarray) -> dict[str, np.ndarray]:
    """The paint pixels each held-out check is scored on, selected once from the fit."""
    out = {}
    for name in ("near_centre", "side_right", "far_base"):
        p, q = transforms.apply(H_court2img, np.array(court.LINES[name]))
        out[name], _ = lines.band(pixels, p, q)
    # Keep only the parts that belong to each check, judged in court coordinates.
    H_img2court = np.linalg.inv(H_court2img)
    far_side = transforms.apply(H_img2court, out["side_right"])
    out["side_right"] = out["side_right"][far_side[:, 0] > L / 2 + K]
    far_base = transforms.apply(H_img2court, out["far_base"])
    out["far_base"] = out["far_base"][far_base[:, 1] > 0.8 * W]
    return out


def held_out(H_img2court: np.ndarray, landmarks_img: dict, evidence: dict) -> dict:
    """Every held-out error, in cm. `landmarks_img` must hold NBC and NKC."""
    res = {}
    for name in ("NBC", "NKC"):
        est = transforms.apply(H_img2court, landmarks_img[name][None])[0]
        res[name] = float(np.linalg.norm(est - court.court_point(name)) * 100)
    c = transforms.apply(H_img2court, evidence["near_centre"])
    res["near_centre_line"] = float(np.median(np.abs(c[:, 1] - W / 2)) * 100)
    c = transforms.apply(H_img2court, evidence["side_right"])
    res["far_right_sideline"] = float(np.median(np.abs(c[:, 1] - (W - HW))) * 100)
    c = transforms.apply(H_img2court, evidence["far_base"])
    res["far_baseline"] = float(np.median(np.abs(c[:, 0] - (L - HW))) * 100)
    res["n_pixels"] = {k: int(len(v)) for k, v in evidence.items()}
    return res


def scale_cm_per_px(H_img2court: np.ndarray, point_img: np.ndarray) -> dict:
    """How many centimetres of court one pixel covers at an image point, per axis."""
    p = np.asarray(point_img, float)
    base = transforms.apply(H_img2court, p[None])[0]
    dx = transforms.apply(H_img2court, (p + [1, 0])[None])[0] - base
    dy = transforms.apply(H_img2court, (p + [0, 1])[None])[0] - base
    return {"x": float(np.linalg.norm(dx) * 100), "y": float(np.linalg.norm(dy) * 100)}
