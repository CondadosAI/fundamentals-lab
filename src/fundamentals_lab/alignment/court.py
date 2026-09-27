"""The court as a set of lines on a plane, in metres.

Court frame: origin at the outside corner where the near baseline meets the left
sideline (the corner that falls off the left edge of the frame), x along the court
away from the camera, y across it::

    (0, W) +---------------------------+ (L, W)
           |      |  near  |  |  far   |
           |      |kitchen |  |kitchen |      x = L/2 - K : near kitchen line
           |------+ - - - -|net - - - -|      x = L/2     : the net
           |      |        |  |        |      x = L/2 + K : far kitchen line
    (0, 0) +---------------------------+ (L, 0)
       near baseline                  far baseline

Every coordinate here is a *line centre*: 3.A.2 measures the court to the outside
edge of its lines, and a fit to paint pixels finds the middle of the stripe.
"""

from __future__ import annotations

import numpy as np

from fundamentals_lab.config import (
    COURT_KITCHEN_M as K,
)
from fundamentals_lab.config import (
    COURT_LENGTH_M as L,
)
from fundamentals_lab.config import (
    COURT_LINE_HALF_M as HW,
)
from fundamentals_lab.config import (
    COURT_WIDTH_M as W,
)

MID = L / 2

#: Each painted line as two court points on its centre line.
LINES: dict[str, tuple[tuple[float, float], tuple[float, float]]] = {
    "near_base": ((HW, 0.0), (HW, W)),
    "far_base": ((L - HW, 0.0), (L - HW, W)),
    "side_left": ((0.0, HW), (L, HW)),
    "side_right": ((0.0, W - HW), (L, W - HW)),
    "near_kitchen": ((MID - K + HW, 0.0), (MID - K + HW, W)),
    "far_kitchen": ((MID + K - HW, 0.0), (MID + K - HW, W)),
    # 3.A.4.d: the centre line runs from the non-volley line to the baseline.
    "near_centre": ((0.0, W / 2), (MID - K, W / 2)),
    "far_centre": ((MID + K, W / 2), (L, W / 2)),
}

#: The lines a camera behind the near corner sees cleanly. The far lines lie
#: behind the net, and `lines.fit_line` refuses the pixels there.
NEAR_LINES = ("near_base", "side_left", "side_right", "near_kitchen", "near_centre")

#: Twelve named intersections. Short names: N/F = near/far half, B/K = baseline /
#: kitchen line, L/R/C = left sideline, right sideline, centre line.
LANDMARKS: dict[str, tuple[str, str]] = {
    "NBL": ("near_base", "side_left"),
    "NBR": ("near_base", "side_right"),
    "NKL": ("near_kitchen", "side_left"),
    "NKR": ("near_kitchen", "side_right"),
    "NBC": ("near_base", "near_centre"),
    "NKC": ("near_kitchen", "near_centre"),
    "FBL": ("far_base", "side_left"),
    "FBR": ("far_base", "side_right"),
    "FKL": ("far_kitchen", "side_left"),
    "FKR": ("far_kitchen", "side_right"),
    "FBC": ("far_base", "far_centre"),
    "FKC": ("far_kitchen", "far_centre"),
}

#: The four that define the fit in every lesson: the corners of the near half.
FIT_LANDMARKS = ("NBL", "NBR", "NKL", "NKR")


def homogeneous_line(p, q) -> np.ndarray:
    """The line through two points, as a 3-vector: their cross product."""
    return np.cross([p[0], p[1], 1.0], [q[0], q[1], 1.0])


def meet(l1: np.ndarray, l2: np.ndarray) -> np.ndarray:
    """Where two lines cross, as a homogeneous point (w = 0 if they are parallel)."""
    return np.cross(l1, l2)


def dehomogenise(x: np.ndarray) -> np.ndarray:
    return x[:2] / x[2]


def court_point(name: str) -> np.ndarray:
    """A landmark's position on the court, in metres."""
    a, b = LANDMARKS[name]
    return dehomogenise(meet(homogeneous_line(*LINES[a]), homogeneous_line(*LINES[b])))


def court_points(names) -> np.ndarray:
    return np.array([court_point(n) for n in names])
