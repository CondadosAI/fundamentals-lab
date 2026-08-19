"""Where the two ColorCheckers sit in the frame, patch by patch.

Fairchild metered 54 points and published their positions as numbered circles drawn
on a JPEG, which is readable by a person and useless to a script. What this module
does is turn four hand-read corners per chart (`config.CHART_CORNERS`) into all 24
patch rectangles, by the one fact that makes it safe: a ColorChecker is a lattice,
so a homography from grid coordinates to pixels places every patch from its corners.

The reading is checked rather than trusted — `sensing-charts` ranks the sampled
patches against the metered luminances, and a lattice that is off by a column or
flipped end for end fails that check immediately.
"""

from dataclasses import dataclass

import cv2
import numpy as np

from fundamentals_lab.config import CHART_CORNERS

COLUMNS, ROWS = 6, 4
GRID_CORNERS = np.float32([[0, 0], [COLUMNS - 1, 0], [COLUMNS - 1, ROWS - 1], [0, ROWS - 1]])


@dataclass(frozen=True)
class Patch:
    """One chart patch: its number, its centre, and the box we sample inside it."""

    chart: str
    number: int  # 1..24, row-major from the top-left patch, as the survey numbers them
    column: int
    row: int
    centre: tuple[float, float]
    box: tuple[int, int, int, int]  # x0, y0, x1, y1, half-open, in full-frame pixels

    @property
    def key(self) -> str:
        return f"{self.chart}:{self.number}"

    def sample(self, plane: np.ndarray) -> np.ndarray:
        """The pixels inside the patch, from any full-frame plane."""
        x0, y0, x1, y1 = self.box
        return plane[y0:y1, x0:x1]

    def sample_cfa(self, raw: np.ndarray, dy: int, dx: int) -> np.ndarray:
        """One colour-filter channel of the patch, by its offset in the 2x2 mosaic.

        A raw frame is still a mosaic, so averaging a rectangle of it averages red,
        green and blue photosites together. Every statistic in unit 1.2 is computed
        on one channel at a time for that reason.
        """
        x0, y0, x1, y1 = self.box
        y0 += (dy - y0) % 2
        x0 += (dx - x0) % 2
        return raw[y0:y1:2, x0:x1:2]


def patches(chart: str, inset: float = 0.3) -> list[Patch]:
    """All 24 patches of one chart.

    `inset` shrinks each sampling box away from the patch edge, where the chart's
    black gap, the neighbouring patch and demosaicing all bleed in. At 0.3 the box
    is the middle 40% of the patch in each direction.
    """
    if chart not in CHART_CORNERS:
        raise KeyError(f"unknown chart {chart!r} — expected one of {sorted(CHART_CORNERS)}")
    homography = cv2.getPerspectiveTransform(
        GRID_CORNERS, np.float32(CHART_CORNERS[chart])
    )
    half = 0.5 - inset
    found: list[Patch] = []
    for row in range(ROWS):
        for column in range(COLUMNS):
            cell = np.float32(
                [
                    [column - half, row - half],
                    [column + half, row - half],
                    [column + half, row + half],
                    [column - half, row + half],
                    [column, row],
                ]
            ).reshape(-1, 1, 2)
            mapped = cv2.perspectiveTransform(cell, homography).reshape(-1, 2)
            corners, centre = mapped[:4], mapped[4]
            x0, y0 = np.floor(corners.min(axis=0)).astype(int)
            x1, y1 = np.ceil(corners.max(axis=0)).astype(int)
            found.append(
                Patch(
                    chart=chart,
                    number=row * COLUMNS + column + 1,
                    column=column,
                    row=row,
                    centre=(float(centre[0]), float(centre[1])),
                    box=(int(x0), int(y0), int(x1), int(y1)),
                )
            )
    return found


def all_patches(inset: float = 0.3) -> list[Patch]:
    """Both charts, lit first."""
    return [patch for chart in CHART_CORNERS for patch in patches(chart, inset)]
