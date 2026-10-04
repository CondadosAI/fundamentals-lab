"""Circles on the board: the two transducers, and the four mounting holes it half-finds."""

from __future__ import annotations

import cv2
import numpy as np

from fundamentals_lab.config import HOLE_CIRCLES, MOUNTING_HOLES, TRANSDUCER_CIRCLES


def transducers(blur: np.ndarray) -> np.ndarray | None:
    p = TRANSDUCER_CIRCLES
    c = cv2.HoughCircles(
        blur,
        cv2.HOUGH_GRADIENT,
        1,
        p["min_dist"],
        param1=p["param1"],
        param2=p["param2"],
        minRadius=p["r"][0],
        maxRadius=p["r"][1],
    )
    return None if c is None else c[0][np.argsort(c[0][:, 0])]


def holes(blur: np.ndarray, param2: int) -> dict:
    p = HOLE_CIRCLES
    c = cv2.HoughCircles(
        blur,
        cv2.HOUGH_GRADIENT,
        1,
        p["min_dist"],
        param1=p["param1"],
        param2=param2,
        minRadius=p["r"][0],
        maxRadius=p["r"][1],
    )
    c = np.empty((0, 3)) if c is None else c[0]
    truth = np.array(MOUNTING_HOLES, float)
    found = 0
    for t in truth:
        if len(c) and np.min(np.hypot(*(c[:, :2] - t).T)) <= 4:
            found += 1
    hits = 0
    for d in c:
        if np.min(np.hypot(*(truth - d[:2]).T)) <= 4:
            hits += 1
    return {
        "detections": int(len(c)),
        "holes_found": found,
        "not_holes": int(len(c) - hits),
        "circles": np.round(c, 1).tolist(),
    }
