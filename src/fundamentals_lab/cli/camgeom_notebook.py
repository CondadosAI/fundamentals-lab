# ruff: noqa: E501  (cell sources mirror the posts verbatim)
"""Build notebooks/camera_geometry.ipynb from the cells unit 4.1's lessons 2 to 4 print.

`uv run camgeom-notebook`. The cells that mirror a post are copied verbatim from it; keep them
in step when a post changes. The notebook is written with its outputs cleared.
"""

import json

import click

from fundamentals_lab.config import PROJECT_ROOT

CELLS: list[tuple[str, str]] = []


def md(text: str) -> None:
    CELLS.append(("markdown", text.strip("\n")))


def code(text: str) -> None:
    CELLS.append(("code", text.strip("\n")))


md("""
# Camera geometry, measured

Companion notebook for the CondadosAI Fundamentals unit **Camera geometry** (F4 / unit 4.1),
lessons 2 to 4: the camera matrix, calibration and PnP. Every Python block printed in those
posts has a cell here, and each lesson's exercises have a cell that computes the answers the
post gives. Lesson 1, homogeneous coordinates, lives in `image_alignment.ipynb`.

| Used for | Source | Licence |
|---|---|---|
| lessons 2 to 4 | the inner corners detected in OpenCV's 13 `samples/data/left*.jpg` chessboard photos, served by condados.ai as `corners.json` | Apache-2.0 |

The board's square size is not published, so every distance is in board squares. The full
pipeline, which detects the corners from the photos, is `uv run formation-download` then
`uv run camgeom-experiments` in the repository.
""")

code("""
# Pinned so a Colab base-image change cannot silently move the numbers.
!pip install -q 'opencv-python>=5.0.0,<6.0.0' 'numpy>=1.26'
""")

code("""
import os
import urllib.request
from pathlib import Path

import cv2
import numpy as np

# condados.ai sits behind Cloudflare, which refuses Python's default user agent.
BASE = os.environ.get('CAMGEOM_BASE', 'https://condados.ai')
UA = {'User-Agent': 'fundamentals-lab/0.1 (+https://github.com/CondadosAI/fundamentals-lab)'}
if not Path('corners.json').exists():
    req = urllib.request.Request(BASE + '/blog/what-is-camera-calibration/corners.json', headers=UA)
    Path('corners.json').write_bytes(urllib.request.urlopen(req).read())
print('OpenCV', cv2.__version__, '| NumPy', np.__version__)
""")

md("## Lesson 2: The camera matrix")

md("### Lesson 2: how close is this to OpenCV?")

code(r"""
import numpy as np

# who the camera is: focal lengths and centre, in pixels
K = np.array([[536.0735, 0, 342.3705],
              [0, 536.0164, 235.5369],
              [0, 0, 1]])
# where it stands for the first photo: rotation, shift
R = np.array([[0.962221, 0.009801, 0.272095],
              [0.036270, 0.985831, -0.163771],
              [-0.269845, 0.167453, 0.948232]])
t = np.array([-3.011188, -4.357567, 15.992873])

P = K @ np.hstack([R, t[:, None]])  # the whole camera
X = np.array([8, 0, 0, 1.0])        # a board corner
u, v, w = P @ X
print("P @ X =", np.round([u, v, w], 3))
print(f"pixel: ({u / w:.2f}, {v / w:.2f})")
""")

code(r"""
import cv2

rvec, _ = cv2.Rodrigues(R)
pt = X[None, :3]
p, _ = cv2.projectPoints(pt, rvec, t, K, None)
print("cv2.projectPoints:", p.ravel().round(2))

dist = np.array([-0.26509, -0.04674, 0.00183,
                 -0.00031, 0.25231])
q, _ = cv2.projectPoints(pt, rvec, t, K, dist)
print("with the lens:", q.ravel().round(2),
      " detected: [513.77  86.53]")

# and back: take P apart again
Kd, Rd, Ch = cv2.decomposeProjectionMatrix(P)[:3]
print("K back:", (Kd / Kd[2, 2]).diagonal().round(2))
print("camera centre:", (Ch[:3] / Ch[3]).ravel().round(3))
""")

md("### Lesson 2: exercises")

code(r"""
def to_pixel(P, X):
    return (0.0, 0.0)  # (u, v) for a point (X, Y, Z, W)

print(to_pixel(P, X))
""")

code(r"""
# checks your answer
print("correct" if (np.allclose(to_pixel(P, X), (u / w, v / w), atol=0.01)) else "not yet")
""")

code(r"""
# Exercises 2 and 3: the numbers under "What you should see"
far = P @ np.array([1, 0, 0, 0.0])   # the board's X axis, at infinity
print("X axis vanishes at",
      np.round(far[:2] / far[2], 1))
C = -R.T @ t                         # camera centre on the board
print("camera centre:", C.round(3),
      " distance:", round(float(np.linalg.norm(C)), 2))
""")

md("## Lesson 3: Calibration")

md("### Lesson 3: how close is this to OpenCV?")

code(r"""
import json
import numpy as np
import cv2

d = json.load(open("corners.json"))
# the board's own coordinates, one unit per square
obj = np.zeros((54, 3), np.float32)
obj[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2)
img = [np.float32(c).reshape(-1, 1, 2)
       for c in d["corners"]]
size = tuple(d["image_size"])

rms, K, dist, rvecs, tvecs = cv2.calibrateCamera(
    [obj] * len(img), img, size, None, None)
print(len(img), "views, RMS", round(rms, 4), "px")
print("fx, fy:", K[0, 0].round(2), K[1, 1].round(2))
print("cx, cy:", K[0, 2].round(2), K[1, 2].round(2))
print("k1, k2, p1, p2, k3:", dist.ravel().round(4))
""")

code(r"""
# a view the fit never saw: calibrate on the other 12,
# then place the first board with that camera
K12, d12 = cv2.calibrateCamera(
    [obj] * 12, img[1:], size, None, None)[1:3]
ok, rv, tv = cv2.solvePnP(obj, img[0], K12, d12)
p, _ = cv2.projectPoints(obj, rv, tv, K12, d12)
e = np.sqrt(np.mean(np.sum(
    (p - img[0]).reshape(-1, 2) ** 2, axis=1)))
print("first view, held out:", round(float(e), 3), "px")

# the same 13 views with no lens model at all
flags = (cv2.CALIB_FIX_K1 | cv2.CALIB_FIX_K2
         | cv2.CALIB_FIX_K3 | cv2.CALIB_ZERO_TANGENT_DIST)
rms0, K0 = cv2.calibrateCamera(
    [obj] * 13, img, size, None, np.zeros(5),
    flags=flags)[:2]
print("pinhole only: RMS", round(rms0, 3),
      "px, fx", K0[0, 0].round(2))
""")

md("### Lesson 3: exercises")

code(r"""
def board(cols, rows):
    return np.zeros((cols * rows, 3), np.float32)

print(board(9, 6)[:3])
""")

code(r"""
# checks your answer
print("correct" if (np.array_equal(board(9, 6), obj) and board(3, 2)[-1].tolist() == [2.0, 1.0, 0.0]) else "not yet")
""")

code(r"""
# Exercises 2 and 3: the numbers under "What you should see"
for views in ((0, 1, 2), (0, 5, 10), (3, 7, 11)):
    Kv = cv2.calibrateCamera(
        [obj] * 3, [img[i] for i in views],
        size, None, None)[1]
    print("views", views, "fx", Kv[0, 0].round(1))
print("all 13: fx", K[0, 0].round(1))
""")

md("## Lesson 4: PnP")

md("### Lesson 4: how close is this to OpenCV?")

code(r"""
import json
import numpy as np
import cv2

d = json.load(open("corners.json"))
obj = np.zeros((54, 3), np.float32)
obj[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2)
seen = np.float32(d["corners"][0]).reshape(-1, 1, 2)

# the camera, from calibration
K = np.array([[536.0735, 0, 342.3705],
              [0, 536.0164, 235.5369], [0, 0, 1]])
dist = np.array([-0.26509, -0.04674, 0.00183,
                 -0.00031, 0.25231])

ok, rvec, tvec = cv2.solvePnP(obj, seen, K, dist)
R, _ = cv2.Rodrigues(rvec)
C = (-R.T @ tvec).ravel()

def rms(rv, tv, o=obj, s=seen):
    p, _ = cv2.projectPoints(o, rv, tv, K, dist)
    return float(np.sqrt(np.mean(np.sum(
        (p - s).reshape(-1, 2) ** 2, axis=1))))

print("camera centre, in squares:", C.round(3))
print("distance:", round(float(np.linalg.norm(C)), 2))
print("misses the corners by", round(rms(rvec, tvec), 3),
      "px RMS")
""")

code(r"""
# five corners moved 25 to 50 px, as a bad detection would
bad = seen.copy().reshape(-1, 2)
idx = [5, 17, 23, 38, 50]
bad[idx] += np.float32([[40, -25], [-35, 30], [30, 35],
                        [-40, -20], [25, -40]])
bad = bad.reshape(-1, 1, 2)
good = np.setdiff1d(np.arange(54), idx)

def shift(rv, tv):
    Rm, _ = cv2.Rodrigues(rv)
    return float(np.linalg.norm((-Rm.T @ tv).ravel() - C))

ok, rv1, tv1 = cv2.solvePnP(obj, bad, K, dist)
print(f"solvePnP:       moves {shift(rv1, tv1):.3f}, "
      f"{rms(rv1, tv1, obj[good], seen[good]):.3f} px")
ok, rv2, tv2, inl = cv2.solvePnPRansac(
    obj, bad, K, dist, reprojectionError=3.0)
print(f"solvePnPRansac: moves {shift(rv2, tv2):.4f}, "
      f"{rms(rv2, tv2, obj[good], seen[good]):.3f} px, "
      f"{len(inl)} inliers")
""")

md("### Lesson 4: exercises")

code(r"""
def centre(rvec, tvec):
    return np.zeros(3)  # camera centre, board units

print(centre(rvec, tvec))
""")

code(r"""
# checks your answer
print("correct" if (np.allclose(centre(rvec, tvec), C, atol=1e-6)) else "not yet")
""")

code(r"""
# Exercises 2 and 3: the numbers under "What you should see"
four = [0, 8, 45, 53]   # the four outer corners
ok, rv4, tv4 = cv2.solvePnP(obj[four], seen[four], K, dist,
                            flags=cv2.SOLVEPNP_IPPE)
print(f"4 corners: moves {shift(rv4, tv4):.3f} squares, "
      f"{rms(rv4, tv4):.3f} px on all 54")

rng = np.random.default_rng(0)
moves = []
for _ in range(50):
    noisy = seen + rng.normal(0, 1, seen.shape).astype(np.float32)
    ok, rvn, tvn = cv2.solvePnP(obj, noisy, K, dist)
    moves.append(shift(rvn, tvn))
print("1 px of noise, 50 trials: median move",
      round(float(np.median(moves)), 3), "squares")
""")


@click.command()
def camgeom_notebook() -> None:
    """Write notebooks/camera_geometry.ipynb, outputs cleared."""
    nb = {
        "cells": [
            {
                "cell_type": t,
                "id": f"cell-{i:03d}",
                "metadata": {},
                "source": src.splitlines(keepends=True),
                **({"execution_count": None, "outputs": []} if t == "code" else {}),
            }
            for i, (t, src) in enumerate(CELLS)
        ],
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.12"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    out = PROJECT_ROOT / "notebooks" / "camera_geometry.ipynb"
    out.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n")
    print(f"wrote {out} ({len(CELLS)} cells)")


if __name__ == "__main__":
    camgeom_notebook()
