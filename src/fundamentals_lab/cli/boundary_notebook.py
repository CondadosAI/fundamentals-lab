# ruff: noqa: E501  (cell sources mirror the posts verbatim, line lengths included)
"""Build notebooks/boundary_detection.ipynb from the code blocks the posts print.

`uv run boundary-notebook`. The cells that mirror a post are copied verbatim from it; keep
them in step when a post changes. The notebook is written with its outputs cleared.
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
# Boundary detection, measured

Companion notebook for the CondadosAI Fundamentals unit **Boundary detection** (F3 / unit 3.2):
fitting a line to edge pixels, the Hough transform, the probabilistic Hough transform and the
generalized Hough transform. Every Python block printed in those posts has a cell here, and each
lesson's exercises have a cell that computes the answers the post gives.

| Used for | Source | Licence |
|---|---|---|
| every lesson | VisA `pcb1`, `Data/Images/Normal/0000.JPG` (Zou et al., ECCV 2022, Amazon.com, Inc. or its affiliates), reduced to 900x686 and served losslessly by condados.ai | CC BY 4.0 |

The full pipeline, with the held-out frames and the in-the-wild photographs, is
`uv run boundary-download` then `uv run boundary-experiments` in the repository.
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
BASE = os.environ.get('BOUNDARY_BASE', 'https://condados.ai/blog/boundary-detection')
UA = {'User-Agent': 'fundamentals-lab/0.1 (+https://github.com/CondadosAI/fundamentals-lab)'}
if not Path('pcb1.webp').exists():
    req = urllib.request.Request(f'{BASE}/pcb1.webp', headers=UA)
    Path('pcb1.webp').write_bytes(urllib.request.urlopen(req).read())
print('OpenCV', cv2.__version__, '| NumPy', np.__version__, '|', cv2.imread('pcb1.webp').shape)
""")

md("""
## 1. Fitting a line to edge pixels

The cells of [the lesson](https://condados.ai/blog/line-fitting-least-squares), in order.
""")

code(r"""
import numpy as np
import cv2

img = cv2.imread("pcb1.webp")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (0, 0), 1.4)
edges = cv2.Canny(blur, 50, 150)

# the board's left side: x 110-128, y 260-450
ys, xs = np.nonzero(edges[260:450, 110:128])
P = np.c_[xs + 110, ys + 260].astype(float)

# least squares: y = m x + b
A = np.c_[P[:, 0], np.ones(len(P))]
m, b = np.linalg.lstsq(A, P[:, 1], rcond=None)[0]
ls = np.degrees(np.arctan(m)) % 180

# total least squares: first principal direction
mu = P.mean(axis=0)
d = np.linalg.svd(P - mu)[2][0]
tls = np.degrees(np.arctan2(d[1], d[0])) % 180

print(len(P), "pixels")
print(f"least squares:       {ls:.2f}")
print(f"total least squares: {tls:.2f}")
""")

code(r"""
v = cv2.fitLine(P.astype(np.float32), cv2.DIST_L2,
                0, 0.01, 0.01).ravel()
fit = np.degrees(np.arctan2(v[1], v[0])) % 180
print(f"cv2.fitLine:         {fit:.2f}")
print(f"difference to ours:  {abs(fit - tls):.4f}")
""")

md("""
### Exercise 1

Fill it in; the last line checks your answer.
""")

code(r"""
def tls_angle(P):
    # centred sums, then the formula
    return 0.0

print(tls_angle(P), fit)

# checks your answer
print("correct" if (abs(tls_angle(P) - fit) < 0.01) else "not yet")
""")

code(r"""
# Exercises 2 and 3: the numbers under "What you should see"
def fits(Q):
    A = np.c_[Q[:, 0], np.ones(len(Q))]
    m = np.linalg.lstsq(A, Q[:, 1], rcond=None)[0][0]
    d = np.linalg.svd(Q - Q.mean(axis=0))[2][0]
    return np.degrees(np.arctan(m)) % 180, np.degrees(np.arctan2(d[1], d[0])) % 180

ys, xs = np.nonzero(edges[198:212, 290:540])
T = np.c_[xs + 290, ys + 198].astype(float)
print("top edge: LS %.4f  TLS %.4f" % fits(T))

mu = P.mean(axis=0)
for deg in (0, 15, 45, 80):
    t = np.radians(-deg)
    R = np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])
    a, b = fits((P - mu) @ R.T + mu)
    print(f"turned {deg:2} deg: LS {a:6.2f}  TLS {b:6.2f}  gap {((a - b + 90) % 180) - 90:+.2f}")
""")

md("""
## 2. The Hough transform

The cells of [the lesson](https://condados.ai/blog/hough-transform), in order.
""")

code(r"""
import numpy as np
import cv2

img = cv2.imread("pcb1.webp")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (0, 0), 1.4)
edges = cv2.Canny(blur, 50, 150)

ys, xs = np.nonzero(edges)
xs = xs.astype(np.float32)
ys = ys.astype(np.float32)
h, w = edges.shape
n_rho = 2 * (w + h) + 1
half = (n_rho - 1) // 2

acc = np.zeros((n_rho, 180), np.int32)
ang = np.float32(0)
step = np.float32(np.pi / 180)
for t in range(180):
    c = np.float32(np.cos(np.float64(ang)))
    s = np.float32(np.sin(np.float64(ang)))
    r = np.rint(xs * c + ys * s).astype(int)
    acc[:, t] = np.bincount(r + half, minlength=n_rho)
    ang = np.float32(ang + step)

print(len(xs), "edge pixels,", len(xs) * 180, "votes")
r, t = np.unravel_index(acc.argmax(), acc.shape)
print("strongest:", acc.max(), "votes at rho",
      r - half, "theta", t)
""")

code(r"""
# OpenCV's rule for a line: over the threshold, and a
# local maximum against its four neighbours
A = np.pad(acc, 1)
m = A[1:-1, 1:-1]
peak = ((m > 120) & (m > A[:-2, 1:-1])
        & (m >= A[2:, 1:-1]) & (m > A[1:-1, :-2])
        & (m >= A[1:-1, 2:]))
ours = set(zip(*np.nonzero(peak)))

ref = cv2.HoughLinesWithAccumulator(
    edges, 1, np.pi / 180, 120).reshape(-1, 3)
cv = {(round(rr) + half, round(np.degrees(tt)))
      for rr, tt, _ in ref}
print("ours:", len(ours), " OpenCV:", len(cv))
print("same lines:", ours == cv)
print("same votes:", all(
    acc[round(rr) + half, round(np.degrees(tt))] == v
    for rr, tt, v in ref))
""")

md("""
### Exercise 1

Fill it in; the last line checks your answer.
""")

code(r"""
pts = [(290, 203), (319, 203), (454, 203)]
cell = (0, 0)  # (rho, theta in degrees)
print(cell)

# checks your answer
print("correct" if (tuple(int(v) for v in cell) == (203, 90)) else "not yet")
""")

code(r"""
# Exercises 2 and 3: the numbers under "What you should see"
acc2 = np.zeros((n_rho, 360), np.int32)
ang = np.float32(0)
step2 = np.float32(np.pi / 360)
for t in range(360):
    c = np.float32(np.cos(np.float64(ang)))
    s = np.float32(np.sin(np.float64(ang)))
    r = np.rint(xs * c + ys * s).astype(int)
    acc2[:, t] = np.bincount(r + half, minlength=n_rho)
    ang = np.float32(ang + step2)
A2 = np.pad(acc2, 1)
m2 = A2[1:-1, 1:-1]
pk2 = ((m2 > 120) & (m2 > A2[:-2, 1:-1]) & (m2 >= A2[2:, 1:-1])
       & (m2 > A2[1:-1, :-2]) & (m2 >= A2[1:-1, 2:]))
print("0.5 deg bins:", int(pk2.sum()), "lines, strongest", int(acc2.max()))

holes = np.array([(140, 225), (713, 225), (141, 455), (710, 455)], float)
for p in (15, 20, 25):
    c = cv2.HoughCircles(blur, cv2.HOUGH_GRADIENT, 1, 40, param1=150,
                         param2=p, minRadius=6, maxRadius=14)
    c = np.empty((0, 3)) if c is None else c[0]
    found = sum(len(c) and np.min(np.hypot(*(c[:, :2] - h).T)) <= 4 for h in holes)
    print(f"param2 {p}: {len(c)} circles, {found} of 4 holes")
""")

md("""
## 3. The probabilistic Hough transform

The cells of [the lesson](https://condados.ai/blog/probabilistic-hough-transform), in order.
""")

code(r"""
import numpy as np
import cv2

img = cv2.imread("pcb1.webp")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (0, 0), 1.4)
edges = cv2.Canny(blur, 50, 150)

gx = cv2.Sobel(blur, cv2.CV_32F, 1, 0)
gy = cv2.Sobel(blur, cv2.CV_32F, 0, 1)
ys, xs = np.nonzero(edges)
grad = np.degrees(np.arctan2(gy, gx)) % 180
centre = np.rint(grad[ys, xs]).astype(int)

# lesson 2's angle table, built the way OpenCV builds it
cos_t = np.zeros(180, np.float32)
sin_t = np.zeros(180, np.float32)
ang = np.float32(0)
for t in range(180):
    cos_t[t] = np.cos(np.float64(ang))
    sin_t[t] = np.sin(np.float64(ang))
    ang = np.float32(ang + np.float32(np.pi / 180))

h, w = edges.shape
n_rho = 2 * (w + h) + 1
half = (n_rho - 1) // 2
acc = np.zeros((n_rho, 180), np.int32)
k = 5
x32, y32 = xs.astype(np.float32), ys.astype(np.float32)
for off in range(-k, k + 1):
    t = (centre + off) % 180
    r = np.rint(x32 * cos_t[t] + y32 * sin_t[t])
    np.add.at(acc, (r.astype(int) + half, t), 1)

print("votes cast:", len(xs) * (2 * k + 1))
""")

code(r"""
A = np.pad(acc, 1)
m = A[1:-1, 1:-1]
peak = ((m > 120) & (m > A[:-2, 1:-1])
        & (m >= A[2:, 1:-1]) & (m > A[1:-1, :-2])
        & (m >= A[1:-1, 2:]))
ours = sorted(zip(*np.nonzero(peak)),
              key=lambda rt: -acc[rt])
ref = cv2.HoughLines(edges, 1, np.pi / 180, 120)
cv = {(round(rr) + half, round(np.degrees(tt)))
      for rr, tt in ref.reshape(-1, 2)}
print("restricted lines:", len(ours))
for r, t in ours:
    print(f"  rho {r - half:5}  theta {t:3}  "
          f"votes {acc[r, t]:3}  in OpenCV's: {(r, t) in cv}")

seg1 = cv2.HoughLinesP(edges, 1, np.pi / 180, 90,
                       minLineLength=80, maxLineGap=5)
seg2 = cv2.HoughLinesP(edges, 1, np.pi / 180, 90,
                       minLineLength=80, maxLineGap=5)
print("segments:", len(seg1),
      " same on a second run:", np.array_equal(seg1, seg2))
""")

md("""
### Exercise 1

Fill it in; the last line checks your answer.
""")

code(r"""
def votes(direction, k):
    return []  # whole-degree thetas in [0, 180)

print(sorted(votes(178.2, 3)))
print(sorted(votes(93.63, 5)))

# checks your answer
print("correct" if (sorted(votes(178.2, 3)) == [0, 1, 175, 176, 177, 178, 179] and sorted(votes(93.63, 5)) == list(range(89, 100))) else "not yet")
""")

code(r"""
# Exercises 2 and 3: the numbers under "What you should see"
acc_k2 = np.zeros((n_rho, 180), np.int32)
for off in range(-2, 3):
    t = (centre + off) % 180
    r = np.rint(x32 * cos_t[t] + y32 * sin_t[t])
    np.add.at(acc_k2, (r.astype(int) + half, t), 1)
A = np.pad(acc_k2, 1)
m = A[1:-1, 1:-1]
pk = ((m > 120) & (m > A[:-2, 1:-1]) & (m >= A[2:, 1:-1])
      & (m > A[1:-1, :-2]) & (m >= A[1:-1, 2:]))
print("k = 2:", int(pk.sum()), "lines, strongest", int(acc_k2.max()))

for gap in (2, 5, 20):
    seg = cv2.HoughLinesP(edges, 1, np.pi / 180, 90, minLineLength=80, maxLineGap=gap)
    print(f"maxLineGap {gap:2}: {0 if seg is None else len(seg)} segments")
""")

md("""
## 4. The generalized Hough transform

The cells of [the lesson](https://condados.ai/blog/generalized-hough-transform), in order.
""")

code(r"""
import numpy as np
import cv2

img = cv2.imread("pcb1.webp")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (0, 0), 1.4)

# the left transducer, mesh flattened (lesson text)
cx, cy, r = 252.5, 345.5, 115.0
R = int(r) + 6
tm = blur[int(cy) - R:int(cy) + R,
          int(cx) - R:int(cx) + R].copy()
yy, xx = np.mgrid[:2 * R, :2 * R]
mesh = np.hypot(xx - R, yy - R) < 0.75 * r
tm[mesh] = int(np.median(tm[~mesh]))

P = [np.float32(v * 180 / np.pi) for v in
     (0.9997878412794807, -0.3258083974640975,
      0.1555786518463281, -0.04432655554792128)]

def fast_atan2(y, x):  # cv::fastAtan2, degrees
    ax, ay = np.abs(x), np.abs(y)
    eps = np.float32(2.220446049250313e-16)
    big = ax >= ay
    c = np.where(big, ay / (ax + eps), ax / (ay + eps))
    c2 = c * c
    a = (((P[3] * c2 + P[2]) * c2 + P[1]) * c2 + P[0]) * c
    a = np.where(big, a, 90 - a)
    a = np.where(x < 0, 180 - a, a)
    return np.where(y < 0, 360 - a, a)

def edges_and_rows(g, levels=360):
    e = cv2.Canny(g, 50, 150)
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1)
    ys, xs = np.nonzero(e)
    ang = fast_atan2(gy[ys, xs], gx[ys, xs])
    rows = np.rint(ang * np.float32(levels / 360))
    return xs, ys, rows.astype(int)

xs, ys, rows = edges_and_rows(tm)
ref = (tm.shape[1] // 2, tm.shape[0] // 2)
table = {}
for x, y, row in zip(xs, ys, rows):
    table.setdefault(row, []).append(
        (ref[0] - x, ref[1] - y))
print(sum(map(len, table.values())), "offsets in",
      len(table), "rows")
""")

code(r"""
xs, ys, rows = edges_and_rows(blur)
h, w = blur.shape
acc = np.zeros((h // 2 + 3, w // 2 + 3), np.int32)
for row in np.unique(rows):
    off = np.float32(table.get(row, np.empty((0, 2))))
    if not len(off):
        continue
    sel = rows == row
    px = xs[sel, None] + off[None, :, 0]
    py = ys[sel, None] + off[None, :, 1]
    X = np.rint(px * np.float32(0.5)).astype(int)
    Y = np.rint(py * np.float32(0.5)).astype(int)
    np.add.at(acc, (Y.ravel(), X.ravel()), 1)

def two_peaks(a):
    out = []
    for _ in range(2):
        y, x = np.unravel_index(a.argmax(), a.shape)
        out.append((int(a[y, x]), 2 * int(x), 2 * int(y)))
        a[max(0, y - 50):y + 51, max(0, x - 50):x + 51] = 0
    return out

print("ours:  ", two_peaks(acc.copy()))
g = cv2.createGeneralizedHoughBallard()
g.setCannyLowThresh(50)
g.setCannyHighThresh(150)
g.setLevels(360)
g.setDp(2)
g.setMinDist(100)
g.setVotesThreshold(40)
g.setTemplate(tm)
pos, votes = g.detect(blur)
print("OpenCV:", [(int(v[0]), round(p[0]), round(p[1]))
                  for p, v in zip(pos[0][:2], votes[0][:2])])
""")

md("""
### Exercise 1

Fill it in; the last line checks your answer.
""")

code(r"""
def offset(phi, r):
    return (0, 0)  # (dx, dy) back to the centre

print(offset(0, 115), offset(90, 115), offset(45, 100))

# checks your answer
print("correct" if (tuple(offset(0, 115)) == (-115, 0) and tuple(offset(90, 115)) == (0, -115) and tuple(offset(45, 100)) == (-71, -71)) else "not yet")
""")

code(r"""
# Exercises 2 and 3: the numbers under "What you should see"
def vote_peaks(template):
    xs_t, ys_t, rows_t = edges_and_rows(template)
    ref_t = (template.shape[1] // 2, template.shape[0] // 2)
    tab = {}
    for x, y, row in zip(xs_t, ys_t, rows_t):
        tab.setdefault(row, []).append((ref_t[0] - x, ref_t[1] - y))
    xs_i, ys_i, rows_i = edges_and_rows(blur)
    a = np.zeros((h // 2 + 3, w // 2 + 3), np.int32)
    for row in np.unique(rows_i):
        off = np.float32(tab.get(row, np.empty((0, 2))))
        if not len(off):
            continue
        sel = rows_i == row
        X = np.rint((xs_i[sel, None] + off[None, :, 0]) * np.float32(0.5)).astype(int)
        Y = np.rint((ys_i[sel, None] + off[None, :, 1]) * np.float32(0.5)).astype(int)
        np.add.at(a, (Y.ravel(), X.ravel()), 1)
    return two_peaks(a)

small = cv2.resize(tm, None, fx=0.95, fy=0.95, interpolation=cv2.INTER_AREA)
print("template at 0.95x:", vote_peaks(small))

cx2, cy2, r2 = 605.5, 338.5, 109.6
R2 = int(r2) + 6
tm2 = blur[int(cy2) - R2:int(cy2) + R2, int(cx2) - R2:int(cx2) + R2].copy()
yy2, xx2 = np.mgrid[:2 * R2, :2 * R2]
mesh2 = np.hypot(xx2 - R2, yy2 - R2) < 0.75 * r2
tm2[mesh2] = int(np.median(tm2[~mesh2]))
print("template from the right transducer:", vote_peaks(tm2))
""")



@click.command()
def boundary_notebook() -> None:
    """Write notebooks/boundary_detection.ipynb, outputs cleared."""
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
    out = PROJECT_ROOT / "notebooks" / "boundary_detection.ipynb"
    out.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n")
    print(f"wrote {out} ({len(CELLS)} cells)")


if __name__ == "__main__":
    boundary_notebook()
