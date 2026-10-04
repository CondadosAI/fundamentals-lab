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
Hough circle transform. Every Python block printed in those posts has a cell here, and each
lesson's exercises have a cell that computes the answers the post gives.

| Used for | Source | Licence |
|---|---|---|
| lessons 1 to 3 | comma10k frame `0825_e61068239ce72500_2018-11-13--21-06-53_13_903` and its hand-painted lane mask (comma.ai), served losslessly by condados.ai | MIT |
| lesson 4 | "Billiards table 2.JPG", "Billiards table 1.JPG" and "Billiards Table.JPG" by MarkBuckawicki (Wikimedia Commons) at 960 px, and the 41 ball labels of OpenCV in Practice M2 | CC0 1.0 |

The full pipeline, with the 299 held-out frames, the comma2k19 video and the in-the-wild
photographs, is `uv run boundary-download` then `uv run boundary-experiments` in the repository.
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
BASE = os.environ.get('BOUNDARY_BASE', 'https://condados.ai')
UA = {'User-Agent': 'fundamentals-lab/0.1 (+https://github.com/CondadosAI/fundamentals-lab)'}
FILES = ['/blog/boundary-detection/highway.webp', '/blog/boundary-detection/highway-lane.webp',
         '/blog/line-fitting-least-squares/dash-mask.webp', '/blog/line-fitting-least-squares/line-mask.webp',
         '/practice/m2/pool.webp', '/practice/m2/pool-close.webp', '/practice/m2/pool-wide.webp',
         '/practice/m2/labels.json']
for f in FILES:
    name = f.rsplit('/', 1)[1]
    if not Path(name).exists():
        req = urllib.request.Request(BASE + f, headers=UA)
        Path(name).write_bytes(urllib.request.urlopen(req).read())
print('OpenCV', cv2.__version__, '| NumPy', np.__version__, '|', cv2.imread('highway.webp').shape)
""")

md("""
## 1. Fitting a line to edge pixels

The cells of [the lesson](https://condados.ai/blog/line-fitting-least-squares), in order.
""")

code(r"""
import numpy as np
import cv2

# the dash's pixels, painted by comma10k's labellers
mask = cv2.imread("dash-mask.webp", cv2.IMREAD_GRAYSCALE)
ys, xs = np.nonzero(mask)
P = np.c_[xs, ys].astype(float)

# least squares, y on x and x on y
m = np.polyfit(P[:, 0], P[:, 1], 1)[0]
m2 = np.polyfit(P[:, 1], P[:, 0], 1)[0]
y_on_x = np.degrees(np.arctan(m)) % 180
x_on_y = np.degrees(np.arctan2(1, m2)) % 180

# total least squares: first principal direction
d = np.linalg.svd(P - P.mean(axis=0))[2][0]
tls = np.degrees(np.arctan2(d[1], d[0])) % 180

print(len(P), "pixels")
print(f"y on x: {y_on_x:.2f}   x on y: {x_on_y:.2f}")
print(f"total least squares: {tls:.2f}")
""")

code(r"""
v = cv2.fitLine(P.astype(np.float32), cv2.DIST_L2,
                0, 0.01, 0.01).ravel()
fit = np.degrees(np.arctan2(v[1], v[0])) % 180
print(f"cv2.fitLine:        {fit:.2f}")
print(f"difference to ours: {abs(fit - tls):.4f}")
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
def three(Q):
    m = np.polyfit(Q[:, 0], Q[:, 1], 1)[0]
    m2 = np.polyfit(Q[:, 1], Q[:, 0], 1)[0]
    d = np.linalg.svd(Q - Q.mean(axis=0))[2][0]
    return (np.degrees(np.arctan(m)) % 180, np.degrees(np.arctan2(1, m2)) % 180,
            np.degrees(np.arctan2(d[1], d[0])) % 180)

R = cv2.getRotationMatrix2D(tuple(P.mean(axis=0)), -60, 1)
print("dash turned 60 deg: y on x %.2f  x on y %.2f  TLS %.2f" % three(P @ R[:, :2].T + R[:, 2]))
ly, lx = np.nonzero(cv2.imread("line-mask.webp", cv2.IMREAD_GRAYSCALE))
print("solid line, %d px: y on x %.2f  x on y %.2f  TLS %.2f" % (len(lx), *three(np.c_[lx, ly].astype(float))))
""")

md("""
## 2. The Hough transform

The cells of [the lesson](https://condados.ai/blog/hough-transform), in order.
""")

code(r"""
import numpy as np
import cv2

img = cv2.imread("highway.webp")
h, w = img.shape[:2]
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (0, 0), 1.4)

# only the road ahead: a fixed trapezoid
road = np.zeros((h, w), np.uint8)
corners = [(0.02, 0.78), (0.42, 0.50),
           (0.58, 0.50), (0.98, 0.78)]
poly = np.array([(fx * w, fy * h)
                 for fx, fy in corners], np.int32)
cv2.fillPoly(road, [poly], 255)
edges = cv2.Canny(blur, 25, 75) & road

ys, xs = np.nonzero(edges)
xs = xs.astype(np.float32)
ys = ys.astype(np.float32)
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
peak = ((m > 60) & (m > A[:-2, 1:-1])
        & (m >= A[2:, 1:-1]) & (m > A[1:-1, :-2])
        & (m >= A[1:-1, 2:]))
ours = set(zip(*np.nonzero(peak)))

ref = cv2.HoughLinesWithAccumulator(
    edges, 1, np.pi / 180, 60).reshape(-1, 3)
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
pts = [(490, 437), (318, 548), (119, 678)]
cell = (0, 0)  # (rho, theta in degrees)
print(cell)

# checks your answer
print("correct" if (tuple(int(v) for v in cell) == (633, 57)) else "not yet")
""")

code(r"""
# Exercises 2 and 3: the numbers under "What you should see".
# A line is on paint when half of the edge pixels within 1.5 px of it lie within 3 px of the
# hand-painted lane mask, the rule the unit scores with.
lane = cv2.imread("highway-lane.webp", cv2.IMREAD_GRAYSCALE) > 0
near = cv2.dilate(lane.astype(np.uint8), np.ones((7, 7), np.uint8)) > 0
ey, ex = np.nonzero(edges)

def peaks(a, thr):
    A = np.pad(a, 1)
    m = A[1:-1, 1:-1]
    pk = ((m > thr) & (m > A[:-2, 1:-1]) & (m >= A[2:, 1:-1])
          & (m > A[1:-1, :-2]) & (m >= A[1:-1, 2:]))
    return [(int(a[r, t]), int(r) - half, int(t)) for r, t in zip(*np.nonzero(pk))]

def on_paint(lines, deg_per_bin=1.0):
    n = 0
    for _, rho, t in lines:
        th = np.radians(t * deg_per_bin)
        d = np.abs(ex * np.cos(th) + ey * np.sin(th) - rho) <= 1.5
        n += bool(d.any() and near[ey[d], ex[d]].mean() >= 0.5)
    return n

acc2 = np.zeros((n_rho, 360), np.int32)
ang = np.float32(0)
step2 = np.float32(np.pi / 360)
for t in range(360):
    c = np.float32(np.cos(np.float64(ang)))
    s = np.float32(np.sin(np.float64(ang)))
    r = np.rint(xs * c + ys * s).astype(int)
    acc2[:, t] = np.bincount(r + half, minlength=n_rho)
    ang = np.float32(ang + step2)
p2 = peaks(acc2, 60)
print("0.5 deg bins:", len(p2), "lines,", on_paint(p2, 0.5), "on paint, strongest", int(acc2.max()))

for thr in (40, 120, 200):
    p = peaks(acc, thr)
    print(f"over {thr}: {len(p)} lines, {on_paint(p)} on paint;",
          "theta of each:", sorted(t for _, _, t in p) if len(p) < 6 else "...")
print("strongest cell of the dash (theta 115-130):", int(acc[:, 115:131].max()))
""")

md("""
## 3. The probabilistic Hough transform

The cells of [the lesson](https://condados.ai/blog/probabilistic-hough-transform), in order.
""")

code(r"""
import numpy as np
import cv2

img = cv2.imread("highway.webp")
h, w = img.shape[:2]
blur = cv2.GaussianBlur(
    cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (0, 0), 1.4)
road = np.zeros((h, w), np.uint8)
corners = [(0.02, 0.78), (0.42, 0.50),
           (0.58, 0.50), (0.98, 0.78)]
cv2.fillPoly(road, [np.array(
    [(fx * w, fy * h) for fx, fy in corners], np.int32)], 255)
edges = cv2.Canny(blur, 25, 75) & road

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
peak = ((m > 60) & (m > A[:-2, 1:-1])
        & (m >= A[2:, 1:-1]) & (m > A[1:-1, :-2])
        & (m >= A[1:-1, 2:]))
# the lane prior: nothing within 20 degrees of flat
ours = [(r, t) for r, t in zip(*np.nonzero(peak))
        if not 70 <= t <= 110]
ref = cv2.HoughLines(edges, 1, np.pi / 180, 60)
cv = {(round(rr) + half, round(np.degrees(tt)))
      for rr, tt in ref.reshape(-1, 2)}
print("restricted lines:", len(ours),
      " also in OpenCV's full set:",
      sum((r, t) in cv for r, t in ours), "of", len(cv))

seg1 = cv2.HoughLinesP(edges, 1, np.pi / 180, 40,
                       minLineLength=40, maxLineGap=20)
seg2 = cv2.HoughLinesP(edges, 1, np.pi / 180, 40,
                       minLineLength=40, maxLineGap=20)
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
print(sorted(votes(61.39, 5)))

# checks your answer
print("correct" if (sorted(votes(178.2, 3)) == [0, 1, 175, 176, 177, 178, 179] and sorted(votes(61.39, 5)) == list(range(56, 67))) else "not yet")
""")

code(r"""
# Exercises 2 and 3: the numbers under "What you should see"
lane = cv2.imread("highway-lane.webp", cv2.IMREAD_GRAYSCALE) > 0
near = cv2.dilate(lane.astype(np.uint8), np.ones((7, 7), np.uint8)) > 0
paint = lane & (road > 0)
ey, ex = np.nonzero(edges)

acc_k2 = np.zeros((n_rho, 180), np.int32)
for off in range(-2, 3):
    t = (centre + off) % 180
    r = np.rint(x32 * cos_t[t] + y32 * sin_t[t])
    np.add.at(acc_k2, (r.astype(int) + half, t), 1)
A = np.pad(acc_k2, 1)
m = A[1:-1, 1:-1]
pk = ((m > 60) & (m > A[:-2, 1:-1]) & (m >= A[2:, 1:-1])
      & (m > A[1:-1, :-2]) & (m >= A[1:-1, 2:]))
k2 = [(int(r) - half, int(t)) for r, t in zip(*np.nonzero(pk)) if not 70 <= t <= 110]
on = 0
for rho, t in k2:
    d = np.abs(ex * np.cos(np.radians(t)) + ey * np.sin(np.radians(t)) - rho) <= 1.5
    on += bool(d.any() and near[ey[d], ex[d]].mean() >= 0.5)
print("k = 2:", len(k2), "lines,", on, "on paint, strongest", int(acc_k2.max()))

for gap in (5, 20, 60):
    seg = cv2.HoughLinesP(edges, 1, np.pi / 180, 40, minLineLength=40, maxLineGap=gap)
    seg = [] if seg is None else seg.reshape(-1, 4)
    cover = np.zeros(lane.shape, np.uint8)
    for x1, y1, x2, y2 in seg:
        cv2.line(cover, (int(x1), int(y1)), (int(x2), int(y2)), 1, 7)
    print(f"maxLineGap {gap:2}: {len(seg)} segments, paint covered {(cover.astype(bool) & paint).sum() / paint.sum():.1%}")
""")

md("""
## 4. The Hough circle transform

The cells of [the lesson](https://condados.ai/blog/hough-circle-transform), in order.
""")

code(r"""
import json
import numpy as np
import cv2

img = cv2.imread("pool-close.webp")
g = cv2.medianBlur(
    cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 5)
labels = json.load(open("labels.json"))
bx, by, br = max(labels["pool-close"], key=lambda b: b[2])

edges = cv2.Canny(g, 50, 150)
gx = cv2.Sobel(g, cv2.CV_32F, 1, 0)
gy = cv2.Sobel(g, cv2.CV_32F, 0, 1)
ys, xs = np.nonzero(edges)
mag = np.hypot(gx[ys, xs], gy[ys, xs]) + 1e-9
ux, uy = gx[ys, xs] / mag, gy[ys, xs] / mag

r = 52
H, W = g.shape
acc = np.zeros((H, W), np.int32)
for s in (1, -1):  # both ways along the gradient
    cx = np.rint(xs + s * r * ux).astype(int)
    cy = np.rint(ys + s * r * uy).astype(int)
    ok = (cx >= 0) & (cy >= 0) & (cx < W) & (cy < H)
    np.add.at(acc, (cy[ok], cx[ok]), 1)
print("edge pixels:", len(xs), " ball:", (bx, by, br))

for cell in (1, 2, 4):
    h, w = H // cell * cell, W // cell * cell
    a = acc[:h, :w].reshape(
        h // cell, cell, w // cell, cell).sum(axis=(1, 3))
    y0, x0 = (by - 120) // cell, (bx - 120) // cell
    win = a[y0:y0 + 240 // cell, x0:x0 + 240 // cell]
    yy, xx = np.unravel_index(win.argmax(), win.shape)
    px, py = (x0 + xx + 0.5) * cell, (y0 + yy + 0.5) * cell
    d = np.hypot(px - bx, py - by)
    print(f"cell {cell} px: {win.max()} votes"
          f" at ({px}, {py}), {d:.1f} px off")
""")

code(r"""
found = cv2.HoughCircles(
    g, cv2.HOUGH_GRADIENT_ALT, 1.5, 12,
    param1=300, param2=0.8,
    minRadius=8, maxRadius=60)[0]
x, y, rr = min(found,
               key=lambda c: np.hypot(c[0] - bx, c[1] - by))
print("circles on the photo:", len(found))
print(f"big ball: ({x:.1f}, {y:.1f}) radius {rr:.1f},"
      f" {np.hypot(x - bx, y - by):.2f} px from the label")
""")

code(r"""
import json
import numpy as np
import cv2

labels = json.load(open("labels.json"))
grey = {n: cv2.medianBlur(cv2.cvtColor(
    cv2.imread(n + ".webp"), cv2.COLOR_BGR2GRAY), 5)
    for n in labels}

def match(found, truth):
    pairs = sorted(
        (np.hypot(x - lx, y - ly), i, j)
        for i, (lx, ly, _) in enumerate(truth)
        for j, (x, y, _) in enumerate(found))
    li, cj = set(), set()
    for d, i, j in pairs:
        if i in li or j in cj or d >= max(truth[i][2], 8):
            continue
        li.add(i)
        cj.add(j)
    return len(li), len(found) - len(cj)

def score(method, dp, p1, p2):
    hit = false = 0
    for n, truth in labels.items():
        c = cv2.HoughCircles(grey[n], method, dp, 12,
                             param1=p1, param2=p2,
                             minRadius=8, maxRadius=60)
        h, f = match([] if c is None else c[0], truth)
        hit, false = hit + h, false + f
    return hit, false

for p2 in (20, 25, 30, 35, 40):
    h, f = score(cv2.HOUGH_GRADIENT, 1, 100, p2)
    print(f"GRADIENT     param2={p2}:  {h}/41, {f} false")
for p2 in (0.7, 0.8, 0.85, 0.9):
    h, f = score(cv2.HOUGH_GRADIENT_ALT, 1.5, 300, p2)
    print(f"GRADIENT_ALT param2={p2}: {h}/41, {f} false")
""")

md("""
### Exercise 1

Fill it in; the last line checks your answer.
""")

code(r"""
import math

def centre(x, y, direction, r):
    return (x, y)  # move r pixels along the direction

print(centre(179, 454, 86.7, 52))
print(centre(216, 466, 130.0, 52))

# checks your answer
print("correct" if ([round(v) for v in centre(179, 454, 86.7, 52)] == [182, 506] and [round(v) for v in centre(216, 466, 130.0, 52)] == [183, 506]) else "not yet")
""")

code(r"""
# Exercises 2 and 3: the numbers under "What you should see"
print("ALT, param2 0.9:", score(cv2.HOUGH_GRADIENT_ALT, 1.5, 300, 0.9))
for p2 in (20, 30):
    print(f"GRADIENT, param2 {p2}:", score(cv2.HOUGH_GRADIENT, 1, 100, p2))
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
