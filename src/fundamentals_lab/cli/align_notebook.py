# ruff: noqa: E501  (cell sources mirror the posts verbatim, line lengths included)
"""Build notebooks/image_alignment.ipynb from the code blocks the posts print.

`uv run align-notebook`. The cells that mirror a post are copied verbatim from it; keep
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
# Image alignment, measured

Companion notebook for the CondadosAI Fundamentals unit **Image alignment** (F3 / unit 3.5)
and for **Homogeneous coordinates** (unit 4.1, lesson 1). Every Python block printed in those
posts has a cell here.

Sources, none of them redistributed by this repository:

| Used for | Source | Licence |
|---|---|---|
| every lesson | frame 45000 and a 31-frame median of "2026.07.25 WD Open … (Gold Medal match)" by pickleball4you (YouTube `T5rmWjvt8Os`), served by condados.ai | CC BY 3.0 |
| court dimensions | USA Pickleball Official Rulebook 2026, Rule 3.A | published rules |

The full pipeline, including the in-the-wild runs, is `uv run align-download` then
`uv run align-experiments` in the repository.
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
import matplotlib.pyplot as plt
import numpy as np

# The posts' cells call show(image, title); in a notebook that is a matplotlib figure.
def show(img, title=''):
    plt.figure(figsize=(6, 6))
    plt.imshow(img[..., ::-1] if img.ndim == 3 else img, cmap='gray')
    plt.title(title)
    plt.axis('off')
    plt.show()

print('OpenCV', cv2.__version__, '| NumPy', np.__version__)
""")

md("""
## Fetch the frames

Frame 45000 and the median plate are lossless WebP, so these are the same pixels the
repository measures. The 31 frames behind the median are served at 960 px for the demo at
the end.
""")

code("""
BASE = os.environ.get('ALIGN_BASE', 'https://condados.ai/blog/what-is-a-homography')
data = Path('data'); (data / 'frames960').mkdir(parents=True, exist_ok=True)

# condados.ai sits behind Cloudflare, which refuses Python's default user agent.
UA = {'User-Agent': 'fundamentals-lab/0.1 (+https://github.com/CondadosAI/fundamentals-lab)'}

def fetch(url, dst):
    if not Path(dst).exists():
        Path(dst).write_bytes(urllib.request.urlopen(urllib.request.Request(url, headers=UA)).read())
    return dst

cv2.imwrite('frame_045000.png', cv2.imread(fetch(f'{BASE}/frames/frame_045000.webp', 'data/frame_045000.webp')))
Path('frame_045000.webp').write_bytes(Path('data/frame_045000.webp').read_bytes())  # the posts' cells read this name
plate = cv2.imread(fetch(f'{BASE}/frames/plate_median31.webp', 'data/plate_median31.webp'))
FRAMES = range(44100, 45901, 60)
small = [cv2.imread(fetch(f'{BASE}/frames/960/frame_{i:06d}.webp', f'data/frames960/frame_{i:06d}.webp'))
         for i in FRAMES]
frame = cv2.imread('frame_045000.png')
print(frame.shape, plate.shape, len(small), 'small frames')
""")

md("""
## Finding the corners (the measurement behind every lesson)

Each painted line is found as a thin bright ridge (white top-hat), fitted by trimmed total
least squares inside a strip around where it is expected, and every corner is the cross
product of two fitted lines. Court coordinates are line centres: 3.A.2 measures to the
outside edge of 2-inch lines, so each line sits 2.54 cm inside its nominal dimension.
""")

code("""
L, W, K, HW = 13.41, 6.10, 2.13, 0.0254
MID = L / 2
LINES = {
    'near_base': ((HW, 0), (HW, W)), 'side_left': ((0, HW), (L, HW)),
    'side_right': ((0, W - HW), (L, W - HW)), 'near_kitchen': ((MID - K + HW, 0), (MID - K + HW, W)),
    'near_centre': ((0, W / 2), (MID - K, W / 2)),
}
CORNERS = {'NBL': ('near_base', 'side_left'), 'NBR': ('near_base', 'side_right'),
           'NKL': ('near_kitchen', 'side_left'), 'NKR': ('near_kitchen', 'side_right'),
           'NBC': ('near_base', 'near_centre'), 'NKC': ('near_kitchen', 'near_centre')}
FIT = ['NBL', 'NBR', 'NKL', 'NKR']
SEED = np.array([(-15, 585), (985, 960), (655, 500), (1555, 665)], float)  # by eye, ~5 px
NET = np.array([(842, 354), (1735, 442), (1735, 566), (842, 447)], np.int32)

def court_xy(name):
    a, b = (np.cross([*LINES[n][0], 1], [*LINES[n][1], 1]) for n in CORNERS[name])
    x = np.cross(a, b); return x[:2] / x[2]

def paint(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    ridge = cv2.morphologyEx(hsv[..., 2], cv2.MORPH_TOPHAT, np.ones((21, 21), np.uint8))
    m = ((ridge > 40) & (hsv[..., 1] < 90)).astype(np.uint8)
    cv2.fillPoly(m, [NET], 0)
    ys, xs = np.nonzero(m); return np.stack([xs, ys], 1).astype(float)

def fit_line(P):
    for _ in range(4):
        c = P.mean(0); n = np.linalg.svd(P - c, full_matrices=False)[2][1]; r = (P - c) @ n
        P = P[np.abs(r) < max(2.5 * 1.4826 * np.median(np.abs(r)), 1.0)]
    c = P.mean(0); n = np.linalg.svd(P - c, full_matrices=False)[2][1]
    return np.array([n[0], n[1], -n @ c])

def project(H, pts):
    x = np.c_[pts, np.ones(len(pts))] @ H.T; return x[:, :2] / x[:, 2:]

px = paint(plate)
H = cv2.getPerspectiveTransform(np.float32([court_xy(n) for n in FIT]), np.float32(SEED))
for _ in range(3):  # fit, intersect, refit H, look again
    fits = {}
    for name, seg in LINES.items():
        p, q = project(H, np.array(seg)); d = (q - p) / np.linalg.norm(q - p); nrm = np.array([-d[1], d[0]])
        t, s = (px - p) @ d, (px - p) @ nrm
        fits[name] = fit_line(px[(np.abs(s) < 10) & (t > 0) & (t < np.linalg.norm(q - p))])
    corner = {k: (lambda x: x[:2] / x[2])(np.cross(fits[a], fits[b])) for k, (a, b) in CORNERS.items()}
    H = cv2.getPerspectiveTransform(np.float32([court_xy(n) for n in FIT]), np.float32([corner[n] for n in FIT]))
for k, v in corner.items():
    print(k, v.round(2))   # NBL is outside the frame, at x < 0
""")

md("## Homogeneous coordinates (unit 4.1, lesson 1)")

code("""
import numpy as np

nbl, nkl = np.array([-18.52, 587.39, 1.0]), np.array([669.68, 500.09, 1.0])
nbr, nkr = np.array([987.25, 959.91, 1.0]), np.array([1555.24, 663.22, 1.0])

left = np.cross(nbl, nkl)          # the line through two points
right = np.cross(nbr, nkr)
vp = np.cross(left, right)         # the point on two lines
print(left, right, vp[:2] / vp[2]) # ... [2251.8  299.4]
""")

md("### Homogeneous coordinates: how close is this to OpenCV?")

code(r"""
import numpy as np

# four near-court corners, as (x, y, 1)
nbl = np.array([-18.52, 587.39, 1.0])
nkl = np.array([669.68, 500.09, 1.0])
nbr = np.array([987.25, 959.91, 1.0])
nkr = np.array([1555.24, 663.22, 1.0])

left = np.cross(nbl, nkl)    # the line through two points
right = np.cross(nbr, nkr)
vp = np.cross(left, right)   # the point on two lines
print("left sideline:", left.round(2))
print("they meet at", (vp[:2] / vp[2]).round(1))
""")

code(r"""
import cv2

# the school way: two equations a x + b y = -c
A = np.array([left[:2], right[:2]])
b = -np.array([left[2], right[2]])
print("np.linalg.solve:", np.linalg.solve(A, b).round(1))

# OpenCV's division by the last coordinate
p = cv2.convertPointsFromHomogeneous(vp[None, None])
print("cv2.convertPointsFromHomogeneous:", p.ravel().round(1))

# the same two sidelines on the court, in metres
g1, g2 = np.array([0, 1, -0.0254]), np.array([0, 1, -6.0746])
print("on the ground they meet at", np.cross(g1, g2).round(4))
# the school way needs this determinant to be nonzero
d = np.linalg.det(np.array([g1[:2], g2[:2]]))
print("school way on the ground: determinant", d)
""")

md("### Homogeneous coordinates: exercises")

code(r"""
def meet(l1, l2):
    return (0.0, 0.0)  # pixel, or None if they meet at infinity

print(meet(left, right))
""")

code(r"""
# checks your answer
print("correct" if (np.allclose(meet(left, right), (2251.7, 299.4), atol=0.1) and meet(np.array([0, 1, -0.0254]), np.array([0, 1, -6.0746])) is None) else "not yet")
""")

code(r"""
# Exercises 2 and 3: the numbers under "What you should see"
nbc = np.array([332.06, 717.24, 1.0])
nkc = np.array([1034.46, 567.28, 1.0])
centre = np.cross(nbc, nkc)
v2 = np.cross(left, centre)
print("centre line meets the left sideline at",
      (v2[:2] / v2[2]).round(1))
base = np.cross(nbl, nbr)
kitchen = np.cross(nkl, nkr)
v3 = np.cross(base, kitchen)
print("baseline meets kitchen line at",
      (v3[:2] / v3[2]).round(1))
""")

md("## Lesson 1: what a 2×2 does")

code("""
import numpy as np

court = np.array([[0.0254, 0.0254], [0.0254, 6.0746], [4.6004, 0.0254], [4.6004, 6.0746]])  # m
image = np.array([[-18.52, 587.39], [987.25, 959.91], [669.68, 500.09], [1555.24, 663.22]])  # px

c0, i0 = court - court.mean(0), image - image.mean(0)
M = np.linalg.lstsq(c0, i0, rcond=None)[0].T        # image ≈ M @ court, both centred
U, s, Vt = np.linalg.svd(M)
print(M.round(2), np.linalg.det(M).round(0), s.round(1))
print(np.linalg.norm(c0 @ M.T - i0, axis=1).round(1))  # [60.4 60.4 60.4 60.4]
""")

md("### Lesson 1: how close is this to OpenCV?")

code(r"""
import numpy as np
import cv2

# the near-half corners: metres and pixels
court = np.array([[0.0254, 0.0254], [0.0254, 6.0746],
                  [4.6004, 0.0254], [4.6004, 6.0746]])
image = np.array([[-18.52, 587.39], [987.25, 959.91],
                  [669.68, 500.09], [1555.24, 663.22]])

# the best 2x2, with both sets centred
c0 = court - court.mean(0)
i0 = image - image.mean(0)
M = np.linalg.lstsq(c0, i0, rcond=None)[0].T

det = M[0, 0] * M[1, 1] - M[0, 1] * M[1, 0]
sigma = np.linalg.svd(M, compute_uv=False)
print(M.round(2))
print("det:", round(det), " sigma:", sigma.round(1))
""")

code(r"""
# OpenCV moves the points and takes the matrix apart
pts = c0[None].astype(np.float32)
moved = cv2.transform(pts, M.astype(np.float32))[0]
print("max diff, cv2.transform:",
      round(float(np.abs(moved - c0 @ M.T).max()), 4))
print("cv2.determinant:", round(cv2.determinant(M)))
w, u, vt = cv2.SVDecomp(M)
print("cv2.SVDecomp:", w.ravel().round(1))

miss = np.linalg.norm(c0 @ M.T - i0, axis=1)
print("miss at each corner (px):", miss.round(1))
""")

md("### Lesson 1: exercises")

code(r"""
def area_scale(M):
    return 0.0  # pixels per square metre

print(area_scale(M))
""")

code(r"""
# checks your answer
print("correct" if (abs(area_scale(M) - 12639) < 1) else "not yet")
""")

code(r"""
# Exercises 2 and 3: the numbers under "What you should see"
d = np.array([1.0, 0.0])  # along both sidelines
a = M @ d
print("both sidelines, through M:",
      np.degrees(np.arctan2(a[1], a[0])).round(2))
for p, q in ((image[0], image[2]), (image[1], image[3])):
    v = q - p
    print("a sideline in the photo:",
          np.degrees(np.arctan2(v[1], v[0])).round(2))

t = np.radians(30)
R = np.array([[np.cos(t), -np.sin(t)],
              [np.sin(t), np.cos(t)]])
print("det, 30 degree rotation:", round(np.linalg.det(R), 4))
print("det, 2M:", round(np.linalg.det(2 * M)))
""")

md("## Lesson 2: affine and projective")

code("""
import cv2
import numpy as np

court = np.float32([[0.0254, 0.0254], [0.0254, 6.0746], [4.6004, 0.0254], [4.6004, 6.0746]])
image = np.float32([[-18.52, 587.39], [987.25, 959.91], [669.68, 500.09], [1555.24, 663.22]])

A_img2court = cv2.getAffineTransform(image[:3], court[:3])      # 2x3, six numbers
H_img2court = cv2.getPerspectiveTransform(image, court)         # 3x3, eight numbers
nbc = np.float32([[[332.06, 717.24]]])
print(cv2.transform(nbc, A_img2court))                          # [[[0.0252 2.1339]]]
print(cv2.perspectiveTransform(nbc, H_img2court))               # [[[0.0254 3.0547]]]
""")

md("### Lesson 2: how close is this to OpenCV?")

code(r"""
import numpy as np
import cv2

court = np.array([[0.0254, 0.0254], [0.0254, 6.0746],
                  [4.6004, 0.0254], [4.6004, 6.0746]])
image = np.array([[-18.52, 587.39], [987.25, 959.91],
                  [669.68, 500.09], [1555.24, 663.22]])
nbc = np.array([332.06, 717.24, 1.0])

# affine: six unknowns, three pairs, six equations
A, b = np.zeros((6, 6)), np.zeros(6)
for k in range(3):
    (x, y), (u, v) = image[k], court[k]
    A[2*k] = [x, y, 1, 0, 0, 0]
    A[2*k + 1] = [0, 0, 0, x, y, 1]
    b[2*k], b[2*k + 1] = u, v
aff = np.linalg.solve(A, b).reshape(2, 3)

# homography: eight unknowns (the ninth is 1),
# four pairs, eight equations
A, b = np.zeros((8, 8)), np.zeros(8)
for k in range(4):
    (x, y), (u, v) = image[k], court[k]
    A[2*k] = [x, y, 1, 0, 0, 0, -u*x, -u*y]
    A[2*k + 1] = [0, 0, 0, x, y, 1, -v*x, -v*y]
    b[2*k], b[2*k + 1] = u, v
H = np.append(np.linalg.solve(A, b), 1).reshape(3, 3)

q = H @ nbc
print("NBC, affine:", (aff @ nbc).round(4))
print("NBC, homography:", (q[:2] / q[2]).round(4))
""")

code(r"""
ref_a = cv2.getAffineTransform(
    image[:3].astype(np.float32),
    court[:3].astype(np.float32))
ref_h = cv2.getPerspectiveTransform(
    image.astype(np.float32), court.astype(np.float32))
print("max diff, affine:", np.abs(aff - ref_a).max())
print("max diff, homography:", np.abs(H - ref_h).max())

# the corner the affine map never saw
nkr = aff @ np.append(image[3], 1)
off = np.linalg.norm(nkr - court[3]) * 100
print("fourth corner, affine:", nkr.round(2),
      "off by", round(float(off), 1), "cm")
""")

md("### Lesson 2: exercises")

code(r"""
def to_court(H, x, y):
    return (0.0, 0.0)  # metres on the court

print(to_court(H, 332.06, 717.24))
""")

code(r"""
# checks your answer
print("correct" if (np.allclose(to_court(H, 332.06, 717.24), (0.0254, 3.0547), atol=1e-3)) else "not yet")
""")

code(r"""
# Exercises 2 and 3: the numbers under "What you should see"
flat = H.copy()
flat[2] = [0, 0, 1]  # drop the bottom row
print("NBC with the bottom row 0, 0, 1:",
      (flat @ nbc)[:2].round(3))

nkc = np.array([1034.46, 567.28, 1.0])
nkc_m = np.array([4.6004, 3.05])
q = H @ nkc
for name, p in (("affine", aff @ nkc),
                ("homography", q[:2] / q[2])):
    e = np.linalg.norm(p - nkc_m) * 100
    print(name, "NKC off by", round(float(e), 1), "cm")
""")

md("## Lesson 3: the DLT")

code("""
import cv2
import numpy as np

def dlt(src, dst):
    \"\"\"Homography with dst ~ H @ src, from four or more (N, 2) point pairs.\"\"\"
    rows = []
    for (x, y), (u, v) in zip(src, dst):
        rows.append([-x, -y, -1, 0, 0, 0, u * x, u * y, u])
        rows.append([0, 0, 0, -x, -y, -1, v * x, v * y, v])
    _, _, vt = np.linalg.svd(np.asarray(rows))
    H = vt[-1].reshape(3, 3)             # the direction A shrinks the most
    return H / H[2, 2]

image = np.array([[-18.52, 587.39], [987.25, 959.91], [669.68, 500.09], [1555.24, 663.22]])
court = np.array([[0.0254, 0.0254], [0.0254, 6.0746], [4.6004, 0.0254], [4.6004, 6.0746]])
H_img2court = dlt(image, court)
print(np.abs(H_img2court - cv2.getPerspectiveTransform(image.astype(np.float32),
                                                       court.astype(np.float32))).max())
""")

code("""
# The condition numbers the lesson quotes: sigma_1 / sigma_8, raw and normalised.
def normaliser(p):
    c = p.mean(0); s = np.sqrt(2) / np.linalg.norm(p - c, axis=1).mean()
    return np.array([[s, 0, -s * c[0]], [0, s, -s * c[1]], [0, 0, 1]])

def rows(src, dst):
    r = []
    for (x, y), (u, v) in zip(src, dst):
        r += [[-x, -y, -1, 0, 0, 0, u * x, u * y, u], [0, 0, 0, -x, -y, -1, v * x, v * y, v]]
    return np.array(r)

def apply(T, p):
    x = np.c_[p, np.ones(len(p))] @ T.T; return x[:, :2] / x[:, 2:]

s_raw = np.linalg.svd(rows(image, court), compute_uv=False)
s_n = np.linalg.svd(rows(apply(normaliser(image), image), apply(normaliser(court), court)), compute_uv=False)
print('raw', s_raw.round(3), '-> sigma1/sigma8 =', round(s_raw[0] / s_raw[7], 1))   # 53,466
print('normalised sigma1/sigma8 =', round(s_n[0] / s_n[7], 2))                         # 6.4
""")

md("### Lesson 3: how close is this to OpenCV?")

code(r"""
import numpy as np
import cv2

image = np.array([[-18.52, 587.39], [987.25, 959.91],
                  [669.68, 500.09], [1555.24, 663.22]])
court = np.array([[0.0254, 0.0254], [0.0254, 6.0746],
                  [4.6004, 0.0254], [4.6004, 6.0746]])

def pair_rows(x, y, u, v):
    # the two equations one pair gives
    return [[-x, -y, -1, 0, 0, 0, u*x, u*y, u],
            [0, 0, 0, -x, -y, -1, v*x, v*y, v]]

def dlt(src, dst):
    A = np.array([r for (x, y), (u, v) in zip(src, dst)
                  for r in pair_rows(x, y, u, v)])
    _, s, vt = np.linalg.svd(A)
    H = vt[-1].reshape(3, 3)  # what A shrinks most
    return H / H[2, 2], s[0] / s[7]  # sigma1/sigma8

def normaliser(p):
    # centroid to 0, mean distance sqrt(2)
    c = p.mean(0)
    k = np.sqrt(2) / np.linalg.norm(p - c, axis=1).mean()
    return np.array([[k, 0, -k*c[0]],
                     [0, k, -k*c[1]], [0, 0, 1]])

def move(T, p):
    q = np.c_[p, np.ones(len(p))] @ T.T
    return q[:, :2] / q[:, 2:]

def dlt_norm(src, dst):
    Ts, Td = normaliser(src), normaliser(dst)
    Hn, cond = dlt(move(Ts, src), move(Td, dst))
    H = np.linalg.inv(Td) @ Hn @ Ts
    return H / H[2, 2], cond

H, cond_raw = dlt(image, court)
Hn, cond_norm = dlt_norm(image, court)
print("sigma1/sigma8, raw:", f"{cond_raw:,.0f}")
print("sigma1/sigma8, normalised:", round(cond_norm, 1))
print("the two answers differ by", np.abs(H - Hn).max())
""")

code(r"""
ref = cv2.getPerspectiveTransform(
    image.astype(np.float32), court.astype(np.float32))
print("max diff, getPerspectiveTransform:",
      np.abs(H - ref).max())
ref2, _ = cv2.findHomography(image, court, 0)
print("max diff, findHomography:", np.abs(H - ref2).max())

def to_court(H, x, y):
    u, v, w = H @ np.array([x, y, 1.0])
    return np.array([u / w, v / w])

gap = np.linalg.norm(to_court(H, 332.06, 717.24)
                     - to_court(ref, 332.06, 717.24))
print("NBC, ours against OpenCV (mm):", gap * 1000)
""")

md("### Lesson 3: exercises")

code(r"""
def two_rows(x, y, u, v):
    return [[0] * 9, [0] * 9]  # the two rows of A

print(np.round(two_rows(-18.52, 587.39, 0.0254, 0.0254), 3))
""")

code(r"""
# checks your answer
print("correct" if (np.allclose(two_rows(-18.52, 587.39, 0.0254, 0.0254), pair_rows(-18.52, 587.39, 0.0254, 0.0254))) else "not yet")
""")

code(r"""
# Exercises 2 and 3: the numbers under "What you should see"
img6 = np.vstack([image, [[332.06, 717.24],
                          [1034.46, 567.28]]])
crt6 = np.vstack([court, [[0.0254, 3.05],
                          [4.6004, 3.05]]])
print("six pairs, sigma1/sigma8:",
      f"{dlt(img6, crt6)[1]:,.0f}",
      "raw,", round(dlt_norm(img6, crt6)[1], 2),
      "normalised")

bad = img6.copy()
bad[5] = image[3]  # NKR's pixel, NKC's position
for name, (a, b) in (("six good", (img6, crt6)),
                     ("one wrong", (bad, crt6))):
    for how, fit in (("normalised", dlt_norm),
                     ("raw", dlt)):
        M = fit(a, b)[0]
        e = np.linalg.norm(to_court(M, 332.06, 717.24)
                           - [0.0254, 3.05]) * 100
        print(f"{name:9} {how:10} NBC off by "
              f"{e:.2f} cm")
""")

md("## Lesson 5: warping and blending")

code("""
import cv2
import numpy as np

frame = cv2.imread("frame_045000.png")
H_court2img = np.array([[214.196342, 90.15869, -26.259561],
                        [28.479672, -12.463675, 587.254072],
                        [0.095124, -0.077166, 1.0]])
s, m = 50, 1.5                                   # 50 px per metre, 1.5 m margin
H_court2top = np.array([[0, s, m * s], [-s, 0, (13.41 + m) * s], [0, 0, 1.0]])
H_img2top = H_court2top @ np.linalg.inv(H_court2img)

top = cv2.warpPerspective(frame, H_img2top, (455, 820), flags=cv2.INTER_LINEAR)
cv2.imwrite("topview.png", top)                  # backward mapping inside
""")

code("""
# Forward mapping, for contrast: push every frame pixel through H and count what is never written.
ys, xs = np.mgrid[0:frame.shape[0], 0:frame.shape[1]]
dst = np.c_[xs.ravel(), ys.ravel(), np.ones(xs.size)] @ H_img2top.T
dx, dy = np.rint(dst[:, 0] / dst[:, 2]).astype(int), np.rint(dst[:, 1] / dst[:, 2]).astype(int)
ok = (dx >= 0) & (dx < 455) & (dy >= 0) & (dy < 820)
hit = np.zeros((820, 455), int); np.add.at(hit, (dy[ok], dx[ok]), 1)
seen = cv2.warpPerspective(np.ones(frame.shape[:2], np.uint8), H_img2top, (455, 820)) > 0
court_top = np.zeros((820, 455), np.uint8)
cv2.fillPoly(court_top, [np.int32([[75, 745], [380, 745], [380, 75], [75, 75]])], 1)
vis = seen & (court_top > 0)
far = vis.copy(); far[410:] = False
print('holes, far half of the court:', round(((hit == 0) & far).sum() / far.sum(), 3))   # ~0.56
""")

code("""
# Blending by median: 31 frames of a fixed camera, players removed.
med = np.median(np.stack(small), axis=0).astype(np.uint8)
single = small[len(small) // 2]
print('share of the 960 px frame differing from the median by > 40 in some channel:',
      round((np.abs(single.astype(int) - med.astype(int)).max(axis=2) > 40).mean(), 3))
cv2.imwrite('median960.png', med)
""")


md("### Lesson 5: how close is this to OpenCV?")

code(r"""
import numpy as np
import cv2

frame = cv2.imread("frame_045000.webp")
H_court2img = np.array(
    [[214.196342, 90.15869, -26.259561],
     [28.479672, -12.463675, 587.254072],
     [0.095124, -0.077166, 1.0]])
s, m = 50, 1.5  # 50 px per metre, 1.5 m margin
H_court2top = np.array([[0, s, m * s],
                        [-s, 0, (13.41 + m) * s],
                        [0, 0, 1.0]])
H_img2top = H_court2top @ np.linalg.inv(H_court2img)

# pull: every top-view pixel back into the photo
w, h = 455, 820
xs, ys = np.meshgrid(np.arange(w), np.arange(h))
p = np.stack([xs.ravel(), ys.ravel(), np.ones(w * h)])
q = np.linalg.inv(H_img2top) @ p
u, v = q[0] / q[2], q[1] / q[2]

def bilinear(img, u, v):
    x0 = np.floor(u).astype(int)
    y0 = np.floor(v).astype(int)
    fx, fy = (u - x0)[:, None], (v - y0)[:, None]
    ok = ((x0 >= 0) & (y0 >= 0)
          & (x0 < img.shape[1] - 1)
          & (y0 < img.shape[0] - 1))
    x0 = np.clip(x0, 0, img.shape[1] - 2)
    y0 = np.clip(y0, 0, img.shape[0] - 2)
    a = img[y0, x0] * (1 - fx) + img[y0, x0 + 1] * fx
    b = (img[y0 + 1, x0] * (1 - fx)
         + img[y0 + 1, x0 + 1] * fx)
    out = a * (1 - fy) + b * fy
    out[~ok] = 0
    return out

ours = bilinear(frame.astype(np.float32), u, v)
ours = ours.reshape(h, w, 3)
show(ours.clip(0, 255).astype(np.uint8), "pulled back")
print("top view:", ours.shape)
""")

code(r"""
ref = cv2.warpPerspective(frame, H_img2top, (w, h),
                          flags=cv2.INTER_LINEAR)
inside = ours.sum(2) > 0
d = np.abs(np.rint(ours) - ref)[inside]
print("pixels compared:", int(inside.sum()))
print("mean diff:", round(float(d.mean()), 3),
      " max:", d.max())
""")

md("### Lesson 5: exercises")

code(r"""
grey = frame.astype(np.float64).mean(axis=2)

def sample(img, u, v):
    return 0.0  # the bilinear value at (u, v)

print(sample(grey, 485.259, 687.625))
""")

code(r"""
# checks your answer
print("correct" if (abs(sample(grey, 485.259, 687.625) - 140.5) < 0.1) else "not yet")
""")

code(r"""
# Exercises 2 and 3: the numbers under "What you should see"
grey = frame.astype(np.float64).mean(axis=2)
print("nearest:", round(grey[688, 485], 1))

# push every photo pixel into the top view instead
yy, xx = np.mgrid[0:frame.shape[0], 0:frame.shape[1]]
P = np.stack([xx.ravel(), yy.ravel(),
              np.ones(xx.size)])
Q = H_img2top @ P
tx = np.rint(Q[0] / Q[2]).astype(int)
ty = np.rint(Q[1] / Q[2]).astype(int)
k = (tx >= 0) & (ty >= 0) & (tx < w) & (ty < h)
hit = np.zeros((h, w), bool)
hit[ty[k], tx[k]] = True
court = np.zeros((h, w), bool)
court[75:746, 75:380] = True   # the court, 50 px/m
far = court & inside
far[h // 2:] = False           # the far half
print("far half never written:",
      round(float((~hit[far]).mean()), 3))
""")


@click.command()
def align_notebook() -> None:
    """Write notebooks/image_alignment.ipynb, outputs cleared."""
    nb = {
        "cells": [
            {
                "cell_type": t,
                "metadata": {},
                "source": src.splitlines(keepends=True),
                **({"execution_count": None, "outputs": []} if t == "code" else {}),
            }
            for t, src in CELLS
        ],
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.12"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    out = PROJECT_ROOT / "notebooks" / "image_alignment.ipynb"
    out.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n")
    print(f"wrote {out} ({len(CELLS)} cells)")


if __name__ == "__main__":
    align_notebook()
