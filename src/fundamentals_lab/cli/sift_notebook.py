# ruff: noqa: E501  (cell sources mirror the posts verbatim, line lengths included)
"""Build notebooks/sift.ipynb from the code blocks the unit 3.3 posts print.

`uv run sift-notebook`. The cells that mirror a post are copied verbatim from it; keep them in
step when a post changes. The notebook is written with its outputs cleared.
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
# SIFT, measured

Companion notebook for the CondadosAI Fundamentals unit **SIFT** (F3 / unit 3.3): what makes a
good keypoint, scale space and the difference of Gaussians, how SIFT finds keypoints, the SIFT
descriptor, and Lowe's ratio test. Every Python block printed in those posts has a cell here, and
each lesson's exercises have a cell that computes the answers the post gives.

| Used for | Source | Licence |
|---|---|---|
| every lesson | "BarcelonaHarbour1.jpg" and "BarcelonaHarbour2.jpg" by Pap3rinik (Wikimedia Commons) at 960×723, as condados.ai serves them, and the reference homography the unit scores matches against (fitted with ORB) | public domain |

The tables in the posts come from `uv run sift-experiments` in the repository, which reads the
original JPEGs (`uv run align-download`) and templeRing (`uv run edge-download`); the cells here
read the WebP copies the site serves, so their counts can differ slightly.
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
BASE = os.environ.get('SIFT_BASE', 'https://condados.ai')
UA = {'User-Agent': 'fundamentals-lab/0.1 (+https://github.com/CondadosAI/fundamentals-lab)'}
FILES = ['/blog/sift-what-makes-a-point-matchable/barcelona-a.webp', '/blog/sift-what-makes-a-point-matchable/barcelona-b.webp', '/blog/sift-what-makes-a-point-matchable/reference-homography.json']
for f in FILES:
    name = f.rsplit('/', 1)[1]
    if not Path(name).exists():
        req = urllib.request.Request(BASE + f, headers=UA)
        Path(name).write_bytes(urllib.request.urlopen(req).read())
print('OpenCV', cv2.__version__, '| NumPy', np.__version__, '|', cv2.imread('barcelona-a.webp').shape)
""")

md(r"""
## 1. SIFT, the whole pipeline (hub)

The cells of [the post](https://condados.ai/blog/sift-what-makes-a-point-matchable), in order.
""")

md(r"""
The post's summary snippet: `img_a` and `img_b` are the two photos.
""")

code(r"""
img_a = cv2.imread('barcelona-a.webp', cv2.IMREAD_GRAYSCALE)
img_b = cv2.imread('barcelona-b.webp', cv2.IMREAD_GRAYSCALE)
import cv2

sift = cv2.SIFT_create()
kp_a, desc_a = sift.detectAndCompute(img_a, None)  # steps 1-5
kp_b, desc_b = sift.detectAndCompute(img_b, None)

knn = cv2.BFMatcher(cv2.NORM_L2).knnMatch(desc_a, desc_b, k=2)
good = [m for m, n in knn if m.distance < 0.75 * n.distance]  # step 6
""")

md(r"""
## 2. What makes a good keypoint?

The cells of [the post](https://condados.ai/blog/what-makes-a-good-keypoint), in order.
""")

code(r"""
import numpy as np
import cv2

img = cv2.imread("barcelona-a.webp", cv2.IMREAD_GRAYSCALE)
h, w = img.shape

def rotate(im, deg, zoom=1.0):
    M = cv2.getRotationMatrix2D((w / 2, h / 2), deg, zoom)
    c, s = abs(M[0, 0]), abs(M[0, 1])
    W2, H2 = round(h * s + w * c), round(h * c + w * s)
    M[0, 2] += W2 / 2 - w / 2
    M[1, 2] += H2 / 2 - h / 2
    return cv2.warpAffine(im, M, (W2, H2)), M

sift = cv2.SIFT_create()
print(img.shape)
""")

code(r"""
ZOOM = 0.5
small, Mz = rotate(img, 0, ZOOM)

def harris(im, n=1000, size=8.0):
    c = cv2.goodFeaturesToTrack(im, n, 1e-4, 3, useHarrisDetector=True,
                                k=0.04, blockSize=5)
    return [cv2.KeyPoint(float(x), float(y), size) for x, y in c.reshape(-1, 2)]

def sift_points(im, n=1000):
    kps = sorted(sift.detect(im, None), key=lambda k: -k.response)
    seen, out = set(), []
    for k in kps:
        key = (round(k.pt[0], 2), round(k.pt[1], 2), round(k.size, 2))
        if key not in seen:
            seen.add(key)
            k.angle = 0  # upright: this cell tests size, not direction
            out.append(k)
    return out[:n]

def correct_matches(ka, kb):
    ka, da = sift.compute(img, ka)
    kb, db = sift.compute(small, kb)
    good = [m for m, n in cv2.BFMatcher().knnMatch(da, db, k=2)
            if m.distance < 0.75 * n.distance]
    p = np.array([ka[m.queryIdx].pt for m in good]).reshape(-1, 2)
    q = np.array([kb[m.trainIdx].pt for m in good]).reshape(-1, 2)
    err = np.linalg.norm(p @ Mz[:, :2].T + Mz[:, 2] - q, axis=1)
    return len(good), int((err <= 3).sum())

print("Harris at 8 px:  matches, correct =",
      correct_matches(harris(img), harris(small)))
print("SIFT, own size:  matches, correct =",
      correct_matches(sift_points(img), sift_points(small)))
""")

code(
    'rot, M = rotate(img, 45)\n\ndef strongest(im, n=1000):\n    """The n strongest SIFT points, one per position."""\n    seen, pts = set(), []\n    kps = sorted(sift.detect(im, None), key=lambda k: -k.response)\n    for k in kps:\n        key = (round(k.pt[0], 2), round(k.pt[1], 2))\n        if key not in seen:\n            seen.add(key)\n            pts.append(k.pt)\n    return np.array(pts[:n])\n\npa, pb = strongest(img), strongest(rot)\nqa = pa @ M[:, :2].T + M[:, 2]  # where A\'s points should land\n\ndef inside(p, shape, b=8):\n    return ((p[:, 0] >= b) & (p[:, 0] < shape[1] - b)\n            & (p[:, 1] >= b) & (p[:, 1] < shape[0] - b))\n\nMi = cv2.invertAffineTransform(M)\nback = pb @ Mi[:, :2].T + Mi[:, 2]\nqa, pb = qa[inside(qa, rot.shape)], pb[inside(back, img.shape)]\n\nd = np.linalg.norm(qa[:, None] - pb[None], axis=2)\nused, hits = set(), 0\nfor i in np.argsort(d.min(axis=1)):\n    for j in np.argsort(d[i])[:3]:\n        if d[i, j] > 1.5:\n            break\n        if j not in used:\n            used.add(j)\n            hits += 1\n            break\nna, nb = len(qa), len(pb)\nprint(f"{hits} repeated of min({na}, {nb})")\nprint(f"repeatability {hits / min(na, nb):.3f}")'
)

code(r"""
bf = cv2.BFMatcher(cv2.NORM_L2)
m = bf.match(qa.astype(np.float32), pb.astype(np.float32))
d_cv = np.array([x.distance for x in m])
print("largest difference in nearest distance:",
      f"{np.abs(d_cv - d.min(axis=1)).max():.2e} px")
print("points with a neighbour within 1.5 px:",
      int((d_cv <= 1.5).sum()), " one-to-one:", hits)
""")

md(r"""
Exercise `pairs`: the starter as printed, then a solution.
""")

code(r"""
import numpy as np

def one_to_one(d, tol):
    return int((d.min(axis=1) <= tol).sum())  # credits shared partners

print(one_to_one(np.array([[0.5, 3.0], [0.8, 2.0]]), 1.5))
print(one_to_one(np.array([[0.5, 3.0], [3.0, 0.8]]), 1.5))
""")

code(r"""
# Solution
def one_to_one(d, tol):
    used, n = set(), 0
    for i in np.argsort(d.min(axis=1)):
        for j in np.argsort(d[i]):
            if d[i, j] > tol:
                break
            if j not in used:
                used.add(j)
                n += 1
                break
    return n

print(one_to_one(np.array([[0.5, 3.0], [0.8, 2.0]]), 1.5))
print(one_to_one(np.array([[0.5, 3.0], [3.0, 0.8]]), 1.5))
print(one_to_one(np.array([[4.0]]), 1.5))
""")

md(r"""
## 3. Scale space and the difference of Gaussians

The cells of [the post](https://condados.ai/blog/scale-space-difference-of-gaussians), in order.
""")

code(
    'import numpy as np\nimport cv2\n\nimg = cv2.imread("barcelona-a.webp", cv2.IMREAD_GRAYSCALE)\nimg = img.astype(np.float32)\n\ndef gauss_kernel(sigma):\n    r = int(np.ceil(3 * sigma))\n    x = np.arange(-r, r + 1)\n    g = np.exp(-x**2 / (2 * sigma**2))\n    return (g / g.sum()).astype(np.float32)\n\ndef blur(im, sigma):\n    """Separable Gaussian: rows, then columns, mirrored borders."""\n    g = gauss_kernel(sigma)\n    r = len(g) // 2\n    p = np.pad(im, r, mode="reflect")\n    rows = sum(g[i] * p[:, i:i + im.shape[1]] for i in range(len(g)))\n    return sum(g[i] * rows[i:i + im.shape[0], :] for i in range(len(g)))\n\nsigma = 2.54\nmine = blur(img, sigma)\nref = cv2.GaussianBlur(img, (0, 0), sigma, borderType=cv2.BORDER_REFLECT_101)\nk = 2 * int(np.ceil(3 * sigma)) + 1\nref_same = cv2.GaussianBlur(img, (k, k), sigma, borderType=cv2.BORDER_REFLECT_101)\nprint("largest difference, same kernel width:",\n      f"{np.abs(mine - ref_same).max():.4f} grey levels")\nprint("largest difference, OpenCV\'s own width:",\n      f"{np.abs(mine - ref).max():.4f} grey levels")'
)

code(r"""
k = 2 ** (1 / 3)  # three levels per octave
d = blur(img, k * sigma) - blur(img, sigma)
lap = cv2.Laplacian(blur(img, sigma), cv2.CV_32F, ksize=1)
approx = (k - 1) * sigma**2 * lap

m = (slice(20, -20), slice(20, -20))  # leave the borders out
a, b = d[m].ravel(), approx[m].ravel()
print(f"k = {k:.4f}, sigma = {sigma}")
print(f"correlation {np.corrcoef(a, b)[0, 1]:.4f}")
print(f"relative residual {np.linalg.norm(a - b) / np.linalg.norm(a):.3f}")
""")

code(r"""
sigmas = np.geomspace(1, 40, 200)

def disc_response(r, s):
    disc = np.zeros((257, 257), np.float32)
    cv2.circle(disc, (128, 128), r, 1.0, -1, lineType=cv2.LINE_AA)
    g = cv2.GaussianBlur(disc, (0, 0), s)
    return abs(s**2 * cv2.Laplacian(g, cv2.CV_32F, ksize=1)[128, 128])

for r in (4, 8, 16):
    resp = [disc_response(r, s) for s in sigmas]
    best = sigmas[int(np.argmax(resp))]
    fixed = disc_response(r, 5.61) / max(resp)
    print(f"radius {r:2d}: best sigma {best:5.2f}, r/sqrt(2) "
          f"{r / np.sqrt(2):5.2f}, at sigma 5.61: {fixed:.0%} of its best")
""")

md(r"""
Exercise `octaves`: the starter as printed, then a solution.
""")

code(r"""
import numpy as np

def n_octaves(w, h):
    return 0  # round(log2(2 * smaller side) - 2) + 1

print(n_octaves(960, 723), n_octaves(640, 480), n_octaves(4000, 3000))
""")

code(r"""
# Solution
def n_octaves(w, h):
    return int(round(np.log2(2 * min(w, h)) - 2)) + 1

print(n_octaves(960, 723), n_octaves(640, 480), n_octaves(4000, 3000))
""")

md(r"""
## 4. How SIFT finds keypoints

The cells of [the post](https://condados.ai/blog/sift-keypoint-detection), in order.
""")

code(r"""
import numpy as np
import cv2

img = cv2.imread("barcelona-a.webp", cv2.IMREAD_GRAYSCALE)
g = img.astype(np.float32)

# SIFT doubles the photo first (assumed blur 0.5 becomes 1.0)
# and blurs it to 1.6; each level is 2^(1/3) blurrier.
k = 2 ** (1 / 3)
sig = [1.6 * k ** i for i in range(6)]
big = cv2.resize(g, None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)
big = cv2.GaussianBlur(big, (0, 0), np.sqrt(1.6**2 - 1.0**2))
for i in (1, 2, 3):  # up to twice the start blur ...
    big = cv2.GaussianBlur(big, (0, 0), np.sqrt(sig[i]**2 - sig[i-1]**2))
levels = [big[::2, ::2]]  # ... then halve: the octave at the photo's size
for i in range(1, 6):
    step = np.sqrt(sig[i]**2 - sig[i - 1]**2)
    levels.append(cv2.GaussianBlur(levels[-1], (0, 0), step))
dog = [levels[i + 1] - levels[i] for i in range(5)]

# A point is kept if it beats (or ties) its 26 neighbours
# in x, y and scale, and is more than 1 grey level from 0.
box = np.ones((3, 3), np.uint8)
pts = []
for i in (1, 2, 3):
    cube = np.stack(dog[i - 1:i + 2])
    hi = cv2.dilate(cube.max(axis=0), box)
    lo = cv2.erode(cube.min(axis=0), box)
    v = dog[i]
    hit = ((v > 1) & (v >= hi)) | ((v < -1) & (v <= lo))
    hit[:5] = hit[-5:] = False
    hit[:, :5] = hit[:, -5:] = False
    for r, c in zip(*np.nonzero(hit)):
        pts.append((i, r, c))
print("octave size:", levels[0].shape[::-1])
print("extrema in this octave:", len(pts))
""")

code(r"""
# Contrast and edge tests at the integer point (no sub-pixel fit).
keep = 0
weak = edge = 0
for i, r, c in pts:
    d = dog[i]
    if abs(d[r, c]) / 255 * 3 < 0.04:   # contrast test
        weak += 1
        continue
    dxx = d[r, c + 1] + d[r, c - 1] - 2 * d[r, c]
    dyy = d[r + 1, c] + d[r - 1, c] - 2 * d[r, c]
    dxy = (d[r + 1, c + 1] - d[r + 1, c - 1]
           - d[r - 1, c + 1] + d[r - 1, c - 1]) / 4
    tr, det = dxx + dyy, dxx * dyy - dxy ** 2
    if det <= 0 or tr * tr / det >= 11 ** 2 / 10:  # edge test, r = 10
        edge += 1
        continue
    keep += 1
print("too weak:", weak, " edge-like:", edge, " kept:", keep)
""")

code(r"""
# How close: OpenCV's keypoints found at this resolution (octave 0)
# against our raw extrema of the same octave.
kps = cv2.SIFT_create().detect(img, None)
octave = [((kp.octave & 255) ^ 128) - 128 for kp in kps]
mine = np.array([(c, r) for i, r, c in pts], np.float32)
theirs = np.array([kp.pt for kp, o in zip(kps, octave) if o == 0],
                  np.float32)
m = cv2.BFMatcher(cv2.NORM_L2).match(theirs, mine)
d = np.array([x.distance for x in m])
print("OpenCV keypoints in octave 0:", len(theirs))
print("within 1 px of one of ours:", f"{(d <= 1).mean():.1%}")
print("within 2 px:", f"{(d <= 2).mean():.1%}")
print("largest distance:", f"{d.max():.2f} px")
""")

code(r"""
# OpenCV's route: switch each test off and count what it removes.
def count(**kw):
    return len(cv2.SIFT_create(**kw).detect(img, None))

both = count()
no_contrast = count(contrastThreshold=1e-9)
no_edge = count(edgeThreshold=1e9)
neither = count(contrastThreshold=1e-9, edgeThreshold=1e9)
print("both tests off:", neither)
print("contrast test only:", no_edge)
print("edge test only:", no_contrast)
print("both on (default):", both)
""")

md(r"""
Exercise `edge`: the starter as printed, then a solution.
""")

code(r"""
def edge_ok(dxx, dyy, dxy, r=10):
    return True  # replace with the test

print(edge_ok(5.2592, 19.0113, 5.2267))   # the compact spot
print(edge_ok(1.0634, 5.0496, -1.6645))   # the straight edge
print(edge_ok(3.0, -2.0, 0.0))       # a saddle
""")

code(r"""
# Solution
def edge_ok(dxx, dyy, dxy, r=10):
    tr, det = dxx + dyy, dxx * dyy - dxy**2
    return bool(det > 0 and tr * tr / det < (r + 1) ** 2 / r)

print(edge_ok(5.2592, 19.0113, 5.2267), edge_ok(1.0634, 5.0496, -1.6645), edge_ok(3.0, -2.0, 0.0))
""")

md(r"""
## 5. The SIFT descriptor

The cells of [the post](https://condados.ai/blog/sift-descriptor), in order.
""")

code(
    'import numpy as np\nimport cv2\n\nimg = cv2.imread("barcelona-a.webp", cv2.IMREAD_GRAYSCALE)\nsift = cv2.SIFT_create()\nkps = sift.detect(img, None)\n# a mid-sized keypoint, so its window is a few dozen pixels wide\nkp = max((k for k in kps if 8 < k.size < 12), key=lambda k: k.response)\nprint(f"keypoint at ({kp.pt[0]:.1f}, {kp.pt[1]:.1f}),",\n      f"size {kp.size:.1f} px, angle {kp.angle:.1f}°")\n\ndef describe(gray, kp, normalize=True):\n    """4x4 cells x 8 directions around kp, rotated by kp.angle."""\n    sigma = kp.size / 2               # the keypoint\'s blur, in pixels\n    cell = 3 * sigma                  # one cell of the 4x4 grid\n    L = cv2.GaussianBlur(gray.astype(np.float32), (0, 0), sigma)\n    gx = cv2.Sobel(L, cv2.CV_32F, 1, 0, ksize=1)\n    gy = -cv2.Sobel(L, cv2.CV_32F, 0, 1, ksize=1)  # y up\n    t = np.radians(360 - kp.angle)    # the keypoint\'s direction\n    # sample points on a 16 x 16 grid in the keypoint\'s frame\n    u = (np.arange(16) - 7.5) / 4     # in cells, -1.875 .. 1.875\n    U, V = np.meshgrid(u, u)\n    x = kp.pt[0] + cell * (U * np.cos(t) + V * np.sin(t))\n    y = kp.pt[1] + cell * (U * np.sin(-t) + V * np.cos(t))\n    sx = cv2.remap(gx, x.astype(np.float32), y.astype(np.float32),\n                   cv2.INTER_LINEAR)\n    sy = cv2.remap(gy, x.astype(np.float32), y.astype(np.float32),\n                   cv2.INTER_LINEAR)\n    mag = np.hypot(sx, sy) * np.exp(-(U**2 + V**2) / 8)\n    ang = (np.degrees(np.arctan2(sy, sx)) - np.degrees(t)) % 360\n    hist = np.zeros((4, 4, 8))\n    o = ang / 45                      # bins centred on 0, 45, 90, ...\n    b0 = np.floor(o).astype(int)\n    w1 = o - b0                       # split between the two nearest\n    for i in range(16):\n        for j in range(16):\n            hist[i // 4, j // 4, b0[i, j] % 8] += mag[i, j] * (1 - w1[i, j])\n            hist[i // 4, j // 4, (b0[i, j] + 1) % 8] += mag[i, j] * w1[i, j]\n    v = hist.ravel()\n    if not normalize:\n        return v\n    v = v / np.linalg.norm(v)\n    v = np.minimum(v, 0.2)            # no single gradient dominates\n    v = v / np.linalg.norm(v)\n    return np.clip(np.round(v * 512), 0, 255)\n\nmine = describe(img, kp)\n_, theirs = sift.compute(img, [kp])\ntheirs = theirs[0]\ncos = mine @ theirs / np.linalg.norm(mine) / np.linalg.norm(theirs)\nprint("numbers:", mine.size, " cosine with OpenCV\'s:", round(cos, 3))\nprint("largest difference:", int(np.abs(mine - theirs).max()), "of 255")\nprint("ours  [:8]", mine[:8].astype(int))\nprint("OpenCV[:8]", theirs[:8].astype(int))'
)

code(
    '# Photo A against a copy of itself turned by 45 degrees.\ndef turned(gray, deg):\n    h, w = gray.shape\n    M = cv2.getRotationMatrix2D((w / 2, h / 2), deg, 1.0)\n    c, s = abs(M[0, 0]), abs(M[0, 1])\n    W, H = int(h * s + w * c), int(h * c + w * s)\n    M[0, 2] += W / 2 - w / 2\n    M[1, 2] += H / 2 - h / 2\n    return cv2.warpAffine(gray, M, (W, H)), M\n\ndef points(gray, n=500):\n    """The n strongest keypoints, one per place (no orientation copies)."""\n    seen, out = set(), []\n    for k in sorted(sift.detect(gray, None), key=lambda k: -k.response):\n        key = (round(k.pt[0], 1), round(k.pt[1], 1))\n        if key not in seen:\n            seen.add(key)\n            out.append(k)\n    return out[:n]\n\ndef correct_matches(deg, upright):\n    img2, M = turned(img, deg)\n    ka, kb = points(img), points(img2)\n    if upright:                      # throw the direction away\n        for k in ka + kb:\n            k.angle = 0.0\n    ka, da = sift.compute(img, ka)\n    kb, db = sift.compute(img2, kb)\n    good = [m for m, n in cv2.BFMatcher().knnMatch(da, db, k=2)\n            if m.distance < 0.75 * n.distance]\n    p = np.array([ka[m.queryIdx].pt for m in good]).reshape(-1, 2)\n    q = np.array([kb[m.trainIdx].pt for m in good]).reshape(-1, 2)\n    where = p @ M[:, :2].T + M[:, 2]   # where each point went\n    ok = np.linalg.norm(where - q, axis=1) <= 3\n    return int(ok.sum()), len(good)\n\nfor deg in (45,):\n    print(f"{deg}°  oriented: %d of %d correct" % correct_matches(deg, False))\n    print(f"{deg}°  upright:  %d of %d correct" % correct_matches(deg, True))'
)

md(r"""
Exercise `norm`: the starter as printed, then a solution.
""")

code(r"""
import numpy as np

def normalize(v):
    return v  # unit length, clip at 0.2, unit length again

v = np.r_[10.0, np.ones(127)]
print(normalize(v)[:3])
print(normalize(2 * v)[:3])
""")

code(r"""
# Solution
def normalize(v):
    v = v / np.linalg.norm(v)
    v = np.minimum(v, 0.2)
    return v / np.linalg.norm(v)

v = np.r_[10.0, np.ones(127)]
print(normalize(v)[:3])
print(normalize(2 * v)[:3])
""")

code(r"""
dark = (img * 0.5).astype(np.uint8)
for norm in (False, True):
    a, b = describe(img, kp, norm), describe(dark, kp, norm)
    print("normalize", norm,
          "distance:", round(float(np.linalg.norm(a - b)), 1),
          "length:", round(float(np.linalg.norm(a)), 1))
""")

md(r"""
## 6. Lowe's ratio test

The cells of [the post](https://condados.ai/blog/lowe-ratio-test), in order.
""")

code(r"""
import numpy as np
import cv2

a = cv2.imread("barcelona-a.webp", cv2.IMREAD_GRAYSCALE)
b = cv2.imread("barcelona-b.webp", cv2.IMREAD_GRAYSCALE)
sift = cv2.SIFT_create()
ka, da = sift.detectAndCompute(a, None)
kb, db = sift.detectAndCompute(b, None)
print("points:", len(ka), len(kb))

# squared distances, all pairs: |a|^2 + |b|^2 - 2 a.b
d2 = ((da**2).sum(1)[:, None] + (db**2).sum(1)[None]
      - 2 * da @ db.T)
d = np.sqrt(np.maximum(d2, 0))
order = np.argsort(d, axis=1)[:, :2]
rows = np.arange(len(da))
d1_np = d[rows, order[:, 0]]
d2_np = d[rows, order[:, 1]]

knn = cv2.BFMatcher(cv2.NORM_L2).knnMatch(da, db, k=2)
d1_cv = np.array([m.distance for m, n in knn])
d2_cv = np.array([n.distance for m, n in knn])
same = np.mean(order[:, 0] == [m.trainIdx for m, n in knn])
print("same nearest point:", round(100 * same, 2), "%")
print("largest d1 difference:",
      round(float(np.abs(d1_np - d1_cv).max()), 4))
""")

code(r"""
import json

H = np.array(json.load(
    open("reference-homography.json"))["H_a_to_b"])
pa = np.float32([k.pt for k in ka])
pb = np.float32([kb[i].pt for i in order[:, 0]])
proj = cv2.perspectiveTransform(pa[None], H)[0]
right = np.linalg.norm(proj - pb, axis=1) <= 3
ratio = d1_np / d2_np
for t in (0.6, 0.75, 0.8, 1.01):
    keep = ratio < t
    print(f"t={t}: kept {keep.sum()}, right {(keep & right).sum()}")
""")

md(r"""
Exercise `keep`: the starter as printed, then a solution.
""")

code(r"""
def keep(d1, d2, t):
    return True  # compare d1 / d2 with t

print(keep(24.4, 284.1, 0.75), keep(252.3, 252.4, 0.75))
""")

code(r"""
# Solution, and the answers to exercises 2 and 3 (uses d1_np, d2_np, right from the cells above).
def keep(d1, d2, t):
    return d1 / d2 < t

print(keep(24.4, 284.1, 0.75), keep(252.3, 252.4, 0.75), keep(60.0, 80.0, 0.75))
for t in (0.75, 0.8):
    k = ratio < t
    print(f"t={t}: kept {k.sum()}, right {(k & right).sum()}")
n = int((ratio < 0.75).sum())
by_d1 = np.argsort(d1_np)[:n]
print(f"absolute threshold {d1_np[by_d1[-1]]:.1f}: kept {n}, right {right[by_d1].sum()}")
""")


@click.command()
def sift_notebook() -> None:
    """Write notebooks/sift.ipynb, outputs cleared."""
    nb = {
        "cells": [
            {"cell_type": t, "metadata": {}, "source": s.splitlines(keepends=True)}
            | ({"execution_count": None, "outputs": []} if t == "code" else {})
            for t, s in CELLS
        ],
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    out = PROJECT_ROOT / "notebooks" / "sift.ipynb"
    out.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n")
    click.echo(f"wrote {out} ({len(CELLS)} cells)")
