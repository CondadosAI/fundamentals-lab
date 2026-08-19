# fundamentals-lab

Computer vision from first principles, measured rather than asserted: how a camera
forms an image, what the number in a pixel actually counts, and how an edge is found
in a grid of those numbers. Every figure and every number published on condados.ai's
Fundamentals track comes out of this repo.

Companion code for the CondadosAI **Fundamentals** track. One repo for the whole
track, one module per unit:

| Unit | Module | Commands |
|---|---|---|
| 1.1 Image formation | `formation/` | `formation-*` |
| 1.2 Image sensing | `sensing/` | `sensing-*` (needs `uv sync --extra sensing`) |
| 3.1 Edge detection | `core/` | `edge-*` |

Run it in the browser, no install — one notebook per unit:
[**Image formation**](https://colab.research.google.com/github/CondadosAI/fundamentals-lab/blob/main/notebooks/image_formation.ipynb) ·
[**Image sensing**](https://colab.research.google.com/github/CondadosAI/fundamentals-lab/blob/main/notebooks/image_sensing.ipynb) ·
[**Edge detection**](https://colab.research.google.com/github/CondadosAI/fundamentals-lab/blob/main/notebooks/edge_detection.ipynb)

## Layout

```
fundamentals-lab/
├── pyproject.toml
├── src/fundamentals_lab/
│   ├── config.py              # paths, sweeps, and every threshold, documented
│   ├── core/                  # unit 3.1, edge detection
│   │   ├── dataset.py         # the canonical frame that unit works on
│   │   ├── gradients.py       # central differences, Prewitt, Sobel
│   │   ├── laplacian.py       # LoG, zero-crossings, contour closure
│   │   ├── canny.py           # NMS and hysteresis, separable and separately measurable
│   │   └── corners.py         # structure tensor → Harris, Shi-Tomasi, Förstner
│   ├── sensing/               # unit 1.2, image sensing
│   │   ├── archive.py         # one scene out of a 14 GB zip, by HTTP byte range
│   │   ├── hdrps.py           # the scene: raw exposures, metered points, capture data
│   │   ├── charts.py          # where the 48 ColorChecker patches are, and the check that says so
│   │   ├── noise.py           # the photon transfer curve → gain, full well, dynamic range
│   │   ├── response.py        # raw against developed: the tone curve, measured
│   │   └── hdr.py             # the bracket merged, then checked against a colorimeter
│   └── cli/                   # one click command per file
├── notebooks/
│   ├── image_formation.ipynb  # unit 1.1, Colab-ready
│   ├── image_sensing.ipynb    # unit 1.2, Colab-ready
│   └── edge_detection.ipynb   # unit 3.1, Colab-ready
├── data/                      # downloaded datasets (gitignored, never redistributed)
└── output/
    ├── edge_numbers.json      # every number unit 3.1's articles cite
    ├── formation_numbers.json # every number unit 1.1's articles cite
    ├── sensing_numbers.json   # every number unit 1.2's articles cite
    ├── figures/               # generated images (gitignored — see Licence)
    └── covers/                # generated cover backgrounds (gitignored)
```

Unit 1.1's module lives beside these under `formation/`.

## Setup

```bash
uv sync
```

## Commands

```bash
uv run edge-download        # Middlebury templeRing (11.7 MB, 47 views)
uv run edge-experiments     # every sweep → output/edge_numbers.json
uv run edge-figures         # one figure per lesson → output/figures/

uv run formation-download   # unit 1.1's photographs (1.5 MB)
uv run formation-calibrate  # intrinsics and distortion → output/formation_numbers.json
uv run formation-distort    # how far the lens moves a pixel, and what straightening costs
uv run formation-dof        # depth of field at f/4 and f/22
uv run formation-fisheye    # both camera models fitted to one fisheye lens

uv sync --extra sensing     # unit 1.2 only: rawpy/LibRaw, kept out of the base install
uv run sensing-download     # one HDRPS scene: 18 raw exposures (220 MB) and its data
uv run sensing-inspect      # what the files declare → output/sensing_frames.json
uv run sensing-experiments  # noise, response and the HDR merge → output/sensing_numbers.json
uv run sensing-figures      # the unit's photographs → output/figures/
uv run sensing-covers       # cover backgrounds → output/covers/
```

`edge-experiments` **fails loudly** if the corner eigenvalues stop matching the
values the SIFT article published. That check is the only thing keeping two
articles' numbers in agreement, so it is an error, not a warning.

## The canonical frame

Every lesson works on one image: Middlebury templeRing view 0, rotated upright,
CLAHE-lifted on the L channel, resized to 360×480. That is the same frame the
SIFT article's interactive lab embeds, which is why the corner numbers here are
the same numbers that article already reports — the reader can carry them across
the boundary between the two units without re-anchoring.

## What unit 3.1 measures

Keys are in `output/edge_numbers.json`.

| Claim | Where |
|---|---|
| Smoothing across rows is what buys noise robustness — and a box average (Prewitt) does it slightly better than a triangular one (Sobel), because it passes less noise power for the same support | `noise_robustness` |
| At a matched edge-pixel count, hysteresis produces about half as many contours, each about twice as long, as a single threshold | `hysteresis_vs_single_threshold` |
| Smoothing is what makes zero-crossings usable: at σ = 1.0 noise multiplies the crossing pixels by 2.76×, at σ = 3.0 by 1.02× | `zero_crossing_closure` |
| The structure-tensor eigenvalues reproduce the published SIFT values exactly | `corner_parity` |

Every sweep runs several configurations. Noise experiments average five
realizations from a fixed seed and report the standard deviation alongside the
mean, because one run of a noisy pipeline is variance, not an effect.

## Unit 1.1 — Image formation

Four commands, one artifact section each, all writing `output/formation_numbers.json`.
The unit measures on three public sources rather than a camera of ours, and none of
them is redistributed here.

| Claim | Where |
|---|---|
| A real lens displaces the worst image corner by 51.22 px on a 640×480 frame | `distortion` |
| Undistorting this lens costs field of view rather than pixels: 0.00% of the output is empty at the same focal length, and the view stays at 61.67° where the lens captured 67.32° | `distortion` |
| Stopping down from f/4 to f/22 deepens focus by 5.54×, against an aperture ratio of 5.50 | `depth_of_field` |
| On a fisheye lens, pinhole plus Brown–Conrady fits to 7.326 px RMS where Kannala–Brandt reaches 0.644 px | `fisheye` |

Two things here will cost you an afternoon if you meet them cold:

**`cv2.fisheye.calibrate` needs an intrinsic guess.** Left to initialise `K` itself
it converges to a focal length it cannot recover from and lands ~200 px from the
corners. Seeded with the equidistant estimate `f = max(w, h) / π` and
`CALIB_USE_INTRINSIC_GUESS`, the same 15 photographs fit to 0.64 px.

**OpenCV 5 moved the fisheye calibration flags.** `cv2.fisheye.CALIB_FIX_SKEW` and
its neighbours no longer exist; they are `cv2.CALIB_FIX_SKEW` now, so 4.x tutorials
raise `AttributeError` on that line.

Also worth knowing: `py-OCamCalib`'s own `checkerboard_sizes.txt` gives a 5×7 board
for this set. It is 6×8 interior corners, and 5×7 *is* accepted on 4 of the 15
images because a smaller grid fits inside a larger one, so trusting the file
calibrates on a third of the data with the wrong geometry.

## Unit 1.2 — Image sensing

Two things in this module exist because the data fought back, and both would cost an
afternoon to rediscover.

**The survey publishes its raw exposures as one 14.3 GB zip.** Reaching one scene does
not require downloading it: the server answers byte-range requests and the archive is
organised one folder per scene, so `sensing-download` fetches about 220 MB. What makes
that harder than it sounds is that the archive was written **without ZIP64 records**
although it is far past 4 GB, so every offset inside it wrapped modulo 2³². Python's
`zipfile` and the off-the-shelf remote-zip readers follow those offsets and land in the
middle of another member's data. `archive.py` adds the wraps back — the central
directory by a known count, per-entry offsets by counting where the sequence decreases.

**The measurement table uses the 1904 date epoch**, because the workbook was written on
a Mac. Read with the usual 1900 epoch, the shoot dates to 2002 instead of 2006, and that
wrong date would have gone straight into a published reproducibility block.

The unit's numbers are checked against an instrument rather than against themselves: the
scene ships with 54 points metered by a colorimeter. Keys are in
`output/sensing_numbers.json`.

| Claim | Where |
|---|---|
| The sensor's gain is 0.294 ± 0.007 DN per electron, so one count is 3.40 electrons and the well holds 13,199 | `noise.channels.G1` |
| Read noise is **bounded**, not measured: the repeated frames are 30 s long, the transfer fit's intercept comes out negative, and the 5.6 e⁻ figure is the spread of a dark corner | `noise.channels.G1.dark_corner_sigma_electrons` |
| The developed file sits a mean of 9.6 code values from the sRGB curve, and no single exponent fits the ramp | `response` |
| A bracket merged from raw, scaled by one factor fitted on one patch, lands a median 0.198 stops from the colorimeter over 48 patches — 0.109 over the twelve neutral ones | `hdr` |
| Every large error is at the dark end, where flare from the bulb and the noise floor both live | `hdr.predictions` |

## Licence

The code in this repository is Apache-2.0 (see `LICENSE`).

**What the licence does not cover.** The pipeline *downloads* data at run time; it
does not redistribute it, and Apache-2.0 says nothing about it:

- **Middlebury Multi-View Stereo — `templeRing`** (Seitz et al., CVPR 2006),
  fetched from <https://vision.middlebury.edu/mview/data/>. The dataset page
  **states no licence**. Treat it as academic-use material, check with the
  authors before any commercial use, and note that nothing in this repository
  grants you rights to it. No dataset file is committed here.

- **OpenCV sample calibration images** (`samples/data/left01–left14.jpg`, the set
  skips `left10`), fetched from the `opencv/opencv` repository. **Apache-2.0** —
  the one source here that could be redistributed. It is downloaded anyway, so
  that this repository holds code and numbers and nothing else.
- **py-OCamCalib fisheye board views** (`test_images/fish_1`), fetched from
  `jakarto3d/py-OCamCalib`. **GPL-2.0 — copyleft.** Downloaded, never vendored,
  and no image or crop of one is published: only fitted numbers leave here.
- **DPDD** (Abuolaim & Brown, ECCV 2020). No dataset file is downloaded at all.
  Unit 1.1 uses only the capture settings printed on `figures/data_example.png`
  in that repository, which is MIT.

- **HDR Photographic Survey — "Luxo Double Checker"** (Fairchild, CIC 15, 2007),
  fetched from <http://markfairchild.org/HDR.html>: eighteen NEF exposures out of the
  survey's RAW archive, the scene's OpenEXR, its measurement table and its map.
  **Research use and non-commercial publication only — commercial publication of these
  images is prohibited**, and the source must be acknowledged as *Mark Fairchild's HDR
  Photographic Survey*. Downloaded, never redistributed; the repository publishes
  numbers derived from the files, not the files. Unit 1.2 uses it as its scenario, on
  educational-use grounds.

No model weights are used or downloaded.
