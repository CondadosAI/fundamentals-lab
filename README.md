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
| 3.5 Image alignment, with 4.1 lesson 1 (homogeneous coordinates) | `alignment/` | `align-*` |

Run it in the browser, no install — one notebook per unit:
[**Image formation**](https://colab.research.google.com/github/CondadosAI/fundamentals-lab/blob/main/notebooks/image_formation.ipynb) ·
[**Image sensing**](https://colab.research.google.com/github/CondadosAI/fundamentals-lab/blob/main/notebooks/image_sensing.ipynb) ·
[**Edge detection**](https://colab.research.google.com/github/CondadosAI/fundamentals-lab/blob/main/notebooks/edge_detection.ipynb) ·
[**Image alignment**](https://colab.research.google.com/github/CondadosAI/fundamentals-lab/blob/main/notebooks/image_alignment.ipynb)

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
│   ├── alignment/             # unit 3.5 (and 4.1 lesson 1), image alignment
│   │   ├── court.py           # the court as lines on a plane, at line centres (rulebook 3.A)
│   │   ├── plate.py           # the frames, and their per-pixel median (the empty court)
│   │   ├── lines.py           # paint as a ridge, trimmed TLS line fits, corners as cross products
│   │   ├── transforms.py      # 2x2, affine and the DLT written out, Hartley normalisation
│   │   ├── evaluate.py        # held-out evidence in centimetres on the court
│   │   ├── distortion.py      # how bent the lines are, and the plumb-line k that straightens them
│   │   ├── warp.py            # forward and backward mapping, the bird's-eye view
│   │   └── wild.py            # the outdoor court and the Barcelona panorama
│   └── cli/                   # one click command per file
├── notebooks/
│   ├── image_formation.ipynb  # unit 1.1, Colab-ready
│   ├── image_sensing.ipynb    # unit 1.2, Colab-ready
│   ├── edge_detection.ipynb   # unit 3.1, Colab-ready
│   └── image_alignment.ipynb  # unit 3.5 and 4.1 lesson 1, Colab-ready
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

uv run data-experiments     # unit 1.3, and every "in the wild" run for the whole
                            # track → output/image_data_numbers.json. Reuses unit 1.2's
                            # scene, so `sensing-download` has to have run first; the
                            # photographs it fetches from Wikimedia Commons add ~75 MB
uv run data-figures         # the unit's figures, plus the in-the-wild panels
uv run data-covers          # cover backgrounds → output/covers/

uv run align-download       # unit 3.5: the court frame and its median plate from condados.ai,
                            # plus the outdoor section (yt-dlp) and the Barcelona pair;
                            # --from-youtube rebuilds the plate from the source video
uv run align-experiments    # every number → output/alignment_numbers.json
uv run align-figures        # overlays, top views and cover backgrounds
uv run align-notebook       # rebuild notebooks/image_alignment.ipynb, outputs cleared
uv run align-media          # the animations the posts open with (needs ffmpeg with libwebp)
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

## Unit 1.3 — The image as data

The unit that had to take a live article apart to be written, and the one that finally
measures the whole track on photographs as well as on charts.

It runs on unit 1.2's scene, one frame of it, developed here rather than taken from the
survey's own renderings — **the published "Rendered" JPEG is 600 × 337**, a thumbnail,
and the OpenEXR is a Photoshop merge on a 3115 × 1752 grid, so neither can be compared
pixel-for-pixel with the raw. Everything comes out of `_MDF0005.NEF` through the same
`rawpy.postprocess` call unit 1.2 used, which puts the 12-bit mosaic and both
developments on one grid.

**A bug worth not rediscovering.** `rawpy`'s `postprocess()` mutates LibRaw's internal
state, so reading `raw_pattern` *after* developing reports the processed layout rather
than the file's: RGGB's `[[0,1],[3,2]]` comes back as `[[0,1],[1,2]]`. It resolves to
the same four channel offsets, so nothing downstream moves and the only casualty is a
wrong CFA pattern printed somewhere. `scene.load()` reads metadata first.

Keys are in `output/image_data_numbers.json`.

| Claim | Where |
|---|---|
| Two thirds of every developed colour image is interpolated; demosaicing costs 4.7 DN on flat patches and 129.9 DN at the strongest edges | `cfa.demosaic_error` |
| Quantization error meets this sensor's measured noise floor at 9.28–9.49 bits, so a 12-bit file carries about 2.5 spare | `l3_quantization.noise_floor_crossing` |
| Δ/√12 predicts the measured error to within 2% on mid-tones and runs 3× optimistic over the whole dark frame | `l3_quantization.error_vs_prediction` |
| A 7.385 px ribbing decimated by 8 returns at 96 px, and the fold predicts the bin it lands in | `l2_sampling.aliasing` |
| Treating a developed file as sRGB lands 9.54 ΔE\*ab from the colorimeter; a 3×3 fitted here and scored on held-out patches reaches 2.41 | `l4_colour.accuracy` |
| **Fairchild's own published D2x matrix does not beat the naive route on this scene** — 9.14–10.29 across four configurations | `l4_colour.accuracy.paths.published_matrix.variants` |
| OpenCV 5.0.0 and Pillow 12.3.0 decode every JPEG here identically; the disagreement is a 16-bit PNG, which Pillow silently returns as 8-bit | `l5_formats.decoders` |

### In the wild

Every lesson on the written track — twenty of them, across units 1.1, 1.2, 1.3 and 3.1 —
repeats its measurement on a photograph, because a calibration target is not a picture
anybody takes. Those runs live under `in_the_wild` and `in_the_wild_other_units` in the
same artifact, and the assets are listed under Licence below.

| Claim | Where |
|---|---|
| 53 detected lines in a temple cloister agree on one vanishing point to 1.6% of the frame diagonal | `in_the_wild_other_units.unit_1_1.l1_vanishing_point` |
| A photograph's own EXIF predicts 0.59 m of depth of field, and its sharpest band is 26× the softest | `…unit_1_1.l2_depth_of_field` |
| Reading a JPEG as if it were linear understates a chart's white-to-black range by 10.5× | `…unit_1_2.l3_response_curve` |
| Not one of 400 edges measured in a real photograph is a step: median equivalent width 2.30 px | `…unit_3_1.l1_edge_profile` |
| 96.3% of an unsmoothed Laplacian's zero-crossings on printed text sit where nothing is changing | `…unit_3_1.l3_zero_crossings` |
| Canny's hysteresis returns 654 contours over 100 px out of 4,340; the low threshold alone needs 73,025 fragments to find 700 | `…unit_3_1.l4_canny_hysteresis` |
| A wheel filmed at 24 fps reports −0.90 apparent revolutions per second — backwards | `in_the_wild.l2_wagon_wheel` |

## Unit 3.5 — Image alignment

One frame of a pickleball final (and the median of 31 around it, because the camera is
fixed and the players are not), measured against the court's published dimensions. Keys
are in `output/alignment_numbers.json`.

| Claim | Where |
|---|---|
| A court corner 18.5 px outside the frame, found as the cross product of two fitted lines | `homogeneous.NBL_img` |
| The sidelines, parallel on the ground, meet at pixel (2252, 299) | `homogeneous.vp_along_court_img` |
| The best 2×2 misses every near-court corner by 60.4 px | `linear.best_2x2_residual_px` |
| On the far right sideline an affine map from 3 corners is 3.97 m out, a homography from 4 is 5.95 cm out | `affine_vs_projective` |
| Hartley normalisation takes cond(A) from 53,466 to 6.4 and moves the answer by under 0.2 cm; it makes the result independent of the pixel origin | `dlt` |
| One mislabelled pair among six puts the far baseline 17.6 m out | `dlt.one_wrong_pair_of_six` |
| Forward mapping leaves 55.8% of the far court's bird's-eye pixels unwritten | `warping` |
| A one-parameter division model straightens the near baseline from 7.71 px of bow to 0.16 | `lens` |
| Moving the four seed corners by up to 5 px moves no fitted corner by more than 0.58 px | `stability` |

Three things here cost time and are worth not rediscovering:

**A white shirt is white paint.** In frame 45000 the near player stands on a sideline in a
white top; the paint detector took her for the line. The median of 31 frames is the court
with nobody on it, and the corners it gives agree with frame 45000's to within 1.3 px.

**A pale surface is a line to a brightness threshold.** The non-volley zone is surfaced in
light grey, and part of it passes any threshold that keeps the paint. It sits on one side
of the kitchen line and drags that line's fit. Paint is found as a ridge instead (white
top-hat, 21 px), which took the far baseline error from 81.3 cm to 38.4.

**A line seen through a net is its own seed.** Behind the mesh a painted line becomes a
dotted band as wide as the search strip, and a line fitted to a uniform band returns the
strip's centre. The net region is excluded by a hand-drawn polygon (`config.NET_POLYGON`).

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

- **Wikimedia Commons photographs and one video**, fetched by `data-experiments` for the
  "in the wild" measurements: a temple cloister, a market stall, a brick wall, a
  circular fisheye frame, a twilight landscape, a long-exposure astrophotograph, a
  photographed ColorChecker, a red telephone box, an 1814 newspaper scan, and a clip of
  the wagon-wheel effect. All are **CC0, public domain or CC BY 4.0 — none is
  share-alike**, so figures derived from them carry no copyleft obligation. Each is
  recorded with its author, licence, source page and a sha256 of the exact bytes
  measured; the fetcher re-checks that hash and refuses a file that changed at the
  source. Downloaded, never committed here. Attribution for each appears in the caption
  of every figure that shows it, on condados.ai.

- **HDR Photographic Survey — "Luxo Double Checker"** (Fairchild, CIC 15, 2007),
  fetched from <http://markfairchild.org/HDR.html>: eighteen NEF exposures out of the
  survey's RAW archive, the scene's OpenEXR, its measurement table and its map.
  **Research use and non-commercial publication only — commercial publication of these
  images is prohibited**, and the source must be acknowledged as *Mark Fairchild's HDR
  Photographic Survey*. Downloaded, never redistributed; the repository publishes
  numbers derived from the files, not the files. Unit 1.2 uses it as its scenario, on
  educational-use grounds.

- **Pickleball footage from pickleball4you on YouTube, CC BY 3.0** (YouTube's only
  Creative Commons option). *"2026.07.25 WD Open - Sabrina Lam + Grace Thomas vs Lingzhe Xu +
  Margit Aardmaa (Gold Medal match)"* (`T5rmWjvt8Os`): frame 45000 and a median of 31
  frames, served by condados.ai with attribution, or fetched from YouTube with
  `--from-youtube`. *"2024.08.30 MS4.0 Philip Wong vs Tristan Clark (Round Robin, match 6)"*
  (`K0qrASvix3Y`): a 70 s section fetched with yt-dlp. Attribution is required by the
  licence and appears on every figure that shows a frame. Nothing is committed here.
- **Two photographs of Barcelona harbour** by Pap3rinik on Wikimedia Commons,
  *BarcelonaHarbour1.jpg* and *BarcelonaHarbour2.jpg*, **public domain** (released by the
  author, `{{PD-self}}`). Fetched through the Commons API, never committed.
- **A printed card on a bench**, *Stationery on a bench (Unsplash).jpg* by Brigitte Tohm on
  Wikimedia Commons, **CC0**, for the document-scanner animation. Fetched, never committed.
- **Player ankle positions for a 10 s clip** (`output/alignment_players_clip.csv`) and ball
  detections (`output/alignment_ball_clip.csv`), exported from
  [CondadosAI/sportcv](https://github.com/CondadosAI/sportcv), which runs the RTMO pose model
  through rtmlib (Apache-2.0) on the same CC BY 3.0 match. The minimap animation maps them
  through this repository's homography, not sportcv's.
- **USA Pickleball Official Rulebook (2026), Rule 3.A.** Only the court dimensions are
  used, as numbers in `config.py`.

No model weights are used or downloaded.
