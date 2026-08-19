# fundamentals-lab

Edge detection from first principles, measured rather than asserted: gradient
operators, the Laplacian and its zero-crossings, Canny's non-maximum suppression
and hysteresis taken apart, and the structure tensor that turns contours into
points you can match.

Companion code for the CondadosAI **Fundamentals** track. One repo for the whole
track; today it carries the **Edge Detection** unit (one hub post and five lessons).

Run it in the browser, no install:
[**Open in Colab**](https://colab.research.google.com/github/CondadosAI/fundamentals-lab/blob/main/notebooks/edge_detection.ipynb)

## Layout

```
fundamentals-lab/
├── pyproject.toml
├── src/edge_lab/
│   ├── config.py              # paths, sweeps, and every threshold, documented
│   ├── core/
│   │   ├── dataset.py         # the canonical frame the whole unit works on
│   │   ├── gradients.py       # central differences, Prewitt, Sobel
│   │   ├── laplacian.py       # LoG, zero-crossings, contour closure
│   │   ├── canny.py           # NMS and hysteresis, separable and separately measurable
│   │   └── corners.py         # structure tensor → Harris, Shi-Tomasi, Förstner
│   └── cli/                   # one click command per file
├── notebooks/
│   └── edge_detection.ipynb   # the same code, section per lesson, Colab-ready
├── data/                      # downloaded dataset (gitignored)
└── output/
    ├── edge_numbers.json      # every number the articles cite
    └── figures/               # one figure per lesson
```

## Setup

```bash
uv sync
```

## Commands

```bash
uv run edge-download        # Middlebury templeRing (11.7 MB, 47 views)
uv run edge-experiments     # every sweep → output/edge_numbers.json
uv run edge-figures         # one figure per lesson → output/figures/
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

## What is measured

| Claim | Where |
|---|---|
| Smoothing across rows is what buys noise robustness — and a box average (Prewitt) does it slightly better than a triangular one (Sobel), because it passes less noise power for the same support | `noise_robustness` |
| At a matched edge-pixel count, hysteresis produces about half as many contours, each about twice as long, as a single threshold | `hysteresis_vs_single_threshold` |
| Smoothing is what makes zero-crossings usable: at σ = 1.0 noise multiplies the crossing pixels by 2.76×, at σ = 3.0 by 1.02× | `zero_crossing_closure` |
| The structure-tensor eigenvalues reproduce the published SIFT values exactly | `corner_parity` |

Every sweep runs several configurations. Noise experiments average five
realizations from a fixed seed and report the standard deviation alongside the
mean, because one run of a noisy pipeline is variance, not an effect.

## Licence

The code in this repository is Apache-2.0 (see `LICENSE`).

**What the licence does not cover.** The pipeline *downloads* data at run time; it
does not redistribute it, and Apache-2.0 says nothing about it:

- **Middlebury Multi-View Stereo — `templeRing`** (Seitz et al., CVPR 2006),
  fetched from <https://vision.middlebury.edu/mview/data/>. The dataset page
  **states no licence**. Treat it as academic-use material, check with the
  authors before any commercial use, and note that nothing in this repository
  grants you rights to it. No dataset file is committed here.

No model weights are used or downloaded.
