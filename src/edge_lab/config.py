"""Project-wide constants and artifact paths.

Paths resolve from the project root (found by walking up to pyproject.toml) so
every CLI works from any CWD; each is overridable via environment variable.
"""

import os
from pathlib import Path


def find_project_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "pyproject.toml").exists():
            return parent
    return Path.cwd()


PROJECT_ROOT = find_project_root()
DATA_DIR = Path(os.environ.get("EDGE_DATA_DIR", PROJECT_ROOT / "data"))
OUTPUT_DIR = Path(os.environ.get("EDGE_OUTPUT_DIR", PROJECT_ROOT / "output"))
FIGURE_DIR = OUTPUT_DIR / "figures"

# Middlebury multi-view stereo datasets (Seitz et al., CVPR 2006). Downloaded,
# never redistributed.
MIDDLEBURY_BASE_URL = "https://vision.middlebury.edu/mview/data/data"
DATASETS = {"templeRing": f"{MIDDLEBURY_BASE_URL}/templeRing.zip"}
DATASET = "templeRing"
VIEW = 0

# --- The canonical frame -----------------------------------------------------
# Every lesson in the unit works on one image so the reader compares operators
# on pixels they already know. It is the frame the SIFT article's interactive
# lab embeds, and reproducing it byte-for-byte is what lets the corner numbers
# here match the ones that article already published.
CLAHE_CLIP = 2.0
CLAHE_TILE = (8, 8)
CANONICAL_SIZE = (360, 480)  # (w, h)

# --- Structure tensor --------------------------------------------------------
# 21x21 window (half = 10). Sobel gradients on intensities in 0..1, with the
# 1-pixel border dropped because Sobel's response there is computed against
# replicated pixels rather than image content.
PATCH_HALF = 10
PATCH_PRESETS = {
    "background": (50, 60),
    "column_edge": (277, 240),
    "base_corner": (150, 320),
}
# Published in the SIFT article and in sfm-from-scratch/output/sift_numbers.json.
# The corners experiment is a parity check against these.
PUBLISHED_EIGENVALUES = {
    "background": (0.001, 0.000),
    "column_edge": (0.411, 0.019),
    "base_corner": (0.442, 0.318),
}

# --- Experiment sweeps -------------------------------------------------------
# Noise is Gaussian, in 0..255 intensity units, applied before any conversion.
NOISE_SIGMAS = (0.0, 2.0, 5.0, 10.0, 20.0)
NOISE_TRIALS = 5  # a single realization is not an effect
SEED = 20260818

# Fraction of pixels kept as "edge" when a magnitude map is thresholded. Fixing
# the count rather than the threshold is what makes two operators comparable:
# they have different response scales.
EDGE_FRACTION = 0.05

# Single-threshold sweep, on the 0..255 magnitude scale.
DILEMMA_THRESHOLDS = (10, 20, 40, 80, 160)

LOG_SIGMAS = (1.0, 1.4, 2.0, 3.0)

# Canny operating points, as (low, high) on the 0..255 magnitude scale.
CANNY_POINTS = ((50, 150), (75, 200), (100, 250))
