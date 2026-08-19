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


# =============================================================================
# Unit 1.2 — Image sensing
# =============================================================================
# The scenario is one scene of Mark Fairchild's HDR Photographic Survey: 18 NEF
# exposures a stop apart, a photometrically linear OpenEXR, and 54 points metered
# in the room with a Konica Minolta CS-100. Terms are research and non-commercial
# publication, so the files are downloaded and never redistributed, and the source
# is acknowledged wherever a pixel of it appears.
#
# The host serves plain HTTP; its HTTPS certificate has expired. Do not "fix"
# these URLs to https — they stop working.
HDRPS_BASE_URL = "http://markfairchild.org/HDRPS"
HDRPS_RAW_ARCHIVE_URL = "http://markfairchild.org/files/HDRPS_Raws.zip"
HDRPS_ARCHIVE_BYTES = 14_296_416_907

# The archive is 14 GB and was written without ZIP64 records, so every offset in
# it wrapped modulo 2**32. These three constants are the end-of-central-directory
# values read once from the file, plus the wrap count that makes them true again:
# 1_411_405_356 + 3 * 2**32 lands on "PK\x01\x02", and nothing else does.
HDRPS_CD_OFFSET_RAW = 1_411_405_356
HDRPS_CD_WRAPS = 3
HDRPS_CD_BYTES = 109_641

# The scene, chosen 19 Aug 2026: a lab setup built to characterise the camera —
# a lit ColorChecker, a second one in ambient light only, and a bulb in frame.
# 54 metered points from 0.015 to 2530 cd/m2, which is 17.4 stops of measured
# luminance, and the reason this scene beat the prettier ones.
SCENE = "Luxo Double Checker"
SCENE_SLUG = "LuxoDoubleChecker"
SENSING_DIR = DATA_DIR / "hdrps"


# =============================================================================
# Unit 1.1 — Image formation
# =============================================================================
# Three public sources, one per problem, none of them redistributed. OpenCV's own
# calibration photographs are Apache-2.0 and carry lessons 1 and 3; the fisheye
# board is GPL-2.0 and carries lesson 4; lesson 2 needs no pixels at all, only the
# capture settings printed on a figure in the DPDD repository (MIT).
FORMATION_NUMBERS_JSON = OUTPUT_DIR / "formation_numbers.json"

FORMATION_OPENCV_RAW = "https://raw.githubusercontent.com/opencv/opencv/4.x/samples/data"
# The set skips left10: the canonical sample data has 01-09 and 11-14, so a loop
# over range(1, 15) silently downloads a 404 page instead of a photograph.
FORMATION_LEFT_IDS = (1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13, 14)
FORMATION_LEFT_IMAGES = tuple(f"left{i:02d}.jpg" for i in FORMATION_LEFT_IDS)
FORMATION_CALIB_DIR = DATA_DIR / "opencv-calib"

# Inner corners, not squares: a 10x7 board of squares has 9x6 interior crossings,
# and findChessboardCorners wants the crossings.
FORMATION_BOARD = (9, 6)

# The physical square size is not published with these images. Calibrating in
# board units leaves fx, fy, cx, cy and every distortion coefficient unchanged --
# only the translation vectors carry the unit, and no lesson here quotes one. Any
# post using these numbers has to say this out loud.
FORMATION_SQUARE_SIZE = 1.0

# Lesson 4's lens. Downloaded, never vendored: GPL-2.0 is copyleft, so no image
# and no crop of one is published, and only fitted numbers leave this repository.
FORMATION_FISHEYE_RAW = "https://raw.githubusercontent.com/jakarto3d/py-OCamCalib/main/test_images/fish_1"
FORMATION_FISHEYE_IMAGES = tuple(f"Fisheye1_{i}.jpg" for i in range(1, 16))
FORMATION_FISHEYE_DIR = DATA_DIR / "fisheye-fish1"

# --- Where the metered points are, in pixels ---------------------------------
# The survey published its measurements as numbered circles drawn on a JPEG, so
# the pixel positions had to be recovered once. These are the centres of the four
# corner patches of each ColorChecker — patch 1, 6, 24 and 19, in that order —
# read off a developed frame (1/45 s for the lit chart, 10 s for the shadowed one,
# the exposures where each is neither blown nor buried).
#
# Four corners are enough because a ColorChecker is a lattice: one homography from
# grid coordinates places all 24. The check that this is right is not the reading,
# it is `sensing-charts`: sampled patch brightness against Fairchild's metered
# luminance ranks at Spearman 0.98 (lit) and 0.96 (shadowed). A lattice off by one
# column collapses both.
CHART_CORNERS = {
    "bright": [(2506, 1544), (3110, 1611), (3073, 1970), (2475, 1910)],
    "dim": [(238, 1576), (926, 1536), (928, 1906), (258, 1986)],
}
# Exposure index (0-based, in shutter order) where each chart is best exposed.
CHART_REFERENCE_FRAME = {"bright": 4, "dim": 13}
# The neutral row: patches 19-24, white through black, the ramp three lessons use.
NEUTRAL_ROW = (19, 20, 21, 22, 23, 24)

# Row of the D2x's published RGB -> CIE XYZ matrix that gives luminance, D65-normalised.
# Source: Fairchild's own "Nikon D2x Characterization for HDR-from-NEF Images"
# (markfairchild.org/HDRPS/HDRcharacterization.html), fitted on this scene's lit chart
# with R^2 of 0.998/0.997/0.992 and a mean Delta-E*ab of 2.5 over 25 patches.
D2X_LUMINANCE_ROW = (0.1904, 0.7646, 0.0450)
# The patch the radiance merge is scaled on: the lit chart's white, metered at
# 382 cd/m2. Every other point is then a prediction rather than a fit.
CALIBRATION_ANCHOR = "bright:19"
