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


# =============================================================================
# Unit 1.3 — The image as data
# =============================================================================
# Same scene as unit 1.2, same frame, same patch lattice: the point of reusing them
# is that a reader who followed 1.2 is looking at pixels they already know. What is
# new here is what we do to those pixels rather than where they came from.
IMAGEDATA_NUMBERS_JSON = OUTPUT_DIR / "image_data_numbers.json"
# Unit 1.3 spends unit 1.2's measurements rather than re-deriving them, so 1.2's
# artifact is an input here. Named once so the dependency is visible.
SENSING_NUMBERS_JSON = OUTPUT_DIR / "sensing_numbers.json"

# The lit chart's reference exposure, 1/45 s -- unit 1.2's `bright` reference frame.
IMAGEDATA_FRAME = "_MDF0005.NEF"

# The survey publishes a rendered JPEG, and it is 600x337: a thumbnail. The EXR is a
# Photoshop merge on a 3115x1752 grid, which is not this frame's grid either. So every
# representation this unit compares is developed here, from the one NEF, with the call
# unit 1.2 used -- which is what keeps all three on one 2868x4312 grid.
DEVELOP_KWARGS = {"use_camera_wb": True, "no_auto_bright": True}

# --- The camera, from Nikon's product page -----------------------------------
# 23.7 x 15.7 mm, 4288 x 2848 effective pixels (12.4 MP; 12.84 MP total). LibRaw
# exposes 4312 visible columns because it includes a masked border, so the pitch is
# computed from Nikon's effective figures and never from the array we happen to read.
D2X_SENSOR_MM = (23.7, 15.7)
D2X_EFFECTIVE_PX = (4288, 2848)
D2X_FOCAL_LENGTH_MM = 18.0  # the survey's recorded focal length for this scene

# --- Fairchild's published D2x colour matrices -------------------------------
# Verbatim from markfairchild.org/HDRPS/D2xCharacterization.pdf (updated 5/21/07).
# Both are RGB -> CIE XYZ. The first is the raw fit on a ColorChecker with no
# intercept; the second is that fit renormalised to D65, which the document calls
# "the one to use for most, if not all, HDR images".
#
# The published Delta-E below is a residual on the patches the matrix was fitted to,
# not a prediction. Quoting it against a number we measured on held-out patches would
# compare a fit to a forecast, so unit 1.3 fits its own matrix and cross-validates it.
D2X_XYZ_FROM_RGB_SCENE = (
    (0.4803, 0.5502, 0.1040),
    (0.1904, 0.7646, 0.0450),
    (-0.0096, 0.0487, 0.3805),
)
D2X_XYZ_FROM_RGB_D65 = (
    (0.4024, 0.4610, 0.0871),
    (0.1904, 0.7646, 0.0450),
    (-0.0249, 0.1264, 0.9873),
)
D2X_PUBLISHED_FIT = {"delta_e_mean": 2.5, "delta_e_std": 1.5, "delta_e_max": 5.5, "patches": 25}

# --- L2: sampling ------------------------------------------------------------
# The table under the lamp is a ribbed surface, and its ribbing is the only strongly
# periodic thing in the frame -- which makes it the one place aliasing can be measured
# rather than asserted. x0, y0, x1, y1 in full-frame pixels.
GRAIN_CROP = (2200, 2100, 3200, 2868)
# A bright column band inside that crop, where the ribbing has the best contrast.
GRAIN_STRIP = (30, 130)
# Rows of the crop the frequency measurement runs on, as (start, length).
# The ribbing is a physical grating photographed at a shallow angle, so its period in
# the image grows from about 6 px at the far edge to about 14 px at the near one. A
# single transform over the whole crop smears those into one broad peak, so the
# aliasing measurement runs on a window short enough for the period to be roughly
# constant across it, and the drift is reported beside it rather than hidden.
GRAIN_WINDOW = (96, 192)
DECIMATION_FACTORS = (2, 4, 8)

# --- L3: quantization --------------------------------------------------------
# The NEF is 12-bit (white level 4095), so these are all reductions from 12.
REQUANT_BITS = (10, 8, 6, 4)
# No SMOOTH_CROP. A textbook banding figure needs a broad, slowly-varying ramp, and
# this scene -- a dark room with one lamp -- has none: every wide tonal range in it is
# an edge. L3 measures patch separability instead, which is what the scene supports.

# The lamp shade's falloff: the only part of this scene with real gradation in it, and
# therefore the only place posterisation can be shown rather than described.
GLOW_CROP = (2090, 1290, 2410, 1610)

# --- L5: file formats --------------------------------------------------------
JPEG_QUALITIES = (95, 85, 75, 50, 25)
# The frame is a dark room with one lamp in it, so nine tenths of it is near-black and
# compresses to almost nothing. Ratios and PSNR measured over the whole frame are
# therefore a statement about the darkness, not about the encoder. Everything is
# measured twice: over the frame, and over the region that actually has content in it.
CONTENT_CROP = (1200, 800, 3200, 2600)
WEBP_QUALITY = 82  # the site's own encoding setting, so the number means something here
# Unit 3.1's middle Canny operating point, reused unchanged so the edge column is
# comparable with what the edge-detection unit published.
IMAGEDATA_CANNY = (75, 200)

# =============================================================================
# Unit 3.5 — Image alignment (and unit 4.1, lesson 1: homogeneous coordinates)
# =============================================================================
ALIGNMENT_NUMBERS_JSON = OUTPUT_DIR / "alignment_numbers.json"
ALIGN_DIR = DATA_DIR / "pickleball"

# "2026.07.25 WD Open - Sabrina Lam + Grace Thomas vs Lingzhe Xu + Margit Aardmaa
# (Gold Medal match)" by pickleball4you on YouTube, CC BY 3.0. One fixed camera
# behind a corner of the court, 1920x1080, 30000/1001 fps.
ALIGN_VIDEO_ID = "T5rmWjvt8Os"
ALIGN_FPS = 30000 / 1001
# The frame every figure shows: 45000 * 1001 / 30000 = 1501.5 s into the match.
ALIGN_FRAME = 45000
# The camera never moves, the players do. The median of 31 frames, one every
# 2 s either side of ALIGN_FRAME, is the court with nobody on it, and every line
# is measured on that plate rather than on a frame where a white shirt stands on
# a white line.
ALIGN_PLATE_FRAMES = tuple(range(44100, 45901, 60))
# Frame 45000 and the plate at full resolution (lossless), the 31 frames at 960 px
# for the notebook's median demo. The full-resolution frames come from YouTube via
# `align-download --from-youtube`.
ALIGN_SITE_BASE = "https://condados.ai/blog/image-alignment-and-stitching/frames"

# --- The court (USA Pickleball Official Rulebook 2026, Rule 3.A) ---------------
COURT_LENGTH_M = 13.41  # 3.A.1: 44 ft
COURT_WIDTH_M = 6.10  # 3.A.1: 20 ft
COURT_KITCHEN_M = 2.13  # 3.A.4.c: non-volley line, 7 ft from the net
# 3.A.4.e: lines are 2 in (5.08 cm) wide, and 3.A.2 measures the court to their
# *outside* edge. A line fitted to paint pixels finds its centre, so the model
# puts every line 2.54 cm inside the nominal dimension. Ignoring this moves each
# landmark by 2.5-3.6 cm, which is the size of the effects the unit measures.
COURT_LINE_HALF_M = 0.0254

# --- Finding the painted lines ------------------------------------------------
# Four approximate pixel positions of the near-half corners, read off the frame by
# eye to about 5 px. They only say where to look: each line is then refitted on
# paint pixels and the corners recomputed from the fits, twice, so the published
# landmarks do not depend on these numbers (checked by `stability`, below).
# Order: near baseline x left sideline (off the frame, at x < 0), near baseline x
# right sideline, near kitchen line x left sideline, near kitchen line x right.
ALIGN_SEED_CORNERS = ((-15.0, 585.0), (985.0, 960.0), (655.0, 500.0), (1555.0, 665.0))
WHITE_MIN_VALUE = 150  # HSV value: bright
WHITE_MAX_SAT = 90  # HSV saturation: close to grey
LINE_BAND_PX = 10.0  # half-width of the strip searched either side of a predicted line
LINE_REFINE_PASSES = 2
# The net, drawn by hand on the plate, with ~10 px of padding. Seen through the
# mesh, a painted line becomes a dotted band as wide as the search strip, and a
# least-squares fit to a uniform band returns the middle of the strip, which is
# the prediction it started from. That is an echo, not a measurement, so every
# pixel inside this polygon is excluded.
NET_POLYGON = ((842, 354), (1735, 442), (1735, 566), (842, 447))
SEED_PERTURB_PX = 5.0
SEED_PERTURB_TRIALS = 20

# --- Lesson 3 (DLT) ------------------------------------------------------------
DLT_NOISE_PX = 1.0
DLT_NOISE_TRIALS = 1000
ALIGN_SEED = 20260926

# --- Lesson 5 (warping) ----------------------------------------------------------
# The bird's-eye view: 1 px = 2 cm, with a 1.5 m margin around the court.
TOPVIEW_PX_PER_M = 50
TOPVIEW_MARGIN_M = 1.5
