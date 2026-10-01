"""OpenCV in Practice, module M2: find every ball on a pool table.

The scene is three CC0 photographs by MarkBuckawicki on Wikimedia Commons, taken on
27 Feb 2014 at the same table (registered in CondadosAI/cv-assets):

    pool.webp        "Billiards table 2.JPG"   15 balls, the cue across the cloth
    pool-close.webp  "Billiards table 1.JPG"   12 balls, two of them close to the lens
    pool-wide.webp   "Billiards Table.JPG"     14 balls, the whole table from one corner

The lessons run on pool.webp; the hub scores the finished pipeline on all three against
the hand-placed centres in LABELS. Every number the posts print is computed here from
the published WebP files, decoded the way the page decodes them.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
from loguru import logger

from fundamentals_lab.config import DATA_DIR, OUTPUT_DIR

#: Published name -> original file in data/pool/ (download: see cv-assets).
PHOTOS = {
    "pool": "Billiards_table_2.JPG",
    "pool-close": "Billiards_table_1.JPG",
    "pool-wide": "Billiards_Table.JPG",
}
WIDTH = 960
QUALITY = 90
SOURCE_DIR = DATA_DIR / "pool"
ASSET_DIR = OUTPUT_DIR / "practice" / "m2"
NUMBERS_JSON = OUTPUT_DIR / "practice_m2_numbers.json"
LABELS_JSON = OUTPUT_DIR / "practice_m2_labels.json"

#: Every ball resting on the playing surface, as (x, y, radius) in the 960 x 720 frame,
#: placed by hand on zoomed crops. One ball in pool-wide sits on top of the far rail,
#: off the cloth, and is left out; that is a choice of what "on the table" means.
LABELS = {
    "pool": [
        (145, 157, 20),
        (167, 197, 22),
        (414, 109, 14),
        (494, 92, 13),
        (615, 120, 13),
        (686, 129, 13),
        (688, 156, 14),
        (727, 155, 14),
        (509, 162, 17),
        (592, 197, 18),
        (896, 220, 17),
        (350, 250, 22),
        (573, 260, 20),
        (424, 298, 23),
        (791, 311, 23),
    ],
    "pool-close": [
        (300, 112, 15),
        (480, 110, 15),
        (539, 83, 13),
        (657, 101, 16),
        (703, 188, 23),
        (841, 156, 22),
        (813, 173, 24),
        (714, 227, 28),
        (201, 266, 30),
        (428, 234, 26),
        (179, 504, 52),
        (329, 473, 48),
    ],
    "pool-wide": [
        (449, 107, 11),
        (210, 160, 8),
        (317, 163, 11),
        (277, 213, 12),
        (178, 222, 11),
        (162, 233, 11),
        (188, 238, 10),
        (28, 294, 11),
        (28, 308, 11),
        (90, 350, 12),
        (118, 333, 12),
        (288, 320, 15),
        (626, 218, 17),
        (292, 596, 23),
    ],
}

#: Why each ball the finished pipeline misses is missed, read off the debug images.
#: experiments() fails if a miss has no cause here, so the table cannot drift from it.
CAUSES = {
    "touching": [
        ("pool", 145, 157),
        ("pool", 167, 197),
        ("pool", 688, 156),
        ("pool-close", 703, 188),
        ("pool-close", 841, 156),
        ("pool-wide", 178, 222),
        ("pool-wide", 162, 233),
        ("pool-wide", 188, 238),
        ("pool-wide", 28, 294),
    ],
    "cloth-coloured": [("pool", 727, 155), ("pool-wide", 90, 350)],
    "shadow": [("pool-close", 329, 473)],
}

# The pipeline's settings, as the lessons' cells write them.
HUE = (75, 100)  # the cloth, on OpenCV's 0-179 hue scale
SAT_MIN = 80
V_MIN = 40  # the brightness floor a first attempt uses
#: Lesson 5's floor. Hue means nothing on a near-black pixel (after WebP compression the
#: room's blacks carry random hues), so the floor drops to 10, not to 0. The hub prints
#: the sweep over 0-40 beside it, because the count moves by a ball or two with it.
V_SHADOW = 10
EDGE = 19  # erosion of the table, in pixels, to stay off the cushions
MIN_AREA = 45
MIN_ROUND = 0.55
POCKET_V = 60  # a blob whose brightest pixel is below this is a hole, not a ball

#: The stages the hub's table reports, each adding one fix to the one before.
STAGES = {
    "first": {"v_min": V_MIN, "pocket_v": 0},
    "pockets": {"v_min": V_MIN, "pocket_v": POCKET_V},
    "shadows": {"v_min": V_SHADOW, "pocket_v": POCKET_V},
}
SWEEP = [0, 5, 10, 20, 30, 40]


def build_assets(out_dir: Path = ASSET_DIR) -> dict[str, Path]:
    """Resize the three originals to WIDTH and write them as the module's WebP files."""
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    for name, src in PHOTOS.items():
        img = cv2.imread(str(SOURCE_DIR / src))
        if img is None:
            raise FileNotFoundError(f"{SOURCE_DIR / src}: download it (see cv-assets)")
        h = round(img.shape[0] * WIDTH / img.shape[1])
        img = cv2.resize(img, (WIDTH, h), interpolation=cv2.INTER_AREA)
        path = out_dir / f"{name}.webp"
        cv2.imwrite(str(path), img, [cv2.IMWRITE_WEBP_QUALITY, QUALITY])
        paths[name] = path
        logger.info(f"{name}: {src} -> {img.shape}")
    LABELS_JSON.write_text(json.dumps({k: [list(b) for b in v] for k, v in LABELS.items()}) + "\n")
    return paths


def load(name: str, asset_dir: Path = ASSET_DIR) -> np.ndarray:
    img = cv2.imread(str(asset_dir / f"{name}.webp"))
    if img is None:
        raise FileNotFoundError(f"{name}.webp: run `uv run practice-m2 --assets` first")
    return img


# --- the pipeline, one function per lesson ---------------------------------------------


def cloth_mask(img: np.ndarray, v_min: int = V_MIN) -> np.ndarray:
    """Lessons 1 and 2: the cloth is one hue, whatever its brightness."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    return cv2.inRange(hsv, (HUE[0], SAT_MIN, v_min), (HUE[1], 255, 255))


def clean(mask: np.ndarray) -> np.ndarray:
    """Lesson 3: opening removes specks, closing fills pinholes."""
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))


def table_mask(cloth: np.ndarray) -> np.ndarray:
    """Lesson 4: the largest cloth contour, filled through its convex hull, pulled in."""
    contours, _ = cv2.findContours(cloth, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    hull = cv2.convexHull(max(contours, key=cv2.contourArea))
    table = np.zeros_like(cloth)
    cv2.fillPoly(table, [hull], 255)
    return cv2.erode(table, np.ones((EDGE, EDGE), np.uint8))


def roundness(contour: np.ndarray) -> float:
    """4 pi area / perimeter^2: 1 for a circle, near 0 for a stick."""
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)
    return 4 * np.pi * area / perimeter**2 if perimeter else 0.0


def find_balls(img: np.ndarray, v_min: int = V_MIN, pocket_v: int = 0) -> dict:
    """The whole pipeline. Returns the kept balls and every rejected blob, with reasons."""
    cloth = cloth_mask(img, v_min)
    table = table_mask(cloth)
    objects = clean(cv2.bitwise_and(cv2.bitwise_not(cloth), table))
    value = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)[..., 2]
    contours, _ = cv2.findContours(objects, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    balls, rejected = [], []
    for c in contours:
        area = cv2.contourArea(c)
        if area < MIN_AREA:
            continue
        (x, y), r = cv2.minEnclosingCircle(c)
        blob = {"x": round(x, 1), "y": round(y, 1), "r": round(r, 1), "area": area}
        blob["round"] = round(roundness(c), 2)
        filled = np.zeros_like(objects)
        cv2.drawContours(filled, [c], -1, 255, -1)
        blob["brightest"] = int(value[filled > 0].max())
        if blob["round"] <= MIN_ROUND:
            rejected.append({**blob, "why": "not round"})
        elif blob["brightest"] < pocket_v:
            rejected.append({**blob, "why": "too dark"})
        else:
            balls.append(blob)
    return {"balls": balls, "rejected": rejected, "objects": objects, "table": table}


# --- scoring against the hand labels ---------------------------------------------------


def match(balls: list[dict], labels: list[tuple]) -> dict:
    """One detection per ball, closest pairs first, within the ball's radius (8 px minimum)."""
    pairs = sorted(
        (float(np.hypot(b["x"] - lx, b["y"] - ly)), i, j)
        for i, (lx, ly, _) in enumerate(labels)
        for j, b in enumerate(balls)
    )
    hit_l, hit_b = set(), set()
    for dist, i, j in pairs:
        if i in hit_l or j in hit_b or dist >= max(labels[i][2], 8):
            continue
        hit_l.add(i)
        hit_b.add(j)
    return {
        "found": len(hit_l),
        "balls": len(labels),
        "false": len(balls) - len(hit_b),
        "missed_at": [list(labels[i][:2]) for i in range(len(labels)) if i not in hit_l],
        "false_at": [[balls[j]["x"], balls[j]["y"]] for j in range(len(balls)) if j not in hit_b],
    }


def _patch(hsv: np.ndarray, x: int, y: int, r: int = 4) -> list[int]:
    return [int(v) for v in np.median(hsv[y - r : y + r + 1, x - r : x + r + 1].reshape(-1, 3), 0)]


def experiments() -> dict:
    out: dict = {
        "settings": {
            "hue": HUE,
            "sat_min": SAT_MIN,
            "v_min": V_MIN,
            "edge": EDGE,
            "min_area": MIN_AREA,
            "min_round": MIN_ROUND,
            "pocket_v": POCKET_V,
            "v_shadow": V_SHADOW,
        },
        "photos": {},
        "stages": {},
    }
    for name in PHOTOS:
        out["photos"][name] = {"balls": len(LABELS[name]), "shape": list(load(name).shape)}
    for stage, kw in STAGES.items():
        row, tot = {}, {"found": 0, "balls": 0, "false": 0}
        for name in PHOTOS:
            m = match(find_balls(load(name), **kw)["balls"], LABELS[name])
            row[name] = m
            for k in tot:
                tot[k] += m[k]
        row["total"] = tot
        out["stages"][stage] = row
    # The pocket rule's margin: the brightest pixel of every blob it rejected, against
    # the dimmest brightest-pixel of any kept blob that lands on a labelled ball.
    dark, lit = [], []
    for name in PHOTOS:
        for stage in ("pockets", "shadows"):
            r = find_balls(load(name), **STAGES[stage])
            dark += [b["brightest"] for b in r["rejected"] if b["why"] == "too dark"]
            for b in r["balls"]:
                if any(np.hypot(b["x"] - x, b["y"] - y) < r_ for x, y, r_ in LABELS[name]):
                    lit.append(b["brightest"])
    out["pocket_margin"] = {"rejected_brightest_max": max(dark), "ball_brightest_min": min(lit)}

    final = out["stages"]["shadows"]
    cause_of = {(n, x, y): c for c, xs in CAUSES.items() for (n, x, y) in xs}
    out["missed_by_cause"] = {c: 0 for c in CAUSES}
    for name in PHOTOS:
        for x, y in final[name]["missed_at"]:
            out["missed_by_cause"][cause_of[(name, x, y)]] += 1
    assert sum(out["missed_by_cause"].values()) == final["total"]["balls"] - final["total"]["found"]

    # How much the final count depends on the brightness floor.
    out["v_min_sweep"] = {}
    for v in SWEEP:
        ms = [match(find_balls(load(n), v, POCKET_V)["balls"], LABELS[n]) for n in PHOTOS]
        out["v_min_sweep"][v] = {
            "found": sum(m["found"] for m in ms),
            "false": sum(m["false"] for m in ms),
        }
    NUMBERS_JSON.write_text(json.dumps(out, indent=1) + "\n")
    logger.info(f"wrote {NUMBERS_JSON}")
    return out


# --- media: the opening image of every post, and the hub's animation ------------------

MEDIA_DIR = OUTPUT_DIR / "practice" / "m2" / "media"
#: The points lesson 1 reads, on pool.webp: (x, y) -> what is there.
SAMPLES = {
    "cloth, near": (400, 240),
    "cloth, far": (700, 95),
    "shadow": (790, 336),
    "green ball": (727, 155),
}
GREEN, RED, YELLOW = (80, 220, 60), (60, 60, 235), (40, 210, 240)


def _grey3(mask: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)


def _scored(img: np.ndarray, name: str, **kw) -> tuple[np.ndarray, dict]:
    """Found balls ringed green, missed balls red, false alarms yellow."""
    balls = find_balls(img, **kw)["balls"]
    m = match(balls, LABELS[name])
    out = img.copy()
    missed = {tuple(p) for p in m["missed_at"]}
    false = {tuple(p) for p in m["false_at"]}
    for x, y, r in LABELS[name]:
        if (x, y) in missed:
            cv2.circle(out, (x, y), r + 4, RED, 3, cv2.LINE_AA)
    for b in balls:
        colour = YELLOW if (b["x"], b["y"]) in false else GREEN
        cv2.circle(out, (round(b["x"]), round(b["y"])), round(b["r"]) + 4, colour, 3, cv2.LINE_AA)
    return out, m


def _contours(img: np.ndarray, **kw) -> np.ndarray:
    """Lesson 4's picture: the table outline in white, kept blobs green, rejected red."""
    r = find_balls(img, **kw)
    out = img.copy()
    cs, _ = cv2.findContours(r["table"], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(out, cs, -1, (255, 255, 255), 2, cv2.LINE_AA)
    cs, _ = cv2.findContours(r["objects"], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in cs:
        if cv2.contourArea(c) < MIN_AREA:
            continue
        colour = GREEN if roundness(c) > MIN_ROUND else RED
        cv2.drawContours(out, [c], -1, colour, 2, cv2.LINE_AA)
    return out


def build_media(out_dir: Path = MEDIA_DIR) -> dict[str, Path]:
    from fundamentals_lab.alignment.media import encode, label
    from fundamentals_lab.practice.m1 import _pair

    out_dir.mkdir(parents=True, exist_ok=True)
    img = load("pool")
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    n = experiments()
    first, _ = _scored(img, "pool", **STAGES["first"])
    final, _ = _scored(img, "pool", **STAGES["shadows"])
    points = img.copy()
    for x, y in SAMPLES.values():
        cv2.circle(points, (x, y), 7, (255, 255, 255), 2, cv2.LINE_AA)
    # Hue runs 0-179 in OpenCV; stretched to 0-255 so it reads as grey levels.
    hue = _grey3(cv2.convertScaleAbs(hsv[..., 0], alpha=255 / 179))
    cloth = cloth_mask(img)
    objects = cv2.bitwise_and(cv2.bitwise_not(cloth), table_mask(cloth))
    s = n["stages"]
    stills = {
        "hsv": _pair(points, hue, "pool.webp", "hue: the cloth is one grey"),
        "mask": _pair(img, _grey3(cloth), "pool.webp", "inRange: the cloth"),
        "clean": _pair(
            _grey3(objects),
            _grey3(clean(objects)),
            "not cloth, on the table",
            "opened, then closed",
        ),
        "contours": _pair(img, _contours(img), "pool.webp", "contours: round kept, others not"),
        "fixes": _pair(
            first,
            final,
            f"first try: {s['first']['pool']['found']} of 15",
            f"with the fixes: {s['shadows']['pool']['found']} of 15",
        ),
    }
    paths = {}
    for name, still in stills.items():
        p = out_dir / f"{name}.webp"
        cv2.imwrite(str(p), still, [cv2.IMWRITE_WEBP_QUALITY, 82])
        paths[name] = p
    # Hub: the failure gallery, four crops of the finished pipeline's output.
    cases = [
        ("pool-wide", (175, 235), 55, "touching: 3 balls, 1 blob"),
        ("pool", (707, 150), 45, "green ball on green cloth"),
        ("pool-close", (320, 490), 90, "shadow joined to the ball"),
        ("pool", (615, 120), 30, "green stripe: two white pieces"),
    ]
    tiles = []
    for name, (cx, cy), half, text in cases:
        scored, _ = _scored(load(name), name, **STAGES["shadows"])
        crop = scored[cy - half : cy + half, cx - round(half * 4 / 3) : cx + round(half * 4 / 3)]
        tile = cv2.resize(crop, (400, 300), interpolation=cv2.INTER_CUBIC)
        label(tile, text, (12, 30), 0.6)
        tiles.append(tile)
    gap = np.full((300, 6, 3), 40, np.uint8)
    row1 = np.hstack([tiles[0], gap, tiles[1]])
    row2 = np.hstack([tiles[2], gap, tiles[3]])
    gallery = np.vstack([row1, np.full((6, row1.shape[1], 3), 40, np.uint8), row2])
    p = out_dir / "failures.webp"
    cv2.imwrite(str(p), gallery, [cv2.IMWRITE_WEBP_QUALITY, 82])
    paths["failures"] = p

    # Hub: the pipeline step by step on pool.webp, then the result on all three photos.
    steps = [
        (img, "1. the photo"),
        (_grey3(cloth_mask(img, V_SHADOW)), "2. the cloth, by its hue"),
        (_grey3(table_mask(cloth_mask(img, V_SHADOW))), "3. the table"),
        (_grey3(find_balls(img, V_SHADOW, POCKET_V)["objects"]), "4. what is not cloth"),
        (_contours(img, **STAGES["shadows"]), "5. keep the round blobs"),
    ]
    frames = []
    for still, text in steps:
        f = still.copy()
        label(f, text, (16, 40), 0.9)
        frames += [f] * 9
    for name in PHOTOS:
        f, m = _scored(load(name), name, **STAGES["shadows"])
        text = f"{name}.webp: {m['found']} of {m['balls']} found, {m['false']} false"
        label(f, text, (16, 40), 0.9)
        frames += [f] * 15
    paths["find"] = encode(frames, 6, out_dir / "find.webp", 640, 50)
    for p in paths.values():
        logger.info(f"{p.name}: {p.stat().st_size / 1e3:.0f} KB")
    return paths


# --- covers: backgrounds for the title card (scripts/gen-thumbnails.mjs in the site) ---

SLUGS = {
    "hub": "find-balls-pool-table-opencv",
    "hsv": "opencv-find-hsv-range",
    "mask": "opencv-inrange-color-mask",
    "clean": "opencv-morphology-erode-dilate",
    "contours": "opencv-find-contours",
    "fixes": "opencv-color-detection-failures",
}


def build_covers(out_dir: Path | None = None) -> dict[str, Path]:
    from fundamentals_lab.practice.m1 import COVER_DIR, _cover_bg

    out_dir = out_dir or COVER_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    img = load("pool")
    final, _ = _scored(img, "pool", **STAGES["shadows"])
    cloth = cloth_mask(img)
    objects = cv2.bitwise_and(cv2.bitwise_not(cloth), table_mask(cloth))
    backgrounds = {
        SLUGS["hub"]: final,
        SLUGS["hsv"]: _grey3(
            cv2.convertScaleAbs(cv2.cvtColor(img, cv2.COLOR_BGR2HSV)[..., 0], alpha=255 / 179)
        ),
        SLUGS["mask"]: _grey3(cloth),
        SLUGS["clean"]: _grey3(clean(objects)),
        SLUGS["contours"]: _contours(img),
        SLUGS["fixes"]: cv2.imread(str(MEDIA_DIR / "failures.webp")),
    }
    paths = {}
    for slug, bg in backgrounds.items():
        p = out_dir / f"{slug}.png"
        cv2.imwrite(str(p), _cover_bg(bg))
        paths[slug] = p
    logger.info(f"wrote {len(paths)} cover backgrounds to {out_dir}")
    return paths
