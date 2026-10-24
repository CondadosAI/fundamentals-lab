"""Every number unit 3.3 publishes, one function per post.

The unit measures on the Barcelona harbour pair at 960 px. Warps we apply ourselves have
an exact answer for every pixel; the real pair is scored against a homography fitted by
ORB (`scene.reference_homography`), so SIFT is never its own answer key. templeRing is
the "in the wild" photo. Each function returns a JSON-able dict;
`cli/sift_experiments.py` writes them to `output/sift_numbers.json`. Worked examples are
computed from the values the posts print.
"""

from __future__ import annotations

import time

import cv2
import numpy as np

from fundamentals_lab.config import (
    DARK_FACTORS,
    HARRIS_DESCRIBE_SIZE,
    MATCH_TOL_PX,
    PATCH_SIDE,
    RATIO,
    RATIOS,
    REPEAT_BORDER_PX,
    REPEAT_COUNTS,
    REPEAT_SIZE_OCTAVES,
    REPEAT_TOL_PX,
    REPEAT_TOLS,
    REPEAT_TOP_N,
    SIFT_SEED,
    TEMPLE_F_PX,
    TEMPLE_STEP_DEG,
    TEMPLE_VIEWS,
    UPRIGHT_ROTATIONS,
    WARP_BLUR,
    WARP_NOISE,
    WARP_ROTATIONS,
    WARP_SCALES,
)
from fundamentals_lab.features import scene
from fundamentals_lab.features import sift_scratch as S


def _r(x, n=3):
    return None if x is None else round(float(x), n)


def _spread(values) -> dict:
    v = np.asarray(values, float)
    if not v.size:
        return {"n": 0}
    return {
        "n": int(v.size),
        "median": _r(np.median(v)),
        "p05": _r(np.percentile(v, 5)),
        "p95": _r(np.percentile(v, 95)),
        "max": _r(v.max()),
    }


def warps(img: np.ndarray) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """The known changes every lesson is tested against, with their exact pixel maps."""
    out = {f"rotate {d}°": scene.rotate_scale(img, d) for d in WARP_ROTATIONS}
    out |= {f"scale ×{s}": scene.rotate_scale(img, 0, s) for s in WARP_SCALES}
    out[f"blur σ {WARP_BLUR:g}"] = (scene.photometric(img, blur=WARP_BLUR), np.eye(3))
    out[f"noise σ {WARP_NOISE:g}"] = (scene.photometric(img, noise=WARP_NOISE), np.eye(3))
    return out


def _common(ka, kb, H, shape_a, shape_b):
    """The keypoints of each image that land inside the other one, away from the border."""
    pa = np.array([k.pt for k in ka]).reshape(-1, 2)
    pb = np.array([k.pt for k in kb]).reshape(-1, 2)
    ia = scene.inside(scene.apply(H, pa), shape_b, REPEAT_BORDER_PX)
    ib = scene.inside(scene.apply(np.linalg.inv(H), pb), shape_a, REPEAT_BORDER_PX)
    keep_a = [k for k, m in zip(ka, ia, strict=True) if m]
    return keep_a, [k for k, m in zip(kb, ib, strict=True) if m]


def repeatability(ka, kb, H, shape_a, shape_b, tol=REPEAT_TOL_PX, size_octaves=None) -> dict:
    """Schmid's repeatability: one-to-one correspondences / min(nA, nB) in the common region.

    The tolerance is in the pixels of the image the points are found in: after a zoom out
    a pixel of the smaller image covers more of the scene, and a detector cannot place a
    point more finely than its own grid. With `size_octaves`, a correspondence also needs the
    two sizes to agree (after the warp's scale) within that many octaves.
    """
    ka, kb = _common(ka, kb, H, shape_a, shape_b)
    if not ka or not kb:
        return {"rate": 0.0, "hits": 0, "na": len(ka), "nb": len(kb)}
    s = float(np.sqrt(abs(np.linalg.det(H[:2, :2]))))
    q = scene.apply(H, [k.pt for k in ka])
    pb = np.array([k.pt for k in kb])
    d = np.linalg.norm(q[:, None] - pb[None], axis=2)
    if size_octaves is not None:
        sa = np.array([k.size for k in ka]) * s
        sb = np.array([k.size for k in kb])
        bad = np.abs(np.log2(sb[None] / sa[:, None])) > size_octaves
        d = np.where(bad, np.inf, d)
    tol_px = tol
    used, hits = set(), 0
    for i in np.argsort(d.min(1)):
        order = np.argsort(d[i])
        for j in order[:3]:
            if d[i, j] > tol_px:
                break
            if j not in used:
                used.add(int(j))
                hits += 1
                break
    return {"rate": _r(hits / min(len(ka), len(kb))), "hits": hits, "na": len(ka), "nb": len(kb)}


def _ratio_matches(da, db, ratio=RATIO):
    out = []
    for m, n in scene.knn(da, db):
        if m.distance < ratio * n.distance:
            out.append(m)
    return out


def _scored(ka, kb, matches, H, tol=MATCH_TOL_PX) -> dict:
    if not matches:
        return {"matches": 0, "correct": 0, "precision": None}
    p = np.array([ka[m.queryIdx].pt for m in matches])
    q = np.array([kb[m.trainIdx].pt for m in matches])
    ok = np.linalg.norm(scene.apply(H, p) - q, axis=1) <= tol
    return {"matches": len(matches), "correct": int(ok.sum()), "precision": _r(ok.mean())}


# ---------------------------------------------------------------- hub ---------


def _patches(img, kps, side=PATCH_SIDE):
    h = side // 2
    pad = cv2.copyMakeBorder(img, h, h, h, h, cv2.BORDER_REFLECT).astype(np.float32)
    out = []
    for k in kps:
        x, y = int(round(k.pt[0])) + h, int(round(k.pt[1])) + h
        p = pad[y - h : y + h + 1, x - h : x + h + 1].ravel()
        p = p - p.mean()
        out.append(p / max(np.linalg.norm(p), 1e-6))
    return np.array(out, np.float32)


def hub(_=None) -> dict:
    """The headline, and the naive representation (a raw patch) against SIFT's."""
    a, b = scene.pair()
    H, ref = scene.reference_homography()
    sift = scene.sift()
    raw_a = sift.detect(a, None)
    ka, da = sift.compute(a, scene.dedupe(raw_a))
    kb, db = sift.compute(b, scene.dedupe(sift.detect(b, None)))
    real = _scored(ka, kb, _ratio_matches(da, db), H)
    rows = {}
    for name, (deg, s) in {
        "the real pair": (0, 1.0),
        "B rotated 20°": (20, 1.0),
        "B scaled ×0.71": (0, 0.71),
    }.items():
        bw, M = scene.rotate_scale(b, deg, s)
        HM = M @ H
        kbw = scene.dedupe(sift.detect(bw, None))
        kbw, dbw = sift.compute(bw, kbw)
        pa_, pb_ = _patches(a, ka), _patches(bw, kbw)
        rows[name] = {
            "patch": _scored(ka, kbw, _ratio_matches(pa_, pb_), HM),
            "sift": _scored(ka, kbw, _ratio_matches(da, dbw), HM),
        }
    return {
        "image_px": [a.shape[1], a.shape[0]],
        "keypoints": [len(raw_a), len(ka)],
        "keypoints_b": len(kb),
        "reference": ref,
        "real_pair": real,
        "patch_vs_sift": rows,
        "patch_side": PATCH_SIDE,
    }


# ---------------------------------------------------------------- lesson 1 ----


def keypoints_lesson(_=None) -> dict:
    """Repeatability (position, then position and size), and what the size buys."""
    a, _b = scene.pair()
    out_pos, out_size = {}, {}
    sa = scene.sift_points(a, REPEAT_TOP_N)
    ha = scene.harris_points(a, REPEAT_TOP_N)
    for name, (img, H) in warps(a).items():
        sb = scene.sift_points(img, REPEAT_TOP_N)
        hb = scene.harris_points(img, REPEAT_TOP_N)
        out_pos[name] = {
            "harris": repeatability(ha, hb, H, a.shape, img.shape),
            "sift": repeatability(sa, sb, H, a.shape, img.shape),
        }
        out_size[name] = repeatability(
            sa, sb, H, a.shape, img.shape, size_octaves=REPEAT_SIZE_OCTAVES
        )
    sens = {}
    for name in ("rotate 45°", "scale ×0.5"):
        img, H = warps(a)[name]
        sens[name] = {
            "tolerance_px": {
                str(t): {
                    "harris": repeatability(
                        ha, scene.harris_points(img, REPEAT_TOP_N), H, a.shape, img.shape, tol=t
                    )["rate"],
                    "sift": repeatability(
                        sa, scene.sift_points(img, REPEAT_TOP_N), H, a.shape, img.shape, tol=t
                    )["rate"],
                }
                for t in REPEAT_TOLS
            },
            "top_n": {
                str(n): {
                    "harris": repeatability(
                        scene.harris_points(a, n),
                        scene.harris_points(img, n),
                        H,
                        a.shape,
                        img.shape,
                    )["rate"],
                    "sift": repeatability(
                        scene.sift_points(a, n), scene.sift_points(img, n), H, a.shape, img.shape
                    )["rate"],
                }
                for n in REPEAT_COUNTS
            },
        }
    # What the size buys: describe each detector's points upright and match across a zoom.
    # Harris has no size, so its points are described at one fixed size.
    sift = scene.sift()
    payoff = {}
    for s in WARP_SCALES:
        img, H = scene.rotate_scale(a, 0, s)
        res = {}
        for det in ("harris", "sift"):
            if det == "harris":
                kA = scene.harris_points(a, REPEAT_TOP_N, HARRIS_DESCRIBE_SIZE)
                kB = scene.harris_points(img, REPEAT_TOP_N, HARRIS_DESCRIBE_SIZE)
            else:
                kA, kB = scene.sift_points(a, REPEAT_TOP_N), scene.sift_points(img, REPEAT_TOP_N)
                for k in kA + kB:
                    k.angle = 0.0
            kA, dA = sift.compute(a, kA)
            kB, dB = sift.compute(img, kB)
            res[det] = _scored(kA, kB, _ratio_matches(dA, dB), H)
        payoff[f"scale ×{s}"] = res
    dark = {}
    for f in DARK_FACTORS:
        img = scene.photometric(a, gain=f)
        dark[str(f)] = {
            "sift_points": len(scene.sift_points(img)),
            "harris_repeatability": repeatability(
                ha, scene.harris_points(img, REPEAT_TOP_N), np.eye(3), a.shape, img.shape
            )["rate"],
            "sift_repeatability": repeatability(
                sa, scene.sift_points(img, REPEAT_TOP_N), np.eye(3), a.shape, img.shape
            )["rate"],
        }
    ex = out_pos["rotate 45°"]["sift"]
    return {
        "top_n": REPEAT_TOP_N,
        "tolerance_px": REPEAT_TOL_PX,
        "size_octaves": REPEAT_SIZE_OCTAVES,
        "position": out_pos,
        "position_and_size": out_size,
        "sensitivity": sens,
        "size_payoff": payoff,
        "harris_describe_size": HARRIS_DESCRIBE_SIZE,
        "darkening": dark,
        "worked_example": {
            "warp": "rotate 45°",
            **ex,
            "rate_from_printed": _r(ex["hits"] / min(ex["na"], ex["nb"])),
        },
    }


# ---------------------------------------------------------------- lesson 2 ----


def scale_space_lesson(_=None) -> dict:
    """The pyramid's size, DoG against the scale-normalized Laplacian, and scale tracking."""
    a, _b = scene.pair()
    pyr = S.build_pyramid(a)
    n_oct = len(pyr.gauss)
    k = 2 ** (1 / S.LAYERS)
    shapes = [list(g[0].shape[::-1]) for g in pyr.gauss]
    # DoG ~ (k - 1) sigma^2 Laplacian(G): compare on octave 0's middle level.
    i = 2
    sig = S.SIGMA * k**i  # blur of level i, in octave-0 pixels (the doubled image)
    g = pyr.gauss[0][i].astype(np.float64)
    lap = cv2.Laplacian(g, cv2.CV_64F, ksize=1)
    approx = (k - 1) * sig**2 * lap
    dog = pyr.dog[0][i].astype(np.float64)
    m = slice(20, -20)
    dd, aa = dog[m, m].ravel(), approx[m, m].ravel()
    corr = float(np.corrcoef(dd, aa)[0, 1])
    rel = float(np.linalg.norm(dd - aa) / np.linalg.norm(dd))
    # Discs of known radius: the sigma at which sigma^2 |Laplacian| peaks at the centre.
    discs = {}
    sigmas = np.geomspace(1, 40, 200)
    for r in (4, 8, 16):
        img = np.zeros((257, 257), np.float64)
        cv2.circle(img, (128, 128), r, 1.0, -1, lineType=cv2.LINE_AA)
        resp = []
        for s in sigmas:
            gs = cv2.GaussianBlur(img, (0, 0), s)
            resp.append(abs(s**2 * cv2.Laplacian(gs, cv2.CV_64F, ksize=1)[128, 128]))
        discs[str(r)] = {
            "best_sigma": _r(sigmas[int(np.argmax(resp))], 2),
            "r_over_sqrt2": _r(r / np.sqrt(2), 2),
        }
    # Scale tracking: the same points, zoomed out, should come back with sizes times s.
    track = {}
    sift = scene.sift()
    for s in WARP_SCALES:
        img, H = scene.rotate_scale(a, 0, s)
        kA = scene.sift_points(a)
        kB = scene.sift_points(img)
        kA, dA = sift.compute(a, kA)
        kB, dB = sift.compute(img, kB)
        ms = _ratio_matches(dA, dB)
        p = np.array([kA[x.queryIdx].pt for x in ms])
        q = np.array([kB[x.trainIdx].pt for x in ms])
        ok = np.linalg.norm(scene.apply(H, p) - q, axis=1) <= MATCH_TOL_PX * max(s, 0.5)
        ratio = np.array([kB[x.trainIdx].size / kA[x.queryIdx].size for x in ms])[ok]
        track[f"×{s}"] = {"correct_matches": int(ok.sum()), "size_ratio": _spread(ratio)}
    sizes = np.array([kp.size for kp in scene.sift_points(a)])
    octs = np.array([((kp.octave & 255) ^ 128) - 128 for kp in scene.sift_points(a)])
    hist = {str(o): int((octs == o).sum()) for o in sorted(set(octs.tolist()))}
    return {
        "image_px": [a.shape[1], a.shape[0]],
        "doubled_px": shapes[0],
        "octaves": n_oct,
        "octave_formula": {
            "min_side_doubled": int(min(a.shape) * 2),
            "log2": _r(np.log2(min(a.shape) * 2), 3),
            "rounded": int(round(np.log2(min(a.shape) * 2) - 2)),
        },
        "gaussians_per_octave": S.LAYERS + 3,
        "dog_per_octave": S.LAYERS + 2,
        "dog_levels": n_oct * (S.LAYERS + 2),
        "searchable_levels": n_oct * S.LAYERS,
        "octave_shapes": shapes,
        "k": _r(k, 4),
        "level_sigmas": [_r(S.SIGMA * k**j, 3) for j in range(S.LAYERS + 3)],
        "dog_vs_log": {
            "octave": 0,
            "level": i,
            "sigma": _r(sig, 3),
            "correlation": _r(corr, 4),
            "relative_residual": _r(rel, 3),
        },
        "discs": discs,
        "scale_tracking": track,
        "sizes_px": _spread(sizes),
        "size_range_px": [_r(sizes.min(), 2), _r(sizes.max(), 1)],
        "points_per_octave": hist,
    }


# ---------------------------------------------------------------- lesson 3 ----


def _edge_example(run: S.Run, kind: str):
    """One real extremum passed and one dropped by the edge test, with its curvature numbers."""
    pyr = run.pyramid
    best = None
    for e in S.raw_extrema(pyr):
        stage, f = S.refine(pyr, e)
        if stage not in ("kept", "edge") or (kind == "kept") != (stage == "kept"):
            continue
        _g, H = S._derivatives(pyr.dog[f.octave], f.layer, f.r, f.c)
        tr, det = H[0, 0] + H[1, 1], H[0, 0] * H[1, 1] - H[0, 1] ** 2
        if det <= 0:
            continue
        score = tr * tr / det
        if f.octave == 1 and (best is None or abs(f.response) > abs(best[0].response)):
            best = (f, H, tr, det, score)
    f, H, tr, det, score = best
    return {
        "pt": [_r(v, 1) for v in f.pt],
        "dxx": _r(H[0, 0], 4),
        "dyy": _r(H[1, 1], 4),
        "dxy": _r(H[0, 1], 4),
        "trace": _r(tr, 4),
        "det": _r(det, 6),
        "tr2_over_det": _r(score, 2),
        "limit": _r((S.EDGE + 1) ** 2 / S.EDGE, 2),
        "kept": bool(score < (S.EDGE + 1) ** 2 / S.EDGE),
    }


def detector_lesson(_=None, img=None) -> dict:
    """The detector from scratch, stage by stage, checked against cv2.SIFT."""
    a = scene.pair()[0] if img is None else img
    t = time.perf_counter()
    run = S.detect(a)
    t_scratch = time.perf_counter() - t
    t = time.perf_counter()
    cv_kps = scene.sift().detect(a, None)
    t_cv = time.perf_counter() - t
    distinct = len({(e.octave, e.layer, e.r, e.c) for e in run.kept})
    P = np.float32([k.pt for k in run.keypoints()])
    Q = np.float32([k.pt for k in cv_kps])
    mm = cv2.BFMatcher(cv2.NORM_L2).match(Q, P)
    d = np.array([m.distance for m in mm])
    j = np.array([m.trainIdx for m in mm])
    close = d < 0.1
    size_ratio = np.array([cv_kps[i].size / run.kept[j[i]].size for i in np.nonzero(close)[0]])
    # A location can carry several orientations; compare each OpenCV angle with the
    # closest of ours at the same place.
    by_place: dict[tuple, list[float]] = {}
    for e in run.kept:
        by_place.setdefault((round(e.pt[0], 1), round(e.pt[1], 1)), []).append(e.angle)
    ang = []
    for i in np.nonzero(close)[0]:
        e = run.kept[j[i]]
        cands = np.array(by_place[(round(e.pt[0], 1), round(e.pt[1], 1))])
        ang.append(np.min(np.abs((cv_kps[i].angle - cands + 180) % 360 - 180)))
    ang = np.array(ang)
    offsets = np.array([np.hypot(e.xc, e.xr) * 2 ** (e.octave - 1) for e in run.kept])
    copies = len(run.kept) - distinct
    multi = {}
    for e in run.kept:
        multi[(e.octave, e.layer, e.r, e.c)] = multi.get((e.octave, e.layer, e.r, e.c), 0) + 1
    out = {
        "image_px": [a.shape[1], a.shape[0]],
        "extrema": run.extrema,
        "dropped": run.dropped,
        "survived_tests": run.extrema - sum(run.dropped.values()),
        "after_tests": distinct,
        "merged_duplicates": run.extrema - sum(run.dropped.values()) - distinct,
        "keypoints": len(run.kept),
        "orientation_copies": copies,
        "points_with_2plus_orientations": sum(1 for v in multi.values() if v > 1),
        "opencv_keypoints": len(cv_kps),
        "opencv_found_by_scratch_0_1px": _r(close.mean(), 4),
        "size_ratio_median": _r(np.median(size_ratio), 6),
        "angle_diff_deg_p95": _r(np.percentile(ang, 95), 5),
        "subpixel_shift_px": _spread(offsets),
        "seconds": {"scratch": _r(t_scratch, 2), "opencv": _r(t_cv, 3)},
    }
    if img is None:
        out["edge_test"] = {
            "kept": _edge_example(run, "kept"),
            "dropped": _edge_example(run, "edge"),
        }
        strongest = max(run.kept, key=lambda e: abs(e.response))
        hist = S.orientation_histogram(
            run.pyramid.gauss[strongest.octave][strongest.layer],
            strongest.r,
            strongest.c,
            strongest.scale,
        )
        out["strongest"] = {
            "pt": [_r(v, 1) for v in strongest.pt],
            "size": _r(strongest.size, 2),
            "angle_opencv_convention": _r(strongest.angle, 1),
            "gradient_direction_deg": _r((360 - strongest.angle) % 360, 1),
            "histogram": [_r(v, 1) for v in hist],
            "orientations": [_r(x, 1) for x in S.orientations(hist)],
            "contrast": _r(strongest.response, 4),
        }
        out["floor_dn"] = float(np.floor(0.5 * S.CONTRAST / S.LAYERS * 255))
        out["contrast_limit"] = _r(S.CONTRAST / S.LAYERS, 4)
    return out


# ---------------------------------------------------------------- lesson 4 ----


def _scratch_set(img, upright=False, **kw):
    run = S.detect(img)
    seen, kept = set(), []
    for e in sorted(run.kept, key=lambda e: -abs(e.response)):
        key = (e.octave, e.layer, e.r, e.c)
        if key in seen:
            continue
        seen.add(key)
        if upright:
            e.angle = 0.0
        kept.append(e)
    run.kept = kept
    return run.keypoints(), run.descriptors(**kw)


def descriptor_lesson(_=None, img=None) -> dict:
    """Ablations: each design step of the descriptor removed, under the change it is for."""
    a = scene.pair()[0] if img is None else img
    out = {}
    # 1. Agreement of our descriptor with OpenCV's, on the same keypoints.
    run = S.detect(a)
    D = run.descriptors()
    _, Dc = scene.sift().compute(a, run.keypoints())
    cos = np.sum(D * Dc, 1) / np.linalg.norm(D, axis=1) / np.linalg.norm(Dc, axis=1)
    out["agreement"] = {
        "keypoints": len(D),
        "identical": _r(np.all(Dc == D, axis=1).mean(), 4),
        "cosine_median": _r(np.median(cos), 6),
        "cosine_p01": _r(np.percentile(cos, 1), 4),
    }
    # 2. Orientation: upright descriptors against a rotated copy.
    up = {}
    for deg in UPRIGHT_ROTATIONS:
        img2, H = scene.rotate_scale(a, deg)
        res = {}
        for name, upright in (("oriented", False), ("upright", True)):
            kA, dA = _scratch_set(a, upright)
            kB, dB = _scratch_set(img2, upright)
            res[name] = _scored(kA, kB, _ratio_matches(dA, dB), H)
        up[f"{deg}°"] = res
    out["orientation"] = up
    # 3. Normalization against a contrast change, clipping against a non-linear one.
    tests = {
        "contrast ×0.5": {"gain": 0.5},
        "gamma 0.5": {"gamma": 0.5},
        "saturate ×1.8": {"gain": 1.8},
    }
    ab = {}
    for tname, kw in tests.items():
        img2 = scene.photometric(a, **kw)
        res = {}
        for name, flags in {
            "full": {},
            "no clip": {"clip": False},
            "no normalization": {"normalize": False},
        }.items():
            kA, dA = _scratch_set(a, **flags)
            kB, dB = _scratch_set(img2, **flags)
            res[name] = _scored(kA, kB, _ratio_matches(dA, dB), np.eye(3))
        ab[tname] = res
    out["photometric"] = ab
    clipped = []
    for e in run.kept[:2000]:
        raw = S.descriptor(
            run.pyramid.gauss[e.octave][e.layer],
            e.c + e.xc,
            e.r + e.xr,
            e.angle,
            e.scale,
            normalize=False,
        )
        u = raw / max(np.linalg.norm(raw), 1e-9)
        clipped.append(int((u > S.DESC_CLIP).sum()))
    out["clip"] = {
        "threshold": S.DESC_CLIP,
        "descriptors_with_a_clipped_entry": _r(np.mean(np.array(clipped) > 0), 3),
        "entries_clipped_median_when_any": _r(
            np.median([c for c in clipped if c]) if any(clipped) else 0, 1
        ),
    }
    if img is None:
        strongest = max(run.kept, key=lambda e: abs(e.response))
        v = S.descriptor(
            run.pyramid.gauss[strongest.octave][strongest.layer],
            strongest.c + strongest.xc,
            strongest.r + strongest.xr,
            strongest.angle,
            strongest.scale,
        )
        out["strongest"] = {
            "pt": [_r(x, 1) for x in strongest.pt],
            "size": _r(strongest.size, 2),
            "descriptor": [int(x) for x in v],
            "window_px": _r(
                S.DESC_SCL
                * strongest.scale
                * (S.DESC_D + 1)
                * np.sqrt(2)
                * 2 ** (strongest.octave - 1),
                1,
            ),
            "cell_px": _r(S.DESC_SCL * strongest.scale * 2 ** (strongest.octave - 1), 2),
        }
    return out


# ---------------------------------------------------------------- lesson 5 ----


def matching_lesson(_=None) -> dict:
    """The ratio test on the real pair, scored against the ORB homography."""
    a, b = scene.pair()
    H, ref = scene.reference_homography()
    sift = scene.sift()
    ka, da = sift.compute(a, scene.sift_points(a))
    kb, db = sift.compute(b, scene.sift_points(b))
    kn = scene.knn(da, db)
    pa = np.array([ka[m.queryIdx].pt for m, _ in kn])
    pb = np.array([kb[m.trainIdx].pt for m, _ in kn])
    d1 = np.array([m.distance for m, _ in kn])
    d2 = np.array([n.distance for _, n in kn])
    r = d1 / d2
    err = np.linalg.norm(scene.apply(H, pa) - pb, axis=1)
    ok = err <= MATCH_TOL_PX
    over = scene.inside(scene.apply(H, pa), b.shape)
    sweep = {}
    for t in RATIOS:
        s = r < t if t < 1 else np.ones_like(r, bool)
        sweep[str(t)] = {
            "kept": int(s.sum()),
            "in_overlap": int((s & over).sum()),
            "correct": int((s & ok).sum()),
            "precision": _r((s & ok).sum() / s.sum()),
            "recall_of_correct": _r((s & ok).sum() / ok.sum()),
        }
    keep = r < RATIO
    n = int(keep.sum())
    by_d1 = np.argsort(d1)[:n]
    # Absolute threshold, tuned after the fact to the count the ratio test keeps.
    absolute = {"kept": n, "threshold": _r(d1[by_d1[-1]], 1), "correct": int(ok[by_d1].sum())}
    # Mutual nearest neighbour (cross-check), the usual alternative.
    back = cv2.BFMatcher(cv2.NORM_L2).match(db, da)
    mutual = np.array([back[m.trainIdx].trainIdx == m.queryIdx for m, _ in kn])
    cross = {"kept": int(mutual.sum()), "correct": int((mutual & ok).sum())}
    both = {"kept": int((mutual & keep).sum()), "correct": int((mutual & keep & ok).sum())}
    clear = int(np.argmin(np.where(keep, r, np.inf)))
    amb = int(np.argmax(np.where(over, r, -np.inf)))
    # SIFT's own homography against the reference, on the points both see.
    cv2.setRNGSeed(SIFT_SEED)
    Hs, mask = cv2.findHomography(np.float32(pa[keep]), np.float32(pb[keep]), cv2.USAC_MAGSAC, 2.0)
    test = pa[keep][mask.ravel().astype(bool)]
    gap = np.linalg.norm(scene.apply(Hs, test) - scene.apply(H, test), axis=1)
    lab = [
        {
            "a": [_r(pa[i, 0], 1), _r(pa[i, 1], 1)],
            "b": [_r(pb[i, 0], 1), _r(pb[i, 1], 1)],
            "d1": _r(d1[i], 1),
            "d2": _r(d2[i], 1),
            "ok": bool(ok[i]),
        }
        for i in np.nonzero(over)[0]
    ]
    return {
        "keypoints": [len(ka), len(kb)],
        "reference": ref,
        "overlap_queries": int(over.sum()),
        "correct_anywhere": int(ok.sum()),
        "best_distance_median": _r(np.median(d1), 1),
        "sweep": sweep,
        "absolute_same_count": absolute,
        "cross_check": cross,
        "cross_check_and_ratio": both,
        "clearest": {
            "d1": _r(d1[clear], 1),
            "d2": _r(d2[clear], 1),
            "ratio": _r(round(d1[clear], 1) / round(d2[clear], 1)),
            "ok": bool(ok[clear]),
        },
        "most_ambiguous": {
            "d1": _r(d1[amb], 1),
            "d2": _r(d2[amb], 1),
            "ratio": _r(round(d1[amb], 1) / round(d2[amb], 1)),
            "ok": bool(ok[amb]),
        },
        "sift_homography": {
            "inliers": int(mask.sum()),
            "of": n,
            "gap_to_reference_px": _spread(gap),
        },
        "lab_matches": lab,
    }


# ---------------------------------------------------------------- in the wild -


def wild(_=None) -> dict:
    """templeRing: lessons 1-4's headline on another photo, and lesson 5's viewpoint sweep."""
    t = scene.temple(1)
    sa, ha = scene.sift_points(t, 500), scene.harris_points(t, 500)
    rep = {}
    for name, (img, H) in {
        "rotate 45°": scene.rotate_scale(t, 45),
        "scale ×0.5": scene.rotate_scale(t, 0, 0.5),
    }.items():
        rep[name] = {
            "harris": repeatability(ha, scene.harris_points(img, 500), H, t.shape, img.shape)[
                "rate"
            ],
            "sift": repeatability(sa, scene.sift_points(img, 500), H, t.shape, img.shape)["rate"],
            "sift_with_size": repeatability(
                sa,
                scene.sift_points(img, 500),
                H,
                t.shape,
                img.shape,
                size_octaves=REPEAT_SIZE_OCTAVES,
            )["rate"],
        }
    pyr_oct = S.n_octaves(t.shape)
    det = detector_lesson(img=t)
    desc = descriptor_lesson(img=t)
    sift = scene.sift()
    k1, d1 = sift.compute(t, scene.sift_points(t))
    sweep = []
    for v in TEMPLE_VIEWS[1:]:
        tv = scene.temple(v)
        k2, d2 = sift.compute(tv, scene.sift_points(tv))
        ms = _ratio_matches(d1, d2)
        p = np.float32([k1[m.queryIdx].pt for m in ms])
        q = np.float32([k2[m.trainIdx].pt for m in ms])
        inl = 0
        if len(ms) >= 8:
            cv2.setRNGSeed(SIFT_SEED)
            _F, mask = cv2.findFundamentalMat(p, q, cv2.FM_RANSAC, TEMPLE_F_PX, 0.999)
            inl = int(mask.sum()) if mask is not None else 0
        sweep.append(
            {
                "view": v,
                "degrees": _r((v - 1) * TEMPLE_STEP_DEG, 1),
                "matches": len(ms),
                "inliers": inl,
            }
        )
    return {
        "image_px": [t.shape[1], t.shape[0]],
        "repeatability_top500": rep,
        "octaves": pyr_oct,
        "detector": {
            k: det[k]
            for k in (
                "extrema",
                "dropped",
                "after_tests",
                "keypoints",
                "orientation_copies",
                "opencv_keypoints",
                "opencv_found_by_scratch_0_1px",
            )
        },
        "orientation": desc["orientation"]["45°"],
        "viewpoint": sweep,
    }
