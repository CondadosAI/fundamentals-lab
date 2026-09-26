"""The "in the wild" runs: the unit's measurements repeated on other photographs.

- a second court, outdoors, from a different camera (same channel, CC BY 3.0), where
  both near corners are outside the frame and are found as line intersections anyway;
- two overlapping photographs of Barcelona harbour from one viewpoint (public domain),
  stitched through a homography and blended three ways.

Every asset is registered in CondadosAI/cv-assets with its licence and sha256, and is
downloaded, never redistributed.
"""

from __future__ import annotations

import cv2
import numpy as np

from fundamentals_lab.alignment import court, evaluate, lines, transforms
from fundamentals_lab.config import DATA_DIR

OUTDOOR_DIR = DATA_DIR / "pickleball-outdoor"
PANO_DIR = DATA_DIR / "wild-panorama"

#: Approximate near-court corners on the outdoor frame, by eye to ~10 px. Both
#: baseline corners are outside the 1920x1080 frame (one below, one to the right).
OUTDOOR_SEED = ((835.0, 1208.0), (2016.0, 677.0), (250.0, 714.0), (1240.0, 540.0))


def _r(x, n=2):
    return round(float(x), n)


def outdoor_plate() -> np.ndarray:
    frames = sorted(OUTDOOR_DIR.glob("frame_*.png"))
    stack = np.stack([cv2.imread(str(f)) for f in frames])
    return np.median(stack, axis=0).astype(np.uint8)


def outdoor() -> dict:
    """Lessons 1-3 again, on a court nobody tuned anything for."""
    plate = outdoor_plate()
    px = lines.paint_pixels(plate, exclude_net=False)
    fit = lines.fit_court(px, seed_corners=OUTDOOR_SEED)
    L = {n: f.line for n, f in fit.lines.items()}
    img4 = np.array([fit.landmarks_img[n] for n in court.FIT_LANDMARKS])
    court4 = court.court_points(court.FIT_LANDMARKS)

    def direction(line):
        return np.degrees(np.arctan2(-line[0], line[1]))

    angle = abs(direction(L["side_left"]) - direction(L["side_right"]))
    angle = min(angle, 180 - angle)
    c0, i0 = court4 - court4.mean(0), img4 - img4.mean(0)
    M = transforms.fit_linear(c0, i0)
    lin_resid = np.linalg.norm(transforms.apply(M, c0) - i0, axis=1)

    H = transforms.dlt(img4, court4)  # image -> court
    three = [0, 1, 2]
    A3 = transforms.fit_affine(img4[three], court4[three])
    p, q = transforms.apply(np.linalg.inv(H), np.array(court.LINES["near_centre"]))
    centre_px, _ = lines.band(px, p, q)

    def score(Mi2c):
        out = {}
        for n in ("NBC", "NKC"):
            est = transforms.apply(Mi2c, fit.landmarks_img[n][None])[0]
            out[n] = _r(np.linalg.norm(est - court.court_point(n)) * 100)
        c = transforms.apply(Mi2c, centre_px)
        out["near_centre_line"] = _r(np.median(np.abs(c[:, 1] - court.W / 2)) * 100)
        return out

    vp = court.meet(L["side_left"], L["side_right"])
    vp_c = court.meet(L["side_left"], L["near_centre"])
    vp_px, vpc_px = court.dehomogenise(vp), court.dehomogenise(vp_c)
    base = fit.landmarks_img["NKC"]
    d1, d2 = vp_px - base, vpc_px - base
    vp_angle = abs(np.degrees(np.arctan2(d1[1], d1[0]) - np.arctan2(d2[1], d2[0])))
    scale = {
        n: {k: _r(v) for k, v in evaluate.scale_cm_per_px(H, fit.landmarks_img[n]).items()}
        for n in ("NBC", "NKL", "NKR")
    }
    return {
        "corners_img": {n: [_r(v) for v in fit.landmarks_img[n]] for n in court.FIT_LANDMARKS},
        "corners_outside_frame": [
            n
            for n in court.FIT_LANDMARKS
            if not (0 <= fit.landmarks_img[n][0] < 1920 and 0 <= fit.landmarks_img[n][1] < 1080)
        ],
        "sideline_angle_img_deg": _r(angle),
        "vp_along_court_img": [_r(v, 1) for v in vp_px],
        "vp_via_centre_line_img": [_r(v, 1) for v in vpc_px],
        "vp_agreement_px": _r(np.linalg.norm(vp_px - vpc_px), 1),
        "vp_direction_disagreement_deg": _r(vp_angle, 3),
        "best_2x2_residual_px": _r(lin_resid.mean(), 1),
        "affine_3pt": score(A3),
        "homography_4pt": score(H),
        "scale_cm_per_px": scale,
        "line_fit_rms_px": {n: _r(f.rms_px) for n, f in fit.lines.items()},
    }


def _pyramid_blend(a, b, mask, levels=5):
    """Burt-Adelson: blend each Laplacian band with a correspondingly blurred mask."""
    ga, gb, gm = [a.astype(np.float32)], [b.astype(np.float32)], [mask.astype(np.float32)]
    for _ in range(levels):
        ga.append(cv2.pyrDown(ga[-1]))
        gb.append(cv2.pyrDown(gb[-1]))
        gm.append(cv2.pyrDown(gm[-1]))
    out = None
    for lvl in range(levels, -1, -1):
        if lvl == levels:
            la, lb = ga[lvl], gb[lvl]
        else:
            size = (ga[lvl].shape[1], ga[lvl].shape[0])
            la = ga[lvl] - cv2.pyrUp(ga[lvl + 1], dstsize=size)
            lb = gb[lvl] - cv2.pyrUp(gb[lvl + 1], dstsize=size)
        m = gm[lvl][..., None]
        band = la * m + lb * (1 - m)
        out = band if out is None else cv2.pyrUp(out, dstsize=(band.shape[1], band.shape[0])) + band
    return np.clip(out, 0, 255).astype(np.uint8)


def panorama(save_dir=None) -> dict:
    """SIFT + RANSAC + DLT + backward warp, then three ways to combine the overlap."""
    a = cv2.imread(str(PANO_DIR / "BarcelonaHarbour1.jpg"))
    b = cv2.imread(str(PANO_DIR / "BarcelonaHarbour2.jpg"))
    scale = 1600 / a.shape[1]
    a = cv2.resize(a, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    b = cv2.resize(b, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    sift = cv2.SIFT_create()
    ka, da = sift.detectAndCompute(cv2.cvtColor(a, cv2.COLOR_BGR2GRAY), None)
    kb, db = sift.detectAndCompute(cv2.cvtColor(b, cv2.COLOR_BGR2GRAY), None)
    knn = cv2.BFMatcher().knnMatch(db, da, k=2)
    good = [m for m, n in knn if m.distance < 0.75 * n.distance]
    src = np.float64([kb[m.queryIdx].pt for m in good])
    dst = np.float64([ka[m.trainIdx].pt for m in good])
    H_b2a, inl = cv2.findHomography(src, dst, cv2.RANSAC, 3.0)
    inl = inl.ravel().astype(bool)
    H_ours = transforms.dlt(src[inl], dst[inl])
    reproj = np.linalg.norm(transforms.apply(H_ours, src[inl]) - dst[inl], axis=1)
    # canvas: a at the origin, b warped into a's frame, extended to hold both
    hb, wb = b.shape[:2]
    corners = transforms.apply(H_ours, np.array([[0, 0], [wb, 0], [wb, hb], [0, hb]], float))
    ha, wa = a.shape[:2]
    xmax = int(np.ceil(max(wa, corners[:, 0].max())))
    ymin = int(np.floor(min(0, corners[:, 1].min())))
    ymax = int(np.ceil(max(ha, corners[:, 1].max())))
    T = np.array([[1, 0, 0], [0, 1, -ymin], [0, 0, 1.0]])
    size = (xmax, ymax - ymin)
    A = cv2.warpPerspective(a, T, size)
    B = cv2.warpPerspective(b, T @ H_ours, size)
    ma = cv2.warpPerspective(np.ones(a.shape[:2], np.uint8), T, size) > 0
    mb = cv2.warpPerspective(np.ones(b.shape[:2], np.uint8), T @ H_ours, size) > 0
    both = ma & mb
    # exposure difference in the overlap
    ga = cv2.cvtColor(A, cv2.COLOR_BGR2GRAY).astype(np.float64)
    gb = cv2.cvtColor(B, cv2.COLOR_BGR2GRAY).astype(np.float64)
    ratio = ga[both].mean() / gb[both].mean()
    # hard cut down the middle of the overlap, feather, multiband
    xs = np.where(both.any(axis=0))[0]
    seam = int((xs.min() + xs.max()) / 2)
    left = np.zeros(size[::-1], bool)
    left[:, :seam] = True
    cut_mask = (ma & (left | ~mb)).astype(np.float32)
    da_ = cv2.distanceTransform(ma.astype(np.uint8), cv2.DIST_L2, 5)
    db_ = cv2.distanceTransform(mb.astype(np.uint8), cv2.DIST_L2, 5)
    feather_w = np.where(both, da_ / np.maximum(da_ + db_, 1e-6), ma.astype(np.float32))
    hard = (A * cut_mask[..., None] + B * (1 - cut_mask[..., None])).astype(np.uint8)
    feather = (A * feather_w[..., None] + B * (1 - feather_w[..., None])).astype(np.uint8)
    multi = _pyramid_blend(A, B, cut_mask)
    multi[~(ma | mb)] = 0

    def seam_step(img):
        g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float64)
        rows = both[:, seam - 1] & both[:, seam + 1]
        return float(np.mean(np.abs(g[rows, seam + 1] - g[rows, seam - 1])))

    def ref_step(img):
        g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float64)
        cols = [c for c in range(seam - 60, seam + 61, 20) if c != seam]
        vals = []
        for c in cols:
            rows = both[:, c - 1] & both[:, c + 1]
            vals.append(np.mean(np.abs(g[rows, c + 1] - g[rows, c - 1])))
        return float(np.mean(vals))

    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(save_dir / "pano_multiband.png"), multi)
        y0 = int(size[1] * 0.55)
        crop = slice(y0, y0 + 260), slice(seam - 200, seam + 200)
        cv2.imwrite(str(save_dir / "pano_seam_hard.png"), hard[crop])
        cv2.imwrite(str(save_dir / "pano_seam_feather.png"), feather[crop])
        cv2.imwrite(str(save_dir / "pano_seam_multiband.png"), multi[crop])
    return {
        "working_width_px": 1600,
        "keypoints": [len(ka), len(kb)],
        "ratio_test_matches": len(good),
        "ransac_inliers": int(inl.sum()),
        "inlier_ratio": _r(inl.mean(), 3),
        "inlier_reprojection_px": {
            "median": _r(np.median(reproj), 3),
            "p95": _r(np.percentile(reproj, 95), 3),
        },
        "overlap_frac_of_canvas_covered": _r(both.sum() / (ma | mb).sum(), 3),
        "exposure_ratio_a_over_b": _r(ratio, 3),
        "seam_step_grey": {
            "hard_cut": _r(seam_step(hard)),
            "feather": _r(seam_step(feather)),
            "multiband": _r(seam_step(multi)),
        },
        "reference_step_grey_near_seam": {"hard_cut": _r(ref_step(hard))},
    }
