"""Every number units 3.5 and 4.1-L1 publish, one function per lesson.

Each returns a JSON-able dict; `cli/align_experiments.py` writes them all to
`output/alignment_numbers.json`.
"""

from __future__ import annotations

import cv2
import numpy as np

from fundamentals_lab.alignment import court, distortion, evaluate, lines, plate, transforms, warp
from fundamentals_lab.config import (
    ALIGN_SEED,
    ALIGN_SEED_CORNERS,
    DLT_NOISE_PX,
    DLT_NOISE_TRIALS,
    SEED_PERTURB_PX,
    SEED_PERTURB_TRIALS,
)


def _r(x, n=3):
    return None if x is None else round(float(x), n)


class Scene:
    """The plate, its paint, the court fit and the held-out evidence, computed once."""

    def __init__(self):
        self.plate = plate.load_plate()
        self.frame = plate.load_frame()
        self.pixels = lines.paint_pixels(self.plate)
        self.fit = lines.fit_court(self.pixels)
        self.H_court2img = self.fit.H_court2img
        self.H_img2court = np.linalg.inv(self.H_court2img)
        self.evidence = evaluate.evidence_pixels(self.pixels, self.H_court2img)
        self.fit_img = np.array([self.fit.landmarks_img[n] for n in court.FIT_LANDMARKS])
        self.fit_court = court.court_points(court.FIT_LANDMARKS)

    def held_out(self, H_img2court):
        return {
            k: (_r(v, 2) if not isinstance(v, dict) else v)
            for k, v in evaluate.held_out(
                H_img2court, self.fit.landmarks_img, self.evidence
            ).items()
        }


# --- The scene itself: what every lesson and lab is seeded with -------------------


def scene(s: Scene) -> dict:
    return {
        "landmarks_img": {n: [_r(v, 2) for v in p] for n, p in s.fit.landmarks_img.items()},
        "landmarks_court_m": {n: [_r(v, 4) for v in court.court_point(n)] for n in court.LANDMARKS},
        "fit_landmarks": list(court.FIT_LANDMARKS),
        "H_court2img": [[_r(v, 6) for v in r] for r in s.H_court2img / s.H_court2img[2, 2]],
        "frame": 45000,
    }


# --- 4.1 L1: homogeneous coordinates --------------------------------------------


def homogeneous(s: Scene) -> dict:
    L = {n: f.line for n, f in s.fit.lines.items()}
    vp_sides = court.meet(L["side_left"], L["side_right"])
    vp_centre = court.meet(L["side_left"], L["near_centre"])
    vp_across = court.meet(L["near_base"], L["near_kitchen"])
    horizon = np.cross(vp_sides, vp_across)
    horizon = horizon / np.linalg.norm(horizon[:2])
    # On the ground the two sidelines are parallel: their meet has w = 0.
    ground = court.meet(
        court.homogeneous_line(*court.LINES["side_left"]),
        court.homogeneous_line(*court.LINES["side_right"]),
    )
    nbl = s.fit.landmarks_img["NBL"]
    lm = s.fit.landmarks_img
    # The worked example: the same lines as joins of two corners, unnormalised.
    join_left = np.cross([*lm["NBL"], 1.0], [*lm["NKL"], 1.0])
    join_right = np.cross([*lm["NBR"], 1.0], [*lm["NKR"], 1.0])
    vp_raw = np.cross(join_left, join_right)
    # Three lines parallel on the ground should share one vanishing point. Seen from
    # NKC, the directions to the two estimates differ by this angle.
    a1 = np.subtract(court.dehomogenise(vp_sides), lm["NKC"])
    a2 = np.subtract(court.dehomogenise(vp_centre), lm["NKC"])
    vp_angle = np.degrees(np.arctan2(a1[1], a1[0]) - np.arctan2(a2[1], a2[0]))
    return {
        "worked_join_left_raw": [_r(v, 2) for v in join_left],
        "worked_join_right_raw": [_r(v, 2) for v in join_right],
        "worked_vp_raw": [_r(v, 1) for v in vp_raw],
        "vp_direction_disagreement_deg": _r(abs(vp_angle), 3),
        "lines_img_normalised": {n: [_r(v, 5) for v in vec] for n, vec in L.items()},
        "NBL_img": [_r(v, 2) for v in nbl],
        "NBL_off_frame_px": _r(-nbl[0], 2),
        "sidelines_meet_on_ground": [_r(v, 4) for v in ground / np.abs(ground).max()],
        "vp_along_court_img": [_r(v, 1) for v in court.dehomogenise(vp_sides)],
        "vp_along_court_via_centre_line_img": [_r(v, 1) for v in court.dehomogenise(vp_centre)],
        "vp_agreement_px": _r(
            np.linalg.norm(court.dehomogenise(vp_sides) - court.dehomogenise(vp_centre)), 1
        ),
        "vp_across_court_img": [_r(v, 1) for v in court.dehomogenise(vp_across)],
        "horizon_line": [_r(v, 5) for v in horizon],
        "horizon_row_at_centre_x": _r(-(horizon[0] * 960 + horizon[2]) / horizon[1], 1),
        # Check: the far baseline and far kitchen line, never used above, must meet on
        # that horizon too. They are fitted here only through the net-free paint.
    }


# --- 3.5 L1: the 2x2 ---------------------------------------------------------------


def linear(s: Scene) -> dict:
    L = {n: f.line for n, f in s.fit.lines.items()}

    def direction_deg(line):
        return float(np.degrees(np.arctan2(-line[0], line[1])))

    angle = abs(direction_deg(L["side_left"]) - direction_deg(L["side_right"]))
    angle = min(angle, 180 - angle)
    # Best 2x2 once both point sets are centred: the most any linear map can do.
    c_court, c_img = s.fit_court.mean(0), s.fit_img.mean(0)
    M = transforms.fit_linear(s.fit_court - c_court, s.fit_img - c_img)
    pred = transforms.apply(M, s.fit_court - c_court) + c_img
    resid = np.linalg.norm(pred - s.fit_img, axis=1)
    # What a perfect map would need: the local area scale at the near and far ends.
    J = {}
    for name in ("NBR", "NKR"):
        p = court.court_point(name)
        a = transforms.apply(s.H_court2img, np.array([p, p + [0.01, 0], p + [0, 0.01]]))
        J[name] = abs(np.linalg.det(np.stack([a[1] - a[0], a[2] - a[0]]) / 0.01))
    return {
        "sideline_angle_img_deg": _r(angle, 2),
        "sideline_angle_ground_deg": 0.0,
        "best_2x2_court2img": [[_r(v, 2) for v in row] for row in M],
        "best_2x2_decomposition": {
            k: (_r(v, 3) if isinstance(v, float) else v)
            for k, v in transforms.decompose_2x2(M).items()
        },
        "best_2x2_residual_px": dict(
            zip(court.FIT_LANDMARKS, [_r(v, 1) for v in resid], strict=True)
        ),
        "px2_per_m2_at_NBR": _r(J["NBR"], 0),
        "px2_per_m2_at_NKR": _r(J["NKR"], 0),
    }


# --- 3.5 L2: affine vs projective -----------------------------------------------


def affine_vs_projective(s: Scene) -> dict:
    three = [court.FIT_LANDMARKS.index(n) for n in ("NBL", "NBR", "NKL")]
    A3 = transforms.fit_affine(s.fit_img[three], s.fit_court[three])
    A4 = transforms.fit_affine(s.fit_img, s.fit_court)
    H = s.H_img2court
    miss_nkr = np.linalg.norm(transforms.apply(A3, s.fit_img[3:4])[0] - s.fit_court[3]) * 100
    return {
        "affine_3pt_img2court": [[_r(v, 6) for v in r] for r in A3],
        "affine_3pt_misses_NKR_cm": _r(miss_nkr, 1),
        "affine_3pt": s.held_out(A3),
        "affine_lsq_4pt": s.held_out(A4),
        "affine_lsq_4pt_fit_residual_cm": [
            _r(v, 1)
            for v in np.linalg.norm(transforms.apply(A4, s.fit_img) - s.fit_court, axis=1) * 100
        ],
        "homography_4pt": s.held_out(H),
        "homography_img2court": [[_r(v, 8) for v in r] for r in H / H[2, 2]],
    }


# --- 3.5 L3: the DLT ----------------------------------------------------------------


def dlt(s: Scene) -> dict:
    src, dst = s.fit_img, s.fit_court
    H_ours = transforms.dlt(src, dst)
    H_cv = cv2.getPerspectiveTransform(src.astype(np.float32), dst.astype(np.float32))
    H_cv = H_cv / H_cv[2, 2]
    probe = np.array([s.fit.landmarks_img["NBC"], s.fit.landmarks_img["NKC"]])
    parity = np.abs(transforms.apply(H_ours, probe) - transforms.apply(H_cv, probe)).max() * 1000
    A = transforms.dlt_matrix(src, dst)
    sv = np.linalg.svd(A, compute_uv=False)

    # More points: add the two near centre-line corners to the fit.
    six = list(court.FIT_LANDMARKS) + ["NBC", "NKC"]
    src6 = np.array([s.fit.landmarks_img[n] for n in six])
    H6 = transforms.dlt(src6, court.court_points(six))
    held6 = s.held_out(H6)

    # Noise: sigma = 1 px on the image corners, with and without normalising. With
    # exactly four points the system has an exact solution and normalising cannot
    # change it (both columns agree to float precision); it matters once there are
    # more equations than unknowns and h is a least-squares compromise, so the
    # six-point fit is the one that shows it.
    rng = np.random.default_rng(ALIGN_SEED)
    spread = {}
    for label, names in (("four_points", list(court.FIT_LANDMARKS)), ("six_points", six)):
        s_img = np.array([s.fit.landmarks_img[n] for n in names])
        s_court = court.court_points(names)
        far = {True: [], False: []}
        for _ in range(DLT_NOISE_TRIALS):
            noisy = s_img + rng.normal(0, DLT_NOISE_PX, s_img.shape)
            for norm in (True, False):
                H = transforms.dlt(noisy, s_court, normalise=norm)
                far[norm].append(
                    evaluate.held_out(H, s.fit.landmarks_img, s.evidence)["far_baseline"]
                )
        spread[label] = {
            ("normalised" if k else "raw"): {
                "median": _r(np.median(v), 2),
                "p95": _r(np.percentile(v, 95), 2),
            }
            for k, v in far.items()
        }
        spread[label]["max_abs_difference_cm"] = _r(
            np.max(np.abs(np.array(far[True]) - np.array(far[False]))), 4
        )
    # H&Z 4.4: the unnormalised DLT depends on where the pixel origin is; the
    # normalised one does not. Same noisy six points, origin moved, pixel-to-pixel
    # (image -> top view), mapped back so every H takes the original pixels.
    H_c2t, _ = warp.court2top()
    H_t2c = np.linalg.inv(H_c2t)
    noisy6 = src6 + np.random.default_rng(ALIGN_SEED).normal(0, DLT_NOISE_PX, src6.shape)
    dst6 = transforms.apply(H_c2t, court.court_points(six))
    invariance = {}
    for off in (0, 1_000, 10_000, 100_000):
        shift = np.array([[1, 0, off], [0, 1, off], [0, 0, 1.0]])
        row = {}
        for norm in (True, False):
            H = (
                H_t2c
                @ transforms.dlt(transforms.apply(shift, noisy6), dst6, normalise=norm)
                @ shift
            )
            row["normalised" if norm else "raw"] = _r(
                evaluate.held_out(H, s.fit.landmarks_img, s.evidence)["far_baseline"], 3
            )
        invariance[str(off)] = row
    pix_cond = {
        "raw": _r(transforms.condition_number(src6, dst6, False), 0),
        "normalised": _r(transforms.condition_number(src6, dst6, True), 2),
    }

    scale = {
        n: {
            k: _r(v, 2)
            for k, v in evaluate.scale_cm_per_px(s.H_img2court, s.fit.landmarks_img[n]).items()
        }
        for n in ("NBR", "NBC", "NKC")
    }
    far_px = transforms.apply(s.H_court2img, court.court_points(["FBR"]))[0]
    scale["FBR"] = {k: _r(v, 2) for k, v in evaluate.scale_cm_per_px(s.H_img2court, far_px).items()}
    # One wrong pair among six: NKC's pixel replaced by a point 25 px away along the
    # kitchen line, the size of error a detector makes on a shoe beside the paint.
    wrong = src6.copy()
    kl = s.fit.lines["near_kitchen"].line
    direction = np.array([-kl[1], kl[0]])
    wrong[5] = wrong[5] + 25 * direction
    H_bad = transforms.dlt(wrong, court.court_points(six))
    outlier = {"slip_25px": s.held_out(H_bad)}
    # And a mislabel: the pixel of NKR given NKC's court position, which is what a
    # matcher does when two corners look alike.
    swapped = src6.copy()
    swapped[5] = s.fit.landmarks_img["NKR"]
    outlier["mislabel_NKR_as_NKC"] = s.held_out(transforms.dlt(swapped, court.court_points(six)))
    rows_nbl = A[:2]

    return {
        "A_shape": list(A.shape),
        "A_rows_NBL": [[_r(v, 3) for v in r] for r in rows_nbl],
        "one_wrong_pair_of_six": outlier,
        "A_singular_values": [_r(v, 6) for v in sv],
        "condition_raw": _r(transforms.condition_number(src, dst, False), 1),
        "condition_normalised": _r(transforms.condition_number(src, dst, True), 2),
        "parity_vs_getPerspectiveTransform_mm": _r(parity, 6),
        "six_point_fit": held6,
        "noise_far_baseline_cm": spread,
        "condition_raw_six": _r(
            transforms.condition_number(src6, court.court_points(six), False), 1
        ),
        "condition_normalised_six": _r(
            transforms.condition_number(src6, court.court_points(six), True), 2
        ),
        "noise_sigma_px": DLT_NOISE_PX,
        "noise_trials": DLT_NOISE_TRIALS,
        "scale_cm_per_px": scale,
        "origin_shift_far_baseline_cm": invariance,
        "condition_pixel_to_pixel_six": pix_cond,
    }


# --- The lens --------------------------------------------------------------------


def lens(s: Scene) -> dict:
    """How bent the near lines are, the k that straightens them, and what it buys."""
    near = distortion.line_bands(s.pixels, s.H_court2img, court.NEAR_LINES)
    k = distortion.plumb_line_k(near)
    before, after = distortion.total_sagitta(near, 0.0), distortion.total_sagitta(near, k)
    # Held-out line: the far part of the right sideline, never used to choose k.
    far = {"far_right_sideline": s.evidence["side_right"]}
    fb, fa = distortion.total_sagitta(far, 0.0), distortion.total_sagitta(far, k)
    # Redo the whole fit in undistorted pixels and score it on the same evidence.
    upix = distortion.undistort(s.pixels, k)
    useeds = distortion.undistort(np.array(ALIGN_SEED_CORNERS), k)
    ufit = lines.fit_court(upix, seed_corners=useeds)
    uev = {n: distortion.undistort(v, k) for n, v in s.evidence.items()}
    uheld = evaluate.held_out(np.linalg.inv(ufit.H_court2img), ufit.landmarks_img, uev)
    return {
        "k_division_model": _r(k, 4),
        "distortion_centre": "frame centre (assumed)",
        "sagitta_px_before": {n: _r(v, 2) for n, v in before.items()},
        "sagitta_px_after": {n: _r(v, 2) for n, v in after.items()},
        "held_out_line_sagitta_px": {
            "before": _r(fb["far_right_sideline"], 2),
            "after": _r(fa["far_right_sideline"], 2),
        },
        "held_out_raw": s.held_out(s.H_img2court),
        "held_out_undistorted": {
            k2: (_r(v, 2) if not isinstance(v, dict) else v) for k2, v in uheld.items()
        },
    }


# --- 3.5 L5: warping and blending ---------------------------------------------------


def warping(s: Scene) -> dict:
    H_i2t, size = warp.img2top(s.H_img2court)
    ours, ok = warp.backward(s.frame, H_i2t, size)
    ref = cv2.warpPerspective(s.frame, H_i2t, size, flags=cv2.INTER_LINEAR)
    inside = ok.copy()
    inside[:2, :], inside[-2:, :], inside[:, :2], inside[:, -2:] = False, False, False, False
    diff = np.abs(ours.astype(int) - ref.astype(int))[inside]
    court_top = warp.court_mask_top(size)
    visible = court_top & ok
    # Forward mapping of the court's image pixels: count holes and double writes.
    region = np.zeros(s.frame.shape[:2], np.uint8)
    court_img = transforms.apply(
        s.H_court2img, np.array([[0, 0], [13.41, 0], [13.41, 6.10], [0, 6.10]])
    )
    cv2.fillPoly(region, [np.rint(court_img).astype(np.int32)], 1)
    _, hit = warp.forward(s.frame, H_i2t, size, region.astype(bool))
    holes = (hit == 0) & visible
    doubles = (hit > 1) & visible
    # Near and far halves separately: the far half is stretched, the near squeezed.
    top_h = size[1]
    near_rows = np.zeros_like(visible)
    near_rows[top_h // 2 :, :] = True
    # Blending: median of the 31 warped frames vs the warp of the median plate.
    from fundamentals_lab.config import ALIGN_PLATE_FRAMES

    warped = np.stack(
        [
            cv2.warpPerspective(plate.load_frame(i), H_i2t, size, flags=cv2.INTER_LINEAR)
            for i in ALIGN_PLATE_FRAMES
        ]
    )
    med_of_warps = np.median(warped, axis=0).astype(np.uint8)
    warp_of_med = cv2.warpPerspective(s.plate, H_i2t, size, flags=cv2.INTER_LINEAR)
    d = np.abs(med_of_warps.astype(int) - warp_of_med.astype(int)).max(axis=2)[visible]
    # Players in the frame: pixels of frame 45000's top view that differ from the plate.
    single = cv2.warpPerspective(s.frame, H_i2t, size, flags=cv2.INTER_LINEAR)
    moved = np.abs(single.astype(int) - warp_of_med.astype(int)).max(axis=2) > 40
    # The worked example: one top-view pixel on the edge of the near centre line,
    # pulled back to the frame, and the four pixels bilinear interpolation mixes.
    H_t2c = np.linalg.inv(warp.court2top()[0])
    tp = np.array([230.0, 711.0])
    c_pt = transforms.apply(H_t2c, tp[None])[0]
    hom = s.H_court2img @ np.array([c_pt[0], c_pt[1], 1.0])
    u, v = hom[:2] / hom[2]
    x0, y0 = int(np.floor(u)), int(np.floor(v))
    fx, fy = u - x0, v - y0
    grey = s.frame.astype(np.float64).mean(axis=2)
    quad = [grey[y0, x0], grey[y0, x0 + 1], grey[y0 + 1, x0], grey[y0 + 1, x0 + 1]]
    wts = [(1 - fx) * (1 - fy), fx * (1 - fy), (1 - fx) * fy, fx * fy]
    worked = {
        "top_px": tp.tolist(),
        "court_m": [_r(v_, 4) for v_ in c_pt],
        "homogeneous_img": [_r(v_, 3) for v_ in hom],
        "img_px": [_r(u, 3), _r(v, 3)],
        "fx_fy": [_r(fx, 3), _r(fy, 3)],
        "grey_quad": [_r(q, 1) for q in quad],
        "weights": [_r(w_, 3) for w_ in wts],
        "bilinear": _r(np.dot(wts, quad), 1),
        "nearest": _r(quad[int(round(fy)) * 2 + int(round(fx))], 1),
    }
    return {
        "worked_backward": worked,
        "topview_size_px": list(size),
        "topview_cm_per_px": 2.0,
        "backward_vs_cv2_max_abs_diff": int(diff.max()),
        "backward_vs_cv2_mean_abs_diff": _r(diff.mean(), 4),
        "court_top_px_visible": int(visible.sum()),
        "forward_holes_frac": _r(holes.sum() / visible.sum(), 4),
        "forward_holes_frac_far_half": _r(
            (holes & ~near_rows).sum() / (visible & ~near_rows).sum(), 4
        ),
        "forward_holes_frac_near_half": _r(
            (holes & near_rows).sum() / (visible & near_rows).sum(), 4
        ),
        "forward_double_writes_frac_near_half": _r(
            (doubles & near_rows).sum() / (visible & near_rows).sum(), 4
        ),
        "median_order_max_diff_p99": _r(np.percentile(d, 99), 1),
        "median_order_mean_diff": _r(d.mean(), 3),
        "frame_pixels_differing_from_plate_frac": _r((moved & visible).sum() / visible.sum(), 4),
    }


# --- Is the measurement a measurement? ---------------------------------------------


def stability(s: Scene) -> dict:
    rng = np.random.default_rng(ALIGN_SEED)
    base = s.fit.landmarks_img
    worst = dict.fromkeys(base, 0.0)
    for _ in range(SEED_PERTURB_TRIALS):
        seeds = np.array(ALIGN_SEED_CORNERS) + rng.uniform(
            -SEED_PERTURB_PX, SEED_PERTURB_PX, (4, 2)
        )
        m = lines.fit_court(s.pixels, seed_corners=seeds).landmarks_img
        for n in base:
            worst[n] = max(worst[n], float(np.linalg.norm(m[n] - base[n])))
    frame_fit = lines.fit_court(lines.paint_pixels(s.frame)).landmarks_img
    colour_only = lines.fit_court(_colour_mask_pixels(s.plate))
    return {
        "seed_perturbation_px": SEED_PERTURB_PX,
        "seed_trials": SEED_PERTURB_TRIALS,
        "max_landmark_shift_px": {n: _r(v, 3) for n, v in worst.items()},
        "frame45000_vs_plate_px": {n: _r(np.linalg.norm(frame_fit[n] - base[n]), 2) for n in base},
        "line_fit": {
            n: {"n_pixels": f.n_pixels, "rms_px": _r(f.rms_px, 2)} for n, f in s.fit.lines.items()
        },
        "crossing_angle_deg": {
            n: _r(
                np.degrees(
                    np.arcsin(
                        np.sqrt(
                            max(0.0, 1 - (s.fit.lines[a].line[:2] @ s.fit.lines[b].line[:2]) ** 2)
                        )
                    )
                ),
                1,
            )
            for n, (a, b) in court.LANDMARKS.items()
            if a in s.fit.lines and b in s.fit.lines
        },
        "colour_threshold_mask_held_out": s.held_out(np.linalg.inv(colour_only.H_court2img)),
    }


def _colour_mask_pixels(image):
    from fundamentals_lab.config import NET_POLYGON, WHITE_MAX_SAT, WHITE_MIN_VALUE

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (0, 0, WHITE_MIN_VALUE), (179, WHITE_MAX_SAT, 255))
    cv2.fillPoly(mask, [np.array(NET_POLYGON, np.int32)], 0)
    ys, xs = np.nonzero(mask)
    return np.stack([xs, ys], axis=1).astype(np.float64)
