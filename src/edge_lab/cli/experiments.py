"""Every measured claim the Edge Detection unit publishes, in one run.

Each experiment sweeps a parameter rather than reporting a single point: a
single run of a noisy pipeline is variance, not an effect. The output is one
JSON file that the articles cite by key.
"""

import json
import platform
import sys

import click
import cv2
import numpy as np
from loguru import logger

from edge_lab.config import (
    CANNY_POINTS,
    DILEMMA_THRESHOLDS,
    EDGE_FRACTION,
    LOG_SIGMAS,
    NOISE_SIGMAS,
    NOISE_TRIALS,
    OUTPUT_DIR,
    PUBLISHED_EIGENVALUES,
    SEED,
)
from edge_lab.core import canny as canny_mod
from edge_lab.core.corners import preset_eigenvalues
from edge_lab.core.dataset import add_noise, canonical_gray
from edge_lab.core.gradients import (
    OPERATORS,
    gradients,
    magnitude,
    orientation_deg,
    top_fraction_mask,
)
from edge_lab.core.laplacian import (
    contour_stats,
    crossing_floor,
    log_response,
    zero_crossings,
)


def _components(mask: np.ndarray) -> int:
    n_labels, _ = cv2.connectedComponents(mask.astype(np.uint8), connectivity=8)
    return int(n_labels - 1)


def noise_robustness(gray: np.ndarray) -> dict:
    """How much of an operator's edge set survives noise.

    Both masks hold the same number of pixels by construction, so precision and
    recall are equal and a single "agreement" number says everything: the
    fraction of the clean edge set the noisy run still finds.
    """
    rng = np.random.default_rng(SEED)
    reference = {}
    for op in OPERATORS:
        gx, gy = gradients(gray, op)
        reference[op] = top_fraction_mask(magnitude(gx, gy), EDGE_FRACTION)

    results = {op: {} for op in OPERATORS}
    for sigma in NOISE_SIGMAS:
        for op in OPERATORS:
            scores = []
            for _ in range(NOISE_TRIALS):
                noisy = add_noise(gray, sigma, rng)
                gx, gy = gradients(noisy, op)
                mask = top_fraction_mask(magnitude(gx, gy), EDGE_FRACTION)
                overlap = int(np.count_nonzero(mask & reference[op]))
                scores.append(overlap / int(np.count_nonzero(reference[op])))
            results[op][f"sigma_{sigma:g}"] = {
                "agreement_mean": round(float(np.mean(scores)), 4),
                "agreement_std": round(float(np.std(scores)), 4),
                "trials": NOISE_TRIALS,
            }
    return {
        "metric": "fraction of the clean-image edge set recovered on the noisy image",
        "edge_fraction": EDGE_FRACTION,
        "noise_sigma_scale": "0..255 intensity units",
        "by_operator": results,
    }


def hysteresis_vs_single_threshold(gray: np.ndarray) -> dict:
    """Does linking weak to strong actually buy longer, fewer contours?

    The comparison is made at a matched edge-pixel count. Comparing at a matched
    threshold would be meaningless, since hysteresis keeps pixels a single
    threshold would have dropped.
    """
    stages = canny_mod.canny_from_scratch(gray, sigma=1.4, low=50, high=150)
    thin = stages["thin"]
    nonzero = thin[thin > 0]

    points = []
    for low, high in CANNY_POINTS:
        edges = canny_mod.hysteresis(thin, low / 255.0, high / 255.0)
        n_px = int(np.count_nonzero(edges))
        if n_px == 0:
            logger.warning(f"({low}, {high}) produced no edges — skipping")
            continue
        n_comp = _components(edges)
        # The single threshold that keeps the same number of pixels.
        cutoff = float(np.quantile(nonzero, 1.0 - n_px / nonzero.size))
        single = thin >= cutoff
        n_comp_single = _components(single)
        points.append(
            {
                "low": low,
                "high": high,
                "edge_pixels": n_px,
                "hysteresis": {
                    "components": n_comp,
                    "mean_component_px": round(n_px / n_comp, 2) if n_comp else None,
                },
                "single_threshold": {
                    "cutoff_0_255": round(cutoff * 255.0, 2),
                    "edge_pixels": int(np.count_nonzero(single)),
                    "components": n_comp_single,
                    "mean_component_px": round(int(np.count_nonzero(single)) / n_comp_single, 2)
                    if n_comp_single
                    else None,
                },
            }
        )
    return {
        "gaussian_sigma": 1.4,
        "note": "both columns are non-maximum-suppressed; only the keep rule differs",
        "operating_points": points,
    }


def threshold_dilemma(gray: np.ndarray) -> dict:
    """What a single threshold on gradient magnitude can and cannot give you.

    Two quantities, swept together, because the usual telling of this ("thick at a
    low threshold, broken at a high one") is only half right. Raising the threshold
    genuinely does thin the ridges. It does not fragment them — fragment *count*
    falls, because weak structure disappears entirely rather than breaking up. What
    actually collapses is coverage.

    thickness = magnitude pixels above t, divided by the non-maximum-suppressed
    pixels above the same t; that ratio is the average width of a ridge.
    coverage   = the fraction of the lowest threshold's thinned edge set retained.
    """
    blurred = cv2.GaussianBlur(gray, (0, 0), sigmaX=1.4, sigmaY=1.4)
    gx, gy = gradients(blurred, "sobel")
    mag = magnitude(gx, gy)
    thin = canny_mod.non_max_suppression(mag, orientation_deg(gx, gy))

    rows, base = [], None
    for t in DILEMMA_THRESHOLDS:
        raw_px = int(np.count_nonzero(mag >= t / 255.0))
        thin_mask = thin >= t / 255.0
        thin_px = int(np.count_nonzero(thin_mask))
        if base is None:
            base = thin_px
        rows.append(
            {
                "threshold_0_255": t,
                "magnitude_px": raw_px,
                "thinned_px": thin_px,
                "ridge_thickness_px": round(raw_px / max(thin_px, 1), 2),
                "coverage_vs_lowest": round(thin_px / max(base, 1), 3),
                "fragments": _components(thin_mask),
            }
        )
    return {
        "gaussian_sigma": 1.4,
        "operator": "sobel",
        "note": (
            "fragment count FALLS as the threshold rises — the folk claim that a high "
            "threshold breaks edges up is not what the aggregate shows. The real cost "
            "is coverage."
        ),
        "by_threshold": rows,
    }


def zero_crossing_closure(gray: np.ndarray) -> dict:
    """Do LoG zero-crossings really close into loops, and what does noise add?"""
    rng = np.random.default_rng(SEED)
    noisy = add_noise(gray, 10.0, rng)
    rows = []
    for sigma in LOG_SIGMAS:
        clean_resp = log_response(gray, sigma)
        noisy_resp = log_response(noisy, sigma)
        # One floor, derived from the clean response, applied to both. Letting
        # each image set its own floor makes the noisy run look better the
        # noisier it gets, which is an artefact of the threshold, not a finding.
        floor = crossing_floor(clean_resp)
        clean_mask = zero_crossings(clean_resp, floor)
        noisy_mask = zero_crossings(noisy_resp, floor)
        clean_stats = contour_stats(clean_mask)
        noisy_stats = contour_stats(noisy_mask)
        px_clean = int(np.count_nonzero(clean_mask))
        px_noisy = int(np.count_nonzero(noisy_mask))
        rows.append(
            {
                "sigma": sigma,
                "clean": {**clean_stats, "crossing_px": px_clean},
                "noisy_sigma255_10": {**noisy_stats, "crossing_px": px_noisy},
                "crossing_px_ratio": round(px_noisy / px_clean, 3),
            }
        )
    # Control: the textbook property is that zero-crossings of a continuous LoG form
    # closed contours. Run it on a rendered disc to show the metric can see closure
    # at all, so a low fraction on the photograph is a fact about photographs.
    disc = np.zeros((240, 240), dtype=np.float32)
    cv2.circle(disc, (120, 120), 70, 1.0, -1)
    control = {
        f"sigma_{s:g}": contour_stats(zero_crossings(log_response(disc, s))) for s in LOG_SIGMAS
    }

    return {
        "definition_closed": "a connected component with no pixel having exactly one neighbour",
        "synthetic_control": {
            "shape": "a filled disc of radius 70 on a 240x240 field",
            "why": "validates the closure metric; a closed fraction of 1.0 here means "
            "a low fraction on the photograph is a property of the image, not the metric",
            "by_sigma": control,
        },
        "noise_metric": (
            "crossing_px_ratio — the factor by which noise multiplies the number of "
            "zero-crossing pixels. Component count is NOT used as the noise metric: at "
            "small sigma the extra crossings connect existing components instead of "
            "adding new ones, so the count falls while the structure grows, which makes "
            "it non-monotone in sigma and unusable as evidence."
        ),
        "by_sigma": rows,
    }


def opencv_reference_check(gray: np.ndarray) -> dict:
    """How far the from-scratch Canny here agrees with OpenCV's, stage by stage.

    Reported per stage rather than as one end-to-end IoU, because the stages do not
    agree equally and a single number hides that. The gradient and the non-maximum
    suppression match; what happens after the threshold does not, and that
    divergence is **not explained**. Ruled out: the gradient norm (OpenCV defaults
    to L1 and is forced to L2 here), the missing Gaussian (OpenCV applies none of
    its own), 8-bit quantization of the input, and the magnitude scale.
    """
    sigma, low, high = 1.4, 50, 150
    blurred = cv2.GaussianBlur(gray, (0, 0), sigmaX=sigma, sigmaY=sigma)
    b8 = np.clip(blurred * 255.0, 0, 255).astype(np.uint8)

    gx, gy = gradients(blurred, "sobel")
    mag = magnitude(gx, gy) * 255.0
    cgx = cv2.Sobel(b8, cv2.CV_32F, 1, 0, ksize=3)
    cgy = cv2.Sobel(b8, cv2.CV_32F, 0, 1, ksize=3)
    strong = mag > 20
    scale_ratio = float(np.median(np.hypot(cgx, cgy)[strong] / mag[strong]))

    thin = canny_mod.non_max_suppression(magnitude(gx, gy), orientation_deg(gx, gy))
    ours_nms = int(np.count_nonzero(thin > 0))
    cv_nms = int(np.count_nonzero(cv2.Canny(b8, 1, 1, L2gradient=True) > 0))

    ours = canny_mod.hysteresis(thin, low / 255.0, high / 255.0)
    ref = cv2.Canny(b8, low, high, L2gradient=True) > 0
    inter = int(np.count_nonzero(ours & ref))
    union = int(np.count_nonzero(ours | ref))

    return {
        "settings": {"sigma": sigma, "low": low, "high": high, "L2gradient": True},
        "agrees": {
            "gradient_magnitude_scale_ratio": round(scale_ratio, 4),
            "nms_survivors_ours": ours_nms,
            "nms_survivors_opencv": cv_nms,
            "nms_relative_difference": round(abs(cv_nms - ours_nms) / ours_nms, 4),
        },
        "diverges": {
            "weak_set_ours": int(np.count_nonzero(thin >= low / 255.0)),
            "weak_set_opencv": int(np.count_nonzero(cv2.Canny(b8, low, low, L2gradient=True) > 0)),
            "final_ours": int(np.count_nonzero(ours)),
            "final_opencv": int(np.count_nonzero(ref)),
            "iou": round(inter / max(union, 1), 3),
            "status": "UNEXPLAINED — see the docstring for what was ruled out",
        },
    }


def corner_parity(gray: np.ndarray) -> dict:
    """Parity check against the eigenvalues the SIFT article already published."""
    measured = preset_eigenvalues(gray)
    checks = {}
    for name, (l1_pub, l2_pub) in PUBLISHED_EIGENVALUES.items():
        got = measured[name]
        checks[name] = {
            "measured": [got["l1"], got["l2"]],
            "published": [l1_pub, l2_pub],
            "match": abs(got["l1"] - l1_pub) <= 0.0005 and abs(got["l2"] - l2_pub) <= 0.0005,
        }
    # The three responses built on the same tensor, so the reader can see where they
    # agree and where the choice of k decides the answer rather than the data.
    responses = {}
    for name, got in measured.items():
        l1, l2 = got["l1"], got["l2"]
        det, tr = l1 * l2, l1 + l2
        responses[name] = {
            "l1": l1,
            "l2": l2,
            "harris_k0.04": round(det - 0.04 * tr * tr, 6),
            "harris_k0.06": round(det - 0.06 * tr * tr, 6),
            "shi_tomasi": l2,
            "forstner_w": round(det / tr, 5) if tr else 0.0,
            "forstner_q": round(4 * det / (tr * tr), 4) if tr else 0.0,
        }

    return {
        "source_of_published": "sfm-from-scratch/output/sift_numbers.json",
        "responses": responses,
        "responses_note": (
            "on the edge patch Harris is POSITIVE at k=0.04 (+0.000413) and NEGATIVE at "
            "k=0.06 (-0.003285); k decides the sign, it is not a cosmetic constant"
        ),
        "presets": checks,
        "all_match": all(c["match"] for c in checks.values()),
    }


@click.command()
@click.option("--out", "out_name", default="edge_numbers.json", show_default=True)
def experiments(out_name: str) -> None:
    """Run every sweep and write the numbers the articles cite."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    gray = canonical_gray()
    logger.info(f"canonical frame {gray.shape[1]}x{gray.shape[0]}, intensities in 0..1")

    numbers = {
        "environment": {
            "python": sys.version.split()[0],
            "opencv": cv2.__version__,
            "numpy": np.__version__,
            "platform": platform.platform(),
            "seed": SEED,
        },
        "image": {"width": int(gray.shape[1]), "height": int(gray.shape[0])},
        "corner_parity": corner_parity(gray),
        "opencv_reference_check": opencv_reference_check(gray),
        "noise_robustness": noise_robustness(gray),
        "threshold_dilemma": threshold_dilemma(gray),
        "hysteresis_vs_single_threshold": hysteresis_vs_single_threshold(gray),
        "zero_crossing_closure": zero_crossing_closure(gray),
    }

    out = OUTPUT_DIR / out_name
    out.write_text(json.dumps(numbers, indent=2))
    parity = numbers["corner_parity"]["all_match"]
    logger.info(f"corner parity vs the published SIFT numbers: {'PASS' if parity else 'FAIL'}")
    logger.info(f"wrote {out}")
    if not parity:
        raise click.ClickException(
            "corner eigenvalues do not match the published values — the canonical "
            "frame pipeline has drifted from the one the SIFT article used"
        )


if __name__ == "__main__":
    experiments()
