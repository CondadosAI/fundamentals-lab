"""Lesson 2's numbers: depth of field at f/4 and f/22 on one published scene."""

import click
from loguru import logger

from fundamentals_lab.formation import numbers
from fundamentals_lab.formation.dof import (
    FOCAL_LENGTH_M,
    FOCUS_DISTANCE_RANGE_M,
    FULL_FRAME_COC_MM,
    PUBLISHED_DOF,
    dof_limits,
    focal_length_implied,
    hyperfocal,
    recover_subject_and_coc,
)


@click.command()
def dof() -> None:
    """Reproduce the published depth of field, then test the other aperture against it."""
    f = FOCAL_LENGTH_M
    near4, far4 = PUBLISHED_DOF[4.0]
    subject, coc = recover_subject_and_coc(f, 4.0, near4, far4)

    result = {
        "source": (
            "DPDD figures/data_example.png (MIT), capture settings read off the figure"
        ),
        "focal_length_mm": f * 1000.0,
        "focus_distance_range_m": list(FOCUS_DISTANCE_RANGE_M),
        "published_dof_m": {str(k): list(v) for k, v in PUBLISHED_DOF.items()},
        "recovered_subject_distance_m": subject,
        "recovered_coc_mm": coc * 1000.0,
        "full_frame_reference_coc_mm": FULL_FRAME_COC_MM,
        "apertures": {},
    }
    logger.info(
        f"f/4 interval {near4}-{far4} m implies subject {subject:.4f} m "
        f"and circle of confusion {coc * 1000:.4f} mm "
        f"(full-frame tables use about {FULL_FRAME_COC_MM} mm)"
    )
    if not FOCUS_DISTANCE_RANGE_M[0] <= subject <= FOCUS_DISTANCE_RANGE_M[1]:
        logger.warning("recovered subject distance falls outside the figure's own range")

    for n in (4.0, 22.0):
        near, far = dof_limits(f, n, coc, subject)
        row = {
            "hyperfocal_m": hyperfocal(f, n, coc),
            "near_m": near,
            "far_m": far,
            "depth_m": far - near,
            "published_m": list(PUBLISHED_DOF[n]),
        }
        result["apertures"][str(n)] = row
        logger.info(
            f"f/{n:.0f}: predicted {near:.3f}-{far:.3f} m (depth {far - near:.3f} m), "
            f"published {PUBLISHED_DOF[n][0]}-{PUBLISHED_DOF[n][1]} m"
        )

    d4 = result["apertures"]["4.0"]["depth_m"]
    d22 = result["apertures"]["22.0"]["depth_m"]
    result["depth_ratio_f22_over_f4"] = d22 / d4
    logger.info(f"depth grows {d22 / d4:.2f}x from f/4 to f/22, against an aperture ratio of 5.50")

    # The f/22 row of the figure does not follow from the figure's own focal
    # length. Rather than call it an error, record what it would imply.
    f_imp, s_imp = focal_length_implied(22.0, coc, *PUBLISHED_DOF[22.0])
    result["published_f22_implies"] = {
        "focal_length_mm": f_imp * 1000.0,
        "subject_distance_m": s_imp,
    }
    logger.info(
        f"the published f/22 interval would need a {f_imp * 1000:.1f} mm lens "
        f"at {s_imp:.3f} m, not {f * 1000:.0f} mm at {subject:.3f} m"
    )

    numbers.save("depth_of_field", result)
    logger.info(f"wrote {numbers.FORMATION_NUMBERS_JSON}")


if __name__ == "__main__":
    dof()
