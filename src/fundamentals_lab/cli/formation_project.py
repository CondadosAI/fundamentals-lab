"""Lesson 1's numbers: the pinhole prediction against the measured corner."""

import click
from loguru import logger

from fundamentals_lab.config import FORMATION_BOARD
from fundamentals_lab.formation import numbers
from fundamentals_lab.formation.calibration import find_corners
from fundamentals_lab.formation.dataset import fetch_opencv_calibration
from fundamentals_lab.formation.projection import project_view


@click.command()
@click.option("--view", default=0, help="Which calibration photograph to work through.")
def project(view: int) -> None:
    """Project one view's board corners with the two scalar equations and measure the gap."""
    paths = fetch_opencv_calibration()
    obj_points, img_points, size, used, _ = find_corners(paths)
    if not 0 <= view < len(used):
        raise click.ClickException(f"view {view} is outside the {len(used)} usable views")

    result = project_view(obj_points, img_points, size, view)
    result["view_file"] = used[view]
    result["board_inner_corners"] = list(FORMATION_BOARD)
    numbers.save("projection", result)

    c = result["worked_corner"]
    logger.info(f"view {used[view]}, board corner {c['board_point']}")
    logger.info(f"  camera frame {[round(v, 4) for v in c['camera_frame_point']]}")
    logger.info(
        f"pinhole predicts ({c['pinhole_prediction_px'][0]:.2f}, "
        f"{c['pinhole_prediction_px'][1]:.2f}), "
        f"measured ({c['measured_px'][0]:.2f}, {c['measured_px'][1]:.2f}), "
        f"off by {c['error_px']:.2f} px at r={c['radius_px']:.0f} px"
    )
    e = result["pinhole_only_error_px"]
    inner, outer = e["innermost_third"], e["outermost_third"]
    logger.info(f"pinhole-only residual: mean {e['mean']:.2f} px, max {e['max']:.2f} px")
    logger.info(
        f"  innermost third (mean r={inner['mean_radius_px']:.0f} px): "
        f"{inner['mean_error_px']:.2f} px"
    )
    logger.info(
        f"  outermost third (mean r={outer['mean_radius_px']:.0f} px): "
        f"{outer['mean_error_px']:.2f} px"
    )
    f = result["full_model_error_px"]
    logger.info(f"with the distortion terms: mean {f['mean']:.3f} px, max {f['max']:.3f} px")


if __name__ == "__main__":
    project()
