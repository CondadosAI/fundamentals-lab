"""Depth of field, worked on a scene whose capture settings were published.

The DPDD paper states the apertures but not the focal length or the subject
distance, so the depth of field cannot be computed from the text. The repository's
own figures/data_example.png carries them: 93 mm, focus distance 1.46-1.59 m,
ISO 100, f/4 at 1/25 s and f/22 at 1.3 s, with a depth of field quoted for each.

That makes the figure checkable rather than quotable, which is the point: the
formula either reproduces the published interval or it does not.
"""

import math

# Read off figures/data_example.png in the DPDD repository (MIT). Metres.
FOCAL_LENGTH_M = 0.093
FOCUS_DISTANCE_RANGE_M = (1.46, 1.59)
PUBLISHED_DOF = {4.0: (1.50, 1.55), 22.0: (0.70, 5.80)}

# Canon EOS 5D Mark IV: full frame, 36 x 24 mm. The usual full-frame circle of
# confusion quoted by depth-of-field tables is around 0.029-0.030 mm, which is
# what the value recovered below has to be compared against.
SENSOR_MM = (36.0, 24.0)
FULL_FRAME_COC_MM = 0.030


def hyperfocal(f: float, n: float, coc: float) -> float:
    """Distance beyond which everything is acceptably sharp."""
    return f * f / (n * coc) + f


def dof_limits(f: float, n: float, coc: float, s: float) -> tuple:
    """Near and far limits of acceptable focus for subject distance s."""
    a = hyperfocal(f, n, coc) - f
    u = s - f
    near = s * a / (a + u)
    far = math.inf if u >= a else s * a / (a - u)
    return near, far


def recover_subject_and_coc(f: float, n: float, near: float, far: float) -> tuple:
    """Invert a published depth-of-field interval into the two numbers behind it.

    The limits give a closed form rather than needing a search:
        1/near + 1/far = 2/s          ->  s = 2 near far / (near + far)
        1/near - 1/far = 2(s-f)/(s A) ->  A = 2 (s-f) near far / (s (far - near))
    with A = H - f, so the circle of confusion is c = f^2 / (n A).
    """
    s = 2.0 * near * far / (near + far)
    a = 2.0 * (s - f) * near * far / (s * (far - near))
    return s, f * f / (n * a)


def focal_length_implied(n: float, coc: float, near: float, far: float) -> tuple:
    """Which lens and subject distance would produce this interval at this f-stop.

    Used to test a published row against the rest of its own caption: solving
    f^2 = n c A with A expressed through the limits gives a quadratic in f.
    """
    s = 2.0 * near * far / (near + far)
    k = 2.0 * n * coc * near * far / (s * (far - near))
    # f^2 + k f - k s = 0
    f = (-k + math.sqrt(k * k + 4.0 * k * s)) / 2.0
    return f, s
