import math
from typing import Sequence


def angle_3d(p1 : Sequence[float], p2: Sequence[float]) -> tuple[float, float, float]:
    """
    Returns the normalized 3D vector of 2 points.

    :param p1: The 3D coordinates of the origin point.
    :param p2: The 3D coordinates of the offset point
    :return: Normalized 3D vector representing the angle between p1 and p2.
    """
    # vector 1 → 2
    x = p2[0] - p1[0]
    y = p2[1] - p1[1]
    z = p2[2] - p1[2]
    # vector magnitude (length)
    magnitude = math.sqrt(x * x + y * y + z * z)
    # avoid division by zero
    if magnitude == 0:
        return 0, 0, 0
    # normalized vector (unit length)
    return x / magnitude, y / magnitude, z / magnitude
