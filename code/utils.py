"""
This module contains some common routines and types used by other modules.
"""
import collections
from enum import Enum
from typing import List, NamedTuple, Tuple

import numpy as np
import shapely.geometry

# We use OpenCV's type as the underlying 2D image type.
Image = np.ndarray

CHECK_RIGHT_ANGLE_THRESH = 0.1


class Point2D(collections.namedtuple("Point2D", "x y")):
    @classmethod
    def from_tuple(cls, t: Tuple[np.float, np.float]):
        return cls._make(t)


# The type of supported polygon/wall/point objects.
class PolygonType(Enum):
    ROOM = "room"
    WINDOW = "window"
    DOOR = "door"
    OPENING = "opening"
    PRIMARY_CAMERA = "primary_camera"
    SECONDARY_CAMERA = "secondary_camera"
    PIN_LABEL = "pin_label"


PolygonTypeMapping = {
    "windows": PolygonType.WINDOW,
    "doors": PolygonType.DOOR,
    "openings": PolygonType.OPENING,
}


class Polygon(
    NamedTuple(
        "Polygon", [("type", PolygonType), ("points", List[Point2D]), ("name", str)]
    )
):
    """
    Polygon class that can be used to represent polygons/lines as a list of points, the type and (optional) name
    """

    __slots__ = ()

    def __new__(cls, type, points, name=""):
        return super(Polygon, cls).__new__(cls, type, points, name)

    @staticmethod
    def list_to_points(points: List[Tuple[np.float, np.float]]):
        return [Point2D._make(p) for p in points]

    @property
    def to_list(self):
        return [(p.x, p.y) for p in self.points]

    @property
    def num_points(self):
        return len(self.points)

    @property
    def to_shapely_poly(self):
        # Use this function when converting a closed room shape polygon
        return shapely.geometry.polygon.Polygon(self.to_list)

    @property
    def to_shapely_line(self):
        # Use this function when converting W/D/O elements since those are represented as lines.
        return shapely.geometry.LineString(self.to_list)


def compute_dot_product(x_prev, y_prev, x_curr, y_curr, x_next, y_next):
    """Compute the oriented angle (in radians) given the camera position and
    two vertices.
    """
    vec_prev = np.array([x_prev - x_curr, y_prev - y_curr])
    vec_prev_norm = np.linalg.norm(vec_prev)
    vec_next = np.array([x_next - x_curr, y_next - y_curr])
    vec_next_norm = np.linalg.norm(vec_next)
    # The function expects non-degenerate case, e.g if one of the line is a point, then this will fail
    return np.dot(vec_prev, vec_next) / (vec_prev_norm * vec_next_norm)


def remove_collinear(room_vertices):
    room_vertices_updated = []
    for idx_curr, vert_curr in enumerate(room_vertices):
        idx_prev = idx_curr - 1
        if idx_prev < 0:
            idx_prev = len(room_vertices) - 1
        idx_next = idx_curr + 1
        if idx_next >= len(room_vertices):
            idx_next = 0
        vert_prev = room_vertices[idx_prev]
        vert_next = room_vertices[idx_next]
        angle = compute_dot_product(
            vert_prev[0],
            vert_prev[1],
            vert_curr[0],
            vert_curr[1],
            vert_next[0],
            vert_next[1],
        )
        if abs(abs(angle) - 1.0) < 1e-3:
            continue
        room_vertices_updated.append([vert_curr[0], vert_curr[1]])
    return np.asarray(room_vertices_updated)
