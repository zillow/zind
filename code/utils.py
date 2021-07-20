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
