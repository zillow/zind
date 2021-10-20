"""
This module provides utilities to handle the various coordinate system transformations:
1. Spherical to/from cartesian
2. 3D room layout to/from pano pixels
3. 3D room floor_plan_layouts to/from 2D top-down merged floor_plan_layouts
"""
import collections
import logging
import math
import sys
from typing import List, Dict, Any

import numpy as np

from utils import Point2D

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
LOG = logging.getLogger(__name__)


class Transformation2D(
    collections.namedtuple("Transformation", "rotation_matrix scale translation")
):
    """
    Class to handle relative translation/rotation/scale of room shape coordinates
    to transform them from local to the global frame of reference.
    """

    @classmethod
    def from_translation_rotation_scale(
        cls, *, position: Point2D, rotation: float, scale: float
    ):
        """
        Create a transformation object from the ZInD merged top-down geometry data
        based on the given 2D translation (position), rotation angle and scale.

        :param position: 2D translation (in the x-y plane)
        :param rotation: Rotation angle in degrees (in the x-y plane)
        :param scale: Scale factor for all the coordinates

        :return: A transformation object that can later be applied on a list of
        coordinates in local frame of reference to move them into the global
        (merged floor map) frame of reference.
        """
        translation = np.array([position.x, position.y]).reshape(1, 2)
        rotation_angle = np.radians(rotation)

        rotation_matrix = np.array(
            [
                [np.cos(rotation_angle), np.sin(rotation_angle)],
                [-np.sin(rotation_angle), np.cos(rotation_angle)],
            ]
        )

        return cls(
            rotation_matrix=rotation_matrix, scale=scale, translation=translation
        )

    @classmethod
    def from_zind_data(cls, zind_transformation: Dict[str, Any]):
        """
        Create a transformation object from the ZInD JSON blob.

        :param zind_transformation: Dict with "translation", "rotation" and "scale" fields.

        :return: A transformation object that can later be applied on a list of
        coordinates in local frame of reference to move them into the global
        (merged floor map) frame of reference.
        """
        return Transformation2D.from_translation_rotation_scale(
            position=Point2D.from_tuple(zind_transformation["translation"]),
            rotation=zind_transformation["rotation"],
            scale=zind_transformation["scale"],
        )

    def to_global(self, coordinates):
        """
        Apply transformation on a list of 2D points to transform them from local to global frame of reference.

        :param coordinates: List of 2D coordinates in local frame of reference.

        :return: The transformed list of 2D coordinates.
        """
        coordinates = coordinates.dot(self.rotation_matrix) * self.scale
        coordinates += self.translation
        return coordinates

    def apply_inverse(self, coordinates):
        coordinates -= self.translation
        coordinates = coordinates.dot(self.rotation_matrix.T) / self.scale
        return coordinates


class TransformationSpherical:
    """
    Class to handle various spherical transformations.
    """

    def __init__(self):
        pass

    @classmethod
    def rotate(cls, input_array: np.ndarray):
        return input_array.dot(cls.ROTATION_MATRIX)

    @staticmethod
    def normalize(points_cart: np.ndarray) -> np.ndarray:
        """
        Normalize a set of 3D vectors.
        """
        num_points = points_cart.shape[0]
        assert num_points > 0

        num_coords = points_cart.shape[1]
        assert num_coords == 3

        rho = np.sqrt(np.sum(np.square(points_cart), axis=1))
        return points_cart / rho.reshape(num_points, 1)

    @staticmethod
    def cartesian_to_sphere(points_cart: np.ndarray) -> np.ndarray:
        """
        Convert cartesian to spherical coordinates.
        """
        output_shape = (points_cart.shape[0], 3)  # type: ignore

        num_points = points_cart.shape[0]
        assert num_points > 0

        num_coords = points_cart.shape[1]
        assert num_coords == 3

        x_arr = points_cart[:, 0]
        y_arr = points_cart[:, 1]
        z_arr = points_cart[:, 2]

        # Azimuth angle is in [-pi, pi].
        # Note the x-axis flip to align the handedness of the pano and room shape coordinate systems.
        theta = np.arctan2(-x_arr, y_arr)
        # Radius can be anything between (0, inf)
        rho = np.sqrt(np.sum(np.square(points_cart), axis=1))
        phi = np.arcsin(z_arr / rho)  # Map elevation to [-pi/2, pi/2]
        return np.column_stack((theta, phi, rho)).reshape(output_shape)

    @staticmethod
    def sphere_to_cartesian(points_sph: np.ndarray) -> np.ndarray:
        """
        Convert spherical coordinates to cartesian.
        """
        EPS = np.deg2rad(1)
        if not isinstance(points_sph, np.ndarray) or points_sph.ndim == 1:
            points_sph = np.reshape(points_sph, (1, -1))
            output_shape = (3,)
        else:
            output_shape = (points_sph.shape[0], 3)  # type: ignore
        num_points = points_sph.shape[0]
        assert num_points > 0
        num_coords = points_sph.shape[1]
        assert num_coords == 2 or num_coords == 3
        theta = points_sph[:, 0]
        # Validate the azimuthal angles.
        assert np.all(np.greater_equal(theta, -math.pi - EPS))
        assert np.all(np.less_equal(theta, math.pi + EPS))
        phi = points_sph[:, 1]
        # Validate the elevation angles.
        assert np.all(np.greater_equal(phi, -math.pi / 2.0 - EPS))
        assert np.all(np.less_equal(phi, math.pi / 2.0 + EPS))
        if num_coords == 2:
            rho = np.ones_like(theta)
        else:
            rho = points_sph[:, 2]
        # Validate the radial distances.
        assert np.all(np.greater(rho, 0.0))
        rho_cos_phi = rho * np.cos(phi)
        x_arr = rho_cos_phi * np.sin(theta)
        y_arr = rho * np.sin(phi)
        z_arr = -rho_cos_phi * np.cos(theta)
        return np.column_stack((x_arr, y_arr, z_arr)).reshape(output_shape)

    @staticmethod
    def cartesian_product(*arrays: List[Any], flip_axis: bool = False):
        """Compute the cartesian product of a set of arrays."""
        len_arr = len(arrays)
        arr = np.empty(
            [len(a) for a in arrays] + [len_arr], dtype=np.result_type(*arrays)
        )
        for i, a in enumerate(np.ix_(*arrays)):
            arr[..., i] = a
        res = arr.reshape(-1, len_arr)
        return res if not flip_axis else np.fliplr(res)

    @staticmethod
    def sphere_to_pixel(points_sph: np.ndarray, width: int) -> np.ndarray:
        """
        Convert spherical coordinates to pixel coordinates inside a 360 pano image with a given width.
        """
        EPS = np.deg2rad(1)
        output_shape = (points_sph.shape[0], 2)  # type: ignore
        num_points = points_sph.shape[0]
        assert num_points > 0
        num_coords = points_sph.shape[1]
        assert num_coords == 2 or num_coords == 3
        height = width / 2
        assert width > 1 and height > 1
        # We only consider the azimuth and elevation angles.
        theta = points_sph[:, 0]
        assert np.all(np.greater_equal(theta, -math.pi - EPS))
        assert np.all(np.less_equal(theta, math.pi + EPS))
        phi = points_sph[:, 1]
        assert np.all(np.greater_equal(phi, -math.pi / 2.0 - EPS))
        assert np.all(np.less_equal(phi, math.pi / 2.0 + EPS))
        # Convert the azimuth to x-coordinates in the pano image, where
        # theta = 0 maps to the horizontal center.
        x_arr = theta + math.pi  # Map to [0, 2*pi]
        x_arr /= 2.0 * math.pi  # Map to [0, 1]
        x_arr *= width - 1  # Map to [0, width)

        # Convert the elevation to y-coordinates in the pano image, where
        # phi = 0 maps to the vertical center.
        y_arr = phi + math.pi / 2.0  # Map to [0, pi]
        y_arr /= math.pi  # Map to [0, 1]
        y_arr = 1.0 - y_arr  # Flip so that y goes up.
        y_arr *= height - 1  # Map to [0, height)

        return np.column_stack((x_arr, y_arr)).reshape(output_shape)

    @classmethod
    def cartesian_to_pixel(cls, points_cart: np.ndarray, width: int):
        return cls.sphere_to_pixel(cls.cartesian_to_sphere(points_cart), width)


class Transformation3D:
    """
    Class to handle transformation from the 2D top-down floor map coordinates to 3D cartesian coordinates
    """

    def __init__(self, ceiling_height: float, camera_height: float):
        """
        :param ceiling_height: The height of the ceiling
        :param camera_height: The height of the camera
        """
        self._ceiling_height = ceiling_height
        self._camera_height = camera_height

    def to_3d(self, room_vertices: List[Point2D]):
        """
        Transform 2D room vertices to 3D cartesian points.

        :param room_vertices: The top-down 2D projected vertices

        :return: Both the floor as well as the ceiling vertices in 3D cartesian coordinates
        """
        # Extract and format room shape coordinates
        num_vertices = room_vertices.shape[0]
        floor_z = np.repeat([-self._camera_height], num_vertices).reshape(
            num_vertices, 1
        )
        ceiling_z = np.repeat(
            [self._ceiling_height - self._camera_height], num_vertices
        ).reshape(num_vertices, 1)

        # Create floor and ceiling coordinates
        floor_coordinates = np.hstack((room_vertices, floor_z))
        ceiling_coordinates = np.hstack((room_vertices, ceiling_z))

        return floor_coordinates, ceiling_coordinates
