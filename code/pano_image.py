"""PanoImage module provides utilities to represent and work with 360 images (in Equirectangular projection).

Typical usage example:
pano_image = PanoImage(image_file_path)     # Load a 360 panorama.
pano_image.draw_XXX(...)                    # Draw various elements like markers and lines.
pano_image_cv = pano_image.opencv_image     # Access the underlying OpenCV representation (in a mutable way).
"""

import logging
import math
import sys
from typing import Tuple

import cv2
import numpy as np

from transformations import TransformationSpherical
from utils import Image

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
LOG = logging.getLogger(__name__)


class PanoImageException(Exception):
    """Custom exception that represents a failure to load a valid 360 panorama."""

    def __init__(self, message):
        message = f"ZInD failed! Error: {message}"

        super(PanoImageException, self).__init__(message)


class PanoImage:
    """Load, parse and represent 360 panorama images in Equirectangular projection.

    The class also implements a few drawing utilities directly in the Equirectangular projection.
    """

    def __init__(self, image_cv: Image) -> None:
        """Initialize 360 panorama from a given OpenCV RGB image.

        :param image_cv: uint8 OpenCV image.
        """
        self._image = image_cv
        self._validate_image()

    @classmethod
    def from_file(cls, image_file_path: str) -> "PanoImage":
        """Initialize 360 panorama from a given RGB image file

        :param image_file_path: The path to the 360 panorama image.

        :return: The corresponding PanoImage object.
        """
        image_cv = cv2.imread(image_file_path, cv2.IMREAD_COLOR)

        # This can happen because of missing file, improper permissions, unsupported or invalid format, etc.
        if image_cv is None:
            raise PanoImageException("Can not load image: {}".format(image_file_path))

        # Convert the underlying data structure from uint8 BGR to uint8 RGB.
        image_cv = cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB)

        return cls(image_cv)

    def write_to_file(self, file_path: str) -> None:
        # Convert the underlying image from RGB floating point to bgr uint8.
        image_bgr_cv = cv2.cvtColor(self.opencv_image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(file_path, image_bgr_cv)

    def _validate_image(self):
        """Verify whether the underlying image represents a valid 360 panorama.

        This method will throw an instance of PanoImageException if validation fails.
        """
        if self._image is None:
            raise PanoImageException("Empty image")

        if not np.issubdtype(self._image.dtype, np.uint8):
            raise PanoImageException("Expecting uint8 image: %s" % self._image.dtype)

        if not self._has_valid_fov:
            raise PanoImageException(
                "Invalid pano dimensions: %d-by-%d" % (self.width, self.height)
            )

    @property
    def _has_valid_fov(self) -> bool:
        """Return true if the pano image dimensions could represent a valid full FoV pano, i.e. 2:1."""
        return self.width == 2 * self.height

    @property
    def width(self) -> int:
        """Return the width of the pano."""
        return self._image.shape[1]

    @property
    def height(self) -> int:
        """Return the height of the pano."""
        return self._image.shape[0]

    @property
    def opencv_image(self) -> Image:
        """Return a shallow copy of the underlying OpenCV image data.

        Note: this is a mutable object and should be treated with care!
        """
        return self._image

    def draw_marker(
        self,
        point_pix: np.ndarray,
        *,
        color: Tuple[int, int, int] = (255, 255, 0),
        marker_size: int = 10,
        thickness: int = 2,
    ):
        """Draw a tilted cross marker (on the underlying pano) given its center in pixel coordinates.

        :param point_pix: The center of the point (in image coordinates).
        :param color: The RGB color of the marker.
        :param marker_size: The size of the marker.
        :param thickness: The thickness of the marker.
        """
        cv2.drawMarker(
            self.opencv_image,
            tuple(np.int_(point_pix)),
            color,
            cv2.MARKER_TILTED_CROSS,
            markerSize=marker_size,
            thickness=thickness,
        )

    def draw_spherical_line(
        self,
        point_start_cart: np.ndarray,
        point_end_cart: np.ndarray,
        color: Tuple[int, int, int] = (255, 255, 255),
        thickness: int = 2,
        thresh_deg: float = 0.5,
    ):
        """Draw a spherical line corresponding to the shorter arc, by properly handling loop-closure crossing lines.

        :param point_start_cart: The start point (in cartesian coordinates).
        :param point_end_cart: The end point (in cartesian coordinates).
        :param color: The RGB color of the line.
        :param thickness: The thickness of the line.
        :param thresh_deg: The angular resolution for approximating spherical line as a set of polylines.
        """
        pt1 = point_start_cart.reshape(1, 3)
        pt2 = point_end_cart.reshape(1, 3)
        thresh_rad = np.deg2rad(thresh_deg)
        pt1_pix = TransformationSpherical.cartesian_to_pixel(pt1, self.width)

        points_stack = [[pt1_pix[0, 0], pt1_pix[0, 1]]]

        lines_stack = [(pt1, pt2)]
        while lines_stack:
            line_curr = lines_stack.pop()

            pt1 = line_curr[0]
            pt2 = line_curr[1]

            angle_curr = np.arccos(np.clip(np.dot(pt1, pt2.T), -1, 1))

            if angle_curr < thresh_rad:
                pt2_pix = TransformationSpherical.cartesian_to_pixel(pt2, self.width)
                points_stack.append([pt2_pix[0, 0], pt2_pix[0, 1]])
            else:
                mid_pt = 0.5 * (pt1 + pt2)
                mid_pt /= np.linalg.norm(mid_pt)
                lines_stack.append((mid_pt, pt2))
                lines_stack.append((pt1, mid_pt))

        # In case of a loop closure line, we split it into two poly lines.
        if self._is_loop_closure_line(self.width, point_start_cart, point_end_cart):
            idx_cut = -1
            for idx, pt_curr in enumerate(points_stack[:-1]):
                pt_next = points_stack[idx + 1]
                if abs(pt_curr[0] - pt_next[0]) > self.width / 2:
                    idx_cut = idx

            assert 0 <= idx_cut < len(points_stack)

            points_left = np.int32([points_stack[0 : idx_cut + 1]])
            cv2.polylines(self.opencv_image, points_left, False, color, thickness)

            points_right = np.int32([points_stack[idx_cut + 1 : -1]])
            cv2.polylines(self.opencv_image, points_right, False, color, thickness)
        else:
            cv2.polylines(
                self.opencv_image, np.int32([points_stack]), False, color, thickness
            )

    def draw_dotted_line(
        self,
        point_start_cart,
        point_end_cart,
        *,
        color: Tuple[int, int, int] = (255, 255, 255),
        thickness: int = 1,
        thresh_deg: float = 2.0,
    ):
        """Draw a dotted spherical line given the two end points in cartesian coordinates.

        :param point_start_cart: The start point.
        :param point_end_cart: The end point.
        :param color: The RGB color of the line.
        :param thickness: The thickness of the line.
        :param thresh_deg: The distance between the dotted points (in degrees).
        """
        # Controls the angular space between the dots.
        thresh_rad = np.deg2rad(thresh_deg)

        pt1 = np.asarray(point_start_cart).reshape(1, 3)
        pt2 = np.asarray(point_end_cart).reshape(1, 3)

        pt1_pix = TransformationSpherical.cartesian_to_pixel(pt1, self.width)

        points_stack = [[pt1_pix[0, 0], pt1_pix[0, 1]]]

        lines_stack = [(pt1, pt2)]
        while lines_stack:
            line_curr = lines_stack.pop()

            pt1 = line_curr[0]
            pt2 = line_curr[1]

            angle_curr = np.arccos(np.clip(np.dot(pt1, pt2.T), -1, 1))

            if angle_curr < thresh_rad:
                pt2_pix = TransformationSpherical.cartesian_to_pixel(pt2, self.width)
                points_stack.append([pt2_pix[0, 0], pt2_pix[0, 1]])
            else:
                mid_pt = 0.5 * (pt1 + pt2)
                mid_pt /= np.linalg.norm(mid_pt)
                lines_stack.append((mid_pt, pt2))
                lines_stack.append((pt1, mid_pt))

                # Transform to pixel coordinates.
                mid_pt_pix = TransformationSpherical.cartesian_to_pixel(
                    mid_pt, self.width
                )
                cv2.circle(
                    self.opencv_image,
                    tuple(np.int_(mid_pt_pix[0])),
                    thickness,
                    color,
                    -1,
                )

    def draw_vertical_line(
        self,
        points: Tuple[np.ndarray, np.ndarray],
        *,
        color: Tuple[int, int, int] = (255, 255, 255),
        thickness: int = 2,
    ):
        """Draw a vertical line given the two points in cartesian coordinates."""
        pt1_pix, pt2_pix = self._get_vertical_line_image_coordinates(points)
        cv2.line(
            self.opencv_image,
            (pt1_pix[0], pt1_pix[1]),
            (pt2_pix[0], pt2_pix[1]),
            color,
            thickness=thickness,
        )

    def _get_vertical_line_image_coordinates(
        self, points: Tuple[np.ndarray, np.ndarray]
    ):
        """Get the endpoints of a vertical line, in img coords, running from 3d points[0] to points[1]"""
        pt1_pix = TransformationSpherical.cartesian_to_pixel(
            points[0].reshape(1, 3), self.width
        )
        pt1_pix = np.int_(np.squeeze(pt1_pix))
        pt2_pix = TransformationSpherical.cartesian_to_pixel(
            points[1].reshape(1, 3), self.width
        )
        pt2_pix = np.int_(np.squeeze(pt2_pix))
        return pt1_pix, pt2_pix

    def _is_loop_closure_line(self, width: int, pt1: np.ndarray, pt2: np.ndarray):
        """Check if a given line is a "loop closure line", meaning that it's rendering on the pano texture would
        wrap around the left/right border.
        """
        pt1 = pt1.reshape(1, 3)
        pt2 = pt2.reshape(1, 3)

        pt1_pix = TransformationSpherical.cartesian_to_pixel(pt1, width)
        pt2_pix = TransformationSpherical.cartesian_to_pixel(pt2, width)

        mid_pt = 0.5 * (pt1 + pt2)
        mid_pt /= np.linalg.norm(mid_pt)

        mid_pt_pix = TransformationSpherical.cartesian_to_pixel(mid_pt, width)

        dist_total = abs(pt1_pix[0, 0] - pt2_pix[0, 0])
        dist_left = abs(pt1_pix[0, 0] - mid_pt_pix[0, 0])
        dist_right = abs(pt2_pix[0, 0] - mid_pt_pix[0, 0])

        return dist_total > width / 2.0 or dist_left + dist_right > dist_total + 1

    def rotate_pano(self, rot: np.matrix):
        """Rotate a pano image by the given rotation matrix.

        Note:
        We are using a coordinate system where the 3D cartesian x-axis maps to
        the horizontal image axis and the 3D cartesian y-axis maps to the
        vertical image axis, and finally the 3D cartesian z-axis maps inwards
        the panorama image plane. Thus a rotation around the y-axis would
        correspond to panning the panorama left-to-right or right-to-left.
        Rotation around the x-axis would correspond to looking up/down, and
        rotation around the z-axis would correspond to tilting the camera, e.g.
        this will "tilt" the straight horizontal lines in the equirectangular
        spherical projection that we are assuming.

        Args:
            pano_image: The source pano image (will not be modified).

            rot: The 3-by-3 rotation matrix.

        Return:
            The rotated panorama image.
        """
        assert self._has_valid_fov
        width = self.width
        height = self.height
        theta = np.linspace(start=-math.pi, stop=math.pi, num=width)
        fov_half = math.pi / 2.0
        # each pixel corresponds to a
        phi = np.linspace(start=fov_half, stop=-fov_half, num=height)
        points_sph_col_major = np.asarray(
            TransformationSpherical.cartesian_product(phi, theta)
        )
        points_sph = np.fliplr(points_sph_col_major)
        points_cart = TransformationSpherical.sphere_to_cartesian(points_sph)
        rot_inv = rot.T
        points_cart_rot = np.matmul(rot_inv, points_cart.T).T
        points_cart_rot_arrange = np.asarray(
            [points_cart_rot[:, 0], -points_cart_rot[:, 2], points_cart_rot[:, 1]]
        ).T
        points_sph_rot = TransformationSpherical.cartesian_to_sphere(
            points_cart_rot_arrange
        )
        points_pix_rot = TransformationSpherical.sphere_to_pixel(points_sph_rot, width)

        # Convert to OpenCV compatible maps for cv2.remap.
        map_x = points_pix_rot[:, 0].reshape((height, width)).astype(np.float32)
        map_y = points_pix_rot[:, 1].reshape((height, width)).astype(np.float32)
        img_dst_cv = cv2.remap(
            self.opencv_image,
            map_x,
            map_y,
            interpolation=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_REPLICATE,
        )
        return PanoImage(img_dst_cv)
