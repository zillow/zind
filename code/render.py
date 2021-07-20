"""
This module contains some common rendering routines.
"""
import itertools
import logging
import os
import sys
from typing import List, Dict, Any

import cv2
import numpy as np
from pano_image import PanoImage
from transformations import TransformationSpherical, Transformation3D
from utils import Polygon, PolygonType

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
LOG = logging.getLogger(__name__)

# Default parameters when drawing ZInD floor plans.
DEFAULT_LINE_THICKNESS = 4
DEFAULT_RENDER_RESOLUTION = 2048

# Polygon colors when we render the floor map as a JPG image.
POLYGON_COLOR = {
    PolygonType.ROOM: (0, 0, 0),  # Black
    PolygonType.WINDOW: (0, 255, 255),  # Cyan
    PolygonType.DOOR: (255, 255, 0),  # Yellow-ish
    PolygonType.OPENING: (0, 0, 255),  # Blue
    PolygonType.PRIMARY_CAMERA: (255, 64, 255),  # Purple
    PolygonType.SECONDARY_CAMERA: (0, 128, 0),  # Green
    PolygonType.PIN_LABEL: (0, 0, 0),  # Black
}


def render_jpg_image(
    polygon_list: List[Polygon],
    *,
    jpg_file_name: str = None,
    thickness: int = DEFAULT_LINE_THICKNESS,
    output_width: int = DEFAULT_RENDER_RESOLUTION
):
    """Render a set of ZInD polygon objects to an image that can be saved to the file system.

    :param polygon_list: List of Polygon objects.
    :param jpg_file_name: File name to save the image to (if None we won't save).
    :param thickness: The line thickness when drawing the polygons.
    :param output_width: The default output resolution.

    :return: An OpenCV image object.
    """
    min_x = polygon_list[0].points[0][0]
    min_y = polygon_list[0].points[0][1]

    for polygon in polygon_list:
        for point in polygon.points:
            min_x = min(min_x, point[0])
            min_y = min(min_y, point[1])

    # Normalize based on the upper-left.
    polygon_list_points = []
    for polygon in polygon_list:
        polygon_modified = []
        for point in polygon.points:
            polygon_modified.append((point[0] - min_x, point[1] - min_y))
        polygon_list_points.append(polygon_modified)

    max_x = polygon_list_points[0][0][0]
    max_y = polygon_list_points[0][0][1]
    for point in itertools.chain.from_iterable(polygon_list_points):
        max_x = max(max_x, point[0])
        max_y = max(max_y, point[1])

    resize_ratio = output_width / max_x

    # Normalize based on the max width.
    polygon_list_points_modified = []
    for polygon in polygon_list_points:
        polygon_modified = []
        for point in polygon:
            polygon_modified.append((point[0] * resize_ratio, point[1] * resize_ratio))
        polygon_list_points_modified.append(polygon_modified)
    polygon_list_points = polygon_list_points_modified

    max_x = max_x * resize_ratio
    max_y = max_y * resize_ratio

    polygon_list = [
        p._replace(points=q) for p, q in zip(polygon_list, polygon_list_points)
    ]

    img_floor_map = np.zeros([int(max_y + 1), int(max_x + 1), 3], dtype=np.uint8)

    img_floor_map[:] = (255, 255, 255)
    for polygon in polygon_list:
        polygon_points = polygon.points

        try:
            # Draw wall elements like windows/doors/openings with increased thickness for better visualization.
            wall_thickness = (
                thickness if polygon.type == PolygonType.ROOM else 2 * thickness
            )

            cv2.polylines(
                img_floor_map,
                [np.int_([polygon_points])],
                True,
                POLYGON_COLOR[polygon.type],
                thickness=wall_thickness,
                lineType=cv2.LINE_AA,
            )

            # Draw the line/polygon points as Red dots, unless this is the camera center or pin label.
            if (
                polygon.type != PolygonType.PRIMARY_CAMERA
                and polygon.type != PolygonType.SECONDARY_CAMERA
                and polygon.type != PolygonType.PIN_LABEL
            ):
                for point in polygon_points:
                    cv2.circle(
                        img_floor_map, tuple(np.int_(point)), thickness, (255, 0, 0), -1
                    )
            elif polygon.type == PolygonType.PIN_LABEL:
                pin_label_position = np.int_(polygon_points[0]) - np.array([40, 15])
                cv2.putText(
                    img_floor_map,
                    polygon.name,
                    tuple(pin_label_position),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    POLYGON_COLOR[polygon.type],
                    2,
                    cv2.LINE_AA,
                )
        except Exception as ex:
            LOG.debug("Error drawing {}: {} {}".format(jpg_file_name, polygon, str(ex)))
            continue

    # Convert from RGB to BGR (the default type OpenCV expects).
    img_floor_map = cv2.cvtColor(img_floor_map, cv2.COLOR_RGB2BGR)

    if jpg_file_name is not None:
        cv2.imwrite(jpg_file_name, img_floor_map)

    return img_floor_map


def render_room_vertices_on_panos(
    panos_list: List[Dict[str, Any]], *, input_folder: str, output_folder: str
):
    """Render room vertices (floor and ceiling) and WDO elements on the pano textures.

    :param panos_list: The list of collected per-pano data.
    :param input_folder: The input folder, used to locate the pano textures.
    :param output_folder: The output folder, where the rendered layouts will be saved to.

    :return: None
    """
    PolygonTypeMapping = {
        "windows": PolygonType.WINDOW,
        "doors": PolygonType.DOOR,
        "openings": PolygonType.OPENING,
    }

    for pano_data in panos_list:
        pano_id = pano_data["pano_id"]
        pano_image_path = os.path.join(input_folder, pano_data["image"])
        pano_image = PanoImage.from_file(pano_image_path)
        pano_width = pano_image.width

        transform = Transformation3D(
            camera_height=pano_data["camera_height"],
            ceiling_height=pano_data["ceiling_height"],
        )

        vertices_types_to_visualize = [pano_data["room_vertices"]]
        # visualize internal_vertices (if any)
        vertices_types_to_visualize.extend(pano_data["internal_vertices"])
        for vertices_to_visualize in vertices_types_to_visualize:
            floor_coordinates, ceiling_coordinates = transform.to_3d(
                vertices_to_visualize
            )

            floor_coords = TransformationSpherical.normalize(floor_coordinates)
            ceiling_coords = TransformationSpherical.normalize(ceiling_coordinates)
            assert floor_coords.shape[0] == ceiling_coords.shape[0]

            num_vertices = ceiling_coords.shape[0]

            for room_coords in [floor_coords, ceiling_coords]:
                for i in range(num_vertices):
                    point_start = room_coords[[i], :].tolist()
                    point_end = room_coords[[(i + 1) % num_vertices], :].tolist()

                    point_start_pix = TransformationSpherical.cartesian_to_pixel(
                        np.asarray(point_start).reshape(1, 3), pano_width
                    )

                    point_end_pix = TransformationSpherical.cartesian_to_pixel(
                        np.asarray(point_end).reshape(1, 3), pano_width
                    )

                    # Draw a dotted spherical line to represent the proxy layout geometry.
                    pano_image.draw_dotted_line(point_start, point_end)

                    # Draw markers to represent the corners.
                    pano_image.draw_marker(point_start_pix[0])
                    pano_image.draw_marker(point_end_pix[0])

        for wdo_type in ["windows", "doors", "openings"]:
            # Skip floor_plan_layouts that might be missing this field.
            if wdo_type not in pano_data:
                continue

            wdo_vertices = pano_data[wdo_type]
            if len(wdo_vertices) == 0:
                continue

            # Each WDO is represented by three continuous elements:
            # (left boundary x,y); (right boundary x,y); (bottom boundary z, top boundary z)
            assert len(wdo_vertices) % 3 == 0
            num_wdo = len(wdo_vertices) // 3
            for wdo_idx in range(num_wdo):
                bottom_z = wdo_vertices[wdo_idx * 3 + 2][0]
                top_z = wdo_vertices[wdo_idx * 3 + 2][1]
                # wdo_bbox_3D contains four points at bottom left, bottom right, top right, top left
                wdo_bbox_3D = np.array(
                    [
                        [
                            wdo_vertices[wdo_idx * 3][0],
                            wdo_vertices[wdo_idx * 3][1],
                            bottom_z,
                        ],
                        [
                            wdo_vertices[wdo_idx * 3 + 1][0],
                            wdo_vertices[wdo_idx * 3 + 1][1],
                            bottom_z,
                        ],
                        [
                            wdo_vertices[wdo_idx * 3 + 1][0],
                            wdo_vertices[wdo_idx * 3 + 1][1],
                            top_z,
                        ],
                        [
                            wdo_vertices[wdo_idx * 3][0],
                            wdo_vertices[wdo_idx * 3][1],
                            top_z,
                        ],
                    ]
                )
                wdo_bbox_3D = TransformationSpherical.normalize(wdo_bbox_3D)
                pano_image.draw_vertical_line(
                    (wdo_bbox_3D[0], wdo_bbox_3D[3]),
                    color=POLYGON_COLOR[PolygonTypeMapping[wdo_type]],
                )
                pano_image.draw_vertical_line(
                    (wdo_bbox_3D[1], wdo_bbox_3D[2]),
                    color=POLYGON_COLOR[PolygonTypeMapping[wdo_type]],
                )
                pano_image.draw_spherical_line(
                    wdo_bbox_3D[0],
                    wdo_bbox_3D[1],
                    color=POLYGON_COLOR[PolygonTypeMapping[wdo_type]],
                )
                pano_image.draw_spherical_line(
                    wdo_bbox_3D[3],
                    wdo_bbox_3D[2],
                    color=POLYGON_COLOR[PolygonTypeMapping[wdo_type]],
                )

        output_file_name = os.path.join(output_folder, "{}_layout.jpg".format(pano_id))
        pano_image.write_to_file(output_file_name)


def render_raster_to_vector_alignment(
    room_wdo_poly_list: List[Polygon],
    transformation: np.ndarray,
    floorplan_image_file_name: str,
    output_file_name: str,
):
    """Render the raster to vector alignment as an image, using a transformation that maps the given vector
    representation to the raster floor plan image.

    Note that the alignment will not match perfectly due to the final floor plan clean-up stage introducing a
    variety of final touchups, such as fixing misalignments of walls, doors, windows, etc.

    :param room_wdo_poly_list: The vectorized floor plan as a set of polygons.
    :param transformation: The 3-by-3 transformation representing translation, rotation and scale.
    :param floorplan_image_file_name: The raster floor plan image.
    :param output_file_name: The output file name where the raster to vector alignment image will be stored.

    :return: None
    """
    png_floor_image = cv2.imread(floorplan_image_file_name)
    png_floor_image = cv2.cvtColor(png_floor_image, cv2.COLOR_BGR2RGB)

    for room_wdo_poly in room_wdo_poly_list:
        if room_wdo_poly.type == PolygonType.ROOM:
            color = (255, 0, 0)  # Red
        else:
            color = POLYGON_COLOR[room_wdo_poly.type]
        room_wdo_vertices = room_wdo_poly.points
        png_floor_coordinates = transformation.apply_inverse(room_wdo_vertices)
        cv2.polylines(
            png_floor_image,
            [np.int_(png_floor_coordinates)],
            True,
            color,
            thickness=2,
            lineType=cv2.LINE_AA,
        )

    png_floor_image = cv2.cvtColor(png_floor_image, cv2.COLOR_RGB2BGR)

    cv2.imwrite(output_file_name, png_floor_image)
