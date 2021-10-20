"""FloorPlan module provides utilities to represent and work with ZInD floor plans.

Typical usage example:
zfp = FloorPlan(input_json_file)  # Load and parse ZInD JSON file as a Zillow FloorPlan object.
top_down_layouts = zfp.floor_plan_layouts["raw"]  # Retrieve the 2D merged top-down floor plan layouts and WDO.
panos_layouts = zfp.panos_layouts["raw"]["primary"] # Retrieve the 3D per-pano floor plan layouts and WDO.
"""

import json
import logging
import sys
from pathlib import Path
from typing import Dict, Any, List

import numpy as np

from transformations import Transformation2D
from utils import Polygon, PolygonType, PolygonTypeMapping

RAW_GEOMETRY_KEY = "raw"
COMPLETE_GEOMETRY_KEY = "complete"
VISIBLE_GEOMETRY_KEY = "visible"
REDRAW_GEOMETRY_KEY = "redraw"
PRIMARY_PANO_KEY = "primary"
SECONDARY_PANO_KEY = "secondary"

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
LOG = logging.getLogger(__name__)


class FloorPlanException(Exception):
    """Exception class for ZInD floor plan data parsing and validation."""

    def __init__(self, message):
        message = f"ZInD failed! Error: {message}"

        super(FloorPlanException, self).__init__(message)


class FloorPlan:
    """Load, parse and retrieve ZInD floor plan data.

    A class that handles the custom ZInD floor plan type as a collection of polygon room shapes, wall objects like
    windows/doors/openings (WDO for short) and their location in the global merged floor plan coordinate system.
    """

    def __init__(self, json_file_name):
        """The init method will create a floor plan object from the custom ZInD JSON file."""
        with open(json_file_name) as json_file:
            floor_map_json = json.load(json_file)

        # We will split the parsed 3D per-pano layouts based on the different layouts and pano types.
        self._panos_layouts = {
            RAW_GEOMETRY_KEY: {PRIMARY_PANO_KEY: [], SECONDARY_PANO_KEY: []},
            COMPLETE_GEOMETRY_KEY: {PRIMARY_PANO_KEY: [], SECONDARY_PANO_KEY: []},
            VISIBLE_GEOMETRY_KEY: {PRIMARY_PANO_KEY: [], SECONDARY_PANO_KEY: []},
        }

        self._json_file_name = json_file_name
        self._input_folder = Path(json_file_name).resolve().parent

        # We will split the parsed 2D floor plan data based on the different layouts, e.g. raw, complete or redraw.
        self._floor_plan_layouts = {
            RAW_GEOMETRY_KEY: {},
            COMPLETE_GEOMETRY_KEY: {},
            REDRAW_GEOMETRY_KEY: {},
        }

        # Collect floor plan redraw alignment data.
        self._floor_plan_redraw_align = {}
        self._floor_plan_complete_align = {}
        self._floor_plan_to_redraw_transformation = {}
        self._floor_plan_image_path = {}

        # Make sure we have the required top-level fields
        assert "merger" in floor_map_json
        assert "redraw" in floor_map_json
        assert "floorplan_to_redraw_transformation" in floor_map_json

        merger_data = floor_map_json["merger"]
        redraw_data = floor_map_json["redraw"]
        floor_plan_redraw_alignment_data = floor_map_json[
            "floorplan_to_redraw_transformation"
        ]

        # Parse the merger data: raw, complete, and visible layouts + WDO as well as the draft floor plan.
        self.parse_merger(merger_data)

        # Parse the redraw data: this is the final cleanup floor-plan.
        self.parse_redraw(redraw_data)

        # Parse the transformations that map between the final floor plan image (raster) and the redraw data (vector).
        self.parse_floor_plan_redraw_alignment(floor_plan_redraw_alignment_data)

    @property
    def input_folder(self):
        return self._input_folder

    @property
    def panos_layouts(self):
        return self._panos_layouts

    @property
    def floor_plan_layouts(self):
        return self._floor_plan_layouts

    @property
    def floor_plan_to_redraw_transformation(self):
        return self._floor_plan_to_redraw_transformation

    @property
    def floor_plan_image_path(self):
        return self._floor_plan_image_path

    def parse_redraw(self, redraw_data: Dict[str, Any]):
        """Parse the final redraw geometry.

        The function will modify self._floor_plan_layouts

        :param redraw_data: The parsed ZInD redraw data.
        :return: None
        """
        for floor_id, floor_data in redraw_data.items():
            self._floor_plan_layouts[REDRAW_GEOMETRY_KEY][floor_id] = []

            redraw_vertices_list = []
            redraw_room_wdo_poly_list = []
            for room_id, room_data in floor_data.items():
                room_vertices_global = room_data["vertices"]
                redraw_room_wdo_poly_list.append(
                    Polygon(
                        type=PolygonType.ROOM,
                        name=self._json_file_name,
                        points=Polygon.list_to_points(np.array(room_vertices_global)),
                    )
                )
                redraw_vertices_list.extend(room_vertices_global)

                zind_points_list = Polygon.list_to_points(room_vertices_global)
                zind_poly = Polygon(type=PolygonType.ROOM, points=zind_points_list)

                # Validate the room polygon: may throw FloorPlanException.
                self.validate_room_polygon(zind_poly)

                self._floor_plan_layouts[REDRAW_GEOMETRY_KEY][floor_id].append(
                    zind_poly
                )

                # Parse windows/doors, note that the redraw geometry does not contain openings.
                for wdo_type in ["windows", "doors"]:
                    for wdo_data in room_data[wdo_type]:
                        redraw_room_wdo_poly_list.append(
                            Polygon(
                                type=PolygonTypeMapping[wdo_type],
                                points=Polygon.list_to_points(np.array(wdo_data)),
                            )
                        )

                        wdo_poly = Polygon(
                            type=PolygonTypeMapping[wdo_type],
                            points=Polygon.list_to_points(wdo_data),
                        )
                        self._floor_plan_layouts[REDRAW_GEOMETRY_KEY][floor_id].append(
                            wdo_poly
                        )

                # Collect pin labels (if any).
                for pin_data in room_data["pins"]:
                    pin_label_position = np.array(pin_data["position"], ndmin=2)

                    pin_label_poly = Polygon(
                        type=PolygonType.PIN_LABEL,
                        name=pin_data["label"],
                        points=Polygon.list_to_points(pin_label_position),
                    )

                    self._floor_plan_layouts[REDRAW_GEOMETRY_KEY][floor_id].append(
                        pin_label_poly
                    )

    def _parse_merger_wdo(
        self,
        wdo_vertices_local: List[float],
        *,
        wdo_type: str,
        transformation: np.ndarray,
        zind_poly: Polygon = None,
    ):
        """Parse the WDO portion of the layout field.

        :param wdo_vertices_local: The WDO positions in the local coordinate system.
        :param wdo_type: The type, e.g. window, door or opening
        :param transformation: The local to global coordinate system transformation.
        :param zind_poly: ZInD room shape polygon object.

        :return: The parsed WDO elements as a list of polygons.
        """
        # Skip if there are no elements of this type.
        if len(wdo_vertices_local) == 0:
            return

        num_wdo = len(wdo_vertices_local) // 3
        wdo_left_right_bound = []
        for wdo_idx in range(num_wdo):
            wdo_left_right_bound.extend(
                wdo_vertices_local[wdo_idx * 3 : wdo_idx * 3 + 2]
            )
        wdo_vertices_global = transformation.to_global(np.array(wdo_left_right_bound))

        wdo_poly_list = []

        # Every two points in the list define windows/doors/openings by left and right boundaries, so
        # for N elements we will have 2 * N pair of points, thus we iterate on every successive pair
        for wdo_points in zip(wdo_vertices_global[::2], wdo_vertices_global[1::2]):
            zind_points_list = Polygon.list_to_points(wdo_points)
            zind_poly_type = PolygonTypeMapping[wdo_type]
            wdo_poly = Polygon(
                type=zind_poly_type, name=self._json_file_name, points=zind_points_list
            )

            # Add the WDO element to the list of polygons/lines.
            wdo_poly_list.append(wdo_poly)

            # Validate the WDO element: may throw FloorPlanException
            self.validate_wdo_polygon(wdo_poly, zind_poly=zind_poly)

        return wdo_poly_list

    def parse_merger(self, merger_data: Dict[str, Any]):
        """Parse the merger data field.

        This includes the following information for every floor:
        1. The tree-like structure of complete, partial rooms and raw layout annotations.
        1. The raw, complete and visible 3D layouts for each pano alongside the WDO elements.
        2. The 2D transformations to build the draft merger floor plan from the individual layout pieces.

        The function will modify (1) self._panos_layouts and (2) self._floor_plan_layouts

        :param merger_data: The merger data field from the ZInD JSON.

        :return: None
        """
        # Top level merger data is per-floor.
        for floor_id, floor_data in merger_data.items():
            # Create a list of all the ZInD polygons for this floor: rooms, windows, doors, openings.
            raw_zind_poly_list = []
            complete_zind_poly_list = []
            for complete_room_id, complete_room_data in floor_data.items():
                complete_geometry_has_collected = False
                for partial_room_id, partial_room_data in complete_room_data.items():
                    for pano_id, pano_data in partial_room_data.items():
                        # Create a transformation object that will move points from local to global coordinates.
                        transformation = Transformation2D.from_zind_data(
                            pano_data["floor_plan_transformation"]
                        )

                        for geometry_type in [
                            "layout_raw",
                            "layout_complete",
                            "layout_visible",
                        ]:
                            if geometry_type not in pano_data:
                                # Note that it is expected behavior to have missing floor plan layouts for some rooms,
                                # e.g. small closet that often times have outside the room annotations.
                                LOG.debug(
                                    "Missing layout {}: {}/{}/{}/{}".format(
                                        geometry_type,
                                        floor_id,
                                        complete_room_id,
                                        partial_room_id,
                                        pano_id,
                                    )
                                )
                                continue

                            zind_poly_list = []

                            # Transform and validate the room shape polygon.
                            room_vertices_local = np.asarray(
                                pano_data[geometry_type]["vertices"]
                            )
                            room_vertices_global = transformation.to_global(
                                room_vertices_local
                            )

                            zind_points_list = Polygon.list_to_points(
                                room_vertices_global
                            )
                            zind_poly = Polygon(
                                type=PolygonType.ROOM, points=zind_points_list
                            )
                            self.validate_room_polygon(zind_poly)
                            zind_poly_list.append(zind_poly)

                            # For complete geometry, we need to visualize the internal vertices.
                            internal_vertices_local_list = []
                            if geometry_type == "layout_complete":
                                for internal_vertices in pano_data[geometry_type][
                                    "internal"
                                ]:
                                    # Validate internal vertices.
                                    internal_vertices_local = np.asarray(
                                        internal_vertices
                                    )
                                    internal_vertices_local_list.append(
                                        internal_vertices_local
                                    )
                                    internal_vertices_global = transformation.to_global(
                                        internal_vertices_local
                                    )
                                    internal_poly = Polygon(
                                        type=PolygonType.ROOM,
                                        points=Polygon.list_to_points(
                                            internal_vertices_global
                                        ),
                                    )
                                    self.validate_room_polygon(internal_poly)
                                    zind_poly_list.append(internal_poly)

                            image_path = pano_data["image_path"]

                            # Collect pano data that we will use later to verify rendering on the pano textures.
                            pano_data_for_render = {}
                            pano_data_for_render["pano_id"] = "_".join(
                                [floor_id, complete_room_id, partial_room_id, pano_id]
                            )
                            pano_data_for_render["room_vertices"] = room_vertices_local
                            pano_data_for_render[
                                "internal_vertices"
                            ] = internal_vertices_local_list
                            pano_data_for_render["image"] = image_path
                            pano_data_for_render["camera_height"] = pano_data[
                                "camera_height"
                            ]
                            pano_data_for_render["ceiling_height"] = pano_data[
                                "ceiling_height"
                            ]

                            # Add the camera center to the list of elements to render later.
                            # Note that the local 2D camera center coordinate is always at (0, 0).
                            camera_center_global = transformation.to_global(
                                np.asarray([[0, 0]])
                            )

                            camera_type = (
                                PolygonType.PRIMARY_CAMERA
                                if pano_data["is_primary"]
                                else PolygonType.SECONDARY_CAMERA
                            )

                            camera_center_poly = Polygon(
                                type=camera_type,
                                points=Polygon.list_to_points(camera_center_global),
                            )

                            wdo_poly_list = []

                            # Parse the WDO elements (if any).
                            for wdo_type in ["windows", "doors", "openings"]:
                                wdo_vertices_local = np.asarray(
                                    pano_data[geometry_type][wdo_type]
                                )
                                pano_data_for_render[wdo_type] = wdo_vertices_local

                                # Skip if there are no elements of this type.
                                if len(wdo_vertices_local) == 0:
                                    continue

                                # Parse the current list of WDO elements and add it to the global list.
                                wdo_poly_list.extend(
                                    self._parse_merger_wdo(
                                        wdo_vertices_local,
                                        wdo_type=wdo_type,
                                        transformation=transformation,
                                        zind_poly=zind_poly,
                                    )
                                )

                            if pano_data["is_primary"]:
                                pano_key = PRIMARY_PANO_KEY
                            else:
                                pano_key = SECONDARY_PANO_KEY

                            # Add the current room shape, camera and wdo polygons to the list of all polygons.
                            if geometry_type == "layout_raw":
                                self._panos_layouts[RAW_GEOMETRY_KEY][pano_key].append(
                                    pano_data_for_render
                                )
                                raw_zind_poly_list.append(camera_center_poly)
                                if pano_data["is_primary"]:
                                    raw_zind_poly_list.append(zind_poly)
                                    raw_zind_poly_list.extend(wdo_poly_list)
                            elif geometry_type == "layout_complete":
                                self._panos_layouts[COMPLETE_GEOMETRY_KEY][
                                    pano_key
                                ].append(pano_data_for_render)
                                complete_zind_poly_list.append(camera_center_poly)

                                if not complete_geometry_has_collected:
                                    complete_zind_poly_list.extend(zind_poly_list)
                                    complete_zind_poly_list.extend(wdo_poly_list)
                                    complete_geometry_has_collected = True

                            elif geometry_type == "layout_visible":
                                self._panos_layouts[VISIBLE_GEOMETRY_KEY][
                                    pano_key
                                ].append(pano_data_for_render)
                                # Note that we do not visualize floor plan for visible geometry
                            else:
                                raise Exception(
                                    "Invalid geometry_type: {}".format(geometry_type)
                                )

            self._floor_plan_layouts[RAW_GEOMETRY_KEY][floor_id] = raw_zind_poly_list
            self._floor_plan_layouts[COMPLETE_GEOMETRY_KEY][
                floor_id
            ] = complete_zind_poly_list

    def parse_floor_plan_redraw_alignment(
        self, floor_plan_redraw_alignment_data: Dict[str, Any]
    ):
        """Parse the alignment between the raster floor plan image and the final redraw geometry.

        The function will modify self._floor_plan_to_redraw_transformation

        :param floor_plan_redraw_alignment_data: The parsed ZInD floor plan to redraw alignment data.
        :return: None
        """
        for floor_id, floor_data in floor_plan_redraw_alignment_data.items():
            transformation = Transformation2D.from_zind_data(floor_data)
            self._floor_plan_to_redraw_transformation[floor_id] = transformation
            self._floor_plan_image_path[floor_id] = floor_data["image_path"]

    def validate_room_polygon(self, zind_poly: Polygon):
        """Validate room polygon vertices:
        (1) Each room polygon must have 3 points or above.
        (2) No self intersections.

        :param zind_poly: ZInD polygon object.

        Throws a FloorPlanException if data can not be validated
        """
        if zind_poly.num_points < 3:
            raise FloorPlanException(
                "Invalid room polygon (insufficient number of corners): {}".format(
                    zind_poly
                )
            )

        shapely_poly = zind_poly.to_shapely_poly
        if not shapely_poly.is_valid:
            raise FloorPlanException(
                "Invalid polygon (self-intersecting): {}".format(zind_poly)
            )

    def validate_wdo_polygon(
        self,
        wdo_poly: Polygon,
        *,
        zind_poly: Polygon = None,
        dist_threshold: float = 1.0,
    ):
        """Validate WDO element vertices against the corresponding ZInD room shape polygon:
        (1) Each WDO element must be defined by exactly two points (left/right boundaries)
        (2) Each WDO element must intersect the room shape geometry
        (3) TODO: verify that each WDO element lies on a single polygon line

        :param wdo_poly: WDO polygon object.
        :param zind_poly: ZInD polygon object (if None then no validation against room shape will be performed).
        :param dist_threshold: Threshold that will be used to check if two geometries intersect (using shapely).

        Throws a FloorPlanException if data can not be validated
        """
        if wdo_poly.num_points != 2:
            raise FloorPlanException(
                "Invalid WDO number of corners: {}".format(wdo_poly)
            )

        wdo_poly_shapely = wdo_poly.to_shapely_line

        if zind_poly is not None:
            zind_poly_shapely = zind_poly.to_shapely_poly

            dist = wdo_poly_shapely.distance(zind_poly_shapely)
            is_intersected = dist < dist_threshold

            if not is_intersected:
                raise FloorPlanException(
                    "Invalid WDO room shape intersection: {} {}".format(wdo_poly, dist)
                )
