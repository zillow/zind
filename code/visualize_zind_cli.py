# """CLI script to visualize & validate data for the public-facing Zillow Indoor Dataset (ZInD).
#
# Validation includes:
#  (1) required JSON fields are presented
#  (2) verify non self-intersection of room floor_plan_layouts
#  (3) verify that windows/doors/openings lie on the room layout geometry
#  (4) verify that windows/doors/openings are defined by two points (left/right boundaries)
#  (5) verify that panos_layouts are RGB images with valid FoV ratio (2:1)
#
# Visualization includes:
#  (1) render the top-down floor map projection: merged room floor_plan_layouts,WDO and camera centers
#  (2) render the room floor_plan_layouts and windows/doors/openings on the pano
#
# Example usage (1): Render all layouts on primary and secondary panos.
#  python visualize_zind_cli.py -i <input_folder> -o <output_folder> --visualize-layout --visualize-floor-plan \
#  --raw --complete --visible --primary --secondary
#
# Example usage (2): Render all vector layouts using merger (based on raw or complete) and the final redraw layouts.
#  python visualize_zind_cli.py -i <input_folder> -o <output_folder> --visualize-floor-plan --redraw --complete --raw
#
# Example usage (3): Render the raster to vector alignments using merger (based on raw or complete) and final redraw.
#  python visualize_zind_cli.py -i <input_folder> -o <output_folder> --visualize-raster --redraw --complete --raw
#

import argparse
import logging
import os
import sys
import traceback
from pathlib import Path
from typing import Dict, Any

from floor_plan import FloorPlan
from render import (
    render_room_vertices_on_panos,
    render_jpg_image,
    render_raster_to_vector_alignment,
)
from tqdm import tqdm

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
LOG = logging.getLogger(__name__)

RENDER_FOLDER = "render_data"


def validate_and_render(
    zillow_floor_plan: "FloorPlan",
    *,
    input_folder: str,
    output_folder: str,
    args: Dict[str, Any]
):
    """Validate and render various ZInD elements, e.g.
    1. Primary/secondary layout and WDO
    2. Raw/complete/visible layouts
    3. Top-down merger results (draft floor-plan)
    4. Top-down redraw results (final floor-plan)
    5. Raster to vector alignment results.

    :param zillow_floor_plan: ZInD floor plan object.
    :param input_folder: Input folder of the current tour.
    :param output_folder: Folder where the renderings will be saved.
    :param args: Input arguments to the script.

    :return: None
    """
    # Get the types of floor_plan_layouts that we should render.
    geometry_to_visualize = []
    if args.raw:
        geometry_to_visualize.append("raw")
    if args.complete:
        geometry_to_visualize.append("complete")
    if args.visible:
        geometry_to_visualize.append("visible")
    if args.redraw:
        geometry_to_visualize.append("redraw")

    # Get the types of panos_layouts that we should render.
    panos_to_visualize = []
    if args.primary:
        panos_to_visualize.append("primary")
    if args.secondary:
        panos_to_visualize.append("secondary")

    # Render the room shape layouts + WDO on top of the pano textures.
    if args.visualize_layout:
        for geometry_type in geometry_to_visualize:
            if geometry_type == "redraw":
                continue
            for pano_type in panos_to_visualize:
                output_folder_layout = os.path.join(
                    output_folder, "layout", geometry_type, pano_type
                )
                os.makedirs(output_folder_layout, exist_ok=True)
                panos_list = zillow_floor_plan.panos_layouts[geometry_type][pano_type]
                render_room_vertices_on_panos(
                    input_folder=zillow_floor_plan.input_folder,
                    panos_list=panos_list,
                    output_folder=output_folder_layout,
                )

    # Render the top-down draft floor plan, result of the merger stage.
    if args.visualize_floor_plan:
        output_folder_floor_plan = os.path.join(output_folder, "floor_plan")
        os.makedirs(output_folder_floor_plan, exist_ok=True)

        for geometry_type in geometry_to_visualize:
            if geometry_type == "visible":
                continue

            zind_dict = zillow_floor_plan.floor_plan_layouts[geometry_type]

            for floor_id, zind_poly_list in zind_dict.items():
                output_file_name = os.path.join(
                    output_folder_floor_plan,
                    "vector_{}_layout_{}.jpg".format(geometry_type, floor_id),
                )

                render_jpg_image(
                    polygon_list=zind_poly_list, jpg_file_name=output_file_name
                )

    # Render vector geometry on top of the raster floor plan image.
    if args.visualize_raster:
        output_folder_floor_plan_alignment = os.path.join(
            output_folder, "floor_plan_raster_to_vector_alignment"
        )
        os.makedirs(output_folder_floor_plan_alignment, exist_ok=True)

        for geometry_type in geometry_to_visualize:
            if geometry_type == "visible":
                continue

            for (
                floor_id,
                raster_to_vector_transformation,
            ) in zillow_floor_plan.floor_plan_to_redraw_transformation.items():
                floor_plan_image_path = os.path.join(
                    input_folder, zillow_floor_plan.floor_plan_image_path[floor_id]
                )

                zind_poly_list = zillow_floor_plan.floor_plan_layouts[geometry_type][
                    floor_id
                ]

                output_file_name = os.path.join(
                    output_folder_floor_plan_alignment,
                    "raster_to_vector_{}_layout_{}.jpg".format(geometry_type, floor_id),
                )

                render_raster_to_vector_alignment(
                    zind_poly_list,
                    raster_to_vector_transformation,
                    floor_plan_image_path,
                    output_file_name,
                )


def main():
    parser = argparse.ArgumentParser(
        description="Visualize & validate Zillow Indoor Dataset (ZInD)"
    )

    parser.add_argument(
        "--input",
        "-i",
        help="Input JSON file (or folder with ZInD data)",
        required=True,
    )

    parser.add_argument(
        "--output",
        "-o",
        help="Output folder where rendered data will be saved to",
        required=True,
    )

    parser.add_argument(
        "--visualize-layout",
        action="store_true",
        help="Render room vertices and WDO on panoramas.",
    )
    parser.add_argument(
        "--visualize-floor-plan",
        action="store_true",
        help="Render the floor plans as top-down projections with floor plan layouts and WDO elements.",
    )

    parser.add_argument(
        "--visualize-raster",
        action="store_true",
        help="Render the vector floor plan (draft or final) on the raster floor plan image.",
    )

    parser.add_argument(
        "--max-tours", default=float("inf"), help="Max tours to process."
    )

    parser.add_argument(
        "--primary", action="store_true", help="Visualize primary panoramas."
    )
    parser.add_argument(
        "--secondary", action="store_true", help="Visualize secondary panoramas."
    )

    parser.add_argument("--raw", action="store_true", help="Visualize raw layout.")
    parser.add_argument(
        "--complete", action="store_true", help="Visualize complete layout."
    )
    parser.add_argument(
        "--visible", action="store_true", help="Visualize visible layout."
    )

    parser.add_argument(
        "--redraw", action="store_true", help="Visualize 2D redraw geometry."
    )

    parser.add_argument(
        "--debug", "-d", action="store_true", help="Set log level to DEBUG"
    )

    args = parser.parse_args()

    if args.debug:
        LOG.setLevel(logging.DEBUG)

    input = args.input

    # Useful to debug, by restricting the number of tours to process.
    max_tours_to_process = args.max_tours

    # Collect all the feasible input JSON files.
    input_files_list = [input]
    if Path(input).is_dir():
        input_files_list = sorted(Path(input).glob("**/zind_data.json"))

    num_failed = 0
    num_success = 0
    failed_tours = []
    for input_file in tqdm(input_files_list, desc="Validating ZInD data"):
        # Try loading and validating the file.
        try:
            zillow_floor_plan = FloorPlan(input_file)

            current_input_folder = os.path.join(str(Path(input_file).parent))
            current_output_folder = os.path.join(
                args.output, RENDER_FOLDER, str(Path(input_file).parent.stem)
            )
            os.makedirs(current_output_folder, exist_ok=True)

            validate_and_render(
                zillow_floor_plan,
                input_folder=current_input_folder,
                output_folder=current_output_folder,
                args=args,
            )
            num_success += 1

            if num_success >= max_tours_to_process:
                LOG.info("Max tours to process reached {}".format(num_success))
                break
        except Exception as ex:
            failed_tours.append(str(Path(input_file).parent.stem))
            num_failed += 1
            track = traceback.format_exc()
            LOG.warning("Error validating {}: {}".format(input_file, str(ex)))
            LOG.debug(track)
            continue

    if num_failed > 0:
        LOG.warning("Failed to validate: {}".format(num_failed))

        LOG.debug("Failed_tours: {}".format(failed_tours))
    else:
        LOG.info("All ZInD validated successfully")


if __name__ == "__main__":
    main()
