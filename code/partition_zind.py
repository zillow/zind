# """ Script to partition ZInD to train/val/test (0.8 : 0.1 : 0.1). 
# The train/val/test splits will have similar distributions under the following metrics:
# 1. Layout complexities (cuboid, L-shape, etc.)
# 2. Number of floors
# 3. Number of primary panoramas
# 4. Number of secondary panoramas
# 5. Total area (to ensure that we have good balance between small/large homes)

# Example usage:
# python partition_zind.py -i <input_folder> -o <output_folder>
#

import argparse
import json
import os
import random

import numpy as np
from tqdm import tqdm
from shapely.geometry import Polygon
import cv2


LAYOUT_TYPES_DICT = {'CUBOID': 0,
                     'MANHATTAN_L': 1,
                     'MANHATTAN_GENERAL': 2,
                     'NON_MANHATTAN': 3}


def angle_between_vectors(vector1, vector2):
    """Return the counterclockwise angle between vector1 and vector2."""

    unit_v1 = vector1 / np.linalg.norm(vector1)
    unit_v2 = vector2 / np.linalg.norm(vector2)
    rotation_angle = np.arccos(np.clip(np.dot(unit_v1, unit_v2), -1, 1))

    return rotation_angle


def get_angle_distribution(room_vertices):
    """Get the distribution of angles for a room shape."""

    num_vertices = room_vertices.shape[0]
    angles = []
    for segment_idx in range(num_vertices):
        v1 = room_vertices[(segment_idx+1)%num_vertices,:] - room_vertices[segment_idx%num_vertices,:]
        v2 = room_vertices[(segment_idx-1)%num_vertices,:] - room_vertices[segment_idx%num_vertices,:]
        angle = np.rad2deg(angle_between_vectors(v1, v2))
        angles.append(angle)

    return angles


def get_layout_type(room_vertices, threshold_degrees=10):
    """
    Get layout type: CUBOID; MANHATTAN_L; MANHATTAN_GENERAL; NON_MANHATTAN.

    :param room_vertices: Coordinates of room vertices.
    :param threshold_degrees: Threshold to determine if a wall angle is NON_MANHATTAN.

    :return: Room layout type.
    """

    num_vertices = room_vertices.shape[0]
    angles = get_angle_distribution(room_vertices)

    angles_mod = np.mod(angles, 90)
    deviation_from_manhattan = np.minimum(angles_mod,
                                          90 - angles_mod)
    non_manhattan_angles = deviation_from_manhattan > threshold_degrees
    if np.any(non_manhattan_angles):
        layout_type = 'NON_MANHATTAN'
    elif num_vertices == 4:
        layout_type = 'CUBOID'
    elif num_vertices == 6:
        layout_type = 'MANHATTAN_L'
    else:
        layout_type = 'MANHATTAN_GENERAL'

    return layout_type


def collect_stats(input_folder, tour_ids_split):
    """Collect stats for input home tours.

    :param input_folder: The folder contains the json files for home tours.
    :param tour_ids_split: Tour ids for train, val, or test.

    :return: Home tours stats.
    """
    num_floors_list = []
    total_sqm_list = []
    num_primary_pano_list = []
    num_secondary_pano_list = []
    layout_type_list = []

    for tour_id in tqdm(tour_ids_split):
        tour_json_path = os.path.join(input_folder, tour_id, "zind_data.json")
        with open(tour_json_path, "r") as fh:
            zillow_json_dict = json.load(fh)

        num_floors = 0
        total_sqm = 0
        num_primary_panos = 0
        num_secondary_panos = 0
        invalid_floor_scale = False
        if "merger" in zillow_json_dict:
            for floor_id, floor_data in zillow_json_dict["merger"].items():
                num_floors += 1
                scale = zillow_json_dict["scale_meters_per_coordinate"][floor_id]
                if scale is None:
                    invalid_floor_scale = True
                for complete_room_data in floor_data.values():
                    for partial_room_data in complete_room_data.values():
                        for pano_data in partial_room_data.values():
                            if pano_data["is_primary"]:
                                num_primary_panos += 1
                                layout_type = get_layout_type(np.array(pano_data["layout_raw"]["vertices"]))
                                layout_type_list.append(LAYOUT_TYPES_DICT[layout_type])
                                if scale is not None:
                                    room_scale = pano_data["floor_plan_transformation"]["scale"]
                                    room_vertices_scaled = [[vertex[0] * room_scale * scale, vertex[1] * room_scale * scale] for vertex in pano_data["layout_raw"]["vertices"]]
                                    room_polygon = Polygon(room_vertices_scaled)
                                    total_sqm += room_polygon.area
                            else:
                                num_secondary_panos += 1
            num_floors_list.append(num_floors)
            if not invalid_floor_scale:
                total_sqm_list.append(total_sqm)
            num_primary_pano_list.append(num_primary_panos)
            num_secondary_pano_list.append(num_secondary_panos)
    
    return [num_floors_list, total_sqm_list, num_primary_pano_list, num_secondary_pano_list, layout_type_list]


def main():
    parser = argparse.ArgumentParser(description="Partition Zillow Indoor Dataset (ZInD)")

    parser.add_argument(
        "--input", "-i", help="Input folder contains all the home tours.", required=True
    )
    parser.add_argument(
        "--output", "-o", help="Output folder where zind_partition.json will be saved to", required=True
    )

    args = parser.parse_args()
    input_folder = args.input
    output_folder = args.output

    os.makedirs(output_folder, exist_ok=True)
    tour_ids =[tour_id for tour_id in os.listdir(input_folder) if os.path.isdir(os.path.join(input_folder, tour_id))]
    num_tours = len(tour_ids)
    train_ratio = 0.8
    val_ratio = 0.1
    test_ratio = 0.1
    assert train_ratio + val_ratio + test_ratio == 1
    num_tours_train = int(num_tours * train_ratio)
    num_tours_val = int(num_tours * val_ratio)
    best_hist_score = 0
    hist_score_list = []
    bins_list = [4, 10, 10, 10, 4]
    for _ in range(50):
        random.shuffle(tour_ids)
        tour_ids_train = tour_ids[:num_tours_train]
        tour_ids_val = tour_ids[num_tours_train:num_tours_train + num_tours_val]
        tour_ids_test = tour_ids[num_tours_train + num_tours_val:]

        stats_list_train = collect_stats(input_folder, tour_ids_train)
        stats_list_val = collect_stats(input_folder, tour_ids_val)
        stats_list_test = collect_stats(input_folder, tour_ids_test)
        hist_score = 0

        for stats_train, stats_val, stats_test, bins in zip(stats_list_train, stats_list_val, stats_list_test, bins_list):
            max_val = max(max(stats_train), max(stats_val), max(stats_test))
            min_val = min(min(stats_train), min(stats_val), min(stats_test))
            hist_train, _ = np.histogram(stats_train, bins=bins, range=(min_val, max_val), density=True)
            hist_val, _ = np.histogram(stats_val, bins=bins, range=(min_val, max_val), density=True)
            hist_test, _ = np.histogram(stats_test, bins=bins, range=(min_val, max_val), density=True)
            hist_score_train_val = cv2.compareHist(hist_train.astype(np.float32), hist_val.astype(np.float32), method=0)
            hist_score_train_test = cv2.compareHist(hist_train.astype(np.float32), hist_test.astype(np.float32), method=0)
            hist_score_val_test = cv2.compareHist(hist_val.astype(np.float32), hist_test.astype(np.float32), method=0)
            hist_score_train_val_test = (hist_score_train_val + hist_score_train_test + hist_score_val_test) / 3
            hist_score += hist_score_train_val_test
        hist_score_list.append(hist_score)
        if hist_score > best_hist_score:
            best_hist_score = hist_score
            best_tour_ids_train = tour_ids_train
            best_tour_ids_val = tour_ids_val
            best_tour_ids_test = tour_ids_test

    # save zind partition
    zind_partition = {
        "train": best_tour_ids_train,
        "val": best_tour_ids_val,
        "test": best_tour_ids_test,
    }
    with open(os.path.join(args.output, "zind_partition.json"), "w") as fh:
        json.dump(zind_partition, fh)


if __name__ == "__main__":
    main()
