import argparse
import hashlib
import json
import logging
import os
import sys
import time
from multiprocessing import cpu_count
from multiprocessing.pool import ThreadPool
from typing import Dict, List

import requests
from tqdm import tqdm as progress_bar

LOG_FILE_NAME = "zind.log"
BRIDGE_API_URL = (
    "https://api.bridgedataoutput.com/api/v2/OData/zgindoor/Indoor/replication"
)
MAX_NUM_RETRIES = 3
JSON_REQUESTS_TIMEOUT = 120  # 120 seconds, will double after every retry
IMAGE_REQUESTS_TIMEOUT = 60  # 60 seconds, will double after every retry
DOWNLOAD_STATUS_FILENAME = "download_status.json"


# TODO Update size based on entire dataset, store ZInD version for the stats

logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def write_dict_to_json(value: Dict, dest_path: str):
    """
    Writes the value to a json
    :param value: Dict to write
    :param dest_path: Path to write json
    """
    with open(dest_path, "w") as f:
        json.dump(value, f)


def create_folder(path: str):
    """
    Creates folder if it does not exist
    :param path: folder path
    """
    if not os.path.isdir(path):
        os.mkdir(path)


def calculate_checksum(dest_path: str):
    """
    Calculate image checksum.
    :param dest_path: Path to the image.
    :return: MD5 checksum
    """
    with open(dest_path, "rb") as file_to_check:
        img_data = file_to_check.read()
        md5_returned = hashlib.md5(img_data).hexdigest()
    return md5_returned


def keep_required_keys(local_house_dict: Dict) -> Dict:
    """
    Only retains the required keys in the house details dict stored on disk
    :param local_house_dict: House details
    :return: Required House details
    """
    keys_to_keep = [
        "merger",
        "redraw",
        "scale_meters_per_coordinate",
        "floorplan_to_redraw_transformation",
    ]
    res = {}
    for key, value in local_house_dict.items():
        if key in keys_to_keep:
            res[key] = value
    return res


def run_imap_unordered_multiprocessing(func, argument_list, num_processes) -> List:
    """
    Runs functions in parallel using imap_unordered multiprocessing
    :param func: Function to run
    :param argument_list: List of arguments to function
    :param num_processes: Number of processes to be used
    :return: result of the function
    """
    pool = ThreadPool(processes=num_processes)

    result_list_tqdm = []
    for result in progress_bar(
        pool.imap_unordered(func=func, iterable=argument_list), total=len(argument_list)
    ):
        result_list_tqdm.append(result)

    return result_list_tqdm


def process_house(details: Dict, output_folder: str) -> List:
    """
    Create required folder structure to store images for the house
    Store details about the house in zind_data.json
    :param details: house details dict from ZInD response json
    :param output_folder: Folder to download images to
    :return: [(url,dest_path,checksum)] List of images to download for house
    """
    house_number = details["home_id"]
    house_path = os.path.join(output_folder, house_number)
    panos_path = os.path.join(house_path, "panos")
    floor_plan_path = os.path.join(house_path, "floor_plans")
    local_house_details = dict(details)
    create_folder(house_path)
    create_folder(panos_path)
    create_folder(floor_plan_path)

    all_images_to_download = []

    for floor_name, floor_details in details["merger"].items():
        for complete_room_name, complete_room_details in floor_details.items():
            for (
                partial_room_name,
                partial_room_details,
            ) in complete_room_details.items():
                for pano_name, pano_details in partial_room_details.items():
                    pano_url = pano_details["image_path"]
                    pano_dest_name = f"{floor_name}_{partial_room_name}_{pano_name}.jpg"
                    pano_dest_path = os.path.join(panos_path, pano_dest_name)
                    pano_checksum = pano_details["checksum"]
                    all_images_to_download.append(
                        (pano_url, pano_dest_path, pano_checksum)
                    )
                    local_house_details["merger"][floor_name][complete_room_name][
                        partial_room_name
                    ][pano_name]["image_path"] = f"panos/{pano_dest_name}"

    for floor_name, floor_details in details[
        "floorplan_to_redraw_transformation"
    ].items():
        image_url = floor_details["image_path"]
        floor_plan_relative_path = os.path.join("floor_plans", f"{floor_name}.png")
        local_house_details["floorplan_to_redraw_transformation"][floor_name][
            "image_path"
        ] = floor_plan_relative_path
        floor_plan_path = os.path.join(house_path, floor_plan_relative_path)
        floor_plan_checksum = floor_details["checksum"]
        all_images_to_download.append([image_url, floor_plan_path, floor_plan_checksum])

    house_details_path = os.path.join(house_path, "zind_data.json")
    local_house_details = keep_required_keys(local_house_details)
    write_dict_to_json(local_house_details, house_details_path)

    return all_images_to_download


def download_json_in_chunks(zind_url, headers, payload, dest_path):
    """Helper function to download the large ZInD json in chunks.
    :param zind_url: The Bridge APU url
    :param headers: The requests headers including authentication
    :param payload: Any requests payload fields
    :param dest_path: Where the json will be saved to (for future use)
    :return: The parses ZInD json as python dict
    """
    response = requests.get(
        zind_url,
        stream=True,
        headers=headers,
        data=payload,
        timeout=JSON_REQUESTS_TIMEOUT,
    )
    total_size_in_bytes = int(response.headers.get("content-length", 0))
    block_size = 1024  # 1 Kb
    all_chunks = []
    with open(dest_path, "wb") as file:
        with progress_bar(
            total=total_size_in_bytes,
            unit="iB",
            unit_scale=True,
            desc="Downloading ZInD json",
        ) as pbar:
            for data in response.iter_content(block_size):
                pbar.update(len(data))
                file.write(data)
                all_chunks.append(data)

    full_content = b"".join(all_chunks)
    logger.info(f"Start loading ZInD json")
    result_dict = json.loads(full_content)
    logger.info(f"Done loading ZInD json")

    return result_dict


def get_zind_json(server_token, output_folder) -> Dict:
    """
    Returns the dict for the ZInD json.
    Sends a request to the BridgeAPI to get details about the ZInD Dataset
    Stores the respose json file in output folder
    :param server_token: token for access to the API
    :param output_folder: path to store response
    :return: ZInD Dict
    """
    dest_path = os.path.join(output_folder, "zind_response.json")
    result_dict = {}
    value_key = "value"
    if os.path.exists(dest_path):
        logger.info(f"Loading ZInD json from {dest_path}")
        try:
            result_dict = json.load(open(dest_path))
            logger.info("Loaded ZInD json successfully")
        except Exception as e:
            logger.info(f"ZInD json invalid, re-downloading file: {e}")

    zind_url = BRIDGE_API_URL

    bearer_token = f"Bearer {server_token}"
    payload = {}
    headers = {"Authorization": bearer_token}

    for retry_count in (1, MAX_NUM_RETRIES + 1):
        if value_key in result_dict:
            break

        logger.info(
            f"Retrieving ZInD json (attempt {retry_count} out of {MAX_NUM_RETRIES})"
        )
        result_dict = download_json_in_chunks(zind_url, headers, payload, dest_path)
        logger.info("Downloaded ZInD json successfully")
    else:
        logger.error(
            "Could not download ZInD json, please check your credentials and internet connection"
        )
        return None

    return result_dict[value_key]


def download_image(url_dest_path) -> bool:
    """
    Downloads file from the url and stores it at the dest path
    :param url_dest_path: [url,dest_path,checksum]
    :return: Returns True is file downloaded successfully
    """
    # Check if file exists is outside the function as we don't want to start another process
    # if file already exists

    url, dest_path, checksum = url_dest_path
    md5_returned = None

    for retry_count in range(1, MAX_NUM_RETRIES + 1):
        try:
            response = requests.get(
                url, stream=True, timeout=retry_count * IMAGE_REQUESTS_TIMEOUT
            )

            if response.status_code == requests.codes.ok:
                with open(dest_path, "wb") as f:
                    for data in response:
                        f.write(data)
                md5_returned = calculate_checksum(dest_path)
        except Exception as e:
            logger.debug(
                f"Exception raised when downloading image {url} to {dest_path}: {str(e)}"
            )
        else:
            if response.status_code == requests.codes.ok and md5_returned == checksum:
                break
            else:
                logger.debug(
                    f"Verification failed when downloading image {url} to {dest_path}: {response}"
                )

        logger.debug(f"Retry {retry_count} downloading image {url} to {dest_path}")
        time.sleep(retry_count * 30)
    else:
        logger.debug(
            f"Failed to download image {url} to {dest_path} after {MAX_NUM_RETRIES} attempts."
        )

    if not os.path.exists(dest_path):
        logger.error(f"Failed to download image {url} to {dest_path}")
        return False

    if md5_returned != checksum:
        logger.error(
            f"Checksum validation failed for {url} to {dest_path}: {md5_returned} vs {checksum}"
        )
        return False

    if retry_count > 1:
        logger.debug(
            f"Successfully downloaded image {url} to {dest_path} after {retry_count - 1} attempt(s)."
        )

    logger.debug(f"Successfully downloaded & verified image from: {url} to {dest_path}")
    return True


def create_folder_structure_and_image_download_list(
    houses: List, output_folder: str, partial_download_percentage: float
) -> List:
    """
    Create folders and zind_data files for all the houses in the output folder.
    Store on disk and return list of images to download
    :param houses: List of Houses returned in the ZInD Dict
    :param output_folder: Path to store dataset
    :param partial_download_percentage: Record percentage downloaded to ensure complete dataset is downloaded
                                        If this value is 100% entire dataset is to be downloaded
    :return: List[[url, dest_path]] List of all images to download for entire dataset
    """
    download_status_path = os.path.join(output_folder, DOWNLOAD_STATUS_FILENAME)
    download_percentage_key = "partial_download_percentage"
    files_list_key = "files_list"

    if os.path.exists(download_status_path):
        logger.info("Folder structure already exists, loading file list")
        download_status = json.load(open(download_status_path))
        if download_status[download_percentage_key] == partial_download_percentage:
            return download_status[files_list_key]

    all_images_to_download = []
    with progress_bar(
        total=len(houses), desc="Preparing the ZInD folder structure"
    ) as pbar:
        for house_details in houses:
            images_to_download = process_house(house_details, output_folder)
            all_images_to_download = all_images_to_download + images_to_download
            pbar.update(1)

    download_status = {
        files_list_key: all_images_to_download,
        download_percentage_key: partial_download_percentage,
    }
    logger.info("Required ZInD folder structure has been created")

    write_dict_to_json(download_status, download_status_path)
    return all_images_to_download


def check_files_left_to_download(image_list: List) -> List:
    """
    Go through entire list of images in dataset and check which ones have already been downloaded.
    Ensure they have correct md5 checksums, and if not we will re-try to download those.
    :param image_list: List of images in dataset
    :return: List of images left to download
    """
    images_left_to_download = []
    with progress_bar(total=len(image_list), desc="Verifying downloaded data") as pbar:
        for image_url, dest_path, checksum in image_list:
            if not os.path.exists(dest_path):
                images_left_to_download.append((image_url, dest_path, checksum))
            else:
                md5_returned = calculate_checksum(dest_path)
                if md5_returned != checksum:
                    images_left_to_download.append((image_url, dest_path, checksum))
                    logger.warning(f"Image is invalid, re-downloading {dest_path}")

            pbar.update(1)

    return images_left_to_download


def download_all(
    output_folder: str,
    server_token: str,
    num_process: int,
    partial_download_percentage=100.0,
):
    """
    Download all images in the ZInD dataset
    :param output_folder: Folder to store dataset
    :param server_token: Token to access API
    :param num_process: Num of process to use while downloading in parallel
    :param partial_download_percentage: Percentage of houses to download

    """
    houses = get_zind_json(server_token, output_folder)
    if houses is None:
        return

    houses = partial_download(partial_download_percentage, houses)

    all_images_to_download = create_folder_structure_and_image_download_list(
        houses, output_folder, partial_download_percentage
    )
    total_num_images = len(all_images_to_download)
    images_remaining = check_files_left_to_download(all_images_to_download)

    successful_downloads = 0
    if len(images_remaining) > 0:
        if len(images_remaining) != total_num_images:
            logger.info(
                "{:.2f}% images left, now resuming to download the rest of the images".format(
                    len(images_remaining) * 100 / total_num_images
                )
            )

        logger.info(f"Images to download: {len(images_remaining)}/{total_num_images}")

        results = run_imap_unordered_multiprocessing(
            download_image, images_remaining, num_process
        )

        for r in results:
            if r:
                successful_downloads += 1

    if successful_downloads == len(images_remaining):
        logger.info(f"All {total_num_images} images downloaded & verified successfully")
    else:
        # This situation should be rare and users are advised to retry.
        logger.info(
            f"There were issues with downloading the data, {len(all_images_to_download) - successful_downloads} images"
            f" left to download. Please, retry running the download script."
        )


def partial_download(percentage_to_download: float, houses: List) -> List:
    total_num_houses = len(houses)
    num_houses_to_download = max(
        1, int((percentage_to_download / 100) * total_num_houses)
    )
    return houses[:num_houses_to_download]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--server_token", "-s", help="Server token for ZInD Access", required=True
    )
    parser.add_argument(
        "--output_folder",
        "-o",
        help="Output folder to store downloaded images",
        required=True,
    )
    parser.add_argument(
        "--num_process",
        "-n",
        help="Number of process to use, by default uses maximum available",
        type=int,
        default=cpu_count(),
    )
    parser.add_argument(
        "-v", "--verbose", help="Show debug information", action="store_true"
    )
    parser.add_argument(
        "--partial_download_percentage",
        "-p",
        help="Percentage of houses to download from dataset",
        type=float,
        default=100.0,
    )

    args = parser.parse_args()

    if args.verbose:
        logger.setLevel(logging.DEBUG)

    logger_path = os.path.join(args.output_folder, LOG_FILE_NAME)
    create_folder(args.output_folder)

    # Create file handler which always logs debug messages.
    fh = logging.FileHandler(logger_path)
    fh.setLevel(logging.DEBUG)
    logger.addHandler(fh)

    download_all(
        args.output_folder,
        args.server_token,
        args.num_process,
        args.partial_download_percentage,
    )


if __name__ == "__main__":
    main()
