import json
import time
import os
import subprocess
import shutil
import math
import re

from argparse import ArgumentParser
from pathlib import Path
from glob import glob


UNIT_CHOICES = {
    "MB": math.pow(1024, 2),
    "GB": math.pow(1024, 3),
    "TB": math.pow(1024, 4),
}


def view_json_data(json_path: str):
    """Assuming that the jsons are not unzipped"""
    # unzip the json locally, process, then remove
    path = Path(json_path)
    new_location = os.getcwd() + f"/{path.name}"
    print(f"Unzipping {new_location}")
    shutil.copy(json_path, os.getcwd())
    subprocess.run(["gzip", "-d", new_location], check=True)

    # read json
    with open(new_location.replace('.gz', ''), 'r') as f:
        for line in f.readlines():
            # Process each line of the JSON file
            json_data = json.loads(line)

    # do some magic here

    print(f"Removing {new_location.replace('.gz', '')}")
    os.remove(f"./{path.name.replace('.gz', '')}")
    time.sleep(3)


def get_corpus_metadata(root: Path, units: int):
    """
    Path is the root directory of the training data
    """
    paths = get_json_paths(root)
    dataset_size = 0
    collection_sizes = {}

    for json in paths:
        collection = re.findall(r"data\/.*?\/", json)
        if collection:
            collection = collection[0].split("/")[1]

        data = view_json_data(json)

        gzip_out = subprocess.run(["gzip", "-l", f"{json}"], capture_output=True)
        result = subprocess.run(["awk", "NR==2 {print $2}"], capture_output=True, input=gzip_out.stdout)
        bytes = int(result.stdout.decode("utf-8"))
        
        if collection not in collection_sizes:
            collection_sizes[collection] = 0
        collection_sizes[collection] += bytes
        dataset_size += bytes

        print(f"\033[K{json}\nTotal Size: {(dataset_size / UNIT_CHOICES[units]):.02f} {units}", end="\r", flush=True)

    print("\nCollection totals:")
    for collection in sorted(collection_sizes.items(), key=lambda x: x[1], reverse=True):
        print(
            f"\t{collection[0]} -> {(collection_sizes[collection[0]] / UNIT_CHOICES[units]):.02f} {units} | " \
            f"{(collection_sizes[collection[0]] / dataset_size) * 100:.02f}%"
        )


def get_json_paths(root: Path):
    glob_str = str(root) + "/**/*.json.gz"
    paths = glob(glob_str, recursive=True)
    return paths


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("data", help="Location")
    parser.add_argument("--units", help="Unit for size output", choices=["MB", "GB", "TB"], default="GB")

    args = parser.parse_args()
    get_corpus_metadata(Path(args.data), args.units)
