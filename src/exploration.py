import json
import time
import gzip
import io
import os
import subprocess
import shutil
import math
import re
import multiprocessing
import concurrent
import zstandard as zstd
import random

from argparse import ArgumentParser
from pathlib import Path
from glob import glob
from multiprocessing import Process,Pool,TimeoutError
from concurrent.futures import ThreadPoolExecutor


UNIT_CHOICES = {
    "MB": math.pow(1024, 2),
    "GB": math.pow(1024, 3),
    "TB": math.pow(1024, 4),
}


dict_lock = multiprocessing.Lock()


def view_json_data(json_path: str):
    """Assuming that the jsons are not unzipped"""
    # unzip the json locally, process, then remove
    path = Path(json_path)
    new_location = os.getcwd() + f"/{path.name}"
    # print(f"Unzipping {new_location}")
    shutil.copy(json_path, os.getcwd())
    subprocess.run(["pigz", "-d", new_location], check=True)
    doc_data = {}

    # read json
    with open(new_location.replace('.gz', ''), 'r') as f:
        for line in f:
            json_data = json.loads(line)
            for key in json_data.keys():
                if key == "text":
                    print(f"{key} : [{json_data[key][:20]} ... {json_data[key][-20:]}]")
                else:
                    print(f"{key} : {json_data[key]}")

    # do some magic analysis here

    print(f"Removing {new_location.replace('.gz', '')}")
    os.remove(f"./{path.name.replace('.gz', '')}")


def process_json_file(path: str):
    # print("Worker started")
    """Threads process a json"""
    # print(path)
    # print(subset)
    
    # get the collection that the file is a part of
    collection = re.findall(r"data\/.*?\/", path)
    if collection:
        collection = collection[0].split("/")[1]

    bytes = 0
    extension = Path(path).suffix
    if extension == ".gz":
        try:
            # decompress into memory and get the full size
            with open(path, 'rb') as f:
                buf = io.BytesIO(f.read())
                bytes = 0
                with gzip.open(buf, 'rb') as f:
                    while chunk := f.read(65536):
                        bytes += len(chunk)
            print(f"Processed {path}")
            # bytes = int(result.stdout.decode("utf-8"))
        except Exception as exc:
            print(f"Error occurred while processing {path}")
            print(exc)
            bytes = 0

    elif extension == ".zstd":
        # unfortunately the zstd files don't easily include the uncompressed sizes so we need to test decompress
        # zstd_out = subprocess.run(["zstd", "-t", "--no-progress", f"{path}"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        try:
            with open(path, 'rb') as f:
                buf = io.BytesIO(f.read())
                dctx = zstd.ZstdDecompressor()
                with dctx.stream_reader(buf) as reader:
                    while chunk := reader.read(65536):
                        bytes += len(chunk)
                # print(f"Uncompressed size: {size} bytes")
                # bytes = int(zstd_out.stderr.decode("utf-8").split(" ")[1])        
           print(f"Processed {path}")
        except Exception as exc:
            print(f"Error occurred while processing {path}")
            print(exc)
            bytes = 0

    else: # probably don't process the file if it's not compressed 
        print(f"Skipped {path}")
        bytes=0
        # skipped_files += 1
        # skipped_file_names.append(json)

    # just need to lock to write to the dictionary
    # lock.acquire()
    # if collection not in collection_sizes:
    #         collection_sizes[collection] = 0
    # collection_sizes[collection] += bytes
    # lock.release()
    return (path, collection, bytes)
    # print(f"\033[K{json}\nTotal Size: {(dataset_size / UNIT_CHOICES[units]):.02f} {units}", end="\r", flush=True)

    # then here we do the unzipping and processing of the actual data in the file
    # path = Path(json_path)
    # new_location = os.getcwd() + f"/{path.name}"
    # # print(f"Unzipping {new_location}")
    # shutil.copy(json_path, os.getcwd())
    # subprocess.run(["pigz", "-d", new_location], check=True)
    # doc_data = {}

    # # read json
    # with open(new_location.replace('.gz', ''), 'r') as f:
    #     for line in f:
    #         json_data = json.loads(line)
    #         for key in json_data.keys():
    #             if key == "text":
    #                 print(f"{key} : [{json_data[key][:20]} ... {json_data[key][-20:]}]")
    #             else:
    #                 print(f"{key} : {json_data[key]}")

    # # do some magic analysis here
    # print(f"Removing {new_location.replace('.gz', '')}")
    # os.remove(f"./{path.name.replace('.gz', '')}")


def get_corpus_metadata(root: Path, units: int, subset: str, nprocs: int):
    """
    Path is the root directory of the training data
    """
    paths = get_json_paths(root)
    random.shuffle(paths) # shuffle the array to attempt to get an even distribution of work for threads
    # paths = paths
    # paths = [path for path in paths if subset != None and subset in path]

    dataset_size = 0
    skipped_file_names = set()
    collection_sizes = {}
    print(f"Total files: {len(paths)}")

    with Pool(processes=nprocs) as pool:
        results = pool.imap(process_json_file, paths, chunksize=1)
        for result in results:
            file,collection,size = result

            # if the size is 0 we skipped the file
            if size == 0:
                skipped_file_names.add(file)

            if collection not in collection_sizes:
                collection_sizes[collection] = 0
            collection_sizes[collection] += size
            dataset_size += size

            print(f"\033[K{file}\nTotal Size: {(dataset_size / UNIT_CHOICES[units]):.02f} {units}", end="\r", flush=True)

    print("\nCollection totals:")
    for collection in sorted(collection_sizes.items(), key=lambda x: x[1], reverse=True):
        print(
            f"\t{collection[0]} -> {(collection_sizes[collection[0]] / UNIT_CHOICES[units]):.02f} {units} | " \
            f"{(collection_sizes[collection[0]] / dataset_size) * 100:.02f}%"
        )

    # for json in paths:
    #     if subset and subset not in json:
    #         continue

    #     collection = re.findall(r"data\/.*?\/", json)
    #     if collection:
    #         collection = collection[0].split("/")[1]

    #     # data = view_json_data(json)
    #     bytes=0
    #     extension = Path(json).suffix
    #     if extension == ".gz":
    #         gzip_out = subprocess.run(["gzip", "-l", f"{json}"], capture_output=True)
    #         result = subprocess.run(["awk", "NR==2 {print $2}"], capture_output=True, input=gzip_out.stdout)
    #         try:
    #             bytes = int(result.stdout.decode("utf-8"))
    #         except:
    #             print(f"Error occurred while processing {json}")
    #             bytes = 0
    #             skipped_files += 1
    #             skipped_file_names.append(json)
    #     elif extension == ".zstd":
    #         # unfortunately the zstd files don't easily include the uncompressed sizes so we need to test decompress
    #         zstd_out = subprocess.run(["zstd", "-t", "--no-progress", f"{json}"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    #         try:
    #             bytes=int(zstd_out.stderr.decode("utf-8").split(" ")[1])
    #         except:
    #             print(f"Error occurred while processing {json}")
    #             bytes = 0
    #             skipped_files += 1
    #             skipped_file_names.append(json)
    #     else: # probably don't process the file if it's not compressed 
    #         print(f"Skipped {json}")
    #         bytes=0
    #         skipped_files += 1
    #         skipped_file_names.append(json)
        
    #     if collection not in collection_sizes:
    #         collection_sizes[collection] = 0
    #     collection_sizes[collection] += bytes
    #     dataset_size += bytes

    #     print(f"\033[K{json}\nTotal Size: {(dataset_size / UNIT_CHOICES[units]):.02f} {units}", end="\r", flush=True)

def get_json_paths(root: Path, subset: str):
    gz_str = str(root) + "/**/*.gz"
    gz_paths = glob(gz_str, recursive=True)
    zstd_str = str(root) + "/**/*.zstd"
    zstd_paths = glob(zstd_str, recursive=True)
    print(f"{len(gz_paths)} gz files")
    print(f"{len(zstd_paths)} zstd files")

    paths = [path for path in gz_paths + zstd_paths if subset and subset in path] 
    print(paths)

    return paths
    # return gz_paths


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("data", help="Location")
    parser.add_argument("--units", help="Unit for size output", choices=["MB", "GB", "TB"], default="GB")
    # hard-code this for now
    parser.add_argument("--subset", help="Data subset to process", choices=["algebraic-stack", "arxiv", "dclm", "open-web-math", "pes2o", "starcoder", "wiki"], default=None)
    parser.add_argument("--threads", help="Number of processes", default=1)

    args = parser.parse_args()
    get_corpus_metadata(Path(args.data), args.units, args.subset, int(args.threads))
