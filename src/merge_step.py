import sys
import json
import gzip
import shutil
import os

from mpi4py import MPI
from argparse import ArgumentParser
from pathlib import Path
from glob import glob


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("jsonregex", help="Regex of all jsons to merge.")
    parser.add_argument("--outfile", help="The merged json file name.", default="output.txt")
    args = parser.parse_args()

    json_paths = glob(args.jsonregex)
    print(f"Found {len(json_paths)} paths to merge")
    outfile = args.outfile
    # clear any existing json data before the job starts
    if os.path.exists(outfile):
        os.remove(outfile)
    
    # merge step (oh god I'm a physicist)
    with open(outfile, 'wb') as combined:
        for path in json_paths:
            with open(path, 'rb') as part:
                shutil.copyfileobj(part, combined)

    print(f"Saved to {outfile}")
