import json
import time

from argparse import ArgumentParser
from pathlib import Path
from glob import glob


def spit_json(str_path: str):
    path = Path(str_path)
    if path.suffix != ".json":
        return
    
    with open(path) as file:
        for line in file.readlines():
            data = json.loads(line)
            for key,value in data.items():            
                print(key, ":", value)
                

def get_corpus_metadata(path: Path):
    """
    Path is the root directory of the training data
    """
    ...


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("data", help="Location")

    args = parser.parse_args()

    # spit_json(args.data)
    get_corpus_metadata(Path(args.data))


