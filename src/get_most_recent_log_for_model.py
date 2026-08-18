import os
import json
import shutil

from argparse import ArgumentParser
from pathlib import Path

def get_model_json(method: str, logdir: str="/eagle/datascience_collab/wkwiecinski/torchtitan/logs/torchtitan.experiments.ezpz.train"):
    logdir = Path(logdir)
    log_files = [log for log in logdir.glob("*.jsonl")]
    log_files.sort(key=os.path.getmtime, reverse=True)

    for file in log_files:
        with open(file, 'r') as json_data:
            for line in json_data:
                data = json.loads(line)
                if data["level"] == "INFO":
                    msg = data["message"]
                    if f"--dataloader.dataset={method}" in msg:
                        return file

if __name__ == "__main__":
    args = ArgumentParser()
    args.add_argument("copy_location", help="The location to copy the log file to.")
    args.add_argument("method", help="The method to find in each log file.")
    args.add_argument("log_location", default="/eagled/datascience_collab/wkwiecinski/torchtitan/logs/torchtitan.experiments.ezpz.train", help="The location of the log files.")

    parsed = args.parse_args()

    json_file = get_model_json(parsed.method, parsed.log_location)

    if json_file:
        shutil.copy2(json_file, parsed.copy_location)