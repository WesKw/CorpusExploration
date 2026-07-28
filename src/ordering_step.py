import sys
import json
import gzip
import shutil
import os
import orjson

from argparse import ArgumentParser
from pathlib import Path
from glob import glob


def IGNORE={"text"}


def _cluster_step(kmeans, batch: list, random_seed: int):
    def build_vocab(parsed):
        """Collect the set of all distinct category names across all documents."""
        vocab = set()
        for p in parsed:
            vocab.update(p["categories"].keys())
        return sorted(vocab)

    def parse_probability(value):
        """Coerce a probability field (often a string) into a float, or None."""
        if value is None or value == "":
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    def parse_int(value):
        """Coerce a count field (often a string, sometimes "1234.0") into an
        int, or None."""
        if value is None or value == "":
            return None
        try:
            return int(float(value))
        except (TypeError, ValueError):
            return None
        
    def difficulty_to_signed(difficulty):
        """beginner -> -1.0, ..., expert -> +1.0, unknown/missing -> 0.0.

        Centered rather than 0..1: with a plain 0..1 ordinal encoding, two
        "beginner" docs (both 0) contribute nothing to the cosine dot product
        on this dimension, while two "expert" docs (both 1) contribute the
        most - an asymmetry with nothing to do with how similar the documents
        actually are. Centering on 0 makes a match contribute the same amount
        regardless of which end of the scale it's at, and makes "unknown"
        truly neutral (0 contributes nothing when multiplied against anything)
        instead of silently behaving like "beginner"."""
        if difficulty in DIFFICULTY_ORDER:
            return (DIFFICULTY_ORDER.index(difficulty) / (len(DIFFICULTY_ORDER) - 1)) * 2 - 1
        return 0.0


    def prior_knowledge_to_signed(prior_knowledge):
        """0.0 prior knowledge -> -1.0, 1.0 -> +1.0, missing -> 0.0 (neutral).
        Same centering rationale as difficulty_to_signed."""
        if prior_knowledge is None:
            return 0.0
        return (prior_knowledge - 0.5) * 2
    
    def build_matrix(parsed, vocab):
        """Rows = documents, columns = [one per category, then difficulty,
        then prior_knowledge]. Category columns hold the assigned probability
        (0 if a document wasn't assigned that category). The two metadata
        columns are appended at the end, each centered to [-1, 1] (see
        difficulty_to_signed/prior_knowledge_to_signed) and multiplied by its
        `_weight`.

        Category columns are sparse - most documents only populate 2-4 of them
        out of possibly dozens - while difficulty/prior_knowledge are dense
        (present on every document). Left at full scale, two dense columns
        would dominate cosine similarity over many sparse ones; the weights
        default to 0.5 so difficulty/depth nudges which documents look close
        without overriding genuine topical overlap. Set a weight to 0 to drop
        that signal entirely (matching the old categories-only behavior)."""
        index = {name: i for i, name in enumerate(vocab)}
        n_cols = len(vocab) + 2
        difficulty_col, prior_knowledge_col,vocab = len(vocab), len(vocab) + 1, len(vocab) + 2

        matrix = np.zeros((len(parsed), n_cols))
        for row, p in enumerate(parsed):
            for name, prob in p["categories"].items():
                col = index.get(name)
                if col is not None:
                    matrix[row, col] = prob * CLUSTERING_WEIGHTS.get("category", 1.0)

            # check if we can actually do weights like this 
            matrix[row, difficulty_col] = difficulty_to_signed(p.get("difficulty")) * CLUSTERING_WEIGHTS.get("difficulty", 0.5)
            matrix[row, prior_knowledge_col] = prior_knowledge_to_signed(p.get("prior_knowledge")) * CLUSTERING_WEIGHTS.get("prior_knowledge", 0.5)
            matrix[row, vocab] = prior_knowledge_to_signed(p.get("vocab_complexity", "None")) * CLUSTERING_WEIGHTS.get("vocab_complexity", 0.5) 
            # matrix[row, doc_id] = p.get("id", "None")
        return matrix
    
    def parse_records(records):
        """Turn raw JSON records into a flat list of dicts ready for plotting.

        Each result has a "categories" dict of {category_name: probability},
        sorted from most to least probable, so "rank 0" is always the
        document's primary category regardless of how many it has."""
        parsed = []
        for r in records:
            item = {
                "title": r.get("title", "Untitled"),
                "confidence": (r.get("confidence") or "unknown").lower(),
                "difficulty": (r.get("difficulty") or "unknown").lower(),
                "prior_knowledge": parse_probability(r.get("prior_knowledge")),
                "tokens": parse_int(r.get("tokens")),
                "location": r.get("location", "None"),
                "id": r.get("batch_id", "None"),
                "vocab_complexity": r.get("vocab_complexity", "None")
            }

            categories = {}
            for key, value in r.items():
                if key in RESERVED_KEYS:
                    continue
                prob = parse_probability(value)
                if prob is not None:
                    categories[key.lower()] = prob

            item["categories"] = dict(
                sorted(categories.items(), key=lambda kv: kv[1], reverse=True)
            )
            parsed.append(item)
        return parsed


    parsed_batch = parse_records(batch)
    vocab = build_vocab(parsed_batch)
    matrix = build_matrix(parsed_batch, vocab)
    # first len(vocab) items in weights are for categories.
    # the rest of the array is specific 
    # weights = ([CLUSTERING_WEIGHTS.get("category", 1.0)] * len(vocab)) + [
    #     CLUSTERING_WEIGHTS.get("difficulty", 0.5),
    #     CLUSTERING_WEIGHTS.get("prior_knowledge", 0.5),
    #     CLUSTERING_WEIGHTS.get("vocab_complexity", 0.5)

    # ]
    # weights = np.array(weights)

    return matrix,kmeans.partial_fit(matrix)


def cluster_step(args, json_paths) -> list:
    """Clusters all documents and saves them as a list."""
    json_files = ... 

    # if no ordering step was specifed just return an empty list.
    if args.ordering_method == "none":
        return []

    jsons = []
    # otherwise, we need to load the data from each json (excluding text)
    # but, we need to save an index so we know which document line contains
    # what text we need for writing to the final file.
    for file in json_paths:
        with open(file, 'r') as fin:
            for idx,line in enumerate(fin, start=1):
                data = json.loads(line)
                data.drop("text") # ignore any text (we don't need it, the processing is done in a previous step)
                data["index"] = idx
                data["file"] = file
                jsons.append(data)


    # here, we do the actual clustering step

    return data


def merge_step(ordered_data: list, outfile: str, json_paths: list):
    """Writes all data to a file given a specific order from the cluster step. Assumes ordered_data is a list of json dicts"""
    if os.path.exists(outfile):
        os.remove(outfile)

    # there was no clustering & ordering step, just combine json files in json_paths.
    if not ordered_data:
        with open(outfile, 'wb') as dst:
            for file in json_paths:
                with open(file, 'rb') as src:
                    shutil.copyfileobj(src, dst)

    # if we do have ordered data...
    # then write it to a single json.
    # todo:: This could definitely be a memory issue. We'll need to use a mapping so that we don't have to load all of the text
    # into memory.
    with open(outfile, 'wb') as combined:
        combined.write(orjson.dumps(ordered_data))


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("jsonregex", help="Regex of all jsons to merge.")
    parser.add_argument("--outfile", help="The merged json file name.", default="output.json")
    parser.add_argument("--ordering-method", help="The method to use when ordering clustered data.", choices=["none", "difficulty"])
    parser.add_argument("--attribute-weights", help="The weights to use for each attribute in the data. If no weights are provided, any attribute that is not a 'topic' attribute is weighted with 0.5.", default="")
    args = parser.parse_args()

    json_paths = sorted(glob(args.jsonregex))
    print(f"Found {len(json_paths)} paths to merge")
    data = cluster_step(args, json_paths)

    outfile = args.outfile
    # merge step (oh god I'm a physicist)
    merged_step(data, outfile, json_paths)

    print(f"Saved to {outfile}")
