import sys
import json
import gzip
import shutil
import os
import orjson
import linecache
import numpy as np

from argparse import ArgumentParser
from pathlib import Path
from glob import glob
from ExplorationArgs import ClusterArgs
from simple_parsing import parse
from sklearn.cluster import MiniBatchKMeans


# Every other key on a record is treated as "<category_name>": "<probability>"
RESERVED_KEYS = {"title", "confidence", "difficulty", "prior_knowledge", "tokens", "vocab_complexity", "sentence_quality", "code_percentage", "stem_like", "batch_id", "row_index", "pid", "rank"}
_RESERVED_KEYS_NORM = {k.lower().replace(" ", "_").replace("-", "_")
                       for k in RESERVED_KEYS}
CONFIDENCE_ORDER = ["low", "medium", "high"]
DIFFICULTY_ORDER = ["beginner", "intermediate", "advanced", "expert"]
CLUSTERING_WEIGHTS = {}


ANCHOR_SUBJECTS = {
    "Physics": 1.0, "Mathematics": 1.0, "Computer Science": 1.0,
    "Biology": 1.0, "Chemistry": 1.0, "Engineering": 1.0,
    "Statistics": 1.0, "Astronomy": 1.0, "Medicine": 0.8,
    "Economics": 0.4, "Psychology": 0.5, "Linguistics": 0.3,
    "History": 0.0, "Literature": 0.0, "Philosophy": 0.1,
    "Art": 0.0, "Music": 0.0, "Law": 0.1, "Sociology": 0.2,
    "Sports": 0.1, "Business": 0.2, "Politics": 0.1, "Religion": 0.0
}


def build_vocab(parsed):
    """Collect the set of all distinct category names across all documents."""
    vocab = set()
    for p in parsed:
        vocab.update(p["categories"].keys())
    return sorted(vocab)


def convert_percentage_to_float(percentage):
    value = 0.0

    if type(percentage) == float:
        if percentage > 1.01:
            percentage = percentage / 100.0
        return percentage
    
    # sometimes the llm is stupid and gives me a percent
    if percentage.endswith("%"):
        value = float(percentage.replace('%', '')) / 100.0
    
    else:
        value = float(percentage)
        if value > 1.01:
            value = value / 100.0

    return value



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


def vocab_complexity_to_signed(vocab_complexity):
    """Same centering rationale as difficulty/prior_knowledge. Uses
    parse_probability so the string "None" (parse_records' default for
    this field) is treated the same as an actual None, rather than
    blowing up trying to subtract 0.5 from a string."""
    value = parse_probability(vocab_complexity)
    if value is None:
        return 0.0
    return (value - 0.5) * 2


def build_stem_weights(vocab, embed_fn, anchor_subjects=ANCHOR_SUBJECTS, sim_floor=0.3):
    """One-time vocab-term -> fractional STEM score mapping.

    vocab is fixed for the batch (built by build_vocab), so this only needs
    to run once per batch, not once per document.
    """
    anchor_names = list(anchor_subjects.keys())
    anchor_embs = np.array([embed_fn(name) for name in anchor_names])
    anchor_norms = np.linalg.norm(anchor_embs, axis=1)
    anchor_scores = np.array([anchor_subjects[name] for name in anchor_names])

    weights = np.zeros(len(vocab))
    for i, term in enumerate(vocab):
        emb = embed_fn(term)
        sims = anchor_embs @ emb / (anchor_norms * np.linalg.norm(emb))
        mask = sims > sim_floor
        if mask.any():
            w = sims[mask]
            w = w / w.sum()
            weights[i] = float((w * anchor_scores[mask]).sum())
    return weights


def document_stemminess(categories: dict, vocab_index: dict, stem_weights: np.ndarray) -> float:
    total = sum(categories.values())
    if total == 0:
        return 0.0
    score = sum(
        prob * stem_weights[col]
        for name, prob in categories.items()
        if (col := vocab_index.get(name)) is not None
    )
    return score / total


def parse_records(records, print_num_docs:bool=False):
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
            "vocab_complexity": r.get("vocab_complexity", "None"),
            "rank": r.get("rank"),
            "pid": r.get("pid"),
            "row_index": r.get("row_index"),
            "stem_like": float(r.get("stem_like", -1.0)),
            "code_percentage": convert_percentage_to_float(r.get("code_percentage", -1.0))
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
    
    if print_num_docs:
        print(f"Parsed {len(parsed)} documents.")
    
    return parsed


def _cluster_step(kmeans, batch: list, random_seed: int, vocab):
    def build_matrix(parsed, vocab):
        """Returns (matrix, n_feature_cols).

        Columns 0..n_feature_cols-1 are what KMeans actually sees: one per
        category (sparse), then difficulty, prior_knowledge, vocab_complexity
        (dense, each centered to [-1, 1] and weighted via CLUSTERING_WEIGHTS
        - see difficulty_to_signed/prior_knowledge_to_signed/
        vocab_complexity_to_signed for the centering rationale; weights
        default to 0.5 so these dense signals nudge similarity without
        drowning out sparse topical overlap. Set a weight to 0 to drop that
        signal entirely).

        Columns n_feature_cols.. onward (row_index, rank, pid) are pure
        identifiers, stored raw and unweighted purely so records can be
        matched back up later. They are NOT part of n_feature_cols and are
        sliced off before anything is fit - they have zero effect on
        clustering. Missing identifiers are stored as NaN rather than 0, so
        a real id of 0 isn't confused with "absent".

        NOTE: this assumes row_index/rank/pid are numeric. If pid is ever a
        non-numeric string in your data, don't put it in this float array -
        return it as a separate parallel list instead.
        """
        index = {name: i for i, name in enumerate(vocab)}
        n_vocab = len(vocab)

        difficulty_col = n_vocab
        prior_knowledge_col = n_vocab + 1
        vocab_complexity_col = n_vocab + 2
        n_feature_cols = n_vocab + 3

        row_index_col = n_feature_cols
        rank_col = n_feature_cols + 1
        pid_col = n_feature_cols + 2
        n_cols = n_feature_cols + 3
 
        matrix = np.zeros((len(parsed), n_cols))
        for row, p in enumerate(parsed):
            for name, prob in p["categories"].items():
                col = index.get(name)
                if col is not None:
                    matrix[row, col] = prob * CLUSTERING_WEIGHTS.get("category", 1.0)

            matrix[row, difficulty_col] = difficulty_to_signed(p.get("difficulty")) * CLUSTERING_WEIGHTS.get("difficulty", 0.5)
            matrix[row, prior_knowledge_col] = prior_knowledge_to_signed(p.get("prior_knowledge")) * CLUSTERING_WEIGHTS.get("prior_knowledge", 0.5)
            matrix[row, vocab_complexity_col] = vocab_complexity_to_signed(p.get("vocab_complexity")) * CLUSTERING_WEIGHTS.get("vocab_complexity", 0.5)

            matrix[row, row_index_col] = p["row_index"] if p["row_index"] is not None else np.nan
            matrix[row, rank_col] = p["rank"] if p["rank"] is not None else np.nan
            matrix[row, pid_col] = p["pid"] if p["pid"] is not None else np.nan

        return matrix, n_feature_cols, difficulty_col

    parsed_batch = parse_records(batch)
    # vocab = build_vocab(parsed_batch)
    matrix, n_feature_cols, difficulty_col = build_matrix(parsed_batch, vocab)

    # Only the feature columns drive clustering - row_index/rank/pid ride
    # along in `matrix` for later recovery but never reach the fit.
    feature_matrix = matrix[:, :n_feature_cols]

    return matrix, kmeans.partial_fit(feature_matrix), n_feature_cols, difficulty_col


def batch_data(data: list, size: int):
    for i in range(0, len(data), size):
        yield data[i:i+size]


def cluster_step(args, json_paths) -> list:
    def _order_by_difficulty(matrix, n_feature_cols, kmeans, column, reverse=False):
        """
        For each cluster, return the rows assigned to it (with identifiers),
        sorted from closest to farthest from that cluster's centroid.

        Returns: dict[int, list[dict]] mapping cluster_label -> ordered records,
        each record carrying its distance plus the recoverable identifier columns.
        """
        feature_matrix = matrix[:, :n_feature_cols]
        labels = kmeans.predict(feature_matrix)
        # column is the overall difficulty column
        centroids = kmeans.cluster_centers_
        order = np.argsort(centroids[:, column])
        if reverse: # reverse is hardest to easiest
            order = order[::-1]

        n_feature_cols = feature_matrix.shape[1]
        row_index_col = n_feature_cols
        rank_col = n_feature_cols + 1
        pid_col = n_feature_cols + 2

        result = []
        for cluster_id in order:
            member_rows = np.where(labels == cluster_id)[0]
            if len(member_rows) == 0: # no members in a cluster, we ignore
                # result[cluster_id] = []
                continue

            centroid = centroids[cluster_id]
            # Euclidean distance in feature space only - identifiers never
            # factor into distance since they were sliced off before fit/predict.
            distances = np.linalg.norm(feature_matrix[member_rows] - centroid, axis=1)

            order = np.argsort(distances)  # ascending: closest first
            ordered_rows = member_rows[order]
            ordered_distances = distances[order]

            records = []
            for row, dist in zip(ordered_rows, ordered_distances):
                records.append({
                    "distance": float(dist),
                    "row_index": None if np.isnan(matrix[row, row_index_col]) else int(matrix[row, row_index_col]),
                    "rank": None if np.isnan(matrix[row, rank_col]) else int(matrix[row, rank_col]),
                    "pid": None if np.isnan(matrix[row, pid_col]) else int(matrix[row, pid_col]),
                })
            result.extend(records)
            # result[cluster_id] = records
        return result
    
    def _order_by_stemminess(parsed_records, reverse=False):
        """Use cosine similarity to orders documents by how close it is to a 'STEM' topic."""
        # parsed_batch = parse_records(batch)
        # matrix, n_feature_cols, difficulty_col = build_matrix(parsed_batch, vocab)

        # stem_weights = build_stem_weights(vocab, embed_fn)  # cache this if vocab is stable across batches
        # stem_scores = np.array([
        #     document_stemminess(p["categories"], index, stem_weights)
        #     for p in parsed_records
        # ])

        # Order documents by STEMminess, independent of clustering
        # stem_scores = -stem_scores if reverse else stem_scores
        # order = np.argsort(stem_scores)
        # ranked_docs = [parsed_records[i] for i in order]
        # LLM includes a "stem_like" field so we probably don't even need to do anything fancy...
        ranked_docs = sorted(parsed_records, key=lambda x: x["stem_like"])
        if reverse:
            ranked_docs = ranked_docs[::-1]
        return ranked_docs

    def _order_by_code_percentage(data, reverse=False):
        records = sorted(data, key=lambda x: x["code_percentage"])
        if reverse:
            records = records[::-1]
        return records

    def _none():
        return

    """Clusters all documents and saves them as a list."""
    jsons = []
    # load the data from each json (excluding text)
    # but, we need to save an index so we know which document line contains
    # what text we need for writing to the final file.
    for file in json_paths:
        with open(file, 'r') as fin:
            for idx,line in enumerate(fin, start=1):
                data = json.loads(line)
                jsons.append(data)

    # this is inefficient but fine for now
    parsed = parse_records(jsons, print_num_docs=True)
    vocab = build_vocab(parsed)

    # if no ordering step was specifed just return the loaded jsons
    if args.method == "none" or args.method == "shuffled":
        return jsons

    data=[]
    method = args.method.replace("-reverse", "")
    # then once we have clusters we need to order
    if method == "difficulty":
        matrix = np.array([])
        # here, we do the actual clustering step
        kmeans = MiniBatchKMeans(n_clusters=args.n_clusters, random_state=args.seed, batch_size=args.batch_size, n_init="auto")
        n_features = 0
        diff_col = 0
        num_processed = 0

        for batch in batch_data(jsons, args.batch_size):
            batch_matrix,kmeans_step,n_feature_cols,difficulty_col=_cluster_step(kmeans, batch, args.seed, vocab)
            kmeans = kmeans_step
            diff_col = difficulty_col
            n_features = n_feature_cols

            if not matrix.any():
                matrix = batch_matrix
            else:
                matrix = np.vstack((matrix, batch_matrix))

            num_processed += len(batch)
            print(f"{num_processed} documents processed")

        data = _order_by_difficulty(matrix, n_features, kmeans, difficulty_col, args.reverse)

    elif method == "stem":
        data = _order_by_stemminess(parsed, args.reverse)

    elif method == "code": # todo:: order by percentage of code
        data = _order_by_code_percentage(parsed, args.reverse)

    else:
        raise Exception(f"Unsupported ordering: {args.ordering_method}")

    return data # return final ordered data.


def merge_step(ordered_data: list, outfile: str, text_attribute_json_paths: list, shard_data_paths: list, args):
    """Writes all data to a file given a specific order from the cluster step. Assumes ordered_data is a list of json dicts"""
    if os.path.exists(outfile):
        os.remove(outfile)

    jsons_location = str(Path(text_attribute_json_paths[0]).parents[0])
    shards_location = str(Path(shard_data_paths[0]).parents[0])

    with open(outfile, 'wb') as out:
        # if we do have ordered data... pull it from the specified json based on metadata then write to the final output file.
        print(f"Ordered {len(ordered_data)} documents. Merging...")
        processed = 0
        skipped = 0
        for obj in ordered_data:
            try:
                rank = int(obj["rank"])
                pid = int(obj["pid"])
                idx = int(obj["row_index"])
                line = json.loads(linecache.getline(f"{shards_location}/merged.out_rank{rank}_dump.json_pid{pid}.json", idx))
                text_only = {"text": line["text"]}
                out.write(orjson.dumps(text_only, option=orjson.OPT_APPEND_NEWLINE))
                processed+=1
                if processed % args.batch_size == 0:
                    print(f"Processed {processed} of {len(ordered_data)}")
                    if "code_percentage" in obj: print(obj["title"], obj["code_percentage"])
            except:
                print(f"Skipped {obj['title']}")
                skipped += 1

        print(f"Processed {processed} / {processed+skipped} docs")


if __name__ == "__main__":
    args = parse(ClusterArgs)

    if args.method.endswith("-reverse"):
        args.reverse = True
        args.method.replace("-reverse", "")

    # gather json paths
    text_attribute_json_paths = glob(args.llm_data_regex)
    shard_data_paths = glob(args.shard_data_regex)
    if args.sort:
        text_attribute_json_paths = sorted(text_attribute_json_paths)
        shard_data_paths = sorted(shard_data_paths)

    try:
        with open(args.weights_json, 'r') as weights:
            CLUSTERING_WEIGHTS = json.load(weights)
    except:
        print(f"Could not locate weights json {args.weights_json}, using default weights.")

    # call cluster step
    print(f"Found {len(text_attribute_json_paths)} paths to merge")
    print(f"Found {len(shard_data_paths)} shards for merging")
    data = cluster_step(args, text_attribute_json_paths)

    # merge results
    outfile = args.outfile
    # merge step (oh god I'm a physicist)
    merge_step(data, outfile, text_attribute_json_paths, shard_data_paths, args)

    print(f"Saved to {outfile}")
