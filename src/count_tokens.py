#!/usr/bin/env python3
"""
Count the number of tokens in a corpus using the Gemma-7B tokenizer.

Two modes:

1) Direct mode (default) - tokenize the text found directly in each input file.
    python count_tokens.py --input /path/to/corpus --workers 8
    python count_tokens.py --input /path/to/corpus --glob 'shard-*.jsonl' --workers 8

   --input can be a single file or a directory (recursively scans
   .txt/.json/.jsonl by default, or use --glob for a custom pattern).
   .json and .jsonl are treated the same: each file is auto-detected as
   either line-delimited JSON (one object per line) or a single JSON
   document (a list of objects, or one object). Set --json-field to the
   key holding the text (default: "text").

2) Pointer mode - the input files are *index* files, each record pointing
   to a document that lives in a separate data file:
    python count_tokens.py --pointer-mode --input /path/to/index_dir \\
        --data-dir /path/to/data_dir --workers 8

   Each index record is a JSON object with "idx", "rank", "pid" keys.
   The data file holding the actual document is named:
       merged.out_rank<rank>_dump.json_pid<pid>.json
   and is itself line-delimited JSON, with "idx" giving the row/line number
   of the document within that file (1-indexed by default; pass
   --index-base 0 if your idx values are 0-indexed instead). Set --doc-field
   for the key holding the document text within the data file (default: "text").
   Index files are matched the same way as direct mode (--glob, or the
   default .txt/.json/.jsonl scan). Requests are grouped by target data
   file so each data file is streamed through exactly once, regardless of
   how many rows are needed from it.

Notes:
  - google/gemma-7b is a gated model on the Hugging Face Hub. You need to:
      1) accept the license at https://huggingface.co/google/gemma-7b
      2) `huggingface-cli login` (or set HF_TOKEN env var) before running
    Or pass --model-path pointing at an already-downloaded local copy.
  - Uses multiprocessing so it scales across cores on a login/compute node.
"""

import argparse
import json
import multiprocessing as mp
import os
import sys
from pathlib import Path

# Path or HF repo id for the tokenizer. Override with --model-path if you
# already have gemma-7b downloaded locally (e.g. on Eagle) to avoid any
# network/auth dependency on compute nodes.
TOKENIZER_NAME = "google/gemma-7b"

# Chunk size (in characters) for streaming large .txt files so we never hold
# a whole file in memory. Chosen to comfortably fit typical tokenizer max
# input handling; token counts near chunk boundaries can be off by at most
# a token or two per boundary, which is negligible at corpus scale.
CHUNK_CHARS = 1_000_000

# Filename template for pointer-mode data files.
DATA_FILENAME_TEMPLATE = "merged.out_rank{rank}_dump.json_pid{pid}.json"

# Loaded once per worker process (not per call) via initializer.
_tokenizer = None
_model_path = None
_data_dir = None
_doc_field = None


def _init_worker(model_path, data_dir=None, doc_field=None):
    global _tokenizer, _model_path, _data_dir, _doc_field
    from transformers import AutoTokenizer

    _model_path = model_path
    _tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=Path(model_path).exists())
    _data_dir = Path(data_dir) if data_dir else None
    _doc_field = doc_field


def _count_text_streaming(f):
    """Tokenize a file object in fixed-size character chunks, streaming."""
    global _tokenizer
    n_tokens = 0
    while True:
        chunk = f.read(CHUNK_CHARS)
        if not chunk:
            break
        n_tokens += len(_tokenizer.encode(chunk, add_special_tokens=False))
    return n_tokens


def _iter_json_records(path, error_counter):
    """
    Yield raw dict objects from a .json or .jsonl file, auto-detecting the
    same two layouts as _iter_json_texts (line-delimited, or a single
    array/object document), but without extracting any particular field.
    """
    with open(path, "r", encoding="utf-8") as f:
        first_line = f.readline()
        stripped = first_line.strip()
        first_obj = None
        looks_like_jsonl = False
        if stripped:
            try:
                first_obj = json.loads(stripped)
                looks_like_jsonl = isinstance(first_obj, dict)
            except json.JSONDecodeError:
                looks_like_jsonl = False

        if looks_like_jsonl:
            if isinstance(first_obj, dict):
                yield first_obj
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    error_counter[0] += 1
                    continue
                if isinstance(obj, dict):
                    yield obj
        else:
            f.seek(0)
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                error_counter[0] += 1
                return
            if isinstance(data, list):
                for obj in data:
                    if isinstance(obj, dict):
                        yield obj
            elif isinstance(data, dict):
                yield data


def _iter_json_texts(path, json_field, error_counter):
    """
    Yield text strings from a .json or .jsonl file, regardless of extension.

    Handles two layouts:
      - line-delimited JSON (one object per line)
      - a single JSON document that is either a list of objects or one object

    error_counter is a 1-element list used as a mutable int to report
    malformed lines/documents back to the caller.
    """
    with open(path, "r", encoding="utf-8") as f:
        first_line = f.readline()
        stripped = first_line.strip()
        first_obj = None
        looks_like_jsonl = False
        if stripped:
            try:
                first_obj = json.loads(stripped)
                # Only line-delimited if the first line is itself a JSON
                # object. A single line containing a whole array (or a
                # scalar) means this is one JSON document, not jsonl.
                looks_like_jsonl = isinstance(first_obj, dict)
            except json.JSONDecodeError:
                looks_like_jsonl = False

        if looks_like_jsonl:
            if isinstance(first_obj, dict):
                text = first_obj.get(json_field)
                if text:
                    yield text
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    error_counter[0] += 1
                    continue
                if isinstance(obj, dict):
                    text = obj.get(json_field)
                    if text:
                        yield text
        else:
            # Not line-delimited (e.g. pretty-printed or a top-level array/object).
            # Falls back to loading the whole document at once.
            f.seek(0)
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                error_counter[0] += 1
                return
            if isinstance(data, list):
                for obj in data:
                    if isinstance(obj, dict):
                        text = obj.get(json_field)
                        if text:
                            yield text
            elif isinstance(data, dict):
                text = data.get(json_field)
                if text:
                    yield text


def _count_file(args):
    path, json_field = args
    global _tokenizer
    n_tokens = 0
    n_docs = 0
    n_errors = 0

    try:
        if path.suffix in (".json", ".jsonl"):
            error_counter = [0]
            for text in _iter_json_texts(path, json_field, error_counter):
                n_tokens += len(_tokenizer.encode(text, add_special_tokens=False))
                n_docs += 1
            n_errors += error_counter[0]
        else:
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                n_tokens += _count_text_streaming(f)
            n_docs += 1
    except Exception as e:  # noqa: BLE001 - report and keep going
        print(f"[warn] failed on {path}: {e}", file=sys.stderr)
        n_errors += 1

    return n_tokens, n_docs, n_errors


def gather_files(input_path: Path, pattern: str = None):
    if input_path.is_file():
        return [input_path]
    if pattern:
        return sorted(input_path.rglob(pattern))
    return sorted(
        f for f in input_path.rglob("*")
        if f.is_file() and f.suffix in (".txt", ".json", ".jsonl")
    )


# ---------------------------------------------------------------------------
# Pointer mode
# ---------------------------------------------------------------------------

def _extract_pointers(path):
    """
    Read one index file and return (pointers, n_errors), where pointers is
    a list of (rank, pid, idx) tuples pulled from each record's
    "rank"/"pid"/"idx" keys. Records missing any of those keys are counted
    as errors and skipped.
    """
    error_counter = [0]
    pointers = []
    for obj in _iter_json_records(path, error_counter):
        try:
            pointers.append((obj["rank"], obj["pid"], int(obj["row_index"])))
        except KeyError:
            error_counter[0] += 1
    return pointers, error_counter[0]


def _resolve_data_file(task):
    """
    Given (rank, pid, idx_list), stream the corresponding data file once and
    tokenize the document text at each requested row. Returns
    (n_tokens, n_docs, n_errors).
    """
    global _tokenizer, _data_dir, _doc_field
    rank, pid, idx_list = task
    path = _data_dir / DATA_FILENAME_TEMPLATE.format(rank=rank, pid=pid)

    n_tokens = 0
    n_docs = 0
    n_errors = 0
    remaining = set(idx_list)

    if not path.exists():
        print(f"[warn] missing data file: {path}", file=sys.stderr)
        return 0, 0, len(remaining)

    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            for i, line in enumerate(f):
                if i not in remaining:
                    continue
                remaining.discard(i)
                line = line.strip()
                obj = None
                if line:
                    try:
                        obj = json.loads(line)
                    except json.JSONDecodeError:
                        obj = None
                text = obj.get(_doc_field) if isinstance(obj, dict) else None
                if text:
                    n_tokens += len(_tokenizer.encode(text, add_special_tokens=False))
                    n_docs += 1
                else:
                    n_errors += 1
                if not remaining:
                    break
    except Exception as e:  # noqa: BLE001
        print(f"[warn] failed reading {path}: {e}", file=sys.stderr)
        return n_tokens, n_docs, n_errors + len(remaining)

    # Any idx never reached means it was out of range for this file.
    if remaining:
        print(f"[warn] {len(remaining)} idx not found in {path.name}", file=sys.stderr)
        n_errors += len(remaining)

    return n_tokens, n_docs, n_errors


def run_pointer_mode(args, index_files):
    print(f"Found {len(index_files)} index file(s). Extracting pointers...")

    # Phase 1: extract (rank, pid, idx) pointers from every index file.
    # Lightweight/no tokenizer needed, so a plain pool is fine here.
    pointer_groups = {}
    total_pointer_errors = 0
    with mp.Pool(processes=args.workers) as pool:
        for pointers, n_errors in pool.imap_unordered(_extract_pointers, index_files):
            total_pointer_errors += n_errors
            for rank, pid, idx in pointers:
                zero_based_idx = idx - args.index_base
                if zero_based_idx < 0:
                    print(f"[warn] idx {idx} with --index-base {args.index_base} is negative, skipping", file=sys.stderr)
                    total_pointer_errors += 1
                    continue
                pointer_groups.setdefault((rank, pid), set()).add(zero_based_idx)

    total_requested = sum(len(v) for v in pointer_groups.values())
    print(
        f"Resolved {total_requested:,} document pointer(s) across "
        f"{len(pointer_groups)} data file(s). Tokenizing with {args.model_path} "
        f"using {args.workers} worker(s)..."
    )

    # Phase 2: resolve each distinct data file exactly once, pulling every
    # requested row out of it in a single streaming pass.
    tasks = [(rank, pid, idx_set) for (rank, pid), idx_set in pointer_groups.items()]

    total_tokens = 0
    total_docs = 0
    total_errors = total_pointer_errors

    with mp.Pool(
        processes=args.workers,
        initializer=_init_worker,
        initargs=(args.model_path, str(args.data_dir), args.doc_field),
    ) as pool:
        for i, (n_tokens, n_docs, n_errors) in enumerate(pool.imap_unordered(_resolve_data_file, tasks), start=1):
            total_tokens += n_tokens
            total_docs += n_docs
            total_errors += n_errors
            if i % 50 == 0 or i == len(tasks):
                print(f"  processed {i}/{len(tasks)} data files | running total: {total_tokens:,} tokens")

    print("\n--- Summary ---")
    print(f"Index files      : {len(index_files)}")
    print(f"Data files       : {len(tasks)}")
    print(f"Documents        : {total_docs:,} / {total_requested:,} requested")
    print(f"Total tokens     : {total_tokens:,}")
    if total_errors:
        print(f"Errors/skips     : {total_errors}")


def main():
    parser = argparse.ArgumentParser(description="Count tokens in a corpus with the Gemma-7B tokenizer.")
    parser.add_argument("--input", required=True, help="File or directory to scan (.txt / .json / .jsonl)")
    parser.add_argument("--json-field", default="text", help="[direct mode] Field name holding text in records")
    parser.add_argument("--workers", type=int, default=max(1, os.cpu_count() - 1), help="Parallel worker processes")
    parser.add_argument(
        "--model-path",
        default=TOKENIZER_NAME,
        help="Local directory with the downloaded gemma-7b tokenizer, or an HF repo id (default: google/gemma-7b)",
    )
    parser.add_argument(
        "--glob",
        default=None,
        help="Glob pattern (relative to --input, searched recursively) to select files, "
             "e.g. '*.jsonl' or 'shard-*.json'. Overrides the default .txt/.json/.jsonl scan.",
    )
    parser.add_argument(
        "--pointer-mode",
        action="store_true",
        help="Treat --input files as index files whose records point ('idx'/'rank'/'pid') "
             "to documents in separate data files. See module docstring for details.",
    )
    parser.add_argument(
        "--data-dir",
        default=None,
        help="[pointer mode] Directory containing the merged.out_rank<rank>_dump.json_pid<pid>.json "
             "data files. Defaults to --input if it's a directory.",
    )
    parser.add_argument(
        "--doc-field",
        default="text",
        help="[pointer mode] Field name holding the document text inside each data file record",
    )
    parser.add_argument(
        "--index-base",
        type=int,
        default=1,
        help="[pointer mode] Whether 'idx' in the index files is 0-indexed or 1-indexed (default: 1)",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Input path does not exist: {input_path}", file=sys.stderr)
        sys.exit(1)

    files = gather_files(input_path, args.glob)
    if not files:
        print("No matching files found.", file=sys.stderr)
        sys.exit(1)

    if args.pointer_mode:
        if args.data_dir:
            args.data_dir = Path(args.data_dir)
        elif input_path.is_dir():
            args.data_dir = input_path
        else:
            print("--data-dir is required in pointer mode when --input is a single file.", file=sys.stderr)
            sys.exit(1)
        run_pointer_mode(args, files)
        return

    print(f"Found {len(files)} file(s). Tokenizing with {args.model_path} using {args.workers} worker(s)...")

    total_tokens = 0
    total_docs = 0
    total_errors = 0

    tasks = [(f, args.json_field) for f in files]

    with mp.Pool(processes=args.workers, initializer=_init_worker, initargs=(args.model_path,)) as pool:
        for i, (n_tokens, n_docs, n_errors) in enumerate(pool.imap_unordered(_count_file, tasks), start=1):
            total_tokens += n_tokens
            total_docs += n_docs
            total_errors += n_errors
            if i % 50 == 0 or i == len(files):
                print(f"  processed {i}/{len(files)} files | running total: {total_tokens:,} tokens")

    print("\n--- Summary ---")
    print(f"Files processed : {len(files)}")
    print(f"Documents       : {total_docs:,}")
    print(f"Total tokens    : {total_tokens:,}")
    if total_errors:
        print(f"Errors/skips    : {total_errors}")


if __name__ == "__main__":
    main()
