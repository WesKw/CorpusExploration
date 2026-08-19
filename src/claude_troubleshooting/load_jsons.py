"""
Load documents from processed dump files that are NOT already accounted for
by their corresponding master index file.

Expected layout (all files live in the same data_dir):
  - merged.out_rank<rank>.json is line-delimited JSON. Each record has
    "pid", "idx", "rank" fields pointing to a document's location.
  - merged.out_rank<rank>_dump.json_pid<pid>.json is line-delimited JSON,
    one document per line/row, addressed by row number ("idx").
  - Indexing per pid is NOT assumed to be contiguous: processing failures
    can leave gaps (a row skipped and never logged), and the master file
    can contain duplicate (pid, idx) records. So "unindexed" documents are
    determined by exact (pid, idx) set membership, not a per-pid high-water
    mark - duplicates are harmless here, and gaps are preserved correctly.

This module assumes data_dir already exists; no filesystem validation is
performed. Callers are expected to check that themselves.
"""

import json
import re
from pathlib import Path

MASTER_FILENAME_RE = re.compile(r"^merged\.out_rank(\d+)\.json$")
DUMP_FILENAME_RE = re.compile(r"^merged\.out_rank(\d+)_dump\.json_pid(.+)\.json$")


def _iter_json_records(path):
    """Yield dict records from a line-delimited JSON file, skipping blank/malformed lines."""
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict):
                yield obj


def _normalize_pid(pid):
    """
    Canonicalize a pid value so that int 5, float 5.0, "5", and "005" all
    compare equal. Falls back to the raw (stripped) string for non-numeric
    pids (e.g. hash-like ids) so those still match on exact string equality.
    """
    s = str(pid).strip()
    try:
        return str(int(float(s)))
    except (TypeError, ValueError):
        return s


def discover_ranks(data_dir):
    """Return sorted rank ints for every merged.out_rank<rank>.json master file present."""
    ranks = set()
    for p in Path(data_dir).iterdir():
        if p.is_file():
            m = MASTER_FILENAME_RE.match(p.name)
            if m:
                ranks.add(int(m.group(1)))
    return sorted(ranks)


def discover_dump_files(data_dir, rank):
    """Return {normalized_pid: Path} for every dump file belonging to the given rank."""
    dump_files = {}
    for p in Path(data_dir).iterdir():
        if p.is_file():
            m = DUMP_FILENAME_RE.match(p.name)
            if m and int(m.group(1)) == rank:
                dump_files[_normalize_pid(m.group(2))] = p
    return dump_files


def load_indexed_sets(master_path, index_base=1):
    """
    Read a merged.out_rank<rank>.json master file and return
    {normalized_pid: set of 0-based idx} already accounted for there.

    Duplicate (pid, idx) records collapse harmlessly into the same set
    entry. Returns an empty dict if master_path doesn't exist (nothing
    accounted for yet).
    """
    if not Path(master_path).exists():
        return {}

    indexed = {}
    for obj in _iter_json_records(master_path):
        pid, idx = obj.get("pid"), int(obj.get("idx"))
        if pid is None or idx is None:
            continue
        zero_based = idx - index_base
        if zero_based < 0:
            continue
        indexed.setdefault(_normalize_pid(pid), set()).add(zero_based)
    return indexed


def diagnose(data_dir, rank, index_base=1):
    """
    Cross-check counts for a rank against what `wc -l` would report, to help
    catch pid/idx mismatches, duplicate master records, and non-contiguous
    idx gaps. Returns a dict of diagnostic counts; doesn't tokenize or load
    document text.

    per_pid_breakdown for each pid reports:
      - master_records: raw record count for this pid in the master file
      - distinct_idx_count: count of unique 0-based idx values seen
      - max_idx: highest 0-based idx seen (-1 if none)
      - duplicate_count: master_records - distinct_idx_count
          (> 0 means the same idx was logged more than once for this pid)
      - gap_count: (max_idx + 1) - distinct_idx_count
          (> 0 means some idx values below the max were never logged at
          all, e.g. from a processing failure)
    """
    data_dir = Path(data_dir)
    master_path = data_dir / f"merged.out_rank{rank}.json"

    master_line_count = 0
    per_pid_stats = {}
    if master_path.exists():
        for obj in _iter_json_records(master_path):
            master_line_count += 1
            pid, idx = obj.get("pid"), int(obj.get("idx"))
            if pid is None or idx is None:
                continue
            key = _normalize_pid(pid)
            stats = per_pid_stats.setdefault(key, {"records": 0, "distinct_idx": set(), "max_idx": -1})
            stats["records"] += 1
            zero_based = idx - index_base
            if zero_based >= 0:
                stats["distinct_idx"].add(zero_based)
                stats["max_idx"] = max(stats["max_idx"], zero_based)

    per_pid_breakdown = {
        pid: {
            "master_records": s["records"],
            "distinct_idx_count": len(s["distinct_idx"]),
            "max_idx": s["max_idx"],
            "duplicate_count": s["records"] - len(s["distinct_idx"]),
            "gap_count": (s["max_idx"] + 1 - len(s["distinct_idx"])) if s["max_idx"] >= 0 else 0,
        }
        for pid, s in per_pid_stats.items()
    }

    master_pids = set(per_pid_stats.keys())
    # accounted_for_total uses distinct idx count (exact), not max+1, since
    # gaps mean max+1 overstates how much is actually accounted for.
    accounted_for_total = sum(len(s["distinct_idx"]) for s in per_pid_stats.values())
    total_gaps = sum(b["gap_count"] for b in per_pid_breakdown.values())
    total_duplicates = sum(b["duplicate_count"] for b in per_pid_breakdown.values())

    dump_files = discover_dump_files(data_dir, rank)
    dump_pids = set(dump_files.keys())
    dump_line_counts = {
        pid: sum(1 for _ in open(path, "r", encoding="utf-8", errors="ignore"))
        for pid, path in dump_files.items()
    }
    dump_total_lines = sum(dump_line_counts.values())

    return {
        "master_line_count": master_line_count,
        "accounted_for_total": accounted_for_total,
        "total_duplicates": total_duplicates,
        "total_gaps": total_gaps,
        "per_pid_breakdown": per_pid_breakdown,
        "master_pids": master_pids,
        "dump_pids": dump_pids,
        "pids_in_master_not_in_dumps": master_pids - dump_pids,
        "pids_in_dumps_not_in_master": dump_pids - master_pids,
        "dump_total_lines": dump_total_lines,
        "dump_line_counts": dump_line_counts,
        "expected_max_unindexed": dump_total_lines - accounted_for_total,
    }


def inspect_duplicates(data_dir, rank, pid, limit=5, index_base=1):
    """
    For a given pid, find idx values that appear more than once in
    merged.out_rank<rank>.json and return the raw JSON records for each,
    so you can inspect whether they're truly identical duplicates or
    actually distinct records that happen to share an idx.

    Returns {zero_based_idx: [raw_record, ...]} for up to `limit` repeated
    idx values (each with 2+ records).
    """
    data_dir = Path(data_dir)
    master_path = data_dir / f"merged.out_rank{rank}.json"
    target_pid = _normalize_pid(pid)

    by_idx = {}
    for obj in _iter_json_records(master_path):
        p, idx = obj.get("pid"), int(obj.get("idx"))
        if p is None or idx is None or _normalize_pid(p) != target_pid:
            continue
        zero_based = idx - index_base
        by_idx.setdefault(zero_based, []).append(obj)

    duplicates = {idx: records for idx, records in by_idx.items() if len(records) > 1}
    # Return the first `limit` repeated idx values (arbitrary order preserved via dict insertion)
    return dict(list(duplicates.items())[:limit])


def load_unindexed_documents(data_dir, rank, doc_field="text", index_base=1, doc_length:int=None):
    """
    Yield the text of every document belonging to `rank` whose (pid, idx)
    is NOT present in merged.out_rank<rank>.json.

    Assumes data_dir exists. Yields plain document-text strings.
    """
    data_dir = Path(data_dir)
    master_path = data_dir / f"merged.out_rank{rank}.json"
    indexed = load_indexed_sets(master_path, index_base=index_base)

    for pid, dump_path in discover_dump_files(data_dir, rank).items():
        already_indexed = indexed.get(pid, set())
        with open(dump_path, "r", encoding="utf-8", errors="ignore") as f:
            for i, line in enumerate(f):
                if i in already_indexed:
                    continue
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(obj, dict):
                    continue
                text = obj.get(doc_field)
                if text:
                    yield {"text": text[:doc_length]}


def load_all_unindexed_documents(data_dir, ranks=None, doc_field="text", index_base=1):
    """
    Convenience wrapper over load_unindexed_documents() that iterates every
    rank found in data_dir (or a specific subset via `ranks`), yielding
    (rank, text) pairs.
    """
    data_dir = Path(data_dir)
    ranks = ranks if ranks is not None else discover_ranks(data_dir)
    for rank in ranks:
        for text in load_unindexed_documents(data_dir, rank, doc_field=doc_field, index_base=index_base):
            yield rank, text