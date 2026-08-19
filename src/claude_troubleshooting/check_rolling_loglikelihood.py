"""
Diagnostic: replicate lm-eval's loglikelihood_rolling behavior on a single
long document, but log PER-TOKEN logprob instead of just a final sum.

We already know:
- Short-context loglikelihood (arc_easy/hellaswag/openbookqa) works
  sanely (numbers near random baseline, not exploding).
- Long-context rolling loglikelihood (wikitext) produces an average
  per-token logprob around -128 nats, ~12x worse than random guessing
  under a 32000-token vocab.
- max_position_embeddings=4096 vs 8192 in config.json made NO difference,
  suggesting the custom architecture doesn't actually read that field.

This script finds WHERE in a long sequence things go wrong:
- If per-token logprob is fine early and collapses after some position
  N, that points to a position-encoding / context-length problem
  specific to long sequences (RoPE table sizing, cache/position_ids
  bug, etc.) - and N will tell us roughly where the real ceiling is.
- If per-token logprob is bad from token 1, the rolling-context
  eval path is doing something structurally different from what we
  tested before (e.g. windowing/stride logic, or how lm-eval
  constructs input_ids for loglikelihood_rolling specifically -
  worth then diffing against lm-eval's actual implementation).

Usage:
    python check_rolling_loglikelihood.py \
        --model_path /path/to/hf/model \
        --text_file /path/to/a/wikitext_doc.txt

    # Or, if your document is stored as JSON with a "page" key holding
    # the text (e.g. {"page": "The history of computing is..."}):
    python check_rolling_loglikelihood.py \
        --model_path /path/to/hf/model \
        --json_file /path/to/doc.json

    # Or, if you have a JSONL file where each line is a JSON object with
    # a "page" key (e.g. one wikitext document per line), analyze one
    # specific line (0-indexed) or all lines in sequence:
    python check_rolling_loglikelihood.py \
        --model_path /path/to/hf/model \
        --jsonl_file /path/to/docs.jsonl \
        --jsonl_line 0

    python check_rolling_loglikelihood.py \
        --model_path /path/to/hf/model \
        --jsonl_file /path/to/docs.jsonl \
        --jsonl_all

If you don't have a text file handy, pass --use_sample_text to fall back
to a built-in long-ish synthetic passage (less representative of real
wikitext formatting/markup, but still useful for isolating a position-
based cutoff).
"""

import argparse
import ast
import json

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


def safe_parse(raw: str, source_label: str) -> dict:
    """
    Parses a JSON-ish object that may use either standard double-quoted
    JSON syntax, or Python dict repr syntax with single-quoted keys
    (common when data was written via str(dict) rather than json.dumps).

    Plain json.loads() is tried first since it's stricter/safer. If that
    fails, falls back to ast.literal_eval, which handles single-quoted
    Python dict literals correctly - including when the document text
    itself contains double quotes, which would otherwise break naive
    quote-swapping approaches (e.g. blindly replacing ' with ").
    """
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        try:
            return ast.literal_eval(raw)
        except (ValueError, SyntaxError) as e:
            raise ValueError(
                f"Could not parse {source_label} as either JSON or a Python "
                f"dict literal: {e}"
            )


def extract_text(data: dict, doc_key: str, json_key: str, source_label: str):
    """
    Pulls the document text out of a parsed JSON object. Supports both:
    - Nested: {"doc": {"page": "..."}}  (default expected shape)
    - Flat:   {"page": "..."}           (fallback, for older files)
    """
    if doc_key in data:
        nested = data[doc_key]
        if not isinstance(nested, dict):
            raise TypeError(
                f"Expected '{doc_key}' in {source_label} to be a dict, "
                f"got {type(nested).__name__}"
            )
        if json_key not in nested:
            raise KeyError(
                f"Key '{json_key}' not found under '{doc_key}' in {source_label}. "
                f"Available keys under '{doc_key}': {list(nested.keys())}"
            )
        return nested[json_key]
    elif json_key in data:
        # Fallback: flat structure, no nesting
        return data[json_key]
    else:
        raise KeyError(
            f"Neither '{doc_key}.{json_key}' nor top-level '{json_key}' found in "
            f"{source_label}. Available top-level keys: {list(data.keys())}"
        )

SAMPLE_TEXT = """
The history of computing is a long and varied one, stretching back
centuries before the invention of the modern electronic computer.
Early mechanical calculating devices, such as the abacus, allowed
merchants and scholars to perform arithmetic more quickly and
reliably than by hand alone. In the seventeenth century, inventors
such as Blaise Pascal and Gottfried Wilhelm Leibniz built mechanical
calculators capable of performing addition, subtraction, and in
Leibniz's case, multiplication and division as well. These devices
were often expensive, delicate, and difficult to manufacture at
scale, which limited their widespread adoption. It was not until the
nineteenth century that Charles Babbage conceived of a more general
purpose calculating engine, one capable of being programmed to
perform a wide range of mathematical operations. Babbage's Analytical
Engine, though never completed in his lifetime, is often cited as a
conceptual precursor to the modern computer, incorporating ideas such
as a central processing unit, memory storage, and conditional
branching. Ada Lovelace, who worked closely with Babbage, is widely
regarded as having written what many consider to be the first
computer program, an algorithm intended to be carried out by the
Analytical Engine to compute Bernoulli numbers. The twentieth century
saw an explosion of progress in the field, driven first by the
demands of the Second World War and subsequently by rapid advances in
electronics. Early electronic computers such as the ENIAC used
thousands of vacuum tubes and consumed enormous amounts of power,
occupying entire rooms while offering computational capabilities that
are now dwarfed by even the simplest of modern handheld devices.
""".strip().replace("\n", " ")


def load(model_path):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float32)
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    return model, tokenizer, device


def per_token_logprobs(model, tokenizer, device, text: str, chunk_report_every: int = 50):
    input_ids = tokenizer(text, return_tensors="pt").input_ids.to(device)
    seq_len = input_ids.shape[1]
    print(f"Total sequence length: {seq_len} tokens")

    with torch.no_grad():
        logits = model(input_ids=input_ids).logits[0]  # [seq_len, vocab]

    log_probs = F.log_softmax(logits, dim=-1)

    per_token = []
    for pos in range(seq_len - 1):
        target = input_ids[0, pos + 1]
        lp = log_probs[pos, target].item()
        per_token.append(lp)

    return per_token


def summarize(per_token, window=50):
    n = len(per_token)
    print(f"\n{'Position':>10} {'Avg logprob (window)':>22} {'Min':>10} {'Max':>10}")
    for start in range(0, n, window):
        chunk = per_token[start:start + window]
        avg = sum(chunk) / len(chunk)
        print(f"{start:>10} {avg:>22.3f} {min(chunk):>10.2f} {max(chunk):>10.2f}")

    print(f"\nOverall average per-token logprob: {sum(per_token)/n:.3f}")
    print(f"Random-guess baseline (ln(32000)):  {-10.37:.3f}")

    # find the first position where things clearly go off the rails
    # (defined loosely as: rolling window average logprob < -20, i.e.
    # roughly 2x worse than random guessing - real collapse, not noise)
    threshold = -20.0
    collapse_pos = None
    for start in range(0, n, window):
        chunk = per_token[start:start + window]
        avg = sum(chunk) / len(chunk)
        if avg < threshold:
            collapse_pos = start
            break

    if collapse_pos is not None:
        print(
            f"\nPer-token logprob first drops below {threshold} (i.e. collapses) "
            f"around position {collapse_pos}. This strongly suggests a "
            f"position/context-length-dependent bug (e.g. RoPE table sizing, "
            f"a cache or position_ids issue) rather than a general model quality "
            f"problem - the model is fine up to ~position {collapse_pos}, then "
            f"breaks down."
        )
    else:
        print(
            "\nNo clear collapse point found - per-token logprob looks roughly "
            "consistent (good or bad) across the whole sequence. If it's "
            "consistently bad from the start, the issue is likely NOT "
            "position-related, and is more likely in how input_ids/BOS/context "
            "are constructed for this rolling-loglikelihood style request "
            "specifically (different from the short-context path already "
            "verified to work)."
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--text_file", default=None, help="Path to a plain text file (e.g. one wikitext document)")
    parser.add_argument("--json_file", default=None, help="Path to a JSON file with a 'doc': {'page': ...} structure holding the document text")
    parser.add_argument("--doc_key", default="doc", help="Outer key holding the nested dict (default: 'doc')")
    parser.add_argument("--json_key", default="page", help="Inner key to read the document text from, nested under --doc_key (default: 'page')")
    parser.add_argument("--jsonl_file", default=None, help="Path to a JSONL file, one JSON object per line, each with a 'doc': {'page': ...} structure")
    parser.add_argument("--jsonl_line", type=int, default=None, help="0-indexed line number to analyze from --jsonl_file. If omitted and --jsonl_all is not set, defaults to line 0.")
    parser.add_argument("--jsonl_all", action="store_true", help="Analyze every line in --jsonl_file, one after another, instead of a single line")
    parser.add_argument("--use_sample_text", action="store_true")
    parser.add_argument("--window", type=int, default=50, help="Window size (in tokens) for summarizing logprob trend")
    args = parser.parse_args()

    texts = []  # list of (label, text) pairs

    if args.text_file:
        with open(args.text_file, "r") as f:
            texts.append((args.text_file, f.read()))
    elif args.json_file:
        with open(args.json_file, "r") as f:
            data = safe_parse(f.read(), args.json_file)
        text = extract_text(data, args.doc_key, args.json_key, args.json_file)
        texts.append((args.json_file, text))
    elif args.jsonl_file:
        with open(args.jsonl_file, "r") as f:
            lines = [line for line in f if line.strip()]

        if args.jsonl_all:
            selected_indices = range(len(lines))
        else:
            line_idx = args.jsonl_line if args.jsonl_line is not None else 0
            selected_indices = [line_idx]

        for idx in selected_indices:
            if idx >= len(lines):
                raise IndexError(f"Line {idx} out of range - {args.jsonl_file} has {len(lines)} lines")
            label = f"{args.jsonl_file}:line{idx}"
            data = safe_parse(lines[idx], label)
            text = extract_text(data, args.doc_key, args.json_key, label)
            texts.append((label, text))
    elif args.use_sample_text:
        texts.append(("sample_text", SAMPLE_TEXT))
    else:
        raise ValueError("Pass one of --text_file, --json_file, --jsonl_file, or --use_sample_text")

    model, tokenizer, device = load(args.model_path)

    for label, text in texts:
        print("\n" + "#" * 70)
        print(f"# Document: {label}")
        print("#" * 70)
        per_token = per_token_logprobs(model, tokenizer, device, text)
        summarize(per_token, window=args.window)