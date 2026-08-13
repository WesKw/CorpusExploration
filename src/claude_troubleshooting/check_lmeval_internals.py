"""
Directly exercises lm-eval-harness's OWN internal functions
(get_rolling_token_windows, make_disjoint_window, HFLM._loglikelihood_tokens)
on a real document, and compares the result token-for-token against a plain
forward() pass computed independently (same method as check_rolling_loglikelihood.py).

We've already ruled out:
- max_length resolution (confirmed correct: prints 4096 as expected)
- Gross window-count blowup (with max_length=4096 on a ~4650 token doc,
  there are only 2 windows - nowhere near enough to explain a ~30x
  discrepancy in average logprob)

This script isolates whether the bug is in:
(a) get_rolling_token_windows / make_disjoint_window - by printing the
    actual windows produced and their lengths, or
(b) HFLM._loglikelihood_tokens itself - by comparing its per-window NLL
    output against an independent forward-pass computation on the exact
    same window token IDs.

Usage:
    python check_lmeval_internals.py \
        --model_path /path/to/hf/model \
        --jsonl_file /path/to/wikitext.jsonl \
        --jsonl_line 61
"""

import argparse
import ast
import json

import torch
import torch.nn.functional as F


def safe_parse(raw: str, source_label: str) -> dict:
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        try:
            return ast.literal_eval(raw)
        except (ValueError, SyntaxError) as e:
            raise ValueError(f"Could not parse {source_label}: {e}")


def extract_text(data: dict, doc_key: str, json_key: str, source_label: str):
    if doc_key in data:
        nested = data[doc_key]
        if json_key not in nested:
            raise KeyError(f"'{json_key}' not found under '{doc_key}' in {source_label}")
        return nested[json_key]
    elif json_key in data:
        return data[json_key]
    else:
        raise KeyError(f"Neither '{doc_key}.{json_key}' nor '{json_key}' found in {source_label}")


def independent_forward_nll(model, input_ids: list[int], device) -> float:
    """
    Plain forward pass, no lm-eval involved. Computes summed NLL of every
    token given all preceding tokens (i.e. treats the whole window as
    context+target, same convention lm-eval uses internally for a single
    window: predict every token from position 0 onward, conditioned on
    everything before it).
    """
    ids = torch.tensor([input_ids], device=device)
    with torch.no_grad():
        logits = model(input_ids=ids).logits[0]
    log_probs = F.log_softmax(logits, dim=-1)
    total_nll = 0.0
    for pos in range(len(input_ids) - 1):
        target = input_ids[pos + 1]
        total_nll += -log_probs[pos, target].item()
    return total_nll


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--jsonl_file", required=True)
    parser.add_argument("--jsonl_line", type=int, default=0)
    parser.add_argument("--doc_key", default="doc")
    parser.add_argument("--json_key", default="page")
    parser.add_argument("--max_length", type=int, default=4096)
    args = parser.parse_args()

    # Import lm-eval's actual internals - this MUST be run in an environment
    # where lm-evaluation-harness is installed (same venv you use for lm_eval).
    from lm_eval import utils as lm_eval_utils
    from lm_eval.models.huggingface import HFLM

    with open(args.jsonl_file, "r") as f:
        lines = [line for line in f if line.strip()]
    data = safe_parse(lines[args.jsonl_line], f"{args.jsonl_file}:line{args.jsonl_line}")
    text = extract_text(data, args.doc_key, args.json_key, f"{args.jsonl_file}:line{args.jsonl_line}")

    print(f"Loading model via HFLM from {args.model_path} ...")
    lm = HFLM(
        pretrained=args.model_path,
        tokenizer=args.model_path,
        max_length=args.max_length,
        batch_size=1,
    )
    device = lm.device

    print(f"lm.max_length = {lm.max_length}")
    print(f"lm.prefix_token_id = {lm.prefix_token_id}")

    token_list = lm.tok_encode(text)
    print(f"\nTotal tokens (lm.tok_encode): {len(token_list)}")

    # Reproduce EXACTLY what loglikelihood_rolling does internally
    rolling_windows = list(
        map(
            lm_eval_utils.make_disjoint_window,
            lm_eval_utils.get_rolling_token_windows(
                token_list=token_list,
                prefix_token=lm.prefix_token_id,
                max_seq_len=lm.max_length,
                context_len=1,
            ),
        )
    )

    print(f"Number of windows: {len(rolling_windows)}")
    for i, (context, continuation) in enumerate(rolling_windows):
        print(
            f"  Window {i}: context_len={len(context)}, "
            f"continuation_len={len(continuation)}, "
            f"context[:5]={context[:5]}, continuation[:5]={continuation[:5]}, "
            f"continuation[-5:]={continuation[-5:]}"
        )

    # Now call lm-eval's ACTUAL scoring function on these exact windows
    windows_for_scoring = [(None,) + w for w in rolling_windows]
    print("\nCalling HFLM._loglikelihood_tokens on the real windows ...")
    lmeval_nlls = lm._loglikelihood_tokens(
        requests=windows_for_scoring,
        disable_tqdm=True,
        override_bs=len(windows_for_scoring),
    )
    for i, (nll, is_greedy) in enumerate(lmeval_nlls):
        print(f"  Window {i}: lm-eval NLL = {nll:.3f} (is_greedy={is_greedy})")

    lmeval_total = sum(nll for nll, _ in lmeval_nlls)
    print(f"\nlm-eval total NLL (sum across windows): {lmeval_total:.3f}")

    # Independent cross-check: for each window, run a plain forward pass
    # on context+continuation concatenated, and compute NLL of the
    # continuation tokens ourselves, bypassing _loglikelihood_tokens
    # entirely.
    print("\nIndependent cross-check via plain forward() on each window's full token span:")
    independent_total = 0.0
    for i, (context, continuation) in enumerate(rolling_windows):
        full_ids = context + continuation
        ids = torch.tensor([full_ids], device=device)
        with torch.no_grad():
            logits = lm.model(input_ids=ids).logits[0]
        log_probs = F.log_softmax(logits, dim=-1)
        cont_start = len(context)
        window_nll = 0.0
        for j in range(len(continuation)):
            pos = cont_start + j - 1
            target = full_ids[cont_start + j]
            window_nll += -log_probs[pos, target].item()
        independent_total += window_nll
        print(f"  Window {i}: independent NLL = {window_nll:.3f}")

    print(f"\nIndependent total NLL (sum across windows): {independent_total:.3f}")
    print(f"\nDIFFERENCE (lm-eval total - independent total): {lmeval_total - independent_total:.3f}")
    if abs(lmeval_total - independent_total) > 1.0:
        print(
            "\nMISMATCH: lm-eval's _loglikelihood_tokens produces a different "
            "NLL than an independent forward-pass computation on the IDENTICAL "
            "token windows. The bug is inside _loglikelihood_tokens itself - "
            "worth diffing its internals (padding, attention_mask construction, "
            "or continuation-slicing logic) against the independent computation "
            "above to find exactly where they diverge."
        )
    else:
        print(
            "\nMATCH: lm-eval's scoring function agrees with the independent "
            "computation on these windows. If the final reported wikitext "
            "number is still wildly different from this, the bug must be in "
            "how per-window NLLs get aggregated/reported downstream (e.g. in "
            "process_results or the metrics registry), not in the scoring "
            "itself."
        )


if __name__ == "__main__":
    main()