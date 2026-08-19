"""
Last remaining variable check: tokenization is confirmed identical between
lm-eval and a plain tokenizer call. This script loads the model TWO ways
in the same process - once via HFLM (what lm-eval actually uses) and once
via plain AutoModelForCausalLM (what check_rolling_loglikelihood.py used) -
and runs both on the IDENTICAL token IDs, then diffs the resulting logits
directly, position by position.

Also explicitly prints tie_word_embeddings behavior for both, since a
loading-time warning suggested embed_tokens and lm_head may not actually
be tied despite config.json specifying they should be - this checks
whether that affects the two loading paths differently.

Usage:
    python check_loader_comparison.py \
        --model_path /path/to/hf/model \
        --jsonl_file /path/to/wikitext.jsonl \
        --jsonl_line 61
"""

import argparse
import ast
import json

import torch
import torch.nn.functional as F


def safe_parse(raw, label):
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return ast.literal_eval(raw)


def extract_text(data, doc_key, json_key):
    if doc_key in data:
        return data[doc_key][json_key]
    return data[json_key]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--jsonl_file", required=True)
    parser.add_argument("--jsonl_line", type=int, default=0)
    parser.add_argument("--doc_key", default="doc")
    parser.add_argument("--json_key", default="page")
    parser.add_argument("--max_tokens", type=int, default=4097, help="Truncate to this many tokens (matches window 0 length)")
    args = parser.parse_args()

    with open(args.jsonl_file) as f:
        lines = [l for l in f if l.strip()]
    data = safe_parse(lines[args.jsonl_line], "line")
    text = extract_text(data, args.doc_key, args.json_key)

    from transformers import AutoTokenizer, AutoModelForCausalLM
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    input_ids_list = tokenizer(text).input_ids[: args.max_tokens]
    print(f"Using {len(input_ids_list)} tokens (truncated to --max_tokens={args.max_tokens})")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    input_ids = torch.tensor([input_ids_list], device=device)

    # --- Loader A: plain AutoModelForCausalLM, exactly like check_rolling_loglikelihood.py ---
    print("\nLoading model via plain AutoModelForCausalLM.from_pretrained(..., torch_dtype=torch.float32) ...")
    model_plain = AutoModelForCausalLM.from_pretrained(args.model_path, torch_dtype=torch.float32)
    model_plain.eval().to(device)
    print(f"  model_plain dtype: {next(model_plain.parameters()).dtype}")
    print(f"  model_plain config.tie_word_embeddings: {model_plain.config.tie_word_embeddings}")
    embed_tied_plain = torch.equal(
        model_plain.get_input_embeddings().weight, model_plain.get_output_embeddings().weight
    )
    print(f"  model_plain embed_tokens.weight == lm_head.weight (actual tensors): {embed_tied_plain}")

    with torch.no_grad():
        logits_plain = model_plain(input_ids=input_ids).logits[0]

    # --- Loader B: HFLM, exactly like lm-eval uses ---
    print("\nLoading model via lm_eval.models.huggingface.HFLM ...")
    from lm_eval.models.huggingface import HFLM
    lm = HFLM(pretrained=args.model_path, tokenizer=args.model_path, max_length=4096)
    print(f"  lm.model dtype: {next(lm.model.parameters()).dtype}")
    print(f"  lm.model config.tie_word_embeddings: {lm.model.config.tie_word_embeddings}")
    embed_tied_hflm = torch.equal(
        lm.model.get_input_embeddings().weight, lm.model.get_output_embeddings().weight
    )
    print(f"  lm.model embed_tokens.weight == lm_head.weight (actual tensors): {embed_tied_hflm}")
    print(f"  lm.model attn_implementation: {getattr(lm.model.config, '_attn_implementation', 'unknown')}")

    with torch.no_grad():
        logits_hflm = lm.model(input_ids=input_ids).logits[0]

    # --- Direct comparison ---
    print("\n" + "=" * 70)
    print("DIRECT LOGIT COMPARISON (identical input_ids, both models)")
    print("=" * 70)

    same_shape = logits_plain.shape == logits_hflm.shape
    print(f"Logits shape match: {same_shape}  (plain={tuple(logits_plain.shape)}, hflm={tuple(logits_hflm.shape)})")

    if same_shape:
        allclose = torch.allclose(logits_plain, logits_hflm, atol=1e-2)
        print(f"Logits allclose (atol=1e-2): {allclose}")
        diff = (logits_plain - logits_hflm).abs()
        print(f"Max abs logit diff: {diff.max().item():.4f}")
        print(f"Mean abs logit diff: {diff.mean().item():.4f}")

        # Show where the biggest divergence starts (first position where diff is large)
        per_pos_max_diff = diff.max(dim=-1).values
        threshold = 1.0
        bad_positions = (per_pos_max_diff > threshold).nonzero(as_tuple=True)[0]
        if len(bad_positions) > 0:
            first_bad = bad_positions[0].item()
            print(f"\nFirst position where logits diverge significantly (max diff > {threshold}): position {first_bad}")
            print(f"  (out of {len(input_ids_list)} total positions)")
        else:
            print(f"\nNo position exceeds diff threshold {threshold} - logits are consistent throughout.")

    # Also compute actual per-token logprob of the real next tokens under both,
    # to directly compare to the -4.165 avg logprob baseline we measured before
    log_probs_plain = F.log_softmax(logits_plain, dim=-1)
    log_probs_hflm = F.log_softmax(logits_hflm, dim=-1)

    total_plain, total_hflm = 0.0, 0.0
    for pos in range(len(input_ids_list) - 1):
        target = input_ids_list[pos + 1]
        total_plain += log_probs_plain[pos, target].item()
        total_hflm += log_probs_hflm[pos, target].item()

    n = len(input_ids_list) - 1
    print(f"\nAvg per-token logprob (plain loader):  {total_plain / n:.3f}")
    print(f"Avg per-token logprob (HFLM loader):    {total_hflm / n:.3f}")

    if abs(total_plain / n - total_hflm / n) > 1.0:
        print(
            "\nCONFIRMED: the two loaders produce meaningfully different logits/"
            "logprobs on the IDENTICAL input. Compare the printed config/dtype/"
            "attn_implementation/tie_word_embeddings values above - whichever "
            "differs between the two is the root cause."
        )
    else:
        print(
            "\nBoth loaders agree. If lm-eval's actual reported wikitext number "
            "is still wildly different from this, the discrepancy must be "
            "somewhere in request batching/aggregation across MULTIPLE "
            "documents, not in per-document scoring."
        )


if __name__ == "__main__":
    main()