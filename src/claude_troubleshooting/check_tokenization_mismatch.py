"""
One conclusive test: does lm-eval's tok_encode() produce different token
IDs than a plain tokenizer call on the exact same text? This is checked
directly, not inferred - prints the actual token ID lists side by side.

Usage:
    python check_tokenization_mismatch.py \
        --model_path /path/to/hf/model \
        --jsonl_file /path/to/wikitext.jsonl \
        --jsonl_line 61
"""

import argparse
import ast
import json


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
    args = parser.parse_args()

    with open(args.jsonl_file) as f:
        lines = [l for l in f if l.strip()]
    data = safe_parse(lines[args.jsonl_line], "line")
    text = extract_text(data, args.doc_key, args.json_key)

    from transformers import AutoTokenizer
    plain_tok = AutoTokenizer.from_pretrained(args.model_path)

    ids_plain_default = plain_tok(text).input_ids
    ids_plain_no_special = plain_tok(text, add_special_tokens=False).input_ids

    print("Method 1: plain AutoTokenizer(text) [default add_special_tokens]")
    print(f"  Length: {len(ids_plain_default)}  First 10: {ids_plain_default[:10]}")

    print("\nMethod 2: plain AutoTokenizer(text, add_special_tokens=False)")
    print(f"  Length: {len(ids_plain_no_special)}  First 10: {ids_plain_no_special[:10]}")

    from lm_eval.models.huggingface import HFLM
    lm = HFLM(pretrained=args.model_path, tokenizer=args.model_path, max_length=4096)
    ids_lmeval = lm.tok_encode(text)

    print("\nMethod 3: lm-eval's HFLM.tok_encode(text)  [what the harness actually uses]")
    print(f"  Length: {len(ids_lmeval)}  First 10: {ids_lmeval[:10]}")

    print("\n" + "=" * 70)
    print("DIRECT COMPARISON")
    print("=" * 70)
    print(f"Method 1 == Method 3? {ids_plain_default == ids_lmeval}")
    print(f"Method 2 == Method 3? {ids_plain_no_special == ids_lmeval}")

    if ids_plain_default != ids_lmeval and ids_plain_no_special != ids_lmeval:
        print(
            "\nlm-eval's tok_encode produces token IDs that don't match EITHER "
            "plain tokenization method. This confirms a real tokenization "
            "difference is in play - the printed ID lists above show exactly "
            "where they diverge (compare the first 10 tokens character by "
            "character)."
        )
    elif len(ids_plain_default) != len(ids_lmeval):
        print(
            f"\nSame starting tokens but DIFFERENT LENGTH: plain={len(ids_plain_default)} "
            f"vs lm-eval={len(ids_lmeval)}. Something is being added or dropped "
            f"partway through - check the END of each list too."
        )
        print(f"\nMethod 1 last 10: {ids_plain_default[-10:]}")
        print(f"Method 3 last 10: {ids_lmeval[-10:]}")
    else:
        print(
            "\nTokenization matches. This rules out tokenization as the cause - "
            "the discrepancy must be in the model loading/invocation itself "
            "(e.g. HFLM's from_pretrained kwargs like attn_implementation, "
            "device_map, or use_cache differing from a plain "
            "AutoModelForCausalLM.from_pretrained call)."
        )


if __name__ == "__main__":
    main()