"""
Compares embedding tensors between the raw DCP checkpoint (source of
truth - what was actually trained) and the HF-converted checkpoint
(what lm-eval and check_generation.py actually load), to check whether
convert_to_hf.py's sd_adapter.to_hf() step is silently corrupting or
misaligning the embedding table.

This matters because:
- Short-context generation (common/low-index tokens) already looked
  coherent - a vocab/embedding misalignment could easily be invisible
  there if common tokens happen to land in overlapping/similar rows.
- Long documents (wikitext) sample a much broader, rarer slice of the
  vocabulary - exactly where a subtle misalignment would first become
  visible as catastrophically bad predictions.

Usage:
    python check_embedding_conversion.py \
        --dcp_checkpoint /path/to/step-XXXX/ \
        --hf_checkpoint /path/to/converted/model/ \
        --vocab_size 32000 \
        --dim 640
"""

import argparse

import torch
import torch.distributed.checkpoint as dcp
from safetensors import safe_open


def load_dcp_tensor(checkpoint_path: str, key: str, shape: tuple):
    sd = {key: torch.zeros(shape)}
    dcp.load(sd, checkpoint_id=checkpoint_path)
    return sd[key]


def load_hf_tensor(hf_checkpoint_path: str, key: str):
    import os

    # Handle both single-file and sharded safetensors checkpoints
    st_files = [f for f in os.listdir(hf_checkpoint_path) if f.endswith(".safetensors")]
    if not st_files:
        raise FileNotFoundError(f"No .safetensors files found in {hf_checkpoint_path}")

    for fname in st_files:
        fpath = os.path.join(hf_checkpoint_path, fname)
        with safe_open(fpath, framework="pt") as f:
            if key in f.keys():
                return f.get_tensor(key)
    raise KeyError(f"Key '{key}' not found in any safetensors file under {hf_checkpoint_path}. "
                    f"Available keys in {st_files[0]}: {list(safe_open(os.path.join(hf_checkpoint_path, st_files[0]), framework='pt').keys())[:20]}")


def compare_embeddings(dcp_path, hf_path, vocab_size, dim,
                        dcp_key="tok_embeddings.weight", hf_key="model.embed_tokens.weight"):
    print(f"Loading DCP tensor '{dcp_key}' with shape ({vocab_size}, {dim}) ...")
    dcp_tensor = load_dcp_tensor(dcp_path, dcp_key, (vocab_size, dim))

    print(f"Loading HF tensor '{hf_key}' ...")
    hf_tensor = load_hf_tensor(hf_path, hf_key).float()

    print(f"\nDCP tensor shape: {tuple(dcp_tensor.shape)}")
    print(f"HF tensor shape:  {tuple(hf_tensor.shape)}")

    if dcp_tensor.shape != hf_tensor.shape:
        print(
            "\nMISMATCH: shapes differ between the DCP checkpoint and the HF "
            "conversion. This alone could explain everything - if the HF "
            "config's vocab_size doesn't match the trained model, rows would "
            "be misaligned, padded, or truncated during conversion."
        )
        return

    # Check overall similarity
    are_close = torch.allclose(dcp_tensor, hf_tensor, atol=1e-3)
    print(f"\nTensors match (allclose, atol=1e-3): {are_close}")

    if not are_close:
        diff = (dcp_tensor - hf_tensor).abs()
        print(f"Max abs diff: {diff.max().item():.6f}")
        print(f"Mean abs diff: {diff.mean().item():.6f}")

        # Find which rows (token IDs) differ most - if it's a broad set of
        # HIGH token IDs specifically, that's a strong hint of an
        # index-shift/misalignment bug rather than random noise/precision
        row_diffs = diff.mean(dim=1)
        worst_rows = torch.topk(row_diffs, min(10, len(row_diffs)))
        print("\nToken IDs (rows) with largest embedding differences:")
        for idx, val in zip(worst_rows.indices.tolist(), worst_rows.values.tolist()):
            print(f"  token_id={idx:6d}  mean_abs_diff={val:.6f}")

        best_rows = torch.topk(row_diffs, min(10, len(row_diffs)), largest=False)
        print("\nToken IDs (rows) with smallest embedding differences:")
        for idx, val in zip(best_rows.indices.tolist(), best_rows.values.tolist()):
            print(f"  token_id={idx:6d}  mean_abs_diff={val:.6f}")
    else:
        print("\nEmbeddings match exactly (within tolerance) - conversion is NOT "
              "corrupting or misaligning the embedding table. The bug is "
              "elsewhere.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dcp_checkpoint", required=True)
    parser.add_argument("--hf_checkpoint", required=True)
    parser.add_argument("--vocab_size", type=int, required=True)
    parser.add_argument("--dim", type=int, required=True)
    parser.add_argument("--dcp_key", default="tok_embeddings.weight")
    parser.add_argument("--hf_key", default="model.embed_tokens.weight")
    args = parser.parse_args()

    compare_embeddings(
        args.dcp_checkpoint, args.hf_checkpoint, args.vocab_size, args.dim,
        args.dcp_key, args.hf_key,
    )