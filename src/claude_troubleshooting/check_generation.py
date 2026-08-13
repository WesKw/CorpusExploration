"""
Sanity-check a trained checkpoint by generating text directly, bypassing
lm-eval entirely. If output is garbage despite low training loss, the
problem is likely in training (e.g. a frozen/broken parameter group).
If output looks roughly coherent, the problem is likely downstream
(checkpoint conversion, tokenizer mismatch, or the eval harness config).

Usage:
    python check_generation.py --model_path /path/to/hf/model --prompt "The quick brown fox"

Also includes an optional norm-weight freeze check (same idea as the
bf16-master-weight bug, but useful as a general "is this param group
dead" diagnostic regardless of root cause).
"""

import argparse

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def generate(model_path: str, prompt: str, max_new_tokens: int = 80):
    print(f"Loading tokenizer from {model_path} ...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    print(f"Loading model from {model_path} ...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.float32
    )
    model.eval()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    print(f"\nPrompt: {prompt!r}\n")

    with torch.no_grad():
        # Greedy decode - most diagnostic, no sampling randomness to confuse things
        greedy_out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            num_beams=1,
            repetition_penalty=1.3,  # light guard against degenerate loops
            pad_token_id=tokenizer.eos_token_id,
        )
        greedy_text = tokenizer.decode(greedy_out[0], skip_special_tokens=True)

        # Also check raw next-token logits/probs on the prompt itself
        logits = model(**inputs).logits
        next_token_logits = logits[0, -1, :]
        next_token_probs = torch.softmax(next_token_logits, dim=-1)
        top5 = torch.topk(next_token_probs, 5)
        top5_tokens = [
            (tokenizer.decode([tok_id]), float(prob))
            for tok_id, prob in zip(top5.indices.tolist(), top5.values.tolist())
        ]

    print("=" * 70)
    print("GREEDY GENERATION (deterministic, most diagnostic):")
    print("=" * 70)
    print(greedy_text)
    print()
    print("=" * 70)
    print("TOP-5 NEXT-TOKEN PREDICTIONS after the prompt:")
    print("=" * 70)
    for tok, prob in top5_tokens:
        print(f"  {tok!r:20s}  p={prob:.4f}")
    print()

    # Rough automatic signal: if the top prediction has near-uniform
    # probability (~1/vocab_size), the model is essentially guessing.
    vocab_size = next_token_logits.shape[-1]
    uniform_prob = 1.0 / vocab_size
    top1_prob = top5.values[0].item()
    print(f"Vocab size: {vocab_size}")
    print(f"Uniform (random-guess) probability: {uniform_prob:.6f}")
    print(f"Actual top-1 probability: {top1_prob:.6f}")
    if top1_prob < uniform_prob * 5:
        print(
            "WARNING: top-1 probability is close to random-guess level. "
            "The model's next-token distribution looks close to uniform, "
            "consistent with a model that hasn't learned useful structure."
        )
    else:
        print(
            "Top-1 probability is well above random-guess level - the "
            "model IS producing a peaked, non-random distribution. If eval "
            "is still at baseline, look downstream (checkpoint conversion, "
            "tokenizer mismatch, or eval harness config) rather than at "
            "training itself."
        )


def check_norm_weights(dcp_checkpoint_path: str, dim: int, num_layers: int):
    """
    General-purpose check: are RMSNorm weights actually moving from their
    1.0 init, or are they stuck? This isn't only relevant to the bf16
    bug - any bug that zeros out gradients/updates for a specific param
    group (wrong param group in optimizer, weight decay misconfigured,
    a stray .requires_grad_(False), etc.) would show the same symptom.
    """
    import torch.distributed.checkpoint as dcp

    print(f"\nChecking norm weights in checkpoint: {dcp_checkpoint_path}")
    sd = {"norm.weight": torch.zeros(dim)}
    for i in range(num_layers):
        sd[f"layers.{i}.attention_norm.weight"] = torch.zeros(dim)
        sd[f"layers.{i}.ffn_norm.weight"] = torch.zeros(dim)

    dcp.load(sd, checkpoint_id=dcp_checkpoint_path)

    all_ones = True
    for name, tensor in sd.items():
        is_ones = torch.allclose(tensor, torch.ones_like(tensor))
        status = "STUCK AT 1.0" if is_ones else "moved from init"
        print(f"  {name:40s} {status}  (mean={tensor.mean().item():.4f}, std={tensor.std().item():.6f})")
        all_ones = all_ones and is_ones

    if all_ones:
        print("\nWARNING: ALL norm weights are stuck at 1.0 init. This "
              "param group is not learning, regardless of root cause.")
    else:
        print("\nNorm weights have moved from init - this specific failure mode is ruled out.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True, help="Path to HF-format converted checkpoint")
    parser.add_argument("--prompt", default="The quick brown fox jumps over the lazy dog. In other news,")
    parser.add_argument("--max_new_tokens", type=int, default=80)
    parser.add_argument("--check_norm", action="store_true", help="Also run the DCP-level norm-weight freeze check")
    parser.add_argument("--dcp_checkpoint_path", default=None, help="Path to raw DCP checkpoint (required if --check_norm)")
    parser.add_argument("--dim", type=int, default=768, help="Model hidden_size (for norm check)")
    parser.add_argument("--num_layers", type=int, default=8, help="Model num_hidden_layers (for norm check)")
    args = parser.parse_args()

    generate(args.model_path, args.prompt, args.max_new_tokens)

    if args.check_norm:
        if not args.dcp_checkpoint_path:
            raise ValueError("--dcp_checkpoint_path is required when using --check_norm")
        check_norm_weights(args.dcp_checkpoint_path, args.dim, args.num_layers)