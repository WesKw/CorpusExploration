"""
Diagnostic: is the model's raw forward() pass actually sensitive to its
input, the way generate() clearly is?

We already confirmed generate() produces coherent, context-appropriate
text. lm-eval's loglikelihood scoring (used for all multiple-choice
accuracy numbers) does NOT call generate() - it calls forward() directly
on context+continuation and reads off log-probabilities. If that code
path has a bug (stale cache state, position_ids issue, etc. left over
from generation-only testing), forward() could produce logits that don't
meaningfully depend on the input - which would explain identical-looking
random-baseline accuracy regardless of batching or BOS settings.

This script:
1. Runs forward() on two clearly different prompts and compares the
   resulting next-token distributions - they should differ a lot if
   the model is behaving correctly.
2. Replicates lm-eval's core loglikelihood computation manually on one
   simple example (a context + two candidate continuations) and prints
   which continuation the model scores higher - this should match
   whichever continuation is obviously more plausible to a human.
"""

import argparse

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


def load(model_path):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float32)
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    return model, tokenizer, device


def test_forward_sensitivity(model, tokenizer, device):
    print("=" * 70)
    print("TEST 1: Does forward() output change with different inputs?")
    print("=" * 70)

    prompt_a = "The capital of France is"
    prompt_b = "Bananas are typically colored"

    for prompt in (prompt_a, prompt_b):
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            logits = model(**inputs).logits
        next_logits = logits[0, -1, :]
        probs = F.softmax(next_logits, dim=-1)
        top3 = torch.topk(probs, 3)
        top3_toks = [tokenizer.decode([t]) for t in top3.indices.tolist()]
        print(f"\nPrompt: {prompt!r}")
        print(f"  Top-3 next tokens: {list(zip(top3_toks, top3.values.tolist()))}")

    print(
        "\nIf the top tokens above are similar/nonsensical for BOTH very "
        "different prompts, forward() is likely not conditioning on input "
        "correctly. If they differ sensibly (e.g. 'Paris' vs a color word), "
        "forward() is working fine and the bug is elsewhere in the eval "
        "harness integration (e.g. how it slices logprobs for scoring)."
    )


def loglikelihood(model, tokenizer, device, context: str, continuation: str):
    """
    Minimal reimplementation of lm-eval's core loglikelihood logic:
    - tokenize context and continuation separately
    - concatenate, forward pass once
    - sum the log-probabilities of the continuation tokens only
      (conditioned on everything before them)
    """
    context_ids = tokenizer(context, return_tensors="pt").input_ids[0]
    # IMPORTANT: match lm-eval's default of NOT using add_special_tokens
    # for the continuation piece, and concatenating at the token level
    full_ids = tokenizer(context + continuation, return_tensors="pt").input_ids[0]
    cont_len = len(full_ids) - len(context_ids)

    input_ids = full_ids.unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(input_ids=input_ids).logits[0]  # [seq_len, vocab]

    log_probs = F.log_softmax(logits, dim=-1)

    # continuation token i is predicted by logits at position (len(context_ids) + i - 1)
    total_logprob = 0.0
    for i in range(cont_len):
        pos = len(context_ids) + i - 1
        target_token = full_ids[len(context_ids) + i]
        total_logprob += log_probs[pos, target_token].item()

    return total_logprob


def test_manual_loglikelihood(model, tokenizer, device):
    print("\n" + "=" * 70)
    print("TEST 2: Manual loglikelihood scoring (mimics lm-eval's core logic)")
    print("=" * 70)

    context = "The sky on a clear day is usually"
    good_continuation = " blue."
    bad_continuation = " purple with orange stripes and singing."

    lp_good = loglikelihood(model, tokenizer, device, context, good_continuation)
    lp_bad = loglikelihood(model, tokenizer, device, context, bad_continuation)

    print(f"\nContext: {context!r}")
    print(f"  logprob({good_continuation!r}) = {lp_good:.3f}")
    print(f"  logprob({bad_continuation!r}) = {lp_bad:.3f}")

    if lp_good > lp_bad:
        print(
            "\nModel correctly scores the plausible continuation higher. "
            "The manual loglikelihood path works - if lm-eval's numbers "
            "still look random, the bug is likely in how the harness "
            "constructs/parses requests for YOUR specific tasks, not in "
            "basic loglikelihood scoring itself."
        )
    else:
        print(
            "\nWARNING: model scores the IMPLAUSIBLE continuation higher "
            "(or they're roughly equal). This points to a real bug in the "
            "forward()/loglikelihood code path itself - worth comparing "
            "this manual result against what lm-eval's internal "
            "_loglikelihood_tokens (or equivalent) computes for the exact "
            "same input, to find where the two implementations diverge."
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    args = parser.parse_args()

    model, tokenizer, device = load(args.model_path)
    test_forward_sensitivity(model, tokenizer, device)
    test_manual_loglikelihood(model, tokenizer, device)