"""WikiText-2 perplexity with Block-GTQ KV cache (skeleton).

This is a minimal evaluation harness scaffolding so external users can plug
Block-GTQ into a perplexity sweep without re-deriving the calibration logic.
The full WikiText-2 PPL evaluation used in the paper applies model surgery
to swap the attention forward, calibrates per-(layer, head), and accumulates
NLL over fixed-stride chunks; see the paper for the experimental details
this skeleton intentionally omits.

Usage::

    HF_HOME=/path/to/hf python examples/wikitext_ppl.py \\
        --model Qwen/Qwen2.5-3B-Instruct --k-bits 3 --v-bits 3 --chunks 4
"""
import argparse

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--k-bits", type=float, default=3.0)
    parser.add_argument("--v-bits", type=int, default=3)
    parser.add_argument("--chunk-tokens", type=int, default=2048)
    parser.add_argument("--chunks", type=int, default=8,
                        help="Number of contiguous WT2 chunks to score.")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    device = torch.device(args.device)
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float16,
    ).to(device).eval()

    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join(ds["text"])
    ids = tok(text, return_tensors="pt").input_ids[0].to(device)

    # Baseline fp16 PPL — a sanity number to compare against the Block-GTQ
    # result the paper reports. Wiring up Block-GTQ here requires the model
    # surgery of replacing the attention forward with one
    # that routes through `BlockGTQKVManager.append_and_attend`. That model-
    # surgery is intentionally out of scope for this skeleton; the paper
    # release will include an integration adapter for the supported model
    # families.
    total_nll, total_tok = 0.0, 0
    for i in range(args.chunks):
        s = i * args.chunk_tokens
        e = s + args.chunk_tokens + 1
        if e > ids.numel():
            break
        chunk = ids[s:e].unsqueeze(0)
        with torch.no_grad():
            out = model(input_ids=chunk, labels=chunk)
        n = chunk.numel() - 1
        total_nll += out.loss.item() * n
        total_tok += n
    print(f"[fp16 baseline] WT2 PPL ≈ {torch.exp(torch.tensor(total_nll/total_tok)).item():.3f}"
          f" over {total_tok} tokens")

    # TODO: hook BlockGTQKVManager into the attention forward and rerun.
    print("[Block-GTQ] not yet wired — see paper for model-surgery details.")


if __name__ == "__main__":
    main()
