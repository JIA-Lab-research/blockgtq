"""Block-GTQ quickstart (basic / unaccelerated).

Demonstrates the *method*: calibrates Block-GTQ, patches every layer's
``k_proj`` / ``v_proj`` to quantize-dequantize K/V on the fly (pure PyTorch,
no packed cache, no custom kernel), then prints a WikiText-2 perplexity
sanity check for the quantized cache next to the fp16 baseline — calibrating
on the Wikitext2 **train** split and scoring the held-out **test** split . For the
memory / latency wins on Hopper use ``quickstart_deployment.py`` instead.

Because the patch wraps the *pre-RoPE* ``k_proj`` output, calibration uses
``post_rope=False`` (the apply point and the calibrate point must match).

Run with the model cached locally (set ``HF_HOME``), e.g.::

    HF_HOME=/path/to/hf python examples/quickstart_basic.py \\
        --model Qwen/Qwen2.5-3B-Instruct \\
        --k-bits 3 --v-bits 3 --chunks 4
"""
import argparse
import math
import os
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from blockgtq import collect_qk_activations
from blockgtq.unaccelerated import (
    build_unaccelerated_quantizers,
    patch_model_kv,
    unpatch_model_kv,
)


@torch.no_grad()
def wikitext_ppl(model, ids, device, chunk_tokens=2048, n_chunks=4):
    """Teacher-forced WikiText-2 perplexity over ``n_chunks`` contiguous
    ``chunk_tokens``-length windows. Returns ``(ppl, n_tokens_scored)``."""
    total_nll, total_tok = 0.0, 0
    for i in range(n_chunks):
        s = i * chunk_tokens
        e = s + chunk_tokens + 1
        if e > ids.numel():
            break
        chunk = ids[s:e].unsqueeze(0).to(device)
        out = model(input_ids=chunk, labels=chunk)
        n = chunk.numel() - 1
        total_nll += out.loss.item() * n
        total_tok += n
    return math.exp(total_nll / total_tok), total_tok


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--k-bits", type=float, default=3.0,
                        help="Average K bits/dim (allocated non-uniformly).")
    parser.add_argument("--v-bits", type=int, default=3)
    parser.add_argument("--n-calib", type=int, default=2048,
                        help="Calibration tokens (2048 = paper canonical).")
    parser.add_argument("--calib-file", default=None,
                        help="Calibration text; defaults to bundled "
                             "calib_data/wikitext2_train_calib_2k.txt "
                             "(WT2-train; disjoint from the WT2-test PPL eval).")
    parser.add_argument("--chunk-tokens", type=int, default=2048)
    parser.add_argument("--chunks", type=int, default=4,
                        help="WikiText-2 test chunks scored for PPL.")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    device = torch.device(args.device)
    print(f"[load] {args.model}")
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float16, trust_remote_code=True,
    ).to(device).eval()

    cfg = model.config
    hd = cfg.hidden_size // cfg.num_attention_heads
    nkv = getattr(cfg, "num_key_value_heads", cfg.num_attention_heads)
    n_layers = len(model.model.layers)
    print(f"[load] {n_layers}L hd={hd} nkv={nkv}")

    # ── Calibration (pre-RoPE: the patch quantizes the pre-RoPE k_proj output) ──
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    calib_path = args.calib_file or os.path.join(
        repo_root, "calib_data", "wikitext2_train_calib_2k.txt")
    calib_ids = tok(open(calib_path).read(), return_tensors="pt").input_ids
    print(f"[calib] n_calib={args.n_calib} from {os.path.basename(calib_path)} "
          f"(post_rope=False)")
    t0 = time.time()
    calib_qk = collect_qk_activations(
        model, calib_ids, args.device,
        n_calib_tokens=min(args.n_calib, calib_ids.shape[1]),
        post_rope=False,
    )
    k_quantizers, v_quantizers = build_unaccelerated_quantizers(
        calib_qk, n_layers, nkv, hd,
        k_avg_bits=args.k_bits, v_bits=args.v_bits, device=args.device,
    )
    print(f"[calib] built per-head quantizers ({time.time() - t0:.1f}s)")

    # ── Evaluation corpus: WikiText-2 test ──
    from datasets import load_dataset
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    eval_ids = tok("\n\n".join(ds["text"]), return_tensors="pt").input_ids[0]

    # ── fp16 baseline ──
    print("[ppl] scoring fp16 baseline ...")
    ppl_fp16, ntok = wikitext_ppl(
        model, eval_ids, device, args.chunk_tokens, args.chunks)

    # ── Block-GTQ (patched) ──
    print(f"[ppl] scoring Block-GTQ K{args.k_bits:g}V{args.v_bits} ...")
    patch_handles = patch_model_kv(model, k_quantizers, v_quantizers, hd, nkv)
    try:
        ppl_bgt, _ = wikitext_ppl(
            model, eval_ids, device, args.chunk_tokens, args.chunks)
    finally:
        unpatch_model_kv(patch_handles)

    delta_pct = 100.0 * (ppl_bgt - ppl_fp16) / ppl_fp16
    print(f"\n[result] WikiText-2 PPL over {ntok} tokens "
          f"({args.chunks} x {args.chunk_tokens}):")
    print(f"  fp16 (uncompressed)        : {ppl_fp16:.3f}")
    print(f"  Block-GTQ K{args.k_bits:g}V{args.v_bits} (this method): "
          f"{ppl_bgt:.3f}  ({delta_pct:+.2f}% vs fp16)")


if __name__ == "__main__":
    main()
