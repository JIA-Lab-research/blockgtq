"""Block-GTQ quickstart (production / accelerated).

prefills and decodes through the fused kernel
It needs a Hopper-class GPU, to validate method quality on any device
(no custom kernel), see ``quickstart_basic.py``.
    python examples/quickstart_deployment.py \\
        --model Qwen/Qwen2.5-3B-Instruct \\
        --prompt "Block-GTQ is" \\
        --k-bits 3 --v-bits 3 \\
        --new-tokens 16
"""
import argparse
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from blockgtq import (
    BlockGTQProductionCache,
    bake_q_rotations,
    build_quantizers,
    collect_qk_activations,
    layer_major_prefill,
    production_decode_step,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--prompt",
                        default="Block-GTQ allocates KV-cache bits over RoPE frequency blocks.")
    parser.add_argument("--k-bits", type=float, default=3.0)
    parser.add_argument("--v-bits", type=int, default=3)
    parser.add_argument("--new-tokens", type=int, default=16)
    parser.add_argument("--n-calib", type=int, default=2048,
                        help="Tokens for per-(layer, head) calibration (2048 = paper canonical).")
    parser.add_argument("--calib-file", default=None,
                        help="Calibration text file; defaults to bundled "
                             "calib_data/wikitext2_calib_2k.txt (WikiText-2 train, the paper "
                             "calibration). Falls back to --prompt if absent.")
    parser.add_argument("--max-seq-len", type=int, default=4096)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--bake-q", action="store_true",
                        help="Bake the Q permutation into q_proj weights "
                             "(one-time amortisation; skips rotate-Q per "
                             "decode step).")
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
    nq = cfg.num_attention_heads
    n_layers = len(model.model.layers)
    print(f"[load] {n_layers}L hd={hd} nkv={nkv} nq={nq}")

    # ── Calibration: post-RoPE Q/K -> per-head quantizers (paper canonical: ──
    # first n-calib tokens of WikiText-2 train; falls back to --prompt if absent).
    import os
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    calib_path = args.calib_file or os.path.join(repo_root, "calib_data", "wikitext2_calib_2k.txt")
    if os.path.exists(calib_path):
        calib_ids = tok(open(calib_path).read(), return_tensors="pt").input_ids
        print(f"[calib] n_calib_tokens={args.n_calib} from {os.path.basename(calib_path)}")
    else:
        calib_ids = tok(args.prompt * 8, return_tensors="pt").input_ids
        print(f"[calib] {calib_path} absent; using --prompt ({calib_ids.shape[1]} tok)")
    t0 = time.time()
    layer_data = collect_qk_activations(
        model, calib_ids, args.device,
        n_calib_tokens=min(args.n_calib, calib_ids.shape[1]),
        post_rope=True,
    )
    quantizers, ba, ka = build_quantizers(
        layer_data, n_layers, nkv, hd,
        k_avg_bits=args.k_bits, v_bits=args.v_bits,
        k_nibble=True, device=args.device,
    )
    if args.bake_q:
        bake_q_rotations(model, quantizers, n_layers, nkv, nq)
    print(f"[calib] done ({time.time() - t0:.1f}s)")

    # ── Build production cache ──
    prompt_ids = tok(args.prompt, return_tensors="pt").input_ids.to(device)
    prompt_len = prompt_ids.shape[1]
    cache = BlockGTQProductionCache(
        quantizers, ba, ka,
        n_layers=n_layers, nkv=nkv, nq=nq, hd=hd,
        max_T=args.max_seq_len,
        v_bits=args.v_bits, k_nibble=True, v_nibble=True,
        device=device,
    )

    # ── Layer-major prefill (one fused kernel call per layer) ──
    print(f"[prefill] T={prompt_len} via layer_major_prefill")
    t0 = time.time()
    layer_major_prefill(model, cache, prompt_ids)
    torch.cuda.synchronize()
    print(f"[prefill] done in {(time.time() - t0)*1000:.1f} ms; "
          f"cache memory = {cache.cache_memory_bytes()/1e6:.2f} MB "
          f"(vs {cache.fp16_cache_bytes()/1e6:.2f} MB fp16)")

    # ── Decode ──
    print(f"[decode] generating {args.new_tokens} tokens with "
          f"k_nibble=True, v_nibble=True ...")
    generated = []
    next_tok = prompt_ids[:, -1:].to(device)
    next_pos = torch.tensor([[prompt_len]], device=device)
    times_ms = []
    for i in range(args.new_tokens):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        logits = production_decode_step(model, next_tok, next_pos, cache)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        times_ms.append((t1 - t0) * 1000)

        next_id = logits[:, -1].argmax(dim=-1)
        generated.append(int(next_id.item()))
        next_tok = next_id.view(1, 1)
        next_pos = next_pos + 1

    # First step pays the CUDA-Graph build cost (~one-time per layer).
    warm_ms = sum(times_ms[1:]) / max(1, len(times_ms) - 1)
    print(f"[decode] step0={times_ms[0]:.2f} ms (graph build), "
          f"steady-state mean={warm_ms:.2f} ms / token")
    print("[decode] continuation:", tok.decode(generated, skip_special_tokens=True))


if __name__ == "__main__":
    main()
