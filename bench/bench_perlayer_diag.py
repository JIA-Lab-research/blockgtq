"""All-layer attention-kernel diagnostics on the model's REAL (normed) K.

Regenerates two paper artifacts on the correct post-`input_layernorm` K,
collected with forward HOOKS on the RoPE-feeding modules (so input_layernorm
and any QK-norm are applied by the real forward -- the same K distribution as
the calibration / deployment path). This replaces a legacy raw-residual
extraction (`proj(hidden_states[li])` with no input_layernorm).

  * Group 3 -- per-(layer, KV-head) RoPE frequency-block energy profiles,
    written as ``results/freq_energy_<tag>.json`` (the input for the
    bit-allocation fingerprint figures).
  * Group 2 -- per-layer kernel-MAE (KV-head 0): mean over 50 RoPE offsets of
    |s(q,k,d) - s(q,k_hat,d)| for TQ-MSE and Block-GTQ, with the 1024-token
    window split 50/50 into calibration / test halves (appendix protocol).

Usage::

    python bench/bench_perlayer_diag.py --model meta-llama/Llama-3.1-8B-Instruct \\
        --device cuda:0 --avg-bits 3 --window-M 1024 --out-tag Llama-3.1-8B
"""
import argparse
import gc
import json
import math
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from blockgtq.rope_utils import (
    decompose_freq_blocks, get_rope_frequencies, compute_rope_kernel,
)
from blockgtq.freq_analysis import compute_freq_importance_energy
from bench._niah_tasks import get_model_layers, get_attn_module

RESULTS_DIR = Path(os.environ.get(
    "BLOCKGTQ_RESULTS_DIR",
    str(Path(__file__).resolve().parent.parent / "results")))
RESULTS_DIR.mkdir(exist_ok=True, parents=True)


# ---------------------------------------------------------------------------
# Energy stats -> per-(layer, KV-head) RoPE frequency-block energy profiles
# (the input for the bit-allocation fingerprint figures).
# ---------------------------------------------------------------------------
def compute_energy_stats(block_energy):
    n = len(block_energy)
    e = block_energy.numpy() if isinstance(block_energy, torch.Tensor) else block_energy
    e = np.array(e, dtype=np.float64)
    sorted_idx = np.argsort(e)[::-1]
    sorted_e = e[sorted_idx]
    total = e.sum()
    if total < 1e-20:
        return {"max_energy": 0, "max_block_idx": 0, "energy_ratio": 1.0,
                "gini": 0.0, "top1_share": 0.0, "top3_share": 0.0,
                "entropy": 1.0, "rank_order": list(range(n))}
    if len(sorted_e) >= 2 and sorted_e[1] > 1e-20:
        ratio = float(sorted_e[0] / sorted_e[1])
    else:
        ratio = float("inf")
    sorted_asc = np.sort(e)
    cumsum = np.cumsum(sorted_asc)
    gini = 1 - 2 * cumsum.sum() / (n * total)
    top1 = float(sorted_e[0] / total)
    top3 = float(sorted_e[:min(3, n)].sum() / total)
    p = e / total
    p = p[p > 0]
    entropy = -np.sum(p * np.log(p)) / np.log(n)
    return {
        "max_energy": float(sorted_e[0]), "max_block_idx": int(sorted_idx[0]),
        "energy_ratio": ratio, "gini": float(gini),
        "top1_share": top1, "top3_share": top3, "entropy": float(entropy),
        "rank_order": sorted_idx.tolist(),
    }


# ---------------------------------------------------------------------------
# Hook-based per-(layer, KV-head) normed Q/K for ALL layers
# ---------------------------------------------------------------------------
def collect_all_layers(model, tokenizer, device, info, text,
                       max_tokens=2048, n_chunks=8, chunk_size=256):
    """Return ``{(li, h): {"q": (T*gqa, hd), "k": (T, hd)}}`` for every layer.

    ``text`` is the corpus the caller passes — WT2-train for calibration, or
    WT2-test for the MAE eval. Q is kept per KV head as all-|G(h)| query-head
    samples (interleaved t0g0,t0g1,...). Hooks capture the RoPE-feeding module
    output of the real forward, so input_layernorm + QK-norm are applied.
    """
    enc = tokenizer(text, return_tensors="pt", truncation=True,
                    max_length=max_tokens)["input_ids"]

    n_heads, n_kv = info["n_heads"], info["n_kv_heads"]
    hd, n_layers = info["head_dim"], info["n_layers"]
    is_mla, qk_rope_dim = info["is_mla"], info["qk_rope_dim"]
    config = info["config"]
    layers_list = get_model_layers(model)
    gqa = n_heads // n_kv if n_kv else 1

    captures = {li: {} for li in range(n_layers)}

    def _mk(li, kind):
        def hook(_m, _i, out):
            t = out[0] if isinstance(out, tuple) else out
            captures[li].setdefault(kind, []).append(t.detach().float())
        return hook

    hooks = []
    for li in range(n_layers):
        attn = get_attn_module(layers_list[li])
        if is_mla:
            hooks.append(attn.kv_a_proj_with_mqa.register_forward_hook(_mk(li, "k")))
            q_mod = getattr(attn, "q_proj", None) or getattr(attn, "q_b_proj")
            hooks.append(q_mod.register_forward_hook(_mk(li, "q")))
        elif hasattr(attn, "k_proj"):
            q_mod = (attn.q_norm if isinstance(getattr(attn, "q_norm", None), nn.Module)
                     else attn.q_proj)
            k_mod = (attn.k_norm if isinstance(getattr(attn, "k_norm", None), nn.Module)
                     else attn.k_proj)
            hooks.append(q_mod.register_forward_hook(_mk(li, "q")))
            hooks.append(k_mod.register_forward_hook(_mk(li, "k")))
        else:  # fused QKV (GLM-4)
            hooks.append(attn.query_key_value.register_forward_hook(_mk(li, "qkv")))

    try:
        with torch.no_grad():
            for ci in range(min(n_chunks, enc.size(1) // chunk_size)):
                s = ci * chunk_size
                model(enc[:, s:s + chunk_size].to(device))
    finally:
        for h in hooks:
            h.remove()

    data = {}
    for li in range(n_layers):
        cap = captures[li]
        if is_mla:
            nope = getattr(config, "qk_nope_head_dim", 0)
            per_q = nope + qk_rope_dim
            k = torch.cat([t[0, :, -qk_rope_dim:] for t in cap["k"]])
            q = torch.cat([
                t[0].reshape(-1, n_heads, per_q)[:, :, -qk_rope_dim:].reshape(-1, qk_rope_dim)
                for t in cap["q"]])
            data[(li, 0)] = {"q": q, "k": k}
            continue
        if "qkv" in cap:
            qd, kd = n_heads * hd, n_kv * hd
            Q = torch.cat([t[0, :, :qd].reshape(-1, n_kv, gqa, hd) for t in cap["qkv"]])
            K = torch.cat([t[0, :, qd:qd + kd].reshape(-1, n_kv, hd) for t in cap["qkv"]])
        else:
            Q = torch.cat([t[0].reshape(-1, n_kv, gqa, hd) for t in cap["q"]])
            K = torch.cat([t[0].reshape(-1, n_kv, hd) for t in cap["k"]])
        for h in range(n_kv):
            data[(li, h)] = {"q": Q[:, h].reshape(-1, hd), "k": K[:, h]}
    return data


# ---------------------------------------------------------------------------
# Group 2: per-layer kernel-MAE metric + quantizers
# ---------------------------------------------------------------------------
def measure_kernel_error(q, k, k_hat, freqs, M, device, n_delta=50, rotary_dim=None):
    n = k.shape[0]
    hd = k.shape[-1]
    rd = hd if rotary_dim is None else rotary_dim
    qb = decompose_freq_blocks(q[:, :rd], rd)
    kb = decompose_freq_blocks(k[:, :rd], rd)
    khb = decompose_freq_blocks(k_hat[:, :rd], rd)
    # Partial-RoPE non-rotary subspace (e.g. GLM-4): a static, offset-independent
    # dot-product term that the K quantization error also perturbs.
    if rd < hd:
        static = (q[:, rd:] * (k[:, rd:] - k_hat[:, rd:])).sum(-1)
    else:
        static = torch.zeros(n, device=q.device, dtype=q.dtype)
    deltas = torch.linspace(-M, M, n_delta, device=device)
    errs = []
    for d in deltas:
        orig = compute_rope_kernel(qb, kb, freqs.to(device), d.expand(n))
        comp = compute_rope_kernel(qb, khb, freqs.to(device), d.expand(n))
        errs.append(((orig - comp) + static).abs().mean().item())
    return float(np.mean(errs))


def _bgt_merge_quantize(pipeline, k_test, hd, rd, budget_bits):
    """Block-GTQ over the rotary subspace [:rd], but the bit group whose width
    equals ``budget_bits`` ALSO absorbs the (hd-rd) non-rotary dims into a single
    TQ-MSE transform (larger rotation -> better whitening). Other rotary bit
    groups are quantized exactly as Block-GTQ would (per-group TurboQuantMSE)."""
    from blockgtq.tq import TurboQuantMSE
    dev = k_test.device
    L = rd // 2                          # rotary blocks; block i -> dims (i, i+L)
    alloc = pipeline.bit_allocation      # (L,) bits per rotary block
    kf = k_test.float()
    kh = torch.empty_like(kf)
    nonrotary = list(range(rd, hd))
    seed = 100
    placed_nonrotary = False
    for bw in sorted({int(b) for b in alloc.tolist()}):
        dims = []
        for i in range(L):
            if int(alloc[i]) == bw:
                dims += [i, i + L]
        if bw == int(budget_bits):
            dims = dims + nonrotary       # fold non-rotary into the budget group
            placed_nonrotary = True
        idx = torch.tensor(dims, device=dev, dtype=torch.long)
        tq = TurboQuantMSE(d=len(dims), bit_width=bw, seed=seed, device=dev)
        kh[:, idx] = tq.compress_decompress(kf[:, idx])
        seed += 1
    if not placed_nonrotary:             # no rotary block landed on budget_bits
        idx = torch.tensor(nonrotary, device=dev, dtype=torch.long)
        tq = TurboQuantMSE(d=len(nonrotary), bit_width=int(budget_bits), seed=seed, device=dev)
        kh[:, idx] = tq.compress_decompress(kf[:, idx])
    return kh.to(k_test.dtype)


def _quantize_k(method, q_cal, k_cal, k_test, hd, bits, rope_base,
                variant="full", split_dim=None):
    """variant: 'full' (Block-GTQ over all hd dims = hd//2 blocks), 'split'
    (Block-GTQ on the rotary [:split_dim] + separate uniform TQ-MSE on the rest),
    or 'merge' (Block-GTQ on the rotary, non-rotary folded into the budget group)."""
    from blockgtq.tq import TurboQuantMSE
    if method == "tq":
        q = TurboQuantMSE(d=hd, bit_width=bits, seed=42, device=k_test.device)
        return q.compress_decompress(k_test.float()).to(k_test.dtype)
    if method == "bgt":
        from blockgtq.block_gtq_pipeline import BlockGTQPipeline
        rd = hd if (variant == "full" or split_dim is None) else split_dim
        q = BlockGTQPipeline(head_dim=rd, avg_bits=float(bits), rotation_threshold=1,
                             min_bits=1, max_bits=8, rope_base=rope_base)
        q.calibrate(q_cal[:, :rd], k_cal[:, :rd])
        if rd == hd:                                  # 'full'
            return q.compress_decompress(k_test.float()).to(k_test.dtype)
        if variant == "merge":
            return _bgt_merge_quantize(q, k_test, hd, rd, bits)
        kh = q.compress_decompress(k_test[:, :rd].float())   # 'split'
        tqn = TurboQuantMSE(d=hd - rd, bit_width=bits, seed=99, device=k_test.device)
        kh = torch.cat([kh, tqn.compress_decompress(k_test[:, rd:].float())], dim=-1)
        return kh.to(k_test.dtype)
    raise ValueError(method)


KERNEL_METHODS = [("tq", "TQ-MSE"), ("bgt", "Block-GTQ")]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--avg-bits", type=int, default=3)
    p.add_argument("--window-M", type=int, default=1024)
    p.add_argument("--rotary-dim", type=int, default=None,
                   help="Rotary subspace dim for partial-RoPE models (e.g. 64 for "
                        "GLM-4: head_dim=128 but only the first 64 dims are rotated). "
                        "Default None = full head_dim. Controls the MAE metric "
                        "(R_d on [:rotary_dim] + static dot on the rest) and the "
                        "Block-GTQ split point for the 'split'/'merge' variants.")
    p.add_argument("--bgt-variant", choices=["split", "merge", "full"], default=None,
                   help="How Block-GTQ quantizes a partial-RoPE head. 'full': over "
                        "all head_dim (hd//2 blocks). 'split': Block-GTQ on rotary "
                        "[:rotary_dim] + separate uniform TQ-MSE on the rest. 'merge': "
                        "Block-GTQ on rotary, non-rotary folded into the budget-bit "
                        "group. Default: split if --rotary-dim set, else full.")
    p.add_argument("--out-tag", default=None)
    args = p.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    tag = args.out_tag or args.model.split("/")[-1]
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float16, trust_remote_code=True,
        device_map={"": args.device}).eval()
    cfg = model.config
    n_heads = cfg.num_attention_heads
    n_kv = getattr(cfg, "num_key_value_heads",
                   getattr(cfg, "multi_query_group_num", n_heads))
    rope_base = getattr(cfg, "rope_theta", 10000.0)
    n_layers = getattr(cfg, "num_hidden_layers", getattr(cfg, "num_layers", None))
    qk_rope_dim = getattr(cfg, "qk_rope_head_dim", None)
    kv_lora = getattr(cfg, "kv_lora_rank", None)
    is_mla = (qk_rope_dim is not None and kv_lora is not None)
    if is_mla:
        hd = qk_rope_dim
    elif hasattr(cfg, "kv_channels"):
        hd = cfg.kv_channels
    elif getattr(cfg, "head_dim", None):
        hd = int(cfg.head_dim)
    else:
        hd = cfg.hidden_size // n_heads
    info = dict(n_heads=n_heads, n_kv_heads=n_kv, head_dim=hd, n_layers=n_layers,
                is_mla=is_mla, qk_rope_dim=qk_rope_dim, config=cfg)
    rotary_dim = args.rotary_dim or hd      # metric subspace + split point
    variant = args.bgt_variant or ("split" if args.rotary_dim else "full")
    # Block-GTQ allocates over alloc_dim//2 blocks: full head_dim for 'full',
    # else just the rotary subspace.
    alloc_dim = hd if variant == "full" else rotary_dim
    freqs = get_rope_frequencies(rotary_dim, rope_base)   # MAE metric (partial-RoPE)
    n_freqs = alloc_dim // 2
    print(f"[variant] bgt={variant} alloc_dim={alloc_dim} "
          f"metric_rotary_dim={rotary_dim} hd={hd}")

    repo_root = Path(__file__).resolve().parent.parent
    calib_text = (repo_root / "calib_data" / "wikitext2_train_calib_2k.txt").read_text()
    test_text = (repo_root / "calib_data" / "wikitext2_calib_2k.txt").read_text()
    t0 = time.time()
    # Calibration (bit allocation + Block-GTQ fit) on the first 2048 WT2-TRAIN
    # tokens; the per-layer MAE is scored on the disjoint first 2048 WT2-TEST
    # tokens (wikitext2_calib_2k.txt) — calib and eval splits do not overlap.
    calib_data = collect_all_layers(model, tok, args.device, info, calib_text, n_chunks=8)
    test_data = collect_all_layers(model, tok, args.device, info, test_text, n_chunks=8)
    print(f"[collect] calib(train)={len(calib_data)} test={len(test_data)} "
          f"combos, {time.time()-t0:.1f}s")

    # ---- Group 3: per-(layer,head) freq-block energy -> freq_energy json ----
    attn0 = get_attn_module(get_model_layers(model)[0])
    arch_type = ("mla" if is_mla
                 else "gqa" if hasattr(attn0, "k_proj")
                 else "fused_qkv")
    energy = {"model": args.model, "timestamp": ts,
              "arch_info": {"n_layers": n_layers, "n_heads": n_heads,
                            "n_kv_heads": n_kv, "head_dim": hd,
                            "rope_head_dim": alloc_dim, "rope_base": rope_base,
                            "n_freqs": n_freqs, "is_mla": is_mla,
                            "arch_type": arch_type, "bgt_variant": variant,
                            "metric_rotary_dim": rotary_dim},
              "extraction": "hook-based (normed: input_layernorm + QK-norm)",
              "layers": {}}
    all_e = []
    for li in range(n_layers):
        heads = sorted([h for (l, h) in calib_data if l == li])
        if not heads:
            continue
        ld, le = {}, []
        for h in heads:
            d = calib_data[(li, h)]
            qb = decompose_freq_blocks(d["q"][:, :alloc_dim], alloc_dim)
            kb = decompose_freq_blocks(d["k"][:, :alloc_dim], alloc_dim)
            be = compute_freq_importance_energy(qb, kb).cpu()
            ld[f"head_{h}"] = {"block_energies": be.tolist(),
                               "stats": compute_energy_stats(be)}
            le.append(be.numpy()); all_e.append(be.numpy())
        avg = np.mean(le, axis=0)
        ld["avg_across_heads"] = {"block_energies": avg.tolist(),
                                  "stats": compute_energy_stats(torch.from_numpy(avg))}
        energy["layers"][f"layer_{li}"] = ld
    if all_e:
        g = np.mean(all_e, axis=0)
        energy["global_summary"] = {"avg_block_energies": g.tolist(),
                                    "avg_stats": compute_energy_stats(torch.from_numpy(g))}
    ej = RESULTS_DIR / f"freq_energy_{tag}.json"
    ej.write_text(json.dumps(energy, indent=2))
    print(f"[group3] saved {ej}")

    # ---- Group 2: per-layer kernel-MAE (mean over ALL KV + query heads) ----
    #   Block-GTQ calibrated on WT2-train (per KV head); MAE scored on disjoint
    #   WT2-test. Per layer we average |q^T R_d k - q^T R_d k_hat| over every KV
    #   head h, every query head g in G(h), all test tokens, and the 50 offsets.
    #   (MLA: one shared KV head; GQA: all heads.) Query heads are batched: q row
    #   t*gqa+g pairs with the tiled k row for token t.
    M = args.window_M
    lines = [f"# Per-layer kernel-MAE (normed K) — {args.model}",
             f"# b={args.avg_bits} b/d, M={M}, ALL KV+query heads, calib=WT2-train test=WT2-test",
             "| layer | " + " | ".join(n for _, n in KERNEL_METHODS) + " | best/TQ Δ% |",
             "|---|" + "---|" * (len(KERNEL_METHODS) + 1)]
    rows = []
    for li in range(n_layers):
        heads = sorted([h for (l, h) in test_data if l == li and (li, h) in calib_data])
        if not heads:
            continue
        per_method = {key: [] for key, _ in KERNEL_METHODS}
        for h in heads:
            cd, td = calib_data[(li, h)], test_data[(li, h)]
            q_cal, k_cal = cd["q"], cd["k"]          # WT2-train (KV head h): all-|G(h)| Q + K
            k_test = td["k"]                          # WT2-test (KV head h)
            Tt = k_test.shape[0]
            if Tt < 8:
                continue
            gqa = td["q"].shape[0] // Tt if Tt else 1
            q_stack = td["q"][:Tt * gqa]                       # (Tt*gqa, hd): t0g0,t0g1,...
            k_stack = k_test.repeat_interleave(gqa, dim=0)     # (Tt*gqa, hd): k aligned per token
            for key, name in KERNEL_METHODS:
                try:
                    kh = _quantize_k(key, q_cal, k_cal, k_test, hd, args.avg_bits,
                                     rope_base, variant=variant, split_dim=rotary_dim)
                    kh_stack = kh.repeat_interleave(gqa, dim=0)
                    per_method[key].append(
                        measure_kernel_error(q_stack, k_stack, kh_stack, freqs, M,
                                             args.device, rotary_dim=rotary_dim))
                except Exception as ex:
                    print(f"  [L{li} h{h} {key}] {type(ex).__name__}: {ex}")
        errs = {key: (float(np.mean(v)) if v else float("nan"))
                for key, v in per_method.items()}
        tq_e = errs.get("tq", float("nan"))
        bgt_e = errs.get("bgt", float("nan"))
        dpct = (bgt_e - tq_e) / tq_e * 100 if tq_e else float("nan")
        lines.append(f"| {li} | " + " | ".join(f"{errs[k]:.4f}" for k, _ in KERNEL_METHODS)
                     + f" | {dpct:+.1f}% |")
        rows.append({"layer": li, **{k: errs[k] for k, _ in KERNEL_METHODS},
                     "bgt_vs_tq_pct": dpct})
    mt = RESULTS_DIR / f"perlayer_kernel_{tag}.md"
    mt.write_text("\n".join(lines))
    (RESULTS_DIR / f"perlayer_kernel_{tag}.json").write_text(
        json.dumps({"model": args.model, "bits": args.avg_bits, "M": M, "rows": rows}, indent=2))
    # quick summary: how many layers BGT < TQ
    valid = [r for r in rows if not math.isnan(r["bgt"]) and not math.isnan(r["tq"])]
    wins = sum(1 for r in valid if r["bgt"] < r["tq"])
    print(f"[group2] saved {mt}  | BGT < TQ on {wins}/{len(valid)} layers")

    del model
    gc.collect(); torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
