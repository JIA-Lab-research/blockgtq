#!/usr/bin/env python3
"""Production E2E decode latency & memory at long context (up to T=1M).

Successor to `bench_production_e2e.py`. Same metrics (decode-step latency,
KV cache memory, peak GPU memory) but uses fast prefill for all methods,
making T={512K, 1M} feasible (the original used token-by-token prefill,
which would take hours at those scales).

Three methods, three prefill paths:

  fp16 + FlashAttention-2:
    Prefill: single HF forward (model() with attn_implementation='flash_attention_2'
             and use_cache=True), past_key_values copied into FP16KVCache.
    Decode:  fp16_decode_step (standard SDPA on cached fp16 KV).

  K3V3 (Block-GTQ, non-uniform K bit allocation):
    Prefill: layer_major_prefill (one fused_packed_attention_prefill call per
             layer, full T at once).
    Decode:  production_decode_step (CUDA-graphed compressed-cache decode).
    Calibrated with min_bits=1, max_bits=8 → per-frequency-block allocation.

  TQ_MSE (uniform K bit allocation, same kernel path):
    Same prefill + decode kernels as K3V3. Calibrated with
    min_bits=max_bits=k_bits to force uniform allocation across all
    frequency blocks. This is the TurboQuant-MSE equivalent within our
    BlockGTQ infrastructure (effectively a single bit-width group, single
    rotation matrix).

Outputs per T: median/mean/p5/p95 ms/step (decode), KV cache size,
peak GPU memory, speedup ratios, OOM markers.

Usage:
  CC=gcc python experiments/bench_decode_e2e_long_t.py \\
      --device cuda:0 \\
      --t-points 4096 16384 65536 131072 262144 524288 1048576 \\
      --n-bench 20
"""
import sys, os, time, math, json, argparse, gc
from pathlib import Path
from datetime import datetime

import torch
import torch.nn.functional as F
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from blockgtq import (
    BlockGTQProductionCache,
    FP16KVCache,
    production_decode_step,
    fp16_decode_step,
    collect_qk_activations,
    layer_major_prefill,
)
from blockgtq.quantizer import BlockGTQQuantizer, build_batched_encode_args

RESULTS_DIR = Path(os.environ.get("BLOCKGTQ_RESULTS_DIR", "./bench_results"))
RESULTS_DIR.mkdir(exist_ok=True, parents=True)


# ────────────────────────────────────────────────────────────────────
# Token loading
# ────────────────────────────────────────────────────────────────────

def get_tokens(tokenizer, max_tokens):
    """Load WikiText-2 train set for the long timing / PPL stream. Block-GTQ /
    TQ-MSE calibrate on the first ``--n-calib`` tokens of THIS same stream
    (the leakage-free n=1000 PPL protocol)."""
    from datasets import load_dataset
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    text = "\n".join([t for t in ds["text"] if len(t) > 50])
    enc = tokenizer(text, return_tensors="pt", truncation=False)["input_ids"]
    n = enc.size(1)
    print(f"  WikiText-2 train (timing stream): {n} tokens")
    if n < max_tokens:
        repeats = (max_tokens // n) + 1
        enc = enc.repeat(1, repeats)
        print(f"  Tiled {repeats}× → {enc.size(1)} tokens")
    return enc[:, :max_tokens]


# ────────────────────────────────────────────────────────────────────
# Fast prefill helpers
# ────────────────────────────────────────────────────────────────────

@torch.no_grad()
def fp16_prefill_fast(model, fp16_cache, prompt_ids, device):
    """One-shot HF forward with FA-2; copy past_key_values → FP16KVCache."""
    T = prompt_ids.shape[1]
    pids = prompt_ids.to(device)
    try:
        out = model(pids, use_cache=True,
                    attn_implementation='flash_attention_2')
    except TypeError:
        out = model(pids, use_cache=True)
    pkv = out.past_key_values

    if hasattr(pkv, 'key_cache'):  # transformers DynamicCache
        ks, vs = pkv.key_cache, pkv.value_cache
    else:
        ks = [layer[0] for layer in pkv]
        vs = [layer[1] for layer in pkv]

    n_layers = len(ks)
    for li in range(n_layers):
        k = ks[li].to(torch.float16)  # (B=1, n_kv, T, hd)
        v = vs[li].to(torch.float16)
        fp16_cache.k_cache[li, :, :, :T] = k
        fp16_cache.v_cache[li, :, :, :T] = v
    fp16_cache.seq_len = T

    del pkv, out, ks, vs
    torch.cuda.empty_cache()


@torch.no_grad()
def k3v3_prefill_fast(model, prod_cache, prompt_ids, device,
                      block_q=32, block_t=128):
    """Layer-major prefill (one big kernel call per layer)."""
    layer_major_prefill(model, prod_cache, prompt_ids.to(device),
                        block_q=block_q, block_t=block_t)


# ────────────────────────────────────────────────────────────────────
# Decode benchmark helpers
# ────────────────────────────────────────────────────────────────────

@torch.no_grad()
def benchmark_decode_steps(model, cache, decode_fn, enc, device,
                           n_warmup=5, n_bench=20, ppl_tokens=0):
    """Time n_bench decode steps after n_warmup. Cache is populated to T_start.
    Steps consume tokens sequentially from enc starting at cache.seq_len.

    If ppl_tokens > 0, after timing also runs `ppl_tokens` more decode steps
    and computes PPL by comparing logits to the next-token ground truth.
    Returns dict with timing + optional 'ppl' field.
    """
    T_start = cache.seq_len
    times_ms = []

    for i in range(n_warmup + n_bench):
        idx = T_start + i
        if idx >= enc.size(1) - 1:
            return None
        token = enc[:, idx:idx+1].to(device)
        pos = torch.tensor([[idx]], device=device)

        torch.cuda.synchronize()
        t0 = time.perf_counter()
        decode_fn(model, token, pos, cache)
        torch.cuda.synchronize()
        t1 = time.perf_counter()

        if i >= n_warmup:
            times_ms.append((t1 - t0) * 1000)

    if not times_ms:
        return None

    # Optional PPL evaluation (after timing, so it doesn't affect latency)
    ppl = None
    n_ppl = 0
    if ppl_tokens > 0:
        nlls = []
        T_after = cache.seq_len
        for i in range(ppl_tokens):
            idx = T_after + i
            if idx >= enc.size(1) - 1:
                break
            token = enc[:, idx:idx+1].to(device)
            pos = torch.tensor([[idx]], device=device)
            logits = decode_fn(model, token, pos, cache)
            target = enc[0, idx + 1].to(device)
            nll = F.cross_entropy(logits[0, 0], target).item()
            nlls.append(nll)
        n_ppl = len(nlls)
        if n_ppl > 0:
            ppl = math.exp(sum(nlls) / n_ppl)

    arr = np.array(times_ms)
    out = {
        'median_ms': float(np.median(arr)),
        'mean_ms': float(np.mean(arr)),
        'p5_ms': float(np.percentile(arr, 5)),
        'p95_ms': float(np.percentile(arr, 95)),
    }
    if ppl is not None:
        out['ppl'] = ppl
        out['n_ppl_tokens'] = n_ppl
    return out


def kv_memory_bytes_prod(cache, T):
    """K3V3 KV cache analytical bytes at T tokens."""
    nl, nkv = cache.n_layers, cache.nkv
    k = T * cache.k_bytes_per_tok * nkv * nl
    kn = T * cache.max_ng * 2 * nkv * nl
    v = T * cache.vpb * nkv * nl
    vn = T * 2 * nkv * nl
    return k + kn + v + vn


def kv_memory_bytes_fp16(n_layers, nkv, hd, T):
    return n_layers * nkv * T * hd * 2 * 2  # K + V, fp16


# ────────────────────────────────────────────────────────────────────
# Per-T cell: setup + bench + memory
# ────────────────────────────────────────────────────────────────────

@torch.no_grad()
def run_fp16_cell(model, T, prompt_ids, device, n_layers, nkv, nq, hd,
                  n_warmup, n_bench, model_mem, ppl_tokens=0):
    """Run one T cell for fp16 path. Returns dict with timing + memory."""
    torch.cuda.empty_cache()
    gc.collect()
    torch.cuda.reset_peak_memory_stats(device)

    try:
        fp16_cache = FP16KVCache(n_layers, nkv, nq, hd,
                                  max_T=T + n_warmup + n_bench + ppl_tokens + 10,
                                  device=device)
        t0 = time.time()
        fp16_prefill_fast(model, fp16_cache, prompt_ids[:, :T], device)
        prefill_time = time.time() - t0
        peak_after_prefill = torch.cuda.max_memory_allocated(device)

        timing = benchmark_decode_steps(model, fp16_cache, fp16_decode_step,
                                         prompt_ids, device,
                                         n_warmup=n_warmup, n_bench=n_bench,
                                         ppl_tokens=ppl_tokens)
        peak_after_decode = torch.cuda.max_memory_allocated(device)

        del fp16_cache
        torch.cuda.empty_cache()
        gc.collect()

        if timing is None:
            return {'T': T, 'method': 'fp16_fa2', 'status': 'no_tokens'}

        return {
            'T': T, 'method': 'fp16_fa2', 'status': 'ok',
            **timing,
            'prefill_s': prefill_time,
            'kv_cache_mb': kv_memory_bytes_fp16(n_layers, nkv, hd, T) / 1e6,
            'peak_gpu_gb': peak_after_decode / 1e9,
            'peak_no_model_gb': (peak_after_decode - model_mem) / 1e9,
        }
    except torch.cuda.OutOfMemoryError as e:
        torch.cuda.empty_cache()
        gc.collect()
        return {
            'T': T, 'method': 'fp16_fa2', 'status': 'oom',
            'kv_cache_mb': kv_memory_bytes_fp16(n_layers, nkv, hd, T) / 1e6,
            'oom_msg': str(e)[:120],
        }


@torch.no_grad()
def run_compressed_cell(model, T, prompt_ids, device,
                         quantizers, ba, ka, n_layers, nkv, nq, hd,
                         v_bits, n_warmup, n_bench, model_mem,
                         method_name, block_q=32, block_t=128,
                         ppl_tokens=0):
    """Run one T cell for any compressed-KV path (K3V3 or TQ_MSE).

    Both methods share the same kernels and prefill driver — they differ
    only in the calibrated quantizers (passed in via `quantizers`/`ba`/`ka`).
    """
    torch.cuda.empty_cache()
    gc.collect()
    torch.cuda.reset_peak_memory_stats(device)

    try:
        prod_cache = BlockGTQProductionCache(
            quantizers, ba, ka, n_layers, nkv, nq, hd,
            max_T=T + n_warmup + n_bench + ppl_tokens + 10,
            v_bits=v_bits, k_nibble=True, v_nibble=True,
            device=device)

        t0 = time.time()
        k3v3_prefill_fast(model, prod_cache, prompt_ids[:, :T], device,
                          block_q=block_q, block_t=block_t)
        prefill_time = time.time() - t0
        peak_after_prefill = torch.cuda.max_memory_allocated(device)

        timing = benchmark_decode_steps(model, prod_cache, production_decode_step,
                                         prompt_ids, device,
                                         n_warmup=n_warmup, n_bench=n_bench,
                                         ppl_tokens=ppl_tokens)
        peak_after_decode = torch.cuda.max_memory_allocated(device)

        kv_mb = kv_memory_bytes_prod(prod_cache, T) / 1e6
        kv_fp16_mb = kv_memory_bytes_fp16(n_layers, nkv, hd, T) / 1e6

        del prod_cache
        torch.cuda.empty_cache()
        gc.collect()

        if timing is None:
            return {'T': T, 'method': method_name, 'status': 'no_tokens'}

        return {
            'T': T, 'method': method_name, 'status': 'ok',
            **timing,
            'prefill_s': prefill_time,
            'kv_cache_mb': kv_mb,
            'kv_fp16_mb': kv_fp16_mb,
            'compression': kv_fp16_mb / kv_mb if kv_mb > 0 else 1.0,
            'peak_gpu_gb': peak_after_decode / 1e9,
            'peak_no_model_gb': (peak_after_decode - model_mem) / 1e9,
        }
    except torch.cuda.OutOfMemoryError as e:
        torch.cuda.empty_cache()
        gc.collect()
        return {
            'T': T, 'method': method_name, 'status': 'oom',
            'kv_cache_mb': kv_memory_bytes_fp16(n_layers, nkv, hd, T) / 1e6,
            'oom_msg': str(e)[:120],
        }


# ────────────────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Long-context E2E decode latency + memory benchmark "
                    "(layer-major prefill, fp16 FA-2 vs K3V3).")
    parser.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--t-points", type=int, nargs='+',
                        default=[4096, 16384, 65536, 131072, 262144,
                                  524288, 1048576])
    parser.add_argument("--n-calib", type=int, default=64)
    parser.add_argument("--configs", nargs='+', default=["K3V3"],
                        help="K/V bit configs to bench, e.g. K3V3 K3V2 K2V2")
    parser.add_argument("--n-warmup", type=int, default=5)
    parser.add_argument("--n-bench", type=int, default=2,
                        help="Decode steps used for latency timing AFTER "
                             "warmup. The PPL window starts at "
                             "T + n_warmup + n_bench; n_bench=2 keeps the "
                             "PPL window close to T (matches the paper "
                             "n=1000 protocol's enc[T+7:T+1007] indexing). "
                             "Raise to ~20 for tighter latency stats when "
                             "PPL alignment is not needed.")
    parser.add_argument("--block-q", type=int, default=32)
    parser.add_argument("--block-t", type=int, default=128)
    parser.add_argument("--skip-fp16", action='store_true',
                        help="Skip fp16 path (e.g., if known OOM at large T)")
    parser.add_argument("--skip-gtq", action='store_true',
                        help="Skip GTQ path (Block-GTQ, non-uniform K bits)")
    parser.add_argument("--skip-tq-mse", action='store_true',
                        help="Skip TQ_MSE path (uniform K bits, same kernel)")
    parser.add_argument("--prewarm-kernel", action='store_true',
                        help="Run small prefills once before the sweep so "
                             "T-sweep results are not contaminated by JIT.")
    parser.add_argument("--prewarm-ts", type=int, nargs='+', default=[512, 4096],
                        help="List of T values to prewarm at (default: 512 4096). "
                             "Multiple T's catch any T-band-specific JIT path.")
    parser.add_argument("--measure-ppl", action='store_true',
                        help="After decode timing, also compute PPL over "
                             "--ppl-tokens additional decode steps.")
    parser.add_argument("--ppl-tokens", type=int, default=1000,
                        help="Number of held-out tokens for the PPL eval "
                             "(only with --measure-ppl). PPL standard "
                             "error scales like 1/sqrt(n); n=1000 keeps "
                             "the relative SE around ~3%% — the value "
                             "used for the paper's long-context PPL "
                             "table. Drop to a few hundred for fast dev "
                             "checks, raise above 1000 to tighten further.")
    args = parser.parse_args()
    ppl_tokens = args.ppl_tokens if args.measure_ppl else 0

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    extra = args.n_warmup + args.n_bench + 50 + (args.ppl_tokens if args.measure_ppl else 0)
    max_t = max(args.t_points)

    # ── Load model ──
    from transformers import AutoModelForCausalLM, AutoTokenizer
    print(f"Loading {args.model} ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    # Try to load with FA-2 attention; fall back if not supported
    try:
        model = AutoModelForCausalLM.from_pretrained(
            args.model, torch_dtype=torch.float16, trust_remote_code=True,
            attn_implementation='flash_attention_2',
        ).to(args.device).eval()
        print("  Loaded with FlashAttention-2")
    except Exception:
        model = AutoModelForCausalLM.from_pretrained(
            args.model, torch_dtype=torch.float16, trust_remote_code=True,
        ).to(args.device).eval()
        print("  Loaded with default attention (FA-2 not specifiable)")

    cfg = model.config
    hd = cfg.hidden_size // cfg.num_attention_heads
    nkv = getattr(cfg, 'num_key_value_heads', cfg.num_attention_heads)
    nq = cfg.num_attention_heads
    n_layers = len(model.model.layers)
    print(f"  {n_layers}L hd={hd} nkv={nkv} nq={nq}")

    model_mem = torch.cuda.max_memory_allocated(args.device)
    print(f"  Model memory: {model_mem/1e9:.2f} GB")

    # ── Load tokens ──
    enc = get_tokens(tokenizer, max_t + extra)
    print(f"  Using {enc.size(1)} tokens")

    # ── Parse configs (e.g. "K3V3" → (k_bits=3, v_bits=3)) ──
    def _parse_cfg(cfg_str):
        # "K3V3" -> (3, 3); "K2V2" -> (2, 2); "K3V2" -> (3, 2)
        s = cfg_str.upper().lstrip('K')
        kpart, vpart = s.split('V')
        return int(kpart), int(vpart)

    config_specs = [(_parse_cfg(c), c) for c in args.configs]

    # ── Shared calibration data (one collection, reused across configs) ──
    # Block-GTQ / TQ-MSE calibrate on the first --n-calib tokens of the same
    # WT2-train stream used for the timing / PPL eval (`enc`). Leakage-free: the
    # n=1000 PPL window sits at [T, T+1000) with T >= 4096, far past the first
    # --n-calib calibration tokens. (Causal attention makes the first --n-calib
    # activations identical whether collected from a 64- or 4096-token forward,
    # so slicing `enc` here is equivalent to collecting from a 4096-token
    # forward and taking the first n_calib.) Calibration-source
    # sensitivity is studied in the calibration ablations, not exposed here.
    rb = getattr(cfg, 'rope_theta', 1e6)
    layer_data = None
    if not (args.skip_gtq and args.skip_tq_mse):
        print(f"\nCollecting QK activations for calibration "
              f"(first {args.n_calib} tokens of the WT2-train stream) ...")
        t_cd0 = time.time()
        layer_data = collect_qk_activations(
            model, enc[:, :args.n_calib], args.device,
            n_calib_tokens=args.n_calib, post_rope=False,
        )
        print(f"  Done ({time.time() - t_cd0:.1f}s)")

    def _calibrate_set(k_bits, v_bits, min_bits, max_bits, label):
        """Build (quantizers, ba, ka) for a given (k_bits, v_bits, allocation)."""
        # BlockGTQQuantizer + build_batched_encode_args are already imported
        # at module scope. Local re-import kept for clarity.
        print(f"\nCalibrating {label} ({n_layers}L × {nkv} heads, "
              f"k_bits={k_bits}, v_bits={v_bits}, "
              f"min_bits={min_bits}, max_bits={max_bits}) ...")
        t_cal0 = time.time()
        quantizers = [[None] * nkv for _ in range(n_layers)]
        ka = [[None] * nkv for _ in range(n_layers)]
        ba = [None] * n_layers
        # GQA: layer_data[li]['q'] has axis-1 = gqa·T (every Q head in
        # the group treated as an independent sample, matching the
        # mean-of-squared-norms convention in `collect_qk_activations`).
        # layer_data[li]['k'] has axis-1 = T. Use both fully — don't cap
        # to `args.n_calib` here, since `n_calib_tokens` already capped
        # the upstream collection.
        gqa = nq // nkv
        for li in range(n_layers):
            q_all = layer_data[li]['q']
            k_all = layer_data[li]['k']
            nc_k = k_all.shape[1]
            nc_q = q_all.shape[1]
            assert nc_q == gqa * nc_k, (
                f"layer {li}: q axis-1 ({nc_q}) != gqa·T ({gqa}·{nc_k}); "
                f"check collect_qk_activations.")
            for hi in range(nkv):
                q8 = BlockGTQQuantizer(
                    head_dim=hd, avg_bits=float(k_bits),
                    rotation_threshold=1,
                    min_bits=min_bits, max_bits=max_bits,
                    rope_base=rb, device=args.device)
                q8.calibrate(q_all[hi, :nc_q].float().contiguous(),
                             k_all[hi, :nc_k].float().contiguous())
                q8.init_v_quantizer(v_bits=v_bits, seed=42 + hi)
                quantizers[li][hi] = q8
                ka[li][hi] = q8.build_kernel_args()
            ba[li] = build_batched_encode_args(quantizers[li], device=args.device)
        print(f"  {label} done ({time.time() - t_cal0:.1f}s)")
        return quantizers, ba, ka

    # ── Build quantiser sets for every (config, method) combination ──
    # `cfg_data[cfg_str]` = {'gtq': (q,ba,ka), 'tqm': (q,ba,ka), 'k_bits': k, 'v_bits': v}
    cfg_data = {}
    for (k_bits, v_bits), cfg_str in config_specs:
        d = {'k_bits': k_bits, 'v_bits': v_bits, 'gtq': None, 'tqm': None}
        if not args.skip_gtq:
            d['gtq'] = _calibrate_set(
                k_bits, v_bits, min_bits=1, max_bits=8,
                label=f"{cfg_str}_GTQ")
        if not args.skip_tq_mse:
            d['tqm'] = _calibrate_set(
                k_bits, v_bits, min_bits=k_bits, max_bits=k_bits,
                label=f"{cfg_str}_TQM")
        cfg_data[cfg_str] = d

    if layer_data is not None:
        del layer_data
        torch.cuda.empty_cache()
        gc.collect()

    # ── Optional kernel prewarm: run small prefills before the sweep so
    #    T-sweep timings are not contaminated by JIT.  We prewarm at MULTIPLE
    #    T values (default 512 + 4096) because some kernel binaries get
    #    cached lazily on the first call within a fresh BlockGTQProductionCache —
    #    a single small T sometimes does not cover all paths used at later
    #    T values.  We prewarm BOTH GTQ and TQM quantiser sets per config
    #    (they have different constexpr signatures even at the same K bits).
    if args.prewarm_kernel and not (args.skip_gtq and args.skip_tq_mse):
        for cfg_str, d in cfg_data.items():
            for variant in ('gtq', 'tqm'):
                quant_set = d.get(variant)
                if quant_set is None:
                    continue
                for pw_t in args.prewarm_ts:
                    if pw_t > enc.size(1) - 50:
                        continue
                    print(f"\nPrewarming {cfg_str} {variant.upper()} kernel "
                          f"at T={pw_t} ...")
                    t_pw0 = time.time()
                    warm_cache = BlockGTQProductionCache(
                        quant_set[0], quant_set[1], quant_set[2],
                        n_layers, nkv, nq, hd,
                        max_T=pw_t + 50,
                        v_bits=d['v_bits'], k_nibble=True, v_nibble=True,
                        device=args.device)
                    k3v3_prefill_fast(model, warm_cache,
                                      enc[:, :pw_t], args.device,
                                      block_q=args.block_q, block_t=args.block_t)
                    del warm_cache
                    torch.cuda.empty_cache()
                    gc.collect()
                    print(f"  {cfg_str} {variant.upper()} T={pw_t} done "
                          f"({time.time() - t_pw0:.1f}s)")

    # ── Run sweep: outer loop over configs, inner over T.
    #    fp16 results are cached per-T (config-independent).
    all_results = []
    json_path = RESULTS_DIR / f"bench_decode_e2e_long_t_{ts}.json"
    fp16_cache_by_t = {}

    def _ms(res):
        if res.get('status') == 'ok':
            return f"{res.get('median_ms', float('nan')):.1f}"
        st = res.get('status', '—')
        return st if st in ('oom', 'skipped', 'no_tokens', '—') else '—'

    def _pk(res):
        return f"{res.get('peak_gpu_gb', float('nan')):.1f}GB" if res.get('status') == 'ok' else "—"

    def _pf(res):
        if res.get('status') == 'ok':
            v = res.get('prefill_s', float('nan'))
            return f"{v:.1f}s" if v < 100 else f"{v:.0f}s"
        return "—"

    def _ppl(res):
        if res.get('status') == 'ok' and 'ppl' in res:
            return f"{res['ppl']:.2f}"
        return "—"

    for cfg_str, d in cfg_data.items():
        k_bits = d['k_bits']
        v_bits = d['v_bits']
        print()
        print(f"{'═'*120}")
        print(f"  Config {cfg_str}  (K={k_bits}-bit, V={v_bits}-bit)")
        print(f"{'═'*120}")
        hdr_extra = f"{'ppl':>6} " if args.measure_ppl else ""
        print(f"{'T':>8} | {'fp16 pf':>7} {'fp16 ms':>8} {'fp16 pk':>8} {hdr_extra}|"
              f" {'GTQ pf':>7} {'GTQ ms':>7} {'GTQ pk':>7} {hdr_extra}{'sp_F':>5} |"
              f" {'TQM pf':>7} {'TQM ms':>7} {'TQM pk':>7} {hdr_extra}{'sp_F':>5} |"
              f" {'TQM/GTQ':>8}")
        print("-" * 130)

        for T in args.t_points:
            prompt = enc[:, :T + extra]
            if prompt.size(1) < T + extra:
                print(f"  T={T}: not enough tokens, breaking")
                break

            # ── fp16 (cached across configs) ──
            if T in fp16_cache_by_t:
                fp16_res = fp16_cache_by_t[T]
            elif args.skip_fp16:
                fp16_res = {'T': T, 'method': 'fp16_fa2', 'status': 'skipped'}
                fp16_cache_by_t[T] = fp16_res
            else:
                fp16_res = run_fp16_cell(
                    model, T, prompt, args.device, n_layers, nkv, nq, hd,
                    args.n_warmup, args.n_bench, model_mem,
                    ppl_tokens=ppl_tokens)
                fp16_cache_by_t[T] = fp16_res
                all_results.append(fp16_res)

            # ── GTQ ──
            if d['gtq'] is not None:
                q, ba, ka = d['gtq']
                gtq_res = run_compressed_cell(
                    model, T, prompt, args.device,
                    q, ba, ka, n_layers, nkv, nq, hd, v_bits,
                    args.n_warmup, args.n_bench, model_mem,
                    method_name=f'{cfg_str}_GTQ',
                    block_q=args.block_q, block_t=args.block_t,
                    ppl_tokens=ppl_tokens)
            else:
                gtq_res = {'T': T, 'method': f'{cfg_str}_GTQ', 'status': 'skipped'}
            all_results.append(gtq_res)

            # ── TQM ──
            if d['tqm'] is not None:
                q, ba, ka = d['tqm']
                tqm_res = run_compressed_cell(
                    model, T, prompt, args.device,
                    q, ba, ka, n_layers, nkv, nq, hd, v_bits,
                    args.n_warmup, args.n_bench, model_mem,
                    method_name=f'{cfg_str}_TQM',
                    block_q=args.block_q, block_t=args.block_t,
                    ppl_tokens=ppl_tokens)
            else:
                tqm_res = {'T': T, 'method': f'{cfg_str}_TQM', 'status': 'skipped'}
            all_results.append(tqm_res)

            def _sp(comp):
                if comp.get('status') == 'ok' and fp16_res.get('status') == 'ok':
                    return f"{fp16_res['median_ms']/comp['median_ms']:.2f}x"
                return "—"

            if (gtq_res.get('status') == 'ok' and tqm_res.get('status') == 'ok'):
                ratio_str = f"{tqm_res['median_ms']/gtq_res['median_ms']:.2f}x"
            else:
                ratio_str = "—"

            extra_f = f"{_ppl(fp16_res):>6} " if args.measure_ppl else ""
            extra_g = f"{_ppl(gtq_res):>6} "  if args.measure_ppl else ""
            extra_t = f"{_ppl(tqm_res):>6} "  if args.measure_ppl else ""
            print(f"{T:>8d} | "
                  f"{_pf(fp16_res):>7s} {_ms(fp16_res):>8s} {_pk(fp16_res):>8s} {extra_f}|"
                  f" {_pf(gtq_res):>7s} {_ms(gtq_res):>7s} {_pk(gtq_res):>7s} {extra_g}{_sp(gtq_res):>5s} |"
                  f" {_pf(tqm_res):>7s} {_ms(tqm_res):>7s} {_pk(tqm_res):>7s} {extra_t}{_sp(tqm_res):>5s} |"
                  f" {ratio_str:>8s}", flush=True)

            with open(json_path, 'w') as f:
                json.dump(all_results, f, indent=2)

    # ── Summary ──
    print()
    print(f"Results: {json_path}")
    print()
    print("Columns:")
    print("  T            : prefill length")
    print("  fp16 pf      : fp16 prefill time (HF FA-2 forward)")
    print("  fp16 ms / pk : fp16 decode latency (median ms/step) and peak GPU memory")
    print("  GTQ pf/ms/pk : K{k}V{v}-GTQ (Block-GTQ, non-uniform K) prefill, decode, peak")
    print("  TQM pf/ms/pk : K{k}V{v}-TQM (TQ_MSE, uniform K)        prefill, decode, peak")
    print("  sp_F         : speedup vs fp16 (fp16_ms / method_ms; >1 = method faster)")
    print("  TQM/GTQ      : tqm_ms / gtq_ms  (<1 = TQM faster than GTQ)")
    if args.measure_ppl:
        print(f"  ppl          : perplexity over {args.ppl_tokens} held-out tokens "
              f"after prefill (lower is better)")


if __name__ == '__main__':
    main()
