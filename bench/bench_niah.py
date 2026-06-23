"""Multi-task Needle-In-A-Haystack benchmark for Block-GTQ.

Patches every layer's ``k_proj`` and ``v_proj`` with quantize-dequantize
wrappers so the attention forward sees the post-decompress K/V, runs the
6 standard NIAH tasks ({single, distractor, multi, multikey, multivalue,
multiquery}) at a sweep of context lengths and needle depths, and scores
the model's answer against the ground-truth needle.

Three K methods are compared, all reading the same per-(layer, KV-head)
calibration cache (see :mod:`examples.build_calib_cache`):

  * ``KIVI-ScaleOnly`` — rolling per-channel asymmetric quant, refreshed
    every ``group_size`` tokens.
  * ``TQ-MSE`` — uniform-rate TurboQuant-MSE, the immediate baseline
    Block-GTQ improves on.
  * ``Block-GTQ`` — paper's RoPE-aware non-uniform bit allocation with
    rotation_threshold = 1 (every block goes through a dense per-bit-width
    rotation group).

V-side is always TurboQuant-MSE per project convention.

Usage::

    python bench/bench_niah.py \\
        --model meta-llama/Llama-3.1-8B-Instruct --device cuda:0 \\
        --context-lengths 4096 8192 16384 32768 65536 131072 \\
        --k-bits 2 --v-bits 2 \\
        --tasks single distractor multi multikey multivalue multiquery \\
        --calib-cache calib_llama31_8b_instruct.pt \\
        --n-trials 3
"""
import argparse
import gc
import json
import os
import random
import sys
import time
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn

# Allow `python bench/bench_niah.py ...` without installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bench._baselines import KIVIScaleOnlyQuantizer
from bench._niah_tasks import (
    NEEDLES, QUERIES, ANSWERS,
    build_single_needle_prompt, build_distractor_prompt,
    build_multi_needle_prompt,
    build_multivalue_prompt, build_multiquery_prompt, build_multikey_prompt,
    get_model_layers, get_attn_module,
)

RESULTS_DIR = Path(os.environ.get(
    "BLOCKGTQ_RESULTS_DIR",
    str(Path(__file__).resolve().parent.parent / "niah_results"),
))
RESULTS_DIR.mkdir(exist_ok=True, parents=True)


# =============================================================================
# Per-head quantizer construction
# =============================================================================

def _build_head_quantizer(method, hd, bits, h, sample_kv, sample_q):
    """Build the per-head K-side quantizer for one KV head.

    ``sample_kv`` / ``sample_q`` are pre-RoPE K and GQA-averaged Q
    activations from the calibration cache, both shaped ``(T, hd)``.
    Both must be provided — the lazy fallback that auto-calibrated from
    the live forward batch is removed because it (a) couples calibration
    to the test prompt and (b) hit a Q-shape mismatch when the live K
    sample was shorter than the energy-score expected.
    """
    assert sample_kv is not None and sample_q is not None, (
        "bench_niah requires --calib-cache; lazy fallback removed.")
    device = sample_kv.device

    if method == "kivi_scale_only":
        q = KIVIScaleOnlyQuantizer(hd, n_bits=bits, group_size=32)
        q.fit(sample_kv)
        return q

    if method == "tq_mse":
        from blockgtq.tq import TurboQuantMSE
        return TurboQuantMSE(d=hd, bit_width=bits, seed=42 + h, device=device)

    if method == "block_gtq":
        from blockgtq.block_gtq_pipeline import BlockGTQPipeline
        q = BlockGTQPipeline(
            head_dim=hd, avg_bits=float(bits),
            rotation_threshold=1, min_bits=1, max_bits=8,
            device=device,
        )
        q.calibrate(sample_q, sample_kv)
        return q

    raise ValueError(f"Unknown method: {method!r}")


# =============================================================================
# Stacked per-method state + batched compress kernels
# =============================================================================

def _stack_kivi_state(head_qs, device):
    cmin_init = torch.stack([q.channel_min for q in head_qs])
    cscale_init = torch.stack([q.channel_scale for q in head_qs])
    n_kv = len(head_qs)
    hd = head_qs[0].head_dim
    g = head_qs[0].group_size
    buf = torch.zeros(n_kv, g, hd, device=device, dtype=torch.float32)
    return {
        "cmin_init": cmin_init, "cscale_init": cscale_init,
        "cmin": cmin_init.clone(), "cscale": cscale_init.clone(),
        "n_levels": head_qs[0].n_levels,
        "group_size": g, "buf": buf, "buf_filled": 0,
    }


def _stack_tq_state(head_qs):
    rot_T = torch.stack([q.rotation_T for q in head_qs])
    rot = torch.stack([q.rotation for q in head_qs])
    return {
        "rot_T": rot_T, "rot": rot,
        "centroids": head_qs[0].centroids,
        "boundaries": head_qs[0].boundaries,
        "norm_correction": head_qs[0].norm_correction,
    }


def _stack_block_state(head_qs):
    n = len(head_qs)
    hd = head_qs[0].head_dim
    device = head_qs[0]._head_perm.device

    head_perm = torch.stack([q._head_perm for q in head_qs])
    head_perm_inv = torch.stack([q._head_perm_inv for q in head_qs])
    block_rot_T = torch.stack([q._block_rot_T for q in head_qs])
    block_rot_unp = torch.stack([q._block_rot_unperm for q in head_qs])

    max_nb = max(q._pos_boundaries.shape[1] for q in head_qs)
    max_nc = max(q._pos_centroids.shape[1] for q in head_qs)
    pos_b = torch.full((n, hd, max_nb), float("inf"),
                       device=device, dtype=torch.float32)
    pos_c = torch.zeros((n, hd, max_nc), device=device, dtype=torch.float32)
    for i, q in enumerate(head_qs):
        nb = q._pos_boundaries.shape[1]
        nc = q._pos_centroids.shape[1]
        pos_b[i, :, :nb] = q._pos_boundaries
        pos_c[i, :, :nc] = q._pos_centroids

    max_g = max(q._group_mask.shape[1] for q in head_qs)
    group_mask = torch.zeros((n, hd, max_g), device=device, dtype=torch.float32)
    group_of = torch.zeros((n, hd), device=device, dtype=torch.long)
    for i, q in enumerate(head_qs):
        g = q._group_mask.shape[1]
        group_mask[i, :, :g] = q._group_mask
        group_of[i] = q._group_of
    return {
        "head_perm": head_perm, "head_perm_inv": head_perm_inv,
        "block_rot_T": block_rot_T, "block_rot_unperm": block_rot_unp,
        "pos_boundaries": pos_b, "pos_centroids": pos_c,
        "group_mask": group_mask, "group_of": group_of,
        "max_nc": max_nc,
    }


@torch.no_grad()
def _compress_kivi(k, st, n_kv, hd):
    """KIVI-ScaleOnly streaming rolling-scale compress."""
    cmin = st["cmin"]
    cscale = st["cscale"]
    nL = st["n_levels"]
    g = st["group_size"]
    buf = st["buf"]
    B, S, _ = k.shape
    if S > 1:
        cmin.copy_(st["cmin_init"])
        cscale.copy_(st["cscale_init"])
        st["buf_filled"] = 0
    filled = int(st["buf_filled"])
    kv = k.float().view(B, S, n_kv, hd)
    out = torch.empty_like(kv)
    i = 0
    while i < S:
        take = min(g - filled, S - i)
        chunk = kv[:, i:i + take, :, :]
        codes = torch.round((chunk - cmin) / cscale).clamp_(0, nL)
        out[:, i:i + take, :, :] = codes * cscale + cmin
        buf[:, filled:filled + take, :] = chunk[0].transpose(0, 1)
        filled += take
        if filled == g:
            new_min = buf.amin(dim=1)
            new_max = buf.amax(dim=1)
            cmin.copy_(new_min)
            cscale.copy_((new_max - new_min).clamp_(min=1e-8) / nL)
            filled = 0
        i += take
    st["buf_filled"] = filled
    return out.view(*k.shape).to(k.dtype)


@torch.no_grad()
def _compress_tq(k, st, n_kv, hd):
    rot_T = st["rot_T"]
    rot = st["rot"]
    cents = st["centroids"]
    bounds = st["boundaries"]
    use_nc = st["norm_correction"]

    kv = k.float().view(*k.shape[:-1], n_kv, hd)
    norms = torch.norm(kv, dim=-1, keepdim=True)
    safe = torch.where(norms > 0, norms, torch.ones_like(norms))
    x_n = kv / safe

    y = torch.einsum("bsnd,nde->bsne", x_n, rot_T)
    y_flat = y.reshape(-1).contiguous()
    indices = torch.searchsorted(bounds, y_flat)
    y_hat = cents[indices].view_as(y)

    if use_nc:
        yh_norms = torch.norm(y_hat, dim=-1, keepdim=True)
        yh_safe = torch.where(yh_norms > 1e-10, yh_norms,
                              torch.ones_like(yh_norms))
        y_hat = y_hat / yh_safe

    x_hat_unit = torch.einsum("bsne,ned->bsnd", y_hat, rot)
    result = (x_hat_unit * norms).reshape(*k.shape)
    return result.to(k.dtype)


@torch.no_grad()
def _compress_block(k, st, n_kv, hd):
    head_perm = st["head_perm"]
    block_rot_T = st["block_rot_T"]
    block_rot_unp = st["block_rot_unperm"]
    pos_b = st["pos_boundaries"]
    pos_c = st["pos_centroids"]
    g_mask = st["group_mask"]
    g_of = st["group_of"]
    max_nc = st["max_nc"]

    batch_shape = k.shape[:-1]
    N = 1
    for s in batch_shape:
        N *= s
    x_full = k.float().view(N, n_kv, hd)

    head_perm_e = head_perm.unsqueeze(0).expand(N, -1, -1)
    x = torch.gather(x_full, -1, head_perm_e)

    sq_norms = torch.einsum("nkd,kdg->nkg", x * x, g_mask)
    norms_g = torch.sqrt(sq_norms)
    safe_g = torch.where(norms_g > 0, norms_g, torch.ones_like(norms_g))
    g_of_e = g_of.unsqueeze(0).expand(N, -1, -1)
    per_pos_norms = torch.gather(safe_g, -1, g_of_e)
    x = x / per_pos_norms

    y = torch.einsum("nkd,kde->nke", x, block_rot_T)

    y_kj = y.permute(1, 2, 0).contiguous()
    indices_kj = torch.searchsorted(pos_b, y_kj).clamp_(max=max_nc - 1)
    y_hat_kj = torch.gather(pos_c, -1, indices_kj)
    y_hat = y_hat_kj.permute(2, 0, 1).contiguous()

    yh_sq_norms = torch.einsum("nkd,kdg->nkg", y_hat * y_hat, g_mask)
    yh_norms = torch.sqrt(yh_sq_norms)
    yh_safe = torch.where(yh_norms > 1e-10, yh_norms,
                          torch.ones_like(yh_norms))
    yh_per_pos = torch.gather(yh_safe, -1, g_of_e)
    y_hat = y_hat / yh_per_pos

    result = torch.einsum("nkd,kde->nke", y_hat * per_pos_norms,
                          block_rot_unp)
    return result.reshape(*batch_shape, n_kv * hd).to(k.dtype)


_METHOD_COMPRESS = {
    "kivi_scale_only": _compress_kivi,
    "tq_mse": _compress_tq,
    "block_gtq": _compress_block,
}


# =============================================================================
# K / V projection patchers
# =============================================================================

class _FastKProj(nn.Module):
    """K-projection patcher: builds per-head state from a calib cache slice
    on the first forward, then runs one batched compress per forward."""

    def __init__(self, orig_proj, head_dim, n_kv_heads, method, k_bits,
                 ext_calib_k, ext_calib_q):
        super().__init__()
        self.orig = orig_proj
        self.hd = head_dim
        self.nkv = n_kv_heads
        self.m = method
        self.b = k_bits
        self.ext_calib_k = ext_calib_k  # (n_kv, T, hd) fp32 CPU
        self.ext_calib_q = ext_calib_q  # (n_kv, T, hd) fp32 CPU
        self._head_qs = []
        self._stacked = None
        self._compress_fn = _METHOD_COMPRESS[method]

    def _ensure_built(self, k_ref):
        if self._stacked is not None:
            return
        dev = k_ref.device
        for h in range(self.nkv):
            sample_kv = self.ext_calib_k[h].to(device=dev, dtype=torch.float32)
            sample_q = self.ext_calib_q[h].to(device=dev, dtype=torch.float32)
            self._head_qs.append(_build_head_quantizer(
                self.m, self.hd, self.b, h, sample_kv, sample_q))
            del sample_kv, sample_q
        torch.cuda.empty_cache()

        if self.m == "kivi_scale_only":
            self._stacked = _stack_kivi_state(self._head_qs, dev)
        elif self.m == "tq_mse":
            self._stacked = _stack_tq_state(self._head_qs)
        elif self.m == "block_gtq":
            self._stacked = _stack_block_state(self._head_qs)
        else:
            raise ValueError(f"Unknown method: {self.m!r}")

    @torch.no_grad()
    def forward(self, x, **kwargs):
        k = self.orig(x, **kwargs)
        self._ensure_built(k)
        return self._compress_fn(k, self._stacked, self.nkv, self.hd).to(k.dtype)


class _FastVProjTQMSE(nn.Module):
    """V-projection patcher: TurboQuant-MSE for every head (project convention)."""

    def __init__(self, orig_proj, head_dim, n_kv_heads, v_bits):
        super().__init__()
        self.orig = orig_proj
        self.hd = head_dim
        self.nkv = n_kv_heads
        self.b = v_bits
        self._head_qs = []
        self._stacked = None

    def _ensure_built(self, v_ref):
        if self._stacked is not None:
            return
        from blockgtq.tq import TurboQuantMSE
        for h in range(self.nkv):
            self._head_qs.append(TurboQuantMSE(
                d=self.hd, bit_width=self.b, seed=1000 + h,
                device=v_ref.device,
            ))
        self._stacked = _stack_tq_state(self._head_qs)

    @torch.no_grad()
    def forward(self, x, **kwargs):
        v = self.orig(x, **kwargs)
        self._ensure_built(v)
        return _compress_tq(v, self._stacked, self.nkv, self.hd).to(v.dtype)


def patch_model_kv(model, head_dim, n_kv_heads, k_method, k_bits, v_bits,
                   calib_cache_k, calib_cache_q):
    """Wrap k_proj and v_proj on every layer. Returns hooks list for unpatch."""
    hooks = []
    layers = get_model_layers(model)
    for li, layer in enumerate(layers):
        attn = get_attn_module(layer)
        orig_k = attn.k_proj
        orig_v = attn.v_proj
        attn.k_proj = _FastKProj(orig_k, head_dim, n_kv_heads, k_method,
                                  k_bits, calib_cache_k[li], calib_cache_q[li])
        attn.v_proj = _FastVProjTQMSE(orig_v, head_dim, n_kv_heads, v_bits)
        hooks.append((attn, orig_k, orig_v))
    return hooks


def unpatch_model_kv(hooks):
    for attn, orig_k, orig_v in hooks:
        attn.k_proj = orig_k
        attn.v_proj = orig_v


# =============================================================================
# NIAH single-test driver + scoring
# =============================================================================

@torch.no_grad()
def _run_niah(model, tokenizer, device, prompt, expected, max_new=30):
    """Run a single NIAH probe. Returns (score, generated, input_tokens).

    Score is 1.0 if the answer matches, 0.0 otherwise. For multi-answer
    tasks (multikey/multivalue/multiquery) score is the fraction of
    needles whose answer appears in the generated text.
    """
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True,
                       max_length=131072).to(device)
    input_len = inputs["input_ids"].shape[1]
    outputs = model.generate(**inputs, max_new_tokens=max_new, do_sample=False,
                              temperature=1.0, pad_token_id=tokenizer.eos_token_id)
    generated = tokenizer.decode(outputs[0][input_len:],
                                 skip_special_tokens=True).strip()
    if isinstance(expected, list):
        hits = sum(1 for ans in expected if ans.lower() in generated.lower())
        score = hits / len(expected)
    else:
        score = 1.0 if expected.lower() in generated.lower() else 0.0
    return score, generated[:120], input_len


def _build_prompt(task, tokenizer, ctx_len, depth):
    if task == "single":
        return build_single_needle_prompt(tokenizer, ctx_len, depth)
    if task == "distractor":
        return build_distractor_prompt(tokenizer, ctx_len, depth)
    if task == "multi":
        return build_multi_needle_prompt(tokenizer, ctx_len, query_key="multi1")
    if task == "multikey":
        return build_multikey_prompt(tokenizer, ctx_len, depth)
    if task == "multivalue":
        return build_multivalue_prompt(tokenizer, ctx_len, depth)
    if task == "multiquery":
        return build_multiquery_prompt(tokenizer, ctx_len, depth)
    raise ValueError(f"Unknown task: {task!r}")


# =============================================================================
# Main
# =============================================================================

ALL_METHODS = [
    ("fp16", "fp16", None, None),
    ("kivi_scale_only", "KIVI-ScaleOnly", None, None),
    ("tq_mse", "TQ-MSE", None, None),
    ("block_gtq", "Block-GTQ", None, None),
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--context-lengths", type=int, nargs="+",
                        default=[4096, 8192, 16384, 32768, 65536, 131072])
    parser.add_argument("--depths", type=float, nargs="+",
                        default=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7,
                                  0.8, 0.9, 1.0])
    parser.add_argument("--n-trials", type=int, default=3)
    parser.add_argument("--k-bits", type=int, default=2)
    parser.add_argument("--v-bits", type=int, default=2)
    parser.add_argument("--tasks", nargs="+",
                        default=["single", "distractor", "multi",
                                 "multikey", "multivalue", "multiquery"])
    parser.add_argument("--methods", nargs="+", default=None,
                        help="Method labels to keep (default: all). "
                             "Choices: fp16, KIVI-ScaleOnly, TQ-MSE, Block-GTQ.")
    parser.add_argument("--calib-cache", required=True,
                        help="Path to the .pt produced by "
                             "examples/build_calib_cache.py.")
    parser.add_argument("--out-tag", default=None,
                        help="Filename tag for the result files.")
    args = parser.parse_args()

    ts = args.out_tag if args.out_tag else datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = RESULTS_DIR / f"niah_{ts}.md"
    json_file = RESULTS_DIR / f"niah_{ts}.json"
    lines = [
        f"# NIAH K{args.k_bits}V{args.v_bits} on {args.model} — {ts}",
        "",
        f"**Context lengths:** {args.context_lengths}",
        f"**Depths:** {args.depths}",
        f"**Trials per cell:** {args.n_trials}",
        f"**Tasks:** {args.tasks}",
        f"**Calib cache:** {args.calib_cache}",
        "",
    ]
    all_results = []

    def log(msg):
        print(msg, flush=True)
        lines.append(str(msg))
        Path(log_file).write_text("\n".join(lines))

    from transformers import AutoModelForCausalLM, AutoTokenizer

    log(f"Loading {args.model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float16, trust_remote_code=True,
        attn_implementation="flash_attention_2",
    ).to(args.device).eval()

    head_dim = getattr(model.config, "head_dim",
                       model.config.hidden_size // model.config.num_attention_heads)
    n_kv_heads = getattr(model.config, "num_key_value_heads",
                         model.config.num_attention_heads)
    log(f"  hd={head_dim}, n_kv={n_kv_heads}")

    ck = torch.load(args.calib_cache, map_location="cpu", weights_only=False)
    calib_cache_k = ck["calib_k"]
    calib_cache_q = ck["calib_q"]
    cl, cn, ct, cd = calib_cache_k.shape
    if cn != n_kv_heads or cd != head_dim:
        raise ValueError(
            f"calib cache shape mismatch: got n_kv={cn}, hd={cd}; "
            f"expected n_kv={n_kv_heads}, hd={head_dim}")
    log(f"**Calibration:** {ck.get('source', '<unknown source>')} "
        f"(shape {tuple(calib_cache_k.shape)})")
    log("")

    methods = [m for m in ALL_METHODS]
    if args.methods:
        wanted = set(args.methods)
        methods = [m for m in methods if m[1] in wanted]

    random.seed(42)

    for task in args.tasks:
        log(f"\n## Task: {task}\n")
        for ctx_len in args.context_lengths:
            log(f"\n### Context: {ctx_len} tokens\n")
            header = "| Method |" + " | ".join(f"d={d:.0%}" for d in args.depths) + " | Score |"
            sep = "|--------|" + "|".join("---" for _ in args.depths) + "|-------|"
            log(header)
            log(sep)

            for mk, mn, _, _ in methods:
                hooks = None
                if mk != "fp16":
                    try:
                        hooks = patch_model_kv(
                            model, head_dim, n_kv_heads,
                            k_method=mk, k_bits=args.k_bits, v_bits=args.v_bits,
                            calib_cache_k=calib_cache_k, calib_cache_q=calib_cache_q,
                        )
                    except Exception as e:
                        log(f"| {mn} | patch ERROR: {type(e).__name__}: {e} |")
                        continue

                results_row = []
                per_depth = {}
                for depth in args.depths:
                    cell_successes = 0.0
                    for trial in range(args.n_trials):
                        try:
                            random.seed(42 + int(depth * 100) + ctx_len + trial * 7919)
                            prompt, expected = _build_prompt(task, tokenizer,
                                                              ctx_len, depth)
                            score, _, _ = _run_niah(model, tokenizer, args.device,
                                                     prompt, expected)
                            cell_successes += score
                        except Exception as e:
                            log(f"  [{mn}] depth={depth} trial={trial}: "
                                f"{type(e).__name__}: {str(e)[:80]}")
                    cell_avg = cell_successes / max(1, args.n_trials)
                    results_row.append(cell_avg)
                    per_depth[str(depth)] = cell_avg

                if hooks is not None:
                    unpatch_model_kv(hooks)
                    gc.collect()
                    torch.cuda.empty_cache()

                avg = sum(results_row) / max(1, len(results_row))
                row_str = " | ".join(f"{r*100:.0f}%" for r in results_row)
                log(f"| {mn} | {row_str} | {avg*100:.1f}% |")
                all_results.append({
                    "task": task, "method": mn, "method_key": mk,
                    "ctx_len": ctx_len, "k_bits": args.k_bits, "v_bits": args.v_bits,
                    "per_depth": per_depth, "avg_score": avg,
                    "n_trials": args.n_trials,
                })
                with open(json_file, "w") as f:
                    json.dump({"model": args.model,
                                "calib_cache": args.calib_cache,
                                "kv_compressed": True,
                                "results": all_results}, f, indent=2)

    print(f"\nSaved log to {log_file}")
    print(f"Saved JSON to {json_file}")


if __name__ == "__main__":
    main()
