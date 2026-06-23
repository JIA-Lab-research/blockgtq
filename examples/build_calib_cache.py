"""Build a per-(layer, KV head) Q + K calibration cache for Block-GTQ.

Supports three attention layouts:

  * **Standard GQA** (Llama, Qwen2.5, Mistral, DeepSeek-R1 distillations):
    separate ``q_proj`` and ``k_proj`` (plus optional ``q_norm`` / ``k_norm``
    pre-RoPE RMSNorm in Qwen3).
  * **Fused QKV** (GLM-4-9B): a single ``query_key_value`` linear that
    outputs ``[Q || K || V]`` concatenated.
  * **MLA** (DeepSeek-V2): ``kv_a_proj_with_mqa`` projects to a low-rank
    latent whose trailing ``qk_rope_head_dim`` columns are the K RoPE
    shard, and Q comes from either a single ``q_proj`` or a low-rank
    ``q_a_proj`` + ``q_b_proj`` pair.

For GQA and fused QKV, K has shape ``(n_layer, n_kv, T, head_dim)`` while Q
keeps all ``|G(h)|`` query heads of each KV group as separate samples,
shape ``(n_layer, n_kv, |G(h)|*T, head_dim)`` (NOT GQA-averaged: the energy
score squares per head, then averages). For MLA the K side has one shared
shard ``(n_layer, 1, T, qk_rope_head_dim)`` and all ``n_q`` query heads are
its samples: ``(n_layer, 1, n_q*T, qk_rope_head_dim)``.

The default calibration text comes from the bundled
``calib_data/wikitext2_calib_2k.txt`` (the file used for the paper's
NIAH and PPL numbers).

Usage::

    python examples/build_calib_cache.py \\
        --model meta-llama/Llama-3.1-8B-Instruct \\
        --device cuda:0 \\
        --n-tokens 2048 \\
        --out calib_llama31_8b_instruct.pt
"""
import argparse
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PROMPT = REPO_ROOT / "calib_data" / "wikitext2_calib_2k.txt"


def _get_layers(model):
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    if hasattr(model, "transformer"):
        tx = model.transformer
        if hasattr(tx, "encoder") and hasattr(tx.encoder, "layers"):
            return tx.encoder.layers          # GLM-4 ChatGLM family
        if hasattr(tx, "h"):
            return tx.h                       # GPT-2 style
        if hasattr(tx, "layers"):
            return tx.layers
    raise ValueError(f"Unknown model architecture: {type(model).__name__}")


def _get_attn(layer):
    for name in ("self_attn", "self_attention", "attn"):
        m = getattr(layer, name, None)
        if m is not None:
            return m
    raise ValueError(f"Unknown layer type: {type(layer).__name__}")


def _detect_arch(attn, config):
    """Return one of 'gqa', 'fused_qkv', 'mla'."""
    qk_rope_dim = getattr(config, "qk_rope_head_dim", None)
    kv_lora_rank = getattr(config, "kv_lora_rank", None)
    if qk_rope_dim is not None and kv_lora_rank is not None:
        return "mla"
    if hasattr(attn, "query_key_value") and not hasattr(attn, "k_proj"):
        return "fused_qkv"
    return "gqa"


def _extract_qk_from_hidden(attn, hidden, arch, n_q, n_kv, head_dim, config):
    """Return ``(q_pre_rope, k_pre_rope)`` tensors at the layout the calib
    cache expects:

      * GQA / fused QKV: ``q = (T, n_q, head_dim)``, ``k = (T, n_kv, head_dim)``.
      * MLA:             ``q = (T, n_q, qk_rope_head_dim)``,
                         ``k = (T, 1, qk_rope_head_dim)``.

    Everything is computed in fp32 on the same device as ``hidden``.
    """
    hs = hidden
    if arch == "gqa":
        q_out = attn.q_proj(hs)
        k_out = attn.k_proj(hs)
        if hasattr(attn, "q_norm") and attn.q_norm is not None:
            q_re = q_out.view(*q_out.shape[:-1], n_q, head_dim)
            q_re = attn.q_norm(q_re)
            q_out = q_re.reshape(*q_out.shape[:-1], n_q * head_dim)
        if hasattr(attn, "k_norm") and attn.k_norm is not None:
            k_re = k_out.view(*k_out.shape[:-1], n_kv, head_dim)
            k_re = attn.k_norm(k_re)
            k_out = k_re.reshape(*k_out.shape[:-1], n_kv * head_dim)
        T = q_out.shape[-2] if q_out.dim() == 3 else q_out.shape[0]
        q = q_out[0].reshape(T, n_q, head_dim).float()
        k = k_out[0].reshape(T, n_kv, head_dim).float()
        return q, k

    if arch == "fused_qkv":
        qkv = attn.query_key_value(hs)
        T = qkv.shape[-2]
        q_dim = n_q * head_dim
        k_dim = n_kv * head_dim
        q_out = qkv[..., :q_dim]
        k_out = qkv[..., q_dim:q_dim + k_dim]
        q = q_out[0].reshape(T, n_q, head_dim).float()
        k = k_out[0].reshape(T, n_kv, head_dim).float()
        return q, k

    if arch == "mla":
        qk_rope_dim = config.qk_rope_head_dim
        nope_dim = getattr(config, "qk_nope_head_dim", 0)
        per_head_q = nope_dim + qk_rope_dim
        compressed = attn.kv_a_proj_with_mqa(hs)
        T = compressed.shape[-2]
        k_rope = compressed[0, :, -qk_rope_dim:].float()  # (T, qk_rope_dim)
        k = k_rope.view(T, 1, qk_rope_dim)
        if hasattr(attn, "q_proj"):
            q_full = attn.q_proj(hs)
        else:
            q_lat = attn.q_a_proj(hs)
            if hasattr(attn, "q_a_layernorm"):
                q_lat = attn.q_a_layernorm(q_lat)
            q_full = attn.q_b_proj(q_lat)
        q_re = q_full[0].reshape(T, n_q, per_head_q)
        q_rope = q_re[:, :, -qk_rope_dim:].float()  # (T, n_q, qk_rope_dim)
        return q_rope, k

    raise ValueError(f"Unknown arch: {arch!r}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--n-tokens", type=int, default=2048,
                    help="Calibration window length (matches paper default).")
    ap.add_argument("--prompt-file", default=str(DEFAULT_PROMPT),
                    help=f"Plain-text calibration source. "
                         f"Default: {DEFAULT_PROMPT.relative_to(REPO_ROOT)}.")
    ap.add_argument("--hf-split", default=None,
                    choices=["train", "validation", "test"],
                    help="If set, load WikiText-2 via HF (this split, lines "
                         ">50 chars) instead of "
                         "--prompt-file. Use 'train' for the attn-diag calib "
                         "so it does NOT overlap the WT2-test eval window "
                         "(the bundled txt is WT2-test -> leakage for "
                         "WT2-test evals).")
    ap.add_argument("--out", required=True,
                    help="Output .pt path (relative or absolute).")
    args = ap.parse_args()

    print(f"[calib] model={args.model}  n_tokens={args.n_tokens}  device={args.device}")
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float16, trust_remote_code=True,
        attn_implementation="flash_attention_2",
    ).to(args.device).eval()

    config = model.config
    n_q = config.num_attention_heads
    n_kv = getattr(config, "num_key_value_heads",
                   getattr(config, "multi_query_group_num", n_q))
    layers = _get_layers(model)
    n_layer = len(layers)

    # Detect architecture from the first layer (assumed homogeneous).
    arch = _detect_arch(_get_attn(layers[0]), config)
    qk_rope_dim = getattr(config, "qk_rope_head_dim", None)
    if arch == "mla":
        head_dim_cache = qk_rope_dim
        n_kv_cache = 1
    elif hasattr(config, "kv_channels"):
        head_dim_cache = config.kv_channels
        n_kv_cache = n_kv
    elif getattr(config, "head_dim", None):
        head_dim_cache = int(config.head_dim)
        n_kv_cache = n_kv
    else:
        head_dim_cache = config.hidden_size // n_q
        n_kv_cache = n_kv

    if arch != "mla" and n_q % n_kv != 0:
        raise ValueError(f"n_q={n_q} not divisible by n_kv={n_kv}")
    q_per_kv = (n_q // n_kv) if arch != "mla" else n_q
    print(f"[calib] arch={arch}  n_layer={n_layer}  "
          f"n_q={n_q}  n_kv_cache={n_kv_cache}  "
          f"head_dim_cache={head_dim_cache}  q_per_kv={q_per_kv}")

    if args.hf_split:
        # Load WikiText-2 directly via HF (filter short lines, join).
        # --hf-split train avoids the leakage of the bundled txt (which is
        # WT2-test) for WT2-test eval windows.
        from datasets import load_dataset
        ds = load_dataset("wikitext", "wikitext-2-raw-v1", split=args.hf_split)
        text = "\n".join([t for t in ds["text"] if len(t) > 50])
        ids = tok.encode(text, add_special_tokens=False)[:args.n_tokens]
        source_label = (f"HF wikitext-2-raw-v1 {args.hf_split} (len>50), "
                        f"first {len(ids)} tokens")
        print(f"[calib] tokenized {len(ids)} tokens from HF wikitext-2 "
              f"{args.hf_split}")
    else:
        prompt_path = Path(args.prompt_file)
        if not prompt_path.exists():
            raise FileNotFoundError(f"--prompt-file not found: {prompt_path}")
        text = prompt_path.read_text()
        ids = tok.encode(text, add_special_tokens=False)[:args.n_tokens]
        source_label = f"{prompt_path.name} first {len(ids)} tokens"
        print(f"[calib] tokenized {len(ids)} tokens from {prompt_path.name}")

    # One forward pass with hidden states; per-layer projections are
    # re-computed from those hidden states so we can handle GQA, fused
    # QKV, and MLA in one place.
    input_ids = torch.tensor([ids], device=args.device, dtype=torch.long)
    with torch.no_grad():
        outputs = model(input_ids, output_hidden_states=True)
    hidden_states = outputs.hidden_states  # tuple of len n_layer+1, each (1, T, hidden)

    # Q-side stores ALL query heads serving each KV head as separate samples
    # (NOT GQA-averaged). The Block-GTQ energy score downstream computes
    #     E_{t, g ∈ G(h)} ||q_{l,g,t}^{(i)}||^2
    # via ``compute_freq_importance_energy``'s ``.mean(dim=0)`` over the
    # ``|G(h)| * T`` Q samples per KV head. Averaging Q vectors first and
    # then squaring underestimates the energy by Jensen's inequality when
    # GQA heads point in different directions (see paper's Calibration
    # Formula appendix).
    n_samples_q = q_per_kv * args.n_tokens  # = |G(h)| * T for GQA / fused;
                                              #   = n_q * T for MLA
    calib_k = torch.zeros(n_layer, n_kv_cache, args.n_tokens, head_dim_cache,
                          dtype=torch.float32)
    calib_q = torch.zeros(n_layer, n_kv_cache, n_samples_q, head_dim_cache,
                          dtype=torch.float32)
    with torch.no_grad():
        for li in range(n_layer):
            attn = _get_attn(layers[li])
            # ``hidden_states[li]`` feeds layer ``li``'s input layernorm; the
            # post-norm projection is what we want.
            hs_pre = hidden_states[li]
            ln = getattr(layers[li], "input_layernorm", None)
            if ln is None:
                raise AttributeError(
                    f"Layer {li} ({type(layers[li]).__name__}) has no "
                    f"'input_layernorm'. Refusing to silently fall back to the "
                    f"raw residual stream -- that would calibrate on the wrong "
                    f"K distribution (the input_layernorm bug). Add this "
                    f"model's pre-attention norm name here.")
            hs_normed = ln(hs_pre)
            q_per_token, k_per_token = _extract_qk_from_hidden(
                attn, hs_normed, arch, n_q, n_kv,
                head_dim_cache if arch != "mla" else config.qk_rope_head_dim,
                config)
            T = min(args.n_tokens, k_per_token.shape[0])
            k_t = k_per_token[:T]  # (T, n_kv_cache, hd_cache)
            q_t = q_per_token[:T]  # (T, n_q, hd_cache)

            calib_k[li, :, :T, :] = k_t.transpose(0, 1).cpu()
            if arch == "mla":
                # (T, n_q, d_rope) -> (1, n_q*T, d_rope) — all query heads
                # are samples for the single shared RoPE-K head.
                q_per_head = q_t.permute(1, 0, 2).reshape(1, n_q * T,
                                                           head_dim_cache)
                calib_q[li, :, :n_q * T, :] = q_per_head.cpu()
            else:
                # (T, n_q, hd) -> (T, n_kv, |G|, hd) -> permute to
                # (n_kv, |G|, T, hd) -> reshape to (n_kv, |G|*T, hd).
                q_grouped = q_t.view(T, n_kv, q_per_kv, head_dim_cache)
                q_per_head = q_grouped.permute(1, 2, 0, 3).reshape(
                    n_kv, q_per_kv * T, head_dim_cache)
                calib_q[li, :, :q_per_kv * T, :] = q_per_head.cpu()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "calib_k": calib_k,
        "calib_q": calib_q,
        "model": args.model,
        "arch": arch,
        "n_tokens": args.n_tokens,
        "head_dim": head_dim_cache,
        "n_kv": n_kv_cache,
        "n_q": n_q,
        "n_layer": n_layer,
        "q_per_kv": q_per_kv,
        "n_samples_q": n_samples_q,
        "q_convention": "all_heads_per_kv_group",
        "source": source_label,
        "score": "energy",
    }
    torch.save(payload, out_path)
    print(f"[calib] saved {out_path}\n"
          f"        calib_k shape {tuple(calib_k.shape)}, "
          f"calib_q shape {tuple(calib_q.shape)}, arch={arch}")
    mean_k_energy = calib_k.pow(2).mean(dim=(2, 3))
    print(f"[calib] K-energy per (layer, kv-head) — "
          f"min={mean_k_energy.min():.4f}  max={mean_k_energy.max():.4f}  "
          f"mean={mean_k_energy.mean():.4f}")
    mean_q_energy = calib_q.pow(2).mean(dim=(2, 3))
    print(f"[calib] Q-energy per (layer, kv-head) — "
          f"min={mean_q_energy.min():.4f}  max={mean_q_energy.max():.4f}  "
          f"mean={mean_q_energy.mean():.4f}")


if __name__ == "__main__":
    main()
