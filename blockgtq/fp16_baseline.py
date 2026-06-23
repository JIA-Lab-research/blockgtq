"""fp16 KV cache + decode glue (no quantization).

Provided as a baseline so :mod:`bench.bench_decode_e2e` and friends can
compare Block-GTQ decode latency / memory against the same model running on
an uncompressed fp16 cache.
"""
import math

import torch
import torch.nn.functional as F


class FP16KVCache:
    """Simple fp16 KV cache for autoregressive decode (no quantization)."""

    def __init__(self, n_layers, nkv, nq, hd, max_T, device='cuda'):
        self.n_layers = n_layers
        self.nkv = nkv
        self.nq = nq
        self.hd = hd
        self.gqa = nq // nkv
        self.scale = 1.0 / math.sqrt(hd)
        self.seq_len = 0
        self.k_cache = torch.zeros(n_layers, 1, nkv, max_T, hd,
                                   dtype=torch.float16, device=device)
        self.v_cache = torch.zeros(n_layers, 1, nkv, max_T, hd,
                                   dtype=torch.float16, device=device)

    def attend(self, layer_idx, q, k, v):
        """Store fp16 K/V and compute standard SDPA."""
        li = layer_idx
        t = self.seq_len
        self.k_cache[li, :, :, t:t + 1] = k
        self.v_cache[li, :, :, t:t + 1] = v
        T = t + 1
        k_all = self.k_cache[li, :, :, :T].repeat_interleave(self.gqa, dim=1)
        v_all = self.v_cache[li, :, :, :T].repeat_interleave(self.gqa, dim=1)
        return F.scaled_dot_product_attention(
            q, k_all, v_all, attn_mask=None, is_causal=False,
            scale=self.scale)

    def increment(self):
        self.seq_len += 1

    def cache_memory_bytes(self) -> int:
        T = self.seq_len
        return 2 * self.n_layers * self.nkv * T * self.hd * 2  # K + V, fp16


@torch.no_grad()
def fp16_decode_step(model, input_ids, position_ids, cache):
    """One decode step using an fp16 KV cache (no quantization).

    Mirrors :func:`blockgtq.production_decode.production_decode_step` so
    that the only difference between fp16 baseline and Block-GTQ decode is
    the cache class.
    """
    from transformers.models.llama.modeling_llama import apply_rotary_pos_emb

    hidden = model.model.embed_tokens(input_ids)

    for li, layer in enumerate(model.model.layers):
        residual = hidden
        hidden = layer.input_layernorm(hidden)
        attn = layer.self_attn
        B, S, _ = hidden.shape

        q = attn.q_proj(hidden)
        k = attn.k_proj(hidden)
        v = attn.v_proj(hidden)

        q = q.view(B, S, cache.nq, cache.hd).transpose(1, 2)
        k = k.view(B, S, cache.nkv, cache.hd).transpose(1, 2)
        v = v.view(B, S, cache.nkv, cache.hd).transpose(1, 2)

        _rope = getattr(attn, 'rotary_emb', None) or model.model.rotary_emb
        cos, sin = _rope(v, position_ids)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        attn_out = cache.attend(li, q, k, v)

        attn_out = attn_out.transpose(1, 2).reshape(B, S, cache.nq * cache.hd)
        hidden = attn.o_proj(attn_out)
        hidden = residual + hidden

        residual = hidden
        hidden = layer.post_attention_layernorm(hidden)
        hidden = layer.mlp(hidden)
        hidden = residual + hidden

    cache.increment()
    hidden = model.model.norm(hidden)
    return model.lm_head(hidden)


__all__ = ["FP16KVCache", "fp16_decode_step"]
