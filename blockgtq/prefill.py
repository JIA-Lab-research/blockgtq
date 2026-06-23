"""Layer-major prefill driver for Block-GTQ.

Processes all T prefill tokens through one transformer layer at a time, so
each layer issues exactly one ``fused_blockgtq_prefill_attention`` call. This
avoids materialising an fp16 KV cache during prefill and is the path that
makes T >= 64K feasible on a single H800.

Contrast with chunk-major prefill (outer loop = chunks, inner loop = layers):

  Chunk-major: L * N_chunks transformer-layer iterations, many small
    QKV / MLP GEMMs with low GPU utilisation.
  Layer-major: L transformer-layer iterations, one big QKV / MLP GEMM per
    layer with high GPU utilisation; reuses
    :func:`fused_blockgtq_prefill_attention` with full-T Q.

Memory note: peak transient memory scales with T (MLP intermediate
dominates). For Qwen2.5-3B at T=256K, expect ~5.6 GB / layer transient.
For T much beyond that, fall back to a chunked prefill driver.
"""
import torch


@torch.no_grad()
def layer_major_prefill(model, cache, prompt_ids, block_q=32, block_t=128):
    """Layer-major prefill into a Block-GTQ packed KV cache.

    Args:
        model: HF Llama-style model.
        cache: :class:`blockgtq.production_cache.BlockGTQProductionCache`.
        prompt_ids: (1, T_total) int input ids.
        block_q, block_t: kernel tile sizes (defaults match production).

    Side effect: populates ``cache.k_packed``, ``cache.k_norms``,
    ``cache.v_packed``, ``cache.v_norms`` for positions ``[0, T_total)`` and
    sets ``cache.seq_len = T_total``.
    """
    from transformers.models.llama.modeling_llama import apply_rotary_pos_emb
    from blockgtq.kernels.fused_packed_attention import fused_blockgtq_prefill_attention

    device = prompt_ids.device
    B, T = prompt_ids.shape
    assert B == 1, "layer_major_prefill currently supports B=1 only"

    hidden = model.model.embed_tokens(prompt_ids)
    pos = torch.arange(T, device=device).unsqueeze(0)

    for li, layer in enumerate(model.model.layers):
        residual = hidden
        hidden = layer.input_layernorm(hidden)
        attn = layer.self_attn

        q = attn.q_proj(hidden)
        k = attn.k_proj(hidden)
        v = attn.v_proj(hidden)

        q = q.view(B, T, cache.nq, cache.hd).transpose(1, 2)
        k = k.view(B, T, cache.nkv, cache.hd).transpose(1, 2)
        v = v.view(B, T, cache.nkv, cache.hd).transpose(1, 2)

        _rope = getattr(attn, 'rotary_emb', None) or model.model.rotary_emb
        cos, sin = _rope(v, pos)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        cache._encode_batched(li, k, v, 0)

        out_parts = []
        for hi in range(cache.nkv):
            q8 = cache.quantizers[li][hi]
            ka = cache.kernel_args[li][hi]
            q_h = q[:, hi * cache.gqa:(hi + 1) * cache.gqa, :, :]

            if q8._q_perm_baked:
                q_h_rot = q_h
            else:
                q_flat = q_h.reshape(-1, cache.hd)
                q_rot = q8.rotate_q_for_packed_kernel(q_flat)
                q_h_rot = q_rot.reshape(q_h.shape)

            kp = cache.k_packed[li, :, hi:hi + 1, :T]
            kn = cache.k_norms[li, :, hi:hi + 1, :T]
            vp = cache.v_packed[li, :, hi:hi + 1, :T]
            vn = cache.v_norms[li, :, hi:hi + 1, :T]
            v_rot = getattr(q8, '_v_rot', None)

            out_h = fused_blockgtq_prefill_attention(
                q_h_rot, kp, kn,
                ka['codebook_flat'], ka['lut_offset'], ka['norm_group'],
                ka['segments_b'],
                vp, vn, q8.v_lut,
                scale=cache.scale,
                block_q=block_q, block_t=block_t,
                v_bits=cache.v_bits,
                v_rotation=v_rot,
                is_causal=True,
                q_pos_offset=0,
                k_nibble=cache.k_nibble,
                v_nibble=cache.v_nibble,
            )
            out_parts.append(out_h)

        attn_out = torch.cat(out_parts, dim=1)
        attn_out = attn_out.transpose(1, 2).reshape(B, T, cache.nq * cache.hd)
        hidden = attn.o_proj(attn_out)
        hidden = residual + hidden

        residual = hidden
        hidden = layer.post_attention_layernorm(hidden)
        hidden = layer.mlp(hidden)
        hidden = residual + hidden

    cache.seq_len = T


__all__ = ["layer_major_prefill"]
