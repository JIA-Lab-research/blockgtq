"""Model-aware production decode + online prefill glue.

These functions assume a Llama-style HF transformer model exposing
``model.model.embed_tokens``, ``model.model.layers``, ``model.model.norm``,
and ``model.lm_head`` (covers Llama, Qwen2, DeepSeek-distill variants).
"""
import torch


@torch.no_grad()
def production_decode_step(model, input_ids, position_ids, cache):
    """One decode step using :class:`BlockGTQProductionCache`.

    The cache class' ``encode_and_attend`` does both K/V encode and fused
    decode attention; this function provides the surrounding model glue
    (embed -> per-layer (norm, qkv proj, RoPE, attention, MLP) -> final
    norm + lm_head).
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

        attn_out = cache.encode_and_attend(li, q, k, v)

        attn_out = attn_out.transpose(1, 2).reshape(B, S, cache.nq * cache.hd)
        hidden = attn.o_proj(attn_out)
        hidden = residual + hidden

        residual = hidden
        hidden = layer.post_attention_layernorm(hidden)
        hidden = layer.mlp(hidden)
        hidden = residual + hidden

    cache.increment_seq_len()
    hidden = model.model.norm(hidden)
    return model.lm_head(hidden)


@torch.no_grad()
def production_prefill(model, input_ids, cache, chunk_encode=64):
    """Online prefill: HF forward to get post-RoPE K/V, then bulk-encode.

    Uses the standard HF attention path for prefill (correct & fast); then
    extracts post-RoPE K/V from ``DynamicCache`` and bulk-encodes them into
    the packed cache. For T >= 64K, prefer
    :func:`blockgtq.prefill.layer_major_prefill` which avoids materialising
    an fp16 KV cache during prefill.

    Args:
        model: HF CausalLM model.
        input_ids: (1, prompt_len) token ids.
        cache: :class:`BlockGTQProductionCache` instance, fresh.
        chunk_encode: encode this many tokens per ``_encode_batched`` call
            to bound scratch memory.

    Returns:
        (1, prompt_len, vocab) logits from the HF forward.
    """
    from transformers import DynamicCache

    prompt_len = input_ids.shape[1]
    device = input_ids.device
    position_ids = torch.arange(prompt_len, device=device).unsqueeze(0)

    hf_cache = DynamicCache()
    out = model(
        input_ids, position_ids=position_ids,
        past_key_values=hf_cache, use_cache=True,
    )

    for li in range(cache.n_layers):
        k = hf_cache.key_cache[li]
        v = hf_cache.value_cache[li]
        for start in range(0, prompt_len, chunk_encode):
            end = min(start + chunk_encode, prompt_len)
            cache._encode_batched(li, k[:, :, start:end], v[:, :, start:end], start)

    cache.seq_len = prompt_len
    del hf_cache
    torch.cuda.empty_cache()
    return out.logits


__all__ = ["production_decode_step", "production_prefill"]
