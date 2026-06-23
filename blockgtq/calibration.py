"""Calibration helpers: collect Q/K activations, build per-head quantizers, bake rotations.

These utilities turn an HF Llama-style model and a calibration token stream
into the ``(quantizers, batched_args, kernel_args)`` triple that
:class:`blockgtq.production_cache.BlockGTQProductionCache` expects.

Recommended workflow::

    from blockgtq import (
        BlockGTQProductionCache,
        BlockGTQQuantizer,
    )
    from blockgtq.quantizer import build_batched_encode_args
    from blockgtq.calibration import (
        collect_qk_activations, build_quantizers, bake_q_rotations,
    )

    layer_data = collect_qk_activations(model, calib_ids, device, post_rope=True)
    quantizers, batched_args, kernel_args = build_quantizers(
        layer_data, n_layers, nkv, hd,
        k_avg_bits=3.0, v_bits=3, k_nibble=True, device=device,
    )
    bake_q_rotations(model, quantizers, n_layers, nkv, nq)

    cache = BlockGTQProductionCache(
        quantizers, batched_args, kernel_args,
        n_layers=n_layers, nkv=nkv, nq=nq, hd=hd, max_T=max_T,
        v_bits=3, k_nibble=True, v_nibble=True, device=device,
    )
"""
import torch


@torch.no_grad()
def collect_qk_activations(model, enc, device, n_calib_tokens=256,
                           post_rope=True, effective_len=None):
    """Collect per-(layer, kv-head) Q/K activations for calibration.

    Args:
        model: HF Llama-style model.
        enc: (1, T) input ids; only the first ``n_calib_tokens`` are used.
        device: target device.
        n_calib_tokens: number of tokens to forward through the model.
        post_rope: if True (recommended for production), apply RoPE to the
            collected Q/K so the quantizer sees the same distribution the
            decode path encodes. Required for ``k_nibble`` cache layout.
        effective_len: if set, run the model with stretched position_ids
            covering ``[0, effective_len)`` at stride
            ``effective_len/n_calib_tokens``. Lets short calib data cover a
            longer-context RoPE phase distribution.

    Returns:
        Dict ``{layer_idx: {'q': (nkv, gqa*T, hd), 'k': (nkv, T, hd)}}``.
    """
    ids = enc[:, :n_calib_tokens].to(device)
    hd = model.config.hidden_size // model.config.num_attention_heads
    nkv = getattr(model.config, 'num_key_value_heads',
                  model.config.num_attention_heads)
    nq = model.config.num_attention_heads
    gqa = nq // nkv

    T = ids.size(1)
    if effective_len is not None:
        if effective_len < T:
            raise ValueError(
                f"effective_len ({effective_len}) must be >= n_calib_tokens ({T})")
        scale = effective_len // T
        position_ids = torch.arange(0, T * scale, scale, device=device).unsqueeze(0)
    else:
        position_ids = None

    layer_data = {}
    handles = []
    for li, layer in enumerate(model.model.layers):
        attn = layer.self_attn
        layer_data[li] = {}

        def make_k_hook(li_):
            def fn(mod, inp, out):
                k = out.detach().reshape(-1, nkv, hd)
                layer_data[li_]['k'] = k.permute(1, 0, 2)
            return fn

        def make_q_hook(li_):
            def fn(mod, inp, out):
                q = out.detach().reshape(-1, nq, hd)
                q_grouped = q.reshape(-1, nkv, gqa, hd)
                layer_data[li_]['q'] = q_grouped.permute(1, 2, 0, 3).reshape(nkv, -1, hd)
            return fn

        handles.append(attn.k_proj.register_forward_hook(make_k_hook(li)))
        handles.append(attn.q_proj.register_forward_hook(make_q_hook(li)))

    if position_ids is not None:
        model(ids, position_ids=position_ids)
    else:
        model(ids)
    for h in handles:
        h.remove()

    if post_rope:
        from transformers.models.llama.modeling_llama import apply_rotary_pos_emb
        rope = (getattr(model.model.layers[0].self_attn, 'rotary_emb', None)
                or model.model.rotary_emb)
        T = min(n_calib_tokens, ids.size(1))
        if effective_len is not None:
            scale = effective_len // T
            position_ids = torch.arange(0, T * scale, scale, device=device).unsqueeze(0)
        else:
            position_ids = torch.arange(T, device=device).unsqueeze(0)
        dummy = torch.zeros(1, 1, T, hd, device=device, dtype=torch.float16)
        cos, sin = rope(dummy, position_ids)

        for li in layer_data:
            k = layer_data[li]['k']
            q = layer_data[li]['q']
            T_actual = k.shape[1]
            q_3d = q.reshape(-1, T_actual, hd)
            q_r, _ = apply_rotary_pos_emb(
                q_3d.unsqueeze(0).to(cos.dtype),
                q_3d.unsqueeze(0).to(cos.dtype), cos, sin)
            _, k_r = apply_rotary_pos_emb(
                k.unsqueeze(0).to(cos.dtype),
                k.unsqueeze(0).to(cos.dtype), cos, sin)
            layer_data[li]['q'] = q_r[0].reshape(nkv, -1, hd).to(q.dtype)
            layer_data[li]['k'] = k_r[0].to(k.dtype)

    return layer_data


def build_quantizers(layer_data, n_layers, nkv, hd,
                     k_avg_bits=3.0, v_bits=3, k_nibble=True,
                     device='cuda'):
    """Build per-(layer, head) quantizers + batched encode args + kernel args.

    Returns:
        Tuple ``(quantizers, batched_args, kernel_args)``:
          * ``quantizers[li][hi]``: calibrated :class:`BlockGTQQuantizer`.
          * ``batched_args[li]``: dict from :func:`build_batched_encode_args`.
          * ``kernel_args[li][hi]``: dict from
            ``quantizer.build_kernel_args``.
    """
    from blockgtq.quantizer import BlockGTQQuantizer, build_batched_encode_args

    quantizers = [[None] * nkv for _ in range(n_layers)]
    kernel_args = [[None] * nkv for _ in range(n_layers)]
    batched_args = [None] * n_layers

    for li in range(n_layers):
        q_data = layer_data[li]['q']
        k_data = layer_data[li]['k']
        for hi in range(nkv):
            q8 = BlockGTQQuantizer(
                head_dim=hd, avg_bits=k_avg_bits, device=device,
            )
            q8.calibrate(q_data[hi].contiguous(), k_data[hi].contiguous())
            q8.init_v_quantizer(v_bits=v_bits, seed=42 + hi)
            quantizers[li][hi] = q8
            kernel_args[li][hi] = q8.build_kernel_args()
        batched_args[li] = build_batched_encode_args(quantizers[li], device=device)

    return quantizers, batched_args, kernel_args


def bake_q_rotations(model, quantizers, n_layers, nkv, nq):
    """Bake per-head Q permutation + rotation into ``q_proj`` weights (in-place).

    This is a one-time amortisation: after baking, the decode path no
    longer needs ``rotate_q_for_packed_kernel`` per step.
    """
    for li in range(n_layers):
        attn = model.model.layers[li].self_attn
        for hi in range(nkv):
            q8 = quantizers[li][hi]
            q8.bake_permutation_into_qproj(
                attn.q_proj.weight.data,
                kv_head_idx=hi,
                n_kv_heads=nkv,
                n_q_heads=nq,
                q_proj_bias=getattr(attn.q_proj, 'bias', None),
            )


__all__ = [
    "collect_qk_activations",
    "build_quantizers",
    "bake_q_rotations",
]
