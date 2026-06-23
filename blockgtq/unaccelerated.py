"""Unaccelerated Block-GTQ: pure-PyTorch path for quality validation.

This is the *unaccelerated* counterpart to the fused production path
(:mod:`blockgtq.production_cache` / :mod:`blockgtq.production_decode`). It
applies Block-GTQ as real-but-unpacked quantization: every layer's
``k_proj`` / ``v_proj`` output is quantized and immediately dequantized in
floating point, then ordinary attention runs on the reconstructed K/V. There
is no packed cache and no custom kernel, so this path:

  * runs on **any** PyTorch device (including CPU),
  * matches the **quality** (NIAH, perplexity) of the fused path,

as opposed to the production path, which delivers the **memory / latency**
wins on Hopper. The quantization is genuine — only the storage is left
unpacked (full-width fp tensors), so the measured quality reflects the true
Block-GTQ reconstruction error.

K side: per-(layer, KV-head)
:class:`~blockgtq.block_gtq_pipeline.BlockGTQPipeline` (RoPE-aware non-uniform
bit allocation). V side: per-head :class:`~blockgtq.tq.TurboQuantMSE`
(uniform), the project convention.

**Calibrate pre-RoPE.** The patch wraps the *pre-RoPE* ``k_proj`` output, and
attention applies RoPE afterwards, so the calibrate point and the apply point
must match: pass ``collect_qk_activations(..., post_rope=False)``. (The
production path calibrates with ``post_rope=True`` instead, because its kernel
applies RoPE internally.)

Example::

    from blockgtq import collect_qk_activations
    from blockgtq.unaccelerated import (
        build_unaccelerated_quantizers, patch_model_kv, unpatch_model_kv,
    )

    layer_data = collect_qk_activations(model, calib_ids, device,
                                        n_calib_tokens=2048, post_rope=False)
    kq, vq = build_unaccelerated_quantizers(
        layer_data, n_layers, nkv, hd, k_avg_bits=3.0, v_bits=3, device=device)
    handles = patch_model_kv(model, kq, vq, hd, nkv)
    # ... run the model normally; K/V are Block-GTQ quantized on the fly ...
    unpatch_model_kv(handles)
"""
import torch
import torch.nn as nn

__all__ = [
    "build_unaccelerated_quantizers",
    "patch_model_kv",
    "unpatch_model_kv",
    "patch_model_kv_buffered",
]


@torch.no_grad()
def build_unaccelerated_quantizers(layer_data, n_layers, nkv, hd,
                                   k_avg_bits=3.0, v_bits=3,
                                   rotation_threshold=1, min_bits=1, max_bits=8,
                                   v_seed_base=1000, device="cuda"):
    """Build per-(layer, KV-head) K and V quantizers for the unaccelerated path.

    Args:
        layer_data: output of
            :func:`blockgtq.calibration.collect_qk_activations` with
            ``post_rope=False`` — ``{li: {'q': (nkv, gqa*T, hd),
            'k': (nkv, T, hd)}}``. Q holds every GQA query head as a sample
            (the correct energy convention); K is pre-RoPE.
        n_layers, nkv, hd: layer count, KV-head count, head dim.
        k_avg_bits: average K bit budget per dim (the allocator distributes it
            non-uniformly across RoPE blocks).
        v_bits: uniform V bit width.
        rotation_threshold, min_bits, max_bits: passed to
            :class:`~blockgtq.block_gtq_pipeline.BlockGTQPipeline`.
        v_seed_base: V quantizer seed is ``v_seed_base + head_index``
            (default 1000).
        device: target device for the quantizer tensors.

    Returns:
        ``(k_quantizers, v_quantizers)``, each an ``[li][hi]`` nested list.
    """
    from blockgtq.block_gtq_pipeline import BlockGTQPipeline
    from blockgtq.tq import TurboQuantMSE

    dev = torch.device(device)
    k_quantizers = [[None] * nkv for _ in range(n_layers)]
    v_quantizers = [[None] * nkv for _ in range(n_layers)]

    for li in range(n_layers):
        q_data = layer_data[li]["q"]   # (nkv, gqa*T, hd)
        k_data = layer_data[li]["k"]   # (nkv, T, hd)
        for hi in range(nkv):
            kq = BlockGTQPipeline(
                head_dim=hd, avg_bits=float(k_avg_bits),
                rotation_threshold=rotation_threshold,
                min_bits=min_bits, max_bits=max_bits, device=dev)
            kq.calibrate(q_data[hi].to(dev).float().contiguous(),
                         k_data[hi].to(dev).float().contiguous())
            k_quantizers[li][hi] = kq
            v_quantizers[li][hi] = TurboQuantMSE(
                d=hd, bit_width=v_bits, seed=v_seed_base + hi, device=dev)

    return k_quantizers, v_quantizers


class _UnaccelProj(nn.Module):
    """Wrap a ``k_proj`` / ``v_proj`` so its per-head (pre-RoPE) output is
    quantized then dequantized by a per-head quantizer.

    The wrapped quantizers expose ``compress_decompress((N, hd)) -> (N, hd)``
    (both :class:`BlockGTQPipeline` and :class:`TurboQuantMSE` do). We flatten
    to 2-D per head because ``TurboQuantMSE`` assumes a 2-D batch.
    """

    def __init__(self, orig_proj, head_dim, n_kv_heads, head_quantizers):
        super().__init__()
        self.orig = orig_proj
        self.hd = head_dim
        self.nkv = n_kv_heads
        self.qs = head_quantizers  # list[len == nkv]

    @torch.no_grad()
    def forward(self, x, **kwargs):
        out = self.orig(x, **kwargs)
        batch_shape = out.shape[:-1]
        kv = out.float().reshape(*batch_shape, self.nkv, self.hd)
        rec = torch.empty_like(kv)
        for h in range(self.nkv):
            flat = kv[..., h, :].reshape(-1, self.hd)
            rec[..., h, :] = self.qs[h].compress_decompress(flat).reshape(
                *batch_shape, self.hd)
        return rec.reshape(*batch_shape, self.nkv * self.hd).to(out.dtype)


def patch_model_kv(model, k_quantizers, v_quantizers, head_dim, n_kv_heads):
    """Wrap every layer's ``k_proj`` / ``v_proj`` with quantize-dequantize.

    Args:
        model: HF Llama-style model (``model.model.layers[*].self_attn``).
        k_quantizers, v_quantizers: ``[li][hi]`` nested lists from
            :func:`build_unaccelerated_quantizers`.
        head_dim, n_kv_heads: per-head dim and KV-head count.

    Returns:
        ``handles`` to pass to :func:`unpatch_model_kv`.
    """
    handles = []
    for li, layer in enumerate(model.model.layers):
        attn = layer.self_attn
        orig_k, orig_v = attn.k_proj, attn.v_proj
        attn.k_proj = _UnaccelProj(orig_k, head_dim, n_kv_heads, k_quantizers[li])
        attn.v_proj = _UnaccelProj(orig_v, head_dim, n_kv_heads, v_quantizers[li])
        handles.append((attn, orig_k, orig_v))
    return handles


def unpatch_model_kv(handles):
    """Restore the original ``k_proj`` / ``v_proj`` modules."""
    for attn, orig_k, orig_v in handles:
        attn.k_proj = orig_k
        attn.v_proj = orig_v


class _BufferedKVPatch:
    """PM-KVQ-aligned three-region KV cache via a post-forward hook.

    Cache layout per layer / KV head, maintained as generation extends it::

        [ sink (n_sink_token, fp16) | quantized blocks | recent fp16 tail (window) ]

    After every model forward, the hook inspects ``past_key_values`` and, once
    the fp16 tail exceeds ``recent_window``, round-trips the oldest
    ``block_size`` tokens through the per-(layer, KV-head) quantizer **in
    place**. The sink prefix and the recent tail stay fp16; everything between
    is quantized — exactly the discipline a streaming KV-quant deployment with
    a recent-token buffer uses.

    Reuses the same quantizers as :func:`patch_model_kv`. Block-GTQ's
    per-RoPE-block bit allocation is rotation-invariant (RoPE rotates within a
    2-D block, preserving its energy), and ``TurboQuantMSE`` is data-free, so
    the (pre-RoPE-calibrated) quantizers apply correctly to the post-RoPE keys
    held in the cache and flushed here. Call :meth:`reset` before each fresh
    generation (the flush counters are per-sequence).
    """

    def __init__(self, model, k_quantizers, v_quantizers, head_dim, n_kv_heads,
                 n_layers, block_size=1, recent_window=128, n_sink_token=4):
        self.kq = k_quantizers
        self.vq = v_quantizers
        self.hd = head_dim
        self.nkv = n_kv_heads
        self.n_layers = n_layers
        self.block_size = int(block_size)
        self.recent_window = int(recent_window)
        self.n_sink = int(n_sink_token)
        self.state = [{"n_flushed": 0} for _ in range(n_layers)]
        self._hook = model.register_forward_hook(self._hook_fn, with_kwargs=True)

    def reset(self):
        """Reset per-sequence flush counters — call before each generation."""
        for st in self.state:
            st["n_flushed"] = 0

    @torch.no_grad()
    def _hook_fn(self, module, args, kwargs, output):
        pkv = kwargs.get("past_key_values", None)
        if pkv is None:
            pkv = getattr(output, "past_key_values", None)
        if pkv is None:
            return
        key_cache = getattr(pkv, "key_cache", None)
        value_cache = getattr(pkv, "value_cache", None)
        if key_cache is None or value_cache is None:
            return
        for li in range(min(self.n_layers, len(key_cache))):
            k = key_cache[li]
            v = value_cache[li]
            if k is None or k.numel() == 0 or k.shape[-1] != self.hd:
                continue
            seq = k.shape[-2]
            st = self.state[li]
            avail = max(0, seq - self.n_sink)
            needed = self.block_size + self.recent_window
            while avail - st["n_flushed"] >= needed:
                start = self.n_sink + st["n_flushed"]
                end = start + self.block_size
                self._flush(k, v, li, start, end)
                st["n_flushed"] += self.block_size

    @torch.no_grad()
    def _flush(self, k_ref, v_ref, li, start, end):
        B = k_ref.shape[0]
        bs = end - start
        for h in range(self.nkv):
            ks = k_ref[:, h, start:end, :].reshape(-1, self.hd).float()
            k_ref[:, h, start:end, :] = self.kq[li][h].compress_decompress(
                ks).reshape(B, bs, self.hd).to(k_ref.dtype)
            vs = v_ref[:, h, start:end, :].reshape(-1, self.hd).float()
            v_ref[:, h, start:end, :] = self.vq[li][h].compress_decompress(
                vs).reshape(B, bs, self.hd).to(v_ref.dtype)

    def remove(self):
        if self._hook is not None:
            self._hook.remove()
            self._hook = None


def patch_model_kv_buffered(model, k_quantizers, v_quantizers, head_dim,
                            n_kv_heads, n_layers, block_size=1,
                            recent_window=128, n_sink_token=4):
    """Install the buffered (three-region) KV patch — sink + quantized +
    recent-fp16-tail. Returns a handle exposing ``.reset()`` (call before each
    generation) and ``.remove()`` (uninstall). See :class:`_BufferedKVPatch`."""
    return _BufferedKVPatch(model, k_quantizers, v_quantizers, head_dim,
                            n_kv_heads, n_layers, block_size=block_size,
                            recent_window=recent_window,
                            n_sink_token=n_sink_token)
