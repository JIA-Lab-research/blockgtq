"""Production Block-GTQ KV cache for decode.

This is the cache class that runs the latency / memory numbers in the
paper. It differs from :class:`blockgtq.packed_kv_manager.BlockGTQKVManager`
(the teaching variant) in three load-bearing ways:

  * K encoded with :func:`compress_k_mixed_batched` (nibble-4 packing for
    bit widths <= 4, raw uint8 for widths 5-8) instead of the bit-packed
    path. The mixed layout matches what the fused decode kernel's
    ``K_NIBBLE`` fast path expects.
  * V encoded with :func:`compress_v_nibble4_batched` (vpb = head_dim / 2)
    instead of the bit-packed V path. Pairs with the kernel's ``V_NIBBLE``
    fast path.
  * Decode passes ``k_nibble=True, v_nibble=True`` to
    :func:`fused_blockgtq_decode_attention`, which selects the
    table-driven paired-LUT decode paths that the bit-packed layout cannot
    feed.

CUDA Graph captures only the per-token K+V encode (2 kernel launches per
layer per token). Decode attention runs in a Python loop over KV heads —
each head's per-segment kernel-args layout is a Triton ``constexpr``, so a
single graph cannot span heads with different bit allocations.

Calibration is owned by the caller: build the per-(layer, head)
:class:`BlockGTQQuantizer` instances, batched encode args, and per-head
kernel args, then hand them in. See :mod:`blockgtq.calibration` for
helpers that do this from an HF model.
"""
import math

import torch


class BlockGTQProductionCache:
    """Production Block-GTQ KV cache with nibble-4 mixed encode + fused decode.

    The class is intentionally minimal: legacy mode only (CUDA-graphed T=1
    encode + per-KV-head fused attention). Long-context streaming (chunked
    encode, fp16 recent buffer, LSE merge) is out of scope for this class.
    """

    def __init__(self, quantizers, batched_args, kernel_args,
                 n_layers, nkv, nq, hd, max_T, v_bits=3,
                 k_nibble=True, v_nibble=True,
                 device='cuda'):
        self.quantizers = quantizers     # quantizers[li][hi]
        self.batched_args = batched_args  # batched_args[li] from build_batched_encode_args
        self.kernel_args = kernel_args    # kernel_args[li][hi] from build_kernel_args
        self.n_layers = n_layers
        self.nkv = nkv
        self.nq = nq
        self.hd = hd
        self.gqa = nq // nkv
        self.v_bits = v_bits
        self.k_nibble = k_nibble
        self.v_nibble = v_nibble
        self.device = device
        self.scale = 1.0 / math.sqrt(hd)
        self.seq_len = 0

        self.max_ng = max(quantizers[li][hi]._n_groups
                          for li in range(n_layers) for hi in range(nkv))

        if k_nibble:
            self.k_bytes_per_tok = max(ba['max_mixed_bytes'] for ba in batched_args)
        else:
            self.k_bytes_per_tok = max(ba['max_tpb'] for ba in batched_args)

        self.k_packed = torch.zeros(
            n_layers, 1, nkv, max_T, self.k_bytes_per_tok,
            dtype=torch.uint8, device=device)
        self.k_norms = torch.zeros(
            n_layers, 1, nkv, max_T, self.max_ng,
            dtype=torch.float16, device=device)

        if v_nibble:
            self.vpb = hd // 2
        else:
            from blockgtq.v_packing import packed_v_bytes
            self.vpb = packed_v_bytes(hd, v_bits)
        self.v_packed = torch.zeros(
            n_layers, 1, nkv, max_T, self.vpb,
            dtype=torch.uint8, device=device)
        self.v_norms = torch.zeros(
            n_layers, 1, nkv, max_T,
            dtype=torch.float16, device=device)

        self._cuda_graphs = [None] * n_layers
        self._graph_k_inputs = [None] * n_layers
        self._graph_v_inputs = [None] * n_layers
        self._graph_k_packed_outs = [None] * n_layers
        self._graph_k_norms_outs = [None] * n_layers
        self._graph_v_packed_outs = [None] * n_layers
        self._graph_v_norms_outs = [None] * n_layers

    # ------------------------------------------------------------------
    # Encode functions
    # ------------------------------------------------------------------

    def _k_encode_fn(self):
        if self.k_nibble:
            from blockgtq.quantizer import compress_k_mixed_batched
            return compress_k_mixed_batched
        from blockgtq.quantizer import compress_k_packed_batched
        return compress_k_packed_batched

    def _v_encode_fn(self):
        if self.v_nibble:
            from blockgtq.quantizer import compress_v_nibble4_batched
            return compress_v_nibble4_batched
        from blockgtq.quantizer import compress_v_packed_batched
        return compress_v_packed_batched

    def _build_cuda_graph(self, layer_idx):
        k_encode = self._k_encode_fn()
        v_encode = self._v_encode_fn()
        ba = self.batched_args[layer_idx]
        li = layer_idx
        H, D = self.nkv, self.hd

        k_in = torch.zeros(H, 1, D, dtype=torch.float16, device=self.device)
        v_in = torch.zeros(H, 1, D, dtype=torch.float16, device=self.device)
        self._graph_k_inputs[li] = k_in
        self._graph_v_inputs[li] = v_in

        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            k_encode(k_in, ba)
            v_encode(v_in, ba)
        torch.cuda.current_stream().wait_stream(s)

        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            kp, kn = k_encode(k_in, ba)
            vp, vn = v_encode(v_in, ba)

        self._cuda_graphs[li] = g
        self._graph_k_packed_outs[li] = kp
        self._graph_k_norms_outs[li] = kn
        self._graph_v_packed_outs[li] = vp
        self._graph_v_norms_outs[li] = vn

    def _encode_graphed(self, layer_idx, k, v, t_pos):
        """T=1 encode via CUDA Graph replay."""
        li = layer_idx
        H, D = self.nkv, self.hd
        ba = self.batched_args[li]

        if self._cuda_graphs[li] is None:
            self._build_cuda_graph(li)

        B = k.shape[0]
        self._graph_k_inputs[li].copy_(
            k.permute(1, 0, 2, 3).reshape(H, B, D))
        self._graph_v_inputs[li].copy_(
            v.permute(1, 0, 2, 3).reshape(H, B, D))

        self._cuda_graphs[li].replay()

        kp = self._graph_k_packed_outs[li]
        kn = self._graph_k_norms_outs[li]
        vp = self._graph_v_packed_outs[li]
        vn = self._graph_v_norms_outs[li]

        ba_ng = ba['max_n_groups']
        k_bytes = kp.shape[-1]
        self.k_packed[li, :, :, t_pos, :k_bytes] = kp.permute(1, 0, 2)
        self.k_norms[li, :, :, t_pos, :ba_ng] = kn.permute(1, 0, 2).to(self.k_norms.dtype)
        vpb = vp.shape[-1]
        self.v_packed[li, :, :, t_pos, :vpb] = vp.permute(1, 0, 2)
        self.v_norms[li, :, :, t_pos] = vn.permute(1, 0).to(torch.float16)

    def _encode_batched(self, layer_idx, k, v, t_pos):
        """Multi-token encode (used by online prefill)."""
        k_encode = self._k_encode_fn()
        v_encode = self._v_encode_fn()

        li = layer_idx
        ba = self.batched_args[li]
        H, D = self.nkv, self.hd
        B, _, S, _ = k.shape

        k_flat = k.permute(1, 0, 2, 3).reshape(H, B * S, D).contiguous()
        v_flat = v.permute(1, 0, 2, 3).reshape(H, B * S, D).contiguous()

        k_packed_raw, k_norms = k_encode(k_flat, ba)
        v_packed_raw, v_norms = v_encode(v_flat, ba)

        ba_ng = ba['max_n_groups']
        vpb = ba['v_nibble_bytes'] if self.v_nibble else ba['v_packed_bytes']
        k_bytes = k_packed_raw.shape[-1]
        k_packed_raw = k_packed_raw.reshape(H, B, S, k_bytes)
        k_norms = k_norms.reshape(H, B, S, ba_ng).to(self.k_norms.dtype)
        v_packed_raw = v_packed_raw.reshape(H, B, S, vpb)
        v_norms = v_norms.reshape(H, B, S)

        self.k_packed[li, :, :, t_pos:t_pos+S, :k_bytes] = k_packed_raw.permute(1, 0, 2, 3)
        self.k_norms[li, :, :, t_pos:t_pos+S, :ba_ng] = k_norms.permute(1, 0, 2, 3)
        self.v_packed[li, :, :, t_pos:t_pos+S, :vpb] = v_packed_raw.permute(1, 0, 2, 3)
        self.v_norms[li, :, :, t_pos:t_pos+S] = v_norms.permute(1, 0, 2).to(torch.float16)

    # ------------------------------------------------------------------
    # Decode entry
    # ------------------------------------------------------------------

    def encode_and_attend(self, layer_idx, q, k, v):
        """T=1 decode: encode (k_new, v_new) → fused packed attention.

        q: (1, n_q, 1, hd) post-RoPE.
        k, v: (1, n_kv, 1, hd) post-RoPE.

        Returns: (1, n_q, 1, hd) fp16.
        """
        from blockgtq.kernels.fused_packed_attention import fused_blockgtq_decode_attention

        li = layer_idx
        S = k.shape[2]
        t = self.seq_len
        T = t + S

        if S == 1:
            self._encode_graphed(li, k, v, t)
        else:
            self._encode_batched(li, k, v, t)

        block_t = 64
        raw_splits = max(1, min((T + block_t - 1) // block_t, 32))
        t_splits = 1
        while t_splits < raw_splits:
            t_splits *= 2

        out_parts = []
        for hi in range(self.nkv):
            q8 = self.quantizers[li][hi]
            ka = self.kernel_args[li][hi]

            q_h = q[:, hi * self.gqa:(hi + 1) * self.gqa, :, :]

            if q8._q_perm_baked:
                q_h_rot = q_h
            else:
                q_flat = q_h.reshape(-1, self.hd)
                q_rot = q8.rotate_q_for_packed_kernel(q_flat)
                q_h_rot = q_rot.reshape(q_h.shape)

            kp = self.k_packed[li, :, hi:hi+1, :T]
            kn = self.k_norms[li, :, hi:hi+1, :T]
            vp = self.v_packed[li, :, hi:hi+1, :T]
            vn = self.v_norms[li, :, hi:hi+1, :T]
            v_rot = getattr(q8, '_v_rot', None)

            attn_h = fused_blockgtq_decode_attention(
                q_h_rot, kp, kn,
                ka['codebook_flat'], ka['lut_offset'],
                ka['norm_group'], ka['segments_b'],
                vp, vn, q8.v_lut,
                scale=self.scale, t_splits=t_splits,
                v_bits=self.v_bits, v_rotation=v_rot,
                k_nibble=self.k_nibble, v_nibble=self.v_nibble,
            )
            out_parts.append(attn_h)

        return torch.cat(out_parts, dim=1)

    def increment_seq_len(self, by=1):
        self.seq_len += by

    # ------------------------------------------------------------------
    # Memory accounting (for bench)
    # ------------------------------------------------------------------

    def cache_memory_bytes(self) -> int:
        T = self.seq_len
        k_bytes = self.k_bytes_per_tok * T * self.nkv * self.n_layers
        v_bytes = self.vpb * T * self.nkv * self.n_layers
        k_norm_bytes = self.max_ng * 2 * T * self.nkv * self.n_layers
        v_norm_bytes = T * self.nkv * self.n_layers * 2
        return k_bytes + v_bytes + k_norm_bytes + v_norm_bytes

    def fp16_cache_bytes(self) -> int:
        T = self.seq_len
        return self.n_layers * self.nkv * T * self.hd * 2 * 2
