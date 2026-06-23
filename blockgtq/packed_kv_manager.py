"""Packed KV cache manager for E2E inference.

Combines Block-GTQ K compression + PolarQuant V compression
with fused_blockgtq_decode_attention for fully-packed decode-time attention.

K: BlockGTQQuantizer compress_packed → (packed_k, k_norms)
V: PolarQuantGPU via compress_v_packed → (packed_v, v_norms)
Attention: fused_blockgtq_decode_attention (in-kernel unpack for both K and V)

Per-head quantizers: each KV head gets its own BlockGTQ quantizer with
independent bit allocation, codebook, and packing layout. This is
necessary because different heads learn different attention patterns
and thus have different frequency importance profiles.

Usage:
    mgr = BlockGTQKVManager(n_layers, n_q_heads, n_kv_heads, head_dim,
                          k_avg_bits=3.0, v_bits=3, device='cuda')
    mgr.calibrate(layer_idx, q_data, k_data)  # (N,D) or (n_kv,N,D)
    mgr.init_from_cache(past_key_value)
    attn_out = mgr.append_and_attend(layer_idx, q, k_new, v_new)
    mgr.increment_seq_len()
"""
import math
import torch


class BlockGTQKVManager:
    """Manages packed KV storage and fused attention for all layers.

    After prefill (init_from_cache), K and V are stored as packed uint8.
    During decode (append_and_attend), new tokens are quantized+packed
    and attention is computed via the fused Triton kernel.

    Each KV head has its own BlockGTQ quantizer (per-head bit allocation).
    Decode dispatches one fused kernel per KV head (each handling its
    GQA Q-head group), since different heads may have different segment
    layouts (Triton constexpr).
    """

    def __init__(self, n_layers: int, n_q_heads: int, n_kv_heads: int,
                 head_dim: int, max_seq_len: int, batch_size: int = 1,
                 k_avg_bits: float = 3.0, v_bits: int = 3,
                 device='cuda'):
        self.n_layers = n_layers
        self.n_q = n_q_heads
        self.n_kv = n_kv_heads
        self.hd = head_dim
        self.max_T = max_seq_len
        self.B = batch_size
        self.k_avg_bits = k_avg_bits
        self.v_bits = v_bits
        self.device = device
        self.gqa_ratio = n_q_heads // n_kv_heads
        self.scale = 1.0 / math.sqrt(head_dim)
        self.seq_len = 0

        # Per-(layer, head) BlockGTQ quantizers
        self.quantizers = [[None] * n_kv_heads for _ in range(n_layers)]

        # Per-(layer, head) kernel dispatch args
        self._kernel_args = [[None] * n_kv_heads for _ in range(n_layers)]

        # Track max dimensions across all heads for padded storage
        self._max_k_packed_bytes = 0
        self._max_n_groups = 0

        # Per-layer batched encode args (built after calibration)
        self._batched_args = [None] * n_layers

        # Per-layer CUDA Graphs for decode (T=1) batched encode
        self._cuda_graphs = [None] * n_layers
        self._graph_k_inputs = [None] * n_layers
        self._graph_v_inputs = [None] * n_layers
        self._graph_k_packed_outs = [None] * n_layers
        self._graph_k_norms_outs = [None] * n_layers
        self._graph_v_packed_outs = [None] * n_layers
        self._graph_v_norms_outs = [None] * n_layers

        # Storage allocated lazily after calibration
        self._k_packed = None
        self._k_norms = None
        self._v_packed = None
        self._v_norms = None

    def calibrate(self, layer_idx: int, q_data: torch.Tensor,
                  k_data: torch.Tensor):
        """Calibrate per-head quantizers for one layer.

        q_data, k_data: (N, head_dim) shared across heads,
                        or (n_kv_heads, N, head_dim) per-head.
        """
        from blockgtq.quantizer import BlockGTQQuantizer

        # Support both (N, D) shared and (n_kv, N, D) per-head
        if q_data.dim() == 2:
            q_data = q_data.unsqueeze(0).expand(self.n_kv, -1, -1)
            k_data = k_data.unsqueeze(0).expand(self.n_kv, -1, -1)

        for hi in range(self.n_kv):
            q8 = BlockGTQQuantizer(
                head_dim=self.hd, avg_bits=self.k_avg_bits, device=self.device)
            q8.calibrate(q_data[hi].contiguous(), k_data[hi].contiguous())
            q8.init_v_quantizer(v_bits=self.v_bits, seed=42 + hi)
            self.quantizers[layer_idx][hi] = q8
            ka = q8.build_kernel_args()
            self._kernel_args[layer_idx][hi] = ka
            self._max_k_packed_bytes = max(
                self._max_k_packed_bytes, ka['total_packed_bytes'])
            self._max_n_groups = max(self._max_n_groups, q8._n_groups)

        # Build batched encode args for this layer
        from blockgtq.quantizer import build_batched_encode_args
        self._batched_args[layer_idx] = build_batched_encode_args(
            self.quantizers[layer_idx], device=self.device)

    def _ensure_storage(self):
        """Allocate padded storage (call after all layers calibrated)."""
        if self._k_packed is not None:
            return
        from blockgtq.v_packing import packed_v_bytes
        vpb = packed_v_bytes(self.hd, self.v_bits)

        self._k_packed = torch.zeros(
            self.n_layers, self.B, self.n_kv, self.max_T,
            self._max_k_packed_bytes,
            dtype=torch.uint8, device=self.device)
        self._k_norms = torch.zeros(
            self.n_layers, self.B, self.n_kv, self.max_T,
            self._max_n_groups,
            dtype=torch.float16, device=self.device)
        self._v_packed = torch.zeros(
            self.n_layers, self.B, self.n_kv, self.max_T, vpb,
            dtype=torch.uint8, device=self.device)
        self._v_norms = torch.zeros(
            self.n_layers, self.B, self.n_kv, self.max_T,
            dtype=torch.float16, device=self.device)

    def _encode_kv(self, layer_idx: int, k: torch.Tensor, v: torch.Tensor,
                   t_start: int, t_end: int):
        """Encode K/V into packed storage at positions [t_start, t_end).

        k, v: (B, n_kv, S, D) fp16

        Uses batched kernels: 2 launches for all heads instead of 2*n_kv.
        """
        self._ensure_storage()
        B, H, S, D = k.shape
        li = layer_idx
        ba = self._batched_args[li]

        from blockgtq.quantizer import (
            compress_k_packed_batched, compress_v_packed_batched)

        # Reshape to (n_kv, B*S, D) for batched dispatch
        k_flat = k.permute(1, 0, 2, 3).reshape(H, B * S, D).contiguous()
        v_flat = v.permute(1, 0, 2, 3).reshape(H, B * S, D).contiguous()

        # 1 launch: batched K encode (all heads)
        k_packed_all, k_norms_all = compress_k_packed_batched(k_flat, ba)
        # k_packed_all: (n_kv, B*S, max_tpb), k_norms_all: (n_kv, B*S, max_ng)

        # 1 launch: batched V encode (all heads)
        v_packed_all, v_norms_all = compress_v_packed_batched(v_flat, ba)
        # v_packed_all: (n_kv, B*S, vpb), v_norms_all: (n_kv, B*S)

        # Write to storage
        max_tpb = ba['max_tpb']
        max_ng = ba['max_n_groups']
        vpb = ba['v_packed_bytes']

        k_packed_all = k_packed_all.reshape(H, B, S, max_tpb)
        k_norms_all = k_norms_all.reshape(H, B, S, max_ng).to(torch.float16)
        v_packed_all = v_packed_all.reshape(H, B, S, vpb)
        v_norms_all = v_norms_all.reshape(H, B, S)

        # (n_kv, B, S, ...) → storage (layers, B, n_kv, T, ...)
        self._k_packed[li, :, :, t_start:t_end, :max_tpb] = k_packed_all.permute(1, 0, 2, 3)
        self._k_norms[li, :, :, t_start:t_end, :max_ng] = k_norms_all.permute(1, 0, 2, 3)
        self._v_packed[li, :, :, t_start:t_end, :vpb] = v_packed_all.permute(1, 0, 2, 3)
        self._v_norms[li, :, :, t_start:t_end] = v_norms_all.permute(1, 0, 2)

    def _build_cuda_graph(self, layer_idx: int):
        """Capture CUDA Graph for T=1 batched encode on a specific layer.

        Called once per layer on first decode step. Graph captures the 2
        batched kernel launches (K + V) for replay on subsequent steps.
        """
        from blockgtq.quantizer import (
            compress_k_packed_batched, compress_v_packed_batched)

        ba = self._batched_args[layer_idx]
        li = layer_idx
        H, D = self.n_kv, self.hd

        # Static input buffers (overwritten before each replay)
        k_in = torch.zeros(H, 1, D, dtype=torch.float16, device=self.device)
        v_in = torch.zeros(H, 1, D, dtype=torch.float16, device=self.device)
        self._graph_k_inputs[li] = k_in
        self._graph_v_inputs[li] = v_in

        # Warmup (required before graph capture)
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            compress_k_packed_batched(k_in, ba)
            compress_v_packed_batched(v_in, ba)
        torch.cuda.current_stream().wait_stream(s)

        # Capture
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            kp, kn = compress_k_packed_batched(k_in, ba)
            vp, vn = compress_v_packed_batched(v_in, ba)

        self._cuda_graphs[li] = g
        self._graph_k_packed_outs[li] = kp
        self._graph_k_norms_outs[li] = kn
        self._graph_v_packed_outs[li] = vp
        self._graph_v_norms_outs[li] = vn

    def _encode_kv_graphed(self, layer_idx: int, k: torch.Tensor,
                           v: torch.Tensor, t_pos: int):
        """T=1 encode via CUDA Graph replay (single-token decode).

        k, v: (B, n_kv, 1, D) fp16
        """
        self._ensure_storage()
        li = layer_idx
        H, D = self.n_kv, self.hd
        ba = self._batched_args[li]

        # Build graph on first use for this layer
        if self._cuda_graphs[li] is None:
            self._build_cuda_graph(li)

        # Copy inputs into static buffers
        B = k.shape[0]
        self._graph_k_inputs[li].copy_(k.permute(1, 0, 2, 3).reshape(H, B, D))
        self._graph_v_inputs[li].copy_(v.permute(1, 0, 2, 3).reshape(H, B, D))

        # Replay
        self._cuda_graphs[li].replay()

        # Write outputs to storage
        max_tpb = ba['max_tpb']
        max_ng = ba['max_n_groups']
        vpb = ba['v_packed_bytes']

        kp = self._graph_k_packed_outs[li].reshape(H, B, 1, max_tpb)
        kn = self._graph_k_norms_outs[li].reshape(H, B, 1, max_ng).to(torch.float16)
        vp = self._graph_v_packed_outs[li].reshape(H, B, 1, vpb)
        vn = self._graph_v_norms_outs[li].reshape(H, B, 1)

        self._k_packed[li, :, :, t_pos:t_pos+1, :max_tpb] = kp.permute(1, 0, 2, 3)
        self._k_norms[li, :, :, t_pos:t_pos+1, :max_ng] = kn.permute(1, 0, 2, 3)
        self._v_packed[li, :, :, t_pos:t_pos+1, :vpb] = vp.permute(1, 0, 2, 3)
        self._v_norms[li, :, :, t_pos:t_pos+1] = vn.permute(1, 0, 2)

    def init_from_cache(self, past_key_value):
        """Convert HF DynamicCache to packed format after prefill."""
        for li in range(self.n_layers):
            k = past_key_value.key_cache[li]   # (B, n_kv, T, D)
            v = past_key_value.value_cache[li]  # (B, n_kv, T, D)
            T = k.shape[2]
            self._encode_kv(li, k, v, 0, T)
        self.seq_len = T

    @torch.no_grad()
    def append_and_attend(self, layer_idx: int, q: torch.Tensor,
                          k_new: torch.Tensor, v_new: torch.Tensor
                          ) -> torch.Tensor:
        """Quantize new K/V, append to cache, run fused packed attention.

        q: (B, n_q, 1, D) fp16
        k_new, v_new: (B, n_kv, 1, D) fp16

        Returns: (B, n_q, 1, D) fp16

        Dispatches one fused kernel per KV head (each head may have
        different segment layouts from per-head bit allocation).
        """
        from blockgtq.kernels.fused_packed_attention import fused_blockgtq_decode_attention

        t = self.seq_len
        S = k_new.shape[2]
        if S == 1 and self.B == 1:
            # T=1 decode: use CUDA Graph replay
            self._encode_kv_graphed(layer_idx, k_new, v_new, t)
        else:
            self._encode_kv(layer_idx, k_new, v_new, t, t + S)
        T = t + S
        li = layer_idx

        # t_splits: number of parallel time-range blocks.
        # Optimal: block_t=64, t_splits=32 (100% SM utilization on H800).
        block_t = 64
        raw_splits = max(1, min((T + block_t - 1) // block_t, 32))
        t_splits = 1
        while t_splits < raw_splits:
            t_splits *= 2

        gqa = self.gqa_ratio
        out_parts = []

        for hi in range(self.n_kv):
            q8 = self.quantizers[li][hi]
            ka = self._kernel_args[li][hi]

            # Slice Q to this KV head's GQA group
            q_h = q[:, hi * gqa:(hi + 1) * gqa, :, :]

            # Per-head packed storage slices
            kp = self._k_packed[li, :, hi:hi + 1, :T]
            kn = self._k_norms[li, :, hi:hi + 1, :T]
            vp = self._v_packed[li, :, hi:hi + 1, :T]
            v_norms = self._v_norms[li, :, hi:hi + 1, :T]

            # Q rotation+permutation: rotate Q into packed K's rotated space.
            # rotate_q_for_packed_kernel does: permute → rotate → pack_perm.
            # If Q perm is baked into q_proj, we still need the rotation,
            # so always use rotate_q_for_packed_kernel for un-baked case.
            if q8._q_perm_baked:
                # TODO: bake rotation into q_proj weight for baked case
                q_h_rot = q_h
            else:
                # Reshape for rotate: (B, gqa, 1, D) → (B*gqa, D)
                q_flat = q_h.reshape(-1, q_h.shape[-1])
                q_rot = q8.rotate_q_for_packed_kernel(q_flat)
                q_h_rot = q_rot.reshape(q_h.shape)

            # V un-rotation
            v_rot = getattr(q8, '_v_rot', None)

            attn_h = fused_blockgtq_decode_attention(
                q_h_rot, kp, kn,
                ka['codebook_flat'],
                ka['lut_offset'],
                ka['norm_group'],
                ka['segments_b'],
                vp, v_norms, q8.v_lut,
                scale=self.scale,
                t_splits=t_splits,
                v_bits=self.v_bits,
                v_rotation=v_rot,
            )
            out_parts.append(attn_h)

        return torch.cat(out_parts, dim=1)

    def increment_seq_len(self):
        """Call after all layers have processed the new decode token."""
        self.seq_len += 1

    def bake_q_permutations(self, model, n_kv_heads: int, n_q_heads: int):
        """Bake per-head composite Q permutation into q_proj for all layers.

        Each KV head's quantizer bakes its own permutation into the
        corresponding GQA Q-head group of q_proj.
        """
        from blockgtq.packed_kv_manager import _get_model_layers
        layers = _get_model_layers(model)

        for li in range(self.n_layers):
            attn = layers[li].self_attn
            for hi in range(self.n_kv):
                q8 = self.quantizers[li][hi]
                q8.bake_permutation_into_qproj(
                    attn.q_proj.weight.data,
                    kv_head_idx=hi,
                    n_kv_heads=n_kv_heads,
                    n_q_heads=n_q_heads,
                    q_proj_bias=getattr(attn.q_proj, 'bias', None),
                )

    def cache_memory_bytes(self) -> int:
        """Compressed KV cache size in bytes (allocated, including padding)."""
        T = self.seq_len
        k_bytes = self._max_k_packed_bytes * T * self.n_kv * self.B * self.n_layers
        v_bytes = self._v_packed[0, 0, 0, 0].numel() * T * self.n_kv * self.B * self.n_layers
        k_norm_bytes = self._max_n_groups * 2 * T * self.n_kv * self.B * self.n_layers
        v_norm_bytes = T * self.n_kv * self.B * self.n_layers * 2  # fp16
        return k_bytes + v_bytes + k_norm_bytes + v_norm_bytes

    def fp16_cache_bytes(self) -> int:
        """Equivalent fp16 KV cache size."""
        T = self.seq_len
        return self.n_layers * self.B * self.n_kv * T * self.hd * 2 * 2


# ============================================================================
# Model hooks for E2E decode
# ============================================================================

def _get_model_layers(model):
    """Return the list of transformer layers from a HuggingFace model."""
    if hasattr(model, 'model') and hasattr(model.model, 'layers'):
        return list(model.model.layers)
    raise ValueError("Cannot find model layers — expected model.model.layers")


def decode_step(model, input_ids, position_ids, kv_manager: BlockGTQKVManager):
    """One full decode step through all layers using fused packed attention.

    input_ids: (B, 1) token ids
    position_ids: (B, 1) position ids
    Returns: logits (B, 1, vocab_size)
    """
    hidden = model.model.embed_tokens(input_ids)

    for li, layer in enumerate(_get_model_layers(model)):
        residual = hidden
        hidden = layer.input_layernorm(hidden)

        attn = layer.self_attn
        B, S, _ = hidden.shape

        q = attn.q_proj(hidden)
        k = attn.k_proj(hidden)
        v = attn.v_proj(hidden)

        n_q = kv_manager.n_q
        n_kv = kv_manager.n_kv
        hd = kv_manager.hd

        q = q.view(B, S, n_q, hd).transpose(1, 2)
        k = k.view(B, S, n_kv, hd).transpose(1, 2)
        v = v.view(B, S, n_kv, hd).transpose(1, 2)

        # RoPE
        _rope = getattr(attn, 'rotary_emb', None) or model.model.rotary_emb
        cos, sin = _rope(v, position_ids)
        from transformers.models.llama.modeling_llama import apply_rotary_pos_emb
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        # Fused quantized attention
        attn_out = kv_manager.append_and_attend(li, q, k, v)

        attn_out = attn_out.transpose(1, 2).reshape(B, S, n_q * hd)
        hidden = attn.o_proj(attn_out)
        hidden = residual + hidden

        # MLP
        residual = hidden
        hidden = layer.post_attention_layernorm(hidden)
        hidden = layer.mlp(hidden)
        hidden = residual + hidden

    # All layers done — advance cache position
    kv_manager.increment_seq_len()

    hidden = model.model.norm(hidden)
    logits = model.lm_head(hidden)
    return logits
