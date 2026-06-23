"""Block-GTQ Triton quantizer: compress-only with 1-4 bit packing.

This is the production K-cache encoder used by :class:`BlockGTQKVManager` and
the fused decode kernel in :mod:`blockgtq.kernels.fused_packed_attention`. Bit-widths
1-4 are packed sub-byte in-kernel; widths 5-8 are stored as raw uint8 so the
decode side only has to handle four sub-byte formats.
"""

from typing import Optional

import torch
import math
import numpy as np

try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False


# ===========================================================================
# Code LUT construction
# ===========================================================================

from blockgtq.kernels.quant_kernels import build_code_lut


# ===========================================================================
# Packing metadata — packs 1-4 bit, 5-8 raw
# ===========================================================================

def build_pack_meta(bit_allocation: torch.Tensor, head_perm: torch.Tensor) -> dict:
    """Compute bit-packing segment metadata from bit allocation.

    Bit-widths 1-4 are packed sub-byte; widths 5-8 are stored as raw uint8.

    Returns dict keyed by bit-width (1..8) + 'total_packed_bytes',
    'nopack_start', 'nopack_len', 'nopack_off', 'pack_perm'.
    Each bw entry: {'start': int, 'length': int, 'pack_offset': int, 'packed_bytes': int}
    """
    HD = head_perm.shape[0]
    bits_per_dim = torch.cat([bit_allocation, bit_allocation]).cpu()
    permuted_bits = bits_per_dim[head_perm.cpu()]

    pack_perm = torch.argsort(permuted_bits, stable=True)
    sorted_bits = permuted_bits[pack_perm]

    meta = {}
    pack_offset = 0

    for bw in range(1, 9):
        mask = (sorted_bits == bw)
        if not mask.any():
            continue
        indices = mask.nonzero(as_tuple=True)[0]
        start = indices[0].item()
        length = indices.shape[0]
        assert indices[-1].item() == start + length - 1, \
            f"Dims with bw={bw} not contiguous after pack_perm sort"

        if bw == 1:
            packed_bytes = (length + 7) // 8
        elif bw == 2:
            packed_bytes = (length + 3) // 4
        elif bw == 3:
            packed_bytes = ((length + 7) // 8) * 3
        elif bw == 4:
            packed_bytes = (length + 1) // 2
        else:  # 5-8: no packing — stored as raw uint8.
            packed_bytes = length

        meta[bw] = {
            'start': start, 'length': length,
            'pack_offset': pack_offset, 'packed_bytes': packed_bytes,
        }
        pack_offset += packed_bytes

    # Merge 5-8 bit into nopack range.
    # Default nopack_start = HD ("uint8 section starts past the end of the
    # vector"), so a low-bit-only allocation (max bit width <= 4) correctly
    # reports the entire vector as nibble-packed.  With the old default of
    # 0, downstream nib_bytes = nopack_start // 2 = 0 -> max_mixed_bytes = 0
    # -> zero-byte K buffer -> fused-attn illegal memory access.
    nopack_start, nopack_len, nopack_off = HD, 0, 0
    for bw in range(5, 9):
        if bw in meta:
            if nopack_len == 0:
                nopack_start = meta[bw]['start']
                nopack_off = meta[bw]['pack_offset']
            nopack_len += meta[bw]['length']

    meta['nopack_start'] = nopack_start
    meta['nopack_len'] = nopack_len
    meta['nopack_off'] = nopack_off
    meta['total_packed_bytes'] = pack_offset
    meta['pack_perm'] = pack_perm
    inv_pack_perm = torch.empty_like(pack_perm)
    inv_pack_perm[pack_perm] = torch.arange(HD)
    meta['inv_pack_perm'] = inv_pack_perm
    return meta


# ===========================================================================
# Bit-packing utilities (CPU/GPU)
# ===========================================================================

from blockgtq.kernels.quant_kernels import pack_codes_mixed_bit, pack_codes_mixed_bit_gpu


# ===========================================================================
# Triton Kernel
# ===========================================================================

if HAS_TRITON:

    from blockgtq.kernels.quant_kernels import _compress_only_kernel

    @triton.jit
    def _compress_pack_kernel_v8(
        X_ptr, PACKED_ptr, NORMS_ptr, SCRATCH_ptr,
        PERM_ptr, PACK_PERM_ptr, BLOCK_ROT_T_ptr, CODE_LUT_ptr, POS_TO_CB_ptr,
        LUT_OFFSETS_ptr, LUT_INV_SCALES_ptr, GROUP_OF_ptr,
        stride_xn, stride_packed_n, stride_norms_n, stride_clut,
        N,
        HD: tl.constexpr,
        N_GROUPS: tl.constexpr,
        N_BINS: tl.constexpr,
        N_CODEBOOKS: tl.constexpr,
        BLOCK_N: tl.constexpr,
        SINGLE_GROUP: tl.constexpr,
        BW1_START: tl.constexpr, BW1_LEN: tl.constexpr, BW1_OFF: tl.constexpr,
        BW2_START: tl.constexpr, BW2_LEN: tl.constexpr, BW2_OFF: tl.constexpr,
        BW3_START: tl.constexpr, BW3_LEN: tl.constexpr, BW3_OFF: tl.constexpr,
        BW4_START: tl.constexpr, BW4_LEN: tl.constexpr, BW4_OFF: tl.constexpr,
        NP_START: tl.constexpr, NP_LEN: tl.constexpr, NP_OFF: tl.constexpr,
    ):
        """Compress + in-kernel bit-pack: bit widths 1-4 are packed, 5-8 raw.
        Step 7 packs 1-4 bit only; 5-8 stored as raw uint8.
        """
        pid = tl.program_id(0)
        row_start = pid * BLOCK_N
        rows = row_start + tl.arange(0, BLOCK_N)
        cols = tl.arange(0, HD)
        mask_n = rows < N
        n_local = tl.arange(0, BLOCK_N)

        # ---- Step 1: Load + permute ----
        perm = tl.load(PERM_ptr + cols)
        x = tl.load(X_ptr + rows[:, None] * stride_xn + perm[None, :],
                     mask=mask_n[:, None], other=0.0).to(tl.float32)

        # ---- Step 2: Per-group norms ----
        if SINGLE_GROUP == 1:
            norm_sq = tl.sum(x * x, axis=1)
            norms_val = tl.sqrt(norm_sq + 1e-30)
            safe_norms = tl.where(norms_val > 1e-10, norms_val, 1.0)
            x_normed = x / safe_norms[:, None]
            tl.store(NORMS_ptr + rows * stride_norms_n,
                     norms_val, mask=mask_n)
        else:
            x_sq = x * x
            group_of = tl.load(GROUP_OF_ptr + cols)
            per_pos_norm_sq = tl.zeros([BLOCK_N, HD], dtype=tl.float32)
            for g in tl.static_range(0, N_GROUPS):
                g_mask = tl.where(group_of == g, 1.0, 0.0)
                g_sq = x_sq * g_mask[None, :]
                g_norm_sq = tl.sum(g_sq, axis=1)
                per_pos_norm_sq += tl.where(group_of[None, :] == g,
                                            g_norm_sq[:, None], 0.0)
            per_pos_norms = tl.sqrt(per_pos_norm_sq + 1e-30)
            per_pos_safe = tl.where(per_pos_norms > 1e-10, per_pos_norms, 1.0)
            x_normed = x / per_pos_safe
            for g in tl.static_range(0, N_GROUPS):
                g_mask = tl.where(group_of == g, 1.0, 0.0)
                g_norms = tl.sum(per_pos_norms * g_mask[None, :], axis=1)
                g_size = tl.sum(g_mask)
                g_norm_val = tl.where(g_size > 0, g_norms / g_size, 0.0)
                tl.store(NORMS_ptr + rows * stride_norms_n + g,
                         g_norm_val, mask=mask_n)

        # ---- Step 3: Forward rotation (tl.dot) ----
        ROT_T = tl.load(
            BLOCK_ROT_T_ptr + cols[:, None] * HD + tl.arange(0, HD)[None, :]
        ).to(tl.float32)
        y = tl.dot(x_normed, ROT_T, allow_tf32=True)

        # ---- Step 4: Discretize to bins ----
        offsets_lut = tl.load(LUT_OFFSETS_ptr + cols).to(tl.float32)
        inv_scales_lut = tl.load(LUT_INV_SCALES_ptr + cols).to(tl.float32)
        bin_idx = (y - offsets_lut[None, :]) * inv_scales_lut[None, :]
        bin_idx = tl.minimum(tl.maximum(bin_idx, 0.0), (N_BINS - 1) + 0.0)
        bin_idx_int = bin_idx.to(tl.int32)

        # ---- Step 5: Code LUT → uint8 codes ----
        pos_cb = tl.load(POS_TO_CB_ptr + cols).to(tl.int32)
        clut_addr = pos_cb[None, :] * stride_clut + bin_idx_int
        codes = tl.load(CODE_LUT_ptr + clut_addr).to(tl.uint8)

        # ---- Step 6: Write to column-major scratch (reordered by bit-width) ----
        sb = pid * HD * BLOCK_N
        inv_pp = tl.load(PACK_PERM_ptr + cols)
        tl.store(SCRATCH_ptr + sb + inv_pp[None, :] * BLOCK_N + n_local[:, None],
                 codes, mask=mask_n[:, None])

        tl.debug_barrier()

        # ---- Step 7: Pack from scratch (1-4 bit only; 5-8 raw) ----
        _sc = SCRATCH_ptr + sb

        # -- 1-bit: 8 codes → 1 byte --
        if BW1_LEN > 0:
            _e1 = BW1_START + BW1_LEN
            for _g in tl.static_range(0, (BW1_LEN + 7) // 8):
                _d = BW1_START + _g * 8
                bv = tl.load(_sc + _d * BLOCK_N + n_local,
                             mask=mask_n, other=0).to(tl.int32) & 1
                if _d + 1 < _e1:
                    bv |= (tl.load(_sc + (_d + 1) * BLOCK_N + n_local,
                                   mask=mask_n, other=0).to(tl.int32) & 1) << 1
                if _d + 2 < _e1:
                    bv |= (tl.load(_sc + (_d + 2) * BLOCK_N + n_local,
                                   mask=mask_n, other=0).to(tl.int32) & 1) << 2
                if _d + 3 < _e1:
                    bv |= (tl.load(_sc + (_d + 3) * BLOCK_N + n_local,
                                   mask=mask_n, other=0).to(tl.int32) & 1) << 3
                if _d + 4 < _e1:
                    bv |= (tl.load(_sc + (_d + 4) * BLOCK_N + n_local,
                                   mask=mask_n, other=0).to(tl.int32) & 1) << 4
                if _d + 5 < _e1:
                    bv |= (tl.load(_sc + (_d + 5) * BLOCK_N + n_local,
                                   mask=mask_n, other=0).to(tl.int32) & 1) << 5
                if _d + 6 < _e1:
                    bv |= (tl.load(_sc + (_d + 6) * BLOCK_N + n_local,
                                   mask=mask_n, other=0).to(tl.int32) & 1) << 6
                if _d + 7 < _e1:
                    bv |= (tl.load(_sc + (_d + 7) * BLOCK_N + n_local,
                                   mask=mask_n, other=0).to(tl.int32) & 1) << 7
                tl.store(PACKED_ptr + rows * stride_packed_n + BW1_OFF + _g,
                         bv.to(tl.uint8), mask=mask_n)

        # -- 2-bit: 4 codes → 1 byte --
        if BW2_LEN > 0:
            _e2 = BW2_START + BW2_LEN
            for _g in tl.static_range(0, (BW2_LEN + 3) // 4):
                _d = BW2_START + _g * 4
                bv = tl.load(_sc + _d * BLOCK_N + n_local,
                             mask=mask_n, other=0).to(tl.int32) & 0x3
                if _d + 1 < _e2:
                    bv |= (tl.load(_sc + (_d + 1) * BLOCK_N + n_local,
                                   mask=mask_n, other=0).to(tl.int32) & 0x3) << 2
                if _d + 2 < _e2:
                    bv |= (tl.load(_sc + (_d + 2) * BLOCK_N + n_local,
                                   mask=mask_n, other=0).to(tl.int32) & 0x3) << 4
                if _d + 3 < _e2:
                    bv |= (tl.load(_sc + (_d + 3) * BLOCK_N + n_local,
                                   mask=mask_n, other=0).to(tl.int32) & 0x3) << 6
                tl.store(PACKED_ptr + rows * stride_packed_n + BW2_OFF + _g,
                         bv.to(tl.uint8), mask=mask_n)

        # -- 3-bit: 8 codes → 3 bytes --
        if BW3_LEN > 0:
            _e3 = BW3_START + BW3_LEN
            for _g in tl.static_range(0, (BW3_LEN + 7) // 8):
                _d = BW3_START + _g * 8
                c0 = tl.load(_sc + _d * BLOCK_N + n_local,
                             mask=mask_n, other=0).to(tl.int32) & 0x7
                if _d + 1 < _e3:
                    c1 = tl.load(_sc + (_d + 1) * BLOCK_N + n_local,
                                 mask=mask_n, other=0).to(tl.int32) & 0x7
                else:
                    c1 = tl.zeros([BLOCK_N], dtype=tl.int32)
                if _d + 2 < _e3:
                    c2 = tl.load(_sc + (_d + 2) * BLOCK_N + n_local,
                                 mask=mask_n, other=0).to(tl.int32) & 0x7
                else:
                    c2 = tl.zeros([BLOCK_N], dtype=tl.int32)
                if _d + 3 < _e3:
                    c3 = tl.load(_sc + (_d + 3) * BLOCK_N + n_local,
                                 mask=mask_n, other=0).to(tl.int32) & 0x7
                else:
                    c3 = tl.zeros([BLOCK_N], dtype=tl.int32)
                if _d + 4 < _e3:
                    c4 = tl.load(_sc + (_d + 4) * BLOCK_N + n_local,
                                 mask=mask_n, other=0).to(tl.int32) & 0x7
                else:
                    c4 = tl.zeros([BLOCK_N], dtype=tl.int32)
                if _d + 5 < _e3:
                    c5 = tl.load(_sc + (_d + 5) * BLOCK_N + n_local,
                                 mask=mask_n, other=0).to(tl.int32) & 0x7
                else:
                    c5 = tl.zeros([BLOCK_N], dtype=tl.int32)
                if _d + 6 < _e3:
                    c6 = tl.load(_sc + (_d + 6) * BLOCK_N + n_local,
                                 mask=mask_n, other=0).to(tl.int32) & 0x7
                else:
                    c6 = tl.zeros([BLOCK_N], dtype=tl.int32)
                if _d + 7 < _e3:
                    c7 = tl.load(_sc + (_d + 7) * BLOCK_N + n_local,
                                 mask=mask_n, other=0).to(tl.int32) & 0x7
                else:
                    c7 = tl.zeros([BLOCK_N], dtype=tl.int32)
                byte0 = c0 | (c1 << 3) | (c2 << 6)
                byte1 = (c2 >> 2) | (c3 << 1) | (c4 << 4) | (c5 << 7)
                byte2 = (c5 >> 1) | (c6 << 2) | (c7 << 5)
                _off = BW3_OFF + _g * 3
                tl.store(PACKED_ptr + rows * stride_packed_n + _off,
                         (byte0 & 0xFF).to(tl.uint8), mask=mask_n)
                tl.store(PACKED_ptr + rows * stride_packed_n + _off + 1,
                         (byte1 & 0xFF).to(tl.uint8), mask=mask_n)
                tl.store(PACKED_ptr + rows * stride_packed_n + _off + 2,
                         (byte2 & 0xFF).to(tl.uint8), mask=mask_n)

        # -- 4-bit: 2 codes → 1 byte --
        if BW4_LEN > 0:
            _e4 = BW4_START + BW4_LEN
            for _g in tl.static_range(0, (BW4_LEN + 1) // 2):
                _d = BW4_START + _g * 2
                bv = tl.load(_sc + _d * BLOCK_N + n_local,
                             mask=mask_n, other=0).to(tl.int32) & 0xF
                if _d + 1 < _e4:
                    bv |= (tl.load(_sc + (_d + 1) * BLOCK_N + n_local,
                                   mask=mask_n, other=0).to(tl.int32) & 0xF) << 4
                tl.store(PACKED_ptr + rows * stride_packed_n + BW4_OFF + _g,
                         bv.to(tl.uint8), mask=mask_n)

        # -- No-pack (5-8 bit): store as uint8 --
        if NP_LEN > 0:
            for _k in tl.static_range(0, NP_LEN):
                _d = NP_START + _k
                c = tl.load(_sc + _d * BLOCK_N + n_local,
                            mask=mask_n, other=0).to(tl.uint8)
                tl.store(PACKED_ptr + rows * stride_packed_n + NP_OFF + _k,
                         c, mask=mask_n)

    @triton.jit
    def _compress_v_pack_kernel(
        V_ptr, PACKED_ptr, NORMS_ptr, SCRATCH_ptr,
        ROT_T_ptr, BOUNDARIES_ptr, CENTROIDS_ptr,
        stride_vn, stride_packed_n,
        N,
        HD: tl.constexpr,
        N_CENTS: tl.constexpr,
        V_BITS: tl.constexpr,
        PACKED_BYTES: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        """Fused V encode: norm → rotate → quantize → corrected_norm → pack.

        Single kernel replacing ~17 PyTorch launches in PolarQuantGPU pipeline.
        Supports V_BITS 1-4 with uniform bit-width packing.
        """
        pid = tl.program_id(0)
        row_start = pid * BLOCK_N
        rows = row_start + tl.arange(0, BLOCK_N)
        cols = tl.arange(0, HD)
        mask_n = rows < N
        n_local = tl.arange(0, BLOCK_N)

        # ---- Step 1: Load V vectors (fp16 → fp32) ----
        x = tl.load(V_ptr + rows[:, None] * stride_vn + cols[None, :],
                     mask=mask_n[:, None], other=0.0).to(tl.float32)

        # ---- Step 2: L2 norms ----
        norm_sq = tl.sum(x * x, axis=1)
        raw_norms = tl.sqrt(norm_sq + 1e-30)
        safe_norms = tl.where(raw_norms > 1e-8, raw_norms, 1.0)

        # ---- Step 3: Normalize ----
        x_normed = x / safe_norms[:, None]

        # ---- Step 4: Rotate y = x_normed @ R^T ----
        ROT_T = tl.load(
            ROT_T_ptr + cols[:, None] * HD + tl.arange(0, HD)[None, :]
        ).to(tl.float32)
        y = tl.dot(x_normed, ROT_T, allow_tf32=True)

        # ---- Step 5: Quantize (searchsorted via linear scan) ----
        codes = tl.zeros([BLOCK_N, HD], dtype=tl.int32)
        for b in tl.static_range(0, N_CENTS - 1):
            boundary_b = tl.load(BOUNDARIES_ptr + b)
            codes += tl.where(y > boundary_b, 1, 0)

        # ---- Step 6: Corrected norms ----
        cent_vals = tl.load(CENTROIDS_ptr + codes).to(tl.float32)
        cent_sq = cent_vals * cent_vals
        cent_norms = tl.sqrt(tl.sum(cent_sq, axis=1) + 1e-20)
        safe_cent = tl.where(cent_norms > 1e-10, cent_norms, 1.0)
        corrected = raw_norms / safe_cent
        tl.store(NORMS_ptr + rows, corrected.to(tl.float16), mask=mask_n)

        # ---- Step 7: Write codes to scratch (col-major) ----
        sb = pid * HD * BLOCK_N
        _sc = SCRATCH_ptr + sb
        tl.store(_sc + cols[None, :] * BLOCK_N + n_local[:, None],
                 codes.to(tl.uint8), mask=mask_n[:, None])
        tl.debug_barrier()

        # ---- Step 8: Pack from scratch ----
        # 1-bit: 8 codes → 1 byte
        if V_BITS == 1:
            for _g in tl.static_range(0, (HD + 7) // 8):
                _d = _g * 8
                bv = tl.load(_sc + _d * BLOCK_N + n_local,
                             mask=mask_n, other=0).to(tl.int32) & 1
                if _d + 1 < HD:
                    bv |= (tl.load(_sc + (_d + 1) * BLOCK_N + n_local,
                                   mask=mask_n, other=0).to(tl.int32) & 1) << 1
                if _d + 2 < HD:
                    bv |= (tl.load(_sc + (_d + 2) * BLOCK_N + n_local,
                                   mask=mask_n, other=0).to(tl.int32) & 1) << 2
                if _d + 3 < HD:
                    bv |= (tl.load(_sc + (_d + 3) * BLOCK_N + n_local,
                                   mask=mask_n, other=0).to(tl.int32) & 1) << 3
                if _d + 4 < HD:
                    bv |= (tl.load(_sc + (_d + 4) * BLOCK_N + n_local,
                                   mask=mask_n, other=0).to(tl.int32) & 1) << 4
                if _d + 5 < HD:
                    bv |= (tl.load(_sc + (_d + 5) * BLOCK_N + n_local,
                                   mask=mask_n, other=0).to(tl.int32) & 1) << 5
                if _d + 6 < HD:
                    bv |= (tl.load(_sc + (_d + 6) * BLOCK_N + n_local,
                                   mask=mask_n, other=0).to(tl.int32) & 1) << 6
                if _d + 7 < HD:
                    bv |= (tl.load(_sc + (_d + 7) * BLOCK_N + n_local,
                                   mask=mask_n, other=0).to(tl.int32) & 1) << 7
                tl.store(PACKED_ptr + rows * stride_packed_n + _g,
                         bv.to(tl.uint8), mask=mask_n)

        # 2-bit: 4 codes → 1 byte
        if V_BITS == 2:
            for _g in tl.static_range(0, (HD + 3) // 4):
                _d = _g * 4
                bv = tl.load(_sc + _d * BLOCK_N + n_local,
                             mask=mask_n, other=0).to(tl.int32) & 0x3
                if _d + 1 < HD:
                    bv |= (tl.load(_sc + (_d + 1) * BLOCK_N + n_local,
                                   mask=mask_n, other=0).to(tl.int32) & 0x3) << 2
                if _d + 2 < HD:
                    bv |= (tl.load(_sc + (_d + 2) * BLOCK_N + n_local,
                                   mask=mask_n, other=0).to(tl.int32) & 0x3) << 4
                if _d + 3 < HD:
                    bv |= (tl.load(_sc + (_d + 3) * BLOCK_N + n_local,
                                   mask=mask_n, other=0).to(tl.int32) & 0x3) << 6
                tl.store(PACKED_ptr + rows * stride_packed_n + _g,
                         bv.to(tl.uint8), mask=mask_n)

        # 3-bit: 8 codes → 3 bytes
        if V_BITS == 3:
            for _g in tl.static_range(0, (HD + 7) // 8):
                _d = _g * 8
                c0 = tl.load(_sc + _d * BLOCK_N + n_local,
                             mask=mask_n, other=0).to(tl.int32) & 0x7
                if _d + 1 < HD:
                    c1 = tl.load(_sc + (_d + 1) * BLOCK_N + n_local,
                                 mask=mask_n, other=0).to(tl.int32) & 0x7
                else:
                    c1 = tl.zeros([BLOCK_N], dtype=tl.int32)
                if _d + 2 < HD:
                    c2 = tl.load(_sc + (_d + 2) * BLOCK_N + n_local,
                                 mask=mask_n, other=0).to(tl.int32) & 0x7
                else:
                    c2 = tl.zeros([BLOCK_N], dtype=tl.int32)
                if _d + 3 < HD:
                    c3 = tl.load(_sc + (_d + 3) * BLOCK_N + n_local,
                                 mask=mask_n, other=0).to(tl.int32) & 0x7
                else:
                    c3 = tl.zeros([BLOCK_N], dtype=tl.int32)
                if _d + 4 < HD:
                    c4 = tl.load(_sc + (_d + 4) * BLOCK_N + n_local,
                                 mask=mask_n, other=0).to(tl.int32) & 0x7
                else:
                    c4 = tl.zeros([BLOCK_N], dtype=tl.int32)
                if _d + 5 < HD:
                    c5 = tl.load(_sc + (_d + 5) * BLOCK_N + n_local,
                                 mask=mask_n, other=0).to(tl.int32) & 0x7
                else:
                    c5 = tl.zeros([BLOCK_N], dtype=tl.int32)
                if _d + 6 < HD:
                    c6 = tl.load(_sc + (_d + 6) * BLOCK_N + n_local,
                                 mask=mask_n, other=0).to(tl.int32) & 0x7
                else:
                    c6 = tl.zeros([BLOCK_N], dtype=tl.int32)
                if _d + 7 < HD:
                    c7 = tl.load(_sc + (_d + 7) * BLOCK_N + n_local,
                                 mask=mask_n, other=0).to(tl.int32) & 0x7
                else:
                    c7 = tl.zeros([BLOCK_N], dtype=tl.int32)
                byte0 = c0 | (c1 << 3) | (c2 << 6)
                byte1 = (c2 >> 2) | (c3 << 1) | (c4 << 4) | (c5 << 7)
                byte2 = (c5 >> 1) | (c6 << 2) | (c7 << 5)
                _off = _g * 3
                tl.store(PACKED_ptr + rows * stride_packed_n + _off,
                         (byte0 & 0xFF).to(tl.uint8), mask=mask_n)
                tl.store(PACKED_ptr + rows * stride_packed_n + _off + 1,
                         (byte1 & 0xFF).to(tl.uint8), mask=mask_n)
                tl.store(PACKED_ptr + rows * stride_packed_n + _off + 2,
                         (byte2 & 0xFF).to(tl.uint8), mask=mask_n)

        # 4-bit: 2 codes → 1 byte
        if V_BITS == 4:
            for _g in tl.static_range(0, (HD + 1) // 2):
                _d = _g * 2
                bv = tl.load(_sc + _d * BLOCK_N + n_local,
                             mask=mask_n, other=0).to(tl.int32) & 0xF
                if _d + 1 < HD:
                    bv |= (tl.load(_sc + (_d + 1) * BLOCK_N + n_local,
                                   mask=mask_n, other=0).to(tl.int32) & 0xF) << 4
                tl.store(PACKED_ptr + rows * stride_packed_n + _g,
                         bv.to(tl.uint8), mask=mask_n)

    # ===================================================================
    # Batched V encode kernel — all heads in one launch
    # ===================================================================

    @triton.jit
    def _compress_v_pack_kernel_batched(
        V_ptr, PACKED_ptr, NORMS_ptr, SCRATCH_ptr,
        ROT_T_ptr, BOUNDARIES_ptr, CENTROIDS_ptr,
        stride_vn, stride_packed_n,
        stride_v_head,       # stride between heads in V (N * HD)
        stride_packed_head,  # stride between heads in PACKED (N * pb)
        stride_norms_head,   # stride between heads in NORMS (N)
        stride_scratch_head, # stride between heads in SCRATCH (n_blocks*HD*BLOCK_N)
        stride_rot_head,     # stride between heads in ROT_T (HD * HD)
        stride_bound_head,   # stride between heads in BOUNDARIES (N_CENTS - 1)
        stride_cent_head,    # stride between heads in CENTROIDS (N_CENTS)
        N, N_BLOCKS_PER_HEAD,
        HD: tl.constexpr,
        N_CENTS: tl.constexpr,
        V_BITS: tl.constexpr,
        PACKED_BYTES: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        """Batched V encode: all KV heads in one kernel launch.

        Grid = (n_kv * n_blocks_per_head,).  Each block computes its
        head_idx and intra-head block_idx from program_id.
        """
        pid = tl.program_id(0)
        head_idx = pid // N_BLOCKS_PER_HEAD
        block_idx = pid % N_BLOCKS_PER_HEAD

        row_start = block_idx * BLOCK_N
        rows = row_start + tl.arange(0, BLOCK_N)
        cols = tl.arange(0, HD)
        mask_n = rows < N
        n_local = tl.arange(0, BLOCK_N)

        # Per-head base pointers
        v_base = V_ptr + head_idx * stride_v_head
        packed_base = PACKED_ptr + head_idx * stride_packed_head
        norms_base = NORMS_ptr + head_idx * stride_norms_head
        scratch_base = SCRATCH_ptr + head_idx * stride_scratch_head
        rot_base = ROT_T_ptr + head_idx * stride_rot_head
        bound_base = BOUNDARIES_ptr + head_idx * stride_bound_head
        cent_base = CENTROIDS_ptr + head_idx * stride_cent_head

        # ---- Step 1: Load V vectors (fp16 → fp32) ----
        x = tl.load(v_base + rows[:, None] * stride_vn + cols[None, :],
                     mask=mask_n[:, None], other=0.0).to(tl.float32)

        # ---- Step 2: L2 norms ----
        norm_sq = tl.sum(x * x, axis=1)
        raw_norms = tl.sqrt(norm_sq + 1e-30)
        safe_norms = tl.where(raw_norms > 1e-8, raw_norms, 1.0)

        # ---- Step 3: Normalize ----
        x_normed = x / safe_norms[:, None]

        # ---- Step 4: Rotate y = x_normed @ R^T ----
        ROT_T = tl.load(
            rot_base + cols[:, None] * HD + tl.arange(0, HD)[None, :]
        ).to(tl.float32)
        y = tl.dot(x_normed, ROT_T, allow_tf32=True)

        # ---- Step 5: Quantize (searchsorted via linear scan) ----
        codes = tl.zeros([BLOCK_N, HD], dtype=tl.int32)
        for b in tl.static_range(0, N_CENTS - 1):
            boundary_b = tl.load(bound_base + b)
            codes += tl.where(y > boundary_b, 1, 0)

        # ---- Step 6: Corrected norms ----
        cent_vals = tl.load(cent_base + codes).to(tl.float32)
        cent_sq = cent_vals * cent_vals
        cent_norms = tl.sqrt(tl.sum(cent_sq, axis=1) + 1e-20)
        safe_cent = tl.where(cent_norms > 1e-10, cent_norms, 1.0)
        corrected = raw_norms / safe_cent
        tl.store(norms_base + rows, corrected.to(tl.float16), mask=mask_n)

        # ---- Step 7: Write codes to scratch (col-major) ----
        sb = block_idx * HD * BLOCK_N
        _sc = scratch_base + sb
        tl.store(_sc + cols[None, :] * BLOCK_N + n_local[:, None],
                 codes.to(tl.uint8), mask=mask_n[:, None])
        tl.debug_barrier()

        # ---- Step 8: Pack from scratch ----
        if V_BITS == 1:
            for _g in tl.static_range(0, (HD + 7) // 8):
                _d = _g * 8
                bv = tl.load(_sc + _d * BLOCK_N + n_local,
                             mask=mask_n, other=0).to(tl.int32) & 1
                if _d + 1 < HD:
                    bv |= (tl.load(_sc + (_d + 1) * BLOCK_N + n_local,
                                   mask=mask_n, other=0).to(tl.int32) & 1) << 1
                if _d + 2 < HD:
                    bv |= (tl.load(_sc + (_d + 2) * BLOCK_N + n_local,
                                   mask=mask_n, other=0).to(tl.int32) & 1) << 2
                if _d + 3 < HD:
                    bv |= (tl.load(_sc + (_d + 3) * BLOCK_N + n_local,
                                   mask=mask_n, other=0).to(tl.int32) & 1) << 3
                if _d + 4 < HD:
                    bv |= (tl.load(_sc + (_d + 4) * BLOCK_N + n_local,
                                   mask=mask_n, other=0).to(tl.int32) & 1) << 4
                if _d + 5 < HD:
                    bv |= (tl.load(_sc + (_d + 5) * BLOCK_N + n_local,
                                   mask=mask_n, other=0).to(tl.int32) & 1) << 5
                if _d + 6 < HD:
                    bv |= (tl.load(_sc + (_d + 6) * BLOCK_N + n_local,
                                   mask=mask_n, other=0).to(tl.int32) & 1) << 6
                if _d + 7 < HD:
                    bv |= (tl.load(_sc + (_d + 7) * BLOCK_N + n_local,
                                   mask=mask_n, other=0).to(tl.int32) & 1) << 7
                tl.store(packed_base + rows * stride_packed_n + _g,
                         bv.to(tl.uint8), mask=mask_n)

        if V_BITS == 2:
            for _g in tl.static_range(0, (HD + 3) // 4):
                _d = _g * 4
                bv = tl.load(_sc + _d * BLOCK_N + n_local,
                             mask=mask_n, other=0).to(tl.int32) & 0x3
                if _d + 1 < HD:
                    bv |= (tl.load(_sc + (_d + 1) * BLOCK_N + n_local,
                                   mask=mask_n, other=0).to(tl.int32) & 0x3) << 2
                if _d + 2 < HD:
                    bv |= (tl.load(_sc + (_d + 2) * BLOCK_N + n_local,
                                   mask=mask_n, other=0).to(tl.int32) & 0x3) << 4
                if _d + 3 < HD:
                    bv |= (tl.load(_sc + (_d + 3) * BLOCK_N + n_local,
                                   mask=mask_n, other=0).to(tl.int32) & 0x3) << 6
                tl.store(packed_base + rows * stride_packed_n + _g,
                         bv.to(tl.uint8), mask=mask_n)

        if V_BITS == 3:
            for _g in tl.static_range(0, (HD + 7) // 8):
                _d = _g * 8
                c0 = tl.load(_sc + _d * BLOCK_N + n_local,
                             mask=mask_n, other=0).to(tl.int32) & 0x7
                if _d + 1 < HD:
                    c1 = tl.load(_sc + (_d + 1) * BLOCK_N + n_local,
                                 mask=mask_n, other=0).to(tl.int32) & 0x7
                else:
                    c1 = tl.zeros([BLOCK_N], dtype=tl.int32)
                if _d + 2 < HD:
                    c2 = tl.load(_sc + (_d + 2) * BLOCK_N + n_local,
                                 mask=mask_n, other=0).to(tl.int32) & 0x7
                else:
                    c2 = tl.zeros([BLOCK_N], dtype=tl.int32)
                if _d + 3 < HD:
                    c3 = tl.load(_sc + (_d + 3) * BLOCK_N + n_local,
                                 mask=mask_n, other=0).to(tl.int32) & 0x7
                else:
                    c3 = tl.zeros([BLOCK_N], dtype=tl.int32)
                if _d + 4 < HD:
                    c4 = tl.load(_sc + (_d + 4) * BLOCK_N + n_local,
                                 mask=mask_n, other=0).to(tl.int32) & 0x7
                else:
                    c4 = tl.zeros([BLOCK_N], dtype=tl.int32)
                if _d + 5 < HD:
                    c5 = tl.load(_sc + (_d + 5) * BLOCK_N + n_local,
                                 mask=mask_n, other=0).to(tl.int32) & 0x7
                else:
                    c5 = tl.zeros([BLOCK_N], dtype=tl.int32)
                if _d + 6 < HD:
                    c6 = tl.load(_sc + (_d + 6) * BLOCK_N + n_local,
                                 mask=mask_n, other=0).to(tl.int32) & 0x7
                else:
                    c6 = tl.zeros([BLOCK_N], dtype=tl.int32)
                if _d + 7 < HD:
                    c7 = tl.load(_sc + (_d + 7) * BLOCK_N + n_local,
                                 mask=mask_n, other=0).to(tl.int32) & 0x7
                else:
                    c7 = tl.zeros([BLOCK_N], dtype=tl.int32)
                byte0 = c0 | (c1 << 3) | (c2 << 6)
                byte1 = (c2 >> 2) | (c3 << 1) | (c4 << 4) | (c5 << 7)
                byte2 = (c5 >> 1) | (c6 << 2) | (c7 << 5)
                _off = _g * 3
                tl.store(packed_base + rows * stride_packed_n + _off,
                         (byte0 & 0xFF).to(tl.uint8), mask=mask_n)
                tl.store(packed_base + rows * stride_packed_n + _off + 1,
                         (byte1 & 0xFF).to(tl.uint8), mask=mask_n)
                tl.store(packed_base + rows * stride_packed_n + _off + 2,
                         (byte2 & 0xFF).to(tl.uint8), mask=mask_n)

        if V_BITS == 4:
            for _g in tl.static_range(0, (HD + 1) // 2):
                _d = _g * 2
                bv = tl.load(_sc + _d * BLOCK_N + n_local,
                             mask=mask_n, other=0).to(tl.int32) & 0xF
                if _d + 1 < HD:
                    bv |= (tl.load(_sc + (_d + 1) * BLOCK_N + n_local,
                                   mask=mask_n, other=0).to(tl.int32) & 0xF) << 4
                tl.store(packed_base + rows * stride_packed_n + _g,
                         bv.to(tl.uint8), mask=mask_n)

    # ===================================================================
    # Batched K encode kernel — all heads in one launch
    # ===================================================================

    @triton.jit
    def _compress_pack_kernel_v8_batched(
        X_ptr, PACKED_ptr, NORMS_ptr, SCRATCH_ptr,
        PERM_ptr, PACK_PERM_ptr, BLOCK_ROT_T_ptr, CODE_LUT_ptr,
        POS_TO_CB_ptr, LUT_OFFSETS_ptr, LUT_INV_SCALES_ptr, GROUP_OF_ptr,
        PACK_PARAMS_ptr,  # (n_kv, 16) int32: per-head segment params
        stride_xn, stride_packed_n, stride_norms_n, stride_clut,
        stride_x_head,       # N * HD
        stride_packed_head,  # N * total_packed_bytes
        stride_norms_head,   # N * max_n_groups
        stride_scratch_head, # n_blocks * HD * BLOCK_N
        stride_perm_head,    # HD
        stride_rot_head,     # HD * HD
        stride_clut_head,    # n_codebooks * clut_stride
        stride_ptcb_head,    # HD
        stride_loff_head,    # HD
        stride_linv_head,    # HD
        stride_grp_head,     # HD
        N, N_BLOCKS_PER_HEAD,
        HD: tl.constexpr,
        MAX_N_GROUPS: tl.constexpr,
        N_BINS: tl.constexpr,
        MAX_N_CODEBOOKS: tl.constexpr,
        BLOCK_N: tl.constexpr,
        MAX_BW1_LEN: tl.constexpr,
        MAX_BW2_LEN: tl.constexpr,
        MAX_BW3_LEN: tl.constexpr,
        MAX_BW4_LEN: tl.constexpr,
        MAX_NP_LEN: tl.constexpr,
    ):
        """Batched K encode: all KV heads in one kernel launch.

        Grid = (n_kv * n_blocks_per_head,).
        Per-head segment params loaded from PACK_PARAMS table at runtime.
        MAX_BW*_LEN constexpr bound the static_range loops; runtime
        masking skips unused iterations for heads with smaller segments.

        PACK_PARAMS layout per head (16 int32):
          [0]=BW1_START, [1]=BW1_LEN, [2]=BW1_OFF,
          [3]=BW2_START, [4]=BW2_LEN, [5]=BW2_OFF,
          [6]=BW3_START, [7]=BW3_LEN, [8]=BW3_OFF,
          [9]=BW4_START, [10]=BW4_LEN, [11]=BW4_OFF,
          [12]=NP_START, [13]=NP_LEN, [14]=NP_OFF,
          [15]=N_GROUPS (1 = single group)
        """
        pid = tl.program_id(0)
        head_idx = pid // N_BLOCKS_PER_HEAD
        block_idx = pid % N_BLOCKS_PER_HEAD

        row_start = block_idx * BLOCK_N
        rows = row_start + tl.arange(0, BLOCK_N)
        cols = tl.arange(0, HD)
        mask_n = rows < N
        n_local = tl.arange(0, BLOCK_N)

        # Load per-head segment params
        pp_base = PACK_PARAMS_ptr + head_idx * 16
        BW1_START = tl.load(pp_base + 0)
        BW1_LEN = tl.load(pp_base + 1)
        BW1_OFF = tl.load(pp_base + 2)
        BW2_START = tl.load(pp_base + 3)
        BW2_LEN = tl.load(pp_base + 4)
        BW2_OFF = tl.load(pp_base + 5)
        BW3_START = tl.load(pp_base + 6)
        BW3_LEN = tl.load(pp_base + 7)
        BW3_OFF = tl.load(pp_base + 8)
        BW4_START = tl.load(pp_base + 9)
        BW4_LEN = tl.load(pp_base + 10)
        BW4_OFF = tl.load(pp_base + 11)
        NP_START = tl.load(pp_base + 12)
        NP_LEN = tl.load(pp_base + 13)
        NP_OFF = tl.load(pp_base + 14)
        N_GROUPS_h = tl.load(pp_base + 15)

        # Per-head base pointers
        x_base = X_ptr + head_idx * stride_x_head
        packed_base = PACKED_ptr + head_idx * stride_packed_head
        norms_base = NORMS_ptr + head_idx * stride_norms_head
        scratch_base = SCRATCH_ptr + head_idx * stride_scratch_head
        perm_base = PERM_ptr + head_idx * stride_perm_head
        pack_perm_base = PACK_PERM_ptr + head_idx * stride_perm_head
        rot_base = BLOCK_ROT_T_ptr + head_idx * stride_rot_head
        clut_base = CODE_LUT_ptr + head_idx * stride_clut_head
        ptcb_base = POS_TO_CB_ptr + head_idx * stride_ptcb_head
        loff_base = LUT_OFFSETS_ptr + head_idx * stride_loff_head
        linv_base = LUT_INV_SCALES_ptr + head_idx * stride_linv_head
        grp_base = GROUP_OF_ptr + head_idx * stride_grp_head

        # ---- Step 1: Load + permute ----
        perm = tl.load(perm_base + cols)
        x = tl.load(x_base + rows[:, None] * stride_xn + perm[None, :],
                     mask=mask_n[:, None], other=0.0).to(tl.float32)

        # ---- Step 2: Per-group norms ----
        if MAX_N_GROUPS == 1:
            norm_sq = tl.sum(x * x, axis=1)
            norms_val = tl.sqrt(norm_sq + 1e-30)
            safe_norms = tl.where(norms_val > 1e-10, norms_val, 1.0)
            x_normed = x / safe_norms[:, None]
            tl.store(norms_base + rows * stride_norms_n,
                     norms_val, mask=mask_n)
        else:
            x_sq = x * x
            group_of = tl.load(grp_base + cols)
            per_pos_norm_sq = tl.zeros([BLOCK_N, HD], dtype=tl.float32)
            for g in tl.static_range(0, MAX_N_GROUPS):
                g_mask = tl.where(group_of == g, 1.0, 0.0)
                g_sq = x_sq * g_mask[None, :]
                g_norm_sq = tl.sum(g_sq, axis=1)
                per_pos_norm_sq += tl.where(group_of[None, :] == g,
                                            g_norm_sq[:, None], 0.0)
            per_pos_norms = tl.sqrt(per_pos_norm_sq + 1e-30)
            per_pos_safe = tl.where(per_pos_norms > 1e-10, per_pos_norms, 1.0)
            x_normed = x / per_pos_safe
            for g in tl.static_range(0, MAX_N_GROUPS):
                g_mask = tl.where(group_of == g, 1.0, 0.0)
                g_norms = tl.sum(per_pos_norms * g_mask[None, :], axis=1)
                g_size = tl.sum(g_mask)
                g_norm_val = tl.where(g_size > 0, g_norms / g_size, 0.0)
                tl.store(norms_base + rows * stride_norms_n + g,
                         g_norm_val, mask=mask_n)

        # ---- Step 3: Forward rotation (tl.dot) ----
        ROT_T = tl.load(
            rot_base + cols[:, None] * HD + tl.arange(0, HD)[None, :]
        ).to(tl.float32)
        y = tl.dot(x_normed, ROT_T, allow_tf32=True)

        # ---- Step 4: Discretize to bins ----
        offsets_lut = tl.load(loff_base + cols).to(tl.float32)
        inv_scales_lut = tl.load(linv_base + cols).to(tl.float32)
        bin_idx = (y - offsets_lut[None, :]) * inv_scales_lut[None, :]
        bin_idx = tl.minimum(tl.maximum(bin_idx, 0.0), (N_BINS - 1) + 0.0)
        bin_idx_int = bin_idx.to(tl.int32)

        # ---- Step 5: Code LUT → uint8 codes ----
        pos_cb = tl.load(ptcb_base + cols).to(tl.int32)
        clut_addr = pos_cb[None, :] * stride_clut + bin_idx_int
        codes = tl.load(clut_base + clut_addr).to(tl.uint8)

        # ---- Step 6: Write to column-major scratch (reordered by bit-width) ----
        sb = block_idx * HD * BLOCK_N
        inv_pp = tl.load(pack_perm_base + cols)
        _sc = scratch_base + sb
        tl.store(_sc + inv_pp[None, :] * BLOCK_N + n_local[:, None],
                 codes, mask=mask_n[:, None])

        tl.debug_barrier()

        # ---- Step 7: Pack from scratch with runtime params ----
        # 1-bit: 8 codes → 1 byte (iterate up to MAX, mask by actual LEN)
        if MAX_BW1_LEN > 0:
            for _g in tl.static_range(0, (MAX_BW1_LEN + 7) // 8):
                if _g * 8 < BW1_LEN:
                    _d = BW1_START + _g * 8
                    _e1 = BW1_START + BW1_LEN
                    bv = tl.load(_sc + _d * BLOCK_N + n_local,
                                 mask=mask_n, other=0).to(tl.int32) & 1
                    if _d + 1 < _e1:
                        bv |= (tl.load(_sc + (_d + 1) * BLOCK_N + n_local,
                                       mask=mask_n, other=0).to(tl.int32) & 1) << 1
                    if _d + 2 < _e1:
                        bv |= (tl.load(_sc + (_d + 2) * BLOCK_N + n_local,
                                       mask=mask_n, other=0).to(tl.int32) & 1) << 2
                    if _d + 3 < _e1:
                        bv |= (tl.load(_sc + (_d + 3) * BLOCK_N + n_local,
                                       mask=mask_n, other=0).to(tl.int32) & 1) << 3
                    if _d + 4 < _e1:
                        bv |= (tl.load(_sc + (_d + 4) * BLOCK_N + n_local,
                                       mask=mask_n, other=0).to(tl.int32) & 1) << 4
                    if _d + 5 < _e1:
                        bv |= (tl.load(_sc + (_d + 5) * BLOCK_N + n_local,
                                       mask=mask_n, other=0).to(tl.int32) & 1) << 5
                    if _d + 6 < _e1:
                        bv |= (tl.load(_sc + (_d + 6) * BLOCK_N + n_local,
                                       mask=mask_n, other=0).to(tl.int32) & 1) << 6
                    if _d + 7 < _e1:
                        bv |= (tl.load(_sc + (_d + 7) * BLOCK_N + n_local,
                                       mask=mask_n, other=0).to(tl.int32) & 1) << 7
                    tl.store(packed_base + rows * stride_packed_n + BW1_OFF + _g,
                             bv.to(tl.uint8), mask=mask_n)

        # 2-bit: 4 codes → 1 byte
        if MAX_BW2_LEN > 0:
            for _g in tl.static_range(0, (MAX_BW2_LEN + 3) // 4):
                if _g * 4 < BW2_LEN:
                    _d = BW2_START + _g * 4
                    _e2 = BW2_START + BW2_LEN
                    bv = tl.load(_sc + _d * BLOCK_N + n_local,
                                 mask=mask_n, other=0).to(tl.int32) & 0x3
                    if _d + 1 < _e2:
                        bv |= (tl.load(_sc + (_d + 1) * BLOCK_N + n_local,
                                       mask=mask_n, other=0).to(tl.int32) & 0x3) << 2
                    if _d + 2 < _e2:
                        bv |= (tl.load(_sc + (_d + 2) * BLOCK_N + n_local,
                                       mask=mask_n, other=0).to(tl.int32) & 0x3) << 4
                    if _d + 3 < _e2:
                        bv |= (tl.load(_sc + (_d + 3) * BLOCK_N + n_local,
                                       mask=mask_n, other=0).to(tl.int32) & 0x3) << 6
                    tl.store(packed_base + rows * stride_packed_n + BW2_OFF + _g,
                             bv.to(tl.uint8), mask=mask_n)

        # 3-bit: 8 codes → 3 bytes
        if MAX_BW3_LEN > 0:
            for _g in tl.static_range(0, (MAX_BW3_LEN + 7) // 8):
                if _g * 8 < BW3_LEN:
                    _d = BW3_START + _g * 8
                    _e3 = BW3_START + BW3_LEN
                    c0 = tl.load(_sc + _d * BLOCK_N + n_local,
                                 mask=mask_n, other=0).to(tl.int32) & 0x7
                    if _d + 1 < _e3:
                        c1 = tl.load(_sc + (_d + 1) * BLOCK_N + n_local,
                                     mask=mask_n, other=0).to(tl.int32) & 0x7
                    else:
                        c1 = tl.zeros([BLOCK_N], dtype=tl.int32)
                    if _d + 2 < _e3:
                        c2 = tl.load(_sc + (_d + 2) * BLOCK_N + n_local,
                                     mask=mask_n, other=0).to(tl.int32) & 0x7
                    else:
                        c2 = tl.zeros([BLOCK_N], dtype=tl.int32)
                    if _d + 3 < _e3:
                        c3 = tl.load(_sc + (_d + 3) * BLOCK_N + n_local,
                                     mask=mask_n, other=0).to(tl.int32) & 0x7
                    else:
                        c3 = tl.zeros([BLOCK_N], dtype=tl.int32)
                    if _d + 4 < _e3:
                        c4 = tl.load(_sc + (_d + 4) * BLOCK_N + n_local,
                                     mask=mask_n, other=0).to(tl.int32) & 0x7
                    else:
                        c4 = tl.zeros([BLOCK_N], dtype=tl.int32)
                    if _d + 5 < _e3:
                        c5 = tl.load(_sc + (_d + 5) * BLOCK_N + n_local,
                                     mask=mask_n, other=0).to(tl.int32) & 0x7
                    else:
                        c5 = tl.zeros([BLOCK_N], dtype=tl.int32)
                    if _d + 6 < _e3:
                        c6 = tl.load(_sc + (_d + 6) * BLOCK_N + n_local,
                                     mask=mask_n, other=0).to(tl.int32) & 0x7
                    else:
                        c6 = tl.zeros([BLOCK_N], dtype=tl.int32)
                    if _d + 7 < _e3:
                        c7 = tl.load(_sc + (_d + 7) * BLOCK_N + n_local,
                                     mask=mask_n, other=0).to(tl.int32) & 0x7
                    else:
                        c7 = tl.zeros([BLOCK_N], dtype=tl.int32)
                    byte0 = c0 | (c1 << 3) | (c2 << 6)
                    byte1 = (c2 >> 2) | (c3 << 1) | (c4 << 4) | (c5 << 7)
                    byte2 = (c5 >> 1) | (c6 << 2) | (c7 << 5)
                    _off = BW3_OFF + _g * 3
                    tl.store(packed_base + rows * stride_packed_n + _off,
                             (byte0 & 0xFF).to(tl.uint8), mask=mask_n)
                    tl.store(packed_base + rows * stride_packed_n + _off + 1,
                             (byte1 & 0xFF).to(tl.uint8), mask=mask_n)
                    tl.store(packed_base + rows * stride_packed_n + _off + 2,
                             (byte2 & 0xFF).to(tl.uint8), mask=mask_n)

        # 4-bit: 2 codes → 1 byte
        if MAX_BW4_LEN > 0:
            for _g in tl.static_range(0, (MAX_BW4_LEN + 1) // 2):
                if _g * 2 < BW4_LEN:
                    _d = BW4_START + _g * 2
                    _e4 = BW4_START + BW4_LEN
                    bv = tl.load(_sc + _d * BLOCK_N + n_local,
                                 mask=mask_n, other=0).to(tl.int32) & 0xF
                    if _d + 1 < _e4:
                        bv |= (tl.load(_sc + (_d + 1) * BLOCK_N + n_local,
                                       mask=mask_n, other=0).to(tl.int32) & 0xF) << 4
                    tl.store(packed_base + rows * stride_packed_n + BW4_OFF + _g,
                             bv.to(tl.uint8), mask=mask_n)

        # No-pack (5-8 bit): store as uint8
        if MAX_NP_LEN > 0:
            for _k in tl.static_range(0, MAX_NP_LEN):
                if _k < NP_LEN:
                    _d = NP_START + _k
                    c = tl.load(_sc + _d * BLOCK_N + n_local,
                                mask=mask_n, other=0).to(tl.uint8)
                    tl.store(packed_base + rows * stride_packed_n + NP_OFF + _k,
                             c, mask=mask_n)


# ===========================================================================
# Python wrapper
# ===========================================================================

from blockgtq.kernels.quant_kernels import build_shared_lut, next_power_of_2


class BlockGTQQuantizer:
    """Compress-only Block-GTQ quantizer with 1-4 bit packing.

    Bit widths 1-4 are packed sub-byte in-kernel; 5-8 bit codes are stored
    as raw uint8. Encoding and round-trip share the same Triton pipeline.
    """

    def __init__(self, head_dim: int, avg_bits: float = 3.0,
                 rotation_threshold: int = 2,
                 min_bits: int = 1, max_bits: int = 8,
                 rope_base: float = 1e6,
                 n_bins: int = 256,
                 device: torch.device = None):
        assert HAS_TRITON, "Triton not available"
        self.head_dim = head_dim
        self.avg_bits = avg_bits
        self.rotation_threshold = rotation_threshold
        self.min_bits = min_bits
        self.max_bits = max_bits
        self.rope_base = rope_base
        self.n_bins = n_bins
        self.device = device or torch.device('cuda:0')
        self._calibrated = False

    def calibrate(self, q_data: torch.Tensor, k_data: torch.Tensor):
        """Calibrate and build code LUT."""
        from blockgtq.block_gtq_pipeline import BlockGTQPipeline

        ref = BlockGTQPipeline(
            head_dim=self.head_dim, avg_bits=self.avg_bits,
            rotation_threshold=self.rotation_threshold,
            min_bits=self.min_bits, max_bits=self.max_bits,
            rope_base=self.rope_base, device=self.device,
        )
        ref.calibrate(q_data, k_data)

        self.bit_allocation = ref.bit_allocation
        self._head_perm = ref._head_perm.contiguous()
        self._n_groups = ref._n_groups
        self._single_group = ref._single_group
        self._group_of = ref._group_of if not ref._single_group else None

        hd = self.head_dim

        self._block_rot_T = ref._block_rot_T.contiguous()
        self._rot_unperm = ref._block_rot_unperm.contiguous()

        self._pos_centroids = ref._pos_centroids.contiguous()
        self._pos_boundaries = ref._pos_boundaries.contiguous()

        self._code_lut, self._pos_to_cb, self._lut_offsets, self._lut_inv_scales = \
            build_code_lut(self._pos_centroids, self._pos_boundaries, n_bins=self.n_bins)
        self._code_lut = self._code_lut.contiguous()
        self._pos_to_cb = self._pos_to_cb.contiguous()
        self._lut_offsets = self._lut_offsets.contiguous()
        self._lut_inv_scales = self._lut_inv_scales.contiguous()
        self._n_codebooks = self._code_lut.shape[0]

        self._shared_lut, _, _, _ = build_shared_lut(
            self._pos_centroids, self._pos_boundaries, n_bins=self.n_bins
        )
        self._shared_lut = self._shared_lut.contiguous()

        # Build packing metadata (1-4 bit packed, 5-8 raw)
        self._pack_meta = build_pack_meta(self.bit_allocation, self._head_perm)
        self._pack_perm = self._pack_meta['pack_perm'].to(self.device).contiguous()
        self._inv_pack_perm = self._pack_meta['inv_pack_perm'].to(self.device).contiguous()
        self._total_packed_bytes = self._pack_meta['total_packed_bytes']
        self._total_bits = int(self.bit_allocation.sum().item()) * 2

        pm = self._pack_meta
        seg_info = ', '.join(
            f"{bw}b:{pm[bw]['length']}d\u2192{pm[bw]['packed_bytes']}B"
            for bw in range(1, 9) if bw in pm
        )
        print(f"  BlockGTQ: HD={hd}, n_groups={self._n_groups}, "
              f"n_codebooks={self._n_codebooks}, "
              f"pack=[{seg_info}], total={self._total_packed_bytes}B")

        self._calibrated = True
        self._q_perm_baked = False

    @property
    def effective_bits_per_dim(self):
        if self.bit_allocation is None:
            return self.avg_bits
        return self.bit_allocation.float().sum().item() / (self.head_dim // 2)

    @property
    def pack_meta(self):
        """Packing metadata for decode-side unpacking."""
        return self._pack_meta

    def build_kernel_args(self) -> dict:
        """Build dispatch arguments for fused_blockgtq_decode_attention.

        Returns dict with:
            codebook_flat: (total_entries,) fp16 — concatenated per-segment centroids
            lut_offset: (HD,) int32 — per-dim offset into codebook_flat
            norm_group: (HD,) int32 — per-dim norm group index
            segments_b: list of (bw, dim_start, dim_len, pack_off, lut_off, norm_idx)
            total_packed_bytes: int
        """
        assert self._calibrated
        hd = self.head_dim
        pm = self._pack_meta
        device = self.device

        codebook_parts = []
        segments_b = []
        lut_offset = torch.zeros(hd, dtype=torch.int32, device=device)
        lut_off = 0
        seg_idx = 0

        for bw in sorted(bw for bw in pm if isinstance(bw, int)):
            info = pm[bw]
            start = info['start']
            length = info['length']
            pack_off = info['pack_offset']
            n_cents = 1 << bw

            # Extract centroids for this segment from _pos_centroids
            centroids = self._pos_centroids[start, :n_cents].to(torch.float16)
            codebook_parts.append(centroids)

            lut_offset[start:start + length] = lut_off
            segments_b.append((bw, start, length, pack_off, lut_off, seg_idx))
            lut_off += n_cents
            seg_idx += 1

        codebook_flat = torch.cat(codebook_parts).to(device)

        # norm_group: map each packed-space dim to its calibration norm group.
        # _group_of is indexed in head_perm space.  pack_perm maps packed dims
        # to head_perm dims.  Since groups are contiguous in head_perm space and
        # segments are contiguous in packed space, and pack_perm preserves this
        # alignment, _group_of[pack_perm] == segment-based mapping.
        if self._group_of is not None:
            norm_group = self._group_of[self._pack_perm].to(torch.int32).to(device)
        else:
            norm_group = torch.zeros(hd, dtype=torch.int32, device=device)

        return {
            'codebook_flat': codebook_flat,
            'lut_offset': lut_offset,
            'norm_group': norm_group,
            'segments_b': segments_b,
            'total_packed_bytes': pm['total_packed_bytes'],
        }

    @property
    def composite_q_perm(self) -> torch.Tensor:
        """Composite permutation for Q: head_perm[pack_perm].

        Maps original Q dimensions to packed-K dimension order.
        ``q_permuted = q[..., composite_q_perm]`` aligns Q with packed K.
        """
        assert self._calibrated
        return self._head_perm[self._pack_perm]

    def rotate_q_for_packed_kernel(self, q: torch.Tensor) -> torch.Tensor:
        """Permute AND rotate Q for fused packed attention kernel.

        The packed K codes live in rotated space: compress does
        ``y = x_normed @ R^T`` (where R^T = block_rot_T).
        Dequant gives back ``centroid ≈ y``, so the kernel computes
        ``Q_input · centroid * norm``.

        For correctness we need ``Q_input = Q_hp @ R^T`` so that
        ``Q_input · centroid ≈ Q_hp @ R^T @ y^T = Q_hp · (y @ R)``,
        which matches the true dot product in head_perm space.

        Args:
            q: (..., HD) in original space
        Returns:
            q_rot: (..., HD) ready for fused kernel (packed order, rotated)
        """
        assert self._calibrated
        leading = q.shape[:-1]
        hd = q.shape[-1]
        q_flat = q.reshape(-1, hd).float()
        # Step 1: permute to head_perm order
        q_hp = q_flat[:, self._head_perm]
        # Step 2: rotate — Q_rot = Q_hp @ R^T (R^T = block_rot_T)
        q_rot = q_hp @ self._block_rot_T
        # Step 3: reorder by pack_perm to packed dim order
        q_packed = q_rot[:, self._pack_perm]
        return q_packed.to(q.dtype).reshape(*leading, hd)

    @torch.no_grad()
    def bake_permutation_into_qproj(
        self,
        q_proj_weight: torch.Tensor,
        kv_head_idx: int,
        n_kv_heads: int,
        n_q_heads: int,
        q_proj_bias: Optional[torch.Tensor] = None,
    ) -> None:
        """Bake permutation + rotation into q_proj weight.

        After this call, Q vectors produced by q_proj are already in
        packed-K dimension order AND rotated to match the K rotation
        applied during compress_packed.  The attention kernel needs
        neither runtime permutation nor rotation.

        Transform: W_baked = (R^T @ W[head_perm, :])[pack_perm, :]
        where R^T = block_rot_T (the K forward rotation matrix).

        Handles GQA: all Q heads mapping to ``kv_head_idx`` are
        transformed using this quantizer's matrices.

        Args:
            q_proj_weight: ``(n_q_heads * head_dim, in_features)`` weight.
                Modified in-place.
            kv_head_idx: which kv head this quantizer belongs to.
            n_kv_heads: total number of kv heads.
            n_q_heads: total number of q heads.
            q_proj_bias: optional bias ``(n_q_heads * head_dim,)``.
        """
        assert self._calibrated, "calibrate() must be called before baking"
        assert kv_head_idx < n_kv_heads
        hd = self.head_dim
        gqa_ratio = n_q_heads // n_kv_heads
        assert q_proj_weight.shape[0] == n_q_heads * hd

        hp = self._head_perm.detach().cpu()
        pp = self._pack_perm.detach().cpu()
        rot_T = self._block_rot_T.detach().cpu().float()

        for g in range(gqa_ratio):
            q_h = kv_head_idx * gqa_ratio + g
            start = q_h * hd
            end = start + hd
            # permute → rotate → pack_perm
            w_hp = q_proj_weight[start:end].clone()[hp].float()
            w_rot = (rot_T @ w_hp).to(q_proj_weight.dtype)
            q_proj_weight[start:end] = w_rot[pp].contiguous()
            if q_proj_bias is not None:
                b_hp = q_proj_bias[start:end].clone()[hp].float()
                b_rot = rot_T @ b_hp.unsqueeze(1)
                q_proj_bias[start:end] = b_rot.squeeze(1)[pp].to(
                    q_proj_bias.dtype).contiguous()

        self._q_perm_baked = True

    @torch.no_grad()
    def compress(self, k: torch.Tensor, block_n: int = 32
                 ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compress fp16 K vectors to uint8 codes + group norms.

        Args:
            k: (..., HD) fp16 input

        Returns:
            codes: (..., HD) uint8 centroid indices
            norms: (..., N_GROUPS) float32 per-group norms
        """
        assert self._calibrated
        batch_shape = k.shape[:-1]
        hd = self.head_dim
        x = k.reshape(-1, hd).float().contiguous()
        N = x.shape[0]

        codes = torch.empty(N, hd, dtype=torch.uint8, device=self.device)
        norms = torch.empty(N, self._n_groups, dtype=torch.float32, device=self.device)

        group_of_ptr = (self._group_of if self._group_of is not None
                       else torch.zeros(1, dtype=torch.long, device=self.device))

        BLOCK_N = block_n
        grid = ((N + BLOCK_N - 1) // BLOCK_N,)

        _compress_only_kernel[grid](
            x, codes, norms,
            self._head_perm,
            self._block_rot_T,
            self._code_lut,
            self._pos_to_cb,
            self._lut_offsets,
            self._lut_inv_scales,
            group_of_ptr,
            x.stride(0),
            codes.stride(0),
            norms.stride(0),
            self._code_lut.stride(0),
            N,
            HD=hd,
            N_GROUPS=self._n_groups,
            N_BINS=self.n_bins,
            N_CODEBOOKS=self._n_codebooks,
            BLOCK_N=BLOCK_N,
            SINGLE_GROUP=1 if self._single_group else 0,
        )

        codes = codes.reshape(*batch_shape, hd)
        norms = norms.reshape(*batch_shape, self._n_groups)
        return codes, norms

    @torch.no_grad()
    def correct_norms(self, k: torch.Tensor, raw_norms: torch.Tensor,
                      block_n: int = 32) -> torch.Tensor:
        """Compute norm-corrected norms matching compress_decompress quality.

        The fused kernel multiplies centroids by raw_norms (= ||x_group||).
        compress_decompress additionally divides by ||y_hat_group|| (the L2
        norm of quantized centroids within each group), which corrects for
        quantization-induced norm distortion.

        This method computes that correction factor from the same quantization
        codes, returning: corrected = raw_norms / ||y_hat_group||.

        Args:
            k: (..., HD) original fp16 K vectors (same input as compress_packed)
            raw_norms: (..., N_GROUPS) raw norms from compress/compress_packed
            block_n: block size for Triton kernel
        Returns:
            corrected_norms: (..., N_GROUPS) float32
        """
        assert self._calibrated
        batch_shape = k.shape[:-1]
        hd = self.head_dim

        # Get per-dim codes via compress (non-packed, same quantization)
        codes, _ = self.compress(k, block_n=block_n)
        codes_flat = codes.reshape(-1, hd)  # (N, HD) uint8 in head_perm order
        norms_flat = raw_norms.reshape(-1, self._n_groups).float()
        N = codes_flat.shape[0]

        # Look up centroid values: pos_centroids[d, code[t,d]]
        dim_idx = torch.arange(hd, device=self.device)
        cent_vals = self._pos_centroids[dim_idx[None, :],
                                        codes_flat.long()]  # (N, HD) float32

        # Compute per-group ||y_hat||
        corrected = norms_flat.clone()
        if self._single_group:
            yhat_norm = cent_vals.norm(dim=-1)  # (N,)
            corrected[:, 0] /= yhat_norm.clamp(min=1e-10)
        else:
            for g in range(self._n_groups):
                mask = self._group_of == g  # (HD,)
                g_cent = cent_vals[:, mask]  # (N, d_g)
                g_norm = g_cent.norm(dim=-1)  # (N,)
                corrected[:, g] /= g_norm.clamp(min=1e-10)

        return corrected.reshape(*batch_shape, self._n_groups)

    @torch.no_grad()
    def compress_packed(self, k: torch.Tensor, block_n: int = 32,
                        norm_correct: bool = False
                        ) -> tuple[torch.Tensor, torch.Tensor, int]:
        """Compress + in-kernel bit-pack (fused single kernel, packs 1-4 bit).

        Args:
            k: (..., HD) fp16 input
            block_n: Triton block size
            norm_correct: if True, apply norm correction to match
                compress_decompress quality (divides by ||y_hat_group||)

        Returns:
            packed: (..., total_packed_bytes) uint8 segment-aligned packed codes
            norms: (..., N_GROUPS) float32 per-group norms
            total_bits: information bits per vector
        """
        assert self._calibrated
        batch_shape = k.shape[:-1]
        hd = self.head_dim
        x = k.reshape(-1, hd).float().contiguous()
        N = x.shape[0]

        tpb = self._total_packed_bytes
        packed = torch.empty(N, tpb, dtype=torch.uint8, device=self.device)
        norms = torch.empty(N, self._n_groups, dtype=torch.float32,
                            device=self.device)

        BLOCK_N = block_n
        n_blocks = (N + BLOCK_N - 1) // BLOCK_N
        scratch = torch.empty(n_blocks * hd * BLOCK_N, dtype=torch.uint8,
                              device=self.device)

        group_of_ptr = (self._group_of if self._group_of is not None
                       else torch.zeros(1, dtype=torch.long, device=self.device))

        m = self._pack_meta
        def _gp(bw):
            if bw in m:
                return m[bw]['start'], m[bw]['length'], m[bw]['pack_offset']
            return 0, 0, 0

        bw1s, bw1l, bw1o = _gp(1)
        bw2s, bw2l, bw2o = _gp(2)
        bw3s, bw3l, bw3o = _gp(3)
        bw4s, bw4l, bw4o = _gp(4)

        grid = (n_blocks,)
        _compress_pack_kernel_v8[grid](
            x, packed, norms, scratch,
            self._head_perm,
            self._inv_pack_perm,
            self._block_rot_T,
            self._code_lut,
            self._pos_to_cb,
            self._lut_offsets,
            self._lut_inv_scales,
            group_of_ptr,
            x.stride(0),
            packed.stride(0),
            norms.stride(0),
            self._code_lut.stride(0),
            N,
            HD=hd,
            N_GROUPS=self._n_groups,
            N_BINS=self.n_bins,
            N_CODEBOOKS=self._n_codebooks,
            BLOCK_N=BLOCK_N,
            SINGLE_GROUP=1 if self._single_group else 0,
            BW1_START=bw1s, BW1_LEN=bw1l, BW1_OFF=bw1o,
            BW2_START=bw2s, BW2_LEN=bw2l, BW2_OFF=bw2o,
            BW3_START=bw3s, BW3_LEN=bw3l, BW3_OFF=bw3o,
            BW4_START=bw4s, BW4_LEN=bw4l, BW4_OFF=bw4o,
            NP_START=m['nopack_start'], NP_LEN=m['nopack_len'],
            NP_OFF=m['nopack_off'],
        )

        packed = packed.reshape(*batch_shape, tpb)
        norms = norms.reshape(*batch_shape, self._n_groups)

        if norm_correct:
            norms = self.correct_norms(k, norms, block_n=block_n)

        return packed, norms, self._total_bits

    @torch.no_grad()
    def compress_decompress(self, k: torch.Tensor, block_n: int = 32) -> torch.Tensor:
        """Reference round-trip for correctness comparison."""
        from blockgtq.kernels.quant_kernels import _roundtrip_block_gtq_v4_kernel

        assert self._calibrated
        batch_shape = k.shape[:-1]
        hd = self.head_dim
        x = k.reshape(-1, hd).float().contiguous()
        N = x.shape[0]
        out = torch.empty_like(x)

        group_of_ptr = (self._group_of if self._group_of is not None
                       else torch.zeros(1, dtype=torch.long, device=self.device))

        BLOCK_N = block_n
        scratch = torch.empty(N, hd, device=self.device, dtype=torch.float32)

        grid = ((N + BLOCK_N - 1) // BLOCK_N,)
        _roundtrip_block_gtq_v4_kernel[grid](
            x, out, scratch,
            self._head_perm,
            self._block_rot_T,
            self._rot_unperm,
            self._shared_lut,
            self._pos_to_cb,
            self._lut_offsets,
            self._lut_inv_scales,
            group_of_ptr,
            x.stride(0),
            scratch.stride(0),
            self._shared_lut.stride(0),
            N,
            HD=hd,
            N_GROUPS=self._n_groups,
            N_BINS=self.n_bins,
            N_CODEBOOKS=self._n_codebooks,
            BLOCK_N=BLOCK_N,
            SINGLE_GROUP=1 if self._single_group else 0,
        )
        return out.reshape(*batch_shape, hd)

    # ------------------------------------------------------------------
    # V cache compression (PolarQuant on GPU)
    # ------------------------------------------------------------------

    def init_v_quantizer(self, v_bits: int = 3, seed: int = 42):
        """Initialize the V cache quantizer (PolarQuant on GPU).

        Call once at setup time. No calibration data needed — the rotation
        matrix and centroids are seed-deterministic.

        Args:
            v_bits: bit width for V codes (1-4).
            seed: random seed for rotation matrix.
        """
        from blockgtq.v_quantize_gpu import PolarQuantGPU
        self._v_quantizer = PolarQuantGPU(
            d=self.head_dim, bit_width=v_bits, seed=seed, device=self.device)
        self._v_bits = v_bits
        # Cache tensors for Triton V encode kernel
        self._v_rot_T = self._v_quantizer.rotation.T.contiguous()
        # Cache R for in-kernel V un-rotation (output @ R undoes PolarQuant rotation)
        self._v_rot = self._v_quantizer.rotation.contiguous()
        self._v_boundaries = self._v_quantizer.boundaries.contiguous()
        self._v_centroids_f32 = self._v_quantizer.centroids.contiguous()

    @property
    def v_bits(self) -> int:
        """V quantization bit width (0 if V quantizer not initialized)."""
        return getattr(self, '_v_bits', 0)

    @property
    def v_lut(self) -> torch.Tensor:
        """V codebook centroids (fp16) for the attention kernel."""
        assert hasattr(self, '_v_quantizer'), "call init_v_quantizer() first"
        return self._v_quantizer.v_lut

    def _corrected_v_norms(self, codes: torch.Tensor, raw_norms: torch.Tensor
                           ) -> torch.Tensor:
        """Compute kernel-compatible corrected norms for V.

        Kernel convention: v_hat[d] = v_lut[code[d]] * corrected_norm.
        corrected_norm = ||v_original|| / ||centroids[codes]||

        This ensures kernel dequant matches PolarQuant's norm-corrected output.
        """
        centroids = self._v_quantizer.centroids[codes.long()].float()  # (N, D)
        cent_norms = centroids.norm(dim=-1).clamp(min=1e-10)  # (N,)
        return (raw_norms.float() / cent_norms).to(torch.float16)

    @torch.no_grad()
    def compress_v(self, v: torch.Tensor
                   ) -> tuple[torch.Tensor, torch.Tensor]:
        """Quantize V vectors on GPU.

        Args:
            v: (..., HD) fp16 value vectors.

        Returns:
            codes: (..., HD) uint8 centroid indices.
            norms: (...,) fp16 corrected norms (kernel-compatible).
        """
        assert hasattr(self, '_v_quantizer'), "call init_v_quantizer() first"
        batch_shape = v.shape[:-1]
        hd = self.head_dim
        flat = v.reshape(-1, hd)
        codes, raw_norms = self._v_quantizer.quantize(flat)
        norms = self._corrected_v_norms(codes, raw_norms)
        return codes.reshape(*batch_shape, hd), norms.reshape(*batch_shape)

    @torch.no_grad()
    def compress_v_packed(self, v: torch.Tensor, block_n: int = 32
                          ) -> tuple[torch.Tensor, torch.Tensor]:
        """Quantize and bit-pack V vectors via fused Triton kernel.

        Fuses norm → rotate → quantize → corrected_norm → pack into
        a single kernel launch (~17 PyTorch launches → 1).

        Args:
            v: (..., HD) fp16 value vectors.
            block_n: rows per thread block.

        Returns:
            packed: (..., packed_bytes) uint8 bit-packed codes.
            norms: (...,) fp16 corrected norms (kernel-compatible).
        """
        assert hasattr(self, '_v_quantizer'), "call init_v_quantizer() first"
        batch_shape = v.shape[:-1]
        hd = self.head_dim
        flat = v.reshape(-1, hd).contiguous()
        N = flat.shape[0]

        from blockgtq.v_packing import packed_v_bytes
        pb = packed_v_bytes(hd, self._v_bits)
        packed = torch.empty(N, pb, dtype=torch.uint8, device=self.device)
        norms = torch.empty(N, dtype=torch.float16, device=self.device)

        BLOCK_N = block_n
        n_blocks = (N + BLOCK_N - 1) // BLOCK_N
        scratch = torch.empty(n_blocks * hd * BLOCK_N, dtype=torch.uint8,
                              device=self.device)

        n_cents = 1 << self._v_bits
        grid = (n_blocks,)
        _compress_v_pack_kernel[grid](
            flat, packed, norms, scratch,
            self._v_rot_T,
            self._v_boundaries,
            self._v_centroids_f32,
            flat.stride(0), packed.stride(0),
            N,
            HD=hd,
            N_CENTS=n_cents,
            V_BITS=self._v_bits,
            PACKED_BYTES=pb,
            BLOCK_N=BLOCK_N,
        )

        return packed.reshape(*batch_shape, pb), norms.reshape(*batch_shape)


# ===========================================================================
# Batched encode helpers (multi-head, 2 kernel launches total)
# ===========================================================================

def build_batched_encode_args(quantizers: list, device='cuda') -> dict:
    """Pre-stack per-head calibration tensors for batched kernel dispatch.

    Args:
        quantizers: list of n_kv calibrated BlockGTQQuantizer instances.

    Returns dict with stacked tensors + constexpr params for batched kernels.
    """
    n_kv = len(quantizers)
    hd = quantizers[0].head_dim
    v_bits = quantizers[0]._v_bits
    n_bins = quantizers[0].n_bins

    # --- V encode stacked tensors ---
    v_rot_T_stack = torch.stack([q._v_rot_T for q in quantizers])    # (n_kv, HD, HD)
    v_bound_stack = torch.stack([q._v_boundaries for q in quantizers])  # (n_kv, N_CENTS-1)
    v_cent_stack = torch.stack([q._v_centroids_f32 for q in quantizers])  # (n_kv, N_CENTS)

    # --- K encode stacked tensors ---
    k_perm_stack = torch.stack([q._head_perm for q in quantizers])  # (n_kv, HD)
    k_inv_pp_stack = torch.stack([q._inv_pack_perm for q in quantizers])  # (n_kv, HD)
    k_rot_stack = torch.stack([q._block_rot_T for q in quantizers])  # (n_kv, HD, HD)

    # Code LUT: all heads must have same n_bins, may differ in n_codebooks
    max_n_codebooks = max(q._n_codebooks for q in quantizers)
    clut_stride = quantizers[0]._code_lut.stride(0)
    clut_stack = torch.zeros(n_kv, max_n_codebooks, clut_stride,
                             dtype=quantizers[0]._code_lut.dtype, device=device)
    # pos_centroids stacked for norm correction (per-dim centroid values)
    max_n_cents = max(q._pos_centroids.shape[1] for q in quantizers)
    k_pos_cents_stack = torch.zeros(n_kv, hd, max_n_cents,
                                     dtype=torch.float32, device=device)
    for i, q in enumerate(quantizers):
        nc = q._n_codebooks
        clut_stack[i, :nc] = q._code_lut
        nc_pc = q._pos_centroids.shape[1]
        k_pos_cents_stack[i, :, :nc_pc] = q._pos_centroids

    k_ptcb_stack = torch.stack([q._pos_to_cb for q in quantizers])  # (n_kv, HD)
    k_loff_stack = torch.stack([q._lut_offsets for q in quantizers])  # (n_kv, HD)
    k_linv_stack = torch.stack([q._lut_inv_scales for q in quantizers])  # (n_kv, HD)

    # Group-of: may be None for single-group heads
    max_n_groups = max(q._n_groups for q in quantizers)
    k_grp_stack = torch.zeros(n_kv, hd, dtype=torch.long, device=device)
    for i, q in enumerate(quantizers):
        if q._group_of is not None:
            k_grp_stack[i] = q._group_of

    # --- K pack params table (n_kv, 16) int32 ---
    pack_params = torch.zeros(n_kv, 16, dtype=torch.int32, device=device)
    max_bw1_len = max_bw2_len = max_bw3_len = max_bw4_len = max_np_len = 0

    for i, q in enumerate(quantizers):
        m = q._pack_meta
        def _gp(bw):
            if bw in m:
                return m[bw]['start'], m[bw]['length'], m[bw]['pack_offset']
            return 0, 0, 0
        bw1s, bw1l, bw1o = _gp(1)
        bw2s, bw2l, bw2o = _gp(2)
        bw3s, bw3l, bw3o = _gp(3)
        bw4s, bw4l, bw4o = _gp(4)
        pack_params[i] = torch.tensor([
            bw1s, bw1l, bw1o,
            bw2s, bw2l, bw2o,
            bw3s, bw3l, bw3o,
            bw4s, bw4l, bw4o,
            m['nopack_start'], m['nopack_len'], m['nopack_off'],
            q._n_groups,
        ], dtype=torch.int32)
        max_bw1_len = max(max_bw1_len, bw1l)
        max_bw2_len = max(max_bw2_len, bw2l)
        max_bw3_len = max(max_bw3_len, bw3l)
        max_bw4_len = max(max_bw4_len, bw4l)
        max_np_len = max(max_np_len, m['nopack_len'])

    max_tpb = max(q._total_packed_bytes for q in quantizers)

    # Pre-computed group masks for norm correction (pack_perm order, fp32)
    # Shape: (n_kv, max_n_groups, HD).  mask[h, g, d] = 1.0 if pack_perm
    # dimension d belongs to group g for head h.  Avoids boolean indexing
    # and .any() checks, making norm correction CUDA-Graph safe.
    k_group_mask_stack = torch.zeros(n_kv, max_n_groups, hd,
                                      dtype=torch.float32, device=device)
    for i, q in enumerate(quantizers):
        # grp_stack is in head_perm order; inv_pack_perm maps packed→head_perm
        ipp = q._inv_pack_perm  # (HD,) — for each packed dim, its head_perm dim
        for g in range(q._n_groups):
            if q._group_of is not None:
                grp_hp = (q._group_of == g)  # (HD,) bool, head_perm order
            else:
                grp_hp = torch.ones(hd, dtype=torch.bool, device=device)
            # Map to pack_perm order
            k_group_mask_stack[i, g] = grp_hp[ipp].float()

    # Mixed packing (K_NIBBLE): nibble-4 for bw≤4, uint8 for bw>4
    max_mixed_bytes = 0
    mixed_nopack_starts = []  # per-head nopack_start (plain int list)
    mixed_nopack_lens = []
    for q in quantizers:
        m = q._pack_meta
        ns = m['nopack_start']
        nl = m['nopack_len']
        mixed_nopack_starts.append(ns)
        mixed_nopack_lens.append(nl)
        max_mixed_bytes = max(max_mixed_bytes, ns // 2 + nl)

    from blockgtq.v_packing import packed_v_bytes
    n_cents = 1 << v_bits
    vpb = packed_v_bytes(hd, v_bits)

    return {
        # V stacked
        'v_rot_T_stack': v_rot_T_stack.contiguous(),
        'v_bound_stack': v_bound_stack.contiguous(),
        'v_cent_stack': v_cent_stack.contiguous(),
        'v_bits': v_bits,
        'v_n_cents': n_cents,
        'v_packed_bytes': vpb,
        'v_nibble_bytes': hd // 2,
        # K stacked
        'k_perm_stack': k_perm_stack.contiguous(),
        'k_inv_pp_stack': k_inv_pp_stack.contiguous(),
        'k_rot_stack': k_rot_stack.contiguous(),
        'k_clut_stack': clut_stack.contiguous(),
        'k_pos_cents_stack': k_pos_cents_stack.contiguous(),
        'k_ptcb_stack': k_ptcb_stack.contiguous(),
        'k_loff_stack': k_loff_stack.contiguous(),
        'k_linv_stack': k_linv_stack.contiguous(),
        'k_grp_stack': k_grp_stack.contiguous(),
        'k_group_mask_stack': k_group_mask_stack.contiguous(),
        'k_pack_params': pack_params.contiguous(),
        'k_clut_stride': clut_stride,
        # Constexpr
        'n_kv': n_kv,
        'hd': hd,
        'n_bins': n_bins,
        'max_n_groups': max_n_groups,
        'max_n_codebooks': max_n_codebooks,
        'max_tpb': max_tpb,
        'max_bw1_len': max_bw1_len,
        'max_bw2_len': max_bw2_len,
        'max_bw3_len': max_bw3_len,
        'max_bw4_len': max_bw4_len,
        'max_np_len': max_np_len,
        'max_mixed_bytes': max_mixed_bytes,
        'mixed_nopack_starts': mixed_nopack_starts,
        'mixed_nopack_lens': mixed_nopack_lens,
    }


@torch.no_grad()
def compress_k_nibble4_batched(k_flat: torch.Tensor, args: dict,
                                block_n: int = 32
                                ) -> tuple[torch.Tensor, torch.Tensor]:
    """Batched K encode → nibble-4 format for K_NIBBLE decode.

    Runs the same compress+pack kernel as compress_k_packed_batched,
    but extracts raw codes from the scratch buffer and repacks as
    nibble-4 (2 codes per byte: lo nibble + hi nibble).

    The mixed-bit packed output is discarded — only nibble-4 + norms
    are returned.

    Args:
        k_flat: (n_kv, N, HD) fp16 key vectors (all heads stacked).
        args: dict from build_batched_encode_args.

    Returns:
        k_nib4: (n_kv, N, HD//2) uint8 nibble-4 packed codes.
        norms: (n_kv, N, max_n_groups) float32 per-group norms.
    """
    n_kv, N, hd = k_flat.shape
    max_tpb = args['max_tpb']
    max_ng = args['max_n_groups']
    device = k_flat.device

    # packed output (kernel writes mixed-bit here — we discard it)
    packed = torch.zeros(n_kv, N, max_tpb, dtype=torch.uint8, device=device)
    norms = torch.zeros(n_kv, N, max_ng, dtype=torch.float32, device=device)

    BLOCK_N = block_n
    n_blocks = (N + BLOCK_N - 1) // BLOCK_N
    scratch = torch.empty(n_kv, n_blocks * hd * BLOCK_N, dtype=torch.uint8,
                          device=device)

    clut_stride = args['k_clut_stride']

    grid = (n_kv * n_blocks,)
    _compress_pack_kernel_v8_batched[grid](
        k_flat, packed, norms, scratch,
        args['k_perm_stack'],
        args['k_inv_pp_stack'],
        args['k_rot_stack'],
        args['k_clut_stack'],
        args['k_ptcb_stack'],
        args['k_loff_stack'],
        args['k_linv_stack'],
        args['k_grp_stack'],
        args['k_pack_params'],
        k_flat.stride(1),   # stride_xn = HD
        packed.stride(1),   # stride_packed_n = max_tpb
        norms.stride(1),    # stride_norms_n = max_ng
        clut_stride,
        N * hd,             # stride_x_head
        N * max_tpb,        # stride_packed_head
        N * max_ng,         # stride_norms_head
        n_blocks * hd * BLOCK_N,  # stride_scratch_head
        hd,                 # stride_perm_head
        hd * hd,            # stride_rot_head
        args['max_n_codebooks'] * clut_stride,  # stride_clut_head
        hd,                 # stride_ptcb_head
        hd,                 # stride_loff_head
        hd,                 # stride_linv_head
        hd,                 # stride_grp_head
        N, n_blocks,
        HD=hd,
        MAX_N_GROUPS=args['max_n_groups'],
        N_BINS=args['n_bins'],
        MAX_N_CODEBOOKS=args['max_n_codebooks'],
        BLOCK_N=BLOCK_N,
        MAX_BW1_LEN=args['max_bw1_len'],
        MAX_BW2_LEN=args['max_bw2_len'],
        MAX_BW3_LEN=args['max_bw3_len'],
        MAX_BW4_LEN=args['max_bw4_len'],
        MAX_NP_LEN=args['max_np_len'],
    )

    # Raw codes live in scratch (column-major, pack_perm order, untouched by Step 7)
    # Layout: scratch[h, b*HD*BLOCK_N + d*BLOCK_N + r] = code(h, b*BLOCK_N+r, d)
    N_padded = n_blocks * BLOCK_N
    codes = (scratch
             .reshape(n_kv, n_blocks, hd, BLOCK_N)
             .permute(0, 1, 3, 2)
             .reshape(n_kv, N_padded, hd)[:, :N, :])

    # Nibble-4 pack: pair adjacent dims into nibble bytes
    even = codes[:, :, 0::2].to(torch.int32)
    odd  = codes[:, :, 1::2].to(torch.int32)
    k_nib4 = ((even & 0xF) | ((odd & 0xF) << 4)).to(torch.uint8)

    return k_nib4.contiguous(), norms


@torch.no_grad()
def compress_k_mixed_batched(k_flat: torch.Tensor, args: dict,
                              block_n: int = 32,
                              norm_correct: bool = True,
                              ) -> tuple[torch.Tensor, torch.Tensor]:
    """Batched K encode → mixed packing: nibble-4 for bw≤4, uint8 for bw>4.

    Output layout per head per token:
      [nibble_bytes: nopack_start//2] [uint8_bytes: nopack_len]

    This fixes the truncation bug in compress_k_nibble4_batched where
    5-8 bit codes were masked to 4 bits via & 0xF.

    Args:
        k_flat: (n_kv, N, HD) fp16 key vectors (all heads stacked).
        args: dict from build_batched_encode_args.
        norm_correct: if True, correct norms by dividing by ||y_hat_group||
            to match compress_decompress quality. Uses the shared LUT
            centroid values from args['k_slut_stack'].

    Returns:
        k_mixed: (n_kv, N, max_mixed_bytes) uint8 mixed-packed codes.
        norms: (n_kv, N, max_n_groups) float32 per-group norms.
    """
    n_kv, N, hd = k_flat.shape
    max_tpb = args['max_tpb']
    max_ng = args['max_n_groups']
    device = k_flat.device

    # packed output (discarded — only codes from scratch are used)
    packed = torch.zeros(n_kv, N, max_tpb, dtype=torch.uint8, device=device)
    norms = torch.zeros(n_kv, N, max_ng, dtype=torch.float32, device=device)

    BLOCK_N = block_n
    n_blocks = (N + BLOCK_N - 1) // BLOCK_N
    scratch = torch.empty(n_kv, n_blocks * hd * BLOCK_N, dtype=torch.uint8,
                          device=device)

    clut_stride = args['k_clut_stride']

    grid = (n_kv * n_blocks,)
    _compress_pack_kernel_v8_batched[grid](
        k_flat, packed, norms, scratch,
        args['k_perm_stack'],
        args['k_inv_pp_stack'],
        args['k_rot_stack'],
        args['k_clut_stack'],
        args['k_ptcb_stack'],
        args['k_loff_stack'],
        args['k_linv_stack'],
        args['k_grp_stack'],
        args['k_pack_params'],
        k_flat.stride(1),   # stride_xn = HD
        packed.stride(1),   # stride_packed_n = max_tpb
        norms.stride(1),    # stride_norms_n = max_ng
        clut_stride,
        N * hd,             # stride_x_head
        N * max_tpb,        # stride_packed_head
        N * max_ng,         # stride_norms_head
        n_blocks * hd * BLOCK_N,  # stride_scratch_head
        hd,                 # stride_perm_head
        hd * hd,            # stride_rot_head
        args['max_n_codebooks'] * clut_stride,  # stride_clut_head
        hd,                 # stride_ptcb_head
        hd,                 # stride_loff_head
        hd,                 # stride_linv_head
        hd,                 # stride_grp_head
        N, n_blocks,
        HD=hd,
        MAX_N_GROUPS=args['max_n_groups'],
        N_BINS=args['n_bins'],
        MAX_N_CODEBOOKS=args['max_n_codebooks'],
        BLOCK_N=BLOCK_N,
        MAX_BW1_LEN=args['max_bw1_len'],
        MAX_BW2_LEN=args['max_bw2_len'],
        MAX_BW3_LEN=args['max_bw3_len'],
        MAX_BW4_LEN=args['max_bw4_len'],
        MAX_NP_LEN=args['max_np_len'],
    )

    # Extract raw codes from scratch (pack_perm order per head)
    N_padded = n_blocks * BLOCK_N
    codes = (scratch
             .reshape(n_kv, n_blocks, hd, BLOCK_N)
             .permute(0, 1, 3, 2)
             .reshape(n_kv, N_padded, hd)[:, :N, :])

    # Per-head split: nibble section (bw≤4) + uint8 section (bw>4)
    # Use precomputed plain-int lists (no .item() calls → CUDA Graph safe)
    nopack_starts = args['mixed_nopack_starts']  # list of int
    nopack_lens = args['mixed_nopack_lens']
    max_mixed = args['max_mixed_bytes']

    k_mixed = torch.zeros(n_kv, N, max_mixed, dtype=torch.uint8, device=device)

    for h in range(n_kv):
        ns = nopack_starts[h]  # nopack_start for this head
        nl = nopack_lens[h]
        nib_bytes = ns // 2

        if ns > 0:
            # Nibble-4 section: pack pairs of ≤4-bit codes
            even = codes[h, :, 0:ns:2].to(torch.int32)
            odd = codes[h, :, 1:ns:2].to(torch.int32)
            k_mixed[h, :, :nib_bytes] = ((even & 0xF) | ((odd & 0xF) << 4)).to(torch.uint8)

        if nl > 0:
            # Uint8 section: raw codes for 5-8 bit dims (no truncation)
            k_mixed[h, :, nib_bytes:nib_bytes + nl] = codes[h, :, ns:ns + nl]

    # --- Norm correction: divide norms by ||y_hat_group|| ---
    # CUDA-Graph safe: no .any(), no boolean indexing, no GPU→CPU sync.
    # Uses pre-computed group masks (float32) for mask-multiply approach.
    if norm_correct and 'k_pos_cents_stack' in args:
        pc = args['k_pos_cents_stack']       # (n_kv, HD, max_cents) fp32
        inv_pp = args['k_inv_pp_stack']      # (n_kv, HD) int64
        gmask = args['k_group_mask_stack']   # (n_kv, max_ng, HD) fp32
        max_ng = args['max_n_groups']

        for h in range(n_kv):
            hp_dims = inv_pp[h]  # (HD,) head_perm pos for each packed dim
            codes_h = codes[h].long()  # (N, HD)
            # Gather centroid values: pc[h, hp_dim, code]
            cent_vals = pc[h, hp_dims.unsqueeze(0).expand(N, -1),
                          codes_h]  # (N, HD) fp32
            cent_sq = cent_vals * cent_vals   # (N, HD)
            for g in range(max_ng):
                g_mask = gmask[h, g]  # (HD,) fp32, pre-computed
                # Per-group ||y_hat|| via mask multiply (fixed-shape, no sync)
                g_norm_sq = (cent_sq * g_mask.unsqueeze(0)).sum(dim=-1)
                g_norm = g_norm_sq.sqrt().clamp(min=1e-10)  # (N,)
                norms[h, :, g] /= g_norm

    return k_mixed.contiguous(), norms


@torch.no_grad()
def compress_v_packed_batched(v_flat: torch.Tensor, args: dict,
                               block_n: int = 32
                               ) -> tuple[torch.Tensor, torch.Tensor]:
    """Batched V encode for all heads in one kernel launch.

    Args:
        v_flat: (n_kv, N, HD) fp16 value vectors (all heads stacked).
        args: dict from build_batched_encode_args.

    Returns:
        packed: (n_kv, N, vpb) uint8 packed codes.
        norms: (n_kv, N) fp16 corrected norms.
    """
    n_kv, N, hd = v_flat.shape
    vpb = args['v_packed_bytes']
    v_bits = args['v_bits']
    n_cents = args['v_n_cents']
    device = v_flat.device

    packed = torch.empty(n_kv, N, vpb, dtype=torch.uint8, device=device)
    norms = torch.empty(n_kv, N, dtype=torch.float16, device=device)

    BLOCK_N = block_n
    n_blocks = (N + BLOCK_N - 1) // BLOCK_N
    scratch = torch.empty(n_kv, n_blocks * hd * BLOCK_N, dtype=torch.uint8,
                          device=device)

    grid = (n_kv * n_blocks,)
    _compress_v_pack_kernel_batched[grid](
        v_flat, packed, norms, scratch,
        args['v_rot_T_stack'],
        args['v_bound_stack'],
        args['v_cent_stack'],
        v_flat.stride(1),   # stride_vn = HD
        packed.stride(1),   # stride_packed_n = vpb
        N * hd,             # stride_v_head
        N * vpb,            # stride_packed_head
        N,                  # stride_norms_head
        n_blocks * hd * BLOCK_N,  # stride_scratch_head
        hd * hd,            # stride_rot_head
        n_cents - 1,        # stride_bound_head
        n_cents,            # stride_cent_head
        N, n_blocks,
        HD=hd,
        N_CENTS=n_cents,
        V_BITS=v_bits,
        PACKED_BYTES=vpb,
        BLOCK_N=BLOCK_N,
    )
    return packed, norms


@torch.no_grad()
def compress_v_nibble4_batched(v_flat: torch.Tensor, args: dict,
                                block_n: int = 32
                                ) -> tuple[torch.Tensor, torch.Tensor]:
    """Batched V encode → nibble-4 format for V_NIBBLE decode.

    Same quantization as compress_v_packed_batched but repacks codes
    as nibble-4 (2 codes per byte: lo = even dim, hi = odd dim).
    Output: (n_kv, N, HD//2) uint8.  Faster to decode in the fused
    attention kernel via V_NIBBLE=1.

    Args:
        v_flat: (n_kv, N, HD) fp16 value vectors (all heads stacked).
        args: dict from build_batched_encode_args.

    Returns:
        nibble4: (n_kv, N, HD//2) uint8 nibble-4 packed codes.
        norms: (n_kv, N) fp16 corrected norms.
    """
    n_kv, N, hd = v_flat.shape
    vpb = args['v_packed_bytes']
    v_bits = args['v_bits']
    n_cents = args['v_n_cents']
    device = v_flat.device

    # Run the same kernel — we only need the scratch (raw codes) + norms
    packed = torch.empty(n_kv, N, vpb, dtype=torch.uint8, device=device)
    norms = torch.empty(n_kv, N, dtype=torch.float16, device=device)

    BLOCK_N = block_n
    n_blocks = (N + BLOCK_N - 1) // BLOCK_N
    scratch = torch.empty(n_kv, n_blocks * hd * BLOCK_N, dtype=torch.uint8,
                          device=device)

    grid = (n_kv * n_blocks,)
    _compress_v_pack_kernel_batched[grid](
        v_flat, packed, norms, scratch,
        args['v_rot_T_stack'],
        args['v_bound_stack'],
        args['v_cent_stack'],
        v_flat.stride(1),
        packed.stride(1),
        N * hd,
        N * vpb,
        N,
        n_blocks * hd * BLOCK_N,
        hd * hd,
        n_cents - 1,
        n_cents,
        N, n_blocks,
        HD=hd,
        N_CENTS=n_cents,
        V_BITS=v_bits,
        PACKED_BYTES=vpb,
        BLOCK_N=BLOCK_N,
    )

    # Extract raw codes from scratch (col-major → row-major)
    N_padded = n_blocks * BLOCK_N
    codes = (scratch
             .reshape(n_kv, n_blocks, hd, BLOCK_N)
             .permute(0, 1, 3, 2)
             .reshape(n_kv, N_padded, hd)[:, :N, :])

    # Pack as nibble-4: byte[i] = code[2i] | (code[2i+1] << 4)
    even = codes[:, :, 0::2].to(torch.int32)
    odd  = codes[:, :, 1::2].to(torch.int32)
    nibble4 = ((even & 0xF) | ((odd & 0xF) << 4)).to(torch.uint8)

    return nibble4.contiguous(), norms


@torch.no_grad()
def compress_k_packed_batched(k_flat: torch.Tensor, args: dict,
                               block_n: int = 32
                               ) -> tuple[torch.Tensor, torch.Tensor]:
    """Batched K encode for all heads in one kernel launch.

    Args:
        k_flat: (n_kv, N, HD) fp16 key vectors (all heads stacked).
        args: dict from build_batched_encode_args.

    Returns:
        packed: (n_kv, N, max_tpb) uint8 packed codes.
        norms: (n_kv, N, max_n_groups) float32 per-group norms.
    """
    n_kv, N, hd = k_flat.shape
    max_tpb = args['max_tpb']
    max_ng = args['max_n_groups']
    device = k_flat.device

    packed = torch.zeros(n_kv, N, max_tpb, dtype=torch.uint8, device=device)
    norms = torch.zeros(n_kv, N, max_ng, dtype=torch.float32, device=device)

    BLOCK_N = block_n
    n_blocks = (N + BLOCK_N - 1) // BLOCK_N
    scratch = torch.empty(n_kv, n_blocks * hd * BLOCK_N, dtype=torch.uint8,
                          device=device)

    clut_stride = args['k_clut_stride']

    grid = (n_kv * n_blocks,)
    _compress_pack_kernel_v8_batched[grid](
        k_flat, packed, norms, scratch,
        args['k_perm_stack'],
        args['k_inv_pp_stack'],
        args['k_rot_stack'],
        args['k_clut_stack'],
        args['k_ptcb_stack'],
        args['k_loff_stack'],
        args['k_linv_stack'],
        args['k_grp_stack'],
        args['k_pack_params'],
        k_flat.stride(1),   # stride_xn = HD
        packed.stride(1),   # stride_packed_n = max_tpb
        norms.stride(1),    # stride_norms_n = max_ng
        clut_stride,
        N * hd,             # stride_x_head
        N * max_tpb,        # stride_packed_head
        N * max_ng,         # stride_norms_head
        n_blocks * hd * BLOCK_N,  # stride_scratch_head
        hd,                 # stride_perm_head
        hd * hd,            # stride_rot_head
        args['max_n_codebooks'] * clut_stride,  # stride_clut_head
        hd,                 # stride_ptcb_head
        hd,                 # stride_loff_head
        hd,                 # stride_linv_head
        hd,                 # stride_grp_head
        N, n_blocks,
        HD=hd,
        MAX_N_GROUPS=args['max_n_groups'],
        N_BINS=args['n_bins'],
        MAX_N_CODEBOOKS=args['max_n_codebooks'],
        BLOCK_N=BLOCK_N,
        MAX_BW1_LEN=args['max_bw1_len'],
        MAX_BW2_LEN=args['max_bw2_len'],
        MAX_BW3_LEN=args['max_bw3_len'],
        MAX_BW4_LEN=args['max_bw4_len'],
        MAX_NP_LEN=args['max_np_len'],
    )
    return packed, norms
