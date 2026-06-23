"""Fused dequant + FlashAttention kernel reading bit-packed K codes.

The decode kernel walks the packed K segments (one per bit-width): load packed
bytes → unpack → codebook gather → per-segment partial dot for the QK score,
with a tensor-core path (tl.dot) for the wider nibble-packed segments. It uses
FlashDecoding split-T (compute + merge), supports GQA, and handles V as TQ-MSE
codes (raw uint8 or packed) with int4 group norms.

K packed layout: (B, H_kv, T, total_packed_bytes) uint8.
  [1b packed | 2b packed | 3b packed | 4b packed | 5b packed | 6b raw | 7b raw | 8b raw]
"""

import torch
import math

try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False


# ===========================================================================
# Triton kernels
# ===========================================================================

if HAS_TRITON:

    # Shared FlashDecoding helpers: per-block norm unpack + split-T merge.
    @triton.jit
    def _unpack_norms_mixed(norms_ptr, scale_ptr,
                            b64, hkv64, offs_t64, block_ids, mask_t,
                            n_stride_b, n_stride_h, n_stride_t, n_stride_blk,
                            s_stride_b, s_stride_h,
                            NORM_BITS: tl.constexpr):
        """Load and dequantize per-block norms (int4 or fp16). Same as _unpack_norms."""
        if NORM_BITS == 4:
            packed_idx = (block_ids // 2).to(tl.int64)
            is_lo = (block_ids & 1) == 1
            offset = (b64 * n_stride_b + hkv64 * n_stride_h
                      + offs_t64[:, None] * n_stride_t
                      + packed_idx[None, :] * n_stride_blk)
            packed = tl.load(norms_ptr + offset,
                             mask=mask_t[:, None], other=0).to(tl.int32)
            hi = (packed >> 4) & 0xF
            lo = packed & 0xF
            norm_int = tl.where(is_lo[None, :], lo, hi)
            scale = tl.load(scale_ptr + b64 * s_stride_b
                            + hkv64 * s_stride_h).to(tl.float32)
            return norm_int.to(tl.float32) / 15.0 * scale
        else:
            offset = (b64 * n_stride_b + hkv64 * n_stride_h
                      + offs_t64[:, None] * n_stride_t
                      + block_ids[None, :] * n_stride_blk)
            return tl.load(norms_ptr + offset,
                           mask=mask_t[:, None], other=1.0).to(tl.float32)

    @triton.jit
    def _split_t_merge_kernel(
        partial_out_ptr,    # (B, n_q_heads, T_SPLITS, D) fp16
        partial_lse_ptr,    # (B, n_q_heads, T_SPLITS) fp32
        out_ptr,            # (B, n_q_heads, D) fp16
        po_stride_b, po_stride_h, po_stride_c, po_stride_d,
        pl_stride_b, pl_stride_h, pl_stride_c,
        o_stride_b, o_stride_h, o_stride_d,
        n_q_heads,
        D: tl.constexpr,
        BLOCK_D: tl.constexpr,
        T_SPLITS: tl.constexpr,
    ):
        """Phase 2: Merge T_SPLITS partial outputs into the final output."""
        pid = tl.program_id(0)
        b = pid // n_q_heads
        h_q = pid % n_q_heads
        b64 = b.to(tl.int64)
        hq64 = h_q.to(tl.int64)

        offs_d = tl.arange(0, BLOCK_D)
        offs_c = tl.arange(0, T_SPLITS)

        # Load partial_lse: (T_SPLITS,)
        pl_offset = (b64 * pl_stride_b + hq64 * pl_stride_h
                     + offs_c.to(tl.int64) * pl_stride_c)
        partial_lse = tl.load(partial_lse_ptr + pl_offset)

        # Find global max LSE for numerical stability
        m = tl.max(partial_lse, axis=0)
        alpha = tl.exp(partial_lse - m)  # (T_SPLITS,)
        denom = tl.sum(alpha, axis=0)    # scalar

        # Load partial_out: (T_SPLITS, BLOCK_D)
        po_offset = (b64 * po_stride_b + hq64 * po_stride_h
                     + offs_c[:, None].to(tl.int64) * po_stride_c
                     + offs_d[None, :] * po_stride_d)
        partial_out = tl.load(partial_out_ptr + po_offset).to(tl.float32)

        # Weighted sum
        weighted_out = tl.sum(alpha[:, None] * partial_out, axis=0)
        final = weighted_out / denom

        o_offset = b64 * o_stride_b + hq64 * o_stride_h + offs_d * o_stride_d
        tl.store(out_ptr + o_offset, final.to(tl.float16))

    # -----------------------------------------------------------------------
    # V codes loading: raw uint8 or packed-bit unpack
    # -----------------------------------------------------------------------

    @triton.jit
    def _load_v_codes(
        v_codes_ptr, v_unpack_ptr,
        b64, hkv64, offs_t64, offs_d, mask_t,
        vc_stride_b, vc_stride_h, vc_stride_t, vc_stride_d,
        vu_stride,
        V_BITS: tl.constexpr,
    ):
        """Load V codes — raw uint8 (V_BITS=0) or unpack from packed bytes.

        When V_BITS > 0, v_codes_ptr points to packed V bytes and
        v_unpack_ptr to a (5, D) int32 table: [byte_lo, byte_hi,
        shift_lo, shift_hi, mask]. Two scattered byte loads + bitwise
        ops extract all codes.
        """
        if V_BITS > 0:
            byte_lo = tl.load(v_unpack_ptr + offs_d).to(tl.int64)
            byte_hi = tl.load(v_unpack_ptr + vu_stride + offs_d).to(tl.int64)
            sh_lo = tl.load(v_unpack_ptr + 2 * vu_stride + offs_d)
            sh_hi = tl.load(v_unpack_ptr + 3 * vu_stride + offs_d)
            vmask = tl.load(v_unpack_ptr + 4 * vu_stride + offs_d)

            vp_base = (b64 * vc_stride_b + hkv64 * vc_stride_h
                       + offs_t64 * vc_stride_t)
            lo = tl.load(v_codes_ptr + vp_base[:, None] + byte_lo[None, :],
                         mask=mask_t[:, None], other=0).to(tl.int32)
            hi = tl.load(v_codes_ptr + vp_base[:, None] + byte_hi[None, :],
                         mask=mask_t[:, None], other=0).to(tl.int32)
            v_codes = ((lo >> sh_lo[None, :])
                       | (hi << sh_hi[None, :])) & vmask[None, :]
        else:
            vc_offset = (b64 * vc_stride_b + hkv64 * vc_stride_h
                         + offs_t64[:, None] * vc_stride_t
                         + offs_d[None, :] * vc_stride_d)
            v_codes = tl.load(v_codes_ptr + vc_offset,
                              mask=mask_t[:, None], other=0).to(tl.int32)
        return v_codes


    # -----------------------------------------------------------------------
    # Decode kernel: per-segment dequant + partial-dot QK (no tensor cores for QK)
    # -----------------------------------------------------------------------

    @triton.jit
    def _packed_decode_kernel(
        q_ptr,
        kp_ptr,             # (B, H_kv, T, total_packed_bytes) uint8
        k_norms_ptr,        # (B, H_kv, T, N_GROUPS) fp16
        v_codes_ptr,        # raw (B,H,T,D) uint8 OR packed (B,H,T,pb) uint8
        v_norms_ptr,
        v_lut_ptr,
        v_norm_scale_ptr,
        v_unpack_ptr,       # (5, D) int32 V unpack tables (or dummy if V_BITS=0)
        v_rot_ptr,          # (D, D) fp32 V un-rotation matrix (or dummy if V_UNROT=0)
        partial_out_ptr,
        partial_lse_ptr,
        codebook_ptr,
        byte_lut_ptr,       # 1-bit byte-LUT (B*H_q, total_1bit_groups, 256) fp32
        # Per-dim LUT offset (D,) int32
        lut_offset_ptr,
        # Per-dim norm group index (D,) int32
        norm_group_ptr,
        # Strides
        q_stride_b, q_stride_h, q_stride_d,
        kp_stride_b, kp_stride_h, kp_stride_t,
        kn_stride_b, kn_stride_h, kn_stride_t, kn_stride_s,
        vc_stride_b, vc_stride_h, vc_stride_t, vc_stride_d,
        vn_stride_b, vn_stride_h, vn_stride_t, vn_stride_blk,
        vns_stride_b, vns_stride_h,
        vu_stride,          # stride between rows of v_unpack (= D)
        po_stride_b, po_stride_h, po_stride_c, po_stride_d,
        pl_stride_b, pl_stride_h, pl_stride_c,
        # Sizes
        T, n_q_heads, n_kv_heads,
        # Segment constexprs (up to 8 segments)
        SEG0_BW: tl.constexpr, SEG0_START: tl.constexpr, SEG0_LEN: tl.constexpr,
        SEG0_PACK_OFF: tl.constexpr, SEG0_LUT_OFF: tl.constexpr,
        SEG0_NORM_IDX: tl.constexpr, SEG0_1BIT_GOFF: tl.constexpr,
        SEG1_BW: tl.constexpr, SEG1_START: tl.constexpr, SEG1_LEN: tl.constexpr,
        SEG1_PACK_OFF: tl.constexpr, SEG1_LUT_OFF: tl.constexpr,
        SEG1_NORM_IDX: tl.constexpr, SEG1_1BIT_GOFF: tl.constexpr,
        SEG2_BW: tl.constexpr, SEG2_START: tl.constexpr, SEG2_LEN: tl.constexpr,
        SEG2_PACK_OFF: tl.constexpr, SEG2_LUT_OFF: tl.constexpr,
        SEG2_NORM_IDX: tl.constexpr, SEG2_1BIT_GOFF: tl.constexpr,
        SEG3_BW: tl.constexpr, SEG3_START: tl.constexpr, SEG3_LEN: tl.constexpr,
        SEG3_PACK_OFF: tl.constexpr, SEG3_LUT_OFF: tl.constexpr,
        SEG3_NORM_IDX: tl.constexpr, SEG3_1BIT_GOFF: tl.constexpr,
        SEG4_BW: tl.constexpr, SEG4_START: tl.constexpr, SEG4_LEN: tl.constexpr,
        SEG4_PACK_OFF: tl.constexpr, SEG4_LUT_OFF: tl.constexpr,
        SEG4_NORM_IDX: tl.constexpr, SEG4_1BIT_GOFF: tl.constexpr,
        SEG5_BW: tl.constexpr, SEG5_START: tl.constexpr, SEG5_LEN: tl.constexpr,
        SEG5_PACK_OFF: tl.constexpr, SEG5_LUT_OFF: tl.constexpr,
        SEG5_NORM_IDX: tl.constexpr, SEG5_1BIT_GOFF: tl.constexpr,
        SEG6_BW: tl.constexpr, SEG6_START: tl.constexpr, SEG6_LEN: tl.constexpr,
        SEG6_PACK_OFF: tl.constexpr, SEG6_LUT_OFF: tl.constexpr,
        SEG6_NORM_IDX: tl.constexpr, SEG6_1BIT_GOFF: tl.constexpr,
        SEG7_BW: tl.constexpr, SEG7_START: tl.constexpr, SEG7_LEN: tl.constexpr,
        SEG7_PACK_OFF: tl.constexpr, SEG7_LUT_OFF: tl.constexpr,
        SEG7_NORM_IDX: tl.constexpr, SEG7_1BIT_GOFF: tl.constexpr,
        N_SEGMENTS: tl.constexpr,
        TOTAL_1BIT_GROUPS: tl.constexpr,
        # Constants
        D: tl.constexpr,
        BLOCK_M: tl.constexpr,
        GQA_RATIO: tl.constexpr,
        BLOCK_T: tl.constexpr,
        BLOCK_D: tl.constexpr,
        T_SPLITS: tl.constexpr,
        SCALE: tl.constexpr,
        V_SKIP_THRESH: tl.constexpr,
        V_QUANT_BLOCK: tl.constexpr,
        V_NORM_BITS: tl.constexpr,
        ONEBIT_FAST: tl.constexpr = 0,
        V_BITS: tl.constexpr = 0,
        V_UNROT: tl.constexpr = 0,
        V_FP16: tl.constexpr = 0,
        V_NIBBLE: tl.constexpr = 0,
        V_PAIRED_2B: tl.constexpr = 0,
        K_NIBBLE: tl.constexpr = 0,
        K_PAIRED_2B: tl.constexpr = 0,
        K_PAIRED_4B: tl.constexpr = 0,
        K_PAIRED_8B: tl.constexpr = 0,
        K_PAIRED_1B: tl.constexpr = 0,
        MIN_SEG_FOR_DOT: tl.constexpr = 16,
    ):
        """Per-segment unpack + partial-dot decode kernel.

        When K_NIBBLE=0 (default): original per-segment bit-packed decode
        with scalar QK dot (no tensor cores for QK).

        When K_NIBBLE=1: K codes are nibble-4 packed (D//2 bytes per token).
        Uses _seg_nibble_dot (tensor-core tl.dot for QK) for segments ≥
        MIN_SEG_FOR_DOT dims, nibble scalar fallback for tiny segments.

        When V_PAIRED_2B=1: 2-bit V paired-decode fast path. One byte per
        4 V dims; 4 mask/shift ops + 4 LUT lookups + 3-stage interleave
        reassemble a (BLOCK_T, D) fp16 tile without the scattered
        _load_v_codes path. Mutually exclusive with V_FP16 / V_NIBBLE.
        """
        pid = tl.program_id(0)
        chunk_id = pid % T_SPLITS
        bh = pid // T_SPLITS
        b = bh // n_kv_heads
        h_kv = bh % n_kv_heads

        b64 = b.to(tl.int64)
        hkv64 = h_kv.to(tl.int64)

        chunk_size = (T + T_SPLITS - 1) // T_SPLITS
        t_start_chunk = chunk_id * chunk_size
        t_end_chunk = tl.minimum(t_start_chunk + chunk_size, T)

        offs_d = tl.arange(0, BLOCK_D)
        offs_m = tl.arange(0, BLOCK_M)
        m_mask = offs_m < GQA_RATIO
        v_block_ids = (offs_d // V_QUANT_BLOCK).to(tl.int64)

        h_q_base = (hkv64 * GQA_RATIO).to(tl.int64)
        q_row = (h_q_base + offs_m.to(tl.int64))
        # q_base: per-head element offset for pointer-based q loading in _seg_partial_dot
        q_base = (b64 * q_stride_b + q_row * q_stride_h)

        # LUT head base: flat index into (B*H_q, total_groups, entries_per_group)
        if ONEBIT_FAST == 2:
            lut_head_base = (b64 * n_q_heads + q_row) * (TOTAL_1BIT_GROUPS * 256)
        elif ONEBIT_FAST == 3:
            lut_head_base = (b64 * n_q_heads + q_row) * (TOTAL_1BIT_GROUPS * 16)
        else:
            lut_head_base = tl.zeros([BLOCK_M], dtype=tl.int64)

        m_i = tl.full((BLOCK_M,), -float('inf'), dtype=tl.float32)
        l_i = tl.zeros((BLOCK_M,), dtype=tl.float32)
        acc = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)

        for t_start in range(t_start_chunk, t_end_chunk, BLOCK_T):
            offs_t = t_start + tl.arange(0, BLOCK_T)
            mask_t = offs_t < t_end_chunk
            offs_t64 = offs_t.to(tl.int64)

            kp_base = (b64 * kp_stride_b + hkv64 * kp_stride_h
                       + offs_t64 * kp_stride_t)

            # Accumulate QK logits per-segment
            logits = tl.zeros((BLOCK_M, BLOCK_T), dtype=tl.float32)

            if K_NIBBLE:
                # ============================================================
                # K_NIBBLE mixed-packing path:
                #   bw ≤ 4 → nibble-4 (tensor-core or scalar)
                #   bw > 4 → uint8 (tensor-core or scalar)
                # Buffer layout: [nibble_bytes | uint8_bytes] per token
                # SEG_PACK_OFF for bw>4 = byte offset into mixed buffer
                # ============================================================

                # ---- Segment 0 ----
                if SEG0_BW <= 4 and SEG0_LEN >= MIN_SEG_FOR_DOT:
                    kn_off = (b64 * kn_stride_b + hkv64 * kn_stride_h
                              + offs_t64 * kn_stride_t + SEG0_NORM_IDX * kn_stride_s)
                    seg0_norm = tl.load(k_norms_ptr + kn_off, mask=mask_t, other=1.0)
                    logits += _seg_nibble_dot(
                        kp_ptr, kp_base, codebook_ptr, seg0_norm,
                        q_ptr, q_base, q_stride_d, m_mask, mask_t,
                        SEG0_START, SEG0_LEN, SEG0_LUT_OFF, BLOCK_M, BLOCK_T)
                elif SEG0_BW <= 4 and SEG0_LEN > 0:
                    kn_off = (b64 * kn_stride_b + hkv64 * kn_stride_h
                              + offs_t64 * kn_stride_t + SEG0_NORM_IDX * kn_stride_s)
                    seg_norm = tl.load(k_norms_ptr + kn_off, mask=mask_t, other=1.0)
                    NIB_OFF_0: tl.constexpr = SEG0_START // 2
                    for _k in tl.static_range(0, (SEG0_LEN + 1) // 2):
                        byte = tl.load(kp_ptr + kp_base + NIB_OFF_0 + _k,
                                       mask=mask_t, other=0).to(tl.int32)
                        lo_code = byte & 0xF
                        hi_code = (byte >> 4) & 0xF
                        if _k * 2 < SEG0_LEN:
                            cent = tl.load(codebook_ptr + SEG0_LUT_OFF + lo_code).to(tl.float32)
                            qv = tl.load(q_ptr + q_base + (SEG0_START + _k * 2) * q_stride_d,
                                         mask=m_mask, other=0.0).to(tl.float32)
                            logits += qv[:, None] * (cent * seg_norm)[None, :]
                        if _k * 2 + 1 < SEG0_LEN:
                            cent = tl.load(codebook_ptr + SEG0_LUT_OFF + hi_code).to(tl.float32)
                            qv = tl.load(q_ptr + q_base + (SEG0_START + _k * 2 + 1) * q_stride_d,
                                         mask=m_mask, other=0.0).to(tl.float32)
                            logits += qv[:, None] * (cent * seg_norm)[None, :]
                elif SEG0_LEN >= MIN_SEG_FOR_DOT:  # bw > 4: uint8 tensor-core
                    kn_off = (b64 * kn_stride_b + hkv64 * kn_stride_h
                              + offs_t64 * kn_stride_t + SEG0_NORM_IDX * kn_stride_s)
                    seg_norm = tl.load(k_norms_ptr + kn_off, mask=mask_t, other=1.0)
                    logits += _seg_uint8_dot(
                        kp_ptr, kp_base, codebook_ptr, seg_norm,
                        q_ptr, q_base, q_stride_d, m_mask, mask_t,
                        SEG0_START, SEG0_LEN, SEG0_LUT_OFF, SEG0_PACK_OFF,
                        BLOCK_M, BLOCK_T)
                elif SEG0_LEN > 0:  # bw > 4: uint8 scalar
                    kn_off = (b64 * kn_stride_b + hkv64 * kn_stride_h
                              + offs_t64 * kn_stride_t + SEG0_NORM_IDX * kn_stride_s)
                    seg_norm = tl.load(k_norms_ptr + kn_off, mask=mask_t, other=1.0)
                    for _k in tl.static_range(0, SEG0_LEN):
                        code = tl.load(kp_ptr + kp_base + SEG0_PACK_OFF + _k,
                                       mask=mask_t, other=0).to(tl.int32)
                        cent = tl.load(codebook_ptr + SEG0_LUT_OFF + code).to(tl.float32)
                        qv = tl.load(q_ptr + q_base + (SEG0_START + _k) * q_stride_d,
                                     mask=m_mask, other=0.0).to(tl.float32)
                        logits += qv[:, None] * (cent * seg_norm)[None, :]

                # ---- Segment 1 ----
                if N_SEGMENTS >= 2:
                    if SEG1_BW <= 4 and SEG1_LEN >= MIN_SEG_FOR_DOT:
                        kn_off = (b64 * kn_stride_b + hkv64 * kn_stride_h
                                  + offs_t64 * kn_stride_t + SEG1_NORM_IDX * kn_stride_s)
                        seg1_norm = tl.load(k_norms_ptr + kn_off, mask=mask_t, other=1.0)
                        logits += _seg_nibble_dot(
                            kp_ptr, kp_base, codebook_ptr, seg1_norm,
                            q_ptr, q_base, q_stride_d, m_mask, mask_t,
                            SEG1_START, SEG1_LEN, SEG1_LUT_OFF, BLOCK_M, BLOCK_T)
                    elif SEG1_BW <= 4 and SEG1_LEN > 0:
                        kn_off = (b64 * kn_stride_b + hkv64 * kn_stride_h
                                  + offs_t64 * kn_stride_t + SEG1_NORM_IDX * kn_stride_s)
                        seg_norm = tl.load(k_norms_ptr + kn_off, mask=mask_t, other=1.0)
                        NIB_OFF_1: tl.constexpr = SEG1_START // 2
                        for _k in tl.static_range(0, (SEG1_LEN + 1) // 2):
                            byte = tl.load(kp_ptr + kp_base + NIB_OFF_1 + _k,
                                           mask=mask_t, other=0).to(tl.int32)
                            lo_code = byte & 0xF
                            hi_code = (byte >> 4) & 0xF
                            if _k * 2 < SEG1_LEN:
                                cent = tl.load(codebook_ptr + SEG1_LUT_OFF + lo_code).to(tl.float32)
                                qv = tl.load(q_ptr + q_base + (SEG1_START + _k * 2) * q_stride_d,
                                             mask=m_mask, other=0.0).to(tl.float32)
                                logits += qv[:, None] * (cent * seg_norm)[None, :]
                            if _k * 2 + 1 < SEG1_LEN:
                                cent = tl.load(codebook_ptr + SEG1_LUT_OFF + hi_code).to(tl.float32)
                                qv = tl.load(q_ptr + q_base + (SEG1_START + _k * 2 + 1) * q_stride_d,
                                             mask=m_mask, other=0.0).to(tl.float32)
                                logits += qv[:, None] * (cent * seg_norm)[None, :]
                    elif SEG1_LEN >= MIN_SEG_FOR_DOT:  # bw > 4: uint8 tensor-core
                        kn_off = (b64 * kn_stride_b + hkv64 * kn_stride_h
                                  + offs_t64 * kn_stride_t + SEG1_NORM_IDX * kn_stride_s)
                        seg_norm = tl.load(k_norms_ptr + kn_off, mask=mask_t, other=1.0)
                        logits += _seg_uint8_dot(
                            kp_ptr, kp_base, codebook_ptr, seg_norm,
                            q_ptr, q_base, q_stride_d, m_mask, mask_t,
                            SEG1_START, SEG1_LEN, SEG1_LUT_OFF, SEG1_PACK_OFF,
                            BLOCK_M, BLOCK_T)
                    elif SEG1_LEN > 0:  # bw > 4: uint8 scalar
                        kn_off = (b64 * kn_stride_b + hkv64 * kn_stride_h
                                  + offs_t64 * kn_stride_t + SEG1_NORM_IDX * kn_stride_s)
                        seg_norm = tl.load(k_norms_ptr + kn_off, mask=mask_t, other=1.0)
                        for _k in tl.static_range(0, SEG1_LEN):
                            code = tl.load(kp_ptr + kp_base + SEG1_PACK_OFF + _k,
                                           mask=mask_t, other=0).to(tl.int32)
                            cent = tl.load(codebook_ptr + SEG1_LUT_OFF + code).to(tl.float32)
                            qv = tl.load(q_ptr + q_base + (SEG1_START + _k) * q_stride_d,
                                         mask=m_mask, other=0.0).to(tl.float32)
                            logits += qv[:, None] * (cent * seg_norm)[None, :]

                # ---- Segment 2 ----
                if N_SEGMENTS >= 3:
                    if SEG2_BW <= 4 and SEG2_LEN >= MIN_SEG_FOR_DOT:
                        kn_off = (b64 * kn_stride_b + hkv64 * kn_stride_h
                                  + offs_t64 * kn_stride_t + SEG2_NORM_IDX * kn_stride_s)
                        seg2_norm = tl.load(k_norms_ptr + kn_off, mask=mask_t, other=1.0)
                        logits += _seg_nibble_dot(
                            kp_ptr, kp_base, codebook_ptr, seg2_norm,
                            q_ptr, q_base, q_stride_d, m_mask, mask_t,
                            SEG2_START, SEG2_LEN, SEG2_LUT_OFF, BLOCK_M, BLOCK_T)
                    elif SEG2_BW <= 4 and SEG2_LEN > 0:
                        kn_off = (b64 * kn_stride_b + hkv64 * kn_stride_h
                                  + offs_t64 * kn_stride_t + SEG2_NORM_IDX * kn_stride_s)
                        seg_norm = tl.load(k_norms_ptr + kn_off, mask=mask_t, other=1.0)
                        NIB_OFF_2: tl.constexpr = SEG2_START // 2
                        for _k in tl.static_range(0, (SEG2_LEN + 1) // 2):
                            byte = tl.load(kp_ptr + kp_base + NIB_OFF_2 + _k,
                                           mask=mask_t, other=0).to(tl.int32)
                            lo_code = byte & 0xF
                            hi_code = (byte >> 4) & 0xF
                            if _k * 2 < SEG2_LEN:
                                cent = tl.load(codebook_ptr + SEG2_LUT_OFF + lo_code).to(tl.float32)
                                qv = tl.load(q_ptr + q_base + (SEG2_START + _k * 2) * q_stride_d,
                                             mask=m_mask, other=0.0).to(tl.float32)
                                logits += qv[:, None] * (cent * seg_norm)[None, :]
                            if _k * 2 + 1 < SEG2_LEN:
                                cent = tl.load(codebook_ptr + SEG2_LUT_OFF + hi_code).to(tl.float32)
                                qv = tl.load(q_ptr + q_base + (SEG2_START + _k * 2 + 1) * q_stride_d,
                                             mask=m_mask, other=0.0).to(tl.float32)
                                logits += qv[:, None] * (cent * seg_norm)[None, :]
                    elif SEG2_LEN >= MIN_SEG_FOR_DOT:  # bw > 4: uint8 tensor-core
                        kn_off = (b64 * kn_stride_b + hkv64 * kn_stride_h
                                  + offs_t64 * kn_stride_t + SEG2_NORM_IDX * kn_stride_s)
                        seg_norm = tl.load(k_norms_ptr + kn_off, mask=mask_t, other=1.0)
                        logits += _seg_uint8_dot(
                            kp_ptr, kp_base, codebook_ptr, seg_norm,
                            q_ptr, q_base, q_stride_d, m_mask, mask_t,
                            SEG2_START, SEG2_LEN, SEG2_LUT_OFF, SEG2_PACK_OFF,
                            BLOCK_M, BLOCK_T)
                    elif SEG2_LEN > 0:  # bw > 4: uint8 scalar
                        kn_off = (b64 * kn_stride_b + hkv64 * kn_stride_h
                                  + offs_t64 * kn_stride_t + SEG2_NORM_IDX * kn_stride_s)
                        seg_norm = tl.load(k_norms_ptr + kn_off, mask=mask_t, other=1.0)
                        for _k in tl.static_range(0, SEG2_LEN):
                            code = tl.load(kp_ptr + kp_base + SEG2_PACK_OFF + _k,
                                           mask=mask_t, other=0).to(tl.int32)
                            cent = tl.load(codebook_ptr + SEG2_LUT_OFF + code).to(tl.float32)
                            qv = tl.load(q_ptr + q_base + (SEG2_START + _k) * q_stride_d,
                                         mask=m_mask, other=0.0).to(tl.float32)
                            logits += qv[:, None] * (cent * seg_norm)[None, :]

                # ---- Segment 3 ----
                if N_SEGMENTS >= 4:
                    if SEG3_BW <= 4 and SEG3_LEN >= MIN_SEG_FOR_DOT:
                        kn_off = (b64 * kn_stride_b + hkv64 * kn_stride_h
                                  + offs_t64 * kn_stride_t + SEG3_NORM_IDX * kn_stride_s)
                        seg3_norm = tl.load(k_norms_ptr + kn_off, mask=mask_t, other=1.0)
                        logits += _seg_nibble_dot(
                            kp_ptr, kp_base, codebook_ptr, seg3_norm,
                            q_ptr, q_base, q_stride_d, m_mask, mask_t,
                            SEG3_START, SEG3_LEN, SEG3_LUT_OFF, BLOCK_M, BLOCK_T)
                    elif SEG3_BW <= 4 and SEG3_LEN > 0:
                        kn_off = (b64 * kn_stride_b + hkv64 * kn_stride_h
                                  + offs_t64 * kn_stride_t + SEG3_NORM_IDX * kn_stride_s)
                        seg_norm = tl.load(k_norms_ptr + kn_off, mask=mask_t, other=1.0)
                        NIB_OFF_3: tl.constexpr = SEG3_START // 2
                        for _k in tl.static_range(0, (SEG3_LEN + 1) // 2):
                            byte = tl.load(kp_ptr + kp_base + NIB_OFF_3 + _k,
                                           mask=mask_t, other=0).to(tl.int32)
                            lo_code = byte & 0xF
                            hi_code = (byte >> 4) & 0xF
                            if _k * 2 < SEG3_LEN:
                                cent = tl.load(codebook_ptr + SEG3_LUT_OFF + lo_code).to(tl.float32)
                                qv = tl.load(q_ptr + q_base + (SEG3_START + _k * 2) * q_stride_d,
                                             mask=m_mask, other=0.0).to(tl.float32)
                                logits += qv[:, None] * (cent * seg_norm)[None, :]
                            if _k * 2 + 1 < SEG3_LEN:
                                cent = tl.load(codebook_ptr + SEG3_LUT_OFF + hi_code).to(tl.float32)
                                qv = tl.load(q_ptr + q_base + (SEG3_START + _k * 2 + 1) * q_stride_d,
                                             mask=m_mask, other=0.0).to(tl.float32)
                                logits += qv[:, None] * (cent * seg_norm)[None, :]
                    elif SEG3_LEN >= MIN_SEG_FOR_DOT:  # bw > 4: uint8 tensor-core
                        kn_off = (b64 * kn_stride_b + hkv64 * kn_stride_h
                                  + offs_t64 * kn_stride_t + SEG3_NORM_IDX * kn_stride_s)
                        seg_norm = tl.load(k_norms_ptr + kn_off, mask=mask_t, other=1.0)
                        logits += _seg_uint8_dot(
                            kp_ptr, kp_base, codebook_ptr, seg_norm,
                            q_ptr, q_base, q_stride_d, m_mask, mask_t,
                            SEG3_START, SEG3_LEN, SEG3_LUT_OFF, SEG3_PACK_OFF,
                            BLOCK_M, BLOCK_T)
                    elif SEG3_LEN > 0:  # bw > 4: uint8 scalar
                        kn_off = (b64 * kn_stride_b + hkv64 * kn_stride_h
                                  + offs_t64 * kn_stride_t + SEG3_NORM_IDX * kn_stride_s)
                        seg_norm = tl.load(k_norms_ptr + kn_off, mask=mask_t, other=1.0)
                        for _k in tl.static_range(0, SEG3_LEN):
                            code = tl.load(kp_ptr + kp_base + SEG3_PACK_OFF + _k,
                                           mask=mask_t, other=0).to(tl.int32)
                            cent = tl.load(codebook_ptr + SEG3_LUT_OFF + code).to(tl.float32)
                            qv = tl.load(q_ptr + q_base + (SEG3_START + _k) * q_stride_d,
                                         mask=m_mask, other=0.0).to(tl.float32)
                            logits += qv[:, None] * (cent * seg_norm)[None, :]

                # ---- Segment 4 ----
                if N_SEGMENTS >= 5:
                    if SEG4_BW <= 4 and SEG4_LEN >= MIN_SEG_FOR_DOT:
                        kn_off = (b64 * kn_stride_b + hkv64 * kn_stride_h
                                  + offs_t64 * kn_stride_t + SEG4_NORM_IDX * kn_stride_s)
                        seg4_norm = tl.load(k_norms_ptr + kn_off, mask=mask_t, other=1.0)
                        logits += _seg_nibble_dot(
                            kp_ptr, kp_base, codebook_ptr, seg4_norm,
                            q_ptr, q_base, q_stride_d, m_mask, mask_t,
                            SEG4_START, SEG4_LEN, SEG4_LUT_OFF, BLOCK_M, BLOCK_T)
                    elif SEG4_BW <= 4 and SEG4_LEN > 0:
                        kn_off = (b64 * kn_stride_b + hkv64 * kn_stride_h
                                  + offs_t64 * kn_stride_t + SEG4_NORM_IDX * kn_stride_s)
                        seg_norm = tl.load(k_norms_ptr + kn_off, mask=mask_t, other=1.0)
                        NIB_OFF_4: tl.constexpr = SEG4_START // 2
                        for _k in tl.static_range(0, (SEG4_LEN + 1) // 2):
                            byte = tl.load(kp_ptr + kp_base + NIB_OFF_4 + _k,
                                           mask=mask_t, other=0).to(tl.int32)
                            lo_code = byte & 0xF
                            hi_code = (byte >> 4) & 0xF
                            if _k * 2 < SEG4_LEN:
                                cent = tl.load(codebook_ptr + SEG4_LUT_OFF + lo_code).to(tl.float32)
                                qv = tl.load(q_ptr + q_base + (SEG4_START + _k * 2) * q_stride_d,
                                             mask=m_mask, other=0.0).to(tl.float32)
                                logits += qv[:, None] * (cent * seg_norm)[None, :]
                            if _k * 2 + 1 < SEG4_LEN:
                                cent = tl.load(codebook_ptr + SEG4_LUT_OFF + hi_code).to(tl.float32)
                                qv = tl.load(q_ptr + q_base + (SEG4_START + _k * 2 + 1) * q_stride_d,
                                             mask=m_mask, other=0.0).to(tl.float32)
                                logits += qv[:, None] * (cent * seg_norm)[None, :]
                    elif SEG4_LEN >= MIN_SEG_FOR_DOT:  # bw > 4: uint8 tensor-core
                        kn_off = (b64 * kn_stride_b + hkv64 * kn_stride_h
                                  + offs_t64 * kn_stride_t + SEG4_NORM_IDX * kn_stride_s)
                        seg_norm = tl.load(k_norms_ptr + kn_off, mask=mask_t, other=1.0)
                        logits += _seg_uint8_dot(
                            kp_ptr, kp_base, codebook_ptr, seg_norm,
                            q_ptr, q_base, q_stride_d, m_mask, mask_t,
                            SEG4_START, SEG4_LEN, SEG4_LUT_OFF, SEG4_PACK_OFF,
                            BLOCK_M, BLOCK_T)
                    elif SEG4_LEN > 0:  # bw > 4: uint8 scalar
                        kn_off = (b64 * kn_stride_b + hkv64 * kn_stride_h
                                  + offs_t64 * kn_stride_t + SEG4_NORM_IDX * kn_stride_s)
                        seg_norm = tl.load(k_norms_ptr + kn_off, mask=mask_t, other=1.0)
                        for _k in tl.static_range(0, SEG4_LEN):
                            code = tl.load(kp_ptr + kp_base + SEG4_PACK_OFF + _k,
                                           mask=mask_t, other=0).to(tl.int32)
                            cent = tl.load(codebook_ptr + SEG4_LUT_OFF + code).to(tl.float32)
                            qv = tl.load(q_ptr + q_base + (SEG4_START + _k) * q_stride_d,
                                         mask=m_mask, other=0.0).to(tl.float32)
                            logits += qv[:, None] * (cent * seg_norm)[None, :]

                # ---- Segments 5-7: scalar only (nibble or uint8) ----
                if N_SEGMENTS >= 6 and SEG5_BW <= 4 and SEG5_LEN > 0:
                    kn_off = (b64 * kn_stride_b + hkv64 * kn_stride_h
                              + offs_t64 * kn_stride_t + SEG5_NORM_IDX * kn_stride_s)
                    seg_norm = tl.load(k_norms_ptr + kn_off, mask=mask_t, other=1.0)
                    NIB_OFF_5: tl.constexpr = SEG5_START // 2
                    for _k in tl.static_range(0, (SEG5_LEN + 1) // 2):
                        byte = tl.load(kp_ptr + kp_base + NIB_OFF_5 + _k,
                                       mask=mask_t, other=0).to(tl.int32)
                        lo_code = byte & 0xF
                        hi_code = (byte >> 4) & 0xF
                        if _k * 2 < SEG5_LEN:
                            cent = tl.load(codebook_ptr + SEG5_LUT_OFF + lo_code).to(tl.float32)
                            qv = tl.load(q_ptr + q_base + (SEG5_START + _k * 2) * q_stride_d,
                                         mask=m_mask, other=0.0).to(tl.float32)
                            logits += qv[:, None] * (cent * seg_norm)[None, :]
                        if _k * 2 + 1 < SEG5_LEN:
                            cent = tl.load(codebook_ptr + SEG5_LUT_OFF + hi_code).to(tl.float32)
                            qv = tl.load(q_ptr + q_base + (SEG5_START + _k * 2 + 1) * q_stride_d,
                                         mask=m_mask, other=0.0).to(tl.float32)
                            logits += qv[:, None] * (cent * seg_norm)[None, :]
                elif N_SEGMENTS >= 6 and SEG5_LEN > 0:  # bw > 4: uint8 scalar
                    kn_off = (b64 * kn_stride_b + hkv64 * kn_stride_h
                              + offs_t64 * kn_stride_t + SEG5_NORM_IDX * kn_stride_s)
                    seg_norm = tl.load(k_norms_ptr + kn_off, mask=mask_t, other=1.0)
                    for _k in tl.static_range(0, SEG5_LEN):
                        code = tl.load(kp_ptr + kp_base + SEG5_PACK_OFF + _k,
                                       mask=mask_t, other=0).to(tl.int32)
                        cent = tl.load(codebook_ptr + SEG5_LUT_OFF + code).to(tl.float32)
                        qv = tl.load(q_ptr + q_base + (SEG5_START + _k) * q_stride_d,
                                     mask=m_mask, other=0.0).to(tl.float32)
                        logits += qv[:, None] * (cent * seg_norm)[None, :]

                if N_SEGMENTS >= 7 and SEG6_BW <= 4 and SEG6_LEN > 0:
                    kn_off = (b64 * kn_stride_b + hkv64 * kn_stride_h
                              + offs_t64 * kn_stride_t + SEG6_NORM_IDX * kn_stride_s)
                    seg_norm = tl.load(k_norms_ptr + kn_off, mask=mask_t, other=1.0)
                    NIB_OFF_6: tl.constexpr = SEG6_START // 2
                    for _k in tl.static_range(0, (SEG6_LEN + 1) // 2):
                        byte = tl.load(kp_ptr + kp_base + NIB_OFF_6 + _k,
                                       mask=mask_t, other=0).to(tl.int32)
                        lo_code = byte & 0xF
                        hi_code = (byte >> 4) & 0xF
                        if _k * 2 < SEG6_LEN:
                            cent = tl.load(codebook_ptr + SEG6_LUT_OFF + lo_code).to(tl.float32)
                            qv = tl.load(q_ptr + q_base + (SEG6_START + _k * 2) * q_stride_d,
                                         mask=m_mask, other=0.0).to(tl.float32)
                            logits += qv[:, None] * (cent * seg_norm)[None, :]
                        if _k * 2 + 1 < SEG6_LEN:
                            cent = tl.load(codebook_ptr + SEG6_LUT_OFF + hi_code).to(tl.float32)
                            qv = tl.load(q_ptr + q_base + (SEG6_START + _k * 2 + 1) * q_stride_d,
                                         mask=m_mask, other=0.0).to(tl.float32)
                            logits += qv[:, None] * (cent * seg_norm)[None, :]
                elif N_SEGMENTS >= 7 and SEG6_LEN > 0:  # bw > 4: uint8 scalar
                    kn_off = (b64 * kn_stride_b + hkv64 * kn_stride_h
                              + offs_t64 * kn_stride_t + SEG6_NORM_IDX * kn_stride_s)
                    seg_norm = tl.load(k_norms_ptr + kn_off, mask=mask_t, other=1.0)
                    for _k in tl.static_range(0, SEG6_LEN):
                        code = tl.load(kp_ptr + kp_base + SEG6_PACK_OFF + _k,
                                       mask=mask_t, other=0).to(tl.int32)
                        cent = tl.load(codebook_ptr + SEG6_LUT_OFF + code).to(tl.float32)
                        qv = tl.load(q_ptr + q_base + (SEG6_START + _k) * q_stride_d,
                                     mask=m_mask, other=0.0).to(tl.float32)
                        logits += qv[:, None] * (cent * seg_norm)[None, :]

                if N_SEGMENTS >= 8 and SEG7_BW <= 4 and SEG7_LEN > 0:
                    kn_off = (b64 * kn_stride_b + hkv64 * kn_stride_h
                              + offs_t64 * kn_stride_t + SEG7_NORM_IDX * kn_stride_s)
                    seg_norm = tl.load(k_norms_ptr + kn_off, mask=mask_t, other=1.0)
                    NIB_OFF_7: tl.constexpr = SEG7_START // 2
                    for _k in tl.static_range(0, (SEG7_LEN + 1) // 2):
                        byte = tl.load(kp_ptr + kp_base + NIB_OFF_7 + _k,
                                       mask=mask_t, other=0).to(tl.int32)
                        lo_code = byte & 0xF
                        hi_code = (byte >> 4) & 0xF
                        if _k * 2 < SEG7_LEN:
                            cent = tl.load(codebook_ptr + SEG7_LUT_OFF + lo_code).to(tl.float32)
                            qv = tl.load(q_ptr + q_base + (SEG7_START + _k * 2) * q_stride_d,
                                         mask=m_mask, other=0.0).to(tl.float32)
                            logits += qv[:, None] * (cent * seg_norm)[None, :]
                        if _k * 2 + 1 < SEG7_LEN:
                            cent = tl.load(codebook_ptr + SEG7_LUT_OFF + hi_code).to(tl.float32)
                            qv = tl.load(q_ptr + q_base + (SEG7_START + _k * 2 + 1) * q_stride_d,
                                         mask=m_mask, other=0.0).to(tl.float32)
                            logits += qv[:, None] * (cent * seg_norm)[None, :]
                elif N_SEGMENTS >= 8 and SEG7_LEN > 0:  # bw > 4: uint8 scalar
                    kn_off = (b64 * kn_stride_b + hkv64 * kn_stride_h
                              + offs_t64 * kn_stride_t + SEG7_NORM_IDX * kn_stride_s)
                    seg_norm = tl.load(k_norms_ptr + kn_off, mask=mask_t, other=1.0)
                    for _k in tl.static_range(0, SEG7_LEN):
                        code = tl.load(kp_ptr + kp_base + SEG7_PACK_OFF + _k,
                                       mask=mask_t, other=0).to(tl.int32)
                        cent = tl.load(codebook_ptr + SEG7_LUT_OFF + code).to(tl.float32)
                        qv = tl.load(q_ptr + q_base + (SEG7_START + _k) * q_stride_d,
                                     mask=m_mask, other=0.0).to(tl.float32)
                        logits += qv[:, None] * (cent * seg_norm)[None, :]

            else:
                # ============================================================
                # Original path: bit-packed per-segment decode (scalar QK dot)
                # ============================================================

                # ---- Segment 0 ----
                if SEG0_LEN > 0:
                    logits = _seg_partial_dot(
                        q_ptr, q_base, q_stride_d, m_mask,
                        kp_ptr, kp_base, codebook_ptr, k_norms_ptr,
                        b64, hkv64, offs_t64, mask_t,
                        kn_stride_b, kn_stride_h, kn_stride_t, kn_stride_s,
                        logits, byte_lut_ptr, lut_head_base,
                        SEG0_BW, SEG0_START, SEG0_LEN, SEG0_PACK_OFF,
                        SEG0_LUT_OFF, SEG0_NORM_IDX, SEG0_1BIT_GOFF,
                        BLOCK_M, BLOCK_T, ONEBIT_FAST, K_PAIRED_2B, K_PAIRED_4B, K_PAIRED_8B, K_PAIRED_1B,
                    )

                # ---- Segment 1 ----
                if N_SEGMENTS >= 2:
                    if SEG1_LEN > 0:
                        logits = _seg_partial_dot(
                            q_ptr, q_base, q_stride_d, m_mask,
                            kp_ptr, kp_base, codebook_ptr, k_norms_ptr,
                            b64, hkv64, offs_t64, mask_t,
                            kn_stride_b, kn_stride_h, kn_stride_t, kn_stride_s,
                            logits, byte_lut_ptr, lut_head_base,
                            SEG1_BW, SEG1_START, SEG1_LEN, SEG1_PACK_OFF,
                            SEG1_LUT_OFF, SEG1_NORM_IDX, SEG1_1BIT_GOFF,
                            BLOCK_M, BLOCK_T, ONEBIT_FAST, K_PAIRED_2B, K_PAIRED_4B, K_PAIRED_8B, K_PAIRED_1B,
                        )

                # ---- Segment 2 ----
                if N_SEGMENTS >= 3:
                    if SEG2_LEN > 0:
                        logits = _seg_partial_dot(
                            q_ptr, q_base, q_stride_d, m_mask,
                            kp_ptr, kp_base, codebook_ptr, k_norms_ptr,
                            b64, hkv64, offs_t64, mask_t,
                            kn_stride_b, kn_stride_h, kn_stride_t, kn_stride_s,
                            logits, byte_lut_ptr, lut_head_base,
                            SEG2_BW, SEG2_START, SEG2_LEN, SEG2_PACK_OFF,
                            SEG2_LUT_OFF, SEG2_NORM_IDX, SEG2_1BIT_GOFF,
                            BLOCK_M, BLOCK_T, ONEBIT_FAST, K_PAIRED_2B, K_PAIRED_4B, K_PAIRED_8B, K_PAIRED_1B,
                        )

                # ---- Segment 3 ----
                if N_SEGMENTS >= 4:
                    if SEG3_LEN > 0:
                        logits = _seg_partial_dot(
                            q_ptr, q_base, q_stride_d, m_mask,
                            kp_ptr, kp_base, codebook_ptr, k_norms_ptr,
                            b64, hkv64, offs_t64, mask_t,
                            kn_stride_b, kn_stride_h, kn_stride_t, kn_stride_s,
                            logits, byte_lut_ptr, lut_head_base,
                            SEG3_BW, SEG3_START, SEG3_LEN, SEG3_PACK_OFF,
                            SEG3_LUT_OFF, SEG3_NORM_IDX, SEG3_1BIT_GOFF,
                            BLOCK_M, BLOCK_T, ONEBIT_FAST, K_PAIRED_2B, K_PAIRED_4B, K_PAIRED_8B, K_PAIRED_1B,
                        )

                # ---- Segment 4 ----
                if N_SEGMENTS >= 5:
                    if SEG4_LEN > 0:
                        logits = _seg_partial_dot(
                            q_ptr, q_base, q_stride_d, m_mask,
                            kp_ptr, kp_base, codebook_ptr, k_norms_ptr,
                            b64, hkv64, offs_t64, mask_t,
                            kn_stride_b, kn_stride_h, kn_stride_t, kn_stride_s,
                            logits, byte_lut_ptr, lut_head_base,
                            SEG4_BW, SEG4_START, SEG4_LEN, SEG4_PACK_OFF,
                            SEG4_LUT_OFF, SEG4_NORM_IDX, SEG4_1BIT_GOFF,
                            BLOCK_M, BLOCK_T, ONEBIT_FAST, K_PAIRED_2B, K_PAIRED_4B, K_PAIRED_8B, K_PAIRED_1B,
                        )

                # ---- Segment 5 ----
                if N_SEGMENTS >= 6:
                    if SEG5_LEN > 0:
                        logits = _seg_partial_dot(
                            q_ptr, q_base, q_stride_d, m_mask,
                            kp_ptr, kp_base, codebook_ptr, k_norms_ptr,
                            b64, hkv64, offs_t64, mask_t,
                            kn_stride_b, kn_stride_h, kn_stride_t, kn_stride_s,
                            logits, byte_lut_ptr, lut_head_base,
                            SEG5_BW, SEG5_START, SEG5_LEN, SEG5_PACK_OFF,
                            SEG5_LUT_OFF, SEG5_NORM_IDX, SEG5_1BIT_GOFF,
                            BLOCK_M, BLOCK_T, ONEBIT_FAST, K_PAIRED_2B, K_PAIRED_4B, K_PAIRED_8B, K_PAIRED_1B,
                        )

                # ---- Segment 6 ----
                if N_SEGMENTS >= 7:
                    if SEG6_LEN > 0:
                        logits = _seg_partial_dot(
                            q_ptr, q_base, q_stride_d, m_mask,
                            kp_ptr, kp_base, codebook_ptr, k_norms_ptr,
                            b64, hkv64, offs_t64, mask_t,
                            kn_stride_b, kn_stride_h, kn_stride_t, kn_stride_s,
                            logits, byte_lut_ptr, lut_head_base,
                            SEG6_BW, SEG6_START, SEG6_LEN, SEG6_PACK_OFF,
                            SEG6_LUT_OFF, SEG6_NORM_IDX, SEG6_1BIT_GOFF,
                            BLOCK_M, BLOCK_T, ONEBIT_FAST, K_PAIRED_2B, K_PAIRED_4B, K_PAIRED_8B, K_PAIRED_1B,
                        )

                # ---- Segment 7 ----
                if N_SEGMENTS >= 8:
                    if SEG7_LEN > 0:
                        logits = _seg_partial_dot(
                            q_ptr, q_base, q_stride_d, m_mask,
                            kp_ptr, kp_base, codebook_ptr, k_norms_ptr,
                            b64, hkv64, offs_t64, mask_t,
                            kn_stride_b, kn_stride_h, kn_stride_t, kn_stride_s,
                            logits, byte_lut_ptr, lut_head_base,
                            SEG7_BW, SEG7_START, SEG7_LEN, SEG7_PACK_OFF,
                            SEG7_LUT_OFF, SEG7_NORM_IDX, SEG7_1BIT_GOFF,
                            BLOCK_M, BLOCK_T, ONEBIT_FAST, K_PAIRED_2B, K_PAIRED_4B, K_PAIRED_8B, K_PAIRED_1B,
                        )

            logits = logits * SCALE
            logits = tl.where(mask_t[None, :], logits, -float('inf'))

            # ---- Online softmax ----
            m_new = tl.maximum(m_i, tl.max(logits, axis=1))
            alpha = tl.exp(m_i - m_new)
            p = tl.exp(logits - m_new[:, None])
            l_new = alpha * l_i + tl.sum(p, axis=1)
            acc = acc * alpha[:, None]
            m_i = m_new
            l_i = l_new

            # ---- V accumulation ----
            do_v = True
            if V_SKIP_THRESH > 0.0:
                do_v = tl.max(p) >= V_SKIP_THRESH

            if do_v:
                if V_FP16:
                    # Direct fp16 V load — pre-decompressed, no unpack/codebook/norms
                    v_fp_offset = (b64 * vc_stride_b + hkv64 * vc_stride_h
                                   + offs_t64[:, None] * vc_stride_t
                                   + offs_d[None, :] * vc_stride_d)
                    v_tile = tl.load(v_codes_ptr + v_fp_offset,
                                     mask=mask_t[:, None],
                                     other=0.0).to(tl.float16)
                elif V_NIBBLE:
                    # Fast nibble-4 V decode: load (BLOCK_T, D//2) bytes,
                    # split into lo/hi nibbles → (BLOCK_T, D) codes.
                    # Layout: byte[i] = (code[2i+1] << 4) | code[2i]
                    # This avoids the slow general _load_v_codes scatter path.
                    HALF_D: tl.constexpr = D // 2
                    offs_half = tl.arange(0, HALF_D)
                    vn_base = (b64 * vc_stride_b + hkv64 * vc_stride_h
                               + offs_t64 * vc_stride_t)
                    packed_bytes = tl.load(
                        v_codes_ptr + vn_base[:, None] + offs_half[None, :].to(tl.int64),
                        mask=mask_t[:, None], other=0).to(tl.int32)
                    v_lo = packed_bytes & 0xF          # even dims (BLOCK_T, D//2)
                    v_hi = (packed_bytes >> 4) & 0xF   # odd dims  (BLOCK_T, D//2)
                    # Codebook lookup → (BLOCK_T, D//2) fp16 each
                    c_lo = tl.load(v_lut_ptr + v_lo).to(tl.float16)
                    c_hi = tl.load(v_lut_ptr + v_hi).to(tl.float16)
                    # Per-token norm (uniform V: one norm per token)
                    v_n = tl.load(
                        v_norms_ptr + b64 * vn_stride_b + hkv64 * vn_stride_h
                        + offs_t64 * vn_stride_t,
                        mask=mask_t, other=1.0).to(tl.float16)
                    v_even = c_lo * v_n[:, None]
                    v_odd  = c_hi * v_n[:, None]
                    # Interleave even/odd → full (BLOCK_T, D) tile
                    v_tile = tl.interleave(v_even, v_odd)
                elif V_PAIRED_2B:
                    # 2-bit paired-decode fast path: each byte packs 4 codes
                    # (2 bits each). Layout matches _compress_v_pack_kernel:
                    #   byte = c0 | (c1 << 2) | (c2 << 4) | (c3 << 6)
                    # where cX occupies dim (4*g + X) within a dim-group g.
                    #
                    # IMPLEMENTATION (3D gather + reshape):
                    # Build a (BLOCK_T, D//4, 4) index tensor where the
                    # last axis enumerates the 4 sub-codes within a byte
                    # via shift+mask. ONE tl.load gathers all centroids
                    # in a single op; tl.reshape collapses the (D//4, 4)
                    # axes to (D) at zero runtime cost (Triton view).
                    # Replaces the previous 4-gather + 3-interleave version
                    # so the inner kernel emits 4× fewer LUT gather ops.
                    QUARTER_D: tl.constexpr = D // 4
                    offs_q = tl.arange(0, QUARTER_D)
                    vn_base = (b64 * vc_stride_b + hkv64 * vc_stride_h
                               + offs_t64 * vc_stride_t)
                    packed_bytes = tl.load(
                        v_codes_ptr + vn_base[:, None] + offs_q[None, :].to(tl.int64),
                        mask=mask_t[:, None], other=0).to(tl.int32)
                    inner = tl.arange(0, 4)
                    codes_3d = (packed_bytes[:, :, None]
                                >> (inner[None, None, :] * 2)) & 0x3
                    cents_3d = tl.load(v_lut_ptr + codes_3d,
                                       mask=mask_t[:, None, None],
                                       other=0.0).to(tl.float16)
                    v_tile = tl.reshape(cents_3d, (BLOCK_T, D))
                    v_n = tl.load(
                        v_norms_ptr + b64 * vn_stride_b + hkv64 * vn_stride_h
                        + offs_t64 * vn_stride_t,
                        mask=mask_t, other=1.0).to(tl.float16)
                    v_tile = v_tile * v_n[:, None]
                else:
                    v_codes = _load_v_codes(
                        v_codes_ptr, v_unpack_ptr,
                        b64, hkv64, offs_t64, offs_d, mask_t,
                        vc_stride_b, vc_stride_h, vc_stride_t, vc_stride_d,
                        vu_stride, V_BITS)
                    v_centroids = tl.load(v_lut_ptr + v_codes).to(tl.float16)

                    v_norms = _unpack_norms_mixed(
                        v_norms_ptr, v_norm_scale_ptr,
                        b64, hkv64, offs_t64, v_block_ids, mask_t,
                        vn_stride_b, vn_stride_h, vn_stride_t, vn_stride_blk,
                        vns_stride_b, vns_stride_h, V_NORM_BITS).to(tl.float16)
                    v_tile = v_centroids * v_norms

                p_fp16 = p.to(tl.float16)
                acc = acc + tl.dot(p_fp16, v_tile)

        partial_out = acc / l_i[:, None]

        # V un-rotation: rotate output back from PolarQuant's rotated space
        # Linearity allows per-split rotation: (Σ wᵢ·outᵢ)@R = Σ wᵢ·(outᵢ@R)
        if V_UNROT == 1:
            rot_cols = tl.arange(0, D)
            rot_block = tl.load(
                v_rot_ptr + offs_d[:, None] * D + rot_cols[None, :],
            ).to(tl.float32)                       # (D, D)
            partial_out = tl.dot(partial_out, rot_block, allow_tf32=True)

        partial_lse = m_i + tl.log(l_i)

        po_offset = (b64 * po_stride_b + q_row[:, None] * po_stride_h
                     + chunk_id.to(tl.int64) * po_stride_c
                     + offs_d[None, :] * po_stride_d)
        tl.store(partial_out_ptr + po_offset,
                 partial_out.to(tl.float16), mask=m_mask[:, None])

        pl_offset = (b64 * pl_stride_b + q_row * pl_stride_h
                     + chunk_id.to(tl.int64) * pl_stride_c)
        tl.store(partial_lse_ptr + pl_offset, partial_lse, mask=m_mask)

    # -----------------------------------------------------------------------
    # Decode-kernel helper: per-segment partial dot
    # -----------------------------------------------------------------------

    @triton.jit
    def _seg_partial_dot(
        q_ptr,              # q data pointer
        q_base,             # (BLOCK_M,) int64 per-head base offsets
        q_stride_d,         # stride along dim D
        m_mask,             # (BLOCK_M,) query head mask
        kp_ptr,             # packed K pointer
        kp_base,            # (BLOCK_T,) int64 base offsets for T-block
        codebook_ptr,       # flat codebook
        k_norms_ptr,        # norms pointer
        b64, hkv64, offs_t64, mask_t,
        kn_stride_b, kn_stride_h, kn_stride_t, kn_stride_s,
        logits,             # (BLOCK_M, BLOCK_T) accumulator, modified in-place
        byte_lut_ptr,       # byte-LUT pointer (used when ONEBIT_FAST==2)
        lut_head_base,      # (BLOCK_M,) int64 per-head LUT base offsets
        # Segment constexprs
        SEG_BW: tl.constexpr,
        SEG_START: tl.constexpr,
        SEG_LEN: tl.constexpr,
        SEG_PACK_OFF: tl.constexpr,
        SEG_LUT_OFF: tl.constexpr,
        SEG_NORM_IDX: tl.constexpr,
        SEG_1BIT_GOFF: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_T: tl.constexpr,
        ONEBIT_FAST: tl.constexpr = 0,
        K_PAIRED_2B: tl.constexpr = 0,
        K_PAIRED_4B: tl.constexpr = 0,
        K_PAIRED_8B: tl.constexpr = 0,
        K_PAIRED_1B: tl.constexpr = 0,
    ):
        """Process one segment: unpack + codebook gather + partial QK dot.

        Loads packed bytes group-by-group, unpacks, gathers centroids,
        and accumulates q[d] * centroid[d] * norm into logits.
        Uses pointer-based q loading (tl.load) to avoid Triton tensor
        indexing limitations with constexpr arithmetic.

        ONEBIT_FAST modes for 1-bit segments:
          0 = standard unpack→codebook gather (baseline)
          1 = (μ,h) sign-dot (no codebook loads, but ~same speed)
          2 = byte-LUT: precomputed 256-entry table per 8-dim group
          3 = nibble-LUT: two 16-entry tables per 8-dim group
          4 = masked_sum: accumulate q where bit=1, final multiply only

        K_PAIRED_2B:
          0 = default scalar loop for 2-bit K segments (backward-compat).
          1 = paired-LUT tensor-core fast path: load D/4 bytes, extract
              4 × 2-bit codes per byte, 3-stage tl.interleave into
              (BLOCK_T, SEG_PAD) fp16 K tile, then tl.dot(Q_seg, K^T).
              Only applied when SEG_BW==2 and SEG_LEN >= 16.

        K_PAIRED_4B:
          0 = default scalar loop for 4-bit K segments (backward-compat).
          1 = paired nibble-4 tensor-core fast path: load D/2 bytes,
              extract 2 × 4-bit codes per byte (low/high nibble), one
              tl.interleave into (BLOCK_T, SEG_PAD) fp16 K tile, then
              tl.dot(Q_seg, K^T). Only applied when SEG_BW==4 and
              SEG_LEN >= 16. Typically the biggest K-decode speed-up
              in HA since the 4-bit segment is usually the largest.

        Returns the updated logits (Triton SSA: in-place += doesn't
        propagate to caller, so we must return the new value).
        """
        # Load segment norm: (BLOCK_T,) fp32
        kn_off = (b64 * kn_stride_b + hkv64 * kn_stride_h
                  + offs_t64 * kn_stride_t
                  + SEG_NORM_IDX * kn_stride_s)
        seg_norm = tl.load(k_norms_ptr + kn_off,
                           mask=mask_t, other=1.0).to(tl.float32)

        # Dispatch on bit width
        if SEG_BW >= 5 and K_PAIRED_8B == 1 and SEG_LEN >= 16:
            # 8-bit (raw uint8) K decode + tensor-core QK dot.
            # Mirrors _seg_uint8_dot but uses SEG_PACK_OFF (default
            # K_NIBBLE=False packing layout) instead of the global
            # BYTE_OFF that the K_NIBBLE=True path requires.
            #
            # SEG_PAD ≥ 32 (skip K=16 minimum MMA tile): defensive fix.
            # Production saw a
            # tl.interleave + tl.dot interaction bug at HALF_PAD=8 with
            # RMSE 18 → +1.747 PPL on Llama-3.1-8B. This 8-bit path has
            # no tl.interleave, but K=16 is the MMA boundary where the
            # bug manifests, so skip it for safety. Cost +0.51% latency.
            SEG_PAD_8B: tl.constexpr = (
                32 if SEG_LEN <= 32 else
                64 if SEG_LEN <= 64 else
                128
            )
            offs_d_8b = tl.arange(0, SEG_PAD_8B)
            codes_8b = tl.load(
                kp_ptr + kp_base[:, None] + SEG_PACK_OFF
                + offs_d_8b[None, :].to(tl.int64),
                mask=mask_t[:, None] & (offs_d_8b[None, :] < SEG_LEN),
                other=0,
            ).to(tl.int32)
            cents_8b = tl.load(
                codebook_ptr + SEG_LUT_OFF + codes_8b,
                mask=mask_t[:, None] & (offs_d_8b[None, :] < SEG_LEN),
                other=0.0,
            ).to(tl.float16)
            k_chunk_8b = cents_8b * seg_norm.to(tl.float16)[:, None]
            q_seg_8b = tl.load(
                q_ptr + q_base[:, None]
                + (SEG_START + offs_d_8b[None, :]) * q_stride_d,
                mask=m_mask[:, None] & (offs_d_8b[None, :] < SEG_LEN),
                other=0.0,
            ).to(tl.float16)
            logits += tl.dot(q_seg_8b, tl.trans(k_chunk_8b))

        elif SEG_BW >= 5:
            # Raw: 1 byte per code (scalar fallback for SEG_LEN < 16)
            for _k in tl.static_range(0, SEG_LEN):
                c = tl.load(kp_ptr + kp_base + SEG_PACK_OFF + _k,
                            mask=mask_t, other=0).to(tl.int32)
                cent = tl.load(codebook_ptr + SEG_LUT_OFF + c).to(tl.float32)
                kv = cent * seg_norm
                qv = tl.load(q_ptr + q_base + (SEG_START + _k) * q_stride_d,
                             mask=m_mask, other=0.0).to(tl.float32)
                logits += qv[:, None] * kv[None, :]

        elif SEG_BW == 1 and K_PAIRED_1B == 1 and SEG_LEN >= 16:
            # 1-bit K decode via 3D-gather + reshape tensor-core path.
            # Each byte stores 8 codes (1 bit each);
            # we extract them into a (BLOCK_T, BYTE_PAD, 8) tensor via
            # shift+mask, gather centroids in ONE tl.load, and reshape
            # to (BLOCK_T, SEG_PAD).  Same 3D-gather + tl.reshape
            # pattern as K_PAIRED_2B/4B — avoids tl.interleave entirely
            # (production found a register-layout bug at HALF_PAD=8 in
            # tl.interleave→tl.dot).
            #
            # SEG_PAD ≥ 32 (skip K=16 minimum MMA tile) defensive fix
            # mirroring the production fix.
            SEG_PAD_1B: tl.constexpr = (
                32 if SEG_LEN <= 32 else
                64 if SEG_LEN <= 64 else
                128
            )
            BYTE_PAD_1B: tl.constexpr = SEG_PAD_1B // 8

            offs_b_1b = tl.arange(0, BYTE_PAD_1B)
            byte_ok_1b = offs_b_1b[None, :] < (SEG_LEN + 7) // 8
            packed_1b = tl.load(
                kp_ptr + kp_base[:, None] + SEG_PACK_OFF
                + offs_b_1b[None, :].to(tl.int64),
                mask=mask_t[:, None] & byte_ok_1b,
                other=0,
            ).to(tl.int32)
            inner_1b = tl.arange(0, 8)
            codes_3d_1b = (packed_1b[:, :, None]
                           >> inner_1b[None, None, :]) & 0x1
            dim_ok_1b = (offs_b_1b[None, :, None] * 8
                         + inner_1b[None, None, :]) < SEG_LEN
            cents_3d_1b = tl.load(
                codebook_ptr + SEG_LUT_OFF + codes_3d_1b,
                mask=mask_t[:, None, None] & dim_ok_1b, other=0.0,
            ).to(tl.float16)
            k_chunk_1b = tl.reshape(cents_3d_1b, (BLOCK_T, SEG_PAD_1B))
            k_chunk_1b = k_chunk_1b * seg_norm.to(tl.float16)[:, None]

            offs_seg_1b = tl.arange(0, SEG_PAD_1B)
            q_seg_1b = tl.load(
                q_ptr + q_base[:, None]
                + (SEG_START + offs_seg_1b[None, :]) * q_stride_d,
                mask=m_mask[:, None] & (offs_seg_1b[None, :] < SEG_LEN),
                other=0.0,
            ).to(tl.float16)

            logits += tl.dot(q_seg_1b, tl.trans(k_chunk_1b))

        elif SEG_BW == 1:
            if ONEBIT_FAST == 4:
                # ---- 1-bit masked_sum fast path (Mode 4) ----
                # Accumulate masked_sum[m,t] = Σ_{d: code=1} q[m,d]
                # Then: score = norm * (c₀ * q_sum + δ * masked_sum)
                # Avoids per-dim centroid multiply; just conditional accumulate.
                c0_val = tl.load(codebook_ptr + SEG_LUT_OFF).to(tl.float32)
                c1_val = tl.load(codebook_ptr + SEG_LUT_OFF + 1).to(tl.float32)
                delta = c1_val - c0_val

                q_sum = tl.zeros([BLOCK_M], dtype=tl.float32)
                masked_sum = tl.zeros([BLOCK_M, BLOCK_T], dtype=tl.float32)

                for _g in tl.static_range(0, (SEG_LEN + 7) // 8):
                    byte = tl.load(kp_ptr + kp_base + SEG_PACK_OFF + _g,
                                   mask=mask_t, other=0).to(tl.int32)
                    for _p in tl.static_range(0, 8):
                        if _g * 8 + _p < SEG_LEN:
                            bit = (byte >> _p) & 1
                            qv = tl.load(q_ptr + q_base + (SEG_START + _g * 8 + _p) * q_stride_d,
                                         mask=m_mask, other=0.0).to(tl.float32)
                            q_sum += qv
                            masked_sum += tl.where(bit[None, :] != 0,
                                                   qv[:, None], 0.0)

                logits += seg_norm[None, :] * (c0_val * q_sum[:, None] + delta * masked_sum)
            elif ONEBIT_FAST == 3:
                # ---- 1-bit nibble-LUT fast path (Mode 3) ----
                # Two 16-entry LUTs per 8-dim group (lo/hi nibble).
                # Each 16-address gather fits in 1 L1 cache line (64B).
                # byte_lut layout: [B*H_q, total_1bit_nibbles, 16] fp32
                #   nibble 2g   = lo (dims d..d+3)
                #   nibble 2g+1 = hi (dims d+4..d+7)
                # Entry bakes in: c₀·Σq_nibble + δ·Σ_{bits=1} q[d+b]
                # Use tl.range (not tl.static_range) to avoid shared memory overflow
                # with large 1-bit segments (e.g. 128d → 16 groups).
                for _g in tl.range(0, (SEG_LEN + 7) // 8, num_stages=1):
                    byte_val = tl.load(kp_ptr + kp_base + SEG_PACK_OFF + _g,
                                       mask=mask_t, other=0).to(tl.int64)
                    lo_nib = byte_val & 0xF                  # [BLOCK_T]
                    hi_nib = (byte_val >> 4) & 0xF           # [BLOCK_T]
                    # Gather lo nibble: [BLOCK_M, BLOCK_T]
                    lo_off = (lut_head_base[:, None]
                              + (SEG_1BIT_GOFF + _g * 2) * 16
                              + lo_nib[None, :])
                    entry_lo = tl.load(byte_lut_ptr + lo_off,
                                       mask=m_mask[:, None] & mask_t[None, :],
                                       other=0.0).to(tl.float32)
                    # Gather hi nibble: [BLOCK_M, BLOCK_T]
                    hi_off = (lut_head_base[:, None]
                              + (SEG_1BIT_GOFF + _g * 2 + 1) * 16
                              + hi_nib[None, :])
                    entry_hi = tl.load(byte_lut_ptr + hi_off,
                                       mask=m_mask[:, None] & mask_t[None, :],
                                       other=0.0).to(tl.float32)
                    logits += seg_norm[None, :] * (entry_lo + entry_hi)
            elif ONEBIT_FAST == 2:
                # ---- 1-bit byte-LUT fast path (Mode 2) ----
                # 256-entry LUT per 8-dim group. One gather per byte.
                # Use tl.range (not tl.static_range) to avoid shared memory overflow.
                for _g in tl.range(0, (SEG_LEN + 7) // 8, num_stages=1):
                    byte_val = tl.load(kp_ptr + kp_base + SEG_PACK_OFF + _g,
                                       mask=mask_t, other=0).to(tl.int64)
                    lut_off = (lut_head_base[:, None]
                               + (SEG_1BIT_GOFF + _g) * 256
                               + byte_val[None, :])
                    entry = tl.load(byte_lut_ptr + lut_off,
                                    mask=m_mask[:, None] & mask_t[None, :],
                                    other=0.0).to(tl.float32)
                    logits += seg_norm[None, :] * entry
            elif ONEBIT_FAST == 1:
                # ---- 1-bit fast path: (μ, h) sign convention ----
                # Load c₀, c₁ ONCE (2 loads vs SEG_LEN × BLOCK_T codebook gathers)
                # Then compute μ = (c₀+c₁)/2, h = (c₁-c₀)/2 inline.
                # Reconstruction: k̂[d] = (μ + sign[d] × h) × norm
                #   where sign = 2 × code - 1 ∈ {-1, +1}
                c0_val = tl.load(codebook_ptr + SEG_LUT_OFF).to(tl.float32)
                c1_val = tl.load(codebook_ptr + SEG_LUT_OFF + 1).to(tl.float32)
                mu_val = (c0_val + c1_val) * 0.5
                h_val = (c1_val - c0_val) * 0.5

                # Accumulate q_sum (BLOCK_M,) and sign_dot (BLOCK_M, BLOCK_T)
                # score[m,t] = norm[t] × (μ × q_sum[m] + h × sign_dot[m,t])
                q_sum = tl.zeros([BLOCK_M], dtype=tl.float32)
                sign_dot = tl.zeros([BLOCK_M, BLOCK_T], dtype=tl.float32)

                for _g in tl.static_range(0, (SEG_LEN + 7) // 8):
                    byte = tl.load(kp_ptr + kp_base + SEG_PACK_OFF + _g,
                                   mask=mask_t, other=0).to(tl.int32)
                    for _p in tl.static_range(0, 8):
                        if _g * 8 + _p < SEG_LEN:
                            bit = (byte >> _p) & 1
                            sign_f = (bit * 2 - 1).to(tl.float32)
                            qv = tl.load(q_ptr + q_base + (SEG_START + _g * 8 + _p) * q_stride_d,
                                         mask=m_mask, other=0.0).to(tl.float32)
                            q_sum += qv
                            sign_dot += qv[:, None] * sign_f[None, :]

                logits += seg_norm[None, :] * (mu_val * q_sum[:, None] + h_val * sign_dot)
            else:
                # ---- 1-bit standard path: unpack → codebook gather ----
                for _g in tl.static_range(0, (SEG_LEN + 7) // 8):
                    byte = tl.load(kp_ptr + kp_base + SEG_PACK_OFF + _g,
                                   mask=mask_t, other=0).to(tl.int32)
                    for _p in tl.static_range(0, 8):
                        if _g * 8 + _p < SEG_LEN:
                            c = (byte >> _p) & 1
                            cent = tl.load(codebook_ptr + SEG_LUT_OFF + c).to(tl.float32)
                            kv = cent * seg_norm
                            qv = tl.load(q_ptr + q_base + (SEG_START + _g * 8 + _p) * q_stride_d,
                                         mask=m_mask, other=0.0).to(tl.float32)
                            logits += qv[:, None] * kv[None, :]

        elif SEG_BW == 2 and K_PAIRED_2B == 1 and SEG_LEN >= 16:
            # Paired-LUT tensor-core 2-bit K decode (3D-gather + reshape):
            # one byte = 4 × 2-bit codes; build a (BLOCK_T, QUARTER_PAD, 4)
            # index tensor and gather all centroids in a single tl.load,
            # then tl.reshape to (BLOCK_T, SEG_PAD_2B) at zero cost.
            # Replaces the older 4-gather + 3-tl.interleave version with a
            # single gather + a no-op view.
            #
            # SEG_PAD ≥ 32 (skip K=16 MMA tile): defensive mirror of the
            # production fix. The
            # production bug was specifically tl.interleave→tl.dot
            # register-layout incompatibility at HALF_PAD=8 (+1.747 PPL
            # on Llama-3.1-8B). This path uses tl.reshape (typically a
            # view) and so likely sidesteps the bug, but skipping K=16
            # entirely is defense-in-depth at +0.51% latency cost.
            SEG_PAD_2B: tl.constexpr = (
                32 if SEG_LEN <= 32 else
                64 if SEG_LEN <= 64 else
                128
            )
            QUARTER_PAD_2B: tl.constexpr = SEG_PAD_2B // 4

            offs_q_2b = tl.arange(0, QUARTER_PAD_2B)
            byte_ok_2b = offs_q_2b[None, :] < (SEG_LEN + 3) // 4
            packed_2b = tl.load(
                kp_ptr + kp_base[:, None] + SEG_PACK_OFF
                + offs_q_2b[None, :].to(tl.int64),
                mask=mask_t[:, None] & byte_ok_2b,
                other=0,
            ).to(tl.int32)
            inner_2b = tl.arange(0, 4)
            codes_3d_2b = (packed_2b[:, :, None]
                           >> (inner_2b[None, None, :] * 2)) & 0x3
            dim_ok_2b = (offs_q_2b[None, :, None] * 4
                         + inner_2b[None, None, :]) < SEG_LEN
            cents_3d_2b = tl.load(
                codebook_ptr + SEG_LUT_OFF + codes_3d_2b,
                mask=mask_t[:, None, None] & dim_ok_2b, other=0.0,
            ).to(tl.float16)
            k_chunk_2b = tl.reshape(cents_3d_2b, (BLOCK_T, SEG_PAD_2B))
            k_chunk_2b = k_chunk_2b * seg_norm.to(tl.float16)[:, None]

            offs_seg_2b = tl.arange(0, SEG_PAD_2B)
            q_seg_2b = tl.load(
                q_ptr + q_base[:, None] + (SEG_START + offs_seg_2b[None, :]) * q_stride_d,
                mask=m_mask[:, None] & (offs_seg_2b[None, :] < SEG_LEN),
                other=0.0,
            ).to(tl.float16)

            logits += tl.dot(q_seg_2b, tl.trans(k_chunk_2b))

        elif SEG_BW == 2:
            for _g in tl.static_range(0, (SEG_LEN + 3) // 4):
                byte = tl.load(kp_ptr + kp_base + SEG_PACK_OFF + _g,
                               mask=mask_t, other=0).to(tl.int32)
                for _p in tl.static_range(0, 4):
                    if _g * 4 + _p < SEG_LEN:
                        c = (byte >> (_p * 2)) & 0x3
                        cent = tl.load(codebook_ptr + SEG_LUT_OFF + c).to(tl.float32)
                        kv = cent * seg_norm
                        qv = tl.load(q_ptr + q_base + (SEG_START + _g * 4 + _p) * q_stride_d,
                                     mask=m_mask, other=0.0).to(tl.float32)
                        logits += qv[:, None] * kv[None, :]

        elif SEG_BW == 3:
            for _g in tl.static_range(0, (SEG_LEN + 7) // 8):
                b0 = tl.load(kp_ptr + kp_base + SEG_PACK_OFF + _g * 3,
                             mask=mask_t, other=0).to(tl.int32)
                b1 = tl.load(kp_ptr + kp_base + SEG_PACK_OFF + _g * 3 + 1,
                             mask=mask_t, other=0).to(tl.int32)
                b2 = tl.load(kp_ptr + kp_base + SEG_PACK_OFF + _g * 3 + 2,
                             mask=mask_t, other=0).to(tl.int32)
                # 8 codes
                codes_3b_0 = b0 & 0x7
                codes_3b_1 = (b0 >> 3) & 0x7
                codes_3b_2 = ((b0 >> 6) | (b1 << 2)) & 0x7
                codes_3b_3 = (b1 >> 1) & 0x7
                codes_3b_4 = (b1 >> 4) & 0x7
                codes_3b_5 = ((b1 >> 7) | (b2 << 1)) & 0x7
                codes_3b_6 = (b2 >> 2) & 0x7
                codes_3b_7 = (b2 >> 5) & 0x7
                # Unrolled per-code dot
                if _g * 8 + 0 < SEG_LEN:
                    cent = tl.load(codebook_ptr + SEG_LUT_OFF + codes_3b_0).to(tl.float32)
                    qv = tl.load(q_ptr + q_base + (SEG_START + _g * 8 + 0) * q_stride_d,
                                 mask=m_mask, other=0.0).to(tl.float32)
                    logits += qv[:, None] * (cent * seg_norm)[None, :]
                if _g * 8 + 1 < SEG_LEN:
                    cent = tl.load(codebook_ptr + SEG_LUT_OFF + codes_3b_1).to(tl.float32)
                    qv = tl.load(q_ptr + q_base + (SEG_START + _g * 8 + 1) * q_stride_d,
                                 mask=m_mask, other=0.0).to(tl.float32)
                    logits += qv[:, None] * (cent * seg_norm)[None, :]
                if _g * 8 + 2 < SEG_LEN:
                    cent = tl.load(codebook_ptr + SEG_LUT_OFF + codes_3b_2).to(tl.float32)
                    qv = tl.load(q_ptr + q_base + (SEG_START + _g * 8 + 2) * q_stride_d,
                                 mask=m_mask, other=0.0).to(tl.float32)
                    logits += qv[:, None] * (cent * seg_norm)[None, :]
                if _g * 8 + 3 < SEG_LEN:
                    cent = tl.load(codebook_ptr + SEG_LUT_OFF + codes_3b_3).to(tl.float32)
                    qv = tl.load(q_ptr + q_base + (SEG_START + _g * 8 + 3) * q_stride_d,
                                 mask=m_mask, other=0.0).to(tl.float32)
                    logits += qv[:, None] * (cent * seg_norm)[None, :]
                if _g * 8 + 4 < SEG_LEN:
                    cent = tl.load(codebook_ptr + SEG_LUT_OFF + codes_3b_4).to(tl.float32)
                    qv = tl.load(q_ptr + q_base + (SEG_START + _g * 8 + 4) * q_stride_d,
                                 mask=m_mask, other=0.0).to(tl.float32)
                    logits += qv[:, None] * (cent * seg_norm)[None, :]
                if _g * 8 + 5 < SEG_LEN:
                    cent = tl.load(codebook_ptr + SEG_LUT_OFF + codes_3b_5).to(tl.float32)
                    qv = tl.load(q_ptr + q_base + (SEG_START + _g * 8 + 5) * q_stride_d,
                                 mask=m_mask, other=0.0).to(tl.float32)
                    logits += qv[:, None] * (cent * seg_norm)[None, :]
                if _g * 8 + 6 < SEG_LEN:
                    cent = tl.load(codebook_ptr + SEG_LUT_OFF + codes_3b_6).to(tl.float32)
                    qv = tl.load(q_ptr + q_base + (SEG_START + _g * 8 + 6) * q_stride_d,
                                 mask=m_mask, other=0.0).to(tl.float32)
                    logits += qv[:, None] * (cent * seg_norm)[None, :]
                if _g * 8 + 7 < SEG_LEN:
                    cent = tl.load(codebook_ptr + SEG_LUT_OFF + codes_3b_7).to(tl.float32)
                    qv = tl.load(q_ptr + q_base + (SEG_START + _g * 8 + 7) * q_stride_d,
                                 mask=m_mask, other=0.0).to(tl.float32)
                    logits += qv[:, None] * (cent * seg_norm)[None, :]

        elif SEG_BW == 4 and K_PAIRED_4B == 1 and SEG_LEN >= 16:
            # Paired nibble-4 tensor-core 4-bit K decode (3D-gather + reshape):
            # one byte = 2 × 4-bit codes; build a (BLOCK_T, HALF_PAD, 2)
            # index tensor and gather all centroids in a single tl.load,
            # then tl.reshape to (BLOCK_T, SEG_PAD_4B) at zero cost.
            # Replaces the previous 2-gather + 1-tl.interleave version.
            #
            # SEG_PAD ≥ 32 (skip K=16 MMA tile): defensive mirror of the
            # production fix. The
            # production bug at HALF_PAD=8 (RMSE 18 → +1.747 PPL on
            # Llama-3.1-8B) was specifically the tl.interleave →
            # tl.dot register-layout boundary. Our 3D-gather path uses
            # tl.reshape so the bug likely doesn't manifest, but K=16 is
            # the same MMA tile the production diagnostic flagged, so
            # we skip it defensively at +0.51% latency cost.
            SEG_PAD_4B: tl.constexpr = (
                32 if SEG_LEN <= 32 else
                64 if SEG_LEN <= 64 else
                128
            )
            HALF_PAD_4B: tl.constexpr = SEG_PAD_4B // 2

            offs_h_4b = tl.arange(0, HALF_PAD_4B)
            byte_ok_4b = offs_h_4b[None, :] < (SEG_LEN + 1) // 2
            packed_4b = tl.load(
                kp_ptr + kp_base[:, None] + SEG_PACK_OFF
                + offs_h_4b[None, :].to(tl.int64),
                mask=mask_t[:, None] & byte_ok_4b,
                other=0,
            ).to(tl.int32)
            inner_4b = tl.arange(0, 2)
            codes_3d_4b = (packed_4b[:, :, None]
                           >> (inner_4b[None, None, :] * 4)) & 0xF
            dim_ok_4b = (offs_h_4b[None, :, None] * 2
                         + inner_4b[None, None, :]) < SEG_LEN
            cents_3d_4b = tl.load(
                codebook_ptr + SEG_LUT_OFF + codes_3d_4b,
                mask=mask_t[:, None, None] & dim_ok_4b, other=0.0,
            ).to(tl.float16)
            k_chunk_4b = tl.reshape(cents_3d_4b, (BLOCK_T, SEG_PAD_4B))
            k_chunk_4b = k_chunk_4b * seg_norm.to(tl.float16)[:, None]

            offs_seg_4b = tl.arange(0, SEG_PAD_4B)
            q_seg_4b = tl.load(
                q_ptr + q_base[:, None] + (SEG_START + offs_seg_4b[None, :]) * q_stride_d,
                mask=m_mask[:, None] & (offs_seg_4b[None, :] < SEG_LEN),
                other=0.0,
            ).to(tl.float16)

            logits += tl.dot(q_seg_4b, tl.trans(k_chunk_4b))

        elif SEG_BW == 4:
            for _g in tl.static_range(0, (SEG_LEN + 1) // 2):
                byte = tl.load(kp_ptr + kp_base + SEG_PACK_OFF + _g,
                               mask=mask_t, other=0).to(tl.int32)
                if _g * 2 < SEG_LEN:
                    c = byte & 0xF
                    cent = tl.load(codebook_ptr + SEG_LUT_OFF + c).to(tl.float32)
                    qv = tl.load(q_ptr + q_base + (SEG_START + _g * 2) * q_stride_d,
                                 mask=m_mask, other=0.0).to(tl.float32)
                    logits += qv[:, None] * (cent * seg_norm)[None, :]
                if _g * 2 + 1 < SEG_LEN:
                    c = (byte >> 4) & 0xF
                    cent = tl.load(codebook_ptr + SEG_LUT_OFF + c).to(tl.float32)
                    qv = tl.load(q_ptr + q_base + (SEG_START + _g * 2 + 1) * q_stride_d,
                                 mask=m_mask, other=0.0).to(tl.float32)
                    logits += qv[:, None] * (cent * seg_norm)[None, :]

        # NOTE: SEG_BW == 5 is handled by SEG_BW >= 5 raw path above
        # (5-8 bit codes are stored as raw uint8).

        return logits

    # -----------------------------------------------------------------------
    # Decode-kernel helper: per-segment nibble-4 decode + tensor-core QK dot
    # -----------------------------------------------------------------------

    @triton.jit
    def _seg_nibble_dot(
        k_nib4_ptr,       # nibble-4 packed K: (B, n_kv, T, D//2) uint8
        kp_base,           # (BLOCK_T,) int64 base offsets into nibble-4 buffer
        codebook_ptr,      # flat codebook
        seg_norm,          # (BLOCK_T,) fp32 segment norm
        q_ptr,             # Q pointer
        q_base,            # (BLOCK_M,) int64 base offsets
        q_stride_d,        # Q dim stride
        m_mask,            # (BLOCK_M,) bool
        mask_t,            # (BLOCK_T,) bool
        # Segment constexprs
        SEG_START: tl.constexpr,
        SEG_LEN: tl.constexpr,
        SEG_LUT_OFF: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_T: tl.constexpr,
    ):
        """Nibble-4 K decode for one segment + tensor-core QK partial dot.

        1. Load (BLOCK_T, seg_half) nibble bytes
        2. Unpack lo/hi nibbles → codes
        3. Gather from segment codebook
        4. Multiply by segment norm
        5. Interleave → K_chunk (BLOCK_T, seg_pad) fp16
        6. Load Q for segment dims
        7. tl.dot(Q_seg, K_chunk.T) → partial logits (BLOCK_M, BLOCK_T)
        """
        # Pad segment length to next power of 2 (≥32) for tl.arange + tl.dot.
        # NOTE: SEG_PAD=16 is intentionally SKIPPED — Triton's `tl.interleave`
        # has a register-layout bug at HALF_PAD=8 that produces wrong tl.dot
        # output (RMSE 18 vs ref).  For SEG_LEN ≤ 16, we use SEG_PAD=32 so HALF_PAD=16 →
        # the buggy codegen path is bypassed.  Mask zeros the unused
        # K positions; second MMA contributes 0.
        SEG_PAD: tl.constexpr = (
            32 if SEG_LEN <= 32 else
            64 if SEG_LEN <= 64 else
            128
        )
        HALF_PAD: tl.constexpr = SEG_PAD // 2
        NIB_OFF: tl.constexpr = SEG_START // 2  # byte offset (SEG_START is always even)

        # 1. Load nibble bytes: (BLOCK_T, HALF_PAD)
        offs_half = tl.arange(0, HALF_PAD)
        packed = tl.load(
            k_nib4_ptr + kp_base[:, None] + NIB_OFF + offs_half[None, :].to(tl.int64),
            mask=mask_t[:, None] & (offs_half[None, :] < (SEG_LEN + 1) // 2),
            other=0,
        ).to(tl.int32)

        # 2. Unpack nibbles
        codes_lo = packed & 0xF           # even dims within segment
        codes_hi = (packed >> 4) & 0xF    # odd dims within segment

        # 3. Codebook gather: segment codebook
        c_lo = tl.load(
            codebook_ptr + SEG_LUT_OFF + codes_lo,
            mask=mask_t[:, None] & (offs_half[None, :] * 2 < SEG_LEN),
            other=0.0,
        ).to(tl.float16)
        c_hi = tl.load(
            codebook_ptr + SEG_LUT_OFF + codes_hi,
            mask=mask_t[:, None] & (offs_half[None, :] * 2 + 1 < SEG_LEN),
            other=0.0,
        ).to(tl.float16)

        # 4. Norm multiply (broadcast segment norm)
        seg_norm_f16 = seg_norm.to(tl.float16)
        k_even = c_lo * seg_norm_f16[:, None]
        k_odd  = c_hi * seg_norm_f16[:, None]

        # 5. Interleave → (BLOCK_T, SEG_PAD) fp16
        k_chunk = tl.interleave(k_even, k_odd)

        # 6. Load Q for this segment's dims: (BLOCK_M, SEG_PAD) fp16
        offs_seg = tl.arange(0, SEG_PAD)
        q_seg = tl.load(
            q_ptr + q_base[:, None] + (SEG_START + offs_seg[None, :]) * q_stride_d,
            mask=m_mask[:, None] & (offs_seg[None, :] < SEG_LEN),
            other=0.0,
        ).to(tl.float16)

        # 7. Tensor-core QK: (BLOCK_M, SEG_PAD) @ (SEG_PAD, BLOCK_T)
        partial = tl.dot(q_seg, tl.trans(k_chunk))  # fp32 result

        return partial

    @triton.jit
    def _seg_uint8_dot(
        k_ptr,             # mixed-packed K data pointer
        kp_base,           # (BLOCK_T,) int64 base offsets into packed buffer
        codebook_ptr,      # flat codebook
        seg_norm,          # (BLOCK_T,) fp32 segment norm
        q_ptr,             # Q pointer
        q_base,            # (BLOCK_M,) int64 base offsets
        q_stride_d,        # Q dim stride
        m_mask,            # (BLOCK_M,) bool
        mask_t,            # (BLOCK_T,) bool
        # Segment constexprs
        SEG_START: tl.constexpr,    # dim start in pack_perm order
        SEG_LEN: tl.constexpr,      # number of dims
        SEG_LUT_OFF: tl.constexpr,  # codebook offset
        BYTE_OFF: tl.constexpr,     # byte offset into packed buffer
        BLOCK_M: tl.constexpr,
        BLOCK_T: tl.constexpr,
    ):
        """Uint8 K decode for one 5-8 bit segment + tensor-core QK dot.

        For segments with bw > 4 in mixed packing format.
        Each code is stored as 1 uint8 byte (no nibble packing).

        1. Load (BLOCK_T, seg_pad) uint8 codes
        2. Gather from segment codebook
        3. Multiply by segment norm → K_chunk
        4. Load Q for segment dims
        5. tl.dot(Q_seg, K_chunk.T) → partial logits
        """
        SEG_PAD: tl.constexpr = (
            16 if SEG_LEN <= 16 else
            32 if SEG_LEN <= 32 else
            64 if SEG_LEN <= 64 else
            128
        )

        # 1. Load uint8 codes: (BLOCK_T, SEG_PAD)
        offs_d = tl.arange(0, SEG_PAD)
        codes = tl.load(
            k_ptr + kp_base[:, None] + BYTE_OFF + offs_d[None, :].to(tl.int64),
            mask=mask_t[:, None] & (offs_d[None, :] < SEG_LEN),
            other=0,
        ).to(tl.int32)

        # 2. Codebook gather
        centroids = tl.load(
            codebook_ptr + SEG_LUT_OFF + codes,
            mask=mask_t[:, None] & (offs_d[None, :] < SEG_LEN),
            other=0.0,
        ).to(tl.float16)

        # 3. Norm multiply → K_chunk (BLOCK_T, SEG_PAD) fp16
        k_chunk = centroids * seg_norm.to(tl.float16)[:, None]

        # 4. Load Q for segment dims: (BLOCK_M, SEG_PAD) fp16
        q_seg = tl.load(
            q_ptr + q_base[:, None] + (SEG_START + offs_d[None, :]) * q_stride_d,
            mask=m_mask[:, None] & (offs_d[None, :] < SEG_LEN),
            other=0.0,
        ).to(tl.float16)

        # 5. Tensor-core QK: (BLOCK_M, SEG_PAD) @ (SEG_PAD, BLOCK_T)
        return tl.dot(q_seg, tl.trans(k_chunk))


# ===========================================================================
# Python wrappers
# ===========================================================================


def fused_blockgtq_decode_attention(
    q: torch.Tensor,              # (B, n_q_heads, 1, D) fp16
    k_packed: torch.Tensor,       # (B, n_kv_heads, T, total_packed_bytes) uint8
    k_norms: torch.Tensor,        # (B, n_kv_heads, T, n_groups) fp16
    codebook_flat: torch.Tensor,  # (total_entries,) fp16
    lut_offset: torch.Tensor,     # (D,) int32 — per-dim LUT offset
    norm_group: torch.Tensor,     # (D,) int32 — per-dim norm group index
    segments: list,               # list of (bw, dim_start, dim_len, pack_off, lut_off, norm_idx)
    v_codes: torch.Tensor,
    v_norms: torch.Tensor,
    v_lut: torch.Tensor,
    scale: float = None,
    block_t: int = 64,
    block_m: int = 16,
    t_splits: int = 32,
    v_skip_thresh: float = 0.0,
    v_norm_scale: torch.Tensor = None,
    onebit_fast = 'auto',
    q_pack_perm: torch.Tensor = None,
    v_bits: int = 0,
    v_rotation: torch.Tensor = None,
    k_nibble: bool = False,
    v_nibble: bool = False,
    v_paired_2b: bool = False,
    k_paired_2b: bool = False,
    k_paired_4b: bool = False,
    k_paired_8b: bool = False,
    k_paired_1b: bool = False,
    return_lse: bool = False,
) -> torch.Tensor:
    """Per-segment partial-dot decode.

    When k_nibble=False (default): original bit-packed K with scalar QK dot.
    When k_nibble=True: k_packed is nibble-4 format (B, n_kv, T, D//2) uint8.
      Uses tensor-core _seg_nibble_dot for QK on segments ≥ 16 dims.
      ONEBIT_FAST LUT building is skipped (all codes are nibble-4).

    onebit_fast: 1-bit segment decode strategy (ignored when k_nibble=True).
      'auto'    = auto-select: masked_sum if total 1-bit dims >= 32, else standard
      True or 4 = masked_sum (conditional accumulate, no per-dim multiply)
      3         = nibble-LUT (two 16-entry tables per 8-dim group)
      2         = byte-LUT (256-entry table, wins only at large T)
      1         = (μ,h) sign-dot (no codebook loads, ~same speed)
      False or 0 = standard unpack→codebook gather (baseline)

    q_pack_perm: optional (D,) int64 tensor — pack_perm to apply to Q at
      runtime. Use when head_perm is baked into q_proj but pack_perm is not.
      If Q is fully pre-permuted (via bake_permutation_into_qproj), omit this.

    v_bits: 0 = v_codes is raw uint8 (D per token), >0 = v_codes is packed
      at this bit width (packed_bytes per token). Kernel unpacks on-the-fly.
      Ignored when v_nibble=True.

    v_nibble: when True, v_codes is nibble-4 packed (D//2 bytes per token).
      Enables fast inline V decode via V_NIBBLE=1 constexpr path.
      Overrides v_bits (sets V_BITS=0, V_NIBBLE=1).
    """
    assert HAS_TRITON
    assert q.dim() == 4 and q.shape[2] == 1
    assert len(segments) <= 8

    # Resolve onebit_fast (skip when K_NIBBLE — all codes are nibble-4)
    if k_nibble:
        _onebit_fast = 0
    else:
        _ONEBIT_MASKED_SUM_THRESHOLD = 32  # use Mode 4 when total 1-bit dims >= this
        if onebit_fast == 'auto':
            total_1bit_dims = sum(length for bw, _, length, _, _, _ in segments if bw == 1)
            _onebit_fast = 4 if total_1bit_dims >= _ONEBIT_MASKED_SUM_THRESHOLD else 0
        elif onebit_fast is True:
            _onebit_fast = 4
        elif onebit_fast is False:
            _onebit_fast = 0
        else:
            _onebit_fast = int(onebit_fast)

    v_norm_bits = 4 if v_norms.dtype == torch.uint8 else 16

    B, n_q_heads, _, D = q.shape
    _, n_kv_heads, T, _ = k_packed.shape
    gqa_ratio = n_q_heads // n_kv_heads

    if v_norms.dim() == 3:
        v_norms = v_norms.unsqueeze(-1)
    if scale is None:
        scale = 1.0 / math.sqrt(D)

    v_nb_raw = v_norms.shape[-1]
    v_nb = v_nb_raw * 2 if v_norm_bits == 4 else v_nb_raw
    v_quant_block = D // v_nb

    if v_norm_scale is None:
        v_norm_scale = v_norms.new_empty(1, 1)

    partial_out = torch.empty(B, n_q_heads, t_splits, D,
                               dtype=torch.float16, device=q.device)
    partial_lse = torch.full((B, n_q_heads, t_splits), -float('inf'),
                              dtype=torch.float32, device=q.device)

    # Apply runtime pack_perm gather on Q if provided (128-element gather, <1μs)
    if q_pack_perm is not None:
        q = q[:, :, :, q_pack_perm]

    q_2d = q.reshape(B, n_q_heads, D).contiguous()

    # ---- Compute 1-bit LUT if needed (Mode 2: byte-LUT, Mode 3: nibble-LUT) ----
    total_1bit_groups = 0  # units depend on mode: groups for mode 2, nibbles for mode 3
    seg_1bit_goffs = []
    lut_entry_size = 1  # entries per unit (256 for byte-LUT, 16 for nibble-LUT)

    for bw, start, length, pack_off, lut_off, norm_idx in segments:
        if bw == 1 and _onebit_fast in (2, 3):
            n_byte_groups = (length + 7) // 8
            seg_1bit_goffs.append(total_1bit_groups)
            if _onebit_fast == 3:
                total_1bit_groups += n_byte_groups * 2  # 2 nibbles per byte group
            else:
                total_1bit_groups += n_byte_groups
        else:
            seg_1bit_goffs.append(0)

    if total_1bit_groups > 0 and _onebit_fast == 3:
        # Build nibble-LUT: shape [B*H_q, total_1bit_nibbles, 16] fp32
        # nibble 2g   → lo (dims d..d+3): entry[n] = c₀·Σq_lo + δ·Σ_{bits=1} q[d+b]
        # nibble 2g+1 → hi (dims d+4..d+7): same for upper 4 dims
        nib_bits = ((torch.arange(16, device=q.device).unsqueeze(1)
                     >> torch.arange(4, device=q.device)) & 1).float()  # [16, 4]

        byte_lut = torch.zeros(B * n_q_heads, total_1bit_groups, 16,
                                dtype=torch.float32, device=q.device)

        for seg_idx, (bw, start, length, pack_off, lut_off, norm_idx) in enumerate(segments):
            if bw != 1:
                continue
            c0 = codebook_flat[lut_off].float().item()
            c1 = codebook_flat[lut_off + 1].float().item()
            delta = c1 - c0
            noff = seg_1bit_goffs[seg_idx]
            n_byte_groups = (length + 7) // 8

            for g in range(n_byte_groups):
                d_base = start + g * 8
                # Lo nibble: dims d_base..d_base+3
                d_lo_end = min(d_base + 4, start + length)
                n_lo = d_lo_end - d_base
                if n_lo > 0:
                    q_lo = q_2d[:, :, d_base:d_lo_end].float().reshape(B * n_q_heads, n_lo)
                    q_sum_lo = q_lo.sum(dim=-1, keepdim=True)
                    lut_lo = q_lo @ nib_bits[:, :n_lo].T  # [B*H_q, 16]
                    byte_lut[:, noff + g * 2, :] = c0 * q_sum_lo + delta * lut_lo

                # Hi nibble: dims d_base+4..d_base+7
                d_hi_start = d_base + 4
                d_hi_end = min(d_base + 8, start + length)
                n_hi = d_hi_end - d_hi_start
                if n_hi > 0:
                    q_hi = q_2d[:, :, d_hi_start:d_hi_end].float().reshape(B * n_q_heads, n_hi)
                    q_sum_hi = q_hi.sum(dim=-1, keepdim=True)
                    lut_hi = q_hi @ nib_bits[:, :n_hi].T  # [B*H_q, 16]
                    byte_lut[:, noff + g * 2 + 1, :] = c0 * q_sum_hi + delta * lut_hi

        byte_lut = byte_lut.contiguous()
        lut_entry_size = 16

    elif total_1bit_groups > 0 and _onebit_fast == 2:
        # Build byte-LUT: shape [B*H_q, total_1bit_groups, 256] fp32
        byte_indices = torch.arange(256, device=q.device)
        bits = ((byte_indices.unsqueeze(1) >> torch.arange(8, device=q.device)) & 1).float()

        byte_lut = torch.zeros(B * n_q_heads, total_1bit_groups, 256,
                                dtype=torch.float32, device=q.device)

        for seg_idx, (bw, start, length, pack_off, lut_off, norm_idx) in enumerate(segments):
            if bw != 1:
                continue
            c0 = codebook_flat[lut_off].float().item()
            c1 = codebook_flat[lut_off + 1].float().item()
            delta = c1 - c0
            goff = seg_1bit_goffs[seg_idx]
            n_groups = (length + 7) // 8

            for g in range(n_groups):
                d_start = start + g * 8
                d_end = min(d_start + 8, start + length)
                n_dims = d_end - d_start
                q_group = q_2d[:, :, d_start:d_end].float().reshape(B * n_q_heads, n_dims)
                q_sum = q_group.sum(dim=-1, keepdim=True)
                lut_val = q_group @ bits[:, :n_dims].T
                byte_lut[:, goff + g, :] = c0 * q_sum + delta * lut_val

        byte_lut = byte_lut.contiguous()
        lut_entry_size = 256
    else:
        byte_lut = torch.empty(1, dtype=torch.float32, device=q.device)
        total_1bit_groups = max(total_1bit_groups, 1)

    # Pad segments to 8 (now 7-tuples with 1bit_goff)
    seg_1bit_goffs = seg_1bit_goffs + [0] * (8 - len(seg_1bit_goffs))
    seg_pad = list(segments) + [(0, 0, 0, 0, 0, 0)] * (8 - len(segments))

    # ---- K_NIBBLE mixed packing: recompute PACK_OFF for bw>4 segments ----
    # In mixed packing layout: [nibble_bytes | uint8_bytes] per token.
    # nibble_bytes = nopack_start // 2 (all ≤4-bit dims packed as nibble-4).
    # For bw>4 segments, PACK_OFF = nib_bytes + (seg_start - nopack_start).
    if k_nibble:
        # Find nopack boundary: first dim of any bw>4 segment
        nopack_start = D  # default: all dims are nibble-packed
        for bw, start, length, pack_off, lut_off, norm_idx in segments:
            if bw > 4 and length > 0:
                nopack_start = min(nopack_start, start)
        nib_bytes = nopack_start // 2
        # Rewrite PACK_OFF for uint8 segments
        for i in range(len(seg_pad)):
            bw, start, length, pack_off, lut_off, norm_idx = seg_pad[i]
            if bw > 4 and length > 0:
                new_pack_off = nib_bytes + (start - nopack_start)
                seg_pad[i] = (bw, start, length, new_pack_off, lut_off, norm_idx)

    # ---- Build V unpack tables if V is packed ----
    # V_PAIRED_2B has its own inline decode path (no v_unpack table), same as
    # V_NIBBLE. It therefore sets _v_bits=0 from the kernel's perspective.
    _v_nibble = 1 if v_nibble else 0
    _v_paired_2b = 1 if v_paired_2b else 0
    if v_nibble or v_paired_2b:
        _v_bits = 0
    else:
        _v_bits = int(v_bits)
    if _v_bits > 0:
        from blockgtq.v_packing import build_v_unpack_tables
        blo, bhi, slo, shi, vmask = build_v_unpack_tables(_v_bits, D, device=q.device)
        v_unpack = torch.stack([blo, bhi, slo, shi, vmask], dim=0).contiguous()
    else:
        v_unpack = torch.zeros(5, D, dtype=torch.int32, device=q.device)

    # V un-rotation matrix
    _v_unrot = 1 if v_rotation is not None else 0
    if v_rotation is not None:
        v_rot_tensor = v_rotation.contiguous().to(torch.float32)
    else:
        v_rot_tensor = torch.empty(1, 1, dtype=torch.float32, device=q.device)

    grid_compute = (B * n_kv_heads * t_splits,)
    _packed_decode_kernel[grid_compute](
        q_2d, k_packed, k_norms, v_codes, v_norms, v_lut,
        v_norm_scale, v_unpack, v_rot_tensor,
        partial_out, partial_lse, codebook_flat,
        byte_lut,
        lut_offset, norm_group,
        q_2d.stride(0), q_2d.stride(1), q_2d.stride(2),
        k_packed.stride(0), k_packed.stride(1), k_packed.stride(2),
        k_norms.stride(0), k_norms.stride(1), k_norms.stride(2), k_norms.stride(3),
        v_codes.stride(0), v_codes.stride(1), v_codes.stride(2),
        v_codes.stride(3) if v_codes.dim() >= 4 else 1,
        v_norms.stride(0), v_norms.stride(1), v_norms.stride(2), v_norms.stride(3),
        v_norm_scale.stride(0),
        v_norm_scale.stride(1) if v_norm_scale.dim() >= 2 else 0,
        v_unpack.stride(0),
        partial_out.stride(0), partial_out.stride(1),
        partial_out.stride(2), partial_out.stride(3),
        partial_lse.stride(0), partial_lse.stride(1), partial_lse.stride(2),
        T, n_q_heads, n_kv_heads,
        SEG0_BW=seg_pad[0][0], SEG0_START=seg_pad[0][1], SEG0_LEN=seg_pad[0][2],
        SEG0_PACK_OFF=seg_pad[0][3], SEG0_LUT_OFF=seg_pad[0][4], SEG0_NORM_IDX=seg_pad[0][5],
        SEG0_1BIT_GOFF=seg_1bit_goffs[0],
        SEG1_BW=seg_pad[1][0], SEG1_START=seg_pad[1][1], SEG1_LEN=seg_pad[1][2],
        SEG1_PACK_OFF=seg_pad[1][3], SEG1_LUT_OFF=seg_pad[1][4], SEG1_NORM_IDX=seg_pad[1][5],
        SEG1_1BIT_GOFF=seg_1bit_goffs[1],
        SEG2_BW=seg_pad[2][0], SEG2_START=seg_pad[2][1], SEG2_LEN=seg_pad[2][2],
        SEG2_PACK_OFF=seg_pad[2][3], SEG2_LUT_OFF=seg_pad[2][4], SEG2_NORM_IDX=seg_pad[2][5],
        SEG2_1BIT_GOFF=seg_1bit_goffs[2],
        SEG3_BW=seg_pad[3][0], SEG3_START=seg_pad[3][1], SEG3_LEN=seg_pad[3][2],
        SEG3_PACK_OFF=seg_pad[3][3], SEG3_LUT_OFF=seg_pad[3][4], SEG3_NORM_IDX=seg_pad[3][5],
        SEG3_1BIT_GOFF=seg_1bit_goffs[3],
        SEG4_BW=seg_pad[4][0], SEG4_START=seg_pad[4][1], SEG4_LEN=seg_pad[4][2],
        SEG4_PACK_OFF=seg_pad[4][3], SEG4_LUT_OFF=seg_pad[4][4], SEG4_NORM_IDX=seg_pad[4][5],
        SEG4_1BIT_GOFF=seg_1bit_goffs[4],
        SEG5_BW=seg_pad[5][0], SEG5_START=seg_pad[5][1], SEG5_LEN=seg_pad[5][2],
        SEG5_PACK_OFF=seg_pad[5][3], SEG5_LUT_OFF=seg_pad[5][4], SEG5_NORM_IDX=seg_pad[5][5],
        SEG5_1BIT_GOFF=seg_1bit_goffs[5],
        SEG6_BW=seg_pad[6][0], SEG6_START=seg_pad[6][1], SEG6_LEN=seg_pad[6][2],
        SEG6_PACK_OFF=seg_pad[6][3], SEG6_LUT_OFF=seg_pad[6][4], SEG6_NORM_IDX=seg_pad[6][5],
        SEG6_1BIT_GOFF=seg_1bit_goffs[6],
        SEG7_BW=seg_pad[7][0], SEG7_START=seg_pad[7][1], SEG7_LEN=seg_pad[7][2],
        SEG7_PACK_OFF=seg_pad[7][3], SEG7_LUT_OFF=seg_pad[7][4], SEG7_NORM_IDX=seg_pad[7][5],
        SEG7_1BIT_GOFF=seg_1bit_goffs[7],
        N_SEGMENTS=len(segments),
        TOTAL_1BIT_GROUPS=total_1bit_groups,
        D=D, BLOCK_M=block_m, GQA_RATIO=gqa_ratio,
        BLOCK_T=block_t, BLOCK_D=D, T_SPLITS=t_splits, SCALE=scale,
        V_SKIP_THRESH=v_skip_thresh,
        V_QUANT_BLOCK=v_quant_block,
        V_NORM_BITS=v_norm_bits,
        ONEBIT_FAST=_onebit_fast,
        V_BITS=_v_bits,
        V_UNROT=_v_unrot,
        K_NIBBLE=1 if k_nibble else 0,
        V_NIBBLE=_v_nibble,
        V_PAIRED_2B=_v_paired_2b,
        K_PAIRED_2B=1 if k_paired_2b else 0,
        K_PAIRED_4B=1 if k_paired_4b else 0,
        K_PAIRED_8B=1 if k_paired_8b else 0,
        K_PAIRED_1B=1 if k_paired_1b else 0,
    )

    out = torch.empty(B, n_q_heads, D, dtype=torch.float16, device=q.device)
    grid_merge = (B * n_q_heads,)
    _split_t_merge_kernel[grid_merge](
        partial_out, partial_lse, out,
        partial_out.stride(0), partial_out.stride(1),
        partial_out.stride(2), partial_out.stride(3),
        partial_lse.stride(0), partial_lse.stride(1), partial_lse.stride(2),
        out.stride(0), out.stride(1), out.stride(2),
        n_q_heads,
        D=D, BLOCK_D=D, T_SPLITS=t_splits,
    )
    if return_lse:
        # Compute total LSE from T_SPLITS partial LSEs.
        # partial_lse: (B, n_q_heads, t_splits) fp32, -inf for empty splits.
        lse = torch.logsumexp(partial_lse, dim=-1)  # (B, n_q_heads)
        return out.unsqueeze(2), lse
    return out.unsqueeze(2)


# ===================================================================
# Phase 2: FA2-style Prefill Kernel for Segmented Packed KV Cache
# ===================================================================
# Grid: (B * n_q_heads * Q_BLOCKS,)
# Q outer loop (FA2): each program owns one (batch, q_head, q_block)
# K/V inner loop: streams packed K/V tokens from HBM
# Segment innermost: reuses _seg_partial_dot for mixed-bit QK
# ===================================================================

if HAS_TRITON:

    @triton.jit
    def _packed_prefill_kernel(
        q_ptr,              # (B, n_q_heads, S_q, D) fp16
        kp_ptr,             # (B, H_kv, T, total_packed_bytes) uint8
        k_norms_ptr,        # (B, H_kv, T, N_GROUPS) fp16
        v_codes_ptr,        # packed (B,H_kv,T,pb) uint8
        v_norms_ptr,
        v_lut_ptr,
        v_norm_scale_ptr,
        v_unpack_ptr,       # (5, D) int32 V unpack tables
        v_rot_ptr,          # (D, D) fp32 V un-rotation matrix
        out_ptr,            # (B, n_q_heads, S_q, D) fp16
        codebook_ptr,
        lut_offset_ptr,
        norm_group_ptr,
        # Strides
        q_stride_b, q_stride_h, q_stride_t, q_stride_d,
        kp_stride_b, kp_stride_h, kp_stride_t,
        kn_stride_b, kn_stride_h, kn_stride_t, kn_stride_s,
        vc_stride_b, vc_stride_h, vc_stride_t, vc_stride_d,
        vn_stride_b, vn_stride_h, vn_stride_t, vn_stride_blk,
        vns_stride_b, vns_stride_h,
        vu_stride,
        o_stride_b, o_stride_h, o_stride_t, o_stride_d,
        # Sizes
        S_q, T, n_q_heads, n_kv_heads,
        # Segment constexprs (up to 8 segments)
        SEG0_BW: tl.constexpr, SEG0_START: tl.constexpr, SEG0_LEN: tl.constexpr,
        SEG0_PACK_OFF: tl.constexpr, SEG0_LUT_OFF: tl.constexpr,
        SEG0_NORM_IDX: tl.constexpr, SEG0_1BIT_GOFF: tl.constexpr,
        SEG1_BW: tl.constexpr, SEG1_START: tl.constexpr, SEG1_LEN: tl.constexpr,
        SEG1_PACK_OFF: tl.constexpr, SEG1_LUT_OFF: tl.constexpr,
        SEG1_NORM_IDX: tl.constexpr, SEG1_1BIT_GOFF: tl.constexpr,
        SEG2_BW: tl.constexpr, SEG2_START: tl.constexpr, SEG2_LEN: tl.constexpr,
        SEG2_PACK_OFF: tl.constexpr, SEG2_LUT_OFF: tl.constexpr,
        SEG2_NORM_IDX: tl.constexpr, SEG2_1BIT_GOFF: tl.constexpr,
        SEG3_BW: tl.constexpr, SEG3_START: tl.constexpr, SEG3_LEN: tl.constexpr,
        SEG3_PACK_OFF: tl.constexpr, SEG3_LUT_OFF: tl.constexpr,
        SEG3_NORM_IDX: tl.constexpr, SEG3_1BIT_GOFF: tl.constexpr,
        SEG4_BW: tl.constexpr, SEG4_START: tl.constexpr, SEG4_LEN: tl.constexpr,
        SEG4_PACK_OFF: tl.constexpr, SEG4_LUT_OFF: tl.constexpr,
        SEG4_NORM_IDX: tl.constexpr, SEG4_1BIT_GOFF: tl.constexpr,
        SEG5_BW: tl.constexpr, SEG5_START: tl.constexpr, SEG5_LEN: tl.constexpr,
        SEG5_PACK_OFF: tl.constexpr, SEG5_LUT_OFF: tl.constexpr,
        SEG5_NORM_IDX: tl.constexpr, SEG5_1BIT_GOFF: tl.constexpr,
        SEG6_BW: tl.constexpr, SEG6_START: tl.constexpr, SEG6_LEN: tl.constexpr,
        SEG6_PACK_OFF: tl.constexpr, SEG6_LUT_OFF: tl.constexpr,
        SEG6_NORM_IDX: tl.constexpr, SEG6_1BIT_GOFF: tl.constexpr,
        SEG7_BW: tl.constexpr, SEG7_START: tl.constexpr, SEG7_LEN: tl.constexpr,
        SEG7_PACK_OFF: tl.constexpr, SEG7_LUT_OFF: tl.constexpr,
        SEG7_NORM_IDX: tl.constexpr, SEG7_1BIT_GOFF: tl.constexpr,
        N_SEGMENTS: tl.constexpr,
        TOTAL_1BIT_GROUPS: tl.constexpr,
        # Constants
        D: tl.constexpr,
        BLOCK_Q: tl.constexpr,
        GQA_RATIO: tl.constexpr,
        BLOCK_T: tl.constexpr,
        BLOCK_D: tl.constexpr,
        SCALE: tl.constexpr,
        V_SKIP_THRESH: tl.constexpr,
        V_QUANT_BLOCK: tl.constexpr,
        V_NORM_BITS: tl.constexpr,
        ONEBIT_FAST: tl.constexpr = 0,
        V_BITS: tl.constexpr = 0,
        V_UNROT: tl.constexpr = 0,
        IS_CAUSAL: tl.constexpr = 1,
        q_pos_offset: tl.int32 = 0,      # global position of Q[0]; runtime (not constexpr) to
                                          # avoid 1 JIT variant per chunk_start value
        K_NIBBLE: tl.constexpr = 0,      # 1: use _seg_nibble_dot tensor-core path for bw≤4 segs
        V_NIBBLE: tl.constexpr = 0,      # 1: use inline nibble V decode (avoids _load_v_codes)
        MIN_SEG_FOR_DOT: tl.constexpr = 16,
    ):
        """FA2-style prefill kernel for segmented packed KV cache.

        Q outer (FA2): each program handles BLOCK_Q query tokens.
        K/V inner: streams compressed KV in BLOCK_T blocks.
        Segment innermost: per-segment QK via _seg_partial_dot.
        Online softmax with causal masking.
        """
        pid = tl.program_id(0)
        q_blocks = (S_q + BLOCK_Q - 1) // BLOCK_Q
        q_block_idx = pid % q_blocks
        bh = pid // q_blocks
        h_q = bh % n_q_heads
        b = bh // n_q_heads

        h_kv = h_q // GQA_RATIO

        b64 = b.to(tl.int64)
        hkv64 = h_kv.to(tl.int64)
        hq64 = h_q.to(tl.int64)

        q_start = q_block_idx * BLOCK_Q
        offs_q = tl.arange(0, BLOCK_Q)
        q_mask = (q_start + offs_q) < S_q
        offs_d = tl.arange(0, BLOCK_D)
        v_block_ids = (offs_d // V_QUANT_BLOCK).to(tl.int64)

        # Q base: each element is a different query token (same head)
        q_base = (b64 * q_stride_b + hq64 * q_stride_h
                  + (q_start + offs_q.to(tl.int64)) * q_stride_t)

        m_i = tl.full((BLOCK_Q,), -float('inf'), dtype=tl.float32)
        l_i = tl.zeros((BLOCK_Q,), dtype=tl.float32)
        acc = tl.zeros((BLOCK_Q, BLOCK_D), dtype=tl.float32)

        # Causal: only attend up to (q_pos_offset + q_start + BLOCK_Q - 1)
        # q_pos_offset shifts Q's global position so chunked prefill (Q at
        # chunk_start, K cache from 0) gets the right K visibility window.
        if IS_CAUSAL == 1:
            t_end = tl.minimum(T, q_pos_offset + q_start + BLOCK_Q)
        else:
            t_end = T

        # Dummy byte_lut for _seg_partial_dot (ONEBIT_FAST=4 doesn't use it)
        byte_lut_dummy = q_ptr  # unused pointer
        lut_head_base = tl.zeros([BLOCK_Q], dtype=tl.int64)

        for t_start in range(0, t_end, BLOCK_T):
            offs_t = t_start + tl.arange(0, BLOCK_T)
            mask_t = offs_t < t_end
            offs_t64 = offs_t.to(tl.int64)

            kp_base = (b64 * kp_stride_b + hkv64 * kp_stride_h
                       + offs_t64 * kp_stride_t)

            logits = tl.zeros((BLOCK_Q, BLOCK_T), dtype=tl.float32)

            # ---- Segment 0 (4-branch K_NIBBLE dispatch + bit-packed fallback) ----
            if K_NIBBLE:
                if SEG0_BW <= 4 and SEG0_LEN >= MIN_SEG_FOR_DOT:
                    # nibble-4 K + tensor-core dot
                    kn_off_0 = (b64 * kn_stride_b + hkv64 * kn_stride_h
                                + offs_t64 * kn_stride_t + SEG0_NORM_IDX * kn_stride_s)
                    seg0_norm = tl.load(k_norms_ptr + kn_off_0, mask=mask_t, other=1.0)
                    logits += _seg_nibble_dot(
                        kp_ptr, kp_base, codebook_ptr, seg0_norm,
                        q_ptr, q_base, q_stride_d, q_mask, mask_t,
                        SEG0_START, SEG0_LEN, SEG0_LUT_OFF, BLOCK_Q, BLOCK_T)
                elif SEG0_BW <= 4 and SEG0_LEN > 0:
                    # nibble-4 K + scalar fallback (small segment)
                    kn_off_0 = (b64 * kn_stride_b + hkv64 * kn_stride_h
                                + offs_t64 * kn_stride_t + SEG0_NORM_IDX * kn_stride_s)
                    seg_norm_0 = tl.load(k_norms_ptr + kn_off_0, mask=mask_t, other=1.0)
                    NIB_OFF_0: tl.constexpr = SEG0_START // 2
                    for _k in tl.static_range(0, (SEG0_LEN + 1) // 2):
                        byte = tl.load(kp_ptr + kp_base + NIB_OFF_0 + _k,
                                       mask=mask_t, other=0).to(tl.int32)
                        lo_code = byte & 0xF
                        hi_code = (byte >> 4) & 0xF
                        if _k * 2 < SEG0_LEN:
                            cent = tl.load(codebook_ptr + SEG0_LUT_OFF + lo_code).to(tl.float32)
                            qv = tl.load(q_ptr + q_base + (SEG0_START + _k * 2) * q_stride_d,
                                         mask=q_mask, other=0.0).to(tl.float32)
                            logits += qv[:, None] * (cent * seg_norm_0)[None, :]
                        if _k * 2 + 1 < SEG0_LEN:
                            cent = tl.load(codebook_ptr + SEG0_LUT_OFF + hi_code).to(tl.float32)
                            qv = tl.load(q_ptr + q_base + (SEG0_START + _k * 2 + 1) * q_stride_d,
                                         mask=q_mask, other=0.0).to(tl.float32)
                            logits += qv[:, None] * (cent * seg_norm_0)[None, :]
                elif SEG0_LEN >= MIN_SEG_FOR_DOT:  # bw > 4: uint8 tensor-core
                    kn_off_0 = (b64 * kn_stride_b + hkv64 * kn_stride_h
                                + offs_t64 * kn_stride_t + SEG0_NORM_IDX * kn_stride_s)
                    seg_norm_0 = tl.load(k_norms_ptr + kn_off_0, mask=mask_t, other=1.0)
                    logits += _seg_uint8_dot(
                        kp_ptr, kp_base, codebook_ptr, seg_norm_0,
                        q_ptr, q_base, q_stride_d, q_mask, mask_t,
                        SEG0_START, SEG0_LEN, SEG0_LUT_OFF, SEG0_PACK_OFF,
                        BLOCK_Q, BLOCK_T)
                elif SEG0_LEN > 0:  # bw > 4: uint8 scalar
                    kn_off_0 = (b64 * kn_stride_b + hkv64 * kn_stride_h
                                + offs_t64 * kn_stride_t + SEG0_NORM_IDX * kn_stride_s)
                    seg_norm_0 = tl.load(k_norms_ptr + kn_off_0, mask=mask_t, other=1.0)
                    for _k in tl.static_range(0, SEG0_LEN):
                        code = tl.load(kp_ptr + kp_base + SEG0_PACK_OFF + _k,
                                       mask=mask_t, other=0).to(tl.int32)
                        cent = tl.load(codebook_ptr + SEG0_LUT_OFF + code).to(tl.float32)
                        qv = tl.load(q_ptr + q_base + (SEG0_START + _k) * q_stride_d,
                                     mask=q_mask, other=0.0).to(tl.float32)
                        logits += qv[:, None] * (cent * seg_norm_0)[None, :]
            elif SEG0_LEN > 0:
                # K_NIBBLE=False: original bit-packed scalar dot
                logits = _seg_partial_dot(
                    q_ptr, q_base, q_stride_d, q_mask,
                    kp_ptr, kp_base, codebook_ptr, k_norms_ptr,
                    b64, hkv64, offs_t64, mask_t,
                    kn_stride_b, kn_stride_h, kn_stride_t, kn_stride_s,
                    logits, byte_lut_dummy, lut_head_base,
                    SEG0_BW, SEG0_START, SEG0_LEN, SEG0_PACK_OFF,
                    SEG0_LUT_OFF, SEG0_NORM_IDX, SEG0_1BIT_GOFF,
                    BLOCK_Q, BLOCK_T, ONEBIT_FAST,
                )

            # ---- Segment 1 ----
            if N_SEGMENTS >= 2:
                if K_NIBBLE:
                    if SEG1_BW <= 4 and SEG1_LEN >= MIN_SEG_FOR_DOT:
                        kn_off_1 = (b64 * kn_stride_b + hkv64 * kn_stride_h
                                    + offs_t64 * kn_stride_t + SEG1_NORM_IDX * kn_stride_s)
                        seg1_norm = tl.load(k_norms_ptr + kn_off_1, mask=mask_t, other=1.0)
                        logits += _seg_nibble_dot(
                            kp_ptr, kp_base, codebook_ptr, seg1_norm,
                            q_ptr, q_base, q_stride_d, q_mask, mask_t,
                            SEG1_START, SEG1_LEN, SEG1_LUT_OFF, BLOCK_Q, BLOCK_T)
                    elif SEG1_BW <= 4 and SEG1_LEN > 0:
                        kn_off_1 = (b64 * kn_stride_b + hkv64 * kn_stride_h
                                    + offs_t64 * kn_stride_t + SEG1_NORM_IDX * kn_stride_s)
                        seg_norm_1 = tl.load(k_norms_ptr + kn_off_1, mask=mask_t, other=1.0)
                        NIB_OFF_1: tl.constexpr = SEG1_START // 2
                        for _k in tl.static_range(0, (SEG1_LEN + 1) // 2):
                            byte = tl.load(kp_ptr + kp_base + NIB_OFF_1 + _k,
                                           mask=mask_t, other=0).to(tl.int32)
                            lo_code = byte & 0xF
                            hi_code = (byte >> 4) & 0xF
                            if _k * 2 < SEG1_LEN:
                                cent = tl.load(codebook_ptr + SEG1_LUT_OFF + lo_code).to(tl.float32)
                                qv = tl.load(q_ptr + q_base + (SEG1_START + _k * 2) * q_stride_d,
                                             mask=q_mask, other=0.0).to(tl.float32)
                                logits += qv[:, None] * (cent * seg_norm_1)[None, :]
                            if _k * 2 + 1 < SEG1_LEN:
                                cent = tl.load(codebook_ptr + SEG1_LUT_OFF + hi_code).to(tl.float32)
                                qv = tl.load(q_ptr + q_base + (SEG1_START + _k * 2 + 1) * q_stride_d,
                                             mask=q_mask, other=0.0).to(tl.float32)
                                logits += qv[:, None] * (cent * seg_norm_1)[None, :]
                    elif SEG1_LEN >= MIN_SEG_FOR_DOT:  # bw > 4: uint8 tensor-core
                        kn_off_1 = (b64 * kn_stride_b + hkv64 * kn_stride_h
                                    + offs_t64 * kn_stride_t + SEG1_NORM_IDX * kn_stride_s)
                        seg_norm_1 = tl.load(k_norms_ptr + kn_off_1, mask=mask_t, other=1.0)
                        logits += _seg_uint8_dot(
                            kp_ptr, kp_base, codebook_ptr, seg_norm_1,
                            q_ptr, q_base, q_stride_d, q_mask, mask_t,
                            SEG1_START, SEG1_LEN, SEG1_LUT_OFF, SEG1_PACK_OFF,
                            BLOCK_Q, BLOCK_T)
                    elif SEG1_LEN > 0:  # bw > 4: uint8 scalar
                        kn_off_1 = (b64 * kn_stride_b + hkv64 * kn_stride_h
                                    + offs_t64 * kn_stride_t + SEG1_NORM_IDX * kn_stride_s)
                        seg_norm_1 = tl.load(k_norms_ptr + kn_off_1, mask=mask_t, other=1.0)
                        for _k in tl.static_range(0, SEG1_LEN):
                            code = tl.load(kp_ptr + kp_base + SEG1_PACK_OFF + _k,
                                           mask=mask_t, other=0).to(tl.int32)
                            cent = tl.load(codebook_ptr + SEG1_LUT_OFF + code).to(tl.float32)
                            qv = tl.load(q_ptr + q_base + (SEG1_START + _k) * q_stride_d,
                                         mask=q_mask, other=0.0).to(tl.float32)
                            logits += qv[:, None] * (cent * seg_norm_1)[None, :]
                elif SEG1_LEN > 0:
                    logits = _seg_partial_dot(
                        q_ptr, q_base, q_stride_d, q_mask,
                        kp_ptr, kp_base, codebook_ptr, k_norms_ptr,
                        b64, hkv64, offs_t64, mask_t,
                        kn_stride_b, kn_stride_h, kn_stride_t, kn_stride_s,
                        logits, byte_lut_dummy, lut_head_base,
                        SEG1_BW, SEG1_START, SEG1_LEN, SEG1_PACK_OFF,
                        SEG1_LUT_OFF, SEG1_NORM_IDX, SEG1_1BIT_GOFF,
                        BLOCK_Q, BLOCK_T, ONEBIT_FAST,
                    )

            # ---- Segment 2 ----
            if N_SEGMENTS >= 3:
                if K_NIBBLE:
                    if SEG2_BW <= 4 and SEG2_LEN >= MIN_SEG_FOR_DOT:
                        kn_off_2 = (b64 * kn_stride_b + hkv64 * kn_stride_h
                                    + offs_t64 * kn_stride_t + SEG2_NORM_IDX * kn_stride_s)
                        seg2_norm = tl.load(k_norms_ptr + kn_off_2, mask=mask_t, other=1.0)
                        logits += _seg_nibble_dot(
                            kp_ptr, kp_base, codebook_ptr, seg2_norm,
                            q_ptr, q_base, q_stride_d, q_mask, mask_t,
                            SEG2_START, SEG2_LEN, SEG2_LUT_OFF, BLOCK_Q, BLOCK_T)
                    elif SEG2_BW <= 4 and SEG2_LEN > 0:
                        kn_off_2 = (b64 * kn_stride_b + hkv64 * kn_stride_h
                                    + offs_t64 * kn_stride_t + SEG2_NORM_IDX * kn_stride_s)
                        seg_norm_2 = tl.load(k_norms_ptr + kn_off_2, mask=mask_t, other=1.0)
                        NIB_OFF_2: tl.constexpr = SEG2_START // 2
                        for _k in tl.static_range(0, (SEG2_LEN + 1) // 2):
                            byte = tl.load(kp_ptr + kp_base + NIB_OFF_2 + _k,
                                           mask=mask_t, other=0).to(tl.int32)
                            lo_code = byte & 0xF
                            hi_code = (byte >> 4) & 0xF
                            if _k * 2 < SEG2_LEN:
                                cent = tl.load(codebook_ptr + SEG2_LUT_OFF + lo_code).to(tl.float32)
                                qv = tl.load(q_ptr + q_base + (SEG2_START + _k * 2) * q_stride_d,
                                             mask=q_mask, other=0.0).to(tl.float32)
                                logits += qv[:, None] * (cent * seg_norm_2)[None, :]
                            if _k * 2 + 1 < SEG2_LEN:
                                cent = tl.load(codebook_ptr + SEG2_LUT_OFF + hi_code).to(tl.float32)
                                qv = tl.load(q_ptr + q_base + (SEG2_START + _k * 2 + 1) * q_stride_d,
                                             mask=q_mask, other=0.0).to(tl.float32)
                                logits += qv[:, None] * (cent * seg_norm_2)[None, :]
                    elif SEG2_LEN >= MIN_SEG_FOR_DOT:  # bw > 4: uint8 tensor-core
                        kn_off_2 = (b64 * kn_stride_b + hkv64 * kn_stride_h
                                    + offs_t64 * kn_stride_t + SEG2_NORM_IDX * kn_stride_s)
                        seg_norm_2 = tl.load(k_norms_ptr + kn_off_2, mask=mask_t, other=1.0)
                        logits += _seg_uint8_dot(
                            kp_ptr, kp_base, codebook_ptr, seg_norm_2,
                            q_ptr, q_base, q_stride_d, q_mask, mask_t,
                            SEG2_START, SEG2_LEN, SEG2_LUT_OFF, SEG2_PACK_OFF,
                            BLOCK_Q, BLOCK_T)
                    elif SEG2_LEN > 0:  # bw > 4: uint8 scalar
                        kn_off_2 = (b64 * kn_stride_b + hkv64 * kn_stride_h
                                    + offs_t64 * kn_stride_t + SEG2_NORM_IDX * kn_stride_s)
                        seg_norm_2 = tl.load(k_norms_ptr + kn_off_2, mask=mask_t, other=1.0)
                        for _k in tl.static_range(0, SEG2_LEN):
                            code = tl.load(kp_ptr + kp_base + SEG2_PACK_OFF + _k,
                                           mask=mask_t, other=0).to(tl.int32)
                            cent = tl.load(codebook_ptr + SEG2_LUT_OFF + code).to(tl.float32)
                            qv = tl.load(q_ptr + q_base + (SEG2_START + _k) * q_stride_d,
                                         mask=q_mask, other=0.0).to(tl.float32)
                            logits += qv[:, None] * (cent * seg_norm_2)[None, :]
                elif SEG2_LEN > 0:
                    logits = _seg_partial_dot(
                        q_ptr, q_base, q_stride_d, q_mask,
                        kp_ptr, kp_base, codebook_ptr, k_norms_ptr,
                        b64, hkv64, offs_t64, mask_t,
                        kn_stride_b, kn_stride_h, kn_stride_t, kn_stride_s,
                        logits, byte_lut_dummy, lut_head_base,
                        SEG2_BW, SEG2_START, SEG2_LEN, SEG2_PACK_OFF,
                        SEG2_LUT_OFF, SEG2_NORM_IDX, SEG2_1BIT_GOFF,
                        BLOCK_Q, BLOCK_T, ONEBIT_FAST,
                    )

            # ---- Segment 3 ----
            if N_SEGMENTS >= 4:
                if K_NIBBLE:
                    if SEG3_BW <= 4 and SEG3_LEN >= MIN_SEG_FOR_DOT:
                        kn_off_3 = (b64 * kn_stride_b + hkv64 * kn_stride_h
                                    + offs_t64 * kn_stride_t + SEG3_NORM_IDX * kn_stride_s)
                        seg3_norm = tl.load(k_norms_ptr + kn_off_3, mask=mask_t, other=1.0)
                        logits += _seg_nibble_dot(
                            kp_ptr, kp_base, codebook_ptr, seg3_norm,
                            q_ptr, q_base, q_stride_d, q_mask, mask_t,
                            SEG3_START, SEG3_LEN, SEG3_LUT_OFF, BLOCK_Q, BLOCK_T)
                    elif SEG3_BW <= 4 and SEG3_LEN > 0:
                        kn_off_3 = (b64 * kn_stride_b + hkv64 * kn_stride_h
                                    + offs_t64 * kn_stride_t + SEG3_NORM_IDX * kn_stride_s)
                        seg_norm_3 = tl.load(k_norms_ptr + kn_off_3, mask=mask_t, other=1.0)
                        NIB_OFF_3: tl.constexpr = SEG3_START // 2
                        for _k in tl.static_range(0, (SEG3_LEN + 1) // 2):
                            byte = tl.load(kp_ptr + kp_base + NIB_OFF_3 + _k,
                                           mask=mask_t, other=0).to(tl.int32)
                            lo_code = byte & 0xF
                            hi_code = (byte >> 4) & 0xF
                            if _k * 2 < SEG3_LEN:
                                cent = tl.load(codebook_ptr + SEG3_LUT_OFF + lo_code).to(tl.float32)
                                qv = tl.load(q_ptr + q_base + (SEG3_START + _k * 2) * q_stride_d,
                                             mask=q_mask, other=0.0).to(tl.float32)
                                logits += qv[:, None] * (cent * seg_norm_3)[None, :]
                            if _k * 2 + 1 < SEG3_LEN:
                                cent = tl.load(codebook_ptr + SEG3_LUT_OFF + hi_code).to(tl.float32)
                                qv = tl.load(q_ptr + q_base + (SEG3_START + _k * 2 + 1) * q_stride_d,
                                             mask=q_mask, other=0.0).to(tl.float32)
                                logits += qv[:, None] * (cent * seg_norm_3)[None, :]
                    elif SEG3_LEN >= MIN_SEG_FOR_DOT:  # bw > 4: uint8 tensor-core
                        kn_off_3 = (b64 * kn_stride_b + hkv64 * kn_stride_h
                                    + offs_t64 * kn_stride_t + SEG3_NORM_IDX * kn_stride_s)
                        seg_norm_3 = tl.load(k_norms_ptr + kn_off_3, mask=mask_t, other=1.0)
                        logits += _seg_uint8_dot(
                            kp_ptr, kp_base, codebook_ptr, seg_norm_3,
                            q_ptr, q_base, q_stride_d, q_mask, mask_t,
                            SEG3_START, SEG3_LEN, SEG3_LUT_OFF, SEG3_PACK_OFF,
                            BLOCK_Q, BLOCK_T)
                    elif SEG3_LEN > 0:  # bw > 4: uint8 scalar
                        kn_off_3 = (b64 * kn_stride_b + hkv64 * kn_stride_h
                                    + offs_t64 * kn_stride_t + SEG3_NORM_IDX * kn_stride_s)
                        seg_norm_3 = tl.load(k_norms_ptr + kn_off_3, mask=mask_t, other=1.0)
                        for _k in tl.static_range(0, SEG3_LEN):
                            code = tl.load(kp_ptr + kp_base + SEG3_PACK_OFF + _k,
                                           mask=mask_t, other=0).to(tl.int32)
                            cent = tl.load(codebook_ptr + SEG3_LUT_OFF + code).to(tl.float32)
                            qv = tl.load(q_ptr + q_base + (SEG3_START + _k) * q_stride_d,
                                         mask=q_mask, other=0.0).to(tl.float32)
                            logits += qv[:, None] * (cent * seg_norm_3)[None, :]
                elif SEG3_LEN > 0:
                    logits = _seg_partial_dot(
                        q_ptr, q_base, q_stride_d, q_mask,
                        kp_ptr, kp_base, codebook_ptr, k_norms_ptr,
                        b64, hkv64, offs_t64, mask_t,
                        kn_stride_b, kn_stride_h, kn_stride_t, kn_stride_s,
                        logits, byte_lut_dummy, lut_head_base,
                        SEG3_BW, SEG3_START, SEG3_LEN, SEG3_PACK_OFF,
                        SEG3_LUT_OFF, SEG3_NORM_IDX, SEG3_1BIT_GOFF,
                        BLOCK_Q, BLOCK_T, ONEBIT_FAST,
                    )

            # ---- Segment 4 ----
            if N_SEGMENTS >= 5:
                if K_NIBBLE:
                    if SEG4_BW <= 4 and SEG4_LEN >= MIN_SEG_FOR_DOT:
                        kn_off_4 = (b64 * kn_stride_b + hkv64 * kn_stride_h
                                    + offs_t64 * kn_stride_t + SEG4_NORM_IDX * kn_stride_s)
                        seg4_norm = tl.load(k_norms_ptr + kn_off_4, mask=mask_t, other=1.0)
                        logits += _seg_nibble_dot(
                            kp_ptr, kp_base, codebook_ptr, seg4_norm,
                            q_ptr, q_base, q_stride_d, q_mask, mask_t,
                            SEG4_START, SEG4_LEN, SEG4_LUT_OFF, BLOCK_Q, BLOCK_T)
                    elif SEG4_BW <= 4 and SEG4_LEN > 0:
                        kn_off_4 = (b64 * kn_stride_b + hkv64 * kn_stride_h
                                    + offs_t64 * kn_stride_t + SEG4_NORM_IDX * kn_stride_s)
                        seg_norm_4 = tl.load(k_norms_ptr + kn_off_4, mask=mask_t, other=1.0)
                        NIB_OFF_4: tl.constexpr = SEG4_START // 2
                        for _k in tl.static_range(0, (SEG4_LEN + 1) // 2):
                            byte = tl.load(kp_ptr + kp_base + NIB_OFF_4 + _k,
                                           mask=mask_t, other=0).to(tl.int32)
                            lo_code = byte & 0xF
                            hi_code = (byte >> 4) & 0xF
                            if _k * 2 < SEG4_LEN:
                                cent = tl.load(codebook_ptr + SEG4_LUT_OFF + lo_code).to(tl.float32)
                                qv = tl.load(q_ptr + q_base + (SEG4_START + _k * 2) * q_stride_d,
                                             mask=q_mask, other=0.0).to(tl.float32)
                                logits += qv[:, None] * (cent * seg_norm_4)[None, :]
                            if _k * 2 + 1 < SEG4_LEN:
                                cent = tl.load(codebook_ptr + SEG4_LUT_OFF + hi_code).to(tl.float32)
                                qv = tl.load(q_ptr + q_base + (SEG4_START + _k * 2 + 1) * q_stride_d,
                                             mask=q_mask, other=0.0).to(tl.float32)
                                logits += qv[:, None] * (cent * seg_norm_4)[None, :]
                    elif SEG4_LEN >= MIN_SEG_FOR_DOT:  # bw > 4: uint8 tensor-core
                        kn_off_4 = (b64 * kn_stride_b + hkv64 * kn_stride_h
                                    + offs_t64 * kn_stride_t + SEG4_NORM_IDX * kn_stride_s)
                        seg_norm_4 = tl.load(k_norms_ptr + kn_off_4, mask=mask_t, other=1.0)
                        logits += _seg_uint8_dot(
                            kp_ptr, kp_base, codebook_ptr, seg_norm_4,
                            q_ptr, q_base, q_stride_d, q_mask, mask_t,
                            SEG4_START, SEG4_LEN, SEG4_LUT_OFF, SEG4_PACK_OFF,
                            BLOCK_Q, BLOCK_T)
                    elif SEG4_LEN > 0:  # bw > 4: uint8 scalar
                        kn_off_4 = (b64 * kn_stride_b + hkv64 * kn_stride_h
                                    + offs_t64 * kn_stride_t + SEG4_NORM_IDX * kn_stride_s)
                        seg_norm_4 = tl.load(k_norms_ptr + kn_off_4, mask=mask_t, other=1.0)
                        for _k in tl.static_range(0, SEG4_LEN):
                            code = tl.load(kp_ptr + kp_base + SEG4_PACK_OFF + _k,
                                           mask=mask_t, other=0).to(tl.int32)
                            cent = tl.load(codebook_ptr + SEG4_LUT_OFF + code).to(tl.float32)
                            qv = tl.load(q_ptr + q_base + (SEG4_START + _k) * q_stride_d,
                                         mask=q_mask, other=0.0).to(tl.float32)
                            logits += qv[:, None] * (cent * seg_norm_4)[None, :]
                elif SEG4_LEN > 0:
                    logits = _seg_partial_dot(
                        q_ptr, q_base, q_stride_d, q_mask,
                        kp_ptr, kp_base, codebook_ptr, k_norms_ptr,
                        b64, hkv64, offs_t64, mask_t,
                        kn_stride_b, kn_stride_h, kn_stride_t, kn_stride_s,
                        logits, byte_lut_dummy, lut_head_base,
                        SEG4_BW, SEG4_START, SEG4_LEN, SEG4_PACK_OFF,
                        SEG4_LUT_OFF, SEG4_NORM_IDX, SEG4_1BIT_GOFF,
                        BLOCK_Q, BLOCK_T, ONEBIT_FAST,
                    )

            # ---- Segment 5 ----
            if N_SEGMENTS >= 6:
                if K_NIBBLE:
                    if SEG5_BW <= 4 and SEG5_LEN >= MIN_SEG_FOR_DOT:
                        kn_off_5 = (b64 * kn_stride_b + hkv64 * kn_stride_h
                                    + offs_t64 * kn_stride_t + SEG5_NORM_IDX * kn_stride_s)
                        seg5_norm = tl.load(k_norms_ptr + kn_off_5, mask=mask_t, other=1.0)
                        logits += _seg_nibble_dot(
                            kp_ptr, kp_base, codebook_ptr, seg5_norm,
                            q_ptr, q_base, q_stride_d, q_mask, mask_t,
                            SEG5_START, SEG5_LEN, SEG5_LUT_OFF, BLOCK_Q, BLOCK_T)
                    elif SEG5_BW <= 4 and SEG5_LEN > 0:
                        kn_off_5 = (b64 * kn_stride_b + hkv64 * kn_stride_h
                                    + offs_t64 * kn_stride_t + SEG5_NORM_IDX * kn_stride_s)
                        seg_norm_5 = tl.load(k_norms_ptr + kn_off_5, mask=mask_t, other=1.0)
                        NIB_OFF_5: tl.constexpr = SEG5_START // 2
                        for _k in tl.static_range(0, (SEG5_LEN + 1) // 2):
                            byte = tl.load(kp_ptr + kp_base + NIB_OFF_5 + _k,
                                           mask=mask_t, other=0).to(tl.int32)
                            lo_code = byte & 0xF
                            hi_code = (byte >> 4) & 0xF
                            if _k * 2 < SEG5_LEN:
                                cent = tl.load(codebook_ptr + SEG5_LUT_OFF + lo_code).to(tl.float32)
                                qv = tl.load(q_ptr + q_base + (SEG5_START + _k * 2) * q_stride_d,
                                             mask=q_mask, other=0.0).to(tl.float32)
                                logits += qv[:, None] * (cent * seg_norm_5)[None, :]
                            if _k * 2 + 1 < SEG5_LEN:
                                cent = tl.load(codebook_ptr + SEG5_LUT_OFF + hi_code).to(tl.float32)
                                qv = tl.load(q_ptr + q_base + (SEG5_START + _k * 2 + 1) * q_stride_d,
                                             mask=q_mask, other=0.0).to(tl.float32)
                                logits += qv[:, None] * (cent * seg_norm_5)[None, :]
                    elif SEG5_LEN >= MIN_SEG_FOR_DOT:  # bw > 4: uint8 tensor-core
                        kn_off_5 = (b64 * kn_stride_b + hkv64 * kn_stride_h
                                    + offs_t64 * kn_stride_t + SEG5_NORM_IDX * kn_stride_s)
                        seg_norm_5 = tl.load(k_norms_ptr + kn_off_5, mask=mask_t, other=1.0)
                        logits += _seg_uint8_dot(
                            kp_ptr, kp_base, codebook_ptr, seg_norm_5,
                            q_ptr, q_base, q_stride_d, q_mask, mask_t,
                            SEG5_START, SEG5_LEN, SEG5_LUT_OFF, SEG5_PACK_OFF,
                            BLOCK_Q, BLOCK_T)
                    elif SEG5_LEN > 0:  # bw > 4: uint8 scalar
                        kn_off_5 = (b64 * kn_stride_b + hkv64 * kn_stride_h
                                    + offs_t64 * kn_stride_t + SEG5_NORM_IDX * kn_stride_s)
                        seg_norm_5 = tl.load(k_norms_ptr + kn_off_5, mask=mask_t, other=1.0)
                        for _k in tl.static_range(0, SEG5_LEN):
                            code = tl.load(kp_ptr + kp_base + SEG5_PACK_OFF + _k,
                                           mask=mask_t, other=0).to(tl.int32)
                            cent = tl.load(codebook_ptr + SEG5_LUT_OFF + code).to(tl.float32)
                            qv = tl.load(q_ptr + q_base + (SEG5_START + _k) * q_stride_d,
                                         mask=q_mask, other=0.0).to(tl.float32)
                            logits += qv[:, None] * (cent * seg_norm_5)[None, :]
                elif SEG5_LEN > 0:
                    logits = _seg_partial_dot(
                        q_ptr, q_base, q_stride_d, q_mask,
                        kp_ptr, kp_base, codebook_ptr, k_norms_ptr,
                        b64, hkv64, offs_t64, mask_t,
                        kn_stride_b, kn_stride_h, kn_stride_t, kn_stride_s,
                        logits, byte_lut_dummy, lut_head_base,
                        SEG5_BW, SEG5_START, SEG5_LEN, SEG5_PACK_OFF,
                        SEG5_LUT_OFF, SEG5_NORM_IDX, SEG5_1BIT_GOFF,
                        BLOCK_Q, BLOCK_T, ONEBIT_FAST,
                    )

            # ---- Segment 6 ----
            if N_SEGMENTS >= 7:
                if K_NIBBLE:
                    if SEG6_BW <= 4 and SEG6_LEN >= MIN_SEG_FOR_DOT:
                        kn_off_6 = (b64 * kn_stride_b + hkv64 * kn_stride_h
                                    + offs_t64 * kn_stride_t + SEG6_NORM_IDX * kn_stride_s)
                        seg6_norm = tl.load(k_norms_ptr + kn_off_6, mask=mask_t, other=1.0)
                        logits += _seg_nibble_dot(
                            kp_ptr, kp_base, codebook_ptr, seg6_norm,
                            q_ptr, q_base, q_stride_d, q_mask, mask_t,
                            SEG6_START, SEG6_LEN, SEG6_LUT_OFF, BLOCK_Q, BLOCK_T)
                    elif SEG6_BW <= 4 and SEG6_LEN > 0:
                        kn_off_6 = (b64 * kn_stride_b + hkv64 * kn_stride_h
                                    + offs_t64 * kn_stride_t + SEG6_NORM_IDX * kn_stride_s)
                        seg_norm_6 = tl.load(k_norms_ptr + kn_off_6, mask=mask_t, other=1.0)
                        NIB_OFF_6: tl.constexpr = SEG6_START // 2
                        for _k in tl.static_range(0, (SEG6_LEN + 1) // 2):
                            byte = tl.load(kp_ptr + kp_base + NIB_OFF_6 + _k,
                                           mask=mask_t, other=0).to(tl.int32)
                            lo_code = byte & 0xF
                            hi_code = (byte >> 4) & 0xF
                            if _k * 2 < SEG6_LEN:
                                cent = tl.load(codebook_ptr + SEG6_LUT_OFF + lo_code).to(tl.float32)
                                qv = tl.load(q_ptr + q_base + (SEG6_START + _k * 2) * q_stride_d,
                                             mask=q_mask, other=0.0).to(tl.float32)
                                logits += qv[:, None] * (cent * seg_norm_6)[None, :]
                            if _k * 2 + 1 < SEG6_LEN:
                                cent = tl.load(codebook_ptr + SEG6_LUT_OFF + hi_code).to(tl.float32)
                                qv = tl.load(q_ptr + q_base + (SEG6_START + _k * 2 + 1) * q_stride_d,
                                             mask=q_mask, other=0.0).to(tl.float32)
                                logits += qv[:, None] * (cent * seg_norm_6)[None, :]
                    elif SEG6_LEN >= MIN_SEG_FOR_DOT:  # bw > 4: uint8 tensor-core
                        kn_off_6 = (b64 * kn_stride_b + hkv64 * kn_stride_h
                                    + offs_t64 * kn_stride_t + SEG6_NORM_IDX * kn_stride_s)
                        seg_norm_6 = tl.load(k_norms_ptr + kn_off_6, mask=mask_t, other=1.0)
                        logits += _seg_uint8_dot(
                            kp_ptr, kp_base, codebook_ptr, seg_norm_6,
                            q_ptr, q_base, q_stride_d, q_mask, mask_t,
                            SEG6_START, SEG6_LEN, SEG6_LUT_OFF, SEG6_PACK_OFF,
                            BLOCK_Q, BLOCK_T)
                    elif SEG6_LEN > 0:  # bw > 4: uint8 scalar
                        kn_off_6 = (b64 * kn_stride_b + hkv64 * kn_stride_h
                                    + offs_t64 * kn_stride_t + SEG6_NORM_IDX * kn_stride_s)
                        seg_norm_6 = tl.load(k_norms_ptr + kn_off_6, mask=mask_t, other=1.0)
                        for _k in tl.static_range(0, SEG6_LEN):
                            code = tl.load(kp_ptr + kp_base + SEG6_PACK_OFF + _k,
                                           mask=mask_t, other=0).to(tl.int32)
                            cent = tl.load(codebook_ptr + SEG6_LUT_OFF + code).to(tl.float32)
                            qv = tl.load(q_ptr + q_base + (SEG6_START + _k) * q_stride_d,
                                         mask=q_mask, other=0.0).to(tl.float32)
                            logits += qv[:, None] * (cent * seg_norm_6)[None, :]
                elif SEG6_LEN > 0:
                    logits = _seg_partial_dot(
                        q_ptr, q_base, q_stride_d, q_mask,
                        kp_ptr, kp_base, codebook_ptr, k_norms_ptr,
                        b64, hkv64, offs_t64, mask_t,
                        kn_stride_b, kn_stride_h, kn_stride_t, kn_stride_s,
                        logits, byte_lut_dummy, lut_head_base,
                        SEG6_BW, SEG6_START, SEG6_LEN, SEG6_PACK_OFF,
                        SEG6_LUT_OFF, SEG6_NORM_IDX, SEG6_1BIT_GOFF,
                        BLOCK_Q, BLOCK_T, ONEBIT_FAST,
                    )

            # ---- Segment 7 ----
            if N_SEGMENTS >= 8:
                if K_NIBBLE:
                    if SEG7_BW <= 4 and SEG7_LEN >= MIN_SEG_FOR_DOT:
                        kn_off_7 = (b64 * kn_stride_b + hkv64 * kn_stride_h
                                    + offs_t64 * kn_stride_t + SEG7_NORM_IDX * kn_stride_s)
                        seg7_norm = tl.load(k_norms_ptr + kn_off_7, mask=mask_t, other=1.0)
                        logits += _seg_nibble_dot(
                            kp_ptr, kp_base, codebook_ptr, seg7_norm,
                            q_ptr, q_base, q_stride_d, q_mask, mask_t,
                            SEG7_START, SEG7_LEN, SEG7_LUT_OFF, BLOCK_Q, BLOCK_T)
                    elif SEG7_BW <= 4 and SEG7_LEN > 0:
                        kn_off_7 = (b64 * kn_stride_b + hkv64 * kn_stride_h
                                    + offs_t64 * kn_stride_t + SEG7_NORM_IDX * kn_stride_s)
                        seg_norm_7 = tl.load(k_norms_ptr + kn_off_7, mask=mask_t, other=1.0)
                        NIB_OFF_7: tl.constexpr = SEG7_START // 2
                        for _k in tl.static_range(0, (SEG7_LEN + 1) // 2):
                            byte = tl.load(kp_ptr + kp_base + NIB_OFF_7 + _k,
                                           mask=mask_t, other=0).to(tl.int32)
                            lo_code = byte & 0xF
                            hi_code = (byte >> 4) & 0xF
                            if _k * 2 < SEG7_LEN:
                                cent = tl.load(codebook_ptr + SEG7_LUT_OFF + lo_code).to(tl.float32)
                                qv = tl.load(q_ptr + q_base + (SEG7_START + _k * 2) * q_stride_d,
                                             mask=q_mask, other=0.0).to(tl.float32)
                                logits += qv[:, None] * (cent * seg_norm_7)[None, :]
                            if _k * 2 + 1 < SEG7_LEN:
                                cent = tl.load(codebook_ptr + SEG7_LUT_OFF + hi_code).to(tl.float32)
                                qv = tl.load(q_ptr + q_base + (SEG7_START + _k * 2 + 1) * q_stride_d,
                                             mask=q_mask, other=0.0).to(tl.float32)
                                logits += qv[:, None] * (cent * seg_norm_7)[None, :]
                    elif SEG7_LEN >= MIN_SEG_FOR_DOT:  # bw > 4: uint8 tensor-core
                        kn_off_7 = (b64 * kn_stride_b + hkv64 * kn_stride_h
                                    + offs_t64 * kn_stride_t + SEG7_NORM_IDX * kn_stride_s)
                        seg_norm_7 = tl.load(k_norms_ptr + kn_off_7, mask=mask_t, other=1.0)
                        logits += _seg_uint8_dot(
                            kp_ptr, kp_base, codebook_ptr, seg_norm_7,
                            q_ptr, q_base, q_stride_d, q_mask, mask_t,
                            SEG7_START, SEG7_LEN, SEG7_LUT_OFF, SEG7_PACK_OFF,
                            BLOCK_Q, BLOCK_T)
                    elif SEG7_LEN > 0:  # bw > 4: uint8 scalar
                        kn_off_7 = (b64 * kn_stride_b + hkv64 * kn_stride_h
                                    + offs_t64 * kn_stride_t + SEG7_NORM_IDX * kn_stride_s)
                        seg_norm_7 = tl.load(k_norms_ptr + kn_off_7, mask=mask_t, other=1.0)
                        for _k in tl.static_range(0, SEG7_LEN):
                            code = tl.load(kp_ptr + kp_base + SEG7_PACK_OFF + _k,
                                           mask=mask_t, other=0).to(tl.int32)
                            cent = tl.load(codebook_ptr + SEG7_LUT_OFF + code).to(tl.float32)
                            qv = tl.load(q_ptr + q_base + (SEG7_START + _k) * q_stride_d,
                                         mask=q_mask, other=0.0).to(tl.float32)
                            logits += qv[:, None] * (cent * seg_norm_7)[None, :]
                elif SEG7_LEN > 0:
                    logits = _seg_partial_dot(
                        q_ptr, q_base, q_stride_d, q_mask,
                        kp_ptr, kp_base, codebook_ptr, k_norms_ptr,
                        b64, hkv64, offs_t64, mask_t,
                        kn_stride_b, kn_stride_h, kn_stride_t, kn_stride_s,
                        logits, byte_lut_dummy, lut_head_base,
                        SEG7_BW, SEG7_START, SEG7_LEN, SEG7_PACK_OFF,
                        SEG7_LUT_OFF, SEG7_NORM_IDX, SEG7_1BIT_GOFF,
                        BLOCK_Q, BLOCK_T, ONEBIT_FAST,
                    )

            logits = logits * SCALE

            # Causal + validity masking
            if IS_CAUSAL == 1:
                q_pos = q_pos_offset + q_start + tl.arange(0, BLOCK_Q)
                k_pos = t_start + tl.arange(0, BLOCK_T)
                causal_mask = q_pos[:, None] >= k_pos[None, :]
                logits = tl.where(causal_mask & mask_t[None, :] & q_mask[:, None],
                                  logits, -float('inf'))
            else:
                logits = tl.where(mask_t[None, :] & q_mask[:, None],
                                  logits, -float('inf'))

            # Online softmax
            m_new = tl.maximum(m_i, tl.max(logits, axis=1))
            alpha = tl.exp(m_i - m_new)
            p = tl.exp(logits - m_new[:, None])
            l_new = alpha * l_i + tl.sum(p, axis=1)
            acc = acc * alpha[:, None]
            m_i = m_new
            l_i = l_new

            # V accumulation
            do_v = True
            if V_SKIP_THRESH > 0.0:
                do_v = tl.max(p) >= V_SKIP_THRESH

            if do_v:
                if V_NIBBLE:
                    # Fast nibble-4 V decode:
                    # load (BLOCK_T, D//2) bytes, split into lo/hi nibbles
                    # → (BLOCK_T, D) codes via codebook + interleave.
                    HALF_D: tl.constexpr = D // 2
                    offs_half = tl.arange(0, HALF_D)
                    vn_base = (b64 * vc_stride_b + hkv64 * vc_stride_h
                               + offs_t64 * vc_stride_t)
                    packed_bytes = tl.load(
                        v_codes_ptr + vn_base[:, None] + offs_half[None, :].to(tl.int64),
                        mask=mask_t[:, None], other=0).to(tl.int32)
                    v_lo = packed_bytes & 0xF
                    v_hi = (packed_bytes >> 4) & 0xF
                    c_lo = tl.load(v_lut_ptr + v_lo).to(tl.float16)
                    c_hi = tl.load(v_lut_ptr + v_hi).to(tl.float16)
                    v_n = tl.load(
                        v_norms_ptr + b64 * vn_stride_b + hkv64 * vn_stride_h
                        + offs_t64 * vn_stride_t,
                        mask=mask_t, other=1.0).to(tl.float16)
                    v_even = c_lo * v_n[:, None]
                    v_odd  = c_hi * v_n[:, None]
                    v_tile = tl.interleave(v_even, v_odd)
                else:
                    v_codes = _load_v_codes(
                        v_codes_ptr, v_unpack_ptr,
                        b64, hkv64, offs_t64, offs_d, mask_t,
                        vc_stride_b, vc_stride_h, vc_stride_t, vc_stride_d,
                        vu_stride, V_BITS)
                    v_centroids = tl.load(v_lut_ptr + v_codes).to(tl.float16)

                    v_norms = _unpack_norms_mixed(
                        v_norms_ptr, v_norm_scale_ptr,
                        b64, hkv64, offs_t64, v_block_ids, mask_t,
                        vn_stride_b, vn_stride_h, vn_stride_t, vn_stride_blk,
                        vns_stride_b, vns_stride_h, V_NORM_BITS).to(tl.float16)
                    v_tile = v_centroids * v_norms

                p_fp16 = p.to(tl.float16)
                acc = acc + tl.dot(p_fp16, v_tile)

        # Final output
        final_out = acc / l_i[:, None]

        # V un-rotation
        if V_UNROT == 1:
            rot_cols = tl.arange(0, D)
            rot_block = tl.load(
                v_rot_ptr + offs_d[:, None] * D + rot_cols[None, :],
            ).to(tl.float32)
            final_out = tl.dot(final_out, rot_block, allow_tf32=True)

        # Store output
        o_offset = (b64 * o_stride_b + hq64 * o_stride_h
                    + (q_start + offs_q.to(tl.int64))[:, None] * o_stride_t
                    + offs_d[None, :] * o_stride_d)
        tl.store(out_ptr + o_offset,
                 final_out.to(tl.float16), mask=q_mask[:, None])


def fused_blockgtq_prefill_attention(
    q: torch.Tensor,              # (B, n_q_heads, S_q, D) fp16
    k_packed: torch.Tensor,       # (B, n_kv_heads, T, total_packed_bytes) uint8
    k_norms: torch.Tensor,        # (B, n_kv_heads, T, n_groups) fp16
    codebook_flat: torch.Tensor,  # (total_entries,) fp16
    lut_offset: torch.Tensor,     # (D,) int32
    norm_group: torch.Tensor,     # (D,) int32
    segments: list,
    v_codes: torch.Tensor,
    v_norms: torch.Tensor,
    v_lut: torch.Tensor,
    scale: float = None,
    block_q: int = 64,
    block_t: int = 64,
    v_skip_thresh: float = 0.0,
    v_norm_scale: torch.Tensor = None,
    v_bits: int = 0,
    v_rotation: torch.Tensor = None,
    is_causal: bool = True,
    q_pos_offset: int = 0,
    k_nibble: bool = False,
    v_nibble: bool = False,
    min_seg_for_dot: int = 16,
) -> torch.Tensor:
    """FlashAttention-2 style prefill over a segmented packed KV cache.

    Used by :func:`blockgtq.prefill.layer_major_prefill` to populate the
    packed cache in one big call per layer for long-context prefill.

    Args:
        q: (B, n_q_heads, S_q, D) query tokens (fp16).
        k_packed, k_norms, codebook_flat: packed K cache and metadata.
        segments: bit-segment definitions (from
            :func:`build_segments_from_pack_meta`).
        v_codes, v_norms, v_lut: packed V cache and codebook.
        scale: attention scale (default 1/sqrt(D)).
        block_q, block_t: kernel tile sizes.
        v_bits: V packing bit width (0 = raw uint8).
        v_rotation: (D, D) fp32 un-rotation matrix or None.
        is_causal: apply causal mask.
        k_nibble: K cache was encoded with ``compress_k_mixed_batched``
            (nibble-4 for bw <= 4, uint8 for bw > 4).
        v_nibble: V cache was encoded with ``compress_v_nibble4_batched``.

    Returns: (B, n_q_heads, S_q, D) fp16.
    """
    assert HAS_TRITON
    assert q.dim() == 4
    assert len(segments) <= 8

    v_norm_bits = 4 if v_norms.dtype == torch.uint8 else 16

    B, n_q_heads, S_q, D = q.shape
    _, n_kv_heads, T, _ = k_packed.shape
    gqa_ratio = n_q_heads // n_kv_heads

    if v_norms.dim() == 3:
        v_norms = v_norms.unsqueeze(-1)
    if scale is None:
        scale = 1.0 / math.sqrt(D)

    v_nb_raw = v_norms.shape[-1]
    v_nb = v_nb_raw * 2 if v_norm_bits == 4 else v_nb_raw
    v_quant_block = D // v_nb

    if v_norm_scale is None:
        v_norm_scale = v_norms.new_empty(1, 1)

    _v_bits = int(v_bits)
    if _v_bits > 0:
        from blockgtq.v_packing import build_v_unpack_tables
        blo, bhi, slo, shi, vmask = build_v_unpack_tables(_v_bits, D, device=q.device)
        v_unpack = torch.stack([blo, bhi, slo, shi, vmask], dim=0).contiguous()
    else:
        v_unpack = torch.zeros(5, D, dtype=torch.int32, device=q.device)

    _v_unrot = 1 if v_rotation is not None else 0
    if v_rotation is not None:
        v_rot_tensor = v_rotation.contiguous().to(torch.float32)
    else:
        v_rot_tensor = torch.empty(1, 1, dtype=torch.float32, device=q.device)

    seg_pad = list(segments) + [(0, 0, 0, 0, 0, 0)] * (8 - len(segments))
    seg_1bit_goffs = [0] * 8

    # When the K cache was encoded with mixed nibble-4 packing, bw>4 segments
    # need their pack_offset recomputed to point past the nibble byte block.
    if k_nibble:
        nopack_start = D
        for bw, start, length, pack_off, lut_off, norm_idx in segments:
            if bw > 4 and length > 0:
                nopack_start = min(nopack_start, start)
        nib_bytes = nopack_start // 2
        for i in range(len(seg_pad)):
            bw, start, length, pack_off, lut_off, norm_idx = seg_pad[i]
            if bw > 4 and length > 0:
                new_pack_off = nib_bytes + (start - nopack_start)
                seg_pad[i] = (bw, start, length, new_pack_off, lut_off, norm_idx)

    out = torch.empty(B, n_q_heads, S_q, D, dtype=torch.float16, device=q.device)
    q = q.contiguous()

    q_blocks = (S_q + block_q - 1) // block_q
    grid = (B * n_q_heads * q_blocks,)

    _onebit_fast = 4  # masked_sum: no byte_lut materialisation needed for prefill.

    _packed_prefill_kernel[grid](
        q, k_packed, k_norms, v_codes, v_norms, v_lut,
        v_norm_scale, v_unpack, v_rot_tensor,
        out, codebook_flat,
        lut_offset, norm_group,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k_packed.stride(0), k_packed.stride(1), k_packed.stride(2),
        k_norms.stride(0), k_norms.stride(1), k_norms.stride(2), k_norms.stride(3),
        v_codes.stride(0), v_codes.stride(1), v_codes.stride(2),
        v_codes.stride(3) if v_codes.dim() >= 4 else 1,
        v_norms.stride(0), v_norms.stride(1), v_norms.stride(2), v_norms.stride(3),
        v_norm_scale.stride(0),
        v_norm_scale.stride(1) if v_norm_scale.dim() >= 2 else 0,
        v_unpack.stride(0),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        S_q, T, n_q_heads, n_kv_heads,
        SEG0_BW=seg_pad[0][0], SEG0_START=seg_pad[0][1], SEG0_LEN=seg_pad[0][2],
        SEG0_PACK_OFF=seg_pad[0][3], SEG0_LUT_OFF=seg_pad[0][4], SEG0_NORM_IDX=seg_pad[0][5],
        SEG0_1BIT_GOFF=seg_1bit_goffs[0],
        SEG1_BW=seg_pad[1][0], SEG1_START=seg_pad[1][1], SEG1_LEN=seg_pad[1][2],
        SEG1_PACK_OFF=seg_pad[1][3], SEG1_LUT_OFF=seg_pad[1][4], SEG1_NORM_IDX=seg_pad[1][5],
        SEG1_1BIT_GOFF=seg_1bit_goffs[1],
        SEG2_BW=seg_pad[2][0], SEG2_START=seg_pad[2][1], SEG2_LEN=seg_pad[2][2],
        SEG2_PACK_OFF=seg_pad[2][3], SEG2_LUT_OFF=seg_pad[2][4], SEG2_NORM_IDX=seg_pad[2][5],
        SEG2_1BIT_GOFF=seg_1bit_goffs[2],
        SEG3_BW=seg_pad[3][0], SEG3_START=seg_pad[3][1], SEG3_LEN=seg_pad[3][2],
        SEG3_PACK_OFF=seg_pad[3][3], SEG3_LUT_OFF=seg_pad[3][4], SEG3_NORM_IDX=seg_pad[3][5],
        SEG3_1BIT_GOFF=seg_1bit_goffs[3],
        SEG4_BW=seg_pad[4][0], SEG4_START=seg_pad[4][1], SEG4_LEN=seg_pad[4][2],
        SEG4_PACK_OFF=seg_pad[4][3], SEG4_LUT_OFF=seg_pad[4][4], SEG4_NORM_IDX=seg_pad[4][5],
        SEG4_1BIT_GOFF=seg_1bit_goffs[4],
        SEG5_BW=seg_pad[5][0], SEG5_START=seg_pad[5][1], SEG5_LEN=seg_pad[5][2],
        SEG5_PACK_OFF=seg_pad[5][3], SEG5_LUT_OFF=seg_pad[5][4], SEG5_NORM_IDX=seg_pad[5][5],
        SEG5_1BIT_GOFF=seg_1bit_goffs[5],
        SEG6_BW=seg_pad[6][0], SEG6_START=seg_pad[6][1], SEG6_LEN=seg_pad[6][2],
        SEG6_PACK_OFF=seg_pad[6][3], SEG6_LUT_OFF=seg_pad[6][4], SEG6_NORM_IDX=seg_pad[6][5],
        SEG6_1BIT_GOFF=seg_1bit_goffs[6],
        SEG7_BW=seg_pad[7][0], SEG7_START=seg_pad[7][1], SEG7_LEN=seg_pad[7][2],
        SEG7_PACK_OFF=seg_pad[7][3], SEG7_LUT_OFF=seg_pad[7][4], SEG7_NORM_IDX=seg_pad[7][5],
        SEG7_1BIT_GOFF=seg_1bit_goffs[7],
        N_SEGMENTS=len(segments),
        TOTAL_1BIT_GROUPS=max(1, 1),
        D=D, BLOCK_Q=block_q, GQA_RATIO=gqa_ratio,
        BLOCK_T=block_t, BLOCK_D=D, SCALE=scale,
        V_SKIP_THRESH=v_skip_thresh,
        V_QUANT_BLOCK=v_quant_block,
        V_NORM_BITS=v_norm_bits,
        ONEBIT_FAST=_onebit_fast,
        V_BITS=_v_bits,
        V_UNROT=_v_unrot,
        IS_CAUSAL=1 if is_causal else 0,
        q_pos_offset=q_pos_offset,
        K_NIBBLE=1 if k_nibble else 0,
        V_NIBBLE=1 if v_nibble else 0,
        MIN_SEG_FOR_DOT=min_seg_for_dot,
    )
    return out


def build_segments_from_pack_meta(pack_meta, codebook_flat, pos_centroids):
    """Build segment tuples for the decode kernel from pack_meta.

    Returns list of (bw, dim_start, dim_len, pack_off, lut_off, norm_idx).
    """
    segments = []
    lut_offset = 0
    norm_idx = 0

    for bw in range(1, 9):
        if bw not in pack_meta:
            continue
        m = pack_meta[bw]
        segments.append((
            bw,
            m['start'],
            m['length'],
            m['pack_offset'],
            lut_offset,      # will be filled properly by caller
            norm_idx,
        ))
        lut_offset += (1 << bw)
        norm_idx += 1

    return segments
