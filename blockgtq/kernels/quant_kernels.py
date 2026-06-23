"""Triton quantization kernels for Block-GTQ.

Internal backend for ``blockgtq.quantizer`` -- not part of the public API
(use the high-level ``build_quantizers`` / ``BlockGTQPipeline``).

Compress / decompress (quantize) kernels for TQ-MSE and Block-GTQ -- not the
fused dequant + attention kernel, which lives in
``kernels/fused_packed_attention.py``.

Consolidates three components that were developed in stages:
  * LUT-based quantize/dequantize -- precomputed look-up tables make each
    quantize-dequantize O(1) per element (replacing per-element searchsorted).
  * tensor-core rotation (``tl.dot``) + a shared 8 KB codebook LUT.
  * compress-only path emitting uint8 codes + group norms at insert time
    (dequant happens later in the fused attention kernel), plus the
    mixed-bit packing helpers.
"""

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
# Shared LUT construction
# ===========================================================================

def build_shared_lut(pos_centroids: torch.Tensor, pos_boundaries: torch.Tensor,
                     n_bins: int = 256
                     ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build shared codebook LUT for Block-GTQ.

    Identifies positions that share the same codebook (same bit-width = same
    number of centroids = same centroid values after normalization). Groups them
    and builds a single shared LUT per group.

    Args:
        pos_centroids: (HD, max_nc) per-position centroids (padded with zeros)
        pos_boundaries: (HD, max_nb) per-position boundaries (padded with inf)
        n_bins: number of uniform bins

    Returns:
        shared_lut: (n_codebooks, n_bins) float32 - shared codebook LUTs
        pos_to_codebook: (HD,) int32 - maps each position to its codebook index
        offsets: (HD,) float32 - per-position discretization offset
        inv_scales: (HD,) float32 - per-position discretization 1/bin_width
    """
    HD = pos_centroids.shape[0]
    device = pos_centroids.device

    offsets = torch.zeros(HD, device=device, dtype=torch.float32)
    inv_scales = torch.zeros(HD, device=device, dtype=torch.float32)

    # Step 1: identify unique codebooks by number of valid centroids
    # and build per-position LUT parameters
    n_valid_per_pos = []
    for j in range(HD):
        bounds_j = pos_boundaries[j]
        valid_mask = bounds_j < 1e30
        n_valid_bounds = valid_mask.sum().item()
        n_valid_cents = int(n_valid_bounds) + 1
        if n_valid_cents <= 1:
            n_valid_cents = 2
        n_valid_per_pos.append(n_valid_cents)

        cents_j = pos_centroids[j, :n_valid_cents]
        c_min = cents_j.min().item()
        c_max = cents_j.max().item()
        spacing = (c_max - c_min) / max(n_valid_cents - 1, 1)
        margin = spacing * 0.5
        range_min = c_min - margin
        range_max = c_max + margin
        bin_width = (range_max - range_min) / n_bins
        offsets[j] = range_min
        inv_scales[j] = 1.0 / bin_width

    # Step 2: group positions by their centroid signature
    # Positions with identical centroid tensors share a codebook
    codebook_map = {}  # signature -> codebook_index
    pos_to_codebook = torch.zeros(HD, dtype=torch.int32, device=device)
    codebook_representatives = []  # list of (j, n_valid_cents) for each unique codebook

    for j in range(HD):
        nc = n_valid_per_pos[j]
        cents_j = pos_centroids[j, :nc]
        # Create a hashable signature from the centroid values
        sig = (nc, tuple(cents_j.cpu().tolist()))
        if sig not in codebook_map:
            codebook_map[sig] = len(codebook_representatives)
            codebook_representatives.append((j, nc))
        pos_to_codebook[j] = codebook_map[sig]

    n_codebooks = len(codebook_representatives)

    # Step 3: build shared LUT for each unique codebook
    shared_lut = torch.zeros(n_codebooks, n_bins, device=device, dtype=torch.float32)

    for cb_idx, (repr_j, nc) in enumerate(codebook_representatives):
        cents = pos_centroids[repr_j, :nc]
        c_min = cents.min().item()
        c_max = cents.max().item()
        spacing = (c_max - c_min) / max(nc - 1, 1)
        margin = spacing * 0.5
        range_min = c_min - margin
        range_max = c_max + margin
        bin_width = (range_max - range_min) / n_bins

        bin_centers = torch.linspace(range_min + bin_width / 2,
                                     range_max - bin_width / 2,
                                     n_bins, device=device, dtype=torch.float32)

        bounds = pos_boundaries[repr_j]
        valid_mask = bounds < 1e30
        n_valid_bounds = valid_mask.sum().item()

        if n_valid_bounds > 0:
            valid_bounds = bounds[:n_valid_bounds]
            indices = torch.searchsorted(valid_bounds, bin_centers)
            indices = indices.clamp(max=nc - 1)
            shared_lut[cb_idx] = cents[indices]
        else:
            mid = (cents[0] + cents[1]) / 2.0
            shared_lut[cb_idx] = torch.where(bin_centers > mid, cents[1], cents[0])

    return shared_lut, pos_to_codebook, offsets, inv_scales


def next_power_of_2(x: int) -> int:
    """Return the smallest power of 2 >= x."""
    if x <= 0:
        return 1
    p = 1
    while p < x:
        p *= 2
    return p


# ===========================================================================
# Triton Kernels
# ===========================================================================

if HAS_TRITON:

    # -------------------------------------------------------------------
    # Roundtrip Block-GTQ V4 kernel: tl.dot rotation + shared LUT
    # Uses full HD x HD rotation via tl.dot (HD must be power of 2)
    # -------------------------------------------------------------------

    @triton.jit
    def _roundtrip_block_gtq_v4_kernel(
        X_ptr, OUT_ptr, SCRATCH_ptr,
        PERM_ptr,               # (HD,) int64 -- forward permutation
        BLOCK_ROT_T_ptr,        # (HD, HD) -- full block-diagonal rotation^T
        ROT_UNPERM_ptr,         # (HD, HD) -- fused inv-rotate+unpermute = R[:, inv_perm]
        SHARED_LUT_ptr,         # (N_CODEBOOKS, N_BINS) shared codebook LUT
        POS_TO_CB_ptr,          # (HD,) int32 -- maps position to codebook index
        LUT_OFFSETS_ptr,        # (HD,) per-position offsets
        LUT_INV_SCALES_ptr,     # (HD,) per-position inv_scales
        GROUP_OF_ptr,           # (HD,) int64 -- maps position to norm group
        stride_xn,              # stride of X/OUT along N dimension
        stride_scratch_n,       # stride of scratch along N dimension
        stride_slut,            # stride of shared_lut along codebook dimension
        N,
        HD: tl.constexpr,
        N_GROUPS: tl.constexpr,
        N_BINS: tl.constexpr,
        N_CODEBOOKS: tl.constexpr,
        BLOCK_N: tl.constexpr,
        SINGLE_GROUP: tl.constexpr,
    ):
        """Roundtrip Block-GTQ V4: tl.dot rotation + shared LUT quantize-dequantize.

        Uses full HD x HD tl.dot for both forward and inverse rotation
        (tensor cores). Shared LUT reduces working set from 128 KB to ~8 KB.

        Hot path:
        1. Permute
        2. Per-group norms + normalize
        3. Forward rotation via tl.dot(x_normed, R^T)
        4. Shared LUT quantize-dequantize (8 KB fits L1)
        5. Norm correction
        6. Inverse rotation + unpermute via tl.dot(y_hat, ROT_UNPERM)
        """
        pid = tl.program_id(0)
        row_start = pid * BLOCK_N
        rows = row_start + tl.arange(0, BLOCK_N)
        cols = tl.arange(0, HD)
        mask_n = rows < N

        # ---- Step 1: Load + permute ----
        perm = tl.load(PERM_ptr + cols)
        x = tl.load(X_ptr + rows[:, None] * stride_xn + perm[None, :],
                     mask=mask_n[:, None], other=0.0).to(tl.float32)

        # ---- Step 2: Per-group norms ----
        if SINGLE_GROUP == 1:
            norm_sq = tl.sum(x * x, axis=1)
            norms = tl.sqrt(norm_sq + 1e-30)
            safe_norms = tl.where(norms > 1e-10, norms, 1.0)
            x_normed = x / safe_norms[:, None]
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

        # ---- Step 3: Forward rotation via tl.dot (tensor core) ----
        # y = x_normed @ BLOCK_ROT_T  -- single tensor core operation
        ROT_T = tl.load(
            BLOCK_ROT_T_ptr + cols[:, None] * HD + tl.arange(0, HD)[None, :]
        ).to(tl.float32)
        y = tl.dot(x_normed, ROT_T, allow_tf32=True)

        # ---- Step 4: Shared LUT quantize-dequantize ----
        offsets_lut = tl.load(LUT_OFFSETS_ptr + cols).to(tl.float32)
        inv_scales_lut = tl.load(LUT_INV_SCALES_ptr + cols).to(tl.float32)

        bin_idx = (y - offsets_lut[None, :]) * inv_scales_lut[None, :]
        bin_idx = tl.minimum(tl.maximum(bin_idx, 0.0), (N_BINS - 1) + 0.0)
        bin_idx_int = bin_idx.to(tl.int32)

        # Two-step gather from shared LUT:
        pos_cb = tl.load(POS_TO_CB_ptr + cols).to(tl.int32)  # (HD,)
        lut_addr = pos_cb[None, :] * stride_slut + bin_idx_int  # (BLOCK_N, HD)
        y_hat = tl.load(SHARED_LUT_ptr + lut_addr)

        # ---- Step 5: Norm correction + rescale ----
        if SINGLE_GROUP == 1:
            yh_norm = tl.sqrt(tl.sum(y_hat * y_hat, axis=1) + 1e-30)
            yh_safe = tl.where(yh_norm > 1e-10, yh_norm, 1.0)
            y_hat = y_hat / yh_safe[:, None]
            y_hat = y_hat * safe_norms[:, None]
        else:
            yh_sq = y_hat * y_hat
            yh_per_pos_norm_sq = tl.zeros([BLOCK_N, HD], dtype=tl.float32)
            for g in tl.static_range(0, N_GROUPS):
                g_mask = tl.where(group_of == g, 1.0, 0.0)
                g_sq = yh_sq * g_mask[None, :]
                g_norm_sq = tl.sum(g_sq, axis=1)
                yh_per_pos_norm_sq += tl.where(group_of[None, :] == g,
                                               g_norm_sq[:, None], 0.0)
            yh_per_pos_norms = tl.sqrt(yh_per_pos_norm_sq + 1e-30)
            yh_per_pos_safe = tl.where(yh_per_pos_norms > 1e-10, yh_per_pos_norms, 1.0)
            y_hat = y_hat / yh_per_pos_safe * per_pos_safe

        # ---- Step 6: Fused inverse rotation + unpermute via tl.dot ----
        # result = y_hat @ ROT_UNPERM where ROT_UNPERM = R[:, inv_perm]
        RU = tl.load(
            ROT_UNPERM_ptr + cols[:, None] * HD + tl.arange(0, HD)[None, :]
        ).to(tl.float32)
        result = tl.dot(y_hat, RU, allow_tf32=True)

        tl.store(OUT_ptr + rows[:, None] * stride_xn + cols[None, :],
                 result, mask=mask_n[:, None])


# ===========================================================================
# Code LUT construction
# ===========================================================================

def build_code_lut(pos_centroids: torch.Tensor, pos_boundaries: torch.Tensor,
                   n_bins: int = 256
                   ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build code LUT for compress-only output.

    Like build_shared_lut but stores centroid INDEX (uint8) instead of value.
    Output codes are compatible with the fused attention kernel's k_codes format.

    Returns:
        code_lut: (n_codebooks, n_bins) uint8 - maps bin_idx → centroid index
        pos_to_codebook: (HD,) int32 - maps position to codebook index
        offsets: (HD,) float32 - per-position discretization offset
        inv_scales: (HD,) float32 - per-position discretization 1/bin_width
    """
    HD = pos_centroids.shape[0]
    device = pos_centroids.device

    offsets = torch.zeros(HD, device=device, dtype=torch.float32)
    inv_scales = torch.zeros(HD, device=device, dtype=torch.float32)

    n_valid_per_pos = []
    for j in range(HD):
        bounds_j = pos_boundaries[j]
        valid_mask = bounds_j < 1e30
        n_valid_bounds = valid_mask.sum().item()
        n_valid_cents = max(int(n_valid_bounds) + 1, 2)
        n_valid_per_pos.append(n_valid_cents)

        cents_j = pos_centroids[j, :n_valid_cents]
        c_min = cents_j.min().item()
        c_max = cents_j.max().item()
        spacing = (c_max - c_min) / max(n_valid_cents - 1, 1)
        margin = spacing * 0.5
        range_min = c_min - margin
        range_max = c_max + margin
        bin_width = (range_max - range_min) / n_bins
        offsets[j] = range_min
        inv_scales[j] = 1.0 / bin_width

    # Group positions by centroid signature
    codebook_map = {}
    pos_to_codebook = torch.zeros(HD, dtype=torch.int32, device=device)
    codebook_representatives = []

    for j in range(HD):
        nc = n_valid_per_pos[j]
        cents_j = pos_centroids[j, :nc]
        sig = (nc, tuple(cents_j.cpu().tolist()))
        if sig not in codebook_map:
            codebook_map[sig] = len(codebook_representatives)
            codebook_representatives.append((j, nc))
        pos_to_codebook[j] = codebook_map[sig]

    n_codebooks = len(codebook_representatives)

    # Build code LUT: bin_idx → centroid index (uint8)
    code_lut = torch.zeros(n_codebooks, n_bins, device=device, dtype=torch.uint8)

    for cb_idx, (repr_j, nc) in enumerate(codebook_representatives):
        cents = pos_centroids[repr_j, :nc]
        c_min = cents.min().item()
        c_max = cents.max().item()
        spacing = (c_max - c_min) / max(nc - 1, 1)
        margin = spacing * 0.5
        range_min = c_min - margin
        range_max = c_max + margin
        bin_width = (range_max - range_min) / n_bins

        bin_centers = torch.linspace(range_min + bin_width / 2,
                                     range_max - bin_width / 2,
                                     n_bins, device=device, dtype=torch.float32)

        bounds = pos_boundaries[repr_j]
        valid_mask = bounds < 1e30
        n_valid_bounds = valid_mask.sum().item()

        if n_valid_bounds > 0:
            valid_bounds = bounds[:n_valid_bounds]
            indices = torch.searchsorted(valid_bounds, bin_centers)
            indices = indices.clamp(max=nc - 1)
        else:
            mid = (cents[0] + cents[1]) / 2.0
            indices = torch.where(bin_centers > mid,
                                  torch.ones_like(bin_centers, dtype=torch.long),
                                  torch.zeros_like(bin_centers, dtype=torch.long))

        code_lut[cb_idx] = indices.to(torch.uint8)

    return code_lut, pos_to_codebook, offsets, inv_scales


# ===========================================================================
# Bit-packing utilities (CPU/GPU post-processing — legacy reference)
# ===========================================================================

def pack_codes_mixed_bit(codes: torch.Tensor, bit_allocation: torch.Tensor,
                         head_perm: torch.Tensor) -> tuple[torch.Tensor, int]:
    """Pack uint8 codes into bit-tight format (CPU, reference implementation).

    codes: (N, HD) uint8 in permuted order (dims sorted by bit-width)
    bit_allocation: per-frequency-pair bit widths
    head_perm: permutation that was applied

    Returns:
        packed: (N, packed_bytes) uint8 - bit-packed codes
        total_bits: total bits per vector
    """
    N, HD = codes.shape
    device = codes.device

    bits_per_dim = torch.cat([bit_allocation, bit_allocation])  # (HD,) split-half RoPE
    permuted_bits = bits_per_dim[head_perm]  # bits per dim in permuted order

    total_bits = int(permuted_bits.sum().item())
    packed_bytes = (total_bits + 7) // 8

    # Pack on CPU for simplicity (this is not the hot path)
    codes_cpu = codes.cpu().numpy().astype(np.uint64)
    permuted_bits_cpu = permuted_bits.cpu().numpy().astype(np.int32)
    packed = np.zeros((N, packed_bytes), dtype=np.uint8)

    for i in range(N):
        bit_pos = 0
        for d in range(HD):
            bw = permuted_bits_cpu[d]
            code_val = codes_cpu[i, d] & ((1 << bw) - 1)
            # Pack into byte array, LSB first
            byte_idx = bit_pos >> 3
            bit_offset = bit_pos & 7
            # May span 2 bytes
            packed[i, byte_idx] |= (code_val << bit_offset) & 0xFF
            if bit_offset + bw > 8 and byte_idx + 1 < packed_bytes:
                packed[i, byte_idx + 1] |= (code_val >> (8 - bit_offset)) & 0xFF
            if bit_offset + bw > 16 and byte_idx + 2 < packed_bytes:
                packed[i, byte_idx + 2] |= (code_val >> (16 - bit_offset)) & 0xFF
            bit_pos += bw

    return torch.from_numpy(packed).to(device), total_bits


def pack_codes_mixed_bit_gpu(codes: torch.Tensor, bit_allocation: torch.Tensor,
                              head_perm: torch.Tensor) -> tuple[torch.Tensor, int]:
    """GPU bit-packing using vectorized shifts (per-dim loop, to be replaced)."""
    N, HD = codes.shape
    device = codes.device

    bits_per_dim = torch.cat([bit_allocation, bit_allocation]).to(device)  # (HD,) split-half RoPE
    permuted_bits = bits_per_dim[head_perm]  # bits in permuted order

    total_bits = int(permuted_bits.sum().item())
    packed_bytes = (total_bits + 7) // 8

    # Precompute bit positions for each dim
    bit_positions = torch.zeros(HD, dtype=torch.int32, device=device)
    bit_positions[1:] = torch.cumsum(permuted_bits[:-1].to(torch.int32), dim=0)

    # Mask codes to their bit width
    masks = ((1 << permuted_bits) - 1).to(torch.int64)
    codes_masked = (codes.to(torch.int64) & masks[None, :])

    # For each output byte, accumulate contributions from all dims
    packed = torch.zeros(N, packed_bytes, dtype=torch.uint8, device=device)

    for d in range(HD):
        bw = permuted_bits[d].item()
        bp = bit_positions[d].item()
        code_val = codes_masked[:, d]  # (N,)

        byte_idx = bp >> 3
        bit_offset = bp & 7

        # First byte
        contrib = ((code_val << bit_offset) & 0xFF).to(torch.uint8)
        packed[:, byte_idx] |= contrib

        # Second byte (if spanning)
        if bit_offset + bw > 8 and byte_idx + 1 < packed_bytes:
            contrib2 = ((code_val >> (8 - bit_offset)) & 0xFF).to(torch.uint8)
            packed[:, byte_idx + 1] |= contrib2

        # Third byte (rare, only for 8-bit codes crossing byte boundary)
        if bit_offset + bw > 16 and byte_idx + 2 < packed_bytes:
            contrib3 = ((code_val >> (16 - bit_offset)) & 0xFF).to(torch.uint8)
            packed[:, byte_idx + 2] |= contrib3

    return packed, total_bits


# ===========================================================================
# Triton Kernel
# ===========================================================================

if HAS_TRITON:

    @triton.jit
    def _compress_only_kernel(
        X_ptr, CODES_ptr, NORMS_ptr,
        PERM_ptr,               # (HD,) int64 — forward permutation
        BLOCK_ROT_T_ptr,        # (HD, HD) — block-diagonal rotation^T
        CODE_LUT_ptr,           # (N_CODEBOOKS, N_BINS) uint8 — code LUT
        POS_TO_CB_ptr,          # (HD,) int32 — maps position to codebook
        LUT_OFFSETS_ptr,        # (HD,) per-position offsets
        LUT_INV_SCALES_ptr,     # (HD,) per-position inv_scales
        GROUP_OF_ptr,           # (HD,) int64 — maps position to norm group
        stride_xn,              # stride of X along N dimension
        stride_codes_n,         # stride of CODES along N dimension
        stride_norms_n,         # stride of NORMS along N dimension
        stride_clut,            # stride of CODE_LUT along codebook dimension
        N,
        HD: tl.constexpr,
        N_GROUPS: tl.constexpr,
        N_BINS: tl.constexpr,
        N_CODEBOOKS: tl.constexpr,
        BLOCK_N: tl.constexpr,
        SINGLE_GROUP: tl.constexpr,
    ):
        """Compress-only Block-GTQ: outputs uint8 codes + group norms.

        Steps:
        1. Load + permute (gather via head_perm)
        2. Per-group norms + normalize
        3. Forward rotation via tl.dot (tensor core)
        4. Discretize to 256 bins
        5. Code LUT: bin_idx → centroid code (uint8)
        6. Store codes + norms
        """
        pid = tl.program_id(0)
        row_start = pid * BLOCK_N
        rows = row_start + tl.arange(0, BLOCK_N)
        cols = tl.arange(0, HD)
        mask_n = rows < N

        # ---- Step 1: Load + permute ----
        perm = tl.load(PERM_ptr + cols)
        x = tl.load(X_ptr + rows[:, None] * stride_xn + perm[None, :],
                     mask=mask_n[:, None], other=0.0).to(tl.float32)

        # ---- Step 2: Per-group norms ----
        if SINGLE_GROUP == 1:
            norm_sq = tl.sum(x * x, axis=1)
            norms = tl.sqrt(norm_sq + 1e-30)
            safe_norms = tl.where(norms > 1e-10, norms, 1.0)
            x_normed = x / safe_norms[:, None]
            # Store single norm per vector
            tl.store(NORMS_ptr + rows * stride_norms_n,
                     norms, mask=mask_n)
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
            # Store per-group norms (one per group per vector)
            for g in tl.static_range(0, N_GROUPS):
                g_mask = tl.where(group_of == g, 1.0, 0.0)
                g_norms = tl.sum(per_pos_norms * g_mask[None, :], axis=1)
                g_size = tl.sum(g_mask)
                g_norm_val = tl.where(g_size > 0, g_norms / g_size, 0.0)
                tl.store(NORMS_ptr + rows * stride_norms_n + g,
                         g_norm_val, mask=mask_n)

        # ---- Step 3: Forward rotation via tl.dot (tensor core) ----
        ROT_T = tl.load(
            BLOCK_ROT_T_ptr + cols[:, None] * HD + tl.arange(0, HD)[None, :]
        ).to(tl.float32)
        y = tl.dot(x_normed, ROT_T, allow_tf32=True)

        # ---- Step 4: Discretize to 256 bins ----
        offsets_lut = tl.load(LUT_OFFSETS_ptr + cols).to(tl.float32)
        inv_scales_lut = tl.load(LUT_INV_SCALES_ptr + cols).to(tl.float32)

        bin_idx = (y - offsets_lut[None, :]) * inv_scales_lut[None, :]
        bin_idx = tl.minimum(tl.maximum(bin_idx, 0.0), (N_BINS - 1) + 0.0)
        bin_idx_int = bin_idx.to(tl.int32)

        # ---- Step 5: Code LUT → centroid code (uint8) ----
        pos_cb = tl.load(POS_TO_CB_ptr + cols).to(tl.int32)
        clut_addr = pos_cb[None, :] * stride_clut + bin_idx_int
        codes = tl.load(CODE_LUT_ptr + clut_addr).to(tl.uint8)

        # ---- Step 6: Store codes ----
        tl.store(CODES_ptr + rows[:, None] * stride_codes_n + cols[None, :],
                 codes, mask=mask_n[:, None])

