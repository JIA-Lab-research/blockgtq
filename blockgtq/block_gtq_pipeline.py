"""Block-GTQ pipeline: GPU-native per-block bit allocation + hybrid quantization.

Each unique bit width forms its own group with an independent rotation +
codebook:

  - Low-bit groups (bits < ``rotation_threshold``): identity rotation, so the
    per-block independence used by the score bound is preserved.
  - High-bit groups (bits >= ``rotation_threshold``): a dense Haar rotation
    per bit width with the MSE-optimal scalar codebook for that bit width.

The hot path issues a single block-diagonal matmul, a per-position 2D
searchsorted, and a fused inverse-rotate-and-unpermute matmul — enough to
benchmark Block-GTQ end-to-end without touching the production Triton kernel
in :mod:`blockgtq.quantizer`.
"""
import torch
import numpy as np
import math
from typing import Optional

from blockgtq.allocator import greedy_bit_allocation


def _optimal_centroids_torch(bit_width: int, d: int, device: torch.device) -> torch.Tensor:
    """Compute optimal MSE centroids as a torch tensor on device."""
    from scipy import stats as sp_stats

    n_centroids = 1 << bit_width

    if bit_width == 1:
        c = math.sqrt(2.0 / (math.pi * d))
        return torch.tensor([-c, c], dtype=torch.float32, device=device)

    if bit_width == 2:
        vals = np.array([-1.51, -0.453, 0.453, 1.51]) / math.sqrt(d)
        return torch.tensor(vals, dtype=torch.float32, device=device)

    sigma = 1.0 / math.sqrt(d)
    boundaries = sp_stats.norm.ppf(
        np.linspace(0, 1, n_centroids + 1)[1:-1], scale=sigma
    )
    centroids = np.zeros(n_centroids)

    def _cond_exp(sig, a, b):
        a_std = a / sig if np.isfinite(a) else a
        b_std = b / sig if np.isfinite(b) else b
        if not np.isfinite(a_std):
            prob = sp_stats.norm.cdf(b_std)
        elif not np.isfinite(b_std):
            prob = sp_stats.norm.sf(a_std)
        else:
            prob = sp_stats.norm.cdf(b_std) - sp_stats.norm.cdf(a_std)
        if prob < 1e-15:
            if np.isfinite(a) and not np.isfinite(b):
                return a + sig
            elif not np.isfinite(a) and np.isfinite(b):
                return b - sig
            elif np.isfinite(a) and np.isfinite(b):
                return (a + b) / 2.0
            else:
                return 0.0
        pdf_diff = sp_stats.norm.pdf(a_std) - sp_stats.norm.pdf(b_std)
        return sig * pdf_diff / prob

    centroids[0] = _cond_exp(sigma, -np.inf, boundaries[0])
    for i in range(1, n_centroids - 1):
        centroids[i] = _cond_exp(sigma, boundaries[i - 1], boundaries[i])
    centroids[-1] = _cond_exp(sigma, boundaries[-1], np.inf)

    for _ in range(100):
        boundaries = (centroids[:-1] + centroids[1:]) / 2.0
        centroids[0] = _cond_exp(sigma, -np.inf, boundaries[0])
        for i in range(1, n_centroids - 1):
            centroids[i] = _cond_exp(sigma, boundaries[i - 1], boundaries[i])
        centroids[-1] = _cond_exp(sigma, boundaries[-1], np.inf)

    centroids.sort()
    return torch.tensor(centroids, dtype=torch.float32, device=device)


def _random_rotation_dense_torch(d: int, seed: int, device: torch.device) -> torch.Tensor:
    """Generate Haar-distributed rotation matrix directly as a GPU tensor."""
    rng = np.random.default_rng(seed)
    G = rng.standard_normal((d, d))
    Q, R = np.linalg.qr(G)
    signs = np.sign(np.diag(R))
    signs[signs == 0] = 1.0
    Q = Q * signs[np.newaxis, :]
    sign, _ = np.linalg.slogdet(Q)
    if sign < 0:
        Q[:, 0] = -Q[:, 0]
    return torch.tensor(Q, dtype=torch.float32, device=device)


class BlockGTQPipeline:
    """GPU-native Block-GTQ: per-bit-width groups with pipeline-style pipeline.

    Each unique bit width forms a separate group:
      - Low-bit (< threshold): identity rotation (preserves per-block independence)
      - High-bit (>= threshold): dense rotation (TQ quality)

    Hot path (pipeline-style, all GPU):
      1. Permute (index)
      2. Per-group norms (group_mask matmul)
      3. Block-diagonal rotation (single hd x hd matmul)
      4. Per-position 2D searchsorted quantization
      5. Per-group norm correction (group_mask matmul)
      6. Rescale + fused inverse rotate + unpermute (single matmul)
    """

    def __init__(self, head_dim: int, avg_bits: float = 3.0,
                 rotation_threshold: int = 3,
                 min_bits: int = 1, max_bits: int = 8,
                 rope_base: float = 1e6,
                 device: Optional[torch.device] = None):
        self.head_dim = head_dim
        self.n_freqs = head_dim // 2
        self.avg_bits = avg_bits
        self.rotation_threshold = rotation_threshold
        self.min_bits = min_bits
        self.max_bits = max_bits
        self.rope_base = rope_base
        self.device = device or torch.device('cuda:0')
        self._calibrated = False
        self.bit_allocation = None
        # If True, the head_perm has been baked into an upstream k_proj weight,
        # so compress_decompress should NOT apply the runtime permutation.
        # Set via bake_permutation_into_kproj().
        self._perm_baked = False

        from blockgtq.rope_utils import get_rope_frequencies
        self.freqs = get_rope_frequencies(head_dim, rope_base)

    def calibrate(self, q_data: torch.Tensor, k_data: torch.Tensor):
        """Calibrate and precompute all GPU tensors."""
        from blockgtq.rope_utils import decompose_freq_blocks
        from blockgtq.freq_analysis import compute_freq_importance_energy

        q_blocks = decompose_freq_blocks(q_data, self.head_dim)
        k_blocks = decompose_freq_blocks(k_data, self.head_dim)
        scores = compute_freq_importance_energy(q_blocks, k_blocks)

        total_budget = int(self.n_freqs * self.avg_bits)
        self.bit_allocation = greedy_bit_allocation(
            scores, total_budget, self.min_bits, self.max_bits
        )
        self._calib_scores = scores

        self._post_allocation_setup(summary="BlockGTQ")

    def _post_allocation_setup(self, summary: str = "BlockGTQ"):
        """Build rotation groups, centroids, and LUTs from ``self.bit_allocation``.

        Safe to re-run after mutating ``self.bit_allocation`` (used by the
        HA-Block-GTQ subclass to rebuild after phi round-up).
        """
        L = self.n_freqs
        hd = self.head_dim
        rt = self.rotation_threshold

        # Sort all freq blocks by bit width
        all_bits = self.bit_allocation
        sorted_order = all_bits.argsort()
        freq_sorted = sorted_order
        bits_sorted = all_bits[freq_sorted]
        unique_bits = sorted(bits_sorted.unique().tolist())

        # Build head_dim permutation: freq blocks sorted by bit width
        head_perm = torch.empty(hd, dtype=torch.long)
        for pos, fi in enumerate(freq_sorted):
            head_perm[2 * pos] = fi.item()
            head_perm[2 * pos + 1] = fi.item() + L

        self._head_perm = head_perm.to(self.device)
        inv_perm = torch.empty_like(self._head_perm)
        inv_perm[self._head_perm] = torch.arange(hd, device=self.device)
        self._head_perm_inv = inv_perm

        # Build block-diagonal rotation: one block per unique bit width
        block_rot = torch.zeros(hd, hd, device=self.device, dtype=torch.float32)
        all_group_slices = []
        all_bits_list = []
        offset = 0

        for bw in unique_bits:
            mask = bits_sorted == bw
            n_bw = mask.sum().item()
            d_bw = n_bw * 2
            s, e = offset, offset + d_bw
            all_group_slices.append((s, e))
            all_bits_list.append(bw)

            if bw < rt:
                block_rot[s:e, s:e] = torch.eye(d_bw, device=self.device)
            else:
                rot = _random_rotation_dense_torch(d_bw, 42 + bw, self.device)
                block_rot[s:e, s:e] = rot

            offset += d_bw

        self._block_rot_T = block_rot.T.contiguous()
        self._block_rot_unperm = block_rot[:, inv_perm].contiguous()

        n_groups = len(all_group_slices)
        self._n_groups = n_groups

        max_n_centroids = max(1 << bw for bw in all_bits_list)
        max_n_boundaries = max_n_centroids - 1

        pos_boundaries = torch.full((hd, max_n_boundaries), float('inf'),
                                     device=self.device, dtype=torch.float32)
        pos_centroids = torch.zeros((hd, max_n_centroids),
                                     device=self.device, dtype=torch.float32)

        for gi, (s, e) in enumerate(all_group_slices):
            d = e - s
            bw = all_bits_list[gi]
            c = _optimal_centroids_torch(bw, d, self.device)
            b = (c[:-1] + c[1:]) / 2.0
            pos_boundaries[s:e, :len(b)] = b.unsqueeze(0).expand(d, -1)
            pos_centroids[s:e, :len(c)] = c.unsqueeze(0).expand(d, -1)

        self._pos_boundaries = pos_boundaries
        self._pos_centroids = pos_centroids
        self._row_idx = torch.arange(hd, device=self.device).unsqueeze(1)

        group_of = torch.zeros(hd, dtype=torch.long, device=self.device)
        group_mask = torch.zeros(hd, n_groups, device=self.device, dtype=torch.float32)
        for gi, (s, e) in enumerate(all_group_slices):
            group_of[s:e] = gi
            group_mask[s:e, gi] = 1.0
        self._group_of = group_of
        self._group_mask = group_mask

        self._single_group = (n_groups == 1)

        self._calibrated = True

        from collections import Counter
        dist = Counter(self.bit_allocation.numpy().tolist())
        eff = self.bit_allocation.float().sum().item() / self.n_freqs
        n_low_grp = sum(1 for bw in all_bits_list if bw < rt)
        n_high_grp = sum(1 for bw in all_bits_list if bw >= rt)
        print(f"  {summary}: eff={eff:.2f}b/d, "
              f"grps={n_groups}(low={n_low_grp},high={n_high_grp}), "
              f"dist={dict(sorted(dist.items()))}")

    @property
    def effective_bits_per_dim(self):
        if self.bit_allocation is None:
            return self.avg_bits
        return self.bit_allocation.float().sum().item() / self.n_freqs

    @torch.no_grad()
    def bake_permutation_into_kproj(
        self,
        k_proj_weight: torch.Tensor,
        kv_head_idx: int,
        n_kv_heads: int,
        k_proj_bias: Optional[torch.Tensor] = None,
    ) -> None:
        """Bake this quantizer's head_perm into the k_proj weight slice.

        This eliminates the runtime permutation in compress_decompress for
        this kv head. The k_proj weight is modified IN PLACE so that the
        rows corresponding to ``kv_head_idx`` produce K vectors already in
        the post-permutation order. Sets ``self._perm_baked = True`` so
        subsequent compress_decompress calls skip the gather.

        Args:
            k_proj_weight: ``(n_kv_heads * head_dim, in_features)`` weight tensor
                of the K projection (nn.Linear). Modified in-place.
            kv_head_idx: which kv head this quantizer is associated with.
            n_kv_heads: total number of kv heads in the layer.
            k_proj_bias: optional bias of shape ``(n_kv_heads * head_dim,)``;
                if provided, the corresponding rows are also reordered.

        After this call, ``compress_decompress(k)`` expects ``k`` to be in
        permuted layout (i.e., the output of the modified k_proj). The output
        is still in the ORIGINAL layout because the inverse rotation already
        absorbs the un-permutation via ``_block_rot_unperm``.

        Mathematically: this is a one-time pre-multiplication of the
        permutation matrix into k_proj's weight. The PPL is unchanged.
        """
        assert self._calibrated, "calibrate() must be called before baking"
        assert kv_head_idx < n_kv_heads
        hd = self.head_dim
        assert k_proj_weight.shape[0] == n_kv_heads * hd, (
            f"k_proj_weight has shape {tuple(k_proj_weight.shape)}, "
            f"expected first dim = n_kv_heads * head_dim = {n_kv_heads * hd}"
        )

        perm = self._head_perm.detach().cpu()
        start = kv_head_idx * hd
        end = start + hd

        # Reorder rows in [start, end) so that the new row i is the old row perm[i].
        # After this change: K_new[h*hd + i] = K_old[h*hd + perm[i]]
        new_rows = k_proj_weight[start:end].clone()[perm].contiguous()
        k_proj_weight[start:end] = new_rows
        if k_proj_bias is not None:
            new_bias = k_proj_bias[start:end].clone()[perm].contiguous()
            k_proj_bias[start:end] = new_bias

        self._perm_baked = True

    @torch.no_grad()
    def compress_decompress(self, k: torch.Tensor) -> torch.Tensor:
        """GPU round-trip using pipeline-style general path."""
        assert self._calibrated
        batch_shape = k.shape[:-1]
        hd = self.head_dim

        # Permute (skipped if the permutation was baked into k_proj's weights)
        if self._perm_baked:
            x = k.reshape(-1, hd).float()
        else:
            x = k.reshape(-1, hd).float()[:, self._head_perm]

        # Per-group norms
        if self._single_group:
            norms = torch.norm(x, dim=1, keepdim=True)
            safe_norms = torch.where(norms > 0, norms, torch.ones_like(norms))
            x = x / safe_norms
        else:
            x_sq = x * x
            sq_norms = x_sq @ self._group_mask
            norms_g = torch.sqrt(sq_norms)
            safe_norms_g = torch.where(norms_g > 0, norms_g, torch.ones_like(norms_g))
            per_pos_norms = safe_norms_g[:, self._group_of]
            x = x / per_pos_norms

        # Block-diagonal rotation
        y = x @ self._block_rot_T

        # Fused 2D searchsorted quantization
        y_t = y.T.contiguous()
        indices_t = torch.searchsorted(self._pos_boundaries, y_t)
        indices_t = indices_t.clamp(max=self._pos_centroids.shape[1] - 1)
        y_hat = self._pos_centroids[self._row_idx, indices_t].T.contiguous()

        # Norm correction + rescale + fused inverse rotate + unpermute
        if self._single_group:
            yh_norms = torch.norm(y_hat, dim=1, keepdim=True)
            yh_safe = torch.where(yh_norms > 1e-10, yh_norms, torch.ones_like(yh_norms))
            y_hat = y_hat / yh_safe
            result = (y_hat * norms) @ self._block_rot_unperm
        else:
            yh_sq = y_hat * y_hat
            yh_sq_norms = yh_sq @ self._group_mask
            yh_norms = torch.sqrt(yh_sq_norms)
            yh_safe = torch.where(yh_norms > 1e-10, yh_norms, torch.ones_like(yh_norms))
            y_hat = y_hat / yh_safe[:, self._group_of]
            result = (y_hat * per_pos_norms) @ self._block_rot_unperm

        return result.reshape(*batch_shape, hd)
