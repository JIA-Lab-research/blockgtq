"""TQ-MSE-fast: GPU-native TurboQuant MSE implementation in PyTorch.

Reimplements the TurboQuant PolarQuant algorithm (Algorithm 1 from the paper)
entirely in PyTorch on GPU, with no NumPy or CPU round-trips.

This provides a fair speed comparison with GTQ-fast (gtq_fast.py), KIVI, and RTN,
which are all PyTorch GPU implementations.

Algorithm:
  1. Extract L2 norms, normalize to unit sphere
  2. Rotate via dense Haar-distributed rotation matrix (QR decomposition)
  3. Scalar quantize each coordinate to nearest MSE-optimal centroid
  4. Norm-correct the quantized rotated vector
  5. Inverse rotate
  6. Rescale by original norms
"""

import math
import torch
import numpy as np
from typing import Optional


def _optimal_centroids_torch(bit_width: int, d: int, device: torch.device) -> torch.Tensor:
    """Compute optimal MSE centroids as a torch tensor on device.

    Mirrors turboquant.codebook.optimal_centroids but returns GPU tensor.
    Uses Lloyd's algorithm on N(0, 1/d) for b >= 3.
    """
    from scipy import stats as sp_stats

    n_centroids = 1 << bit_width

    if bit_width == 1:
        c = math.sqrt(2.0 / (math.pi * d))
        return torch.tensor([-c, c], dtype=torch.float32, device=device)

    if bit_width == 2:
        vals = np.array([-1.51, -0.453, 0.453, 1.51]) / math.sqrt(d)
        return torch.tensor(vals, dtype=torch.float32, device=device)

    # For b >= 3, use Lloyd's algorithm on N(0, 1/d)
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
    """Generate Haar-distributed rotation matrix directly as a GPU tensor.

    Uses QR decomposition of a random Gaussian matrix, with sign correction
    to ensure det(Q) = +1 (proper rotation).
    """
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


class TurboQuantMSE:
    """GPU-native TurboQuant MSE: dense rotation + optimal scalar quantization.

    All operations (norm extraction, rotation, quantization, dequantization)
    run entirely in PyTorch on the specified device. No NumPy, no CPU transfers
    in the hot path.

    Usage:
        tq = TurboQuantMSE(d=128, bit_width=3, device=torch.device('cuda:0'))

        # Round-trip (most common for KV cache compression benchmarks)
        x_hat = tq.quantize_dequantize(x)   # x: (batch, d) -> (batch, d)

        # Or separately
        indices, norms = tq.quantize(x)
        x_hat = tq.dequantize(indices, norms)
    """

    def __init__(self, d: int, bit_width: int, seed: int = 42,
                 norm_correction: bool = True,
                 device: Optional[torch.device] = None):
        self.d = d
        self.bit_width = bit_width
        self.n_centroids = 1 << bit_width
        self.norm_correction = norm_correction
        self.device = device or torch.device('cpu')

        # Precompute rotation matrix on device (done once at init, not in hot path)
        self.rotation = _random_rotation_dense_torch(d, seed, self.device)
        self.rotation_T = self.rotation.T.contiguous()

        # Precompute centroids and boundaries on device
        self.centroids = _optimal_centroids_torch(bit_width, d, self.device)
        self.boundaries = (self.centroids[:-1] + self.centroids[1:]) / 2.0

    def quantize(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Quantize a batch of vectors.

        Args:
            x: Input vectors, shape (batch, d) or (d,). Must be on self.device.

        Returns:
            (indices, norms) where:
                indices: integer centroid indices, shape same as input
                norms: L2 norms, shape (batch,) or scalar
        """
        single = x.ndim == 1
        if single:
            x = x.unsqueeze(0)

        # Extract norms
        norms = torch.norm(x, dim=1)  # (batch,)
        safe_norms = torch.where(norms > 0, norms, torch.ones_like(norms))
        x_normalized = x / safe_norms.unsqueeze(1)

        # Rotate: (batch, d) @ (d, d) -> (batch, d)
        y = x_normalized @ self.rotation_T

        # Nearest centroid via searchsorted on boundaries
        indices = torch.searchsorted(self.boundaries, y.contiguous())

        if single:
            return indices.squeeze(0), norms.squeeze(0)
        return indices, norms

    def dequantize(self, indices: torch.Tensor, norms: torch.Tensor) -> torch.Tensor:
        """Dequantize indices back to vectors.

        Args:
            indices: Integer centroid indices, shape (batch, d) or (d,).
            norms: Original L2 norms, shape (batch,) or scalar.

        Returns:
            Reconstructed vectors, same shape as original input.
        """
        single = indices.ndim == 1
        if single:
            indices = indices.unsqueeze(0)
            norms = norms.unsqueeze(0)

        # Look up centroids
        y_hat = self.centroids[indices]  # (batch, d)

        # Norm correction: re-normalize y_hat to unit norm
        if self.norm_correction:
            y_hat_norms = torch.norm(y_hat, dim=1, keepdim=True)
            y_hat_norms = torch.where(y_hat_norms > 1e-10, y_hat_norms,
                                       torch.ones_like(y_hat_norms))
            y_hat = y_hat / y_hat_norms

        # Inverse rotate: (batch, d) @ (d, d) -> (batch, d)
        x_hat_unit = y_hat @ self.rotation  # rotation_T.T = rotation

        # Rescale by original norms
        x_hat = x_hat_unit * norms.unsqueeze(1)

        if single:
            return x_hat.squeeze(0)
        return x_hat

    @torch.no_grad()
    def quantize_dequantize(self, x: torch.Tensor) -> torch.Tensor:
        """Full round-trip on device. x: (batch, d). Returns: (batch, d).

        This is the main entry point for benchmarking, equivalent to
        compress_decompress in baselines and GTQ.
        """
        single = x.ndim == 1
        if single:
            x = x.unsqueeze(0)

        # Extract norms
        norms = torch.norm(x, dim=1, keepdim=True)  # (batch, 1)
        safe_norms = torch.where(norms > 0, norms, torch.ones_like(norms))
        x_normalized = x / safe_norms

        # Rotate
        y = x_normalized @ self.rotation_T

        # Quantize: nearest centroid via searchsorted
        indices = torch.searchsorted(self.boundaries, y.contiguous())
        y_hat = self.centroids[indices]

        # Norm correction
        if self.norm_correction:
            y_hat_norms = torch.norm(y_hat, dim=1, keepdim=True)
            y_hat_norms = torch.where(y_hat_norms > 1e-10, y_hat_norms,
                                       torch.ones_like(y_hat_norms))
            y_hat = y_hat / y_hat_norms

        # Inverse rotate
        x_hat_unit = y_hat @ self.rotation

        # Rescale
        result = x_hat_unit * norms

        if single:
            return result.squeeze(0)
        return result

    def compress_decompress(self, x: torch.Tensor) -> torch.Tensor:
        """Alias for quantize_dequantize, matching baselines API."""
        return self.quantize_dequantize(x)


# ============================================================================
# TurboQuant FULL: Algorithm 2 = PolarQuant (b-1 bits) + QJL (1 bit residual)
# ============================================================================
#
# The paper's Algorithm 2 for inner-product preservation: spend (b-1) bits on
# MSE-optimal scalar quant of the rotated unit vector, then add a 1-bit QJL
# (Quantized Johnson–Lindenstrauss) sign code on the residual to remove the
# 2/π bias. Total b bits per coordinate.
#
# This class verifies the empirical claim that the QJL variant actually HURTS
# V-cache quality despite its theoretical superiority on inner products.

QJL_CONST_TORCH = math.sqrt(math.pi / 2.0)


def _qjl_projection_torch(d: int, seed: int, device: torch.device) -> torch.Tensor:
    """Random Gaussian projection matrix S ∈ R^{d×d} for QJL."""
    rng = np.random.default_rng(seed)
    S = rng.standard_normal((d, d))
    return torch.tensor(S, dtype=torch.float32, device=device)


class TurboQuantFull:
    """GPU-native TurboQuant FULL: PolarQuant at (b-1) bits + QJL at 1 bit.

    Two stages:
      1. Rotate x to y = R · (x / ||x||) ∈ unit sphere.
      2. Quantize y at (b-1) bits MSE-optimal scalar -> ŷ_mse, residual r = y - ŷ_mse.
      3. QJL on r: signs = sign(S · r), reconstruct r̂ = (√(π/2) / d) · S^T signs · ||r||.
      4. ŷ = ŷ_mse + r̂; x̂ = ŷ · R^{-1} · ||x||.

    For bit_width = b, the encoded record per vector contains:
      - mse indices: d × (b-1) bits
      - qjl signs:   d × 1 bit
      - vector norm: 1 fp16
      - residual norm: 1 fp16
    matching the bit budget b per coordinate (excluding norm overhead).

    For b=1 we fall back to QJL-only (sign(R x)). For b>=2 we use (b-1)+1 split.
    """

    def __init__(self, d: int, bit_width: int, seed: int = 42,
                 norm_correction: bool = True,
                 device: Optional[torch.device] = None):
        assert bit_width >= 1, f"bit_width must be >=1, got {bit_width}"
        self.d = d
        self.bit_width = bit_width
        self.norm_correction = norm_correction
        self.device = device or torch.device('cpu')

        # Stage-1 rotation R (same as TurboQuantMSE)
        self.rotation = _random_rotation_dense_torch(d, seed, self.device)
        self.rotation_T = self.rotation.T.contiguous()

        # Stage-1 centroids at (b-1) bits if b>=2, else degenerate (no MSE stage)
        self.has_mse_stage = bit_width >= 2
        if self.has_mse_stage:
            self.mse_bits = bit_width - 1
            self.n_centroids = 1 << self.mse_bits
            self.centroids = _optimal_centroids_torch(self.mse_bits, d, self.device)
            self.boundaries = (self.centroids[:-1] + self.centroids[1:]) / 2.0
        else:
            self.mse_bits = 0
            self.n_centroids = 1
            self.centroids = torch.zeros(1, dtype=torch.float32, device=self.device)
            self.boundaries = torch.empty(0, dtype=torch.float32, device=self.device)

        # Stage-2 QJL projection (separate seed so it's uncorrelated with R)
        self.qjl_S = _qjl_projection_torch(d, seed + 1000, self.device)
        self.qjl_S_T = self.qjl_S.T.contiguous()
        self.qjl_const = QJL_CONST_TORCH / d

    @torch.no_grad()
    def quantize_dequantize(self, x: torch.Tensor) -> torch.Tensor:
        """Full Alg-2 round-trip. x: (batch, d). Returns (batch, d)."""
        single = x.ndim == 1
        if single:
            x = x.unsqueeze(0)

        # Norm extraction
        norms = torch.norm(x, dim=1, keepdim=True)
        safe_norms = torch.where(norms > 0, norms, torch.ones_like(norms))
        x_normalized = x / safe_norms

        # Rotate to y on unit sphere
        y = x_normalized @ self.rotation_T

        # Stage 1: scalar MSE quant at (b-1) bits → ŷ_mse
        if self.has_mse_stage:
            indices = torch.searchsorted(self.boundaries, y.contiguous())
            y_hat_mse = self.centroids[indices]
            if self.norm_correction:
                yh_norms = torch.norm(y_hat_mse, dim=1, keepdim=True)
                yh_norms = torch.where(yh_norms > 1e-10, yh_norms,
                                       torch.ones_like(yh_norms))
                y_hat_mse = y_hat_mse / yh_norms
        else:
            y_hat_mse = torch.zeros_like(y)

        # Stage 2: QJL on residual r = y - ŷ_mse
        r = y - y_hat_mse                                     # (batch, d)
        r_proj = r @ self.qjl_S_T                              # (batch, d)
        signs = torch.sign(r_proj)                             # (batch, d), {±1}
        # zeros -> +1 (matches QJL convention)
        signs = torch.where(signs == 0, torch.ones_like(signs), signs)
        r_norms = torch.norm(r, dim=1, keepdim=True)           # (batch, 1)
        # Reconstruct residual: r̂ = (√(π/2) / d) · S^T · signs · ||r||
        r_hat_unit = signs @ self.qjl_S * self.qjl_const       # (batch, d)
        # r_hat_unit is approx r / ||r||; rescale by ||r||
        r_hat = r_hat_unit * r_norms

        # Combine
        y_hat = y_hat_mse + r_hat

        # Inverse rotate, rescale
        x_hat_unit = y_hat @ self.rotation
        result = x_hat_unit * norms

        if single:
            return result.squeeze(0)
        return result

    def compress_decompress(self, x: torch.Tensor) -> torch.Tensor:
        return self.quantize_dequantize(x)
