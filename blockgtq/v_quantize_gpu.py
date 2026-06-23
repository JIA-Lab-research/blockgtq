"""GPU V cache quantizer: PolarQuant in PyTorch.

Drop-in replacement for TurboQuantMSE that runs entirely on GPU.
Algorithm: norm extraction → random rotation → per-dim scalar quantize.

Usage:
    vq = PolarQuantGPU(d=128, bit_width=3, seed=42, device='cuda')
    codes, norms = vq.quantize(v)        # v: (N, D) fp16/fp32 on GPU
    v_hat = vq.dequantize(codes, norms)   # reconstructed fp16

    # With packing:
    packed = vq.quantize_packed(v)        # → (packed_codes, norms)
"""

import torch
import numpy as np


def _build_rotation_matrix(d: int, seed: int, device) -> torch.Tensor:
    """Build Haar-distributed random rotation via QR (deterministic by seed)."""
    rng = np.random.default_rng(seed)
    G = rng.standard_normal((d, d))
    Q, R = np.linalg.qr(G)
    # Ensure det(Q) = +1 (proper rotation)
    diag_signs = np.sign(np.diag(R))
    Q = Q * diag_signs[np.newaxis, :]
    if np.linalg.det(Q) < 0:
        Q[:, 0] = -Q[:, 0]
    return torch.from_numpy(Q.astype(np.float32)).to(device)


def _build_centroids(bit_width: int, d: int) -> np.ndarray:
    """Optimal MSE centroids for post-rotation coordinate distribution."""
    n = 1 << bit_width
    if bit_width == 1:
        c = np.sqrt(2.0 / (np.pi * d))
        return np.array([-c, c])
    if bit_width == 2:
        return np.array([-1.51, -0.453, 0.453, 1.51]) / np.sqrt(d)
    # Lloyd's algorithm on N(0, 1/d)
    sigma = 1.0 / np.sqrt(d)
    centroids = np.linspace(-3 * sigma, 3 * sigma, n)
    for _ in range(100):
        boundaries = (centroids[:-1] + centroids[1:]) / 2.0
        samples = np.random.default_rng(0).normal(0, sigma, 100000)
        bins = np.searchsorted(boundaries, samples)
        new_centroids = np.array([
            samples[bins == i].mean() if (bins == i).any() else centroids[i]
            for i in range(n)
        ])
        if np.allclose(centroids, new_centroids, atol=1e-8):
            break
        centroids = new_centroids
    return np.sort(centroids)


class PolarQuantGPU:
    """GPU PolarQuant for V cache compression.

    Seed-deterministic: same seed → same rotation matrix and centroids.
    No calibration data needed.
    """

    def __init__(self, d: int, bit_width: int, seed: int = 42,
                 norm_correction: bool = True, device=None):
        self.d = d
        self.bit_width = bit_width
        self.n_centroids = 1 << bit_width
        self.norm_correction = norm_correction
        self.device = device or torch.device('cuda')

        # Build rotation matrix (numpy QR, then move to GPU)
        self.rotation = _build_rotation_matrix(d, seed, self.device)  # (D, D) fp32

        # Build centroids and boundaries
        centroids_np = _build_centroids(bit_width, d)
        self.centroids = torch.from_numpy(
            centroids_np.astype(np.float32)).to(self.device)  # (n_centroids,)
        boundaries_np = (centroids_np[:-1] + centroids_np[1:]) / 2.0
        self.boundaries = torch.from_numpy(
            boundaries_np.astype(np.float32)).to(self.device)  # (n_centroids-1,)

    @torch.no_grad()
    def quantize(self, v: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Quantize V vectors on GPU.

        Args:
            v: (N, D) or (D,) fp16/fp32 tensor on GPU.

        Returns:
            codes: (N, D) uint8 centroid indices
            norms: (N,) fp16 L2 norms
        """
        single = v.dim() == 1
        if single:
            v = v.unsqueeze(0)

        x = v.float()

        # 1. Extract norms
        norms = x.norm(dim=1)  # (N,)
        safe_norms = norms.clamp(min=1e-8)

        # 2. Normalize
        x_norm = x / safe_norms.unsqueeze(1)

        # 3. Rotate: y = R @ x_norm^T → (D, N) → transpose → (N, D)
        y = (self.rotation @ x_norm.T).T

        # 4. Per-dim nearest centroid via searchsorted
        codes = torch.searchsorted(self.boundaries, y.contiguous())
        codes = codes.to(torch.uint8)

        norms_out = norms.to(torch.float16)
        if single:
            return codes.squeeze(0), norms_out.squeeze(0)
        return codes, norms_out

    @torch.no_grad()
    def quantize_packed(self, v: torch.Tensor
                        ) -> tuple[torch.Tensor, torch.Tensor]:
        """Quantize and pack V codes in one call.

        Returns:
            packed: (..., packed_bytes) uint8
            norms: (N,) fp16
        """
        from blockgtq.v_packing import pack_v_codes
        codes, norms = self.quantize(v)
        packed = pack_v_codes(codes, self.bit_width)
        return packed, norms

    @torch.no_grad()
    def dequantize(self, codes: torch.Tensor, norms: torch.Tensor
                   ) -> torch.Tensor:
        """Dequantize codes + norms back to V vectors.

        Args:
            codes: (N, D) uint8 centroid indices
            norms: (N,) fp16 norms

        Returns:
            v_hat: (N, D) fp16 reconstructed vectors
        """
        single = codes.dim() == 1
        if single:
            codes = codes.unsqueeze(0)
            norms = norms.unsqueeze(0)

        y_hat = self.centroids[codes.long()].float()  # (N, D)

        if self.norm_correction:
            y_norms = y_hat.norm(dim=1, keepdim=True).clamp(min=1e-10)
            y_hat = y_hat / y_norms

        # Inverse rotation: x_hat = R^T @ y_hat^T
        x_hat = (self.rotation.T @ y_hat.T).T  # (N, D)
        x_hat = x_hat * norms.float().unsqueeze(1)

        result = x_hat.to(torch.float16)
        return result.squeeze(0) if single else result

    @property
    def v_lut(self) -> torch.Tensor:
        """Codebook centroids as fp16 for the attention kernel."""
        return self.centroids.to(torch.float16)
