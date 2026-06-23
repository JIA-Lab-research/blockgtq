"""RoPE frequency extraction, block decomposition, and rotation utilities.

Handles the HuggingFace "split-half" layout where frequency block i
pairs dimensions (i, i + head_dim//2).
"""
import torch
import math


def get_rope_frequencies(head_dim: int, rope_base: float = 10000.0) -> torch.Tensor:
    """Extract RoPE frequency for each 2D block.

    theta_i = 1 / (base^(2i/d)) for i = 0, ..., d/2-1

    Returns: (n_freqs,) tensor of angular frequencies.
    """
    n_freqs = head_dim // 2
    freq_indices = torch.arange(0, head_dim, 2, dtype=torch.float32)
    freqs = 1.0 / (rope_base ** (freq_indices / head_dim))
    return freqs  # (n_freqs,)


def decompose_freq_blocks(x: torch.Tensor, head_dim: int) -> torch.Tensor:
    """Decompose vectors into RoPE frequency blocks (split-half layout).

    x: (..., head_dim)
    Returns: (..., n_freqs, 2) where block[i] = (x[i], x[i + head_dim//2])
    """
    L = head_dim // 2
    first_half = x[..., :L]   # (..., L)
    second_half = x[..., L:]  # (..., L)
    return torch.stack([first_half, second_half], dim=-1)  # (..., L, 2)


def compose_freq_blocks(blocks: torch.Tensor) -> torch.Tensor:
    """Inverse of decompose_freq_blocks.

    blocks: (..., n_freqs, 2)
    Returns: (..., head_dim) in split-half layout
    """
    first_half = blocks[..., 0]   # (..., n_freqs)
    second_half = blocks[..., 1]  # (..., n_freqs)
    return torch.cat([first_half, second_half], dim=-1)  # (..., head_dim)


def apply_rope_to_blocks(blocks: torch.Tensor, freqs: torch.Tensor,
                         positions: torch.Tensor) -> torch.Tensor:
    """Apply RoPE rotation to frequency-decomposed blocks.

    blocks: (..., n_freqs, 2)
    freqs: (n_freqs,) angular frequencies
    positions: (...,) position indices

    Returns: (..., n_freqs, 2) rotated blocks
    """
    # angles: (..., n_freqs)
    angles = positions.unsqueeze(-1).float() * freqs.to(positions.device)
    cos_a = torch.cos(angles)
    sin_a = torch.sin(angles)

    x0 = blocks[..., 0]  # (..., n_freqs)
    x1 = blocks[..., 1]  # (..., n_freqs)

    y0 = x0 * cos_a - x1 * sin_a
    y1 = x0 * sin_a + x1 * cos_a
    return torch.stack([y0, y1], dim=-1)


def undo_rope_from_blocks(blocks_rot: torch.Tensor, freqs: torch.Tensor,
                          positions: torch.Tensor) -> torch.Tensor:
    """Undo RoPE rotation (apply R(-pos*theta))."""
    return apply_rope_to_blocks(blocks_rot, freqs, -positions)


def compute_rope_kernel(q_blocks: torch.Tensor, k_blocks: torch.Tensor,
                        freqs: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
    """Compute RoPE attention kernel q^T R_delta k.

    q_blocks, k_blocks: (..., n_freqs, 2)
    freqs: (n_freqs,)
    delta: (...,) relative position (scalar or batch)

    Returns: (...,) scalar attention scores
    """
    # Rotate k by delta
    k_rot = apply_rope_to_blocks(k_blocks, freqs, delta)
    # Inner product: sum over (n_freqs, 2)
    return (q_blocks * k_rot).sum(dim=(-2, -1))


def select_freq_dims(selected_indices: torch.Tensor, head_dim: int) -> torch.Tensor:
    """Get the actual dimension indices for selected frequency blocks.

    For split-half layout, block i uses dims (i, i + head_dim//2).

    Returns: (2 * n_selected,) sorted dimension indices
    """
    L = head_dim // 2
    first = selected_indices
    second = selected_indices + L
    return torch.sort(torch.cat([first, second]))[0]
