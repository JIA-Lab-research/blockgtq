"""Frequency importance scoring.

Block-GTQ scores each RoPE 2D frequency block by its calibrated energy:

    s_i = ( E[||q^(i)||^2] + E[||k^(i)||^2] ) / 2

The expectation is taken over a small held-out calibration set
(WikiText-2 train tokens). For GQA, every query head in a KV
head's group is kept as a separate sample: the per-head block norm is
squared first and then averaged (mean-of-squared-norms), which avoids the
Jensen underestimate that averaging the Q vectors before squaring would
incur. ``s_i`` is the only knob the greedy allocator consumes, so this
module needs to expose just one function.
"""
import torch


def compute_freq_importance_energy(q_blocks: torch.Tensor,
                                   k_blocks: torch.Tensor) -> torch.Tensor:
    """Energy score for each RoPE 2D block.

    Args:
        q_blocks: (n_samples_q, n_freqs, 2) pre-RoPE Q activations decomposed
            into 2-D frequency blocks. ``n_samples_q`` differs from
            ``n_samples_k``: every query head of a KV group is kept as a
            separate sample, so ``n_samples_q = |G(h)| * n_samples_k``.
        k_blocks: (n_samples_k, n_freqs, 2) pre-RoPE K activations.

    Returns:
        (n_freqs,) per-block importance scores
        ``s_i = (E[||q^(i)||^2] + E[||k^(i)||^2]) / 2``.
    """
    q_energy = (q_blocks ** 2).sum(dim=-1).mean(dim=0)  # (n_freqs,)
    k_energy = (k_blocks ** 2).sum(dim=-1).mean(dim=0)
    return (q_energy + k_energy) / 2
