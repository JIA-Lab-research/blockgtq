"""Greedy per-block bit allocator for Block-GTQ.

Given a length-L vector of per-block importance scores ``s`` and a total bit
budget ``B``, ``greedy_bit_allocation`` assigns each block an integer bit
width ``b_i`` minimising the surrogate objective

    sum_i  s_i * 4^{-b_i}     subject to   sum_i b_i = B,
                                            min_bits <= b_i <= max_bits

by repeatedly handing the next bit to the block with the largest marginal
distortion-reduction.  The ``4^{-b}`` rate law mirrors the TQ-MSE primitive
used downstream (one extra bit quarters the local MSE bound), so the score
``s_i`` is the only allocation knob — typically the calibrated energy
``s_i = E[||q^(i)||^2 + ||k^(i)||^2] / 2`` per RoPE 2D block.
"""
import numpy as np
import torch


def greedy_bit_allocation(scores: torch.Tensor, total_budget: int,
                          min_bits: int = 1, max_bits: int = 8) -> torch.Tensor:
    """Greedy per-block bit allocation.

    Args:
        scores: (n_blocks,) per-block importance scores (>=0).
        total_budget: total bits to distribute (typically n_blocks * avg_bits).
        min_bits, max_bits: per-block constraints.

    Returns:
        (n_blocks,) int64 tensor of bit allocations summing to ``total_budget``
        (or to ``n_blocks * max_bits`` if the budget exceeds what every block
        can absorb).
    """
    n = len(scores)
    alloc = torch.full((n,), min_bits, dtype=torch.long)
    remaining = total_budget - alloc.sum().item()
    if remaining <= 0:
        return alloc

    scores_np = scores.cpu().float().numpy()
    alloc_np = alloc.numpy().copy()

    while remaining > 0:
        # Priority is proportional to s_i * 4^{-b_i} (one extra bit quarters
        # the contribution to the surrogate distortion).
        priorities = scores_np * (4.0 ** (-alloc_np.astype(float)))
        priorities[alloc_np >= max_bits] = -1.0

        best = int(np.argmax(priorities))
        if priorities[best] <= 0:
            break
        alloc_np[best] += 1
        remaining -= 1

    return torch.from_numpy(alloc_np).long()
