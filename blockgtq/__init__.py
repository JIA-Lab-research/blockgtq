"""Block-GTQ: a RoPE-aware mixed-rate KV-cache quantizer."""

from blockgtq.allocator import greedy_bit_allocation
from blockgtq.tq import TurboQuantMSE
from blockgtq.block_gtq_pipeline import BlockGTQPipeline
from blockgtq.quantizer import BlockGTQQuantizer
from blockgtq.packed_kv_manager import BlockGTQKVManager
from blockgtq.production_cache import BlockGTQProductionCache
from blockgtq.production_decode import (
    production_decode_step,
    production_prefill,
)
from blockgtq.fp16_baseline import FP16KVCache, fp16_decode_step
from blockgtq.calibration import (
    collect_qk_activations,
    build_quantizers,
    bake_q_rotations,
)
from blockgtq.prefill import layer_major_prefill

__all__ = [
    "greedy_bit_allocation",
    "TurboQuantMSE",
    "BlockGTQPipeline",
    "BlockGTQQuantizer",
    "BlockGTQKVManager",
    "BlockGTQProductionCache",
    "production_decode_step",
    "production_prefill",
    "FP16KVCache",
    "fp16_decode_step",
    "collect_qk_activations",
    "build_quantizers",
    "bake_q_rotations",
    "layer_major_prefill",
]
