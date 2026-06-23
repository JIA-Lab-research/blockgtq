"""Baseline KV-cache quantizers used by the public benches.

Block-GTQ is the paper's main method; the baselines below are the ones
Tab. 1 and the 10-model attention diagnostic table compare against:

  * :class:`KIVIScaleOnlyQuantizer` — KIVI's per-channel asymmetric
    quantization with a rolling-scale refresh every ``group_size``
    tokens, plus an offline calibration so the very first window starts
    with a sensible (channel_min, channel_max). The paper calls this
    variant *KIVI-ScaleOnly*; older code paths referred to it as
    "KIVI-rolling".

Both classes expose a uniform ``fit(k_data)`` / ``compress_decompress(k)``
interface so the bench scripts can plug them in alongside Block-GTQ
without special-casing per method.
"""
import torch


class KIVIScaleOnlyQuantizer:
    """KIVI per-channel asymmetric quantization with rolling-scale refresh.

    Every incoming token is immediately quantize-dequantized (no fp16 token
    ever persists in the cache). The ``(channel_min, channel_max)`` used
    for quantization is refreshed every ``group_size`` tokens from a
    stats-only fp16 buffer holding the most recent ``group_size`` tokens.
    The scale therefore lags the input by one group: tokens in window
    ``g`` are quantized with the scale derived from window ``g-1``.

    Calibration via :meth:`fit` seeds the first ``group_size`` tokens with
    a per-channel min/max computed from the held-out calibration data so
    the cold-start window is not random.
    """

    def __init__(self, head_dim: int, n_bits: int = 4, group_size: int = 32):
        self.head_dim = head_dim
        self.n_bits = n_bits
        self.n_levels = (1 << n_bits) - 1
        self.group_size = group_size
        self.channel_min = None
        self.channel_max = None
        self.channel_scale = None
        self._buf = None
        self._buf_filled = 0
        self._fitted = False

    def fit(self, k_data: torch.Tensor):
        """Compute per-channel min/max from calibration. ``k_data: (N, hd)``."""
        device = k_data.device
        self.channel_min = k_data.float().amin(dim=0).to(device)
        self.channel_max = k_data.float().amax(dim=0).to(device)
        self.channel_scale = (self.channel_max - self.channel_min
                              ).clamp(min=1e-8) / self.n_levels
        self._fitted = True

    def reset_rolling(self):
        """Drop the rolling window. The next ``group_size`` tokens will be
        quantized with whatever scale is currently active."""
        self._buf_filled = 0

    def compress_decompress(self, k: torch.Tensor) -> torch.Tensor:
        assert self._fitted, "call fit() before compress_decompress()"
        if self.channel_min.device != k.device:
            self.channel_min = self.channel_min.to(k.device)
            self.channel_max = self.channel_max.to(k.device)
            self.channel_scale = self.channel_scale.to(k.device)
        if self._buf is None or self._buf.device != k.device:
            self._buf = torch.empty(self.group_size, self.head_dim,
                                    device=k.device, dtype=torch.float32)
        orig_shape = k.shape
        k_flat = k.reshape(-1, self.head_dim).float()
        out = torch.empty_like(k_flat)
        N = k_flat.shape[0]
        g = self.group_size
        i = 0
        while i < N:
            take = min(g - self._buf_filled, N - i)
            chunk = k_flat[i:i + take]
            codes = torch.round((chunk - self.channel_min) /
                                self.channel_scale).clamp_(0, self.n_levels)
            out[i:i + take] = codes * self.channel_scale + self.channel_min
            self._buf[self._buf_filled:self._buf_filled + take] = chunk
            self._buf_filled += take
            if self._buf_filled == g:
                new_min = self._buf.amin(dim=0)
                new_max = self._buf.amax(dim=0)
                self.channel_min = new_min
                self.channel_max = new_max
                self.channel_scale = (new_max - new_min).clamp(min=1e-8) / self.n_levels
                self._buf_filled = 0
            i += take
        return out.reshape(orig_shape).to(k.dtype)


__all__ = [
    "KIVIScaleOnlyQuantizer",
]
