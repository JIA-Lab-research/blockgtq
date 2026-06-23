"""V-side bit packing utilities.

Packs/unpacks V cache codes with uniform bit width (1-4 bits).
V uses TurboQuant-MSE with a single bit width for all dimensions,
so packing is simpler than K-side (which has mixed bit widths per segment).

Packing format (same as K-side per-segment packing):
  1-bit: 8 codes per byte, bit 0 = first code
  2-bit: 4 codes per byte, bits[1:0] = first code
  3-bit: 8 codes per 3 bytes (sequential bit packing)
  4-bit: 2 codes per byte (nibble), bits[3:0] = first code
"""

import torch


def packed_v_bytes(D: int, bits: int) -> int:
    """Number of packed bytes per token for V at given bit width."""
    if bits == 1:
        return (D + 7) // 8
    elif bits == 2:
        return (D + 3) // 4
    elif bits == 3:
        return ((D + 7) // 8) * 3
    elif bits == 4:
        return (D + 1) // 2
    else:
        raise ValueError(f"V packing supports 1-4 bits, got {bits}")


def pack_v_codes(codes: torch.Tensor, bits: int) -> torch.Tensor:
    """Pack V codes with uniform bit width.

    Args:
        codes: (..., D) uint8 codes, values in [0, 2^bits)
        bits: bit width (1, 2, 3, or 4)

    Returns:
        packed: (..., packed_bytes) uint8
    """
    shape = codes.shape
    D = shape[-1]
    flat = codes.reshape(-1, D).to(torch.int32)
    N = flat.shape[0]
    pb = packed_v_bytes(D, bits)
    device = codes.device
    out = torch.zeros(N, pb, dtype=torch.uint8, device=device)

    if bits == 4:
        n_pairs = D // 2
        lo = flat[:, 0::2][:, :n_pairs] & 0xF          # even dims
        hi = (flat[:, 1::2][:, :n_pairs] & 0xF) << 4    # odd dims
        out[:, :n_pairs] = (lo | hi).to(torch.uint8)
        if D % 2 == 1:
            out[:, n_pairs] = (flat[:, D - 1] & 0xF).to(torch.uint8)

    elif bits == 2:
        n_quads = D // 4
        for p in range(4):
            end = min(n_quads * 4 + p, D)
            if p == 0:
                bv = flat[:, 0::4][:, :n_quads] & 0x3
            elif p < 4 and n_quads > 0:
                bv = bv | ((flat[:, p::4][:, :n_quads] & 0x3) << (p * 2))
        out[:, :n_quads] = bv.to(torch.uint8)
        # Handle remainder
        rem = D % 4
        if rem > 0:
            bv_rem = flat[:, n_quads * 4] & 0x3
            for p in range(1, rem):
                bv_rem |= (flat[:, n_quads * 4 + p] & 0x3) << (p * 2)
            out[:, n_quads] = bv_rem.to(torch.uint8)

    elif bits == 1:
        n_bytes = D // 8
        for p in range(8):
            if p == 0:
                bv = flat[:, 0::8][:, :n_bytes] & 1
            elif n_bytes > 0:
                bv = bv | ((flat[:, p::8][:, :n_bytes] & 1) << p)
        if n_bytes > 0:
            out[:, :n_bytes] = bv.to(torch.uint8)
        rem = D % 8
        if rem > 0:
            bv_rem = flat[:, n_bytes * 8] & 1
            for p in range(1, rem):
                bv_rem |= (flat[:, n_bytes * 8 + p] & 1) << p
            out[:, n_bytes] = bv_rem.to(torch.uint8)

    elif bits == 3:
        n_groups = (D + 7) // 8
        for g in range(n_groups):
            d = g * 8
            cs = []
            for p in range(8):
                if d + p < D:
                    cs.append(flat[:, d + p] & 0x7)
                else:
                    cs.append(torch.zeros(N, dtype=torch.int32, device=device))
            # 8 codes × 3 bits = 24 bits = 3 bytes
            byte0 = cs[0] | (cs[1] << 3) | (cs[2] << 6)
            byte1 = (cs[2] >> 2) | (cs[3] << 1) | (cs[4] << 4) | (cs[5] << 7)
            byte2 = (cs[5] >> 1) | (cs[6] << 2) | (cs[7] << 5)
            off = g * 3
            out[:, off] = (byte0 & 0xFF).to(torch.uint8)
            out[:, off + 1] = (byte1 & 0xFF).to(torch.uint8)
            out[:, off + 2] = (byte2 & 0xFF).to(torch.uint8)

    return out.reshape(*shape[:-1], pb)


def build_v_unpack_tables(bits: int, D: int, device=None):
    """Build per-dim unpack tables for V decode kernel.

    Same format as K unpack tables (Variant A style), but for uniform
    bit width. Used by the decode attention kernel to unpack V on-the-fly.

    For each dim d, the packed code is extracted as:
        code = ((packed[byte_lo[d]] >> shift_lo[d])
                | (packed[byte_hi[d]] << shift_hi[d])) & mask[d]

    Dims that don't span byte boundaries: byte_hi = byte_lo, shift_hi = 8
    (high byte contributes nothing after masking).

    Returns:
        byte_off_lo: (D,) int32
        byte_off_hi: (D,) int32
        shift_lo: (D,) int32
        shift_hi: (D,) int32
        mask: (D,) int32
    """
    byte_lo = torch.zeros(D, dtype=torch.int32)
    byte_hi = torch.zeros(D, dtype=torch.int32)
    sh_lo = torch.zeros(D, dtype=torch.int32)
    sh_hi = torch.zeros(D, dtype=torch.int32)
    code_mask = torch.full((D,), (1 << bits) - 1, dtype=torch.int32)

    if bits == 3:
        # 3-bit: 8 codes per 3-byte group
        for d in range(D):
            group = d // 8
            pos = d % 8
            bit_start = pos * 3  # within the 3-byte group
            lo_byte = group * 3 + bit_start // 8
            lo_bit = bit_start % 8
            hi_bit = lo_bit + bits
            byte_lo[d] = lo_byte
            sh_lo[d] = lo_bit
            if hi_bit <= 8:
                byte_hi[d] = lo_byte
                sh_hi[d] = 8  # no contribution from hi byte
            else:
                byte_hi[d] = lo_byte + 1
                sh_hi[d] = 8 - lo_bit
    else:
        # 1, 2, 4-bit: simple sequential bit packing
        for d in range(D):
            bit_start = d * bits
            lo_byte = bit_start // 8
            lo_bit = bit_start % 8
            byte_lo[d] = lo_byte
            sh_lo[d] = lo_bit
            # These widths never cross byte boundaries:
            # 1-bit: 8 codes per byte, 2-bit: 4 codes, 4-bit: 2 codes
            byte_hi[d] = lo_byte
            sh_hi[d] = 8

    if device is not None:
        byte_lo = byte_lo.to(device)
        byte_hi = byte_hi.to(device)
        sh_lo = sh_lo.to(device)
        sh_hi = sh_hi.to(device)
        code_mask = code_mask.to(device)

    return byte_lo, byte_hi, sh_lo, sh_hi, code_mask

