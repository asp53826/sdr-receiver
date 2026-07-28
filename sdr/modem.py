"""Gray-labelled QPSK modulation and soft demapping.

The symbol mapping is deliberately explicit:

    bits  symbol
    00    (+1 + j) / sqrt(2)
    01    (+1 - j) / sqrt(2)
    11    (-1 - j) / sqrt(2)
    10    (-1 + j) / sqrt(2)

Each bit therefore controls one orthogonal BPSK component.  This makes the
exact AWGN log-likelihood ratio a two-line calculation rather than a nearest-
constellation approximation.
"""

import numpy as np

SQRT2 = np.sqrt(2.0)


def qpsk_modulate(bits):
    """Map an even-length bit vector to unit-energy complex QPSK symbols."""
    bits = np.asarray(bits, dtype=np.uint8).reshape(-1)
    if len(bits) % 2:
        raise ValueError("QPSK needs an even number of bits")
    if np.any(bits > 1):
        raise ValueError("bits must contain only 0 and 1")
    pairs = bits.reshape(-1, 2)
    i = 1.0 - 2.0 * pairs[:, 0].astype(float)
    q = 1.0 - 2.0 * pairs[:, 1].astype(float)
    return (i + 1j * q) / SQRT2


def qpsk_hard_demodulate(symbols):
    """Nearest-quadrant hard decision, returned in transmit bit order."""
    symbols = np.asarray(symbols, dtype=complex).reshape(-1)
    bits = np.empty(2 * len(symbols), dtype=np.uint8)
    bits[0::2] = np.real(symbols) < 0.0
    bits[1::2] = np.imag(symbols) < 0.0
    return bits


def qpsk_soft_llr(symbols, noise_variance):
    """Exact bit LLRs for Gray QPSK in circular complex AWGN.

    ``noise_variance`` is E[|n|^2] at the symbol sampler. Positive LLR means
    bit 0 is more likely; negative means bit 1 is more likely.
    """
    if noise_variance <= 0.0:
        raise ValueError("noise_variance must be positive")
    symbols = np.asarray(symbols, dtype=complex).reshape(-1)
    scale = 2.0 * SQRT2 / float(noise_variance)
    llr = np.empty(2 * len(symbols), dtype=float)
    llr[0::2] = scale * np.real(symbols)
    llr[1::2] = scale * np.imag(symbols)
    return llr


def qpsk_theoretical_ber(ebn0_db):
    """Uncoded coherent Gray-QPSK BER, identical to BPSK: Q(sqrt(2 Eb/N0))."""
    from scipy.special import erfc

    gamma_b = 10.0 ** (np.asarray(ebn0_db, dtype=float) / 10.0)
    return 0.5 * erfc(np.sqrt(gamma_b))

