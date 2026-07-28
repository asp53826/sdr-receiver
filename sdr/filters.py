"""Pulse-shaping and interpolation primitives."""

import numpy as np


def root_raised_cosine(beta=0.35, sps=4, span=10):
    """Unit-energy root-raised-cosine FIR.

    ``span`` is in symbols and must be even so the impulse response has one
    unambiguous centre sample. The two removable singularities are evaluated
    from their analytic limits instead of relying on an epsilon nudge.
    """
    if not 0.0 <= beta <= 1.0:
        raise ValueError("beta must be in [0, 1]")
    if sps < 2:
        raise ValueError("sps must be at least 2")
    if span <= 0 or span % 2:
        raise ValueError("span must be a positive even integer")

    t = np.arange(-span * sps // 2, span * sps // 2 + 1, dtype=float) / sps
    h = np.empty_like(t)

    if beta == 0.0:
        h[:] = np.sinc(t)
    else:
        at_zero = np.isclose(t, 0.0, atol=1e-14)
        at_singularity = np.isclose(np.abs(t), 1.0 / (4.0 * beta), atol=1e-14)
        regular = ~(at_zero | at_singularity)

        h[at_zero] = 1.0 + beta * (4.0 / np.pi - 1.0)
        h[at_singularity] = (
            beta
            / np.sqrt(2.0)
            * (
                (1.0 + 2.0 / np.pi) * np.sin(np.pi / (4.0 * beta))
                + (1.0 - 2.0 / np.pi) * np.cos(np.pi / (4.0 * beta))
            )
        )
        tr = t[regular]
        numerator = (
            np.sin(np.pi * tr * (1.0 - beta))
            + 4.0 * beta * tr * np.cos(np.pi * tr * (1.0 + beta))
        )
        denominator = np.pi * tr * (1.0 - (4.0 * beta * tr) ** 2)
        h[regular] = numerator / denominator

    return h / np.sqrt(np.sum(h * h))


def pulse_shape(symbols, taps, sps):
    """Upsample by ``sps`` and apply the transmit FIR."""
    symbols = np.asarray(symbols, dtype=complex).reshape(-1)
    up = np.zeros(len(symbols) * sps, dtype=complex)
    up[::sps] = symbols
    return np.convolve(up, np.asarray(taps, dtype=float), mode="full")


def matched_filter(samples, taps):
    """Apply the conjugate time-reversed receive filter."""
    return np.convolve(
        np.asarray(samples, dtype=complex),
        np.conjugate(np.asarray(taps)[::-1]),
        mode="full",
    )


def fractional_delay(samples, delay):
    """Band-limited-enough fractional delay using windowed-sinc interpolation.

    A 21-tap kernel is long enough for the RRC-shaped waveforms used here and
    keeps the operation deterministic. Positive ``delay`` moves the waveform
    later in time.
    """
    samples = np.asarray(samples, dtype=complex).reshape(-1)
    if abs(delay) < 1e-15:
        return samples.copy()
    half = 10
    n = np.arange(-half, half + 1, dtype=float)
    kernel = np.sinc(n - delay) * np.hamming(2 * half + 1)
    kernel /= kernel.sum()
    return np.convolve(samples, kernel, mode="same")

