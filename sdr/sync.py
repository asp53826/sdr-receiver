"""Carrier, timing and frame synchronization.

Acquisition is split in two:

1. A blind fourth-power estimator removes most QPSK carrier offset without
   knowing any data. QPSK symbols raised to the fourth power lose their data
   modulation, leaving a tone at four times the carrier offset.
2. A known preamble jointly resolves frame start, sub-sample timing, residual
   frequency and phase. The phase of ``received * conj(preamble)`` is a line;
   its slope is residual carrier frequency and its intercept is phase.

That split mirrors a real burst receiver: a wide-pull-in blind estimator gets
close enough for a precise data-aided acquisition stage.
"""

from dataclasses import dataclass

import numpy as np


def agc(samples, target_rms=1.0):
    """One-shot burst AGC. Returns scaled samples and the applied gain."""
    samples = np.asarray(samples, dtype=complex).reshape(-1)
    rms = np.sqrt(np.mean(np.abs(samples) ** 2))
    if rms <= 1e-15:
        return samples.copy(), 1.0
    gain = target_rms / rms
    return samples * gain, gain


def estimate_qpsk_cfo(samples, sps=1, trim=0.1):
    """Blind fourth-power carrier estimate in cycles per symbol.

    The lag-one phase of y^4 avoids an FFT-bin quantisation floor. Samples with
    very small magnitude carry mostly pulse-transition noise, so a configurable
    lower-amplitude fraction is removed before the weighted average.
    """
    y = np.asarray(samples, dtype=complex).reshape(-1)
    if len(y) < 3:
        raise ValueError("at least three samples are required")
    z = y**4
    products = z[1:] * np.conjugate(z[:-1])
    weights = np.abs(y[1:] * y[:-1]) ** 2
    if trim > 0.0:
        threshold = np.quantile(weights, min(max(trim, 0.0), 0.95))
        keep = weights >= threshold
        products = products[keep]
        weights = weights[keep]
    phasor = np.sum(weights * products)
    if abs(phasor) <= 1e-30:
        return 0.0
    cycles_per_sample = np.angle(phasor) / (2.0 * np.pi * 4.0)
    return float(cycles_per_sample * sps)


def estimate_symbol_spaced_cfo(samples, sps):
    """Choose the most open eye phase, then run the fourth-power estimator.

    Applying y^4 directly to an oversampled pulse-shaped waveform is wrong:
    between symbol centres the RRC pulse is a mixture of adjacent symbols, so
    the fourth power no longer removes the data. Sampling each candidate eye
    phase first restores the estimator's assumption.
    """
    y = np.asarray(samples, dtype=complex).reshape(-1)
    best_metric = -np.inf
    best = None
    for phase in range(sps):
        candidate = y[phase::sps]
        if len(candidate) < 8:
            continue
        power = np.abs(candidate) ** 2
        mean_power = np.mean(power)
        if mean_power <= 1e-30:
            continue
        # A clean symbol-centre sequence has the sharpest constant-modulus eye.
        metric = -float(np.var(power) / (mean_power * mean_power))
        if metric > best_metric:
            best_metric = metric
            best = candidate
    if best is None:
        return 0.0
    return estimate_qpsk_cfo(best, sps=1)


def search_preamble_cfo(
    samples,
    preamble,
    sps,
    payload_symbols,
    max_offset=0.05,
    steps=201,
):
    """Data-aided coarse CFO search, independent of frame start.

    For each frequency hypothesis and eye phase, a normalized sliding preamble
    correlation is evaluated. Its magnitude is insensitive to constant phase,
    so the maximizer jointly identifies carrier offset, symbol phase and frame
    lag. The later acquisition stage refines the remaining sub-grid error.
    """
    y = np.asarray(samples, dtype=complex).reshape(-1)
    preamble = np.asarray(preamble, dtype=complex).reshape(-1)
    total_symbols = len(preamble) + int(payload_symbols)
    if steps < 3 or steps % 2 == 0:
        raise ValueError("steps must be an odd integer >= 3")
    if max_offset <= 0.0 or max_offset >= 0.125:
        raise ValueError("max_offset must be in (0, 0.125) for QPSK")

    frequencies = np.linspace(-max_offset, max_offset, steps)
    preamble_energy = np.vdot(preamble, preamble).real
    best_frequency = 0.0
    best_score = -1.0

    for frequency in frequencies:
        for phase in range(sps):
            sequence = y[phase::sps]
            latest = len(sequence) - total_symbols
            if latest < 0:
                continue
            index = np.arange(len(sequence), dtype=float)
            corrected = sequence * np.exp(-2j * np.pi * frequency * index)
            correlation = np.correlate(corrected, preamble, mode="valid")
            window_energy = np.convolve(
                np.abs(corrected) ** 2,
                np.ones(len(preamble), dtype=float),
                mode="valid",
            )
            usable = latest + 1
            score = np.abs(correlation[:usable]) / np.sqrt(
                np.maximum(preamble_energy * window_energy[:usable], 1e-30)
            )
            candidate = float(np.max(score))
            if candidate > best_score:
                best_score = candidate
                best_frequency = float(frequency)
    return best_frequency, best_score


def correct_cfo(samples, frequency_offset, sps=1):
    """Remove ``frequency_offset`` expressed in cycles per symbol."""
    y = np.asarray(samples, dtype=complex).reshape(-1)
    n = np.arange(len(y), dtype=float)
    return y * np.exp(-2j * np.pi * frequency_offset * n / sps)


def _sample_linear(samples, positions):
    """Complex linear interpolation at arbitrary real sample positions."""
    samples = np.asarray(samples, dtype=complex)
    positions = np.asarray(positions, dtype=float)
    grid = np.arange(len(samples), dtype=float)
    real = np.interp(positions, grid, np.real(samples), left=0.0, right=0.0)
    imag = np.interp(positions, grid, np.imag(samples), left=0.0, right=0.0)
    return real + 1j * imag


def _normalised_correlation(reference, observed):
    denom = np.linalg.norm(reference) * np.linalg.norm(observed)
    if denom <= 1e-30:
        return 0.0
    return float(abs(np.vdot(reference, observed)) / denom)


@dataclass
class Acquisition:
    start_sample: float
    score: float
    residual_cfo: float
    phase: float
    gain: complex
    noise_variance: float


def acquire_preamble(samples, preamble, sps, payload_symbols, refinement=21):
    """Find and calibrate a burst from its symbol preamble.

    Returns the corrected unit-gain symbol sequence (preamble + payload) and
    diagnostics. The search is exhaustive over sample start because the
    benchmark bursts are short; production receivers replace this with a
    polyphase correlator that computes the same statistic incrementally.
    """
    y = np.asarray(samples, dtype=complex).reshape(-1)
    preamble = np.asarray(preamble, dtype=complex).reshape(-1)
    total_symbols = len(preamble) + int(payload_symbols)
    latest = len(y) - 1 - (total_symbols - 1) * sps
    if latest <= 0:
        raise ValueError("samples are too short for the expected frame")

    symbol_offsets = np.arange(len(preamble), dtype=float) * sps
    best_start, best_score = 0, -1.0
    for start in range(latest + 1):
        observed = y[start + symbol_offsets.astype(int)]
        score = _normalised_correlation(preamble, observed)
        if score > best_score:
            best_start, best_score = start, score

    # Resolve the fractional sample phase around the best integer start.
    fractions = np.linspace(-0.75, 0.75, refinement)
    refined_start = float(best_start)
    for frac in fractions:
        candidate = best_start + frac
        observed = _sample_linear(y, candidate + symbol_offsets)
        score = _normalised_correlation(preamble, observed)
        if score > best_score:
            refined_start, best_score = float(candidate), score

    positions = refined_start + np.arange(total_symbols, dtype=float) * sps
    symbols = _sample_linear(y, positions)
    observed_preamble = symbols[: len(preamble)]

    # Residual carrier is a narrow maximum-likelihood tone search after
    # removing the known preamble symbols. Fitting an unwrapped noisy phase
    # ramp is brittle: one ±2pi unwrap mistake tilts the whole line and rotates
    # the payload into the wrong quadrant. A lag-one estimate avoids unwraps
    # but has too much variance on a short burst. The coarse search already
    # leaves less than one grid bin, so a dense local periodogram is both cheap
    # and substantially more precise.
    phase_error = observed_preamble * np.conjugate(preamble)
    index = np.arange(len(preamble), dtype=float)
    residual_grid = np.linspace(-0.0025, 0.0025, 501)
    steering = np.exp(-2j * np.pi * residual_grid[:, None] * index[None, :])
    scores = np.abs(steering @ phase_error)
    residual_cfo = float(residual_grid[int(np.argmax(scores))])
    slope = 2.0 * np.pi * residual_cfo
    de_ramped = phase_error * np.exp(-1j * slope * index)
    intercept = float(np.angle(np.sum(np.abs(phase_error) * de_ramped)))
    rotation = np.exp(-1j * (intercept + slope * np.arange(total_symbols)))
    symbols = symbols * rotation

    # Least-squares complex gain and a data-aided noise estimate.
    observed_preamble = symbols[: len(preamble)]
    gain = np.vdot(preamble, observed_preamble) / np.vdot(preamble, preamble)
    if abs(gain) <= 1e-12:
        raise RuntimeError("preamble gain estimate collapsed to zero")
    symbols = symbols / gain
    residual = symbols[: len(preamble)] - preamble
    noise_variance = float(max(np.mean(np.abs(residual) ** 2), 1e-8))

    return symbols, Acquisition(
        start_sample=refined_start,
        score=float(best_score),
        residual_cfo=residual_cfo,
        phase=float(intercept),
        gain=complex(gain),
        noise_variance=noise_variance,
    )


def decision_directed_costas(symbols, loop_bandwidth=0.02, damping=0.707):
    """Second-order QPSK Costas loop for residual phase/frequency tracking.

    This is used after burst acquisition, not as the wide-pull-in estimator.
    The detector compares each rotated symbol with its nearest QPSK quadrant.
    """
    y = np.asarray(symbols, dtype=complex).reshape(-1)
    if not 0.0 < loop_bandwidth < 1.0:
        raise ValueError("loop_bandwidth must be in (0, 1)")

    denominator = 1.0 + 2.0 * damping * loop_bandwidth + loop_bandwidth**2
    alpha = 4.0 * damping * loop_bandwidth / denominator
    beta = 4.0 * loop_bandwidth**2 / denominator

    out = np.empty_like(y)
    phase = 0.0
    frequency = 0.0
    for i, sample in enumerate(y):
        rotated = sample * np.exp(-1j * phase)
        decision = (
            (1.0 if rotated.real >= 0.0 else -1.0)
            + 1j * (1.0 if rotated.imag >= 0.0 else -1.0)
        ) / np.sqrt(2.0)
        error = float(np.imag(rotated * np.conjugate(decision)))
        frequency += beta * error
        phase += frequency + alpha * error
        out[i] = rotated
    return out
