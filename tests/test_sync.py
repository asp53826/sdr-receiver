import numpy as np
import pytest

from sdr.filters import matched_filter, pulse_shape, root_raised_cosine
from sdr.modem import qpsk_modulate
from sdr.sync import (
    acquire_preamble,
    correct_cfo,
    decision_directed_costas,
    estimate_qpsk_cfo,
    estimate_symbol_spaced_cfo,
    search_preamble_cfo,
)


def test_fourth_power_cfo_estimator_on_qpsk_symbols():
    rng = np.random.default_rng(2)
    symbols = qpsk_modulate(rng.integers(0, 2, 4000, dtype=np.uint8))
    offset = 0.023
    n = np.arange(len(symbols))
    shifted = symbols * np.exp(2j * np.pi * offset * n)
    estimate = estimate_qpsk_cfo(shifted)
    assert estimate == pytest.approx(offset, abs=2e-5)
    corrected = correct_cfo(shifted, estimate)
    assert np.mean(np.abs(corrected - symbols) ** 2) < 1e-5


def test_cfo_estimator_uses_symbol_centres_not_pulse_transitions():
    rng = np.random.default_rng(8)
    symbols = qpsk_modulate(rng.integers(0, 2, 4000, dtype=np.uint8))
    h = root_raised_cosine(sps=4, span=10)
    waveform = matched_filter(pulse_shape(symbols, h, 4), h)
    offset = -0.017
    n = np.arange(len(waveform))
    shifted = waveform * np.exp(2j * np.pi * offset * n / 4)
    estimate = estimate_symbol_spaced_cfo(shifted, 4)
    assert estimate == pytest.approx(offset, abs=5e-4)


def test_preamble_search_recovers_cfo_without_knowing_frame_start():
    rng = np.random.default_rng(14)
    preamble = qpsk_modulate(rng.integers(0, 2, 128, dtype=np.uint8))
    payload = qpsk_modulate(rng.integers(0, 2, 160, dtype=np.uint8))
    h = root_raised_cosine(sps=4, span=10)
    waveform = matched_filter(pulse_shape(np.r_[preamble, payload], h, 4), h)
    offset = -0.031
    n = np.arange(len(waveform))
    shifted = waveform * np.exp(2j * np.pi * offset * n / 4)
    estimate, score = search_preamble_cfo(
        shifted, preamble, 4, len(payload), max_offset=0.06, steps=121
    )
    assert score > 0.98
    assert estimate == pytest.approx(offset, abs=1.1e-3)


def test_preamble_acquisition_finds_timing_phase_and_residual_frequency():
    rng = np.random.default_rng(3)
    preamble = qpsk_modulate(rng.integers(0, 2, 128, dtype=np.uint8))
    payload = qpsk_modulate(rng.integers(0, 2, 80, dtype=np.uint8))
    symbols = np.concatenate([preamble, payload])
    h = root_raised_cosine(sps=4, span=10)
    waveform = pulse_shape(symbols, h, 4)
    phase = 0.61
    residual = 0.0008
    n = np.arange(len(waveform))
    waveform *= np.exp(1j * (phase + 2 * np.pi * residual * n / 4))
    filtered = matched_filter(waveform, h)

    recovered, acq = acquire_preamble(filtered, preamble, 4, len(payload))
    assert acq.score > 0.98
    assert acq.residual_cfo == pytest.approx(residual, abs=2e-4)
    assert np.mean(np.abs(recovered[len(preamble) :] - payload) ** 2) < 2e-3


def test_costas_loop_tracks_small_residual_frequency():
    rng = np.random.default_rng(4)
    symbols = qpsk_modulate(rng.integers(0, 2, 4000, dtype=np.uint8))
    n = np.arange(len(symbols))
    impaired = symbols * np.exp(1j * (0.4 + 2 * np.pi * 0.0005 * n))
    recovered = decision_directed_costas(impaired, loop_bandwidth=0.025)
    # Ignore acquisition transient.
    error = np.mean(np.abs(recovered[300:] - symbols[300:]) ** 2)
    assert error < 2e-3
