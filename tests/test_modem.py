import numpy as np
import pytest

from sdr.modem import (
    qpsk_hard_demodulate,
    qpsk_modulate,
    qpsk_soft_llr,
    qpsk_theoretical_ber,
)


def test_all_qpsk_labels_round_trip():
    bits = np.array([0, 0, 0, 1, 1, 1, 1, 0], dtype=np.uint8)
    symbols = qpsk_modulate(bits)
    assert np.array_equal(qpsk_hard_demodulate(symbols), bits)
    assert np.allclose(np.abs(symbols), 1.0)


def test_llr_sign_matches_transmitted_bits():
    bits = np.array([0, 1, 1, 0, 0, 0], dtype=np.uint8)
    llr = qpsk_soft_llr(qpsk_modulate(bits), noise_variance=0.25)
    assert np.array_equal(llr < 0.0, bits)


def test_qpsk_rejects_odd_bit_count():
    with pytest.raises(ValueError):
        qpsk_modulate([0, 1, 0])


def test_theoretical_curve_is_monotone_and_known_at_zero_db():
    ber = qpsk_theoretical_ber(np.array([0.0, 3.0, 6.0]))
    assert np.all(np.diff(ber) < 0.0)
    assert ber[0] == pytest.approx(0.0786496, rel=1e-5)

