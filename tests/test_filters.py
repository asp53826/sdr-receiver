import numpy as np

from sdr.filters import matched_filter, pulse_shape, root_raised_cosine


def test_rrc_is_symmetric_and_unit_energy():
    h = root_raised_cosine(beta=0.35, sps=4, span=10)
    assert len(h) == 41
    assert np.allclose(h, h[::-1])
    np.testing.assert_allclose(np.sum(h * h), 1.0)


def test_rrc_pair_has_nyquist_zero_crossings():
    h = root_raised_cosine(beta=0.35, sps=4, span=12)
    raised_cosine = np.convolve(h, h[::-1])
    centre = len(raised_cosine) // 2
    symbol_spaced = raised_cosine[centre - 20 : centre + 21 : 4]
    assert symbol_spaced[len(symbol_spaced) // 2] == np.max(symbol_spaced)
    sidelobes = np.delete(symbol_spaced, len(symbol_spaced) // 2)
    assert np.max(np.abs(sidelobes)) < 1e-2


def test_pulse_shape_and_match_filter_recover_impulse_symbols():
    h = root_raised_cosine(beta=0.35, sps=4, span=10)
    symbols = np.array([1 + 0j, 0j, 0j])
    waveform = pulse_shape(symbols, h, 4)
    out = matched_filter(waveform, h)
    delay = len(h) - 1
    np.testing.assert_allclose(out[delay], 1.0, atol=1e-12)
    assert abs(out[delay + 4]) < 1e-2
