import numpy as np

from sdr.channel import ChannelConfig
from sdr.frame import simulate
from sdr.ldpc import SparseLDPC


def test_uncoded_receiver_survives_joint_impairments():
    rng = np.random.default_rng(11)
    bits = rng.integers(0, 2, 1000, dtype=np.uint8)
    channel = ChannelConfig(
        ebn0_db=12.0,
        frequency_offset=0.012,
        phase_offset=0.8,
        timing_offset=0.33,
        gain=0.35,
    )
    _, _, _, result = simulate(bits, channel, rng=rng)
    assert result.acquisition.score > 0.9
    assert np.mean(result.information != bits) < 0.01


def test_coded_receiver_round_trip_at_high_snr():
    rng = np.random.default_rng(12)
    code = SparseLDPC(information_bits=144)
    bits = rng.integers(0, 2, code.k, dtype=np.uint8)
    channel = ChannelConfig(
        ebn0_db=10.0,
        code_rate=code.rate,
        frequency_offset=-0.009,
        phase_offset=-0.55,
        timing_offset=-0.24,
        gain=1.7,
    )
    _, _, _, result = simulate(bits, channel, code=code, rng=rng)
    assert result.decode is not None
    assert result.decode.converged
    assert np.array_equal(result.information, bits)

