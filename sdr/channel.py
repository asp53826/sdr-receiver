"""A reproducible complex-baseband channel.

The impairments are intentionally separate parameters. A receiver that only
works when carrier, clock and gain are already perfect is a modem simulation,
not an SDR receiver.
"""

from dataclasses import dataclass

import numpy as np

from .filters import fractional_delay


@dataclass(frozen=True)
class ChannelConfig:
    ebn0_db: float
    bits_per_symbol: int = 2
    code_rate: float = 1.0
    sps: int = 4
    frequency_offset: float = 0.0
    phase_offset: float = 0.0
    timing_offset: float = 0.0
    gain: float = 1.0

    @property
    def noise_variance(self):
        """Complex matched-filter noise variance N0.

        Unit-energy symbols have Eb = Es / (bits_per_symbol * code_rate).
        Unit-energy RRC filtering preserves that variance at the sampler.
        """
        ebn0 = 10.0 ** (self.ebn0_db / 10.0)
        return 1.0 / (self.bits_per_symbol * self.code_rate * ebn0)


def impair(samples, cfg, rng=None):
    """Apply timing, gain, carrier offset and circular complex AWGN.

    ``frequency_offset`` is cycles per symbol, converted internally to cycles
    per sample. Returns ``(received, noise_variance)``.
    """
    rng = np.random.default_rng() if rng is None else rng
    y = fractional_delay(samples, cfg.timing_offset)
    n = np.arange(len(y), dtype=float)
    phase = cfg.phase_offset + 2.0 * np.pi * cfg.frequency_offset * n / cfg.sps
    y = cfg.gain * y * np.exp(1j * phase)
    # ``ebn0_db`` defines SNR at the receiver. The arbitrary complex-path gain
    # therefore scales signal and noise together; otherwise ``gain`` would be a
    # second hidden SNR knob and a 0.1 gain would silently subtract 20 dB.
    sigma = abs(cfg.gain) * np.sqrt(cfg.noise_variance / 2.0)
    noise = sigma * (rng.normal(size=len(y)) + 1j * rng.normal(size=len(y)))
    return y + noise, cfg.noise_variance
