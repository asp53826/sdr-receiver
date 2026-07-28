"""Burst construction and the complete receive path."""

from dataclasses import dataclass

import numpy as np

from .channel import ChannelConfig, impair
from .filters import matched_filter, pulse_shape, root_raised_cosine
from .ldpc import DecodeResult, SparseLDPC
from .modem import qpsk_hard_demodulate, qpsk_modulate, qpsk_soft_llr
from .sync import (
    Acquisition,
    acquire_preamble,
    agc,
    correct_cfo,
    decision_directed_costas,
    search_preamble_cfo,
)


def make_preamble(length=64, seed=0):
    """Deterministic balanced QPSK acquisition preamble."""
    if length < 16:
        raise ValueError("preamble must contain at least 16 symbols")
    rng = np.random.default_rng(seed)
    bits = rng.integers(0, 2, 2 * length, dtype=np.uint8)
    return qpsk_modulate(bits)


@dataclass
class Frame:
    information: np.ndarray
    transmitted_bits: np.ndarray
    preamble: np.ndarray
    waveform: np.ndarray
    taps: np.ndarray
    sps: int
    code: SparseLDPC | None

    @property
    def payload_symbols(self):
        return len(self.transmitted_bits) // 2


@dataclass
class ReceiveResult:
    information: np.ndarray
    hard_bits: np.ndarray
    payload_symbols: np.ndarray
    acquisition: Acquisition
    coarse_cfo: float
    decode: DecodeResult | None


def build_frame(
    information,
    code=None,
    sps=4,
    rolloff=0.35,
    span=10,
    preamble_length=96,
):
    information = np.asarray(information, dtype=np.uint8).reshape(-1)
    transmitted = code.encode(information) if code is not None else information.copy()
    if len(transmitted) % 2:
        raise ValueError("the transmitted bit count must be even")
    preamble = make_preamble(preamble_length)
    symbols = np.concatenate([preamble, qpsk_modulate(transmitted)])
    taps = root_raised_cosine(rolloff, sps, span)
    return Frame(
        information=information,
        transmitted_bits=transmitted,
        preamble=preamble,
        waveform=pulse_shape(symbols, taps, sps),
        taps=taps,
        sps=sps,
        code=code,
    )


def transmit(frame, channel, rng=None):
    """Pass a frame waveform through the configured channel."""
    expected_rate = frame.code.rate if frame.code is not None else 1.0
    if not np.isclose(channel.code_rate, expected_rate):
        raise ValueError(
            f"channel code_rate={channel.code_rate} does not match frame rate={expected_rate}"
        )
    return impair(frame.waveform, channel, rng)


def receive(frame, received, use_costas=True):
    """Run AGC, blind CFO, matched filter, acquisition, demap and decode."""
    y, _agc_gain = agc(received)
    y = matched_filter(y, frame.taps)
    coarse_cfo, _coarse_score = search_preamble_cfo(
        y,
        frame.preamble,
        frame.sps,
        frame.payload_symbols,
    )
    y = correct_cfo(y, coarse_cfo, sps=frame.sps)
    symbols, acquisition = acquire_preamble(
        y,
        frame.preamble,
        frame.sps,
        frame.payload_symbols,
    )
    payload = symbols[len(frame.preamble) :]
    if use_costas:
        payload = decision_directed_costas(payload, loop_bandwidth=0.01)

    llr = qpsk_soft_llr(payload, acquisition.noise_variance)
    hard = (llr < 0.0).astype(np.uint8)
    if frame.code is None:
        information = hard
        decoded = None
    else:
        decoded = frame.code.decode(llr)
        information = decoded.bits
    return ReceiveResult(
        information=information,
        hard_bits=hard,
        payload_symbols=payload,
        acquisition=acquisition,
        coarse_cfo=coarse_cfo,
        decode=decoded,
    )


def simulate(information, channel, code=None, rng=None, **frame_kwargs):
    frame = build_frame(information, code=code, **frame_kwargs)
    received, noise_variance = transmit(frame, channel, rng)
    return frame, received, noise_variance, receive(frame, received)
