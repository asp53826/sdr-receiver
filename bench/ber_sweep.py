#!/usr/bin/env python3
"""Reproducible BER/FER sweep for the complete impaired receiver.

The uncoded and coded paths see the same carrier offset, phase, fractional
timing and gain. Eb/N0 is defined per information bit, so the rate-1/2 coded
waveform correctly pays twice the transmitted energy per information bit.
"""

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from sdr.channel import ChannelConfig
from sdr.frame import simulate
from sdr.ldpc import SparseLDPC
from sdr.modem import qpsk_theoretical_ber


def run_point(ebn0_db, frames, coded, rng, code):
    bit_errors = 0
    total_bits = 0
    frame_errors = 0
    converged = 0
    iterations = []
    acquisition_scores = []

    for _ in range(frames):
        count = code.k if coded else code.n
        information = rng.integers(0, 2, count, dtype=np.uint8)
        active_code = code if coded else None
        channel = ChannelConfig(
            ebn0_db=ebn0_db,
            code_rate=code.rate if coded else 1.0,
            frequency_offset=0.013,
            phase_offset=0.72,
            timing_offset=0.31,
            gain=0.45,
        )
        _frame, _received, _n0, result = simulate(
            information,
            channel,
            code=active_code,
            rng=rng,
        )
        errors = int(np.count_nonzero(result.information != information))
        bit_errors += errors
        total_bits += len(information)
        frame_errors += int(errors > 0)
        acquisition_scores.append(result.acquisition.score)
        if result.decode is not None:
            converged += int(result.decode.converged)
            iterations.append(result.decode.iterations)

    # Report an explicit upper plotting bound for a zero-error observation;
    # the CSV keeps the measured zero rather than laundering it into a claim.
    return {
        "ber": bit_errors / total_bits,
        "bit_errors": bit_errors,
        "total_bits": total_bits,
        "fer": frame_errors / frames,
        "frame_errors": frame_errors,
        "frames": frames,
        "decoder_convergence": converged / frames if coded else "",
        "mean_iterations": float(np.mean(iterations)) if iterations else "",
        "mean_acquisition_score": float(np.mean(acquisition_scores)),
        "zero_error_upper_bound": 0.5 / total_bits if bit_errors == 0 else "",
    }


def benchmark(points, frames, seed):
    rng = np.random.default_rng(seed)
    code = SparseLDPC(information_bits=144, seed=7)
    rows = []
    for ebn0_db in points:
        theory = float(qpsk_theoretical_ber(ebn0_db))
        for coded in (False, True):
            result = run_point(ebn0_db, frames, coded, rng, code)
            rows.append(
                {
                    "ebn0_db": ebn0_db,
                    "mode": "ldpc" if coded else "uncoded",
                    "theoretical_uncoded_ber": theory,
                    **result,
                }
            )
            print(
                f"{ebn0_db:>4.1f} dB  {'LDPC' if coded else 'raw ':>4}"
                f"  BER={result['ber']:.6g} ({result['bit_errors']}/{result['total_bits']})"
                f"  FER={result['fer']:.4f}"
            )
    return rows


def save_csv(rows, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def save_plot(rows, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    uncoded = [row for row in rows if row["mode"] == "uncoded"]
    coded = [row for row in rows if row["mode"] == "ldpc"]
    x = np.array([row["ebn0_db"] for row in uncoded])
    raw_ber = np.array(
        [
            row["ber"] if row["ber"] > 0 else row["zero_error_upper_bound"]
            for row in uncoded
        ]
    )
    coded_ber = np.array(
        [
            row["ber"] if row["ber"] > 0 else row["zero_error_upper_bound"]
            for row in coded
        ]
    )
    raw_fer = np.array(
        [
            row["fer"] if row["fer"] > 0 else 0.5 / row["frames"]
            for row in uncoded
        ]
    )
    coded_fer = np.array(
        [
            row["fer"] if row["fer"] > 0 else 0.5 / row["frames"]
            for row in coded
        ]
    )

    plt.style.use("dark_background")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2))
    fig.patch.set_facecolor("#071018")
    fig.subplots_adjust(left=0.07, right=0.985, bottom=0.18, top=0.82, wspace=0.13)
    for axis in axes:
        axis.set_facecolor("#0b1722")
        axis.grid(True, which="both", alpha=0.18)
        axis.set_xlabel("Eb/N0 (dB)")

    axes[0].semilogy(
        x,
        [row["theoretical_uncoded_ber"] for row in uncoded],
        "--",
        color="#94a3b8",
        label="uncoded theory",
    )
    axes[0].semilogy(x, raw_ber, "o-", color="#38bdf8", label="full receiver")
    axes[0].semilogy(x, coded_ber, "s-", color="#a3e635", label="LDPC")
    axes[0].set_ylabel("information-bit BER")
    axes[0].set_title("Bit errors")
    axes[0].legend()

    axes[1].semilogy(x, raw_fer, "o-", color="#38bdf8", label="uncoded")
    axes[1].semilogy(x, coded_fer, "s-", color="#a3e635", label="LDPC")
    axes[1].set_ylabel("frame-error rate")
    axes[1].set_title("Whole-frame reliability")
    axes[1].legend()
    fig.suptitle(
        "QPSK receiver under CFO + phase + fractional timing + AWGN", y=0.95
    )
    fig.text(
        0.5,
        0.035,
        "Zero-error observations are plotted at 0.5 / trials as a visual bound; measured CSV values remain zero.",
        ha="center",
        color="#94a3b8",
        fontsize=8,
    )
    fig.savefig(path, dpi=180, facecolor=fig.get_facecolor())
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260728)
    parser.add_argument("--points", type=float, nargs="+", default=[0, 2, 4, 6, 8])
    parser.add_argument("--output", type=Path, default=Path("results/ber.csv"))
    args = parser.parse_args()

    rows = benchmark(args.points, args.frames, args.seed)
    save_csv(rows, args.output)
    save_plot(rows, args.output.with_suffix(".png"))


if __name__ == "__main__":
    main()
