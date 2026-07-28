"""A compact regular LDPC code and normalized min-sum decoder.

This is not a wrapped communications-library codec. The Tanner graph is built
progressively with variable degree 3 and check degree 6. Once two checks share
a variable, the constructor prevents them from sharing another; that removes
length-4 cycles, the shortest feedback loops that make belief propagation
recycle its own evidence.

Encoding is derived from the parity-check matrix rather than hard-coded. A
GF(2) row reduction identifies pivot (parity) and free (information) columns.
The reduced matrix then solves every parity bit directly from the information
bits while the transmitted graph remains the original sparse regular H.
"""

from dataclasses import dataclass

import numpy as np


@dataclass
class DecodeResult:
    bits: np.ndarray
    codeword: np.ndarray
    iterations: int
    converged: bool
    used_channel_fallback: bool = False


class SparseLDPC:
    def __init__(self, information_bits=144, column_weight=3, seed=7):
        if information_bits < 12 or information_bits % 2:
            raise ValueError("information_bits must be an even integer >= 12")
        if column_weight < 2:
            raise ValueError("column_weight must be at least 2")
        self.k = int(information_bits)
        self.m = self.k
        self.n = self.k + self.m
        self.rate = self.k / self.n
        self.column_weight = int(column_weight)
        self.seed = int(seed)
        self.H = self._build_h()
        self._reduced, pivots = self._rref(self.H)
        if len(pivots) != self.m:
            raise RuntimeError("parity-check matrix is not full row rank")
        self.parity_indices = np.asarray(pivots, dtype=int)
        self.information_indices = np.asarray(
            [j for j in range(self.n) if j not in set(pivots)], dtype=int
        )
        if len(self.information_indices) != self.k:
            raise RuntimeError("matrix dimension does not match requested code rate")
        self._check_to_vars = [np.flatnonzero(row) for row in self.H]
        self._var_to_checks = [np.flatnonzero(self.H[:, j]) for j in range(self.n)]

    @staticmethod
    def _rref(matrix):
        """Reduced row-echelon form over GF(2), plus pivot columns."""
        out = np.asarray(matrix, dtype=np.uint8).copy()
        row = 0
        pivots = []
        for column in range(out.shape[1]):
            candidates = np.flatnonzero(out[row:, column])
            if len(candidates) == 0:
                continue
            pivot = row + int(candidates[0])
            out[[row, pivot]] = out[[pivot, row]]
            for other in np.flatnonzero(out[:, column]):
                if other != row:
                    out[other] ^= out[row]
            pivots.append(column)
            row += 1
            if row == out.shape[0]:
                break
        return out, pivots

    def _build_h(self):
        """Progressive (3, 6)-regular graph with no length-4 cycles."""
        edges = self.n * self.column_weight
        if edges % self.m:
            raise ValueError("dimensions do not produce an integer check degree")
        check_degree = edges // self.m
        rng = np.random.default_rng(self.seed)
        for _attempt in range(100):
            H = np.zeros((self.m, self.n), dtype=np.uint8)
            capacity = np.full(self.m, check_degree, dtype=int)
            used_check_pairs = set()
            complete = True

            for column in range(self.n):
                chosen = []
                for _edge in range(self.column_weight):
                    candidates = [
                        check
                        for check in range(self.m)
                        if capacity[check] > 0
                        and check not in chosen
                        and all(
                            (min(check, other), max(check, other))
                            not in used_check_pairs
                            for other in chosen
                        )
                    ]
                    if not candidates:
                        complete = False
                        break
                    # Squared residual capacity strongly balances row degree
                    # while retaining randomness among equally loaded checks.
                    weight = np.asarray(
                        [capacity[check] ** 2 for check in candidates], dtype=float
                    )
                    weight /= weight.sum()
                    chosen.append(int(rng.choice(candidates, p=weight)))
                if not complete:
                    break

                H[chosen, column] = 1
                for check in chosen:
                    capacity[check] -= 1
                for i, check in enumerate(chosen):
                    for other in chosen[:i]:
                        used_check_pairs.add(
                            (min(check, other), max(check, other))
                        )

            if complete and not np.any(capacity):
                _, pivots = self._rref(H)
                if len(pivots) == self.m:
                    return H
        raise RuntimeError("could not construct a full-rank regular LDPC graph")

    def encode(self, information):
        u = np.asarray(information, dtype=np.uint8).reshape(-1)
        if len(u) != self.k:
            raise ValueError(f"expected {self.k} information bits, got {len(u)}")
        if np.any(u > 1):
            raise ValueError("information must contain only 0 and 1")

        codeword = np.zeros(self.n, dtype=np.uint8)
        codeword[self.information_indices] = u
        # Each pivot row is x_p + sum_j R[row,j] x_j = 0.
        for row, pivot in enumerate(self.parity_indices):
            codeword[pivot] = (
                self._reduced[row, self.information_indices] @ u
            ) & 1
        if not self.is_codeword(codeword):
            raise RuntimeError("encoder produced a non-codeword")
        return codeword

    def syndrome(self, bits):
        bits = np.asarray(bits, dtype=np.uint8).reshape(-1)
        if len(bits) != self.n:
            raise ValueError(f"expected {self.n} code bits, got {len(bits)}")
        return (self.H @ bits) & 1

    def is_codeword(self, bits):
        return not np.any(self.syndrome(bits))

    def decode(self, channel_llr, max_iterations=50, alpha=0.8):
        """Normalized min-sum belief propagation.

        LLR sign convention matches ``qpsk_soft_llr``: positive favours zero.
        ``alpha < 1`` corrects the raw min-sum magnitude overestimate.
        """
        llr = np.asarray(channel_llr, dtype=float).reshape(-1)
        if len(llr) != self.n:
            raise ValueError(f"expected {self.n} LLRs, got {len(llr)}")
        if not 0.0 < alpha <= 1.0:
            raise ValueError("alpha must be in (0, 1]")
        channel_hard = (llr < 0.0).astype(np.uint8)

        # Dense message arrays keep the implementation inspectable. Values on
        # non-edges stay zero and are never read.
        q = np.zeros((self.m, self.n), dtype=float)
        r = np.zeros_like(q)
        for j, checks in enumerate(self._var_to_checks):
            q[checks, j] = llr[j]

        posterior = llr.copy()
        for iteration in range(1, max_iterations + 1):
            # Check-to-variable messages: sign product times first/second min.
            for i, variables in enumerate(self._check_to_vars):
                values = q[i, variables]
                signs = np.where(values < 0.0, -1.0, 1.0)
                magnitudes = np.abs(values)
                min_index = int(np.argmin(magnitudes))
                first = magnitudes[min_index]
                if len(magnitudes) > 1:
                    second = np.min(np.delete(magnitudes, min_index))
                else:
                    second = first
                sign_product = np.prod(signs)
                for local, j in enumerate(variables):
                    magnitude = second if local == min_index else first
                    r[i, j] = alpha * sign_product * signs[local] * magnitude

            posterior = llr.copy()
            for j, checks in enumerate(self._var_to_checks):
                posterior[j] += r[checks, j].sum()
            hard = (posterior < 0.0).astype(np.uint8)
            if self.is_codeword(hard):
                return DecodeResult(
                    hard[self.information_indices],
                    hard,
                    iteration,
                    True,
                    False,
                )

            # Extrinsic variable-to-check update.
            for j, checks in enumerate(self._var_to_checks):
                total = posterior[j]
                q[checks, j] = total - r[checks, j]

        # A failed parity check is an explicit erasure signal. Iterative
        # messages can oscillate and leave a harder-decision vector worse than
        # the channel input, so do not silently turn a detected decoder failure
        # into extra bit errors. Return the channel hard decisions and mark the
        # fallback; system-level evaluation still counts the frame as failed.
        hard = channel_hard
        return DecodeResult(
            hard[self.information_indices],
            hard,
            max_iterations,
            False,
            True,
        )
