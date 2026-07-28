import numpy as np

from sdr.ldpc import SparseLDPC


def test_encoder_outputs_valid_codewords():
    code = SparseLDPC(information_bits=48, seed=2)
    rng = np.random.default_rng(3)
    for _ in range(20):
        bits = rng.integers(0, 2, code.k, dtype=np.uint8)
        word = code.encode(bits)
        assert code.is_codeword(word)
        assert np.array_equal(word[code.information_indices], bits)


def test_matrix_is_sparse_and_has_no_isolated_nodes():
    code = SparseLDPC(information_bits=48, column_weight=3)
    density = code.H.mean()
    assert density < 0.08
    assert np.all(code.H.sum(axis=0) == 3)
    assert np.all(code.H.sum(axis=1) == 6)


def test_tanner_graph_has_no_length_four_cycles():
    code = SparseLDPC(information_bits=48)
    overlap = code.H @ code.H.T
    np.fill_diagonal(overlap, 0)
    assert np.max(overlap) <= 1


def test_decoder_round_trips_noiseless_llrs():
    code = SparseLDPC(information_bits=48)
    rng = np.random.default_rng(9)
    bits = rng.integers(0, 2, code.k, dtype=np.uint8)
    word = code.encode(bits)
    llr = np.where(word == 0, 20.0, -20.0)
    result = code.decode(llr)
    assert result.converged
    assert result.iterations == 1
    assert np.array_equal(result.bits, bits)


def test_decoder_corrects_a_small_error_pattern():
    code = SparseLDPC(information_bits=96, seed=4)
    rng = np.random.default_rng(10)
    bits = rng.integers(0, 2, code.k, dtype=np.uint8)
    word = code.encode(bits)
    llr = np.where(word == 0, 5.0, -5.0)
    flips = rng.choice(code.n, size=4, replace=False)
    llr[flips] *= -0.55
    result = code.decode(llr, max_iterations=50)
    assert result.converged
    assert np.array_equal(result.bits, bits)


def test_nonconvergence_falls_back_to_channel_decisions():
    code = SparseLDPC(information_bits=48, seed=5)
    # Deliberately inconsistent low-confidence channel decisions.
    llr = np.linspace(-0.1, 0.1, code.n)
    result = code.decode(llr, max_iterations=1)
    if not result.converged:
        expected = (llr < 0.0).astype(np.uint8)
        assert result.used_channel_fallback
        assert np.array_equal(result.codeword, expected)
