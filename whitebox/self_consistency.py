"""Self-consistency checks for RNA structure prediction round-trips."""

import os
import sys

import numpy as np
from numba import njit

# Ensure the project root is importable when running as a standalone script.
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

try:
    from metrics.tm_score import tm_score
    HAS_TM_SCORE = True
except ImportError:
    HAS_TM_SCORE = False


@njit(cache=True)
def sequence_recovery_rate(true_seq: np.ndarray, pred_seq: np.ndarray) -> float:
    """Fraction of positions where predicted sequence matches the true sequence.

    Args:
        true_seq: Ground-truth sequence, int8 array of shape (L,).
        pred_seq: Predicted sequence, int8 array of shape (L,).

    Returns:
        Recovery rate as a float in [0, 1].
    """
    L = true_seq.shape[0]
    matches = 0
    for i in range(L):
        if true_seq[i] == pred_seq[i]:
            matches += 1
    return matches / L


class RoundtripConsistencyFilter:
    """Filter structure candidates by self-consistency TM-score.

    Performs an inverse-folding round-trip: structure -> sequence -> structure,
    then keeps candidates whose round-trip TM-score meets the threshold.

    Args:
        inverse_model: Duck-typed object with a ``.predict(structure)`` method
            that returns a recovered sequence (int8 array).
        forward_model: Duck-typed object with a ``.predict(sequence)`` method
            that returns a predicted structure (float32 coordinate array).
        sc_tm_threshold: Minimum self-consistency TM-score to keep a candidate.
    """

    def __init__(self, inverse_model, forward_model, sc_tm_threshold: float = 0.6):
        if not HAS_TM_SCORE:
            raise ImportError(
                "metrics.tm_score is required for RoundtripConsistencyFilter"
            )
        self.inverse_model = inverse_model
        self.forward_model = forward_model
        self.sc_tm_threshold = sc_tm_threshold

    def compute_sc_tm(self, structure: np.ndarray, orig_sequence: np.ndarray) -> float:
        """Compute self-consistency TM-score for a single structure.

        Args:
            structure: Coordinate array, shape (L, 3) or (L, N, 3).
            orig_sequence: Original sequence, int8 array of shape (L,).

        Returns:
            Self-consistency TM-score as a float in [0, 1].
        """
        recovered_seq = self.inverse_model.predict(structure)
        sc_structure = self.forward_model.predict(recovered_seq)

        # Use first-atom coords for TM-score comparison
        s1 = np.asarray(structure, dtype=np.float32)
        s2 = np.asarray(sc_structure, dtype=np.float32)
        if s1.ndim == 3:
            s1 = s1[:, 0, :]
        if s2.ndim == 3:
            s2 = s2[:, 0, :]

        return tm_score(s1, s2)

    def filter(self, candidates: list, orig_sequence: np.ndarray) -> list:
        """Keep candidates whose sc-TM meets the threshold.

        Args:
            candidates: List of coordinate arrays (structures).
            orig_sequence: Original sequence, int8 array of shape (L,).

        Returns:
            Filtered list of candidates. Falls back to the first candidate
            if none pass the threshold.
        """
        passed = []
        for cand in candidates:
            sc = self.compute_sc_tm(cand, orig_sequence)
            if sc >= self.sc_tm_threshold:
                passed.append(cand)
        if len(passed) == 0:
            return [candidates[0]]
        return passed


if __name__ == "__main__":
    np.random.seed(42)
    L = 50

    # --- Test sequence_recovery_rate ---
    true_seq = np.random.randint(0, 4, size=L).astype(np.int8)
    pred_seq = true_seq.copy()
    rate_perfect = sequence_recovery_rate(true_seq, pred_seq)
    assert rate_perfect == 1.0, f"Expected 1.0 for identical seqs, got {rate_perfect}"

    pred_seq_noisy = true_seq.copy()
    pred_seq_noisy[0] = (true_seq[0] + 1) % 4
    rate_one_off = sequence_recovery_rate(true_seq, pred_seq_noisy)
    expected = (L - 1) / L
    assert abs(rate_one_off - expected) < 1e-9, f"Expected {expected}, got {rate_one_off}"
    print(f"sequence_recovery_rate (perfect): {rate_perfect:.4f}")
    print(f"sequence_recovery_rate (1 diff):  {rate_one_off:.4f}")

    # --- Test RoundtripConsistencyFilter with mock models ---
    class MockInverseModel:
        """Returns a fixed recovered sequence."""
        def predict(self, structure):
            L = structure.shape[0]
            return np.zeros(L, dtype=np.int8)

    class MockForwardModel:
        """Returns structure close to a stored reference (simulates good round-trip)."""
        def __init__(self, ref_structure):
            self.ref = ref_structure

        def predict(self, sequence):
            noise = 0.05 * np.random.randn(*self.ref.shape).astype(np.float32)
            return self.ref + noise

    true_coords = np.random.randn(L, 3).astype(np.float32)

    inv_model = MockInverseModel()
    fwd_model = MockForwardModel(true_coords)
    filt = RoundtripConsistencyFilter(inv_model, fwd_model, sc_tm_threshold=0.6)

    sc = filt.compute_sc_tm(true_coords, true_seq)
    print(f"sc-TM (good round-trip): {sc:.4f}")
    assert 0.0 <= sc <= 1.0, f"sc-TM out of range: {sc}"
    assert sc > 0.8, f"Expected high sc-TM for near-identical round-trip, got {sc}"

    # Filter: good candidate should pass, random candidate may not
    good_candidate = true_coords + 0.01 * np.random.randn(L, 3).astype(np.float32)
    bad_candidate = np.random.randn(L, 3).astype(np.float32) * 50.0

    # Use a forward model that returns structure close to good_candidate
    fwd_good = MockForwardModel(good_candidate)
    filt_good = RoundtripConsistencyFilter(inv_model, fwd_good, sc_tm_threshold=0.6)
    result = filt_good.filter([good_candidate, bad_candidate], true_seq)
    assert len(result) >= 1, "Filter should return at least one candidate"
    print(f"Filter kept {len(result)} of 2 candidates")

    # Fallback: if threshold is impossibly high, fall back to first candidate
    filt_strict = RoundtripConsistencyFilter(inv_model, fwd_model, sc_tm_threshold=1.1)
    fallback = filt_strict.filter([good_candidate, bad_candidate], true_seq)
    assert len(fallback) == 1, "Should fall back to first candidate"
    assert np.array_equal(fallback[0], good_candidate), "Fallback should be first candidate"
    print("Fallback to first candidate: OK")

    print("All self-tests passed.")
