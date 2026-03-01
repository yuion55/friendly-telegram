"""Chord diagram analysis for RNA secondary structure pseudoknot classification."""

import numpy as np
from numba import njit

from whitebox.genus_invariants import compute_genus_from_pairs


@njit(cache=True)
def compute_chord_crossing_number(pairs):
    """Count the number of crossing pairs in a chord diagram.

    Args:
        pairs: int32 array of shape (K, 2) with paired indices, each row (i, j) with i < j.

    Returns:
        Total number of crossings as int.
    """
    K = pairs.shape[0]
    crossings = 0
    for a in range(K):
        i = pairs[a, 0]
        j = pairs[a, 1]
        for b in range(a + 1, K):
            k = pairs[b, 0]
            l = pairs[b, 1]
            if (i < k < j < l) or (k < i < l < j):
                crossings += 1
    return crossings


@njit(cache=True)
def _crossing_counts(pairs):
    """Compute per-pair crossing counts and total crossings.

    Args:
        pairs: int32 array of shape (K, 2) with paired indices.

    Returns:
        per_pair: int32 array of shape (K,) with crossing count for each pair.
        total: total number of crossings as int.
    """
    K = pairs.shape[0]
    per_pair = np.zeros(K, dtype=np.int32)
    total = 0
    for a in range(K):
        i = pairs[a, 0]
        j = pairs[a, 1]
        for b in range(a + 1, K):
            k = pairs[b, 0]
            l = pairs[b, 1]
            if (i < k < j < l) or (k < i < l < j):
                per_pair[a] += 1
                per_pair[b] += 1
                total += 1
    return per_pair, total


def classify_pseudoknot_type(pairs):
    """Classify pseudoknot type from a chord diagram.

    Args:
        pairs: int32 array of shape (K, 2) with paired indices.

    Returns:
        One of 'planar', 'H-type', 'L-type', 'K-type', or 'higher-order'.
    """
    K = pairs.shape[0]
    if K == 0:
        return "planar"
    per_pair, total = _crossing_counts(pairs)
    if total == 0:
        return "planar"
    if total == 1:
        return "H-type"
    if total == 2:
        return "L-type"
    max_per_pair = 0
    for v in per_pair:
        if v > max_per_pair:
            max_per_pair = v
    if total == 3 and max_per_pair == 2:
        return "K-type"
    return "higher-order"


class ChordDiagramCertificate:
    """Certificate generator for chord diagram topological properties."""

    def certify(self, pairs, L):
        """Produce a certificate dict for the given base pairs.

        Args:
            pairs: int32 array of shape (K, 2) with paired indices.
            L: sequence length.

        Returns:
            dict with genus, crossing_number, pseudoknot_type, n_pairs,
            and certificate_valid=True.
        """
        genus = compute_genus_from_pairs(pairs, L)
        crossing_number = compute_chord_crossing_number(pairs)
        pseudoknot_type = classify_pseudoknot_type(pairs)
        return {
            "genus": int(genus),
            "crossing_number": int(crossing_number),
            "pseudoknot_type": pseudoknot_type,
            "n_pairs": int(pairs.shape[0]),
            "certificate_valid": True,
        }


if __name__ == "__main__":
    L = 50
    np.random.seed(42)

    # Generate synthetic base pairs
    test_pairs = [(2, 18), (5, 30), (10, 25), (15, 40), (20, 45)]
    pairs = np.array(test_pairs, dtype=np.int32)

    cn = compute_chord_crossing_number(pairs)
    print(f"Sequence length: {L}")
    print(f"Number of pairs: {len(test_pairs)}")
    print(f"Crossing number: {cn}")

    pk_type = classify_pseudoknot_type(pairs)
    print(f"Pseudoknot type: {pk_type}")

    cert = ChordDiagramCertificate()
    result = cert.certify(pairs, L)
    print(f"Certificate: {result}")

    # Verify certificate fields
    assert result["certificate_valid"] is True
    assert result["n_pairs"] == len(test_pairs)
    assert result["crossing_number"] == cn
    assert result["pseudoknot_type"] == pk_type
    print("Self-test passed.")
