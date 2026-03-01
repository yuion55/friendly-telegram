"""Topological genus invariants for RNA secondary structure."""

import numpy as np
from numba import njit


@njit(cache=True)
def compute_genus_from_pairs(pairs, L):
    """Estimate genus from base pairs by counting crossings.

    Args:
        pairs: int32 array of shape (K, 2) with paired indices.
        L: sequence length.

    Returns:
        Genus estimate as int: (crossings + 1) // 2.
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
    return (crossings + 1) // 2


def genus_from_bpp(bpp_matrix, threshold=0.5):
    """Compute genus from a base-pair probability matrix.

    Args:
        bpp_matrix: float32 array of shape (L, L) with pair probabilities.
        threshold: minimum probability to consider a pair.

    Returns:
        Genus estimate as int.
    """
    L = bpp_matrix.shape[0]
    pair_list = []
    for i in range(L):
        for j in range(i + 4, L):
            if bpp_matrix[i, j] > threshold:
                pair_list.append((i, j))
    if len(pair_list) == 0:
        pairs = np.empty((0, 2), dtype=np.int32)
    else:
        pairs = np.array(pair_list, dtype=np.int32)
    return compute_genus_from_pairs(pairs, L)


@njit(cache=True)
def genus_consistency_loss(pred_genus, true_genus):
    """Absolute difference between predicted and true genus.

    Args:
        pred_genus: predicted genus value.
        true_genus: true genus value.

    Returns:
        Loss as float.
    """
    diff = pred_genus - true_genus
    if diff < 0:
        diff = -diff
    return np.float32(diff)


if __name__ == "__main__":
    L = 50
    np.random.seed(42)
    bpp = np.zeros((L, L), dtype=np.float32)
    # Insert a few synthetic pairs with high probability
    test_pairs = [(2, 18), (5, 30), (10, 25), (15, 40), (20, 45)]
    for i, j in test_pairs:
        bpp[i, j] = 0.9
        bpp[j, i] = 0.9

    g = genus_from_bpp(bpp, threshold=0.5)
    print(f"Sequence length: {L}")
    print(f"Number of pairs above threshold: {len(test_pairs)}")
    print(f"Estimated genus: {g}")

    loss = genus_consistency_loss(np.float32(g), np.float32(0))
    print(f"Consistency loss (pred={g}, true=0): {loss}")

    # Verify crossing count manually for sanity
    pairs_arr = np.array(test_pairs, dtype=np.int32)
    g2 = compute_genus_from_pairs(pairs_arr, L)
    assert g == g2, "genus_from_bpp and compute_genus_from_pairs must agree"
    print("Self-test passed.")
