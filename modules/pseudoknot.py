"""
pseudoknot.py
=============
Pseudoknot detection and resolution for RNA base-pair sets.

A pseudoknot occurs when two base pairs (i, j) and (k, l) cross:
``i < k < j < l`` or ``k < i < l < j``.  This module provides a
Numba-accelerated detector that flags every pair participating in at
least one crossing, and a greedy resolver that keeps the higher-scoring
pair (by base-pair probability) when two pairs cross.

Functions:
  detect_pseudoknots          — flag pairs involved in crossings
  resolve_pseudoknots_by_score — greedily remove lower-scored crossing pairs

Usage:
  from modules.pseudoknot import detect_pseudoknots, resolve_pseudoknots_by_score
  flags = detect_pseudoknots(pairs)
  filtered = resolve_pseudoknots_by_score(pairs, bpp_matrix)
"""

from __future__ import annotations

import numpy as np
from numba import njit


# ---------------------------------------------------------------------------
# 1. DETECT PSEUDOKNOTS  (sequential — pairs flag each other: write conflicts)
# ---------------------------------------------------------------------------

@njit(cache=True)
def detect_pseudoknots(pairs: np.ndarray) -> np.ndarray:
    """
    Flag base pairs that participate in at least one pseudoknot crossing.

    Two pairs (i, j) and (k, l) cross iff ``i < k < j < l`` or
    ``k < i < l < j``.  Both pairs are flagged when a crossing is found.

    Parameters
    ----------
    pairs : ndarray, shape (K, 2), int32
        Each row is a base pair (i, j) with ``i < j``.

    Returns
    -------
    flags : ndarray, shape (K,), bool
        True for every pair involved in at least one crossing.
    """
    K = pairs.shape[0]
    flags = np.zeros(K, dtype=np.bool_)
    for a in range(K):
        i = pairs[a, 0]
        j = pairs[a, 1]
        for b in range(a + 1, K):
            k = pairs[b, 0]
            l = pairs[b, 1]
            if (i < k < j < l) or (k < i < l < j):
                flags[a] = True
                flags[b] = True
    return flags


# ---------------------------------------------------------------------------
# 2. RESOLVE PSEUDOKNOTS BY SCORE  (greedy removal of lower-scored crossings)
# ---------------------------------------------------------------------------

@njit(cache=True)
def _score_pairs(pairs: np.ndarray, bpp_matrix: np.ndarray) -> np.ndarray:
    """
    Look up the base-pair probability for each pair.

    Parameters
    ----------
    pairs : ndarray, shape (K, 2), int32
    bpp_matrix : ndarray, shape (L, L), float32

    Returns
    -------
    scores : ndarray, shape (K,), float32
    """
    K = pairs.shape[0]
    scores = np.empty(K, dtype=np.float32)
    for a in range(K):
        scores[a] = bpp_matrix[pairs[a, 0], pairs[a, 1]]
    return scores


@njit(cache=True)
def _resolve_crossings(
    pairs: np.ndarray,
    scores: np.ndarray,
) -> np.ndarray:
    """
    Greedily remove the lower-scored pair from every crossing.

    Parameters
    ----------
    pairs : ndarray, shape (K, 2), int32
    scores : ndarray, shape (K,), float32

    Returns
    -------
    keep : ndarray, shape (K,), bool
        True for pairs that survive resolution.
    """
    K = pairs.shape[0]
    keep = np.ones(K, dtype=np.bool_)
    for a in range(K):
        if not keep[a]:
            continue
        i = pairs[a, 0]
        j = pairs[a, 1]
        for b in range(a + 1, K):
            if not keep[b]:
                continue
            k = pairs[b, 0]
            l = pairs[b, 1]
            if (i < k < j < l) or (k < i < l < j):
                if scores[a] >= scores[b]:
                    keep[b] = False
                else:
                    keep[a] = False
                    break  # a is removed; no need to check further
    return keep


def resolve_pseudoknots_by_score(
    pairs: np.ndarray,
    bpp_matrix: np.ndarray,
) -> np.ndarray:
    """
    Remove pseudoknot-forming pairs, keeping the higher-probability one.

    For every pair of crossing base pairs the one with the lower
    ``bpp_matrix[i, j]`` value is discarded.

    Parameters
    ----------
    pairs : ndarray, shape (K, 2), int32
        Base pairs with ``i < j``.
    bpp_matrix : ndarray, shape (L, L), float32
        Base-pair probability matrix.

    Returns
    -------
    filtered : ndarray, shape (K', 2), int32
        Subset of *pairs* with all pseudoknot crossings resolved.
    """
    pairs = np.asarray(pairs, dtype=np.int32)
    bpp_matrix = np.asarray(bpp_matrix, dtype=np.float32)
    scores = _score_pairs(pairs, bpp_matrix)
    keep = _resolve_crossings(pairs, scores)
    return pairs[keep]


# ---------------------------------------------------------------------------
# Self-test on synthetic data
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    rng = np.random.RandomState(42)
    L = 50

    # Build synthetic base pairs: some nested, some crossing
    nested_pairs = np.array([
        [6, 15],
        [7, 14],
        [8, 12],
        [36, 45],
        [37, 44],
    ], dtype=np.int32)

    # Crossing pairs: (5, 30) and (20, 35) cross because 5 < 20 < 30 < 35
    crossing_pairs = np.array([
        [5, 30],
        [20, 35],
    ], dtype=np.int32)

    pairs = np.concatenate((nested_pairs, crossing_pairs), axis=0)
    K = pairs.shape[0]

    # --- detect_pseudoknots ---
    flags = detect_pseudoknots(pairs)
    print(f"Pair count         : {K}")
    print(f"Pseudoknot flags   : {flags}")
    assert flags.shape == (K,), f"Bad shape: {flags.shape}"
    assert flags.dtype == np.bool_, f"Bad dtype: {flags.dtype}"
    # Nested pairs should NOT be flagged
    assert not flags[:5].any(), "Nested pairs incorrectly flagged"
    # Crossing pairs SHOULD be flagged
    assert flags[5] and flags[6], "Crossing pairs not flagged"
    print(f"Flagged pairs      : {int(flags.sum())} / {K}")

    # --- resolve_pseudoknots_by_score ---
    bpp_matrix = rng.rand(L, L).astype(np.float32)
    # Make (5,30) higher-scored than (20,35) so (20,35) is removed
    bpp_matrix[5, 30] = np.float32(0.95)
    bpp_matrix[20, 35] = np.float32(0.10)

    filtered = resolve_pseudoknots_by_score(pairs, bpp_matrix)
    assert filtered.dtype == np.int32, f"Bad dtype: {filtered.dtype}"
    assert filtered.ndim == 2 and filtered.shape[1] == 2, (
        f"Bad shape: {filtered.shape}"
    )
    # The low-score crossing pair (20,35) should have been removed
    filtered_set = {(int(r[0]), int(r[1])) for r in filtered}
    assert (5, 30) in filtered_set, "(5,30) should survive"
    assert (20, 35) not in filtered_set, "(20,35) should be removed"
    # All nested pairs should survive
    for row in nested_pairs:
        assert (int(row[0]), int(row[1])) in filtered_set, (
            f"Nested pair {row} incorrectly removed"
        )
    # Resolved set should have no remaining crossings
    resolved_flags = detect_pseudoknots(filtered)
    assert not resolved_flags.any(), "Crossings remain after resolution"

    print(f"Pairs after resolve: {filtered.shape[0]} / {K}")
    print("Self-test PASSED ✓")
