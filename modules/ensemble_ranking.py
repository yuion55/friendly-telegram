"""Ensemble ranking via pairwise TM-score matrix and proxy score fusion."""

import numpy as np
from numba import njit, prange


@njit(cache=True, parallel=True)
def pairwise_tm_matrix(coords_flat, N, L):
    """Compute pairwise TM-scores for N structures of length L.

    Parameters
    ----------
    coords_flat : ndarray, shape (N*L, 3), float32
        Flat array of coordinates; conceptually (N, L, 3).
    N : int
        Number of structures.
    L : int
        Number of residues per structure.

    Returns
    -------
    tm : ndarray, shape (N, N), float32
    """
    d0 = max(0.5, 1.24 * (L - 15) ** (1.0 / 3.0) - 1.8)
    d0_sq = np.float32(d0 * d0)
    tm = np.zeros((N, N), dtype=np.float32)
    for i in prange(N):
        off_i = i * L
        for j in range(N):
            if i == j:
                tm[i, j] = np.float32(1.0)
                continue
            score = np.float32(0.0)
            off_j = j * L
            for k in range(L):
                dx = coords_flat[off_i + k, 0] - coords_flat[off_j + k, 0]
                dy = coords_flat[off_i + k, 1] - coords_flat[off_j + k, 1]
                dz = coords_flat[off_i + k, 2] - coords_flat[off_j + k, 2]
                di_sq = dx * dx + dy * dy + dz * dz
                score += np.float32(1.0) / (np.float32(1.0) + di_sq / d0_sq)
            tm[i, j] = score / np.float32(L)
    return tm


def rank_ensemble(candidates, proxy_scores, n_select=5):
    """Rank candidate structures by centroid TM-score fused with proxy scores.

    Final score = 0.6 * centroid (mean pairwise TM-score) +
                  0.4 * proxy_scores (normalized to [0, 1]).

    Parameters
    ----------
    candidates : list of ndarray, each shape (L, 3)
        Candidate coordinate arrays (float32).
    proxy_scores : array-like, shape (N,)
        Per-candidate quality proxy scores.
    n_select : int
        Number of top candidates to return.

    Returns
    -------
    selected : ndarray of int
        Indices of the top *n_select* candidates.
    """
    N = len(candidates)
    L = candidates[0].shape[0]
    coords_flat = np.concatenate(candidates, axis=0).astype(np.float32)

    tm = pairwise_tm_matrix(coords_flat, N, L)

    centroid = tm.mean(axis=1).astype(np.float32)

    proxy = np.asarray(proxy_scores, dtype=np.float32)
    p_min = proxy.min()
    p_max = proxy.max()
    if p_max - p_min > 0:
        proxy_norm = (proxy - p_min) / (p_max - p_min)
    else:
        proxy_norm = np.ones_like(proxy)

    combined = np.float32(0.6) * centroid + np.float32(0.4) * proxy_norm
    order = np.argsort(-combined)
    return order[:n_select]


if __name__ == "__main__":
    rng = np.random.default_rng(42)
    L, N = 50, 10
    candidates = [rng.standard_normal((L, 3)).astype(np.float32) for _ in range(N)]
    proxy_scores = rng.random(N).astype(np.float32)

    selected = rank_ensemble(candidates, proxy_scores, n_select=5)

    coords_flat = np.concatenate(candidates, axis=0).astype(np.float32)
    tm = pairwise_tm_matrix(coords_flat, N, L)

    assert tm.shape == (N, N), f"Expected ({N},{N}), got {tm.shape}"
    assert tm.dtype == np.float32, f"Expected float32, got {tm.dtype}"
    for i in range(N):
        assert abs(tm[i, i] - 1.0) < 1e-5, f"Diagonal tm[{i},{i}]={tm[i, i]}"
    for i in range(N):
        for j in range(N):
            assert abs(tm[i, j] - tm[j, i]) < 1e-5, "TM matrix not symmetric"

    assert len(selected) == 5, f"Expected 5 selected, got {len(selected)}"
    assert selected.dtype in (np.intp, np.int64, np.int32)
    print("All self-tests passed.")
