"""TM-score and Kabsch alignment for RNA/protein structure comparison."""

import numpy as np
from numba import njit, prange


@njit(cache=True, parallel=True)
def _tm_score_inner(pred: np.ndarray, true: np.ndarray, d0: np.float32) -> np.float32:
    """Inner loop: accumulate 1/(1 + di^2/d0^2) over residues with prange."""
    L = pred.shape[0]
    d0_sq = d0 * d0
    total = np.float32(0.0)
    for i in prange(L):
        dx = pred[i, 0] - true[i, 0]
        dy = pred[i, 1] - true[i, 1]
        dz = pred[i, 2] - true[i, 2]
        di_sq = dx * dx + dy * dy + dz * dz
        total += np.float32(1.0) / (np.float32(1.0) + di_sq / d0_sq)
    return total / np.float32(L)


def tm_score(pred: np.ndarray, true: np.ndarray) -> float:
    """Compute TM-score between predicted and true structures.

    Args:
        pred: Predicted coordinates, shape (L, 3) or (L, N, 3).
        true: True coordinates, shape (L, 3) or (L, N, 3).

    Returns:
        TM-score as a float in [0, 1].
    """
    pred = np.asarray(pred, dtype=np.float32)
    true = np.asarray(true, dtype=np.float32)

    # Use first atom if 3-D input
    if pred.ndim == 3:
        pred = pred[:, 0, :]
    if true.ndim == 3:
        true = true[:, 0, :]

    L = pred.shape[0]
    # Standard TM-score length-dependent distance threshold (Zhang & Skolnick, 2004)
    d0 = np.float32(max(0.5, 1.24 * (L - 15) ** (1.0 / 3.0) - 1.8))

    return float(_tm_score_inner(pred, true, d0))


def kabsch_align(P: np.ndarray, Q: np.ndarray) -> np.ndarray:
    """Kabsch alignment: rotate P onto Q minimising RMSD.

    Uses numpy SVD (not njit-compatible) for the rotation matrix,
    then applies the rotation.

    Args:
        P: Mobile coordinates, shape (L, 3) or (L, N, 3).
        Q: Target coordinates, shape (L, 3) or (L, N, 3).

    Returns:
        Aligned copy of P as np.float32 array with same shape as input.
    """
    P = np.asarray(P, dtype=np.float32)
    Q = np.asarray(Q, dtype=np.float32)

    if P.ndim == 3:
        P = P[:, 0, :]
    if Q.ndim == 3:
        Q = Q[:, 0, :]

    # Center both point clouds
    Pc = P - P.mean(axis=0)
    Qc = Q - Q.mean(axis=0)

    # Cross-covariance matrix
    H = Pc.T @ Qc

    U, _S, Vt = np.linalg.svd(H)

    # Correct for reflection (use 1.0 when det is exactly zero)
    d = np.linalg.det(Vt.T @ U.T)
    sign_val = np.float32(np.sign(d)) if d != 0.0 else np.float32(1.0)
    sign_matrix = np.diag(np.array([1.0, 1.0, sign_val], dtype=np.float32))

    R = (Vt.T @ sign_matrix @ U.T).astype(np.float32)
    aligned = (Pc @ R.T + Q.mean(axis=0)).astype(np.float32)
    return aligned


if __name__ == "__main__":
    np.random.seed(42)
    L = 50

    # Synthetic ground-truth structure
    true_coords = np.random.randn(L, 3).astype(np.float32)

    # Predicted = true + small noise  ->  expect high TM-score
    pred_coords = true_coords + 0.1 * np.random.randn(L, 3).astype(np.float32)

    score = tm_score(pred_coords, true_coords)
    print(f"TM-score (close structures):  {score:.4f}")
    assert 0.0 <= score <= 1.0, f"TM-score out of range: {score}"
    assert score > 0.9, f"Expected high TM-score for near-identical structures, got {score}"

    # Random structures -> expect low TM-score
    random_pred = np.random.randn(L, 3).astype(np.float32) * 10.0
    low_score = tm_score(random_pred, true_coords)
    print(f"TM-score (random structures): {low_score:.4f}")
    assert 0.0 <= low_score <= 1.0, f"TM-score out of range: {low_score}"

    # Kabsch alignment should reduce RMSD
    aligned = kabsch_align(pred_coords, true_coords)
    rmsd_before = np.sqrt(np.mean(np.sum((pred_coords - true_coords) ** 2, axis=1)))
    rmsd_after = np.sqrt(np.mean(np.sum((aligned - true_coords) ** 2, axis=1)))
    print(f"RMSD before alignment: {rmsd_before:.4f}")
    print(f"RMSD after  alignment: {rmsd_after:.4f}")
    assert rmsd_after <= rmsd_before + 1e-6, "Alignment should not increase RMSD"

    # Test 3-D input path (L, N, 3)
    pred_3d = np.random.randn(L, 4, 3).astype(np.float32)
    true_3d = pred_3d + 0.05 * np.random.randn(L, 4, 3).astype(np.float32)
    score_3d = tm_score(pred_3d, true_3d)
    print(f"TM-score (3-D input):         {score_3d:.4f}")
    assert 0.0 <= score_3d <= 1.0
    aligned_3d = kabsch_align(pred_3d, true_3d)
    assert aligned_3d.shape == (L, 3)

    print("All self-tests passed.")
