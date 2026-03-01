"""TM-score and Kabsch alignment for RNA/protein structure comparison.

Provides CPU (@njit) and GPU (numba.cuda) implementations with auto-selection.
"""

import numpy as np
from numba import njit, prange

# Graceful CUDA import
try:
    from numba import cuda as _numba_cuda
    _CUDA_AVAILABLE = _numba_cuda.is_available()
except Exception:
    _CUDA_AVAILABLE = False


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


def _compute_d0(L):
    """Standard TM-score length-dependent distance threshold (Zhang & Skolnick, 2004)."""
    return np.float32(max(0.5, 1.24 * (L - 15) ** (1.0 / 3.0) - 1.8))


def _prep_coords(arr):
    """Normalize input to (L, 3) float32."""
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim == 3:
        arr = arr[:, 0, :]
    return np.ascontiguousarray(arr)


def tm_score_cpu(pred: np.ndarray, true: np.ndarray) -> float:
    """Compute TM-score on CPU using @njit inner loop.

    Args:
        pred: Predicted coordinates, shape (L, 3) or (L, N, 3).
        true: True coordinates, shape (L, 3) or (L, N, 3).

    Returns:
        TM-score as a float in [0, 1].
    """
    pred = _prep_coords(pred)
    true = _prep_coords(true)
    L = pred.shape[0]
    d0 = _compute_d0(L)
    return float(_tm_score_inner(pred, true, d0))


def tm_score_gpu(pred: np.ndarray, true: np.ndarray) -> float:
    """Compute TM-score on GPU using numba.cuda kernel.

    Falls back to CPU if CUDA is not available.

    Args:
        pred: Predicted coordinates, shape (L, 3) or (L, N, 3).
        true: True coordinates, shape (L, 3) or (L, N, 3).

    Returns:
        TM-score as a float in [0, 1].
    """
    if not _CUDA_AVAILABLE:
        return tm_score_cpu(pred, true)

    from gpu_kernels import tm_score_gpu_kernel

    pred = _prep_coords(pred)
    true = _prep_coords(true)
    L = pred.shape[0]
    d0 = _compute_d0(L)

    pred_flat = _numba_cuda.to_device(np.ascontiguousarray(pred.reshape(-1)))
    true_flat = _numba_cuda.to_device(np.ascontiguousarray(true.reshape(-1)))
    d0_arr = _numba_cuda.to_device(np.array([d0], dtype=np.float32))
    L_arr = np.array([L], dtype=np.int32)
    out = _numba_cuda.device_array(1, dtype=np.float32)

    block_size = min(L, 1024)
    tm_score_gpu_kernel[(1,), (block_size,)](pred_flat, true_flat, L_arr, d0_arr, out)
    return float(out.copy_to_host()[0])


def tm_score_batch_gpu(preds: list, true: np.ndarray) -> np.ndarray:
    """Compute TM-scores of N predictions against one true structure on GPU.

    Args:
        preds: List of (L, 3) float32 predicted coordinate arrays.
        true: True coordinates, shape (L, 3) or (L, N, 3).

    Returns:
        (N,) float32 array of TM-scores.
    """
    true = _prep_coords(true)
    N = len(preds)
    L = true.shape[0]
    d0 = _compute_d0(L)

    preds_stacked = np.concatenate(
        [_prep_coords(p) for p in preds], axis=0
    ).astype(np.float32)

    if _CUDA_AVAILABLE:
        from gpu_kernels import batch_tm_score_kernel

        flat_cands = _numba_cuda.to_device(
            np.ascontiguousarray(preds_stacked.reshape(-1))
        )
        flat_true = _numba_cuda.to_device(
            np.ascontiguousarray(true.reshape(-1))
        )
        d0_arr = _numba_cuda.to_device(np.array([d0], dtype=np.float32))
        L_arr = np.array([L], dtype=np.int32)
        N_arr = np.array([N], dtype=np.int32)
        d_scores = _numba_cuda.device_array(N, dtype=np.float32)

        block_size = min(L, 512)
        batch_tm_score_kernel[(N,), (block_size,)](
            flat_cands, flat_true, N_arr, L_arr, d0_arr, d_scores
        )
        return d_scores.copy_to_host()
    else:
        from gpu_kernels import _batch_tm_score_cpu
        return _batch_tm_score_cpu(preds_stacked, true, N, L, d0)


def tm_score(pred: np.ndarray, true: np.ndarray) -> float:
    """Compute TM-score — auto-selects GPU if CUDA is available, else CPU.

    Args:
        pred: Predicted coordinates, shape (L, 3) or (L, N, 3).
        true: True coordinates, shape (L, 3) or (L, N, 3).

    Returns:
        TM-score as a float in [0, 1].
    """
    if _CUDA_AVAILABLE:
        return tm_score_gpu(pred, true)
    return tm_score_cpu(pred, true)


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
