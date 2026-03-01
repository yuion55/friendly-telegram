"""Numba CUDA kernels for TM-score, distance matrix, and batch scoring.

Falls back gracefully when CUDA is not available.
"""

import math
import numpy as np
from numba import njit, prange

# Graceful CUDA import
try:
    from numba import cuda
    _CUDA_AVAILABLE = cuda.is_available()
except Exception:
    _CUDA_AVAILABLE = False

# ---------------------------------------------------------------------------
# CPU fallback implementations (njit + prange)
# ---------------------------------------------------------------------------


@njit(cache=True, parallel=True)
def _tm_score_cpu(pred_flat, true_flat, L, d0):
    """CPU TM-score between two structures (L,3) stored flat."""
    d0_sq = np.float32(d0 * d0)
    total = np.float32(0.0)
    for i in prange(L):
        dx = pred_flat[i, 0] - true_flat[i, 0]
        dy = pred_flat[i, 1] - true_flat[i, 1]
        dz = pred_flat[i, 2] - true_flat[i, 2]
        di_sq = dx * dx + dy * dy + dz * dz
        total += np.float32(1.0) / (np.float32(1.0) + di_sq / d0_sq)
    return total / np.float32(L)


@njit(cache=True, parallel=True)
def _pairwise_distance_matrix_cpu(coords, N):
    """CPU pairwise Euclidean distance matrix for (N,3) coords."""
    out = np.zeros((N, N), dtype=np.float32)
    for i in prange(N):
        for j in range(i + 1, N):
            dx = coords[i, 0] - coords[j, 0]
            dy = coords[i, 1] - coords[j, 1]
            dz = coords[i, 2] - coords[j, 2]
            d = np.float32(math.sqrt(dx * dx + dy * dy + dz * dz))
            out[i, j] = d
            out[j, i] = d
    return out


@njit(cache=True, parallel=True)
def _batch_tm_score_cpu(candidates_flat, true_flat, N, L, d0):
    """CPU batch TM-score: N candidates vs one true structure."""
    d0_sq = np.float32(d0 * d0)
    scores = np.zeros(N, dtype=np.float32)
    for n in prange(N):
        total = np.float32(0.0)
        off = n * L
        for k in range(L):
            dx = candidates_flat[off + k, 0] - true_flat[k, 0]
            dy = candidates_flat[off + k, 1] - true_flat[k, 1]
            dz = candidates_flat[off + k, 2] - true_flat[k, 2]
            di_sq = dx * dx + dy * dy + dz * dz
            total += np.float32(1.0) / (np.float32(1.0) + di_sq / d0_sq)
        scores[n] = total / np.float32(L)
    return scores


# ---------------------------------------------------------------------------
# CUDA kernel implementations
# ---------------------------------------------------------------------------

if _CUDA_AVAILABLE:

    @cuda.jit
    def tm_score_gpu_kernel(pred_flat, true_flat, L, d0, out):
        """CUDA kernel: TM-score between one predicted and one true structure.

        Grid: (1,). Block: (min(L, 1024),).
        Uses shared memory for tree reduction.
        """
        tid = cuda.threadIdx.x
        block_size = cuda.blockDim.x

        shared = cuda.shared.array(1024, dtype=np.float32)

        d0_sq = d0[0] * d0[0]
        val = np.float32(0.0)

        # Each thread accumulates over its assigned residues
        idx = tid
        while idx < L[0]:
            dx = pred_flat[idx * 3] - true_flat[idx * 3]
            dy = pred_flat[idx * 3 + 1] - true_flat[idx * 3 + 1]
            dz = pred_flat[idx * 3 + 2] - true_flat[idx * 3 + 2]
            di_sq = dx * dx + dy * dy + dz * dz
            val += np.float32(1.0) / (np.float32(1.0) + di_sq / d0_sq)
            idx += block_size

        shared[tid] = val
        cuda.syncthreads()

        # Tree reduction
        s = block_size // 2
        while s > 0:
            if tid < s:
                shared[tid] += shared[tid + s]
            cuda.syncthreads()
            s //= 2

        if tid == 0:
            out[0] = shared[0] / np.float32(L[0])

    @cuda.jit
    def pairwise_distance_matrix_kernel(coords, N, D, out):
        """CUDA kernel: pairwise distance matrix with shared memory tiling.

        Grid: ((N+31)//32, (N+31)//32). Block: (32, 32).
        """
        tx = cuda.threadIdx.x
        ty = cuda.threadIdx.y
        bx = cuda.blockIdx.x
        by = cuda.blockIdx.y

        i = bx * 32 + tx
        j = by * 32 + ty

        # Shared memory tiles for coordinates: 32x3 each
        tile_i = cuda.shared.array((32, 3), dtype=np.float32)
        tile_j = cuda.shared.array((32, 3), dtype=np.float32)

        n_val = N[0]

        # Load tiles
        if i < n_val and ty < 3:
            tile_i[tx, ty] = coords[i * 3 + ty]
        if j < n_val and tx < 3:
            tile_j[ty, tx] = coords[j * 3 + tx]
        cuda.syncthreads()

        if i < n_val and j < n_val:
            dx = tile_i[tx, 0] - tile_j[ty, 0]
            dy = tile_i[tx, 1] - tile_j[ty, 1]
            dz = tile_i[tx, 2] - tile_j[ty, 2]
            dist = math.sqrt(dx * dx + dy * dy + dz * dz)
            out[i * n_val + j] = np.float32(dist)
            out[j * n_val + i] = np.float32(dist)

    @cuda.jit
    def batch_tm_score_kernel(candidates_flat, true_flat, N, L, d0, scores_out):
        """CUDA kernel: TM-score for N candidates against one true structure.

        Grid: (N,). Block: (min(L, 512),).
        Each block handles one candidate.
        """
        bid = cuda.blockIdx.x
        tid = cuda.threadIdx.x
        block_size = cuda.blockDim.x
        l_val = L[0]
        d0_sq = d0[0] * d0[0]

        shared = cuda.shared.array(512, dtype=np.float32)

        val = np.float32(0.0)
        idx = tid
        off = bid * l_val
        while idx < l_val:
            dx = candidates_flat[(off + idx) * 3] - true_flat[idx * 3]
            dy = candidates_flat[(off + idx) * 3 + 1] - true_flat[idx * 3 + 1]
            dz = candidates_flat[(off + idx) * 3 + 2] - true_flat[idx * 3 + 2]
            di_sq = dx * dx + dy * dy + dz * dz
            val += np.float32(1.0) / (np.float32(1.0) + di_sq / d0_sq)
            idx += block_size

        shared[tid] = val
        cuda.syncthreads()

        s = block_size // 2
        while s > 0:
            if tid < s:
                shared[tid] += shared[tid + s]
            cuda.syncthreads()
            s //= 2

        if tid == 0:
            scores_out[bid] = shared[0] / np.float32(l_val)


# ---------------------------------------------------------------------------
# Python wrappers
# ---------------------------------------------------------------------------


def pairwise_tm_matrix_gpu(candidates_list, N, L):
    """Compute pairwise TM-score matrix for N candidate structures.

    Parameters
    ----------
    candidates_list : list of np.ndarray
        Each element is (L, 3) float32.
    N : int
        Number of candidates.
    L : int
        Sequence length.

    Returns
    -------
    np.ndarray
        Shape (N, N) float32 pairwise TM-score matrix.
    """
    d0 = np.float32(max(0.5, 1.24 * (L - 15) ** (1.0 / 3.0) - 1.8))
    candidates_flat = np.concatenate(candidates_list, axis=0).astype(np.float32)

    if _CUDA_AVAILABLE:
        # Flatten to (N*L*3,) for CUDA kernels
        flat_1d = np.ascontiguousarray(candidates_flat.reshape(-1))
        d_candidates = cuda.to_device(flat_1d)
        d0_arr = np.array([d0], dtype=np.float32)
        d_d0 = cuda.to_device(d0_arr)
        L_arr = np.array([L], dtype=np.int32)
        d_L = cuda.to_device(L_arr)
        N_arr = np.array([N], dtype=np.int32)

        tm = np.zeros((N, N), dtype=np.float32)
        block_size = min(L, 512)

        for i in range(N):
            true_1d = np.ascontiguousarray(
                candidates_flat[i * L:(i + 1) * L].reshape(-1)
            )
            d_true = cuda.to_device(true_1d)
            d_scores = cuda.device_array(N, dtype=np.float32)

            batch_tm_score_kernel[(N,), (block_size,)](
                d_candidates, d_true, N_arr, d_L, d_d0, d_scores
            )
            tm[i, :] = d_scores.copy_to_host()

        return tm
    else:
        # CPU fallback
        return _pairwise_tm_matrix_cpu(candidates_flat, N, L, d0)


@njit(cache=True, parallel=True)
def _pairwise_tm_matrix_cpu(coords_flat, N, L, d0):
    """CPU pairwise TM-score matrix."""
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


def distance_matrix_gpu(coords, N):
    """Compute pairwise distance matrix for (N, 3) coordinates.

    Parameters
    ----------
    coords : np.ndarray
        Shape (N, 3) float32 coordinates.
    N : int
        Number of points.

    Returns
    -------
    np.ndarray
        Shape (N, N) float32 distance matrix.
    """
    coords = np.ascontiguousarray(coords, dtype=np.float32)

    if _CUDA_AVAILABLE:
        flat = coords.reshape(-1)
        d_coords = cuda.to_device(flat)
        N_arr = np.array([N], dtype=np.int32)
        D_arr = np.array([3], dtype=np.int32)
        d_out = cuda.device_array(N * N, dtype=np.float32)

        grid = ((N + 31) // 32, (N + 31) // 32)
        block = (32, 32)
        pairwise_distance_matrix_kernel[grid, block](
            d_coords, N_arr, D_arr, d_out
        )
        return d_out.copy_to_host().reshape(N, N)
    else:
        return _pairwise_distance_matrix_cpu(coords, N)


if __name__ == "__main__":
    rng = np.random.default_rng(42)
    L, N = 50, 10

    # Test pairwise TM matrix
    candidates = [rng.standard_normal((L, 3)).astype(np.float32) for _ in range(N)]
    tm = pairwise_tm_matrix_gpu(candidates, N, L)
    assert tm.shape == (N, N), f"Expected ({N},{N}), got {tm.shape}"
    assert tm.dtype == np.float32
    for i in range(N):
        assert abs(tm[i, i] - 1.0) < 1e-4, f"Diagonal tm[{i},{i}]={tm[i, i]}"

    # Test distance matrix
    coords = rng.standard_normal((N, 3)).astype(np.float32)
    dm = distance_matrix_gpu(coords, N)
    assert dm.shape == (N, N)
    assert dm.dtype == np.float32
    for i in range(N):
        assert abs(dm[i, i]) < 1e-5, f"Diagonal dm[{i},{i}]={dm[i, i]}"

    # Test batch TM-score CPU
    true_coords = rng.standard_normal((L, 3)).astype(np.float32)
    cands_flat = np.concatenate(candidates, axis=0).astype(np.float32)
    d0 = np.float32(max(0.5, 1.24 * (L - 15) ** (1.0 / 3.0) - 1.8))
    scores = _batch_tm_score_cpu(cands_flat, true_coords, N, L, d0)
    assert scores.shape == (N,)
    assert scores.dtype == np.float32
    assert all(0.0 <= s <= 1.0 for s in scores)

    print(f"CUDA available: {_CUDA_AVAILABLE}")
    print("gpu_kernels self-test PASSED")
