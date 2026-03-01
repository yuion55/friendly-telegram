"""Persistent homology features for RNA 3D structure analysis."""

import numpy as np
from numba import njit, prange

try:
    from ripser import ripser
    HAS_RIPSER = True
except ImportError:
    HAS_RIPSER = False


@njit(cache=True, parallel=True)
def compute_distance_matrix(coords):
    """Compute pairwise Euclidean distance matrix from 3D coordinates.

    Args:
        coords: float32 array of shape (N, 3) with atom positions.

    Returns:
        Symmetric float32 array of shape (N, N) with pairwise distances.
    """
    N = coords.shape[0]
    D = np.zeros((N, N), dtype=np.float32)
    for i in prange(N):
        for j in range(i + 1, N):
            dx = coords[i, 0] - coords[j, 0]
            dy = coords[i, 1] - coords[j, 1]
            dz = coords[i, 2] - coords[j, 2]
            d = np.sqrt(dx * dx + dy * dy + dz * dz)
            D[i, j] = d
            D[j, i] = d
    return D


def compute_persistence_diagram(coords, max_dim=2):
    """Compute persistence diagram from 3D coordinates using Vietoris-Rips.

    Args:
        coords: float32 array of shape (N, 3) with atom positions.
        max_dim: maximum homology dimension to compute (default 2).

    Returns:
        Dict with 'dgms' key mapping to list of persistence diagrams,
        one per dimension up to max_dim.
    """
    distance_matrix = compute_distance_matrix(coords)
    if HAS_RIPSER:
        result = ripser(distance_matrix, distance_matrix=True, maxdim=max_dim)
        return result
    # Fallback: return empty diagrams when ripser is not installed
    dgms = [np.empty((0, 2), dtype=np.float32) for _ in range(max_dim + 1)]
    return {"dgms": dgms}


@njit(cache=True)
def wasserstein_distance_approx(dgm1, dgm2):
    """Approximate Wasserstein distance via nearest-neighbor matching.

    For each point in dgm1, finds the closest point in dgm2 by
    Euclidean distance, then averages these minimum distances.

    Args:
        dgm1: float32 array of shape (K, 2) with (birth, death) pairs.
        dgm2: float32 array of shape (K, 2) with (birth, death) pairs.

    Returns:
        Approximate Wasserstein distance as float32.
    """
    K1 = dgm1.shape[0]
    K2 = dgm2.shape[0]
    if K1 == 0 or K2 == 0:
        return np.float32(0.0)
    total = np.float32(0.0)
    for i in range(K1):
        min_dist = np.float32(1e30)
        for j in range(K2):
            db = dgm1[i, 0] - dgm2[j, 0]
            dd = dgm1[i, 1] - dgm2[j, 1]
            d = np.sqrt(db * db + dd * dd)
            if d < min_dist:
                min_dist = d
        total += min_dist
    return total / np.float32(K1)


class TDAFeatureExtractor:
    """Extract topological data analysis features from 3D coordinates.

    Wraps ripser to produce Betti numbers and persistence diagrams.

    Args:
        max_dim: maximum homology dimension (default 2).
        threshold: persistence threshold below which features are ignored.
    """

    def __init__(self, max_dim=2, threshold=0.0):
        self.max_dim = max_dim
        self.threshold = np.float32(threshold)

    def extract(self, coords):
        """Extract TDA features from 3D coordinates.

        Args:
            coords: float32 array of shape (N, 3) with atom positions.

        Returns:
            Dict with keys:
                'betti_numbers': tuple (beta0, beta1, beta2).
                'persistence_diagrams': list of float32 arrays per dimension.
        """
        coords = np.asarray(coords, dtype=np.float32)
        result = compute_persistence_diagram(coords, max_dim=self.max_dim)
        dgms = result["dgms"]

        betti = []
        filtered_dgms = []
        for dim in range(self.max_dim + 1):
            dgm = dgms[dim]
            if dgm.shape[0] > 0:
                persist = dgm[:, 1] - dgm[:, 0]
                # Filter finite features above threshold
                mask = np.isfinite(persist) & (persist > self.threshold)
                filtered = dgm[mask].astype(np.float32)
            else:
                filtered = np.empty((0, 2), dtype=np.float32)
            filtered_dgms.append(filtered)
            betti.append(filtered.shape[0])

        return {
            "betti_numbers": tuple(betti),
            "persistence_diagrams": filtered_dgms,
        }


if __name__ == "__main__":
    L = 50
    np.random.seed(42)
    coords = np.random.randn(L, 3).astype(np.float32)

    # Test distance matrix
    D = compute_distance_matrix(coords)
    assert D.shape == (L, L), "Distance matrix shape mismatch"
    assert D.dtype == np.float32, "Distance matrix dtype must be float32"
    assert np.allclose(D, D.T), "Distance matrix must be symmetric"
    assert np.allclose(np.diag(D), 0.0), "Diagonal must be zero"
    print(f"Distance matrix: shape={D.shape}, dtype={D.dtype}")

    # Test persistence diagram
    result = compute_persistence_diagram(coords, max_dim=2)
    assert "dgms" in result, "Result must contain 'dgms' key"
    print(f"Persistence diagrams: {len(result['dgms'])} dimensions")
    for dim, dgm in enumerate(result["dgms"]):
        print(f"  H{dim}: {dgm.shape[0]} features")

    # Test Wasserstein distance approximation
    dgm_a = np.random.rand(10, 2).astype(np.float32)
    dgm_b = np.random.rand(10, 2).astype(np.float32)
    w = wasserstein_distance_approx(dgm_a, dgm_b)
    assert isinstance(float(w), float), "Wasserstein distance must be float"
    w_self = wasserstein_distance_approx(dgm_a, dgm_a)
    assert w_self < 1e-6, "Self-distance must be ~0"
    print(f"Wasserstein approx: d(a,b)={w:.4f}, d(a,a)={w_self:.6f}")

    # Test TDA feature extractor
    extractor = TDAFeatureExtractor(max_dim=2, threshold=0.0)
    features = extractor.extract(coords)
    b0, b1, b2 = features["betti_numbers"]
    print(f"Betti numbers: beta0={b0}, beta1={b1}, beta2={b2}")
    assert len(features["persistence_diagrams"]) == 3
    for dgm in features["persistence_diagrams"]:
        assert dgm.dtype == np.float32, "Diagrams must be float32"

    print("Self-test passed.")
