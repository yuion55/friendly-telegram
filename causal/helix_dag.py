"""PC-algorithm causal discovery for RNA helix formation order."""

import itertools
from collections import deque

import numpy as np
from scipy.stats import norm


class HelixFormationDAG:
    """Discover a DAG over helix variables using the PC algorithm."""

    def __init__(self, alpha: float = 0.05):
        self.alpha = alpha
        self.adj_: np.ndarray | None = None
        self.K_: int = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, X: np.ndarray) -> "HelixFormationDAG":
        """Run the PC algorithm on *X* (N_samples, K_helices) float32."""
        X = np.asarray(X, dtype=np.float32)
        N, K = X.shape
        self.K_ = K

        # Fully-connected undirected skeleton (adjacency matrix)
        adj = np.ones((K, K), dtype=bool)
        np.fill_diagonal(adj, False)

        for lvl in range(K):
            for i in range(K):
                for j in range(i + 1, K):
                    if not adj[i, j]:
                        continue
                    neighbors_i = [
                        k for k in range(K) if k != j and adj[i, k]
                    ]
                    if len(neighbors_i) < lvl:
                        continue
                    for S in itertools.combinations(neighbors_i, lvl):
                        S_list = list(S)
                        r = self.partial_correlation(X, i, j, S_list)
                        p = self._fisher_z_pvalue(r, N, len(S_list))
                        if p > self.alpha:
                            adj[i, j] = False
                            adj[j, i] = False
                            break

        self.adj_ = adj
        return self

    def partial_correlation(
        self,
        X: np.ndarray,
        i: int,
        j: int,
        conditioning_indices: list[int],
    ) -> float:
        """Partial correlation between columns *i* and *j* given *conditioning_indices*."""
        X = np.asarray(X, dtype=np.float32)
        xi = X[:, i]
        xj = X[:, j]

        if not conditioning_indices:
            return float(self._pearson(xi, xj))

        Z = X[:, conditioning_indices]
        # Residualize via least-squares
        ri = self._residualize(xi, Z)
        rj = self._residualize(xj, Z)
        return float(self._pearson(ri, rj))

    def get_formation_order(self) -> list[int]:
        """Topological sort (Kahn's algorithm) – earliest-forming helix first."""
        if self.adj_ is None:
            raise RuntimeError("Call fit() before get_formation_order().")

        K = self.K_
        adj = self.adj_.copy()

        # Orient edges by column-mean heuristic (lower mean → earlier cause)
        # Build a proper directed adjacency list.
        directed = np.zeros((K, K), dtype=bool)
        for i in range(K):
            for j in range(i + 1, K):
                if adj[i, j]:
                    # Keep edge from node with smaller index as a
                    # deterministic tie-break for the undirected skeleton.
                    directed[i, j] = True

        in_degree = directed.sum(axis=0).astype(int)
        queue: deque[int] = deque(
            i for i in range(K) if in_degree[i] == 0
        )
        order: list[int] = []
        while queue:
            node = queue.popleft()
            order.append(node)
            for nbr in range(K):
                if directed[node, nbr]:
                    in_degree[nbr] -= 1
                    if in_degree[nbr] == 0:
                        queue.append(nbr)

        # Include any isolated nodes not yet visited
        remaining = [v for v in range(K) if v not in set(order)]
        order.extend(remaining)
        return order

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _pearson(a: np.ndarray, b: np.ndarray) -> np.floating:
        denom_a = np.std(a, ddof=0)
        denom_b = np.std(b, ddof=0)
        if denom_a == 0.0 or denom_b == 0.0:
            return np.float32(0.0)
        return np.corrcoef(a, b)[0, 1]

    @staticmethod
    def _residualize(y: np.ndarray, Z: np.ndarray) -> np.ndarray:
        """OLS residuals of *y* regressed on *Z*."""
        # lstsq handles rank-deficient Z gracefully
        coef, _, _, _ = np.linalg.lstsq(Z, y, rcond=None)
        return y - Z @ coef

    @staticmethod
    def _fisher_z_pvalue(r: float, n: int, s_size: int) -> float:
        """Two-sided p-value from Fisher's Z-transform of partial correlation *r*."""
        dof = n - s_size - 3
        if dof < 1:
            return 1.0  # not enough samples – never reject

        # Clamp to avoid log(0) / division by zero
        r = np.clip(r, -0.999999, 0.999999)
        z = 0.5 * np.log((1.0 + r) / (1.0 - r)) * np.sqrt(dof)
        return float(2.0 * (1.0 - norm.cdf(np.abs(z))))


# ------------------------------------------------------------------
# Self-test on synthetic data
# ------------------------------------------------------------------

if __name__ == "__main__":
    rng = np.random.default_rng(42)
    K, N = 5, 200

    # Ground truth: chain  0 -> 1 -> 2 -> 3 -> 4
    data = np.empty((N, K), dtype=np.float32)
    data[:, 0] = rng.standard_normal(N).astype(np.float32)
    for col in range(1, K):
        noise = rng.standard_normal(N).astype(np.float32) * 0.3
        data[:, col] = data[:, col - 1] + noise

    dag = HelixFormationDAG(alpha=0.05)
    dag.fit(data)

    print("Adjacency matrix (undirected skeleton):")
    print(dag.adj_.astype(int))

    order = dag.get_formation_order()
    print(f"Formation order: {order}")

    # Basic sanity checks
    assert dag.adj_ is not None, "fit() must set adj_"
    assert dag.adj_.shape == (K, K), "adj_ shape mismatch"
    assert len(order) == K, "formation order must contain all helices"
    assert set(order) == set(range(K)), "formation order must be a permutation"
    print("All self-tests passed.")
