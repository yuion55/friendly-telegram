"""Abstract interpretation of geometric predicates for RNA helices."""

from dataclasses import dataclass

import numpy as np
from numba import njit


# ------------------------------------------------------------------
# Geometric interval dataclass
# ------------------------------------------------------------------

@dataclass
class GeometricInterval:
    """Closed interval [lo, hi] over geometric quantities."""

    lo: np.float32
    hi: np.float32

    def join(self, other: "GeometricInterval") -> "GeometricInterval":
        """Least upper bound (union hull)."""
        return GeometricInterval(
            lo=np.float32(min(self.lo, other.lo)),
            hi=np.float32(max(self.hi, other.hi)),
        )

    def meet(self, other: "GeometricInterval") -> "GeometricInterval":
        """Greatest lower bound (intersection)."""
        return GeometricInterval(
            lo=np.float32(max(self.lo, other.lo)),
            hi=np.float32(min(self.hi, other.hi)),
        )

    def satisfies(self, value: float) -> bool:
        """Return *True* when *value* falls within [lo, hi]."""
        return bool(self.lo <= value <= self.hi)


# ------------------------------------------------------------------
# Numba-accelerated geometric helpers
# ------------------------------------------------------------------

@njit(cache=True)
def compute_helix_axis(coords, start, end):
    """First principal component of the segment coords[start:end] via power iteration.

    Parameters
    ----------
    coords : (L, 3) float32 array
    start, end : int  – row range (exclusive end)

    Returns
    -------
    axis : (3,) float32 unit vector
    """
    seg = coords[start:end].astype(np.float32)
    n = seg.shape[0]
    # Center the segment
    cx = np.float32(0.0)
    cy = np.float32(0.0)
    cz = np.float32(0.0)
    for i in range(n):
        cx += seg[i, 0]
        cy += seg[i, 1]
        cz += seg[i, 2]
    cx /= np.float32(n)
    cy /= np.float32(n)
    cz /= np.float32(n)
    for i in range(n):
        seg[i, 0] -= cx
        seg[i, 1] -= cy
        seg[i, 2] -= cz

    # Covariance matrix (3x3)
    cov = np.zeros((3, 3), dtype=np.float32)
    for i in range(n):
        for p in range(3):
            for q in range(3):
                cov[p, q] += seg[i, p] * seg[i, q]

    # Power iteration – 20 steps
    v = np.ones(3, dtype=np.float32)
    norm_v = np.float32(0.0)
    for _ in range(20):
        w = np.zeros(3, dtype=np.float32)
        for p in range(3):
            for q in range(3):
                w[p] += cov[p, q] * v[q]
        norm_v = np.float32(0.0)
        for p in range(3):
            norm_v += w[p] * w[p]
        norm_v = np.sqrt(norm_v)
        if norm_v < np.float32(1e-12):
            break
        for p in range(3):
            v[p] = w[p] / norm_v
    return v


@njit(cache=True)
def inter_helix_angle(ax1, ax2):
    """Angle in degrees between two helix axes.

    Uses the absolute dot product so that anti-parallel axes give 0°.

    Parameters
    ----------
    ax1, ax2 : (3,) float32 arrays

    Returns
    -------
    angle : float32 – degrees in [0, 90]
    """
    dot = np.float32(0.0)
    for p in range(3):
        dot += ax1[p] * ax2[p]
    dot = np.abs(dot)
    if dot > np.float32(1.0):
        dot = np.float32(1.0)
    return np.float32(np.degrees(np.arccos(dot)))


# ------------------------------------------------------------------
# Abstract interpreter
# ------------------------------------------------------------------

class GeometricPredicateAbstractInterpreter:
    """Registry of geometric predicates verified against concrete coordinates."""

    def __init__(self):
        self._parallelism: list[dict] = []
        self._aminor: list[dict] = []

    # -- registration --------------------------------------------------

    def add_parallelism_predicate(
        self,
        h1_range: tuple[int, int],
        h2_range: tuple[int, int],
        angle_lo: float,
        angle_hi: float,
    ) -> None:
        """Register an inter-helix parallelism predicate."""
        self._parallelism.append(
            {
                "h1_range": h1_range,
                "h2_range": h2_range,
                "interval": GeometricInterval(
                    lo=np.float32(angle_lo), hi=np.float32(angle_hi)
                ),
            }
        )

    def add_aminor_predicate(
        self,
        res_i: int,
        stem_start: int,
        stem_end: int,
        dist_lo: float,
        dist_hi: float,
    ) -> None:
        """Register an A-minor interaction distance predicate."""
        self._aminor.append(
            {
                "res_i": res_i,
                "stem_start": stem_start,
                "stem_end": stem_end,
                "interval": GeometricInterval(
                    lo=np.float32(dist_lo), hi=np.float32(dist_hi)
                ),
            }
        )

    # -- verification --------------------------------------------------

    def verify(self, coords: np.ndarray) -> dict:
        """Check every registered predicate against *coords* (L, 3).

        Returns
        -------
        dict mapping predicate name → {satisfied: bool, violation: float}
        """
        coords = np.asarray(coords, dtype=np.float32)
        results: dict[str, dict] = {}

        for idx, pred in enumerate(self._parallelism):
            s1, e1 = pred["h1_range"]
            s2, e2 = pred["h2_range"]
            ax1 = compute_helix_axis(coords, s1, e1)
            ax2 = compute_helix_axis(coords, s2, e2)
            angle = float(inter_helix_angle(ax1, ax2))
            interval: GeometricInterval = pred["interval"]
            satisfied = interval.satisfies(angle)
            violation = np.float32(0.0)
            if not satisfied:
                violation = np.float32(
                    min(abs(angle - interval.lo), abs(angle - interval.hi))
                )
            results[f"parallelism_{idx}"] = {
                "satisfied": satisfied,
                "violation": float(violation),
            }

        for idx, pred in enumerate(self._aminor):
            res_i = pred["res_i"]
            s, e = pred["stem_start"], pred["stem_end"]
            # Distance from residue to nearest atom in stem segment
            dist = _min_dist_to_segment(coords, res_i, s, e)
            interval = pred["interval"]
            satisfied = interval.satisfies(dist)
            violation = np.float32(0.0)
            if not satisfied:
                violation = np.float32(
                    min(abs(dist - interval.lo), abs(dist - interval.hi))
                )
            results[f"aminor_{idx}"] = {
                "satisfied": satisfied,
                "violation": float(violation),
            }

        return results

    def total_violation_penalty(self, coords: np.ndarray) -> float:
        """Sum of violation magnitudes across all predicates."""
        results = self.verify(coords)
        return float(sum(r["violation"] for r in results.values()))


# ------------------------------------------------------------------
# Internal helpers
# ------------------------------------------------------------------

def _min_dist_to_segment(
    coords: np.ndarray, res_i: int, seg_start: int, seg_end: int
) -> float:
    """Minimum Euclidean distance from coords[res_i] to any row in segment."""
    pt = coords[res_i].astype(np.float32)
    seg = coords[seg_start:seg_end].astype(np.float32)
    diffs = seg - pt[None, :]
    dists = np.sqrt(np.sum(diffs ** 2, axis=1))
    return float(np.min(dists))


# ------------------------------------------------------------------
# Self-test
# ------------------------------------------------------------------

if __name__ == "__main__":
    rng = np.random.RandomState(42)
    L = 50
    coords = rng.randn(L, 3).astype(np.float32)

    # -- GeometricInterval tests --
    iv1 = GeometricInterval(lo=np.float32(0.0), hi=np.float32(10.0))
    iv2 = GeometricInterval(lo=np.float32(5.0), hi=np.float32(15.0))
    joined = iv1.join(iv2)
    assert joined.lo == 0.0 and joined.hi == 15.0, "join failed"
    met = iv1.meet(iv2)
    assert met.lo == 5.0 and met.hi == 10.0, "meet failed"
    assert iv1.satisfies(5.0) and not iv1.satisfies(11.0), "satisfies failed"

    # -- compute_helix_axis test --
    axis = compute_helix_axis(coords, 0, 10)
    assert axis.shape == (3,), f"axis shape {axis.shape}"
    norm_val = float(np.sqrt(np.sum(axis ** 2)))
    assert abs(norm_val - 1.0) < 1e-4, f"axis not unit: {norm_val}"

    # -- inter_helix_angle test --
    ax1 = compute_helix_axis(coords, 0, 10)
    ax2 = compute_helix_axis(coords, 20, 30)
    angle = inter_helix_angle(ax1, ax2)
    assert 0.0 <= angle <= 90.0, f"angle out of range: {angle}"

    # -- Interpreter tests --
    interp = GeometricPredicateAbstractInterpreter()
    interp.add_parallelism_predicate((0, 10), (20, 30), 0.0, 45.0)
    interp.add_aminor_predicate(15, 0, 10, 0.0, 20.0)
    report = interp.verify(coords)
    assert "parallelism_0" in report, "missing parallelism key"
    assert "aminor_0" in report, "missing aminor key"
    assert isinstance(report["parallelism_0"]["satisfied"], bool)
    penalty = interp.total_violation_penalty(coords)
    assert isinstance(penalty, float) and penalty >= 0.0

    print("All self-tests passed.")
