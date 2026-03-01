"""
rna_briq_refinement.py
======================
Knowledge-Based Post-Hoc Refinement for RNA Structures
Implements BRiQ / QRNAS-style energy minimization with:
  - Numba JIT compilation for hot inner loops
  - Vectorized pairwise energy evaluations
  - Orientation-dependent base-base, base-oxygen, and oxygen-oxygen potentials
  - Rotameric backbone state scoring
  - Harmonic P-atom restraints to anchor global fold
  - Gradient-free minimization (L-BFGS-B) + optional Metropolis MC

Usage
-----
>>> from rna_briq_refinement import BRiQRefinement
>>> refiner = BRiQRefinement(n_steps=300, restraint_weight=10.0, seed=42)
>>> refined_coords, info = refiner.refine(coords, sequence)

Coordinate convention
---------------------
coords : np.ndarray, shape (N_atoms, 3), float64, Angstrom
  Expected atom order per nucleotide (canonical heavy atoms):
    P, OP1, OP2, O5', C5', C4', C3', O3', C2', O2', C1', N1/N9, base_heavy...
  A simpler "coarse" mode uses only [P, C4', N-base] per residue.

Reference
---------
Yu, J. et al. BRiQ (2021) - Knowledge-based potential for RNA.
QRNAS - Nucleic Acids Res. 2012;40(19):e156.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize
from scipy.spatial.distance import cdist

# ── Numba (optional but strongly recommended) ──────────────────────────────────
try:
    import numba as nb
    from numba import njit, prange
    _NUMBA = True
except ImportError:
    warnings.warn(
        "Numba not found – falling back to pure NumPy (slow). "
        "Install numba for ~50× speed-up: pip install numba",
        RuntimeWarning,
        stacklevel=2,
    )
    _NUMBA = False
    # Shim so decorated functions still work
    def njit(*a, **kw):          # type: ignore[misc]
        def _dec(f): return f
        return _dec
    prange = range                # type: ignore[misc]

# ═══════════════════════════════════════════════════════════════════════════════
# 1.  CONSTANTS & KNOWLEDGE-BASED POTENTIAL TABLES
# ═══════════════════════════════════════════════════════════════════════════════

_AA_CODES = {"A": 0, "C": 1, "G": 2, "U": 3}

# ── Ideal backbone torsion targets (deg) from PDB statistics ──────────────────
# alpha, beta, gamma, delta, epsilon, zeta, chi  (A-form helix mode)
IDEAL_TORSIONS_AFORM = np.array(
    [-68.0, 178.0, 54.0, 79.0, -153.0, -75.0, -160.0], dtype=np.float64
)

# ── Pairwise distance potential bins (Å) ──────────────────────────────────────
# BRiQ uses 26 bins from 2 to 18 Å; we use a condensed 16-bin version
_R_MIN, _R_MAX = 2.0, 18.0
_N_BINS = 16
_BIN_EDGES = np.linspace(_R_MIN, _R_MAX, _N_BINS + 1, dtype=np.float64)
_BIN_CENTERS = 0.5 * (_BIN_EDGES[:-1] + _BIN_EDGES[1:])
_BIN_WIDTH = _BIN_EDGES[1] - _BIN_EDGES[0]

# Statistical potentials (PMF): shape (4, 4, N_BINS)
# In production these are derived from PDB statistics; here we use analytic
# proxies that capture the essential physics of base stacking / H-bonding.
def _build_statistical_potentials() -> NDArray:
    """Build toy statistical potentials with correct physical shape."""
    n_bases = 4
    pmf = np.zeros((n_bases, n_bases, _N_BINS), dtype=np.float64)
    rc = _BIN_CENTERS

    for i in range(n_bases):
        for j in range(n_bases):
            # Stacking WELL (attractive) near 3.4 Å; pairing shoulder near 9–11 Å;
            # short-range hard-core repulsion below 2 Å handled by bin clamping.
            # Sign convention: negative = attractive (lower energy = preferred).
            pmf[i, j] = (
                - 8.0 * np.exp(-0.5 * ((rc - 3.4) / 0.6) ** 2)   # stacking well
                - 2.0 * np.exp(-0.5 * ((rc - 10.0) / 1.2) ** 2)  # generic pairing shoulder
                + 0.5 * (rc / _R_MAX) ** 2                         # flat repulsion at long range
            )
            # Watson-Crick pairs (A-U: 0-3, G-C: 2-1) get an additional deeper pairing well
            if (i, j) in {(0, 3), (3, 0), (2, 1), (1, 2)}:
                pmf[i, j] -= 1.5 * np.exp(-0.5 * ((rc - 9.5) / 0.8) ** 2)

    # Smooth
    from scipy.ndimage import gaussian_filter1d
    pmf = gaussian_filter1d(pmf, sigma=0.7, axis=-1)
    return pmf.astype(np.float64)

_PMF_BB = _build_statistical_potentials()   # base-base
# Simplified base-oxygen and oxygen-oxygen use scaled versions
_PMF_BO = (_PMF_BB * 0.6).astype(np.float64)
_PMF_OO = (_PMF_BB * 0.3).astype(np.float64)

# ── Backbone force constants (kcal/mol/rad²) ──────────────────────────────────
_K_TORSION = np.array([1.2, 0.8, 1.5, 2.0, 1.5, 1.2, 3.0], dtype=np.float64)

# ── Harmonic restraint on P atoms ─────────────────────────────────────────────
# E_restr = 0.5 * k * |r - r0|²
_K_RESTR_DEFAULT = 10.0   # kcal/mol/Å²

# ═══════════════════════════════════════════════════════════════════════════════
# 2.  JIT-COMPILED CORE KERNELS
# ═══════════════════════════════════════════════════════════════════════════════

@njit(cache=True, fastmath=True)
def _bin_index(r: float) -> int:
    """Return the PMF bin index for distance r (clamped)."""
    idx = int((r - _R_MIN) / _BIN_WIDTH)
    if idx < 0:
        return 0
    if idx >= _N_BINS:
        return _N_BINS - 1
    return idx


@njit(cache=True, fastmath=True, parallel=True)
def _pairwise_energy_jit(
    base_coords: NDArray,   # (N, 3)
    base_types: NDArray,    # (N,) int  0-3
    oxy_coords: NDArray,    # (M, 3)
    pmf_bb: NDArray,        # (4,4,N_BINS)
    pmf_bo: NDArray,        # (4,4,N_BINS)
    pmf_oo: NDArray,        # (4,4,N_BINS)
    cutoff: float,
) -> float:
    """Sum all pairwise knowledge-based terms."""
    N = base_coords.shape[0]
    M = oxy_coords.shape[0]
    E_bb = 0.0
    E_bo = 0.0
    E_oo = 0.0

    # Base-base (parallelised over i)
    for i in prange(N - 1):
        xi = base_coords[i, 0]; yi = base_coords[i, 1]; zi = base_coords[i, 2]
        ti = base_types[i]
        for j in range(i + 1, N):
            dx = xi - base_coords[j, 0]
            dy = yi - base_coords[j, 1]
            dz = zi - base_coords[j, 2]
            r = (dx * dx + dy * dy + dz * dz) ** 0.5
            if r < cutoff:
                b = _bin_index(r)
                E_bb += pmf_bb[ti, base_types[j], b]

    # Base-oxygen
    for i in prange(N):
        xi = base_coords[i, 0]; yi = base_coords[i, 1]; zi = base_coords[i, 2]
        ti = base_types[i]
        for k in range(M):
            dx = xi - oxy_coords[k, 0]
            dy = yi - oxy_coords[k, 1]
            dz = zi - oxy_coords[k, 2]
            r = (dx * dx + dy * dy + dz * dz) ** 0.5
            if r < cutoff:
                b = _bin_index(r)
                # oxygen-type index clamped to 0 for all oxygens
                E_bo += pmf_bo[ti, 0, b]

    # Oxygen-oxygen
    for k in prange(M - 1):
        xk = oxy_coords[k, 0]; yk = oxy_coords[k, 1]; zk = oxy_coords[k, 2]
        for l in range(k + 1, M):
            dx = xk - oxy_coords[l, 0]
            dy = yk - oxy_coords[l, 1]
            dz = zk - oxy_coords[l, 2]
            r = (dx * dx + dy * dy + dz * dz) ** 0.5
            if r < cutoff:
                b = _bin_index(r)
                E_oo += pmf_oo[0, 0, b]

    return E_bb + E_bo + E_oo


@njit(cache=True, fastmath=True)
def _torsion_angle_jit(
    a: NDArray, b: NDArray, c: NDArray, d: NDArray
) -> float:
    """Praxelis torsion from four Cartesian points (radians)."""
    b1 = b - a; b2 = c - b; b3 = d - c
    n1 = np.cross(b1, b2); n2 = np.cross(b2, b3)
    n1_norm = (n1[0]**2 + n1[1]**2 + n1[2]**2) ** 0.5
    n2_norm = (n2[0]**2 + n2[1]**2 + n2[2]**2) ** 0.5
    if n1_norm < 1e-9 or n2_norm < 1e-9:
        return 0.0
    n1 /= n1_norm; n2 /= n2_norm
    b2u = b2 / (b2[0]**2 + b2[1]**2 + b2[2]**2) ** 0.5
    m1 = np.cross(n1, b2u)
    x = n1[0]*n2[0] + n1[1]*n2[1] + n1[2]*n2[2]
    y = m1[0]*n2[0] + m1[1]*n2[1] + m1[2]*n2[2]
    return np.arctan2(y, x)


@njit(cache=True, fastmath=True)
def _backbone_torsion_energy_jit(
    phosphate_coords: NDArray,   # (N, 3) – P atoms in order
    c4p_coords: NDArray,         # (N, 3) – C4' atoms
    ideal_torsions: NDArray,     # (7,) radians
    k_torsion: NDArray,          # (7,) kcal/mol/rad²
) -> float:
    """Harmonic torsion energy around backbone dihedrals (alpha proxy)."""
    N = phosphate_coords.shape[0]
    if N < 4:
        return 0.0
    E = 0.0
    for i in range(N - 3):
        # Use P...P torsion as a proxy for backbone state
        tau = _torsion_angle_jit(
            phosphate_coords[i],
            phosphate_coords[i + 1],
            phosphate_coords[i + 2],
            phosphate_coords[i + 3],
        )
        # Compare to alpha ideal (index 0)
        diff = tau - ideal_torsions[0]
        # Wrap to [-pi, pi]
        while diff > np.pi:  diff -= 2.0 * np.pi
        while diff < -np.pi: diff += 2.0 * np.pi
        E += 0.5 * k_torsion[0] * diff * diff
    # C4' torsion as gamma proxy
    for i in range(N - 3):
        tau = _torsion_angle_jit(
            c4p_coords[i], c4p_coords[i+1],
            c4p_coords[i+2], c4p_coords[i+3],
        )
        diff = tau - ideal_torsions[2]
        while diff > np.pi:  diff -= 2.0 * np.pi
        while diff < -np.pi: diff += 2.0 * np.pi
        E += 0.5 * k_torsion[2] * diff * diff
    return E


@njit(cache=True, fastmath=True)
def _harmonic_restraint_jit(
    p_coords: NDArray,    # (N, 3) current
    p_ref: NDArray,       # (N, 3) reference (DL model)
    k: float,
) -> float:
    """Sum of 0.5*k*|Δr|² over all P atoms."""
    E = 0.0
    for i in range(p_coords.shape[0]):
        dx = p_coords[i, 0] - p_ref[i, 0]
        dy = p_coords[i, 1] - p_ref[i, 1]
        dz = p_coords[i, 2] - p_ref[i, 2]
        E += dx*dx + dy*dy + dz*dz
    return 0.5 * k * E


# ═══════════════════════════════════════════════════════════════════════════════
# 3.  SUGAR PUCKER SCORING (C3'-endo / C2'-endo)
# ═══════════════════════════════════════════════════════════════════════════════

def _pseudorotation_phase(
    c1p: NDArray, c2p: NDArray, c3p: NDArray, c4p: NDArray, o4p: NDArray
) -> float:
    """
    Altona-Sundaralingam pseudorotation phase P (degrees).
    Preferred RNA: C3'-endo  (P ≈ 0–36°).
    """
    def dih(a, b, c, d):
        b1 = b - a; b2 = c - b; b3 = d - c
        n1 = np.cross(b1, b2); n2 = np.cross(b2, b3)
        nn1 = np.linalg.norm(n1); nn2 = np.linalg.norm(n2)
        if nn1 < 1e-9 or nn2 < 1e-9: return 0.0
        n1 /= nn1; n2 /= nn2
        x = np.dot(n1, n2)
        m = np.cross(n1, b2 / np.linalg.norm(b2))
        y = np.dot(m, n2)
        return np.degrees(np.arctan2(y, x))

    v0 = dih(c4p, o4p, c1p, c2p)
    v1 = dih(o4p, c1p, c2p, c3p)
    v2 = dih(c1p, c2p, c3p, c4p)
    v3 = dih(c2p, c3p, c4p, o4p)
    v4 = dih(c3p, c4p, o4p, c1p)

    # Tan formula
    num = (v4 + v1) - (v3 + v0)
    den = 2.0 * v2 * (np.sin(np.radians(36.0)) + np.sin(np.radians(72.0)))
    P = np.degrees(np.arctan2(num, den))
    return P % 360.0


def _sugar_pucker_energy(
    nucleotide_atoms: Dict[str, NDArray],
    preferred: str = "C3'-endo",
    k_sugar: float = 2.0,
) -> float:
    """Harmonic energy around preferred sugar pucker."""
    required = {"C1'", "C2'", "C3'", "C4'", "O4'"}
    if not required.issubset(nucleotide_atoms):
        return 0.0
    P_obs = _pseudorotation_phase(
        nucleotide_atoms["C1'"], nucleotide_atoms["C2'"],
        nucleotide_atoms["C3'"], nucleotide_atoms["C4'"],
        nucleotide_atoms["O4'"],
    )
    P_target = 18.0 if preferred == "C3'-endo" else 162.0
    diff = P_obs - P_target
    # Wrap
    if diff > 180.0:  diff -= 360.0
    if diff < -180.0: diff += 360.0
    return 0.5 * k_sugar * (np.radians(diff)) ** 2


# ═══════════════════════════════════════════════════════════════════════════════
# 4.  COARSE-GRAINED STRUCTURE REPRESENTATION
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class CoarseRNA:
    """
    Minimal coarse representation: P, C4', and glycosidic N per residue.

    Parameters
    ----------
    p_coords  : (N, 3) phosphate positions
    c4p_coords: (N, 3) C4' positions
    n_coords  : (N, 3) N1 (pyrimidine) or N9 (purine) positions
    seq_idx   : (N,)   base type 0=A 1=C 2=G 3=U
    """
    p_coords:   NDArray
    c4p_coords: NDArray
    n_coords:   NDArray
    seq_idx:    NDArray

    # ── optional full-atom oxygen positions for base-oxygen terms ──
    oxy_coords: Optional[NDArray] = field(default=None)

    @classmethod
    def from_flat(cls, flat: NDArray, n_residues: int, seq_idx: NDArray) -> "CoarseRNA":
        """Reconstruct from flat parameter vector (3*3*N)."""
        coords = flat.reshape(n_residues, 3, 3)
        return cls(
            p_coords   = coords[:, 0, :],
            c4p_coords = coords[:, 1, :],
            n_coords   = coords[:, 2, :],
            seq_idx    = seq_idx,
        )

    def to_flat(self) -> NDArray:
        """Flatten to 1-D for scipy.optimize."""
        N = self.p_coords.shape[0]
        coords = np.stack([self.p_coords, self.c4p_coords, self.n_coords], axis=1)
        return coords.ravel()


# ═══════════════════════════════════════════════════════════════════════════════
# 5.  FULL ENERGY FUNCTION
# ═══════════════════════════════════════════════════════════════════════════════

class BRiQEnergyFunction:
    """
    Vectorized + JIT-compiled knowledge-based energy for coarse RNA.

    Energy terms
    ------------
    E_total = w_bb * E_bb_bo_oo    (statistical pairwise potentials)
            + w_tor * E_backbone   (torsion restraints)
            + w_res * E_restraint  (P-atom anchor to DL model)
    """

    def __init__(
        self,
        rna_ref: CoarseRNA,
        restraint_weight: float = _K_RESTR_DEFAULT,
        cutoff: float = 18.0,
        w_bb: float = 1.0,
        w_tor: float = 0.5,
    ):
        self.ref = rna_ref
        self.p_ref = rna_ref.p_coords.copy()
        self.seq_idx = rna_ref.seq_idx.astype(np.int32)
        self.N = rna_ref.p_coords.shape[0]
        self.k_res = restraint_weight
        self.cutoff = cutoff
        self.w_bb = w_bb
        self.w_tor = w_tor

        # Dummy oxygen coords if none provided (use P atoms)
        self._oxy = (
            rna_ref.oxy_coords
            if rna_ref.oxy_coords is not None
            else rna_ref.p_coords
        )

        # JIT-warm-up on a tiny fake structure so first real call is fast
        _dummy = np.zeros((2, 3), dtype=np.float64)
        _dummy_i = np.zeros(2, dtype=np.int32)
        _pairwise_energy_jit(
            _dummy, _dummy_i, _dummy,
            _PMF_BB, _PMF_BO, _PMF_OO, self.cutoff,
        )

    # ── internal helpers ────────────────────────────────────────────────────

    def _unpack(self, flat: NDArray) -> CoarseRNA:
        return CoarseRNA.from_flat(flat, self.N, self.seq_idx)

    # ── public interface ────────────────────────────────────────────────────

    def __call__(self, flat: NDArray) -> float:
        return self.energy(flat)

    def energy(self, flat: NDArray) -> float:
        rna = self._unpack(flat)

        E_pair = _pairwise_energy_jit(
            rna.n_coords.astype(np.float64),
            self.seq_idx,
            self._oxy.astype(np.float64),
            _PMF_BB, _PMF_BO, _PMF_OO,
            self.cutoff,
        )

        E_tor = _backbone_torsion_energy_jit(
            rna.p_coords.astype(np.float64),
            rna.c4p_coords.astype(np.float64),
            np.radians(IDEAL_TORSIONS_AFORM),
            _K_TORSION,
        )

        E_res = _harmonic_restraint_jit(
            rna.p_coords.astype(np.float64),
            self.p_ref.astype(np.float64),
            self.k_res,
        )

        return self.w_bb * E_pair + self.w_tor * E_tor + E_res

    def numerical_gradient(self, flat: NDArray, eps: float = 1e-4) -> NDArray:
        """
        Forward finite-difference gradient (used by scipy L-BFGS-B).

        Works on an internal copy of ``flat`` so the caller's array is
        never mutated — important when Numba fastmath+prange non-determinism
        makes repeated in-place +/-eps operations produce floating-point drift.
        """
        x  = flat.copy()            # defensive copy — never mutate caller's array
        E0 = self.energy(x)
        grad = np.empty_like(x)
        for i in range(len(x)):
            x[i] += eps
            grad[i] = (self.energy(x) - E0) / eps
            x[i] -= eps
        return grad

    def energy_and_gradient(self, flat: NDArray) -> Tuple[float, NDArray]:
        """
        Return ``(energy, gradient)`` from a single coordinated pass.

        E0 is computed once and reused inside the FD loop so that
        ``energy_and_gradient(x)[0]`` is bit-identical to ``energy(x)``
        — avoids any inconsistency from independent double energy calls
        with Numba fastmath+prange enabled.
        """
        x  = flat.copy()            # defensive copy
        E0 = self.energy(x)
        grad = np.empty_like(x)
        eps = 1e-4
        for i in range(len(x)):
            x[i] += eps
            grad[i] = (self.energy(x) - E0) / eps
            x[i] -= eps
        return E0, grad


# ═══════════════════════════════════════════════════════════════════════════════
# 6.  METROPOLIS MONTE CARLO (QRNAS-style backbone moves)
# ═══════════════════════════════════════════════════════════════════════════════

class MetropolisMC:
    """
    Fast QRNAS-style Metropolis MC for backbone-specific fixes.

    Moves
    -----
    - Random Gaussian displacement of a single P atom (backbone)
    - Random Gaussian displacement of a single C4' atom (sugar pucker proxy)
    - Small rotation of a whole nucleotide around its C4'–P axis
    """

    def __init__(
        self,
        energy_fn: BRiQEnergyFunction,
        temperature: float = 300.0,   # K
        step_size: float = 0.2,       # Å
        seed: Optional[int] = None,
    ):
        self.efn = energy_fn
        self.kT = 0.001987 * temperature   # kcal/mol
        self.step = step_size
        self.rng = np.random.default_rng(seed)
        self.n_accept = 0
        self.n_total = 0

    @property
    def acceptance_rate(self) -> float:
        return self.n_accept / max(1, self.n_total)

    def run(self, flat: NDArray, n_steps: int) -> NDArray:
        """Run n_steps of Metropolis MC; returns updated flat coords."""
        x = flat.copy()
        E = self.efn.energy(x)
        n = self.efn.N

        for _ in range(n_steps):
            # choose move type
            move_type = self.rng.integers(0, 3)
            x_new = x.copy()

            if move_type == 0:
                # perturb a random P
                res = self.rng.integers(0, n)
                x_new[res * 9: res * 9 + 3] += self.rng.normal(0, self.step, 3)
            elif move_type == 1:
                # perturb a random C4'
                res = self.rng.integers(0, n)
                x_new[res * 9 + 3: res * 9 + 6] += self.rng.normal(0, self.step, 3)
            else:
                # rigid translation of whole residue (small)
                res = self.rng.integers(0, n)
                delta = self.rng.normal(0, self.step * 0.5, 3)
                x_new[res * 9:     res * 9 + 3] += delta
                x_new[res * 9 + 3: res * 9 + 6] += delta
                x_new[res * 9 + 6: res * 9 + 9] += delta

            E_new = self.efn.energy(x_new)
            dE = E_new - E
            self.n_total += 1

            if dE < 0 or self.rng.uniform() < np.exp(-dE / self.kT):
                x = x_new
                E = E_new
                self.n_accept += 1

            # Adaptive step size
            if self.n_total % 100 == 0:
                if self.acceptance_rate < 0.2:
                    self.step *= 0.9
                elif self.acceptance_rate > 0.5:
                    self.step *= 1.1

        return x


# ═══════════════════════════════════════════════════════════════════════════════
# 7.  HIGH-LEVEL REFINEMENT ORCHESTRATOR
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class RefinementResult:
    """Output of a single BRiQ refinement run."""
    refined_coords: NDArray          # (N_residues, 3, 3) – P / C4' / N
    initial_energy: float
    final_energy:   float
    delta_energy:   float
    n_steps_lbfgs:  int
    n_steps_mc:     int
    mc_acceptance:  float
    converged:      bool
    sequence:       str


class BRiQRefinement:
    """
    End-to-end BRiQ + QRNAS-style post-hoc refinement.

    Parameters
    ----------
    n_steps : int
        Total refinement iterations (split between L-BFGS-B and MC).
    restraint_weight : float
        Spring constant for P-atom anchor (kcal/mol/Å²).
        Higher = tighter anchor, less drift from DL model.
    temperature : float
        Temperature for MC phase (K).
    lbfgs_frac : float
        Fraction of n_steps used for gradient minimization.
    w_bb, w_tor : float
        Weights for pairwise and backbone torsion terms.
    seed : int | None
        Random seed for reproducibility.
    verbose : bool
    """

    def __init__(
        self,
        n_steps: int = 300,
        restraint_weight: float = _K_RESTR_DEFAULT,
        temperature: float = 300.0,
        lbfgs_frac: float = 0.6,
        w_bb: float = 1.0,
        w_tor: float = 0.5,
        seed: Optional[int] = None,
        verbose: bool = True,
    ):
        self.n_steps         = n_steps
        self.k_res           = restraint_weight
        self.temperature     = temperature
        self.lbfgs_frac      = lbfgs_frac
        self.w_bb            = w_bb
        self.w_tor           = w_tor
        self.seed            = seed
        self.verbose         = verbose

    # ── helpers ─────────────────────────────────────────────────────────────

    @staticmethod
    def _parse_sequence(sequence: str) -> NDArray:
        """Convert nucleotide string to integer array (0-3)."""
        seq = sequence.upper().replace("T", "U")
        return np.array([_AA_CODES.get(b, 0) for b in seq], dtype=np.int32)

    @staticmethod
    def _coords_to_coarse(coords: NDArray) -> Tuple[NDArray, NDArray, NDArray]:
        """
        Extract P (idx 0), C4' (idx 3), N1/N9 (idx 11) per residue
        from a full-atom array of shape (N_residues * N_atoms_per_res, 3).

        If coords is already (N, 3, 3) treat as [P, C4', N] directly.
        """
        if coords.ndim == 3 and coords.shape[1] == 3:
            return coords[:, 0, :], coords[:, 1, :], coords[:, 2, :]

        # Assume 12 heavy atoms per nucleotide (simplified)
        n_atoms_per_res = 12
        N = coords.shape[0] // n_atoms_per_res
        if N < 1:
            raise ValueError(
                f"Expected at least 12 atoms, got {coords.shape[0]}. "
                "Pass (N,3,3) array with [P, C4', N] per residue."
            )
        p   = coords[0::n_atoms_per_res][:N]
        c4p = coords[3::n_atoms_per_res][:N]
        n   = coords[11::n_atoms_per_res][:N]
        return p, c4p, n

    # ── main API ────────────────────────────────────────────────────────────

    def refine(
        self,
        coords: NDArray,
        sequence: str,
        oxy_coords: Optional[NDArray] = None,
    ) -> Tuple[NDArray, RefinementResult]:
        """
        Refine an RNA structure.

        Parameters
        ----------
        coords : NDArray
            Either (N_residues, 3, 3) coarse [P/C4'/N] or
            (N_atoms_total, 3) full-atom (12 heavy atoms/residue assumed).
        sequence : str
            RNA sequence string (same length as N_residues).
        oxy_coords : NDArray | None
            Optional (M, 3) oxygen coordinates for base-oxygen terms.

        Returns
        -------
        refined_coords : NDArray  (N_residues, 3, 3)
        result : RefinementResult
        """
        seq_idx = self._parse_sequence(sequence)
        p, c4p, n = self._coords_to_coarse(coords)
        N = p.shape[0]

        if len(seq_idx) != N:
            raise ValueError(
                f"Sequence length {len(sequence)} ≠ N_residues {N}"
            )

        rna_ref = CoarseRNA(
            p_coords=p.astype(np.float64),
            c4p_coords=c4p.astype(np.float64),
            n_coords=n.astype(np.float64),
            seq_idx=seq_idx,
            oxy_coords=oxy_coords,
        )

        efn = BRiQEnergyFunction(
            rna_ref,
            restraint_weight=self.k_res,
            w_bb=self.w_bb,
            w_tor=self.w_tor,
        )

        x0 = rna_ref.to_flat()
        E0 = efn.energy(x0)

        if self.verbose:
            print(f"[BRiQ] N_residues={N}  E_initial={E0:.4f} kcal/mol")

        # ── Phase 1: L-BFGS-B gradient descent ─────────────────────────────
        n_lbfgs = int(self.n_steps * self.lbfgs_frac)
        if self.verbose:
            print(f"[BRiQ] Phase 1 – L-BFGS-B  ({n_lbfgs} max iter)")

        result_opt = minimize(
            efn.energy_and_gradient,
            x0,
            method="L-BFGS-B",
            jac=True,
            options={"maxiter": n_lbfgs, "ftol": 1e-8, "gtol": 1e-5},
        )
        x_lbfgs = result_opt.x
        E_lbfgs = result_opt.fun

        if self.verbose:
            print(f"[BRiQ] After L-BFGS-B: E={E_lbfgs:.4f}  converged={result_opt.success}")

        # ── Phase 2: Metropolis MC (QRNAS backbone moves) ──────────────────
        n_mc = self.n_steps - n_lbfgs
        mc = MetropolisMC(
            efn,
            temperature=self.temperature,
            step_size=0.15,
            seed=self.seed,
        )
        if self.verbose:
            print(f"[BRiQ] Phase 2 – Metropolis MC ({n_mc} steps)")

        x_final = mc.run(x_lbfgs, n_mc)
        E_final = efn.energy(x_final)

        if self.verbose:
            print(
                f"[BRiQ] Final E={E_final:.4f}  ΔE={E_final-E0:.4f}  "
                f"MC acceptance={mc.acceptance_rate:.2%}"
            )

        # ── Unpack result ───────────────────────────────────────────────────
        rna_final = CoarseRNA.from_flat(x_final, N, seq_idx)
        refined_coords = np.stack(
            [rna_final.p_coords, rna_final.c4p_coords, rna_final.n_coords],
            axis=1,
        )

        info = RefinementResult(
            refined_coords = refined_coords,
            initial_energy = E0,
            final_energy   = E_final,
            delta_energy   = E_final - E0,
            n_steps_lbfgs  = n_lbfgs,
            n_steps_mc     = n_mc,
            mc_acceptance  = mc.acceptance_rate,
            converged      = result_opt.success,
            sequence       = sequence,
        )
        return refined_coords, info

    # ── Batch refinement for multiple candidate structures ──────────────────

    def refine_candidates(
        self,
        candidate_coords: List[NDArray],
        sequence: str,
        oxy_coords_list: Optional[List[Optional[NDArray]]] = None,
    ) -> List[Tuple[NDArray, RefinementResult]]:
        """
        Refine a list of candidate structures (e.g. 5 DL predictions).

        Returns results sorted by final energy (best first).
        """
        if oxy_coords_list is None:
            oxy_coords_list = [None] * len(candidate_coords)

        results = []
        for i, (coords, oxy) in enumerate(zip(candidate_coords, oxy_coords_list)):
            if self.verbose:
                print(f"\n{'='*60}")
                print(f"  Refining candidate {i+1}/{len(candidate_coords)}")
                print(f"{'='*60}")
            refined, info = self.refine(coords, sequence, oxy)
            results.append((refined, info))

        # Sort by final energy
        results.sort(key=lambda r: r[1].final_energy)
        if self.verbose:
            print("\n[BRiQ] Candidate ranking after refinement:")
            for rank, (_, info) in enumerate(results, 1):
                print(
                    f"  #{rank}  E={info.final_energy:.4f}  "
                    f"ΔE={info.delta_energy:.4f}  "
                    f"converged={info.converged}"
                )
        return results


# ═══════════════════════════════════════════════════════════════════════════════
# 8.  GEOMETRY SCORING (MolProbity proxy metrics)
# ═══════════════════════════════════════════════════════════════════════════════

def score_geometry(
    p_coords: NDArray,
    c4p_coords: NDArray,
    n_coords: NDArray,
    sequence: str,
) -> Dict[str, float]:
    """
    Compute lightweight geometry quality metrics analogous to MolProbity.

    Returns
    -------
    dict with keys:
        rms_backbone_deviation : float  Å (lower = better local geometry)
        mean_stacking_distance : float  Å (ideal ≈ 3.3–3.8)
        fraction_WC_contacts   : float  fraction of potential WC pairs within 11 Å
        pseudorotation_score   : float  fraction near C3'-endo (RNA ideal)
    """
    N = p_coords.shape[0]
    metrics: Dict[str, float] = {}

    # RMS deviation of consecutive P–P distances from A-form ideal (5.9 Å)
    pp_ideal = 5.9
    pp_dists = np.linalg.norm(np.diff(p_coords, axis=0), axis=1)
    metrics["rms_backbone_deviation"] = float(
        np.sqrt(np.mean((pp_dists - pp_ideal) ** 2))
    )

    # Mean N-N distances between adjacent residues (stacking proxy)
    nn_dists = np.linalg.norm(np.diff(n_coords, axis=0), axis=1)
    metrics["mean_stacking_distance"] = float(np.mean(nn_dists)) if len(nn_dists) > 0 else 0.0

    # Fraction of potential WC pairs within pairing distance
    seq = sequence.upper().replace("T", "U")
    wc_pairs = {("A", "U"), ("U", "A"), ("G", "C"), ("C", "G")}
    n_possible = 0
    n_contact = 0
    for i in range(N):
        for j in range(i + 4, N):   # skip neighbours
            if (seq[i], seq[j]) in wc_pairs:
                n_possible += 1
                dist = np.linalg.norm(n_coords[i] - n_coords[j])
                if 8.0 < dist < 14.0:
                    n_contact += 1
    metrics["fraction_WC_contacts"] = n_contact / max(1, n_possible)

    # Pseudorotation score: penalise large deviations from C3'-endo
    # (we use C4'-P distance as a 1D proxy: ideal A-form ≈ 5.1 Å)
    c4p_p_dists = np.linalg.norm(p_coords - c4p_coords, axis=1)
    ideal_c4p_p = 5.1
    pseudo_score = float(np.mean(np.abs(c4p_p_dists - ideal_c4p_p) < 0.5))
    metrics["pseudorotation_score"] = pseudo_score

    return metrics


# ═══════════════════════════════════════════════════════════════════════════════
# 9.  DEMONSTRATION / SMOKE TEST
# ═══════════════════════════════════════════════════════════════════════════════

def _demo_random_helix(n_residues: int = 12, seed: int = 7) -> NDArray:
    """
    Build a noisy A-form helix as a test structure.
    Returns (N, 3, 3) coarse coords [P / C4' / N].
    """
    rng = np.random.default_rng(seed)
    # Ideal A-form helix parameters
    rise_per_res = 2.81   # Å
    twist_per_res = 32.7  # deg
    r_p = 9.0; r_c4 = 7.5; r_n = 5.5

    coords = np.zeros((n_residues, 3, 3))
    for i in range(n_residues):
        angle = np.radians(i * twist_per_res)
        z = i * rise_per_res
        for r_idx, radius in enumerate([r_p, r_c4, r_n]):
            coords[i, r_idx, 0] = radius * np.cos(angle)
            coords[i, r_idx, 1] = radius * np.sin(angle)
            coords[i, r_idx, 2] = z

    # Add Gaussian noise to simulate DL model imperfections
    coords += rng.normal(0, 0.5, coords.shape)
    return coords


def demo(n_residues: int = 12, n_steps: int = 200) -> None:
    """Run a quick end-to-end demo."""
    print("=" * 65)
    print("  BRiQ / QRNAS-style RNA Refinement  –  Demo")
    print("=" * 65)

    sequence = ("AUGCAUGCAUGC")[:n_residues]
    print(f"\nSequence : {sequence}  ({n_residues} nt)")

    # Build 5 slightly different "DL predictions"
    candidates = [
        _demo_random_helix(n_residues, seed=s) for s in range(5)
    ]
    print(f"Generated {len(candidates)} candidate structures\n")

    refiner = BRiQRefinement(
        n_steps=n_steps,
        restraint_weight=10.0,
        temperature=300.0,
        lbfgs_frac=0.6,
        seed=42,
        verbose=True,
    )

    ranked = refiner.refine_candidates(candidates, sequence)

    print("\n── Geometry metrics (best candidate) ──")
    best_coords, best_info = ranked[0]
    metrics = score_geometry(
        best_coords[:, 0, :],
        best_coords[:, 1, :],
        best_coords[:, 2, :],
        sequence,
    )
    for k, v in metrics.items():
        print(f"  {k:<35s}: {v:.4f}")

    print("\nDone ✓")
    return ranked


# ═══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    demo()
