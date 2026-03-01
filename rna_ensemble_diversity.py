"""
RNA Ensemble Diversity via Torsion-Space Diffusion + Consensus Reranking
========================================================================
Numba-JIT and vectorized implementation for high-performance RNA structure
ensemble generation and scoring for competition "best-of-5" submission.

Pipeline:
  1. Torsion-space diffusion → generate 20-50 diverse candidates
  2. Consensus contact map scoring (base-pair frequency)
  3. lDDT-RNA proxy scoring via graph features
  4. Combined reranking → select top-5

RNA backbone torsion angles:  α, β, γ, δ, ε, ζ  (backbone) + χ (glycosidic)
All angles in radians; coordinates in Ångströms.

Dependencies: numpy, numba, scipy
Optional:      torch (for learned score model), networkx (graph scoring)

References
----------
- lDDT-RNA: RNA-Puzzles / CASP15 evaluation metric
- DFold: torsion-space generative model without MSA
- RNArank: graph-network quality assessment
- RNAdvisor 2: multi-metric ensemble reranking
"""

from __future__ import annotations

import math
import time
import warnings
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple

import numpy as np
from numba import njit, prange, vectorize, float64, int32
from scipy.special import softmax

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class RNAStructure:
    """Lightweight container for a single RNA structure candidate."""
    sequence: str                          # one-letter RNA sequence (ACGU)
    coords: np.ndarray                     # shape (N_res, 3), P-atom trace
    torsions: np.ndarray                   # shape (N_res, 7)  [α,β,γ,δ,ε,ζ,χ]
    atom_names: List[str] = field(default_factory=list)
    residue_ids: Optional[np.ndarray] = None
    # Scoring outputs (populated by ConsensusReranker)
    consensus_score: float = 0.0
    lddt_score: float = 0.0
    combined_score: float = 0.0

    def __post_init__(self):
        self.coords   = np.asarray(self.coords,   dtype=np.float64)
        self.torsions = np.asarray(self.torsions, dtype=np.float64)
        if self.residue_ids is None:
            self.residue_ids = np.arange(len(self.sequence), dtype=np.int32)


# ---------------------------------------------------------------------------
# ── JIT-compiled geometry kernels ──────────────────────────────────────────
# ---------------------------------------------------------------------------

@njit(cache=True, fastmath=True)
def _dihedral_angle(p0: np.ndarray, p1: np.ndarray,
                    p2: np.ndarray, p3: np.ndarray) -> float:
    """Dihedral angle (radians) from four 3-D points using Praxeolitic formula."""
    b1 = p1 - p0
    b2 = p2 - p1
    b3 = p3 - p2
    n1 = np.cross(b1, b2)
    n2 = np.cross(b2, b3)
    b2_norm = b2 / (math.sqrt(b2[0]**2 + b2[1]**2 + b2[2]**2) + 1e-12)
    m1 = np.cross(n1, b2_norm)
    x = n1[0]*n2[0] + n1[1]*n2[1] + n1[2]*n2[2]
    y = m1[0]*n2[0] + m1[1]*n2[1] + m1[2]*n2[2]
    return math.atan2(y, x)


@njit(cache=True, fastmath=True, parallel=True)
def compute_all_dihedrals(coords: np.ndarray,
                           backbone_indices: np.ndarray) -> np.ndarray:
    """
    Parallel batch dihedral computation.

    Parameters
    ----------
    coords           : (N_atoms, 3)
    backbone_indices : (N_res, 4)  — four atom indices per torsion window

    Returns
    -------
    angles : (N_res,)  in radians [-π, π]
    """
    n = backbone_indices.shape[0]
    angles = np.empty(n, dtype=np.float64)
    for i in prange(n):
        a, b, c, d = (backbone_indices[i, 0], backbone_indices[i, 1],
                      backbone_indices[i, 2], backbone_indices[i, 3])
        angles[i] = _dihedral_angle(coords[a], coords[b],
                                     coords[c], coords[d])
    return angles


@njit(cache=True, fastmath=True)
def _local_displacement(alpha: float, beta: float,
                         bond_len: float = 5.9) -> np.ndarray:
    """
    Displacement vector from α and β torsion angles.
    bond_len ~ 5.9 Å is the mean P–P distance in RNA.
    """
    sa, ca = math.sin(alpha), math.cos(alpha)
    sb, cb = math.sin(beta),  math.cos(beta)
    # x along chain, y/z from torsion projection
    return np.array([bond_len * cb,
                     bond_len * sb * ca,
                     bond_len * sb * sa], dtype=np.float64)


@njit(cache=True, fastmath=True)
def batch_torsion_to_coords(torsions: np.ndarray,
                             seed_coord: np.ndarray) -> np.ndarray:
    """
    Reconstruct P-atom trace from torsion angles using chain-growth kinematics.
    NOTE: This is a simplified forward model; a full reconstruction would use
    the NERF/NeRF algorithm with full bond geometry.

    Parameters
    ----------
    torsions   : (N_res, 7)  [α, β, γ, δ, ε, ζ, χ]
    seed_coord : (3,)  starting phosphorus

    Returns
    -------
    p_coords : (N_res, 3)
    """
    n = torsions.shape[0]
    p_coords = np.zeros((n, 3), dtype=np.float64)
    p_coords[0, 0] = seed_coord[0]
    p_coords[0, 1] = seed_coord[1]
    p_coords[0, 2] = seed_coord[2]
    for i in range(1, n):
        d = _local_displacement(torsions[i, 0], torsions[i, 1])
        p_coords[i, 0] = p_coords[i-1, 0] + d[0]
        p_coords[i, 1] = p_coords[i-1, 1] + d[1]
        p_coords[i, 2] = p_coords[i-1, 2] + d[2]
    return p_coords


# ---------------------------------------------------------------------------
# ── Contact map kernels ─────────────────────────────────────────────────────
# ---------------------------------------------------------------------------

@njit(cache=True, fastmath=True, parallel=True)
def contact_map_from_coords(coords: np.ndarray,
                             cutoff: float = 20.0) -> np.ndarray:
    """
    Parallel binary contact map for P-atom traces.
    Excludes immediately bonded pairs |i-j| < 2.

    NOTE: P-atom traces use a 20 Å cutoff (vs 8 Å for all-atom C1'/base).
    Sequential P–P distance is ~5.9 Å, so non-sequential contacts only
    appear at 12–25 Å in folded RNA.

    Parameters
    ----------
    coords  : (N, 3)  phosphorus coordinates
    cutoff  : distance threshold in Å  (default 20.0 for P-atom traces)

    Returns
    -------
    cmap : (N, N)  int8
    """
    n = coords.shape[0]
    cmap = np.zeros((n, n), dtype=np.int8)
    c2 = cutoff * cutoff
    for i in prange(n):
        xi, yi, zi = coords[i, 0], coords[i, 1], coords[i, 2]
        for j in range(i + 2, n):          # exclude only direct bond (|i-j|<2)
            dx = xi - coords[j, 0]
            dy = yi - coords[j, 1]
            dz = zi - coords[j, 2]
            if dx*dx + dy*dy + dz*dz < c2:
                cmap[i, j] = 1
                cmap[j, i] = 1
    return cmap


@njit(cache=True, fastmath=True)
def consensus_contact_map(contact_maps: np.ndarray) -> np.ndarray:
    """
    Per-pair contact frequency over an ensemble.

    Parameters
    ----------
    contact_maps : (M, N, N)  int8

    Returns
    -------
    freq_map : (N, N)  float64  ∈ [0, 1]
    """
    m, n, _ = contact_maps.shape
    freq = np.zeros((n, n), dtype=np.float64)
    # Serial over structures to avoid write-race on freq[i,j];
    # inner i/j loops are fast (N typically < 500).
    for k in range(m):
        for i in range(n):
            for j in range(n):
                freq[i, j] += contact_maps[k, i, j]
    inv_m = 1.0 / m
    for i in range(n):
        for j in range(n):
            freq[i, j] *= inv_m
    return freq


@njit(cache=True, fastmath=True)
def score_against_consensus(cmap: np.ndarray,
                             freq_map: np.ndarray) -> float:
    """
    Agreement of a single structure's contact map with the consensus.
    Score = mean freq_map value over the structure's contact pairs.
    High score → structure makes the most frequently observed contacts.

    Returns
    -------
    score : float  (higher = more consensus-like)
    """
    n = cmap.shape[0]
    total, count = 0.0, 0
    for i in range(n):
        for j in range(i + 3, n):
            if cmap[i, j] == 1:
                total += freq_map[i, j]
                count += 1
    return total / max(count, 1)


# ---------------------------------------------------------------------------
# ── lDDT-RNA kernel ─────────────────────────────────────────────────────────
# ---------------------------------------------------------------------------

@njit(cache=True, fastmath=True)
def lddt_rna_proxy(pred_coords: np.ndarray,
                   ref_coords:  np.ndarray,
                   thresholds:  np.ndarray) -> float:
    """
    Vectorised P-atom lDDT-RNA (mirrors CASP15 RNA evaluation).

    For every pair (i,j) within 15 Å in the reference, check whether
    |d_pred(i,j) - d_ref(i,j)| is below each threshold ∈ {0.5,1,2,4} Å.

    Parameters
    ----------
    pred_coords : (N, 3)
    ref_coords  : (N, 3)
    thresholds  : (T,)  e.g. [0.5, 1.0, 2.0, 4.0]

    Returns
    -------
    lddt : float ∈ [0, 1]
    """
    n = pred_coords.shape[0]
    T = thresholds.shape[0]
    if n < 2:
        return 0.0

    preserved = 0.0
    total     = 0.0
    inv_T     = 1.0 / T

    for i in prange(n):
        for j in range(n):
            if i == j:
                continue
            dx = ref_coords[i,0] - ref_coords[j,0]
            dy = ref_coords[i,1] - ref_coords[j,1]
            dz = ref_coords[i,2] - ref_coords[j,2]
            d_ref = math.sqrt(dx*dx + dy*dy + dz*dz)
            if d_ref > 15.0:
                continue

            dx = pred_coords[i,0] - pred_coords[j,0]
            dy = pred_coords[i,1] - pred_coords[j,1]
            dz = pred_coords[i,2] - pred_coords[j,2]
            d_pred = math.sqrt(dx*dx + dy*dy + dz*dz)

            delta = abs(d_pred - d_ref)
            score_ij = 0.0
            for t in range(T):
                if delta < thresholds[t]:
                    score_ij += 1.0
            preserved += score_ij * inv_T
            total += 1.0

    return preserved / max(total, 1.0)


# ---------------------------------------------------------------------------
# ── Graph-feature lDDT predictor (reference-free) ──────────────────────────
# ---------------------------------------------------------------------------

@njit(cache=True, fastmath=True, parallel=True)
def _knn_stats(coords: np.ndarray, k: int) -> np.ndarray:
    """
    Per-residue k-NN distance statistics: [mean, std, min, max].
    Used as structural regularity features.

    Parameters
    ----------
    coords : (N, 3)
    k      : number of nearest neighbours

    Returns
    -------
    feats : (N, 4)
    """
    n = coords.shape[0]
    k = min(k, n - 1)
    feats = np.zeros((n, 4), dtype=np.float64)

    for i in prange(n):
        # compute all distances from i
        dists = np.empty(n - 1, dtype=np.float64)
        idx = 0
        for j in range(n):
            if j == i:
                continue
            dx = coords[i,0]-coords[j,0]
            dy = coords[i,1]-coords[j,1]
            dz = coords[i,2]-coords[j,2]
            dists[idx] = math.sqrt(dx*dx+dy*dy+dz*dz)
            idx += 1

        # partial selection sort for k smallest values
        for p in range(k):
            m = p
            for q in range(p+1, n-1):
                if dists[q] < dists[m]:
                    m = q
            tmp = dists[p]; dists[p] = dists[m]; dists[m] = tmp

        kd = dists[:k]
        s = 0.0
        for v in kd:
            s += v
        mean_d = s / k
        s2 = 0.0
        for v in kd:
            diff = v - mean_d
            s2 += diff * diff
        feats[i, 0] = mean_d
        feats[i, 1] = math.sqrt(s2 / max(k, 1))
        feats[i, 2] = kd[0]
        feats[i, 3] = kd[k-1]

    return feats


def graph_lddt_score(structure: RNAStructure, k: int = 12) -> float:
    """
    Reference-free structural regularity score ∈ [0, 1].
    Based on k-NN distance variance: well-packed P-traces have low variance.
    Higher = more globally regular ≈ better-structured candidate.
    """
    feats     = _knn_stats(structure.coords, k)
    mean_std  = float(feats[:, 1].mean())   # mean per-residue spread
    mean_d    = float(feats[:, 0].mean())   # mean inter-P distance
    regularity = 1.0 / (1.0 + mean_std / (mean_d + 1e-6))
    return float(np.clip(regularity, 0.0, 1.0))


def torsions_to_structure(
    sequence:    str,
    torsions:    np.ndarray,
    seed_coord:  Optional[np.ndarray] = None,
) -> RNAStructure:
    """
    Convenience utility: convert a torsion array to an RNAStructure.

    Reconstructs the P-atom backbone trace from the torsion angles using
    chain-growth kinematics (``batch_torsion_to_coords``), then packages
    the result into an ``RNAStructure`` object ready for scoring.

    Parameters
    ----------
    sequence   : RNA sequence string (ACGU; T→U automatically)
    torsions   : (N_res, 7)  backbone torsion angles [α,β,γ,δ,ε,ζ,χ] in radians
    seed_coord : (3,) starting phosphorus position in Å.
                 Defaults to the origin if not provided.

    Returns
    -------
    structure : RNAStructure  with coords and torsions populated
    """
    sequence = sequence.upper().replace("T", "U")
    torsions = np.asarray(torsions, dtype=np.float64)
    if seed_coord is None:
        seed_coord = np.zeros(3, dtype=np.float64)
    else:
        seed_coord = np.asarray(seed_coord, dtype=np.float64)

    coords = batch_torsion_to_coords(torsions, seed_coord)
    return RNAStructure(
        sequence   = sequence,
        coords     = coords,
        torsions   = torsions,
        atom_names = ["P"] * len(sequence),
    )



# ---------------------------------------------------------------------------
# ── Torsion-space diffusion sampler ────────────────────────────────────────
# ---------------------------------------------------------------------------

# JIT helper — one Overdamped Langevin step on the torus T^7
@njit(cache=True, fastmath=True)
def _langevin_step(torsions: np.ndarray,
                   score:    np.ndarray,
                   eps:      float,
                   noise_s:  float,
                   noise_n:  np.ndarray) -> np.ndarray:
    """
    θ_{t+1} = θ_t + ε·∇log p_t(θ_t) + √(2ε)·noise_scale·η,  η~N(0,I)
    Result is wrapped to [-π, π].
    """
    n, d = torsions.shape
    sq2e = math.sqrt(2.0 * eps)
    out  = np.empty_like(torsions)
    for i in range(n):
        for j in range(d):
            v = (torsions[i,j]
                 + eps  * score[i,j]
                 + sq2e * noise_s * noise_n[i,j])
            # wrap to [-π, π]
            v = v - 2.0*math.pi * math.floor((v + math.pi) / (2.0*math.pi))
            out[i,j] = v
    return out


@njit(cache=True, fastmath=True)
def _analytical_score(torsions: np.ndarray, sigma: float) -> np.ndarray:
    """
    Score function for isotropic wrapped Gaussian diffusion on T^7:
      s(θ,t) = -wrap(θ) / σ²(t)
    """
    sigma2 = sigma * sigma + 1e-8
    wrapped = torsions - 2.0*math.pi * np.round(torsions / (2.0*math.pi))
    return -wrapped / sigma2


class TorsionDiffusionSampler:
    """
    Score-based reverse-diffusion sampler in RNA backbone torsion space.

    Uses annealed Langevin dynamics (DDPM-style) on the 7-torus T^7.
    Works out-of-the-box with an analytical Gaussian score function (no
    training needed); plug in a learned score network via `score_fn`
    for improved mode coverage.

    Parameters
    ----------
    n_residues      : RNA sequence length
    sigma_min/max   : noise schedule endpoints (geometric)
    n_steps         : reverse-process discretisation steps
    step_size       : Langevin step size ε
    noise_scale     : stochasticity multiplier (1=full SDE, 0=prob-flow ODE)
    score_fn        : optional callable(torsions, t) → score (N_res, 7)
    seed            : RNG seed for reproducibility
    """

    # Prior statistics from RNA crystal structure torsion distributions
    # (Schneider et al. 2004; Richardson et al. 2008)
    _MEANS = np.array([-1.07, -1.60, 0.58, 1.36, -1.52, -1.60, -2.34])
    _STDS  = np.array([ 0.52,  0.45, 0.70, 0.41,  0.48,  0.58,  0.52])

    def __init__(
        self,
        n_residues:  int,
        sigma_min:   float = 0.01,
        sigma_max:   float = math.pi,
        n_steps:     int   = 200,
        step_size:   float = 5e-4,
        noise_scale: float = 1.0,
        score_fn:    Optional[Callable] = None,
        seed:        Optional[int] = 42,
    ):
        self.n_res       = n_residues
        self.sigma_min   = sigma_min
        self.sigma_max   = sigma_max
        self.n_steps     = n_steps
        self.step_size   = step_size
        self.noise_scale = noise_scale
        self.score_fn    = score_fn
        self.rng         = np.random.default_rng(seed)

    # ----
    def _sigma(self, t: float) -> float:
        """Geometric noise schedule: σ(t) = σ_min · (σ_max/σ_min)^t."""
        return self.sigma_min * (self.sigma_max / self.sigma_min) ** t

    def _score(self, torsions: np.ndarray, t: float) -> np.ndarray:
        if self.score_fn is not None:
            return self.score_fn(torsions, t)
        return _analytical_score(torsions, self._sigma(t))

    # ----
    def sample(
        self,
        n_samples:    int = 30,
        conditioning: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Generate `n_samples` diverse torsion arrays via reverse diffusion.

        Parameters
        ----------
        n_samples    : number of structure candidates to generate
        conditioning : (N_res, D) optional feature array for conditional generation
                       (currently used to perturb initial particles)

        Returns
        -------
        samples : (n_samples, N_res, 7)  all angles ∈ [-π, π]
        """
        n = self.n_res

        # ── initialise from N(0, σ_max²) on torus ──
        particles = self.rng.normal(0.0, self.sigma_max,
                                    size=(n_samples, n, 7))

        if conditioning is not None:
            # Weak conditioning: bias initial noise toward conditioning signal
            cond = np.asarray(conditioning, dtype=np.float64)
            if cond.shape == (n, 7):
                particles += 0.1 * cond[np.newaxis, :, :]

        # ── reverse diffusion: t: 1 → 0 ──
        times      = np.linspace(1.0, 0.0, self.n_steps + 1)
        step_sizes = np.abs(np.diff(times)) * self.step_size   # all positive

        print(f"\n  [TorsionDiffusion] Reverse SDE — "
              f"{self.n_steps} steps · {n_samples} particles · N_res={n}")
        t0 = time.perf_counter()

        for idx, (t, eps) in enumerate(zip(times[:-1], step_sizes)):
            noise = self.rng.standard_normal((n_samples, n, 7))
            for k in range(n_samples):
                score = self._score(particles[k], t)
                particles[k] = _langevin_step(
                    particles[k], score, eps, self.noise_scale, noise[k])

            if idx % 50 == 0 or idx == self.n_steps - 1:
                elapsed = time.perf_counter() - t0
                print(f"    step {idx+1:4d}/{self.n_steps}  "
                      f"t={t:.3f}  elapsed={elapsed:.2f}s")

        # ── de-normalise toward known RNA torsion statistics ──
        mean = self._MEANS[np.newaxis, np.newaxis, :]
        std  = self._STDS [np.newaxis, np.newaxis, :]
        particles = particles * std + mean
        # final wrap to [-π, π]
        particles = (particles + math.pi) % (2 * math.pi) - math.pi

        elapsed = time.perf_counter() - t0
        print(f"  [TorsionDiffusion] Completed in {elapsed:.2f}s")
        return particles.astype(np.float64)


# ---------------------------------------------------------------------------
# ── Consensus reranker ──────────────────────────────────────────────────────
# ---------------------------------------------------------------------------

class ConsensusReranker:
    """
    Rerank a structure ensemble using:
      1. Contact-map consensus score  (C)
      2. lDDT-RNA proxy score         (L)
      3. Combined score  =  w_c · C + w_l · L

    Parameters
    ----------
    cutoff      : contact distance threshold in Å (8.0 recommended for P-atoms)
    w_consensus : weight for consensus score
    w_lddt      : weight for lDDT-RNA proxy score
    ref_coords  : (N, 3) optional reference P-trace;
                  if None, ensemble centroid is used as pseudo-reference
    lddt_thresholds : lDDT distance bins in Å
    """

    _LDDT_THRESHOLDS = np.array([0.5, 1.0, 2.0, 4.0], dtype=np.float64)

    def __init__(
        self,
        cutoff:          float = 20.0,
        w_consensus:     float = 0.55,
        w_lddt:          float = 0.45,
        ref_coords:      Optional[np.ndarray] = None,
        lddt_thresholds: Optional[np.ndarray] = None,
    ):
        self.cutoff      = cutoff
        self.w_consensus = w_consensus
        self.w_lddt      = w_lddt
        self.ref_coords  = ref_coords
        self.thresholds  = (lddt_thresholds if lddt_thresholds is not None
                            else self._LDDT_THRESHOLDS)

    # ----
    def _build_contact_maps(self, structures: List[RNAStructure]) -> np.ndarray:
        m = len(structures)
        n = structures[0].coords.shape[0]
        cmaps = np.zeros((m, n, n), dtype=np.int8)
        for k, s in enumerate(structures):
            cmaps[k] = contact_map_from_coords(s.coords, self.cutoff)
        return cmaps

    # ----
    def rerank(
        self,
        structures: List[RNAStructure],
        top_k:      int  = 5,
        verbose:    bool = True,
    ) -> List[RNAStructure]:
        """
        Score all structures; return top-k sorted best-first.

        Parameters
        ----------
        structures : list of RNAStructure candidates
        top_k      : number of structures to return
        verbose    : print scoring table

        Returns
        -------
        top_structures : list of RNAStructure, len == top_k
        """
        m = len(structures)
        if verbose:
            print(f"\n  [Reranker] Scoring {m} candidates …")
        t0 = time.perf_counter()

        # ── Step 1: contact maps ──
        cmaps    = self._build_contact_maps(structures)
        # ── Diagnostic: warn if no contacts found (cutoff too small) ──
        total_contacts = int(cmaps.sum())
        if total_contacts == 0:
            import warnings as _w
            _w.warn(
                f"[ConsensusReranker] Zero contacts found across all {m} structures! "
                f"cutoff={self.cutoff} Å may be too small for this coordinate scale. "
                f"Try cutoff=20.0 for P-atom traces.", stacklevel=2)

        freq_map = consensus_contact_map(cmaps)

        # ── Step 2: reference for lDDT ──
        if self.ref_coords is not None:
            ref = self.ref_coords.astype(np.float64)
        else:
            all_coords = np.stack([s.coords for s in structures], axis=0)
            ref = all_coords.mean(axis=0)   # pseudo-reference = centroid

        # ── Step 3: per-structure scoring ──
        for k, s in enumerate(structures):
            s.consensus_score = float(score_against_consensus(cmaps[k], freq_map))
            s.lddt_score      = float(lddt_rna_proxy(s.coords, ref,
                                                      self.thresholds))

        # ── Step 4: min-max normalise and combine ──
        c = np.array([s.consensus_score for s in structures])
        l = np.array([s.lddt_score      for s in structures])

        def _norm(x: np.ndarray) -> np.ndarray:
            rng = x.max() - x.min()
            return (x - x.min()) / (rng + 1e-12)

        c_n = _norm(c)
        l_n = _norm(l)
        for k, s in enumerate(structures):
            s.combined_score = self.w_consensus * c_n[k] + self.w_lddt * l_n[k]

        # ── Step 5: sort ──
        ranked  = sorted(structures, key=lambda s: s.combined_score, reverse=True)
        elapsed = time.perf_counter() - t0

        if verbose:
            pad = 60
            print(f"\n  {'─'*pad}")
            print(f"  {'Rank':>4}  {'Consensus':>9}  {'lDDT-proxy':>10}  "
                  f"{'Combined':>8}")
            print(f"  {'─'*pad}")
            for r, s in enumerate(ranked[:top_k], start=1):
                print(f"  {r:4d}  {s.consensus_score:9.4f}  "
                      f"{s.lddt_score:10.4f}  {s.combined_score:8.4f}")
            print(f"  {'─'*pad}")
            print(f"  [Reranker] Completed in {elapsed:.3f}s")

        return ranked[:top_k]


# ---------------------------------------------------------------------------
# ── Vectorised batch TM-score proxy ────────────────────────────────────────
# ---------------------------------------------------------------------------

@njit(cache=True, fastmath=True, parallel=True)
def batch_tm_score_proxy(pred_all: np.ndarray,
                          ref:     np.ndarray,
                          d0:      float = 5.0) -> np.ndarray:
    """
    Vectorised TM-score proxy for a batch of P-traces vs. one reference.
    Uses simplified formula (no superposition) — use as fast filter only.

    Parameters
    ----------
    pred_all : (M, N, 3)
    ref      : (N, 3)
    d0       : normalisation distance in Å

    Returns
    -------
    tm_proxy : (M,)  ∈ [0, 1]
    """
    m, n, _ = pred_all.shape
    tm   = np.zeros(m, dtype=np.float64)
    d02  = d0 * d0
    invN = 1.0 / n
    for k in prange(m):
        s = 0.0
        for i in range(n):
            dx = pred_all[k,i,0] - ref[i,0]
            dy = pred_all[k,i,1] - ref[i,1]
            dz = pred_all[k,i,2] - ref[i,2]
            di2 = dx*dx + dy*dy + dz*dz
            s  += 1.0 / (1.0 + di2 / d02)
        tm[k] = s * invN
    return tm


# ---------------------------------------------------------------------------
# ── Vectorized angle utilities ──────────────────────────────────────────────
# ---------------------------------------------------------------------------

@vectorize([float64(float64, float64, float64)], cache=True, fastmath=True,
           target='parallel')
def wrap_angle_vec(angle, lo, hi):
    """Element-wise angle wrapping to [lo, hi); works on NumPy arrays."""
    span = hi - lo
    return angle - span * math.floor((angle - lo) / span)


@vectorize([float64(float64, float64)], cache=True, fastmath=True,
           target='parallel')
def von_mises_log_prob(angle, kappa):
    """Log-probability of von Mises distribution (unnormalised)."""
    return kappa * math.cos(angle)


# ---------------------------------------------------------------------------
# ── Full pipeline ───────────────────────────────────────────────────────────
# ---------------------------------------------------------------------------

class EnsembleDiversityPipeline:
    """
    End-to-end RNA ensemble diversity pipeline.

      Torsion diffusion  →  P-trace reconstruction
      →  Consensus contact map scoring
      →  lDDT-RNA proxy scoring
      →  Top-5 selection

    This addresses the "best-of-5" scoring regime: generate many diverse
    candidates so at least one is near-native, then use consensus + lDDT
    to surface it.

    Usage
    -----
    >>> pipeline = EnsembleDiversityPipeline("GCGGAUUUAGCUCAGUUGGG")
    >>> top5 = pipeline.run(n_candidates=30, top_k=5)
    >>> for i, s in enumerate(top5, 1):
    ...     print(f"#{i}  combined={s.combined_score:.4f}")

    Parameters
    ----------
    sequence         : RNA sequence (ACGU; T→U automatically)
    sigma_min/max    : diffusion noise schedule endpoints
    n_diffusion_steps: reverse SDE discretisation (200 recommended)
    contact_cutoff   : Å threshold for base-pair contacts
    w_consensus      : weight for consensus score in final ranking
    w_lddt           : weight for lDDT-proxy in final ranking
    ref_coords       : optional (N,3) reference for lDDT computation
    score_fn         : optional learned score model callable
    seed             : global RNG seed
    """

    def __init__(
        self,
        sequence:          str,
        sigma_min:         float  = 0.01,
        sigma_max:         float  = math.pi,
        n_diffusion_steps: int    = 200,
        contact_cutoff:    float  = 20.0,
        w_consensus:       float  = 0.55,
        w_lddt:            float  = 0.45,
        ref_coords:        Optional[np.ndarray] = None,
        score_fn:          Optional[Callable]   = None,
        seed:              int = 42,
    ):
        self.sequence = sequence.upper().replace("T", "U")
        self.n_res    = len(self.sequence)

        self.sampler = TorsionDiffusionSampler(
            n_residues   = self.n_res,
            sigma_min    = sigma_min,
            sigma_max    = sigma_max,
            n_steps      = n_diffusion_steps,
            score_fn     = score_fn,
            seed         = seed,
        )
        self.reranker = ConsensusReranker(
            cutoff      = contact_cutoff,
            w_consensus = w_consensus,
            w_lddt      = w_lddt,
            ref_coords  = ref_coords,
        )

    # ----
    def run(
        self,
        n_candidates: int  = 30,
        top_k:        int  = 5,
        conditioning: Optional[np.ndarray] = None,
        verbose:      bool = True,
    ) -> List[RNAStructure]:
        """
        Run the full pipeline.

        Parameters
        ----------
        n_candidates : structures to generate (recommend 20–50)
        top_k        : structures to return   (5 for competition)
        conditioning : (N_res, 7) or (N_res, D) optional features
        verbose      : print progress

        Returns
        -------
        top_structures : list of RNAStructure, best-first
        """
        if verbose:
            bar = "═" * 62
            print(f"\n{bar}")
            print(f"  RNA Ensemble Diversity Pipeline   "
                  f"(N_res={self.n_res}, candidates={n_candidates})")
            print(f"{bar}")

        t_total = time.perf_counter()

        # ── 1. Torsion-space diffusion ──
        torsion_samples = self.sampler.sample(
            n_samples=n_candidates, conditioning=conditioning)

        # ── 2. Reconstruct P-atom traces ──
        if verbose:
            print(f"\n  [Reconstruction] Building {n_candidates} P-traces …")
        t1 = time.perf_counter()

        seed_origin = np.zeros(3, dtype=np.float64)
        structures: List[RNAStructure] = []
        for k in range(n_candidates):
            coords = batch_torsion_to_coords(torsion_samples[k], seed_origin)
            s = RNAStructure(
                sequence   = self.sequence,
                coords     = coords,
                torsions   = torsion_samples[k],
                atom_names = ["P"] * self.n_res,
            )
            # Augment with reference-free regularity score
            s.lddt_score = graph_lddt_score(s)
            structures.append(s)

        if verbose:
            print(f"  [Reconstruction] Done in {time.perf_counter()-t1:.3f}s")

        # ── 3. Consensus reranking ──
        top_structures = self.reranker.rerank(
            structures, top_k=top_k, verbose=verbose)

        if verbose:
            elapsed = time.perf_counter() - t_total
            print(f"\n  ✓ Pipeline complete in {elapsed:.2f}s")
            print(f"    Returning top-{top_k} structures for submission.")
            print("═" * 62 + "\n")

        return top_structures


# ---------------------------------------------------------------------------
# ── CLI demo / smoke test ───────────────────────────────────────────────────
# ---------------------------------------------------------------------------

def _demo():
    """Smoke test on a short tRNA fragment (reduced steps for speed)."""
    print("\n" + "─"*60)
    print("  RNA Ensemble Diversity — demo")
    print("─"*60)

    # 70-nt tRNA-Phe-like sequence
    seq = ("GCGGAUUUAGCUCAGUUGGGAGAGCGCCAGACUGAAGAUCUGGAGG"
           "UCCUGUGUUCGAUCCACAGAAUUCGCA")
    print(f"\n  Sequence : {seq[:40]}… ({len(seq)} nt)\n")

    pipeline = EnsembleDiversityPipeline(
        sequence          = seq,
        n_diffusion_steps = 50,    # use 200+ for production runs
        seed              = 0,
    )

    top5 = pipeline.run(n_candidates=12, top_k=5)

    print("\n  ┌─────────────────────────────────────────────────┐")
    print("  │  Final top-5 submission candidates               │")
    print("  ├──────┬───────────┬────────────┬──────────────────┤")
    print("  │ Rank │ Consensus │ lDDT-proxy │ Combined score   │")
    print("  ├──────┼───────────┼────────────┼──────────────────┤")
    for i, s in enumerate(top5, start=1):
        print(f"  │  {i:3d} │  {s.consensus_score:7.4f}  │   "
              f"{s.lddt_score:7.4f}  │  {s.combined_score:14.4f}  │")
    print("  └──────┴───────────┴────────────┴──────────────────┘\n")


if __name__ == "__main__":
    _demo()
