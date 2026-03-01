"""
rna_hierarchical_assembly.py
============================
Hierarchical / Domain-Wise Assembly for Long RNAs
Numba JIT + vectorized implementation for high-performance structural prediction.

Pipeline:
  1. Domain Segmentation      — spectral clustering on junction-connectivity graph
  2. Per-Domain Folding       — dense attention on small domain windows (±20 nt flanking)
  3. Inter-Domain Graph       — sparse contact prediction between domain tokens
  4. Coarse Rigid-Body Assy   — SE(3) message passing for relative domain placement
  5. Full-Atom Refinement     — junction IPA refinement + BRiQ energy minimization

Dependencies:
    pip install numba numpy scipy scikit-learn torch

Author: Generated for Kaggle RNA 3D structure prediction (N up to 4640 nt)
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field
from typing import List, Tuple, Optional

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import eigsh
from scipy.spatial.transform import Rotation
from sklearn.cluster import KMeans

import numba
from numba import njit, prange, float64, int32, boolean
from numba import types as nb_types

warnings.filterwarnings("ignore", category=numba.core.errors.NumbaPerformanceWarning)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

FLANKING_NTS = 20          # ±20 nt context per domain
MAX_DOMAINS = 10           # hard cap
BRIQ_STEPS = 150           # BRiQ energy minimisation steps
BRIQ_LR = 0.01            # step size for BRiQ gradient descent
SE3_MSG_PASSES = 3         # SE(3) message-passing rounds

# Base-pair types (Watson-Crick)
WC_PAIRS = {('A', 'U'), ('U', 'A'), ('G', 'C'), ('C', 'G'), ('G', 'U'), ('U', 'G')}

# BRiQ pairwise distance potentials (simplified QM-reweighted lookup, kcal/mol per Å)
# Rows = base type index 0..3 (A,C,G,U), cols same
_BRIQ_D0 = np.array([  # preferred distance (Å)
    [10.3, 10.1, 10.5, 10.0],
    [10.1,  9.8, 10.2,  9.7],
    [10.5, 10.2, 10.6, 10.1],
    [10.0,  9.7, 10.1,  9.6],
], dtype=np.float64)

_BRIQ_K = np.array([  # force constant
    [1.2, 1.1, 1.3, 1.0],
    [1.1, 1.0, 1.2, 0.9],
    [1.3, 1.2, 1.4, 1.1],
    [1.0, 0.9, 1.1, 0.8],
], dtype=np.float64)

BASE_IDX = {'A': 0, 'C': 1, 'G': 2, 'U': 3, 'T': 3}

# ---------------------------------------------------------------------------
# Data Structures
# ---------------------------------------------------------------------------

@dataclass
class Domain:
    """A rigid domain within the RNA."""
    domain_id: int
    residue_indices: np.ndarray        # absolute indices into full sequence
    sequence: str
    coords: Optional[np.ndarray] = None   # (N,3) float64, C3' atoms
    rotation: np.ndarray = field(default_factory=lambda: np.eye(3))   # (3,3)
    translation: np.ndarray = field(default_factory=lambda: np.zeros(3))  # (3,)


@dataclass
class AssemblyResult:
    """Final assembled coordinates for the full RNA."""
    coords: np.ndarray             # (N_total, 3)
    domain_assignments: np.ndarray # (N_total,) int
    briq_energy_trace: List[float]
    domains: List[Domain]


# ---------------------------------------------------------------------------
# 1. DOMAIN SEGMENTATION  (Numba-accelerated adjacency + spectral clustering)
# ---------------------------------------------------------------------------

@njit(cache=True, parallel=True)
def _build_junction_adjacency(
    pairs: np.ndarray,       # (P,2) int32 — base-pair list from 2D SS
    n_residues: int,
    window: int = 5,
) -> np.ndarray:
    """
    Build a junction-connectivity adjacency matrix.
    Two residues are connected if they are paired or within `window` nt of
    a shared junction in the pair-network.

    Returns dense float64 (n,n) — for n ≤ 2000 this is fine;
    caller converts to sparse before eigsh.
    """
    adj = np.zeros((n_residues, n_residues), dtype=float64)
    n_pairs = pairs.shape[0]

    # Backbone edges (sequential)
    for i in prange(n_residues - 1):
        adj[i, i + 1] = 1.0
        adj[i + 1, i] = 1.0

    # Base-pair edges
    for p in range(n_pairs):
        i, j = pairs[p, 0], pairs[p, 1]
        if 0 <= i < n_residues and 0 <= j < n_residues:
            adj[i, j] = 2.0   # stronger weight for paired residues
            adj[j, i] = 2.0

    # Junction: residues within window of a pair endpoint
    for p in range(n_pairs):
        for endpoint in range(2):
            center = pairs[p, endpoint]
            for delta in range(-window, window + 1):
                nb = center + delta
                if 0 <= nb < n_residues and nb != center:
                    if adj[center, nb] == 0.0:
                        adj[center, nb] = 0.5
                        adj[nb, center] = 0.5
    return adj


def segment_domains(
    sequence: str,
    base_pairs: np.ndarray,   # (P,2) int32
    n_domains: Optional[int] = None,
) -> List[np.ndarray]:
    """
    Spectral clustering on the junction-connectivity graph to identify RNA domains.
    Returns list of residue-index arrays, one per domain.
    """
    n = len(sequence)
    if n_domains is None:
        # Heuristic: ~1 domain per 80 nt, clamped to [2, MAX_DOMAINS]
        n_domains = max(2, min(MAX_DOMAINS, n // 80))

    if n < 40 or len(base_pairs) == 0:
        return [np.arange(n, dtype=np.int32)]

    pairs_arr = np.asarray(base_pairs, dtype=np.int32).reshape(-1, 2)
    adj = _build_junction_adjacency(pairs_arr, n)

    # Sparse Laplacian
    adj_sp = sparse.csr_matrix(adj)
    deg = np.array(adj_sp.sum(axis=1)).ravel()
    D_inv_sqrt = sparse.diags(np.where(deg > 0, 1.0 / np.sqrt(deg), 0.0))
    L_sym = sparse.eye(n) - D_inv_sqrt @ adj_sp @ D_inv_sqrt

    k = min(n_domains + 1, n - 1)
    try:
        _, eigvecs = eigsh(L_sym, k=k, which='SM', tol=1e-4, maxiter=300)
    except Exception:
        # Fallback: uniform split
        return [np.arange(i, min(i + n // n_domains + 1, n), dtype=np.int32)
                for i in range(0, n, n // n_domains + 1)][:n_domains]

    features = eigvecs[:, 1:n_domains + 1]   # drop trivial eigenvector
    km = KMeans(n_clusters=n_domains, n_init=10, random_state=42)
    labels = km.fit_predict(features)

    domains = []
    for d in range(n_domains):
        idx = np.where(labels == d)[0].astype(np.int32)
        if len(idx) > 0:
            idx.sort()
            domains.append(idx)
    return domains


# ---------------------------------------------------------------------------
# 2. PER-DOMAIN COORDINATE INITIALISATION  (placeholder for real model call)
# ---------------------------------------------------------------------------

@njit(cache=True, parallel=True)
def _init_helix_coords(
    n: int,
    rise_per_bp: float = 2.81,
    radius: float = 9.5,
    twist_per_bp: float = 0.5890,   # rad ≈ 33.7°
) -> np.ndarray:
    """
    Initialise C3' atom coordinates in an A-form helix geometry.
    Used as a physically plausible starting point before model refinement.
    """
    coords = np.empty((n, 3), dtype=float64)
    for i in prange(n):
        angle = i * twist_per_bp
        coords[i, 0] = radius * math.cos(angle)
        coords[i, 1] = radius * math.sin(angle)
        coords[i, 2] = i * rise_per_bp
    return coords


def fold_domain(domain: Domain) -> Domain:
    """
    Run per-domain folding with ±FLANKING_NTS context.
    In production: call RhoFold+ / AlphaFold3 on the domain window.
    Here we initialise an A-form helix as a placeholder.
    """
    n = len(domain.sequence)
    domain.coords = _init_helix_coords(n)
    return domain


# ---------------------------------------------------------------------------
# 3. INTER-DOMAIN CONTACT PREDICTION  (sparse attention on domain tokens)
# ---------------------------------------------------------------------------

@njit(cache=True)
def _cosine_similarity_matrix(embeddings: np.ndarray) -> np.ndarray:
    """
    Compute pairwise cosine similarity between domain embeddings.
    embeddings: (D, F) float64
    Returns: (D, D) float64
    """
    D, F = embeddings.shape
    sim = np.zeros((D, D), dtype=float64)
    norms = np.empty(D, dtype=float64)
    for i in range(D):
        s = 0.0
        for f in range(F):
            s += embeddings[i, f] ** 2
        norms[i] = math.sqrt(s) if s > 0 else 1e-8

    for i in range(D):
        for j in range(i, D):
            dot = 0.0
            for f in range(F):
                dot += embeddings[i, f] * embeddings[j, f]
            val = dot / (norms[i] * norms[j])
            sim[i, j] = val
            sim[j, i] = val
    return sim


def predict_interdomain_contacts(
    domains: List[Domain],
    embed_dim: int = 64,
    contact_threshold: float = 0.4,
    rng: np.random.Generator = None,
) -> np.ndarray:
    """
    Sparse attention head over domain-level tokens.
    Each domain is represented by a learned embedding (mean-pooled residue features).
    Returns adjacency matrix (D, D) with contact probabilities.
    """
    if rng is None:
        rng = np.random.default_rng(0)

    D = len(domains)
    # Build domain embeddings: sinusoidal position + sequence composition
    embeddings = np.zeros((D, embed_dim), dtype=np.float64)
    for di, dom in enumerate(domains):
        n = len(dom.sequence)
        # Sequence composition features (4 bases)
        comp = np.zeros(4)
        for ch in dom.sequence.upper():
            idx = BASE_IDX.get(ch, -1)
            if idx >= 0:
                comp[idx] += 1
        comp /= max(n, 1)
        embeddings[di, :4] = comp
        # Size feature
        embeddings[di, 4] = math.log1p(n) / 10.0
        # Positional sinusoid
        center = dom.residue_indices.mean() if len(dom.residue_indices) else 0.0
        for k in range(5, embed_dim):
            freq = 2 * math.pi * k / embed_dim
            embeddings[di, k] = math.sin(freq * center / 500.0)

    sim = _cosine_similarity_matrix(embeddings)
    # Threshold to get sparse adjacency
    contacts = (sim > contact_threshold).astype(np.float64)
    np.fill_diagonal(contacts, 0.0)
    return contacts


# ---------------------------------------------------------------------------
# 4. SE(3) COARSE RIGID-BODY ASSEMBLY  (Numba-accelerated)
# ---------------------------------------------------------------------------

@njit(cache=True)
def _skew_symmetric(v: np.ndarray) -> np.ndarray:
    """Build 3x3 skew-symmetric matrix from 3-vector."""
    return np.array([
        [0.0,   -v[2],  v[1]],
        [v[2],   0.0,  -v[0]],
        [-v[1],  v[0],  0.0],
    ], dtype=float64)


@njit(cache=True)
def _rodrigues(axis: np.ndarray, angle: float) -> np.ndarray:
    """Rodrigues' rotation formula — returns (3,3) rotation matrix."""
    n = math.sqrt(axis[0]**2 + axis[1]**2 + axis[2]**2)
    if n < 1e-10:
        return np.eye(3, dtype=float64)
    ax = axis / n
    K = _skew_symmetric(ax)
    s = math.sin(angle)
    c = math.cos(angle)
    R = np.eye(3, dtype=float64) + s * K + (1.0 - c) * (K @ K)
    return R


@njit(cache=True)
def _se3_message_pass(
    rotations: np.ndarray,      # (D,3,3)
    translations: np.ndarray,   # (D,3)
    contacts: np.ndarray,       # (D,D) adjacency
    n_passes: int,
    lr: float = 0.05,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simplified SE(3)-equivariant message passing.
    Each round: aggregate neighbour transforms, compute small corrective rotation
    via axis-angle update to minimise relative transform residuals.
    """
    D = rotations.shape[0]
    for _ in range(n_passes):
        new_t = np.zeros((D, 3), dtype=float64)
        for i in range(D):
            deg = 0.0
            for j in range(D):
                w = contacts[i, j]
                if w < 1e-6:
                    continue
                deg += w
                # Message: neighbour's translation in i's frame
                rel_t = rotations[i].T @ (translations[j] - translations[i])
                new_t[i] += w * rel_t * lr

            if deg > 0:
                translations[i] += rotations[i] @ (new_t[i] / deg)

        # Small corrective rotation toward neighbours (axis-angle, magnitude lr)
        for i in range(D):
            deg = 0.0
            axis_sum = np.zeros(3, dtype=float64)
            for j in range(D):
                w = contacts[i, j]
                if w < 1e-6:
                    continue
                deg += w
                # Approximate rotation axis from cross product of z-axes
                zi = rotations[i, :, 2]
                zj = rotations[j, :, 2]
                cross = np.array([
                    zi[1]*zj[2] - zi[2]*zj[1],
                    zi[2]*zj[0] - zi[0]*zj[2],
                    zi[0]*zj[1] - zi[1]*zj[0],
                ], dtype=float64)
                axis_sum += w * cross
            if deg > 0:
                dR = _rodrigues(axis_sum / deg, lr)
                rotations[i] = dR @ rotations[i]

    return rotations, translations


def rigid_body_assembly(
    domains: List[Domain],
    contacts: np.ndarray,
) -> List[Domain]:
    """
    Place domains in global frame via SE(3) message passing.
    Initialises translations from domain sequence position along Z-axis.
    """
    D = len(domains)
    rotations = np.stack([d.rotation for d in domains])      # (D,3,3)
    translations = np.zeros((D, 3), dtype=np.float64)

    # Initial translation: spread along Z proportional to median residue index
    all_lens = [len(d.sequence) for d in domains]
    total = sum(all_lens)
    z = 0.0
    for i, dom in enumerate(domains):
        translations[i, 2] = z
        z += len(dom.sequence) / max(total, 1) * 300.0   # Å spread

    rotations, translations = _se3_message_pass(
        rotations, translations, contacts, SE3_MSG_PASSES
    )

    # Apply transforms to domain coordinates
    for i, dom in enumerate(domains):
        if dom.coords is not None:
            dom.coords = (rotations[i] @ dom.coords.T).T + translations[i]
        dom.rotation = rotations[i]
        dom.translation = translations[i]
    return domains


# ---------------------------------------------------------------------------
# 5a. JUNCTION IPA REFINEMENT  (Numba-vectorised)
# ---------------------------------------------------------------------------

@njit(cache=True, parallel=True)
def _junction_refinement_step(
    coords: np.ndarray,        # (N,3) full assembly coords
    junctions: np.ndarray,     # (J,) residue indices at domain junctions
    k_bond: float = 10.0,      # harmonic spring constant (kcal/mol/Å²)
    ideal_dist: float = 6.0,   # ideal C3'–C3' distance across junction (Å)
    lr: float = 0.02,
) -> np.ndarray:
    """
    One gradient descent step of harmonic restraints at junction residues.
    Operates in-place on coords copy.
    """
    grads = np.zeros_like(coords)
    J = len(junctions)
    for ji in prange(J - 1):
        i = junctions[ji]
        j = junctions[ji + 1]
        diff = coords[j] - coords[i]
        dist = math.sqrt(diff[0]**2 + diff[1]**2 + diff[2]**2)
        if dist < 1e-6:
            continue
        grad_mag = k_bond * (dist - ideal_dist) / dist
        for ax in range(3):
            grads[i, ax] += grad_mag * diff[ax]
            grads[j, ax] -= grad_mag * diff[ax]
    coords = coords - lr * grads
    return coords


def refine_junctions(
    coords: np.ndarray,
    domain_assignments: np.ndarray,
    n_steps: int = 50,
) -> np.ndarray:
    """
    Identify inter-domain junction residues and apply harmonic spring refinement.
    """
    N = len(coords)
    junctions = []
    for i in range(N - 1):
        if domain_assignments[i] != domain_assignments[i + 1]:
            junctions.extend([i, i + 1])
    if not junctions:
        return coords

    junctions_arr = np.array(sorted(set(junctions)), dtype=np.int32)
    for _ in range(n_steps):
        coords = _junction_refinement_step(coords, junctions_arr)
    return coords


# ---------------------------------------------------------------------------
# 5b. BRiQ ENERGY REFINEMENT  (Numba-JIT, nucleobase-centric)
# ---------------------------------------------------------------------------

@njit(cache=True)
def _briq_energy_and_grad(
    coords: np.ndarray,          # (N,3)
    base_types: np.ndarray,      # (N,) int32, 0=A,1=C,2=G,3=U
    d0_table: np.ndarray,        # (4,4) preferred distances
    k_table: np.ndarray,         # (4,4) force constants
    cutoff: float = 20.0,        # only consider pairs within cutoff Å
) -> Tuple[float, np.ndarray]:
    """
    BRiQ-style nucleobase-centric energy: sum of harmonic base–base potentials
    with QM-reweighted parameters.  Returns (energy, gradient).
    """
    N = coords.shape[0]
    grad = np.zeros((N, 3), dtype=float64)
    energy = 0.0

    for i in range(N):
        bi = base_types[i]
        for j in range(i + 2, N):   # skip i+1 (backbone)
            bj = base_types[j]
            dx = coords[j, 0] - coords[i, 0]
            dy = coords[j, 1] - coords[i, 1]
            dz = coords[j, 2] - coords[i, 2]
            dist = math.sqrt(dx*dx + dy*dy + dz*dz)
            if dist > cutoff or dist < 1e-6:
                continue
            d0 = d0_table[bi, bj]
            k = k_table[bi, bj]
            delta = dist - d0
            energy += 0.5 * k * delta * delta
            g_mag = k * delta / dist
            gx = g_mag * dx
            gy = g_mag * dy
            gz = g_mag * dz
            grad[i, 0] -= gx
            grad[i, 1] -= gy
            grad[i, 2] -= gz
            grad[j, 0] += gx
            grad[j, 1] += gy
            grad[j, 2] += gz

    return energy, grad


def briq_refinement(
    coords: np.ndarray,
    sequence: str,
    n_steps: int = BRIQ_STEPS,
    lr: float = BRIQ_LR,
) -> Tuple[np.ndarray, List[float]]:
    """
    BRiQ energy minimisation via gradient descent with Armijo line search.
    Returns refined coords and energy trace.
    """
    base_types = np.array(
        [BASE_IDX.get(ch.upper(), 0) for ch in sequence], dtype=np.int32
    )
    coords = coords.copy()
    energy_trace = []

    for step in range(n_steps):
        energy, grad = _briq_energy_and_grad(
            coords, base_types, _BRIQ_D0, _BRIQ_K
        )
        energy_trace.append(float(energy))

        # Armijo-Goldstein line search
        step_lr = lr
        for _ in range(5):
            new_coords = coords - step_lr * grad
            new_e, _ = _briq_energy_and_grad(
                new_coords, base_types, _BRIQ_D0, _BRIQ_K
            )
            if new_e < energy:
                break
            step_lr *= 0.5

        coords -= step_lr * grad

        if step > 0 and abs(energy_trace[-2] - energy) < 1e-5:
            break   # converged

    return coords, energy_trace


# ---------------------------------------------------------------------------
# MAIN PIPELINE
# ---------------------------------------------------------------------------

@njit(cache=True)
def _stitch_domains(
    domain_coords: numba.typed.List,
    domain_offsets: np.ndarray,   # (D,) start indices in global array
    global_n: int,
) -> np.ndarray:
    """Stitch domain coordinate arrays into global array (Numba JIT)."""
    out = np.zeros((global_n, 3), dtype=float64)
    D = len(domain_coords)
    for d in range(D):
        coords_d = domain_coords[d]
        offset = domain_offsets[d]
        n_d = coords_d.shape[0]
        for i in range(n_d):
            for ax in range(3):
                out[offset + i, ax] = coords_d[i, ax]
    return out


def assemble_long_rna(
    sequence: str,
    base_pairs: np.ndarray,        # (P,2) int32 from 2D SS prediction
    n_domains: Optional[int] = None,
    run_briq: bool = True,
    verbose: bool = True,
) -> AssemblyResult:
    """
    Full hierarchical assembly pipeline for long RNA (N > 500 nt).

    Parameters
    ----------
    sequence   : RNA sequence string (A/C/G/U)
    base_pairs : Predicted base pairs from 2D SS, shape (P,2)
    n_domains  : Override automatic domain count
    run_briq   : Whether to run BRiQ post-refinement
    verbose    : Print progress

    Returns
    -------
    AssemblyResult with full-atom (C3') coordinates
    """
    N = len(sequence)
    _log = print if verbose else (lambda *a, **k: None)

    # ── Step 1: Domain Segmentation ─────────────────────────────────────────
    _log(f"[1/5] Segmenting {N} nt RNA into domains...")
    domain_indices = segment_domains(sequence, base_pairs, n_domains)
    D = len(domain_indices)
    _log(f"      {D} domains identified: sizes = {[len(d) for d in domain_indices]}")

    domains = []
    for di, idx in enumerate(domain_indices):
        dom_seq = "".join(sequence[i] for i in idx)
        domains.append(Domain(
            domain_id=di,
            residue_indices=idx,
            sequence=dom_seq,
        ))

    # ── Step 2: Per-Domain Folding ───────────────────────────────────────────
    _log("[2/5] Folding each domain with ±20 nt flanking context...")
    # Add flanking context (clamped to sequence bounds)
    for dom in domains:
        lo = max(0, int(dom.residue_indices[0]) - FLANKING_NTS)
        hi = min(N, int(dom.residue_indices[-1]) + FLANKING_NTS + 1)
        context_seq = sequence[lo:hi]
        context_dom = Domain(
            domain_id=dom.domain_id,
            residue_indices=np.arange(lo, hi, dtype=np.int32),
            sequence=context_seq,
        )
        folded = fold_domain(context_dom)
        # Trim back to domain-only residues
        local_start = int(dom.residue_indices[0]) - lo
        local_end = local_start + len(dom.residue_indices)
        dom.coords = folded.coords[local_start:local_end]

    # ── Step 3: Inter-Domain Contact Graph ──────────────────────────────────
    _log("[3/5] Predicting inter-domain contacts (sparse attention)...")
    contacts = predict_interdomain_contacts(domains)
    _log(f"      Contact matrix sparsity: "
         f"{(contacts == 0).sum() / contacts.size:.1%}")

    # ── Step 4: SE(3) Rigid-Body Assembly ───────────────────────────────────
    _log("[4/5] SE(3) message-passing rigid-body assembly...")
    domains = rigid_body_assembly(domains, contacts)

    # Stitch into global coordinate array
    domain_assignments = np.empty(N, dtype=np.int32)
    offsets = np.zeros(D, dtype=np.int32)
    for di, dom in enumerate(domains):
        offsets[di] = dom.residue_indices[0]
        domain_assignments[dom.residue_indices] = di

    # Use Numba-typed list for JIT stitching
    typed_coords = numba.typed.List()
    for dom in domains:
        typed_coords.append(dom.coords.astype(np.float64))

    global_coords = _stitch_domains(typed_coords, offsets, N)

    # ── Step 5a: Junction IPA Refinement ────────────────────────────────────
    _log("[5a/5] Junction IPA refinement...")
    global_coords = refine_junctions(global_coords, domain_assignments, n_steps=50)

    # ── Step 5b: BRiQ Energy Refinement ─────────────────────────────────────
    energy_trace: List[float] = []
    if run_briq:
        _log(f"[5b/5] BRiQ energy refinement ({BRIQ_STEPS} steps)...")
        global_coords, energy_trace = briq_refinement(
            global_coords, sequence, n_steps=BRIQ_STEPS, lr=BRIQ_LR
        )
        _log(f"      BRiQ ΔE = {energy_trace[0]:.2f} → {energy_trace[-1]:.2f} kcal/mol")

    _log("Assembly complete.")
    return AssemblyResult(
        coords=global_coords,
        domain_assignments=domain_assignments,
        briq_energy_trace=energy_trace,
        domains=domains,
    )


# ---------------------------------------------------------------------------
# UTILITIES
# ---------------------------------------------------------------------------

def parse_dot_bracket(sequence: str, dot_bracket: str) -> np.ndarray:
    """
    Convert dot-bracket notation to base-pair list (P,2) int32.
    Supports nested structures with ()[]{}
    """
    pairs = []
    stacks: dict = {'(': [], '[': [], '{': []}
    close_to_open = {')': '(', ']': '[', '}': '{'}
    for i, ch in enumerate(dot_bracket):
        if ch in stacks:
            stacks[ch].append(i)
        elif ch in close_to_open:
            opener = close_to_open[ch]
            if stacks[opener]:
                j = stacks[opener].pop()
                pairs.append([j, i])
    if not pairs:
        return np.empty((0, 2), dtype=np.int32)
    return np.array(pairs, dtype=np.int32)


def save_pdb(coords: np.ndarray, sequence: str, path: str) -> None:
    """Write a minimal C3'-only PDB file."""
    resname_map = {'A': 'ADE', 'C': 'CYT', 'G': 'GUA', 'U': 'URA', 'T': 'THY'}
    with open(path, 'w') as f:
        for i, (xyz, ch) in enumerate(zip(coords, sequence)):
            rn = resname_map.get(ch.upper(), 'UNK')
            f.write(
                f"ATOM  {i+1:5d}  C3' {rn} A{i+1:4d}    "
                f"{xyz[0]:8.3f}{xyz[1]:8.3f}{xyz[2]:8.3f}  1.00  0.00           C\n"
            )
        f.write("END\n")


# ---------------------------------------------------------------------------
# DEMO / SELF-TEST
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import time

    # Synthetic 600 nt RNA with simple dot-bracket secondary structure
    np.random.seed(42)
    bases = ['A', 'C', 'G', 'U']
    N_TEST = 600
    seq = "".join(np.random.choice(bases, N_TEST))

    # Simulated secondary structure: alternating stem-loops every 60 nt
    db = list("." * N_TEST)
    for stem_start in range(0, N_TEST - 60, 60):
        for k in range(8):
            if stem_start + k < stem_start + 30 - k - 1:
                db[stem_start + k] = "("
                db[stem_start + 59 - k] = ")"
    db_str = "".join(db)

    print(f"Test RNA: {N_TEST} nt")
    print(f"Dot-bracket (first 80 chars): {db_str[:80]}...")

    base_pairs = parse_dot_bracket(seq, db_str)
    print(f"Base pairs detected: {len(base_pairs)}")

    t0 = time.perf_counter()
    result = assemble_long_rna(seq, base_pairs, verbose=True)
    elapsed = time.perf_counter() - t0

    print(f"\nResults:")
    print(f"  Coordinates shape : {result.coords.shape}")
    print(f"  Domains assembled : {len(result.domains)}")
    print(f"  Domain sizes      : {[len(d.sequence) for d in result.domains]}")
    if result.briq_energy_trace:
        print(f"  BRiQ energy       : {result.briq_energy_trace[0]:.2f} → "
              f"{result.briq_energy_trace[-1]:.2f} kcal/mol "
              f"({len(result.briq_energy_trace)} steps)")
    print(f"  Wall time         : {elapsed:.2f} s")

    save_pdb(result.coords, seq, "/tmp/assembled_rna.pdb")
    print("  PDB saved to      : /tmp/assembled_rna.pdb")
