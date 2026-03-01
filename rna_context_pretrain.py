"""
rna_context_pretrain.py
=======================
Context-Aware Pretraining for Ligand/Complex RNAs
---------------------------------------------------
Implements Priority-7 of the RNA folding improvement roadmap:

  • Numba JIT-compiled kernels for pairwise distance / contact-map computation
  • Vectorised partner-masking for RNA–protein and RNA–ligand PDB complexes
  • Context-type token embedding conditioning (riboswitch, ribozyme, tRNA, …)
  • Lightweight sequence-motif classifier for inference-time context injection
  • Metal-binding auxiliary head with Mg²⁺ / ion restraint injection
  • Minibatch pretraining loop skeleton (framework-agnostic numpy arrays)

Requirements
------------
    pip install numba numpy scipy

Optional (for real PDB parsing):
    pip install biopython
"""

from __future__ import annotations

import math
import time
import warnings
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Dict, List, Optional, Tuple

import numpy as np
from numba import njit, prange, float32, int32, boolean
from numba import typed as nb_typed
from scipy.spatial.distance import cdist

# ---------------------------------------------------------------------------
# 0.  CONSTANTS & ENUMERATIONS
# ---------------------------------------------------------------------------

RNA_NUCLEOTIDES = ("A", "U", "G", "C", "N")
PROTEIN_RESIDUES = tuple("ACDEFGHIKLMNPQRSTVWY")
METAL_IONS = ("MG", "ZN", "CA", "FE", "MN", "CO", "NI", "CU", "K", "NA")

CONTACT_THRESHOLD_A = 8.0   # Å – heavy-atom contact cutoff
METAL_CONTACT_A    = 4.0   # Å – coordination-sphere cutoff


class RNAContextType(IntEnum):
    """Biological context types used as conditioning tokens."""
    UNKNOWN        = 0
    RIBOSWITCH     = 1
    RIBOZYME       = 2
    SPLICEOSOMAL   = 3
    APTAMER        = 4
    RIBOSOMAL      = 5
    TRNA           = 6
    SNRNA          = SNORNA = 7
    MRNA_ELEMENT   = 8
    G_QUADRUPLEX   = 9
    NAKED_RNA      = 10   # no complex partner – baseline

    @classmethod
    def num_types(cls) -> int:
        return max(c.value for c in cls) + 1


# ---------------------------------------------------------------------------
# 1.  NUMBA JIT KERNELS
# ---------------------------------------------------------------------------

@njit(parallel=True, fastmath=True, cache=True)
def compute_pairwise_distances(
    coords_a: np.ndarray,   # (N, 3) float32
    coords_b: np.ndarray,   # (M, 3) float32
    out: np.ndarray,        # (N, M) float32  – pre-allocated
) -> None:
    """
    L2 pairwise distance matrix between two atom-coordinate arrays.
    Parallelised over rows of coords_a with prange.
    """
    N = coords_a.shape[0]
    M = coords_b.shape[0]
    for i in prange(N):
        ax = coords_a[i, 0]
        ay = coords_a[i, 1]
        az = coords_a[i, 2]
        for j in range(M):
            dx = ax - coords_b[j, 0]
            dy = ay - coords_b[j, 1]
            dz = az - coords_b[j, 2]
            out[i, j] = math.sqrt(dx * dx + dy * dy + dz * dz)


@njit(parallel=True, fastmath=True, cache=True)
def build_contact_map(
    dist_mat: np.ndarray,    # (N, M) float32
    threshold: float,
    out: np.ndarray,         # (N, M) int32  – pre-allocated
) -> None:
    """Binary contact map from a precomputed distance matrix."""
    N = dist_mat.shape[0]
    M = dist_mat.shape[1]
    for i in prange(N):
        for j in range(M):
            out[i, j] = 1 if dist_mat[i, j] <= threshold else 0


@njit(parallel=True, fastmath=True, cache=True)
def compute_rna_self_contacts(
    rna_coords: np.ndarray,  # (L, 3) float32
    threshold: float,
    min_seq_sep: int,
    out: np.ndarray,         # (L, L) int32
) -> None:
    """
    Self-contact map for the RNA chain with a minimum sequence-separation
    filter (avoids trivial consecutive-residue contacts).
    """
    L = rna_coords.shape[0]
    for i in prange(L):
        for j in range(L):
            if abs(i - j) < min_seq_sep:
                out[i, j] = 0
                continue
            dx = rna_coords[i, 0] - rna_coords[j, 0]
            dy = rna_coords[i, 1] - rna_coords[j, 1]
            dz = rna_coords[i, 2] - rna_coords[j, 2]
            d = math.sqrt(dx * dx + dy * dy + dz * dz)
            out[i, j] = 1 if d <= threshold else 0


@njit(fastmath=True, cache=True)
def vectorised_partner_mask(
    rna_atom_indices: np.ndarray,    # (N_rna,) int32 – global atom indices
    partner_atom_indices: np.ndarray,# (N_partner,) int32
    all_atom_coords: np.ndarray,     # (N_total, 3) float32
    threshold: float,
    out_mask: np.ndarray,            # (N_rna,) boolean
) -> None:
    """
    For each RNA atom, set True if ANY partner atom is within threshold.
    Used to create 'context-contact' boolean mask for conditioning.
    """
    N_r = rna_atom_indices.shape[0]
    N_p = partner_atom_indices.shape[0]
    for i in range(N_r):
        ri = rna_atom_indices[i]
        ax = all_atom_coords[ri, 0]
        ay = all_atom_coords[ri, 1]
        az = all_atom_coords[ri, 2]
        found = False
        for j in range(N_p):
            pj = partner_atom_indices[j]
            dx = ax - all_atom_coords[pj, 0]
            dy = ay - all_atom_coords[pj, 1]
            dz = az - all_atom_coords[pj, 2]
            if math.sqrt(dx * dx + dy * dy + dz * dz) <= threshold:
                found = True
                break
        out_mask[i] = found


@njit(parallel=True, fastmath=True, cache=True)
def compute_metal_binding_features(
    rna_coords: np.ndarray,       # (L, 3) float32  – per-residue C4' coords
    metal_coords: np.ndarray,     # (K, 3) float32  – metal ion positions
    radii: np.ndarray,            # (K,) float32    – ion-specific radii
    out_prob: np.ndarray,         # (L,) float32    – binding "score"
) -> None:
    """
    Gaussian proximity score: for each RNA residue, sum exp(-d²/2r²) over
    all metal ions.  Output is used as soft metal-binding probability.
    """
    L = rna_coords.shape[0]
    K = metal_coords.shape[0]
    for i in prange(L):
        score = float32(0.0)
        for k in range(K):
            dx = rna_coords[i, 0] - metal_coords[k, 0]
            dy = rna_coords[i, 1] - metal_coords[k, 1]
            dz = rna_coords[i, 2] - metal_coords[k, 2]
            d2 = dx * dx + dy * dy + dz * dz
            r2 = radii[k] * radii[k]
            score += math.exp(-d2 / (2.0 * r2))
        out_prob[i] = score


@njit(parallel=True, fastmath=True, cache=True)
def apply_metal_distance_restraints(
    dist_mat: np.ndarray,          # (L, L) float32  – RNA self-distance matrix
    metal_binding_mask: np.ndarray,# (L,) boolean    – residues near metal
    restraint_strength: float,
    out_restraint: np.ndarray,     # (L, L) float32  – additive restraint bias
) -> None:
    """
    Adds distance-restraint bias between pairs of metal-binding residues,
    encouraging the model to enforce compact coordination geometry.
    """
    L = dist_mat.shape[0]
    for i in prange(L):
        for j in range(L):
            if metal_binding_mask[i] and metal_binding_mask[j] and i != j:
                # Pull metal-binding residues together proportionally to distance
                out_restraint[i, j] = restraint_strength / (1.0 + dist_mat[i, j])
            else:
                out_restraint[i, j] = float32(0.0)


# ---------------------------------------------------------------------------
# 2.  DATA STRUCTURES
# ---------------------------------------------------------------------------

@dataclass
class PDBComplex:
    """Holds parsed coordinates for one PDB entry with multiple chains."""
    pdb_id:            str
    rna_sequence:      str                        # single-letter codes
    rna_coords:        np.ndarray                 # (L, 3) float32  C4'
    rna_atom_indices:  np.ndarray                 # (N_rna,) int32
    partner_indices:   np.ndarray                 # (N_partner,) int32
    all_atom_coords:   np.ndarray                 # (N_total, 3) float32
    metal_coords:      Optional[np.ndarray]       # (K, 3) or None
    metal_types:       Optional[List[str]]        # e.g. ['MG','MG','ZN']
    partner_type:      str                        # 'protein'|'ligand'|'ion'|'naked'
    context_type:      RNAContextType = RNAContextType.UNKNOWN
    chain_id:          str = "A"


@dataclass
class ContextFeatures:
    """Derived features for one RNA complex, ready for model consumption."""
    pdb_id:              str
    rna_length:          int
    context_type_id:     int                       # RNAContextType int value
    contact_mask:        np.ndarray                # (L,) bool  – residues near partner
    self_contact_map:    np.ndarray                # (L, L) int32
    metal_binding_prob:  np.ndarray                # (L,) float32
    metal_restraints:    np.ndarray                # (L, L) float32
    partner_dist_mat:    Optional[np.ndarray]      # (L, N_partner_residues) float32
    sequence_one_hot:    np.ndarray                # (L, 5) float32


# ---------------------------------------------------------------------------
# 3.  SEQUENCE UTILITIES  (vectorised numpy)
# ---------------------------------------------------------------------------

_NUC_TO_IDX: Dict[str, int] = {n: i for i, n in enumerate(RNA_NUCLEOTIDES)}


def sequence_to_one_hot(seq: str) -> np.ndarray:
    """Convert RNA sequence string → (L, 5) float32 one-hot array."""
    L = len(seq)
    out = np.zeros((L, len(RNA_NUCLEOTIDES)), dtype=np.float32)
    for i, ch in enumerate(seq.upper()):
        idx = _NUC_TO_IDX.get(ch, _NUC_TO_IDX["N"])
        out[i, idx] = 1.0
    return out


def one_hot_to_sequence(oh: np.ndarray) -> str:
    indices = np.argmax(oh, axis=-1)
    return "".join(RNA_NUCLEOTIDES[i] for i in indices)


# ---------------------------------------------------------------------------
# 4.  MOTIF-BASED SEQUENCE CLASSIFIER
# ---------------------------------------------------------------------------

# Canonical sequence motifs (simplified consensus) for context classification
_CONTEXT_MOTIFS: Dict[RNAContextType, List[str]] = {
    RNAContextType.RIBOSWITCH:   ["GGCGUA", "GCAUAC", "CCUAACG", "UGAGAG"],
    RNAContextType.RIBOZYME:     ["CUGAUGA", "GGCGAAAC", "GAAA", "AACUGGUG"],
    RNAContextType.TRNA:         ["GCGCCUA", "UUCGAAU", "GCUCCA", "TΨCG"],
    RNAContextType.RIBOSOMAL:    ["AAGACG", "GAUAAG", "GGUGCG", "CACGCC"],
    RNAContextType.SPLICEOSOMAL: ["GUAAGU", "CAUCAG", "ACUAAC", "UACUAAC"],
    RNAContextType.APTAMER:      ["UGGGGG", "GGGCGG", "GGGGUG"],
    RNAContextType.G_QUADRUPLEX: ["GGGG", "GGGCG", "TGGGG"],
}


def classify_rna_context(
    sequence: str,
    min_matches: int = 1,
) -> RNAContextType:
    """
    Lightweight motif-scan classifier.
    Returns the RNAContextType with the most motif hits (>= min_matches).
    Falls back to NAKED_RNA if nothing matches.

    In production: replace / augment with a trained k-mer / CNN classifier.
    """
    seq_upper = sequence.upper().replace("T", "U")
    scores: Dict[RNAContextType, int] = {}
    for ctx, motifs in _CONTEXT_MOTIFS.items():
        count = sum(seq_upper.count(m.upper().replace("T", "U")) for m in motifs)
        scores[ctx] = count

    best_ctx  = max(scores, key=lambda k: scores[k])
    best_score = scores[best_ctx]
    return best_ctx if best_score >= min_matches else RNAContextType.NAKED_RNA


# ---------------------------------------------------------------------------
# 5.  CONTEXT EMBEDDING (numpy – plug into any framework)
# ---------------------------------------------------------------------------

class ContextTypeEmbedding:
    """
    Learnable context-type embedding table.
    Stored as a (num_types, embed_dim) float32 array.
    In PyTorch / JAX you would wrap this as nn.Embedding / flax.linen.Embed.
    """

    def __init__(self, embed_dim: int = 64, seed: int = 42) -> None:
        rng = np.random.default_rng(seed)
        n = RNAContextType.num_types()
        # Xavier uniform initialisation
        limit = math.sqrt(6.0 / (n + embed_dim))
        self.weight = rng.uniform(-limit, limit, (n, embed_dim)).astype(np.float32)
        self.embed_dim = embed_dim

    def __call__(self, context_ids: np.ndarray) -> np.ndarray:
        """
        context_ids : (B,) int32
        returns     : (B, embed_dim) float32
        """
        return self.weight[context_ids]

    def inject_into_single_repr(
        self,
        single_repr: np.ndarray,  # (B, L, d_model) float32
        context_ids: np.ndarray,  # (B,) int32
        projection: Optional[np.ndarray] = None,  # (embed_dim, d_model) – linear proj
    ) -> np.ndarray:
        """
        Adds the context embedding (broadcast over L) to the single representation.
        If projection is None and embed_dim != d_model a ValueError is raised.
        """
        emb = self(context_ids)                             # (B, embed_dim)
        if projection is not None:
            emb = emb @ projection                          # (B, d_model)
        elif emb.shape[-1] != single_repr.shape[-1]:
            raise ValueError(
                f"embed_dim={emb.shape[-1]} != d_model={single_repr.shape[-1]}. "
                "Provide a projection matrix."
            )
        # Broadcast over sequence length
        return single_repr + emb[:, np.newaxis, :]          # (B, L, d_model)


# ---------------------------------------------------------------------------
# 6.  METAL-BINDING AUXILIARY HEAD (numpy forward pass)
# ---------------------------------------------------------------------------

def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x.astype(np.float64))).astype(np.float32)


class MetalBindingHead:
    """
    Two-layer MLP auxiliary head: single_repr → metal-binding probability per residue.
    Weights are stored as numpy arrays; framework wrappers are trivial to add.
    """

    def __init__(self, d_model: int, hidden_dim: int = 128, seed: int = 7) -> None:
        rng = np.random.default_rng(seed)

        def xavier(fan_in, fan_out):
            lim = math.sqrt(6.0 / (fan_in + fan_out))
            return rng.uniform(-lim, lim, (fan_in, fan_out)).astype(np.float32)

        self.W1 = xavier(d_model, hidden_dim)
        self.b1 = np.zeros(hidden_dim, dtype=np.float32)
        self.W2 = xavier(hidden_dim, 1)
        self.b2 = np.zeros(1, dtype=np.float32)

    def forward(self, single_repr: np.ndarray) -> np.ndarray:
        """
        single_repr : (B, L, d_model) float32
        returns     : (B, L) float32  – probability in [0, 1]
        """
        h = np.maximum(0.0, single_repr @ self.W1 + self.b1)   # ReLU  (B,L,hidden)
        logits = (h @ self.W2 + self.b2).squeeze(-1)             # (B, L)
        return _sigmoid(logits)

    def binary_cross_entropy_loss(
        self,
        pred_prob: np.ndarray,   # (B, L) float32
        targets: np.ndarray,     # (B, L) float32  – 0/1 labels
        eps: float = 1e-7,
    ) -> float:
        p = np.clip(pred_prob, eps, 1.0 - eps)
        return float(-np.mean(targets * np.log(p) + (1.0 - targets) * np.log(1.0 - p)))


# ---------------------------------------------------------------------------
# 7.  FEATURE EXTRACTION PIPELINE
# ---------------------------------------------------------------------------

_ION_RADII: Dict[str, float] = {
    "MG": 3.5, "ZN": 3.0, "CA": 4.5, "FE": 3.5,
    "MN": 3.5, "CO": 3.2, "NI": 3.2, "CU": 3.0,
    "K":  4.0, "NA": 4.0,
}


def extract_context_features(
    complex_: PDBComplex,
    contact_threshold: float = CONTACT_THRESHOLD_A,
    metal_threshold:   float = METAL_CONTACT_A,
    metal_restraint_strength: float = 0.5,
    min_seq_sep: int = 4,
) -> ContextFeatures:
    """
    Full feature-extraction pipeline for one PDB complex.

    Steps
    -----
    1. Partner-contact mask (JIT kernel)
    2. RNA self-contact map (JIT kernel)
    3. Metal-binding probability (JIT kernel)
    4. Metal distance restraints (JIT kernel)
    5. Partner distance matrix
    6. Sequence one-hot
    """
    L = complex_.rna_coords.shape[0]

    # -- 1. Partner contact mask -----------------------------------------
    contact_mask = np.zeros(len(complex_.rna_atom_indices), dtype=np.bool_)
    if complex_.partner_indices.size > 0:
        vectorised_partner_mask(
            complex_.rna_atom_indices.astype(np.int32),
            complex_.partner_indices.astype(np.int32),
            complex_.all_atom_coords.astype(np.float32),
            contact_threshold,
            contact_mask,
        )
    # Aggregate to residue level (any atom in residue contacted)
    rna_len = L
    residue_contact = np.zeros(rna_len, dtype=np.bool_)
    atoms_per_residue = max(1, len(complex_.rna_atom_indices) // rna_len)
    for r in range(rna_len):
        start = r * atoms_per_residue
        end   = min(start + atoms_per_residue, len(contact_mask))
        if np.any(contact_mask[start:end]):
            residue_contact[r] = True

    # -- 2. RNA self-contact map ------------------------------------------
    self_contact = np.zeros((L, L), dtype=np.int32)
    compute_rna_self_contacts(
        complex_.rna_coords.astype(np.float32),
        contact_threshold,
        min_seq_sep,
        self_contact,
    )

    # -- 3. Metal binding probability ------------------------------------
    metal_prob = np.zeros(L, dtype=np.float32)
    if complex_.metal_coords is not None and len(complex_.metal_coords) > 0:
        metal_np = complex_.metal_coords.astype(np.float32)
        radii    = np.array(
            [_ION_RADII.get(t, 3.5) for t in (complex_.metal_types or [])],
            dtype=np.float32,
        )
        compute_metal_binding_features(
            complex_.rna_coords.astype(np.float32),
            metal_np,
            radii,
            metal_prob,
        )
        # Sigmoid normalise
        metal_prob = _sigmoid(metal_prob - metal_prob.mean())

    # -- 4. Metal distance restraints ------------------------------------
    rna_dist_mat = np.zeros((L, L), dtype=np.float32)
    compute_pairwise_distances(
        complex_.rna_coords.astype(np.float32),
        complex_.rna_coords.astype(np.float32),
        rna_dist_mat,
    )
    metal_binding_mask = (metal_prob > 0.5)
    metal_restraints   = np.zeros((L, L), dtype=np.float32)
    apply_metal_distance_restraints(
        rna_dist_mat, metal_binding_mask, metal_restraint_strength, metal_restraints,
    )

    # -- 5. Partner distance matrix (residue level) ----------------------
    partner_dist = None
    if complex_.partner_indices.size > 0:
        # Use partner Cα / representative atom per residue
        partner_coords = complex_.all_atom_coords[complex_.partner_indices]
        partner_dist   = np.zeros(
            (L, len(complex_.partner_indices)), dtype=np.float32
        )
        compute_pairwise_distances(
            complex_.rna_coords.astype(np.float32),
            partner_coords.astype(np.float32),
            partner_dist,
        )

    # -- 6. Sequence one-hot ---------------------------------------------
    one_hot = sequence_to_one_hot(complex_.rna_sequence)

    return ContextFeatures(
        pdb_id            = complex_.pdb_id,
        rna_length        = L,
        context_type_id   = int(complex_.context_type),
        contact_mask      = residue_contact,
        self_contact_map  = self_contact,
        metal_binding_prob= metal_prob,
        metal_restraints  = metal_restraints,
        partner_dist_mat  = partner_dist,
        sequence_one_hot  = one_hot,
    )


# ---------------------------------------------------------------------------
# 8.  MASKING / DATA-AUGMENTATION FOR PRETRAINING
# ---------------------------------------------------------------------------

def mask_partner_coordinates(
    complex_: PDBComplex,
    mask_protein: bool = True,
    mask_ligand:  bool = True,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns (visible_coords, masked_coords) for the all-atom array.
    Partner atoms are zeroed-out; RNA atoms remain intact.
    This mimics the pretraining objective: 'fold RNA as if partner context present'.

    The masked partner coordinates are kept as training targets for
    optional partner-reconstruction auxiliary loss.
    """
    if rng is None:
        rng = np.random.default_rng()

    coords = complex_.all_atom_coords.copy().astype(np.float32)
    original_partner = coords[complex_.partner_indices].copy()

    if mask_protein and complex_.partner_type in ("protein", "naked"):
        coords[complex_.partner_indices] = 0.0
    if mask_ligand and complex_.partner_type == "ligand":
        coords[complex_.partner_indices] = 0.0

    return coords, original_partner


def random_crop_rna(
    seq: str,
    coords: np.ndarray,   # (L, 3)
    max_length: int = 512,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[str, np.ndarray]:
    """Random contiguous crop for memory-efficient pretraining."""
    if rng is None:
        rng = np.random.default_rng()
    L = len(seq)
    if L <= max_length:
        return seq, coords
    start = int(rng.integers(0, L - max_length))
    return seq[start:start + max_length], coords[start:start + max_length]


# ---------------------------------------------------------------------------
# 9.  PRETRAINING BATCH COLLATOR
# ---------------------------------------------------------------------------

@dataclass
class PretrainingBatch:
    """A collated mini-batch of ContextFeatures, padded to the same length."""
    pdb_ids:           List[str]
    padded_length:     int
    context_type_ids:  np.ndarray     # (B,) int32
    one_hot:           np.ndarray     # (B, L, 5) float32
    contact_masks:     np.ndarray     # (B, L) bool
    self_contacts:     np.ndarray     # (B, L, L) int32
    metal_probs:       np.ndarray     # (B, L) float32
    metal_restraints:  np.ndarray     # (B, L, L) float32
    seq_lengths:       np.ndarray     # (B,) int32


def collate_features(features: List[ContextFeatures]) -> PretrainingBatch:
    """Pad a list of ContextFeatures to a uniform length."""
    B   = len(features)
    L   = max(f.rna_length for f in features)

    ctx_ids   = np.array([f.context_type_id for f in features], dtype=np.int32)
    lengths   = np.array([f.rna_length for f in features], dtype=np.int32)
    one_hot   = np.zeros((B, L, 5), dtype=np.float32)
    c_masks   = np.zeros((B, L), dtype=np.bool_)
    s_cont    = np.zeros((B, L, L), dtype=np.int32)
    m_probs   = np.zeros((B, L), dtype=np.float32)
    m_restr   = np.zeros((B, L, L), dtype=np.float32)

    for i, f in enumerate(features):
        l = f.rna_length
        one_hot[i, :l]      = f.sequence_one_hot
        c_masks[i, :l]      = f.contact_mask
        s_cont[i, :l, :l]   = f.self_contact_map
        m_probs[i, :l]      = f.metal_binding_prob
        m_restr[i, :l, :l]  = f.metal_restraints

    return PretrainingBatch(
        pdb_ids           = [f.pdb_id for f in features],
        padded_length     = L,
        context_type_ids  = ctx_ids,
        one_hot           = one_hot,
        contact_masks     = c_masks,
        self_contacts     = s_cont,
        metal_probs       = m_probs,
        metal_restraints  = m_restr,
        seq_lengths       = lengths,
    )


# ---------------------------------------------------------------------------
# 10.  PRETRAINING LOOP SKELETON
# ---------------------------------------------------------------------------

class RNAContextPretrainer:
    """
    Lightweight pretraining orchestrator.

    Wires together:
      - ContextTypeEmbedding
      - MetalBindingHead
      - Partner-masking data augmentation
      - Mini-batch feature extraction

    The 'model' parameter is a callable:
        single_repr = model(one_hot, context_emb)  → (B, L, d_model)
    In practice, replace with your PyTorch / JAX model.
    """

    def __init__(
        self,
        d_model:   int = 256,
        embed_dim: int = 64,
        hidden_dim:int = 128,
        seed:      int = 0,
    ) -> None:
        self.ctx_emb    = ContextTypeEmbedding(embed_dim=embed_dim, seed=seed)
        self.metal_head = MetalBindingHead(d_model=d_model, hidden_dim=hidden_dim, seed=seed)
        self.proj       = np.random.default_rng(seed).standard_normal(
            (embed_dim, d_model)
        ).astype(np.float32) * 0.02
        self.d_model    = d_model
        self.rng        = np.random.default_rng(seed)

    def preprocess_complex(self, complex_: PDBComplex) -> ContextFeatures:
        """Classify context type (if unknown) then extract features."""
        if complex_.context_type == RNAContextType.UNKNOWN:
            complex_.context_type = classify_rna_context(complex_.rna_sequence)
        return extract_context_features(complex_)

    def training_step(
        self,
        complexes: List[PDBComplex],
        model_fn,   # callable(one_hot, ctx_emb) -> (B, L, d_model)
    ) -> Dict[str, float]:
        """
        One pretraining step.

        Returns a dict of scalar losses:
          - 'metal_binding_loss'
          - 'contact_mask_loss'  (simple cross-entropy on partner contact mask)
        """
        # 1. Augment: mask partner coordinates
        for c in complexes:
            c.all_atom_coords, _ = mask_partner_coordinates(c, rng=self.rng)

        # 2. Feature extraction
        features = [self.preprocess_complex(c) for c in complexes]
        batch    = collate_features(features)

        # 3. Build context embeddings
        ctx_embs = self.ctx_emb(batch.context_type_ids)            # (B, embed_dim)
        ctx_proj = ctx_embs @ self.proj                            # (B, d_model)

        # 4. Forward through model  (stub: returns random single repr)
        B, L = batch.padded_length, self.d_model
        single_repr = model_fn(batch.one_hot, ctx_proj)            # (B, L, d_model)

        # 5. Metal-binding auxiliary head
        metal_pred  = self.metal_head.forward(single_repr)         # (B, L)
        metal_loss  = self.metal_head.binary_cross_entropy_loss(
            metal_pred, batch.metal_probs,
        )

        # 6. Simple logistic loss on partner contact mask
        contact_pred = _sigmoid(single_repr[:, :, 0])              # (B, L) – first dim as logit
        contact_loss = self.metal_head.binary_cross_entropy_loss(
            contact_pred,
            batch.contact_masks.astype(np.float32),
        )

        return {
            "metal_binding_loss": metal_loss,
            "contact_mask_loss":  contact_loss,
            "total_loss":         metal_loss + 0.5 * contact_loss,
        }


# ---------------------------------------------------------------------------
# 11.  INFERENCE HELPER: inject context + restraints
# ---------------------------------------------------------------------------

def inference_prepare(
    sequence: str,
    model_fn,                     # callable(one_hot, ctx_emb) -> single_repr
    ctx_embedding: ContextTypeEmbedding,
    metal_head: MetalBindingHead,
    projection: np.ndarray,       # (embed_dim, d_model)
    known_metal_coords: Optional[np.ndarray] = None,  # (K, 3) float32
    known_metal_types:  Optional[List[str]] = None,
    rna_coords_init: Optional[np.ndarray] = None,     # (L, 3) initialisation guess
) -> Dict[str, np.ndarray]:
    """
    Prepare conditioned single representation for inference.

    1. Classify context type from sequence
    2. Embed context type
    3. Call model to get initial single repr
    4. Predict metal binding sites
    5. Compute metal restraint bias
    6. Return dict of conditioning arrays for the folding model

    Returns
    -------
    {
      'single_repr':      (1, L, d_model),
      'context_type_id':  int,
      'metal_prob':       (L,) float32,
      'metal_restraints': (L, L) float32 – zero if no coords provided,
    }
    """
    L = len(sequence)
    ctx_type    = classify_rna_context(sequence)
    ctx_id      = np.array([int(ctx_type)], dtype=np.int32)
    ctx_emb     = ctx_embedding(ctx_id) @ projection                # (1, d_model)
    one_hot     = sequence_to_one_hot(sequence)[np.newaxis]          # (1, L, 5)
    single_repr = model_fn(one_hot, ctx_emb)                        # (1, L, d_model)

    # Metal binding prediction
    metal_prob = metal_head.forward(single_repr).squeeze(0)         # (L,)

    # Metal restraints if coordinates supplied
    metal_restraints = np.zeros((L, L), dtype=np.float32)
    if known_metal_coords is not None and rna_coords_init is not None:
        metal_coords_f  = known_metal_coords.astype(np.float32)
        radii = np.array(
            [_ION_RADII.get(t, 3.5) for t in (known_metal_types or [])],
            dtype=np.float32,
        )
        prob_from_coords = np.zeros(L, dtype=np.float32)
        compute_metal_binding_features(
            rna_coords_init.astype(np.float32),
            metal_coords_f,
            radii,
            prob_from_coords,
        )
        # Merge with head prediction
        metal_prob = np.maximum(metal_prob, _sigmoid(prob_from_coords))

        rna_dist = np.zeros((L, L), dtype=np.float32)
        compute_pairwise_distances(
            rna_coords_init.astype(np.float32),
            rna_coords_init.astype(np.float32),
            rna_dist,
        )
        apply_metal_distance_restraints(
            rna_dist, metal_prob > 0.5, 0.5, metal_restraints
        )

    return {
        "single_repr":     single_repr,
        "context_type_id": int(ctx_type),
        "context_label":   ctx_type.name,
        "metal_prob":      metal_prob,
        "metal_restraints":metal_restraints,
    }


# ---------------------------------------------------------------------------
# 12.  SYNTHETIC DATA GENERATOR (for testing / benchmarking)
# ---------------------------------------------------------------------------

def make_synthetic_complex(
    rna_length:    int = 64,
    n_protein_atoms: int = 200,
    n_metals:       int = 3,
    seed:          int  = 0,
    context_type:  RNAContextType = RNAContextType.RIBOSWITCH,
) -> PDBComplex:
    """Generate a random synthetic PDB complex for unit tests and benchmarks."""
    rng = np.random.default_rng(seed)

    rna_seq    = "".join(rng.choice(list("AUGC"), size=rna_length))
    rna_coords = rng.standard_normal((rna_length, 3)).astype(np.float32) * 15.0

    # Protein placed near RNA centre
    protein_coords = rng.standard_normal((n_protein_atoms, 3)).astype(np.float32) * 20.0
    metal_coords   = rng.standard_normal((n_metals, 3)).astype(np.float32) * 5.0

    # Stack all atoms: first rna_length (C4' only), then protein, then metals
    rna_atom_idx    = np.arange(rna_length, dtype=np.int32)
    partner_idx     = np.arange(rna_length, rna_length + n_protein_atoms, dtype=np.int32)
    metal_atom_idx  = np.arange(
        rna_length + n_protein_atoms,
        rna_length + n_protein_atoms + n_metals,
        dtype=np.int32,
    )

    all_coords = np.vstack([rna_coords, protein_coords, metal_coords]).astype(np.float32)

    metal_type_list = [METAL_IONS[i % len(METAL_IONS)] for i in range(n_metals)]

    return PDBComplex(
        pdb_id           = f"SYNTH_{seed:04d}",
        rna_sequence     = rna_seq,
        rna_coords       = rna_coords,
        rna_atom_indices = rna_atom_idx,
        partner_indices  = partner_idx,
        all_atom_coords  = all_coords,
        metal_coords     = metal_coords,
        metal_types      = metal_type_list,
        partner_type     = "protein",
        context_type     = context_type,
    )


# ---------------------------------------------------------------------------
# 13.  BENCHMARK / SELF-TEST
# ---------------------------------------------------------------------------

def _stub_model(one_hot: np.ndarray, ctx_emb: np.ndarray) -> np.ndarray:
    """Placeholder model: returns Gaussian noise single representation."""
    B, L, _ = one_hot.shape
    d_model  = ctx_emb.shape[-1]
    return np.random.randn(B, L, d_model).astype(np.float32) * 0.1


def run_benchmark(n_complexes: int = 8, rna_length: int = 128) -> None:
    print("=" * 60)
    print("RNA Context-Aware Pretraining — Benchmark")
    print("=" * 60)

    # Warm up Numba JIT (first call triggers compilation)
    print("\n[1/4] Warming up Numba JIT kernels …")
    dummy = make_synthetic_complex(rna_length=16, seed=999)
    _ = extract_context_features(dummy)
    print("      JIT compilation done.")

    # Build synthetic complexes
    print(f"\n[2/4] Generating {n_complexes} synthetic complexes (L={rna_length}) …")
    ctx_types = list(RNAContextType)
    complexes = [
        make_synthetic_complex(
            rna_length   = rna_length,
            seed         = i,
            context_type = ctx_types[i % len(ctx_types)],
        )
        for i in range(n_complexes)
    ]

    # Feature extraction timing
    print("\n[3/4] Feature extraction …")
    t0 = time.perf_counter()
    features = [extract_context_features(c) for c in complexes]
    t1 = time.perf_counter()
    print(f"      {n_complexes} complexes in {(t1-t0)*1000:.1f} ms  "
          f"({(t1-t0)/n_complexes*1000:.1f} ms/complex)")

    for f in features[:3]:
        print(f"      [{f.pdb_id}] L={f.rna_length} "
              f"ctx={RNAContextType(f.context_type_id).name} "
              f"metal_mean={f.metal_binding_prob.mean():.3f} "
              f"contacts={f.contact_mask.sum()}")

    # Pretraining step timing
    print("\n[4/4] Pretraining step …")
    trainer = RNAContextPretrainer(d_model=256, embed_dim=64, hidden_dim=128)
    t0 = time.perf_counter()
    losses = trainer.training_step(complexes[:4], _stub_model)
    t1 = time.perf_counter()
    print(f"      Step time: {(t1-t0)*1000:.1f} ms")
    for k, v in losses.items():
        print(f"      {k}: {v:.4f}")

    # Sequence classifier
    print("\n--- Sequence Classifier Demo ---")
    test_seqs = {
        "riboswitch_like": "GGCGUAAUUCGAAUGGCGUA" * 3,
        "tRNA_like":       "GCGCCUAUUCGAAUGCUCCA" * 3,
        "unknown":         "AUGUAUGCUAGCUAGCUAGC" * 3,
    }
    for name, seq in test_seqs.items():
        ctx = classify_rna_context(seq)
        print(f"  {name:20s} → {ctx.name}")

    print("\n✓ Benchmark complete.")


if __name__ == "__main__":
    run_benchmark(n_complexes=8, rna_length=128)
