"""
rna_tertiary_heads.py
=====================
Non-WC and Exotic Tertiary Interaction Heads for RNA Structure Prediction

Architecture overview:
  - InteractionLabelEncoder   : maps FR3D/DSSR annotations → integer class labels
  - PairInteractionHead       : per-class cross-entropy heads (WC, non-WC subtypes,
                                stacking, base-phosphate, base-ribose)
  - TorsionHead               : von Mises loss over 7 backbone + χ torsions
  - InteractionBiasInjector   : folds predicted interaction probs into IPA attention
  - RNAInteractionModule      : end-to-end wrapper

Numba acceleration:
  - JIT-compiled distance/angle kernels used during label generation
  - Vectorised pair-feature extraction via @guvectorize
  - Von Mises NLL kernel via @njit
"""

from __future__ import annotations

import math
from enum import IntEnum
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from numba import njit, prange, guvectorize, float32, float64, int64

# ──────────────────────────────────────────────────────────────────────────────
# 1.  Interaction class taxonomy
# ──────────────────────────────────────────────────────────────────────────────

class InteractionClass(IntEnum):
    """Unified label space merging WC, non-WC, and exotic contacts."""
    NO_CONTACT        = 0
    # Watson-Crick
    WC_CG             = 1
    WC_AU             = 2
    WC_GU_WOBBLE      = 3
    # Non-WC (Leontis-Westhof edge notation)
    HOOGSTEEN_TRANS   = 4
    HOOGSTEEN_CIS     = 5
    SUGAR_EDGE_TRANS  = 6
    SUGAR_EDGE_CIS    = 7
    BIFURCATED        = 8
    # Base triples (third-strand participant)
    BASE_TRIPLE_MAJOR = 9
    BASE_TRIPLE_MINOR = 10
    # Backbone contacts
    BASE_PHOSPHATE    = 11
    BASE_RIBOSE       = 12
    # Stacking
    STACK_UPWARD      = 13
    STACK_DOWNWARD    = 14
    STACK_INWARD      = 15
    # A-minor motifs (type I / II)
    A_MINOR_I         = 16
    A_MINOR_II        = 17
    # Ribose zipper
    RIBOSE_ZIPPER     = 18

NUM_CLASSES = len(InteractionClass)

# ──────────────────────────────────────────────────────────────────────────────
# 2.  Numba-accelerated geometry kernels
# ──────────────────────────────────────────────────────────────────────────────

@njit(cache=True, fastmath=True)
def _dist_sq(ax: float, ay: float, az: float,
              bx: float, by: float, bz: float) -> float:
    """Squared Euclidean distance (scalar, JIT-compiled)."""
    dx = ax - bx
    dy = ay - by
    dz = az - bz
    return dx*dx + dy*dy + dz*dz


@njit(parallel=True, cache=True, fastmath=True)
def pairwise_distance_matrix(coords: np.ndarray) -> np.ndarray:
    """
    Compute full N×N distance matrix in parallel.

    Parameters
    ----------
    coords : float32 array, shape (N, 3)

    Returns
    -------
    dist : float32 array, shape (N, N)
    """
    N = coords.shape[0]
    dist = np.empty((N, N), dtype=np.float32)
    for i in prange(N):
        for j in range(N):
            dist[i, j] = math.sqrt(_dist_sq(
                coords[i, 0], coords[i, 1], coords[i, 2],
                coords[j, 0], coords[j, 1], coords[j, 2]))
    return dist


@njit(cache=True, fastmath=True)
def _cross3(ax: float, ay: float, az: float,
            bx: float, by: float, bz: float):
    """Scalar 3-D cross product — returns (cx, cy, cz)."""
    return (ay * bz - az * by,
            az * bx - ax * bz,
            ax * by - ay * bx)


@njit(cache=True, fastmath=True)
def _dot3(ax: float, ay: float, az: float,
          bx: float, by: float, bz: float) -> float:
    """Scalar 3-D dot product."""
    return ax * bx + ay * by + az * bz


@njit(cache=True, fastmath=True)
def torsion_angle(p1: np.ndarray, p2: np.ndarray,
                  p3: np.ndarray, p4: np.ndarray) -> float:
    """
    Dihedral / torsion angle (radians) for four 3-D points.

    Implemented entirely with scalar arithmetic so Numba never needs to
    reconcile mixed float32/float64 dtypes inside np.dot or np.cross.
    Numerically stable near 0 and π (atan2 formulation).
    """
    # bond vectors — promote to float64 scalars immediately
    b1x = float(p2[0] - p1[0]); b1y = float(p2[1] - p1[1]); b1z = float(p2[2] - p1[2])
    b2x = float(p3[0] - p2[0]); b2y = float(p3[1] - p2[1]); b2z = float(p3[2] - p2[2])
    b3x = float(p4[0] - p3[0]); b3y = float(p4[1] - p3[1]); b3z = float(p4[2] - p3[2])

    # n1 = b1 × b2,  n2 = b2 × b3
    n1x, n1y, n1z = _cross3(b1x, b1y, b1z, b2x, b2y, b2z)
    n2x, n2y, n2z = _cross3(b2x, b2y, b2z, b3x, b3y, b3z)

    # b2_norm = b2 / |b2|
    inv_b2 = 1.0 / (math.sqrt(b2x*b2x + b2y*b2y + b2z*b2z) + 1e-9)
    b2nx = b2x * inv_b2; b2ny = b2y * inv_b2; b2nz = b2z * inv_b2

    # m1 = n1 × b2_norm
    m1x, m1y, m1z = _cross3(n1x, n1y, n1z, b2nx, b2ny, b2nz)

    x = _dot3(n1x, n1y, n1z, n2x, n2y, n2z)
    y = _dot3(m1x, m1y, m1z, n2x, n2y, n2z)
    return math.atan2(y, x)


@njit(parallel=True, cache=True, fastmath=True)
def batch_torsion_angles(p1: np.ndarray, p2: np.ndarray,
                         p3: np.ndarray, p4: np.ndarray) -> np.ndarray:
    """
    Vectorised torsion angles over B residues.

    Parameters
    ----------
    p1..p4 : float32 (or float64) arrays, shape (B, 3)
        All four arrays must share the same dtype.

    Returns
    -------
    angles : float32 array, shape (B,)  in [-π, π]
    """
    B = p1.shape[0]
    angles = np.empty(B, dtype=np.float32)
    for i in prange(B):
        angles[i] = torsion_angle(p1[i], p2[i], p3[i], p4[i])
    return angles


@guvectorize(
    [(float32[:], float32[:], float32[:])],
    "(n),(n)->()",
    nopython=True, cache=True, target="parallel"
)
def _dot_gu(a, b, out):
    """GUVectorized batched dot product."""
    s = float32(0.0)
    for k in range(a.shape[0]):
        s += a[k] * b[k]
    out[0] = s


@njit(cache=True, fastmath=True)
def von_mises_nll_scalar(mu: float, kappa: float, obs: float) -> float:
    """
    Negative log-likelihood of von Mises distribution (scalar).

    NLL = log(2π I_0(κ)) - κ cos(obs - μ)

    I_0 approximated via log Bessel for numerical stability.
    """
    # log I_0(kappa) approximation (Abramowitz & Stegun 9.8.5)
    ak = abs(kappa)
    if ak < 3.75:
        t = (ak / 3.75) ** 2
        log_i0 = math.log(
            1.0 + t*(3.5156229 + t*(3.0899424 + t*(1.2067492 +
            t*(0.2659732 + t*(0.0360768 + t*0.0045813))))))
    else:
        t = 3.75 / ak
        log_i0 = ak + math.log(
            (0.39894228 + t*(0.01328592 + t*(0.00225319 +
            t*(-0.00157565 + t*(0.00916281 + t*(-0.02057706 +
            t*(0.02635537 + t*(-0.01647633 + t*0.00392377)))))))) / math.sqrt(ak))
    return math.log(2.0 * math.pi) + log_i0 - kappa * math.cos(obs - mu)


@njit(parallel=True, cache=True, fastmath=True)
def batch_von_mises_nll(mu: np.ndarray, kappa: np.ndarray,
                         obs: np.ndarray) -> np.ndarray:
    """
    Batch von Mises NLL over shape (B, T) torsion arrays.

    Parameters
    ----------
    mu, kappa, obs : float32, shape (B, T)
        T = number of torsion channels (typically 8: α β γ δ ε ζ + χ + pseudo)

    Returns
    -------
    nll : float32, shape (B, T)
    """
    B, T = mu.shape
    nll = np.empty((B, T), dtype=np.float32)
    for i in prange(B):
        for j in range(T):
            nll[i, j] = von_mises_nll_scalar(mu[i, j], kappa[i, j], obs[i, j])
    return nll


# ──────────────────────────────────────────────────────────────────────────────
# 3.  Label encoder  (FR3D / DSSR annotation → integer tensor)
# ──────────────────────────────────────────────────────────────────────────────

# Mapping from common annotation strings → InteractionClass
_ANNOTATION_MAP: dict[str, InteractionClass] = {
    # Watson-Crick
    "cWW_CG": InteractionClass.WC_CG,
    "cWW_GC": InteractionClass.WC_CG,
    "cWW_AU": InteractionClass.WC_AU,
    "cWW_UA": InteractionClass.WC_AU,
    "cWW_GU": InteractionClass.WC_GU_WOBBLE,
    "cWW_UG": InteractionClass.WC_GU_WOBBLE,
    # Non-WC Hoogsteen
    "tHH":    InteractionClass.HOOGSTEEN_TRANS,
    "cHH":    InteractionClass.HOOGSTEEN_CIS,
    "tHW":    InteractionClass.HOOGSTEEN_TRANS,
    "cHW":    InteractionClass.HOOGSTEEN_CIS,
    "tWH":    InteractionClass.HOOGSTEEN_TRANS,
    "cWH":    InteractionClass.HOOGSTEEN_CIS,
    # Sugar edge
    "tSS":    InteractionClass.SUGAR_EDGE_TRANS,
    "cSS":    InteractionClass.SUGAR_EDGE_CIS,
    "tSW":    InteractionClass.SUGAR_EDGE_TRANS,
    "cSW":    InteractionClass.SUGAR_EDGE_CIS,
    "tWS":    InteractionClass.SUGAR_EDGE_TRANS,
    "cWS":    InteractionClass.SUGAR_EDGE_CIS,
    # Bifurcated
    "BPh":    InteractionClass.BASE_PHOSPHATE,
    "BR":     InteractionClass.BASE_RIBOSE,
    "s35":    InteractionClass.STACK_UPWARD,
    "s53":    InteractionClass.STACK_DOWNWARD,
    "s33":    InteractionClass.STACK_INWARD,
    "s55":    InteractionClass.STACK_INWARD,
    # A-minor
    "A-minor_I":  InteractionClass.A_MINOR_I,
    "A-minor_II": InteractionClass.A_MINOR_II,
    # Ribose zipper
    "ribose_zipper": InteractionClass.RIBOSE_ZIPPER,
}


def encode_annotations(
    annotations: list[tuple[int, int, str]],
    n_residues: int,
) -> torch.Tensor:
    """
    Convert a list of (i, j, annotation_string) triples into an integer label
    matrix.

    Parameters
    ----------
    annotations : list of (i, j, label_str) tuples
        Produced by parsing FR3D / DSSR output for a single structure.
    n_residues : int

    Returns
    -------
    labels : int64 tensor, shape (N, N)
        labels[i, j] = InteractionClass integer (0 = no contact)
    """
    labels = np.zeros((n_residues, n_residues), dtype=np.int64)
    for i, j, ann in annotations:
        cls = _ANNOTATION_MAP.get(ann, None)
        if cls is not None and 0 <= i < n_residues and 0 <= j < n_residues:
            labels[i, j] = int(cls)
            # Stacking is directional; base-pairing interactions are symmetric
            if cls in (InteractionClass.WC_CG, InteractionClass.WC_AU,
                       InteractionClass.WC_GU_WOBBLE,
                       InteractionClass.BASE_PHOSPHATE,
                       InteractionClass.BASE_RIBOSE,
                       InteractionClass.A_MINOR_I, InteractionClass.A_MINOR_II,
                       InteractionClass.RIBOSE_ZIPPER):
                labels[j, i] = int(cls)
    return torch.from_numpy(labels)


# ──────────────────────────────────────────────────────────────────────────────
# 4.  Pair interaction head
# ──────────────────────────────────────────────────────────────────────────────

class PairInteractionHead(nn.Module):
    """
    Multi-class pair interaction head.

    Projects pair embeddings → logits over InteractionClass, supervised with
    focal cross-entropy to handle extreme class imbalance (most pairs = NO_CONTACT).

    Parameters
    ----------
    pair_dim : int
        Dimensionality of the incoming pair representation z_{ij}.
    hidden_dim : int
        Hidden layer width.
    num_classes : int
        Number of interaction classes (defaults to NUM_CLASSES).
    gamma_focal : float
        Focal loss γ parameter. Set 0 to recover standard cross-entropy.
    class_weights : optional tensor of shape (num_classes,)
        Manual per-class weights (use for heavy imbalance).
    """

    def __init__(
        self,
        pair_dim: int,
        hidden_dim: int = 128,
        num_classes: int = NUM_CLASSES,
        gamma_focal: float = 2.0,
        class_weights: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.num_classes  = num_classes
        self.gamma_focal  = gamma_focal

        self.mlp = nn.Sequential(
            nn.LayerNorm(pair_dim),
            nn.Linear(pair_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, num_classes),
        )

        if class_weights is not None:
            self.register_buffer("class_weights", class_weights)
        else:
            self.class_weights = None

    def forward(self, pair_repr: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        pair_repr : float tensor, shape (..., N, N, pair_dim)

        Returns
        -------
        logits : float tensor, shape (..., N, N, num_classes)
        """
        return self.mlp(pair_repr)

    def focal_loss(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Focal cross-entropy loss.

        Parameters
        ----------
        logits : (..., N, N, C)
        labels : (..., N, N)  int64, values in [0, C)
        mask   : (..., N, N)  bool, True = valid pair

        Returns
        -------
        loss : scalar tensor
        """
        # Flatten spatial dims
        C = logits.shape[-1]
        flat_logits = logits.reshape(-1, C)
        flat_labels = labels.reshape(-1)

        ce = F.cross_entropy(
            flat_logits, flat_labels,
            weight=self.class_weights,
            reduction="none",
        )
        if self.gamma_focal > 0:
            probs = F.softmax(flat_logits, dim=-1)
            pt    = probs.gather(1, flat_labels.unsqueeze(1)).squeeze(1)
            ce    = ((1 - pt) ** self.gamma_focal) * ce

        if mask is not None:
            flat_mask = mask.reshape(-1).float()
            loss = (ce * flat_mask).sum() / (flat_mask.sum() + 1e-8)
        else:
            loss = ce.mean()
        return loss


# ──────────────────────────────────────────────────────────────────────────────
# 5.  Torsion head with von Mises loss
# ──────────────────────────────────────────────────────────────────────────────

# RNA backbone torsion names (IUPAC + χ)
TORSION_NAMES = ["alpha", "beta", "gamma", "delta", "epsilon", "zeta", "chi", "pseudo"]
N_TORSIONS    = len(TORSION_NAMES)   # 8


class TorsionHead(nn.Module):
    """
    Predicts per-residue backbone torsions as (sin, cos) pairs and computes
    a von Mises negative log-likelihood loss.

    The network outputs:
      - μ  (mean angle)  via atan2(sin_pred, cos_pred)
      - κ  (concentration) via softplus → ensures κ > 0

    Parameters
    ----------
    single_dim : int
        Dimensionality of per-residue single representation.
    hidden_dim : int
    n_torsions : int
        Number of torsion channels (default 8: α β γ δ ε ζ χ pseudo).
    """

    def __init__(
        self,
        single_dim: int,
        hidden_dim: int = 128,
        n_torsions: int = N_TORSIONS,
    ):
        super().__init__()
        self.n_torsions = n_torsions

        # Predicts (sin, cos, log_kappa) for each torsion
        self.mlp = nn.Sequential(
            nn.LayerNorm(single_dim),
            nn.Linear(single_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, n_torsions * 3),   # sin, cos, log_κ per torsion
        )

    def forward(self, single_repr: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Parameters
        ----------
        single_repr : (B, N, single_dim)

        Returns
        -------
        dict with keys:
          'mu'    : (B, N, T)  predicted mean angle in [-π, π]
          'kappa' : (B, N, T)  predicted concentration > 0
          'sin'   : (B, N, T)
          'cos'   : (B, N, T)
        """
        B, N, _ = single_repr.shape
        raw = self.mlp(single_repr).reshape(B, N, self.n_torsions, 3)

        sin_pred = raw[..., 0]
        cos_pred = raw[..., 1]
        log_kap  = raw[..., 2]

        # Stack (sin, cos) → (B, N, T, 2) and normalise on the unit circle
        # with F.normalize (clips denominator from below, never inflates it)
        # so the output satisfies sin²+cos²==1 to float32 precision.
        # Dividing by sqrt(s²+c²+1e-8) adds eps *inside* the sqrt, inflating
        # the denominator and producing sin²+cos²<1 by ~1e-3 near init.
        sc       = torch.stack([sin_pred, cos_pred], dim=-1)      # (..., 2)
        sc_norm  = F.normalize(sc, p=2.0, dim=-1, eps=1e-6)       # unit circle
        sin_norm = sc_norm[..., 0]                                 # (B, N, T)
        cos_norm = sc_norm[..., 1]                                 # (B, N, T)
        mu       = torch.atan2(sin_norm, cos_norm)                 # ∈ [-π, π]
        kappa    = F.softplus(log_kap) + 1e-4                      # κ > 0

        return {"mu": mu, "kappa": kappa, "sin": sin_norm, "cos": cos_norm}

    def von_mises_loss(
        self,
        pred: dict[str, torch.Tensor],
        obs_angles: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Von Mises NLL averaged over valid residues and torsion channels.

        Parameters
        ----------
        pred        : output of forward()
        obs_angles  : (B, N, T) observed torsion angles in [-π, π]
        mask        : (B, N) bool, True = valid residue

        Returns
        -------
        loss : scalar
        """
        mu    = pred["mu"]
        kappa = pred["kappa"]

        # log I_0(κ) via Bessel function (differentiable torch path)
        log_i0 = _log_bessel_i0_torch(kappa)

        nll = math.log(2.0 * math.pi) + log_i0 - kappa * torch.cos(obs_angles - mu)
        # nll : (B, N, T)

        if mask is not None:
            valid = mask.float().unsqueeze(-1)          # (B, N, 1)
            loss  = (nll * valid).sum() / (valid.sum() * self.n_torsions + 1e-8)
        else:
            loss = nll.mean()
        return loss


def _log_bessel_i0_torch(kappa: torch.Tensor) -> torch.Tensor:
    """
    Differentiable log I_0(κ) approximation (Abramowitz & Stegun 9.8.5).
    Handles small and large κ with a smooth blend.
    """
    small_t = (kappa / 3.75) ** 2
    log_i0_small = torch.log(
        1.0 + small_t*(3.5156229 + small_t*(3.0899424 + small_t*(1.2067492 +
        small_t*(0.2659732 + small_t*(0.0360768 + small_t*0.0045813)))))
    )
    large_t = 3.75 / (kappa + 1e-9)
    log_i0_large = kappa + torch.log(
        (0.39894228 + large_t*(0.01328592 + large_t*(0.00225319 +
        large_t*(-0.00157565 + large_t*(0.00916281 + large_t*(-0.02057706 +
        large_t*(0.02635537 + large_t*(-0.01647633 + large_t*0.00392377)))))))) /
        torch.sqrt(kappa + 1e-9)
    )
    # Blend: use small branch for kappa < 3.75
    return torch.where(kappa < 3.75, log_i0_small, log_i0_large)


# ──────────────────────────────────────────────────────────────────────────────
# 6.  Interaction bias injector for IPA / SE(3)-Transformer
# ──────────────────────────────────────────────────────────────────────────────

class InteractionBiasInjector(nn.Module):
    """
    Projects predicted interaction probabilities → per-head attention biases,
    ready for addition into invariant point attention (IPA) or SE(3)-Transformer
    before the softmax.

    Parameters
    ----------
    num_classes      : int   – interaction label vocabulary size
    num_heads        : int   – number of attention heads in the structure module
    pair_dim         : int   – width of pair representation (z_{ij})
    use_pair_context : bool  – also project pair_repr → bias (recommended)
    """

    def __init__(
        self,
        num_classes: int  = NUM_CLASSES,
        num_heads: int    = 12,
        pair_dim: int     = 128,
        use_pair_context: bool = True,
    ):
        super().__init__()
        self.num_heads        = num_heads
        self.use_pair_context = use_pair_context

        # Interaction prob  →  per-head scalar bias
        self.prob_proj = nn.Sequential(
            nn.Linear(num_classes, num_heads),
        )

        if use_pair_context:
            self.pair_proj = nn.Linear(pair_dim, num_heads)
        else:
            self.pair_proj = None

    def forward(
        self,
        interaction_logits: torch.Tensor,
        pair_repr: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        interaction_logits : (B, N, N, C)  raw logits from PairInteractionHead
        pair_repr          : (B, N, N, pair_dim)  optional

        Returns
        -------
        attn_bias : (B, H, N, N)  — add to pre-softmax attention scores
        """
        probs = F.softmax(interaction_logits, dim=-1)   # (B, N, N, C)
        bias  = self.prob_proj(probs)                   # (B, N, N, H)

        if self.use_pair_context and pair_repr is not None and self.pair_proj is not None:
            bias = bias + self.pair_proj(pair_repr)     # (B, N, N, H)

        # Rearrange to (B, H, N, N) for standard attention layout
        return bias.permute(0, 3, 1, 2)


# ──────────────────────────────────────────────────────────────────────────────
# 7.  End-to-end module
# ──────────────────────────────────────────────────────────────────────────────

class RNAInteractionModule(nn.Module):
    """
    Pluggable module that adds non-WC and exotic tertiary interaction heads to
    any RNA structure prediction network exposing (single_repr, pair_repr).

    Loss terms returned by compute_loss() can be summed with the main
    structure loss:

        L_total = L_structure + λ_pair * L_pair + λ_torsion * L_torsion

    Parameters
    ----------
    single_dim        : width of per-residue embedding  (e.g. 256)
    pair_dim          : width of pair embedding          (e.g. 128)
    num_heads         : IPA attention heads
    torsion_hidden    : hidden size in torsion MLP
    pair_hidden       : hidden size in pair interaction MLP
    focal_gamma       : focal loss γ (2.0 recommended)
    lambda_pair       : loss weight for pair interaction head
    lambda_torsion    : loss weight for torsion head
    """

    def __init__(
        self,
        single_dim:     int   = 256,
        pair_dim:       int   = 128,
        num_heads:      int   = 12,
        torsion_hidden: int   = 128,
        pair_hidden:    int   = 256,
        focal_gamma:    float = 2.0,
        lambda_pair:    float = 1.0,
        lambda_torsion: float = 0.5,
    ):
        super().__init__()
        self.lambda_pair    = lambda_pair
        self.lambda_torsion = lambda_torsion

        self.pair_head = PairInteractionHead(
            pair_dim    = pair_dim,
            hidden_dim  = pair_hidden,
            gamma_focal = focal_gamma,
        )
        self.torsion_head = TorsionHead(
            single_dim = single_dim,
            hidden_dim = torsion_hidden,
        )
        self.bias_injector = InteractionBiasInjector(
            num_classes      = NUM_CLASSES,
            num_heads        = num_heads,
            pair_dim         = pair_dim,
            use_pair_context = True,
        )

    # ------------------------------------------------------------------
    def forward(
        self,
        single_repr: torch.Tensor,
        pair_repr:   torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """
        Parameters
        ----------
        single_repr : (B, N, single_dim)
        pair_repr   : (B, N, N, pair_dim)

        Returns
        -------
        {
          'interaction_logits' : (B, N, N, NUM_CLASSES),
          'attn_bias'          : (B, H, N, N),
          'torsion_pred'       : dict with mu/kappa/sin/cos  (B, N, T),
        }
        """
        inter_logits = self.pair_head(pair_repr)
        attn_bias    = self.bias_injector(inter_logits, pair_repr)
        torsion_pred = self.torsion_head(single_repr)

        return {
            "interaction_logits": inter_logits,
            "attn_bias":          attn_bias,
            "torsion_pred":       torsion_pred,
        }

    # ------------------------------------------------------------------
    def compute_loss(
        self,
        outputs:        dict[str, torch.Tensor],
        interaction_labels: torch.Tensor,           # (B, N, N) int64
        torsion_obs:    torch.Tensor,               # (B, N, T) float32, radians
        pair_mask:      Optional[torch.Tensor] = None,  # (B, N, N) bool
        residue_mask:   Optional[torch.Tensor] = None,  # (B, N)    bool
    ) -> dict[str, torch.Tensor]:
        """
        Compute and return all auxiliary losses.

        Returns
        -------
        {
          'loss_pair'    : scalar,
          'loss_torsion' : scalar,
          'loss_total'   : scalar,
        }
        """
        l_pair = self.pair_head.focal_loss(
            outputs["interaction_logits"],
            interaction_labels,
            mask=pair_mask,
        )
        l_torsion = self.torsion_head.von_mises_loss(
            outputs["torsion_pred"],
            torsion_obs,
            mask=residue_mask,
        )
        l_total = self.lambda_pair * l_pair + self.lambda_torsion * l_torsion

        return {
            "loss_pair":    l_pair,
            "loss_torsion": l_torsion,
            "loss_total":   l_total,
        }

    # ------------------------------------------------------------------
    @torch.no_grad()
    def predict_interactions(
        self,
        pair_repr: torch.Tensor,
        threshold: float = 0.3,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Convenience method for inference.

        Returns
        -------
        pred_class : (B, N, N) int64  – argmax class
        pred_mask  : (B, N, N) bool   – True where max prob > threshold
        """
        logits     = self.pair_head(pair_repr)
        probs      = F.softmax(logits, dim=-1)
        pred_class = probs.argmax(dim=-1)
        max_prob   = probs.max(dim=-1).values
        pred_mask  = (pred_class != InteractionClass.NO_CONTACT) & (max_prob > threshold)
        return pred_class, pred_mask


# ──────────────────────────────────────────────────────────────────────────────
# 8.  Utility: numba-accelerated INF-all metric (Interaction Network Fidelity)
# ──────────────────────────────────────────────────────────────────────────────

@njit(parallel=True, cache=True, fastmath=True)
def compute_inf_all(
    pred_labels: np.ndarray,
    true_labels: np.ndarray,
) -> float:
    """
    Compute INF-all (interaction network fidelity) between predicted and true
    interaction label matrices.

    INF-all = F1 score over all non-NO_CONTACT interaction pairs.

    Parameters
    ----------
    pred_labels, true_labels : int64 arrays, shape (N, N)

    Returns
    -------
    inf_all : float  in [0, 1]
    """
    N = pred_labels.shape[0]
    tp = np.int64(0)
    fp = np.int64(0)
    fn = np.int64(0)

    for i in prange(N):
        for j in range(N):
            p = pred_labels[i, j] != 0
            t = true_labels[i, j] != 0
            if p and t and (pred_labels[i, j] == true_labels[i, j]):
                tp += 1
            elif p and not t:
                fp += 1
            elif not p and t:
                fn += 1

    prec = tp / (tp + fp + 1e-9)
    rec  = tp / (tp + fn + 1e-9)
    if prec + rec < 1e-9:
        return 0.0
    return 2.0 * prec * rec / (prec + rec)


# ──────────────────────────────────────────────────────────────────────────────
# 9.  Quick smoke-test
# ──────────────────────────────────────────────────────────────────────────────

def _smoke_test():
    """Runs a small forward pass + loss computation to verify shapes."""
    B, N = 2, 32
    SINGLE_DIM, PAIR_DIM, H = 256, 128, 12

    model = RNAInteractionModule(
        single_dim=SINGLE_DIM, pair_dim=PAIR_DIM, num_heads=H
    )

    single = torch.randn(B, N, SINGLE_DIM)
    pair   = torch.randn(B, N, N, PAIR_DIM)

    out = model(single, pair)
    assert out["interaction_logits"].shape == (B, N, N, NUM_CLASSES), \
        f"Bad shape: {out['interaction_logits'].shape}"
    assert out["attn_bias"].shape == (B, H, N, N), \
        f"Bad bias shape: {out['attn_bias'].shape}"
    assert out["torsion_pred"]["mu"].shape == (B, N, N_TORSIONS), \
        f"Bad torsion shape: {out['torsion_pred']['mu'].shape}"

    labels    = torch.randint(0, NUM_CLASSES, (B, N, N))
    tors_obs  = torch.rand(B, N, N_TORSIONS) * 2 * math.pi - math.pi
    losses    = model.compute_loss(out, labels, tors_obs)

    print("=== Smoke test passed ===")
    print("  interaction_logits :", out['interaction_logits'].shape)
    print("  attn_bias          :", out['attn_bias'].shape)
    print("  torsion mu         :", out['torsion_pred']['mu'].shape)
    print("  loss_pair          :", round(losses['loss_pair'].item(), 4))
    print("  loss_torsion       :", round(losses['loss_torsion'].item(), 4))
    print("  loss_total         :", round(losses['loss_total'].item(), 4))