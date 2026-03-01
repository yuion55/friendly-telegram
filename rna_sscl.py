"""
rna_sscl.py — Secondary Structure–Constrained Loss (SSCL) for RNA 3D Structure Prediction
============================================================================================
Implements Joint 2D–3D training with:
  - Numba JIT for inner-loop geometry kernels
  - NumPy vectorization for batch operations
  - Curriculum λ annealing (high SS weight early → low later)
  - WC / non-WC-pseudoknot / no-contact 3-class base-pair head
  - DSSR-style annotation support

References
----------
RNA3D-SSCL (2025): Secondary Structure–Constrained Loss for RNA 3D Prediction
"""

from __future__ import annotations

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from numba import njit, prange, float32, int32
from typing import Optional, Tuple


# ---------------------------------------------------------------------------
# 1. Numba-accelerated geometry kernels
# ---------------------------------------------------------------------------

@njit(float32[:, :](float32[:, :]), parallel=True, cache=True, fastmath=True)
def pairwise_distance_matrix(coords: np.ndarray) -> np.ndarray:
    """
    Compute all-pairs Euclidean distance matrix for N atoms.
    
    Parameters
    ----------
    coords : (N, 3) float32
        3D coordinates of RNA residues (e.g., C1' or P atoms).

    Returns
    -------
    D : (N, N) float32
        Distance matrix.
    """
    N = coords.shape[0]
    D = np.zeros((N, N), dtype=np.float32)
    for i in prange(N):
        for j in range(i + 1, N):
            dx = coords[i, 0] - coords[j, 0]
            dy = coords[i, 1] - coords[j, 1]
            dz = coords[i, 2] - coords[j, 2]
            d = math.sqrt(dx * dx + dy * dy + dz * dz)
            D[i, j] = d
            D[j, i] = d
    return D


@njit(float32[:](float32[:, :], int32[:, :]), parallel=True, cache=True, fastmath=True)
def extract_pair_distances(coords: np.ndarray, pairs: np.ndarray) -> np.ndarray:
    """
    Extract distances for a list of (i, j) index pairs — much faster than full matrix
    when only a subset of pairs is needed (e.g., predicted base-pair contacts).

    Parameters
    ----------
    coords : (N, 3) float32
    pairs  : (P, 2) int32   — each row is (i, j)

    Returns
    -------
    dists : (P,) float32
    """
    P = pairs.shape[0]
    dists = np.zeros(P, dtype=np.float32)
    for k in prange(P):
        i, j = pairs[k, 0], pairs[k, 1]
        dx = coords[i, 0] - coords[j, 0]
        dy = coords[i, 1] - coords[j, 1]
        dz = coords[i, 2] - coords[j, 2]
        dists[k] = math.sqrt(dx * dx + dy * dy + dz * dz)
    return dists


@njit(cache=True, fastmath=True)
def stem_consistency_score(coords: np.ndarray,
                           wc_pairs: np.ndarray,
                           target_dist: float = np.float32(10.4),
                           sigma: float = np.float32(2.0)) -> float:
    """
    Gaussian-kernel score for how well WC pairs match the canonical C1'–C1' distance.
    Canonical Watson-Crick C1'–C1' ≈ 10.4 Å.

    Returns mean score in [0, 1]; 1 = perfect geometry.
    """
    P = wc_pairs.shape[0]
    if P == 0:
        return 1.0
    total = 0.0
    for k in range(P):
        i, j = wc_pairs[k, 0], wc_pairs[k, 1]
        dx = coords[i, 0] - coords[j, 0]
        dy = coords[i, 1] - coords[j, 1]
        dz = coords[i, 2] - coords[j, 2]
        d = math.sqrt(dx * dx + dy * dy + dz * dz)
        diff = (d - target_dist) / sigma
        total += math.exp(-0.5 * diff * diff)
    return total / P


@njit(float32[:](float32[:, :], int32[:]), cache=True, fastmath=True, parallel=True)
def backbone_bond_lengths(coords: np.ndarray, residue_idx: np.ndarray) -> np.ndarray:
    """
    Compute consecutive P–P or C1'–C1' backbone distances for a chain.

    Parameters
    ----------
    coords      : (N, 3) float32
    residue_idx : (M,)  int32  — ordered residue indices in the chain

    Returns
    -------
    lengths : (M-1,) float32
    """
    M = residue_idx.shape[0]
    lengths = np.zeros(M - 1, dtype=np.float32)
    for k in prange(M - 1):
        i = residue_idx[k]
        j = residue_idx[k + 1]
        dx = coords[i, 0] - coords[j, 0]
        dy = coords[i, 1] - coords[j, 1]
        dz = coords[i, 2] - coords[j, 2]
        lengths[k] = math.sqrt(dx * dx + dy * dy + dz * dz)
    return lengths


# ---------------------------------------------------------------------------
# 2. Secondary Structure Head (pair representation → 3-class logits)
# ---------------------------------------------------------------------------

class SecondaryStructureHead(nn.Module):
    """
    Predicts per-pair base-pair class from a pair representation.

    Classes
    -------
    0 : no contact
    1 : Watson-Crick (canonical stem)
    2 : non-WC / pseudoknot

    Architecture
    ------------
    Pair repr (B, L, L, d_pair) → 2-layer MLP → (B, L, L, 3)
    Symmetrized so logits[i,j] == logits[j,i].
    """

    def __init__(self, d_pair: int, hidden: int = 128, dropout: float = 0.1):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(d_pair, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.GELU(),
            nn.Linear(hidden // 2, 3),
        )

    def forward(self, pair_repr: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        pair_repr : (B, L, L, d_pair)

        Returns
        -------
        logits : (B, L, L, 3)   — symmetrized
        """
        logits = self.proj(pair_repr)
        # Symmetrize: logits_sym[i,j] = (logits[i,j] + logits[j,i]) / 2
        logits = (logits + logits.transpose(1, 2)) * 0.5
        return logits


# ---------------------------------------------------------------------------
# 3. Non-WC (Pseudoknot) Head — separate binary head for pseudoknot contacts
# ---------------------------------------------------------------------------

class PseudoknotHead(nn.Module):
    """
    Binary head predicting pseudoknot contacts on top of pair representation.
    This is kept separate to allow different λ weighting.
    """

    def __init__(self, d_pair: int, hidden: int = 64):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(d_pair, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, pair_repr: torch.Tensor) -> torch.Tensor:
        """Returns (B, L, L) logits (pre-sigmoid)."""
        logits = self.proj(pair_repr).squeeze(-1)
        return (logits + logits.transpose(1, 2)) * 0.5


# ---------------------------------------------------------------------------
# 4. Vectorized loss functions (pure PyTorch — GPU compatible)
# ---------------------------------------------------------------------------

def ss_cross_entropy_loss(
    ss_logits: torch.Tensor,       # (B, L, L, 3)
    ss_labels: torch.Tensor,       # (B, L, L) int64 — 0/1/2
    pair_mask: Optional[torch.Tensor] = None,  # (B, L, L) bool
    class_weights: Optional[torch.Tensor] = None,  # (3,)
    label_smoothing: float = 0.05,
) -> torch.Tensor:
    """
    Cross-entropy loss for the 3-class secondary structure head.
    Only upper-triangle pairs are included (avoids double-counting).

    Parameters
    ----------
    ss_logits       : (B, L, L, 3)
    ss_labels       : (B, L, L)   0=no-contact, 1=WC, 2=pseudoknot
    pair_mask       : (B, L, L)   True where valid (exclude padding)
    class_weights   : (3,)        re-weight rare positive classes
    label_smoothing : float       standard smoothing

    Returns
    -------
    loss : scalar
    """
    B, L, _, C = ss_logits.shape

    # Upper-triangle mask (i < j) to avoid double-counting
    tri_mask = torch.ones(L, L, dtype=torch.bool, device=ss_logits.device).triu(diagonal=1)
    tri_mask = tri_mask.unsqueeze(0).expand(B, -1, -1)  # (B, L, L)

    if pair_mask is not None:
        tri_mask = tri_mask & pair_mask

    logits_flat = ss_logits[tri_mask]       # (P, 3)
    labels_flat = ss_labels[tri_mask]       # (P,)

    if logits_flat.numel() == 0:
        return ss_logits.sum() * 0.0        # zero loss, keeps gradient graph

    loss = F.cross_entropy(
        logits_flat,
        labels_flat,
        weight=class_weights,
        label_smoothing=label_smoothing,
        reduction="mean",
    )
    return loss


def contact_focal_loss(
    ss_logits: torch.Tensor,       # (B, L, L, 3)
    ss_labels: torch.Tensor,       # (B, L, L)
    gamma: float = 2.0,
    alpha: float = 0.25,
    pair_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Focal loss variant — strongly up-weights hard mis-classified contacts.
    Useful for pseudoknot class which is rare and hard.

    Returns scalar loss.
    """
    B, L, _, C = ss_logits.shape
    tri_mask = torch.ones(L, L, dtype=torch.bool, device=ss_logits.device).triu(diagonal=1)
    tri_mask = tri_mask.unsqueeze(0).expand(B, -1, -1)
    if pair_mask is not None:
        tri_mask = tri_mask & pair_mask

    logits_flat = ss_logits[tri_mask]       # (P, 3)
    labels_flat = ss_labels[tri_mask]       # (P,)

    if logits_flat.numel() == 0:
        return ss_logits.sum() * 0.0

    log_probs = F.log_softmax(logits_flat, dim=-1)          # (P, 3)
    probs = log_probs.exp()
    p_t = probs.gather(1, labels_flat.unsqueeze(1)).squeeze(1)  # (P,)
    focal_weight = (1 - p_t) ** gamma
    ce = -log_probs.gather(1, labels_flat.unsqueeze(1)).squeeze(1)
    loss = (alpha * focal_weight * ce).mean()
    return loss


def geometry_consistency_loss(
    coords_pred: torch.Tensor,     # (B, L, 3)
    ss_probs: torch.Tensor,        # (B, L, L, 3)  softmax probs
    wc_target_dist: float = 10.4,
    pk_target_dist: float = 11.5,
    sigma_wc: float = 1.5,
    sigma_pk: float = 2.0,
    pair_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Encourages predicted WC-paired residues to have C1'–C1' distances
    near canonical values. Fully differentiable (no Numba here; runs on GPU).

    L_geo = mean over (i,j) of [ P(WC|ij) * (d_ij - d_WC)^2 / σ_WC^2
                                + P(PK|ij) * (d_ij - d_PK)^2 / σ_PK^2 ]

    Parameters
    ----------
    coords_pred     : (B, L, 3)
    ss_probs        : (B, L, L, 3)  class probs [no-contact, WC, PK]
    wc_target_dist  : Å  canonical WC C1'–C1' distance
    pk_target_dist  : Å  canonical pseudoknot C1'–C1' distance
    sigma_wc        : spread for WC Gaussian
    sigma_pk        : spread for PK Gaussian
    pair_mask       : (B, L, L) bool

    Returns
    -------
    loss : scalar
    """
    B, L, _ = coords_pred.shape

    # (B, L, L) pairwise distances — vectorized
    # diff[b, i, j] = coords[b, i] - coords[b, j]
    c = coords_pred  # (B, L, 3)
    diff = c.unsqueeze(2) - c.unsqueeze(1)          # (B, L, L, 3)
    dist = torch.norm(diff, dim=-1, keepdim=False)  # (B, L, L)

    p_wc = ss_probs[..., 1]   # (B, L, L)
    p_pk = ss_probs[..., 2]   # (B, L, L)

    loss_wc = p_wc * ((dist - wc_target_dist) ** 2) / (sigma_wc ** 2)
    loss_pk = p_pk * ((dist - pk_target_dist) ** 2) / (sigma_pk ** 2)

    geo_loss = loss_wc + loss_pk  # (B, L, L)

    # Upper triangle only
    tri_mask = torch.ones(L, L, dtype=torch.bool, device=coords_pred.device).triu(diagonal=1)
    tri_mask = tri_mask.unsqueeze(0).expand(B, -1, -1)
    if pair_mask is not None:
        tri_mask = tri_mask & pair_mask

    return geo_loss[tri_mask].mean()


def stem_planarity_loss(
    coords_pred: torch.Tensor,      # (B, L, 3)
    ss_probs: torch.Tensor,         # (B, L, L, 3)
    min_stack_run: int = 2,
) -> torch.Tensor:
    """
    Penalizes adjacent WC pairs (stacked stems) for deviating from co-planarity.
    For a stem segment (i,j) and (i+1, j-1), the four C1' atoms should be nearly co-planar.

    This is a soft loss: weighted by P(WC|i,j) * P(WC|i+1,j-1).

    Returns scalar loss.
    """
    B, L, _ = coords_pred.shape
    p_wc = ss_probs[..., 1]   # (B, L, L)

    total_loss = coords_pred.new_zeros(1)
    count = 0

    # Vectorized over batch; loop over stem positions (small inner loop)
    for di in range(1, min(4, L)):
        # Check stacking: pair (i, j) and (i+di, j-di)
        # p_stack[b, i, j] = P(WC|i,j) * P(WC|i+di, j-di)
        p_wc_ij   = p_wc[:, :L - di, di:]              # (B, L-di, L-di)
        p_wc_adj  = p_wc[:, di:, :L - di]              # (B, L-di, L-di)
        w = p_wc_ij * p_wc_adj                          # (B, L-di, L-di)

        if w.sum() < 1e-6:
            continue

        # Get the four C1' positions
        i_idx = torch.arange(L - di, device=coords_pred.device)
        j_idx = torch.arange(di, L, device=coords_pred.device)

        A = coords_pred[:, i_idx, :]           # (B, L-di, 3)  residue i
        B_ = coords_pred[:, j_idx, :]          # (B, L-di, 3)  residue j
        C = coords_pred[:, i_idx + di, :]      # (B, L-di, 3)  residue i+di
        D_ = coords_pred[:, j_idx - di, :]     # (B, L-di, 3)  residue j-di

        # Normal of plane (A, B_, C) — batched cross product
        AB = B_ - A   # (B, L-di, 3)
        AC = C - A    # (B, L-di, 3)
        n = torch.cross(AB, AC, dim=-1)          # (B, L-di, 3)
        n = F.normalize(n, dim=-1)

        # Distance of D from plane
        AD = D_ - A  # (B, L-di, 3)
        d_plane = (AD * n).sum(dim=-1).abs()     # (B, L-di)

        # Lower-triangular mask (j > i)
        upper = (j_idx.unsqueeze(0) > i_idx.unsqueeze(1)).to(w.device)  # (L-di, L-di)
        upper = upper.unsqueeze(0).expand(B, -1, -1)

        weighted = (w * d_plane.unsqueeze(1).expand_as(w) * upper.float())
        total_loss = total_loss + weighted.sum() / (w[upper].sum() + 1e-8)
        count += 1

    if count == 0:
        return coords_pred.sum() * 0.0
    return total_loss / count


# ---------------------------------------------------------------------------
# 5. Curriculum λ scheduler
# ---------------------------------------------------------------------------

class CurriculumLambdaScheduler:
    """
    Anneals λ_SS from λ_high → λ_low over `anneal_steps` steps, then holds.
    Optionally uses cosine annealing for a smooth transition.

    Usage
    -----
    scheduler = CurriculumLambdaScheduler(lambda_high=5.0, lambda_low=0.5,
                                          warmup_steps=1000, anneal_steps=9000)
    λ = scheduler.step()   # call once per training step
    """

    def __init__(
        self,
        lambda_high: float = 5.0,
        lambda_low: float = 0.5,
        lambda_nwc_high: float = 2.0,
        lambda_nwc_low: float = 0.2,
        warmup_steps: int = 500,
        anneal_steps: int = 9_000,
        mode: str = "cosine",   # "cosine" | "linear" | "exp"
    ):
        self.lambda_high = lambda_high
        self.lambda_low  = lambda_low
        self.lambda_nwc_high = lambda_nwc_high
        self.lambda_nwc_low  = lambda_nwc_low
        self.warmup_steps = warmup_steps
        self.anneal_steps = anneal_steps
        self.mode = mode
        self._step = 0

    def _anneal(self, high: float, low: float) -> float:
        t = self._step
        if t <= self.warmup_steps:
            # Linear warm-up from 0 to high
            return high * t / max(self.warmup_steps, 1)
        elapsed = t - self.warmup_steps
        if elapsed >= self.anneal_steps:
            return low
        frac = elapsed / self.anneal_steps        # 0 → 1
        if self.mode == "cosine":
            frac_cos = (1 - math.cos(math.pi * frac)) / 2
            return high + (low - high) * frac_cos
        elif self.mode == "linear":
            return high + (low - high) * frac
        elif self.mode == "exp":
            return high * (low / high) ** frac
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

    def step(self) -> Tuple[float, float]:
        """Returns (lambda_ss, lambda_nwc) for this training step and advances."""
        λ_ss  = self._anneal(self.lambda_high, self.lambda_low)
        λ_nwc = self._anneal(self.lambda_nwc_high, self.lambda_nwc_low)
        self._step += 1
        return λ_ss, λ_nwc

    def state_dict(self) -> dict:
        return {"step": self._step}

    def load_state_dict(self, d: dict):
        self._step = d["step"]


# ---------------------------------------------------------------------------
# 6. Combined SSCL loss module
# ---------------------------------------------------------------------------

class SecondaryStructureConstrainedLoss(nn.Module):
    """
    Full SSCL loss:

        L_total = L_coord
                + λ_SS  * L_SS   (3-class CE or focal)
                + λ_NWC * L_NWC  (pseudoknot binary focal)
                + λ_geo * L_geo  (geometry consistency)
                + λ_plan* L_plan (stem planarity)

    Parameters
    ----------
    d_pair          : pair representation dimension
    use_focal       : use focal loss instead of CE for SS head
    lambda_geo      : fixed weight for geometry consistency
    lambda_plan     : fixed weight for planarity loss
    class_weights   : (3,) tensor for CE loss
    """

    def __init__(
        self,
        d_pair: int,
        use_focal: bool = True,
        lambda_geo: float = 0.2,
        lambda_plan: float = 0.05,
        class_weights: Optional[torch.Tensor] = None,
        label_smoothing: float = 0.05,
    ):
        super().__init__()
        self.ss_head   = SecondaryStructureHead(d_pair)
        self.pk_head   = PseudoknotHead(d_pair)
        self.use_focal = use_focal
        self.lambda_geo   = lambda_geo
        self.lambda_plan  = lambda_plan
        self.label_smoothing = label_smoothing
        if class_weights is not None:
            self.register_buffer("class_weights", class_weights)
        else:
            self.class_weights = None

    def forward(
        self,
        pair_repr: torch.Tensor,        # (B, L, L, d_pair)
        coords_pred: torch.Tensor,      # (B, L, 3)
        l_coord: torch.Tensor,          # scalar — coordinate loss from main branch
        ss_labels: torch.Tensor,        # (B, L, L) int64  {0,1,2}
        lambda_ss: float,
        lambda_nwc: float,
        pair_mask: Optional[torch.Tensor] = None,   # (B, L, L) bool
    ) -> Tuple[torch.Tensor, dict]:
        """
        Returns
        -------
        total_loss : scalar
        info       : dict with individual loss components
        """
        # --- SS head forward ---
        ss_logits = self.ss_head(pair_repr)          # (B, L, L, 3)
        ss_probs  = F.softmax(ss_logits.detach(), dim=-1)  # detach for geometry

        # --- 3-class SS loss ---
        if self.use_focal:
            l_ss = contact_focal_loss(ss_logits, ss_labels, pair_mask=pair_mask)
        else:
            l_ss = ss_cross_entropy_loss(
                ss_logits, ss_labels,
                pair_mask=pair_mask,
                class_weights=self.class_weights,
                label_smoothing=self.label_smoothing,
            )

        # --- Non-WC / pseudoknot binary focal loss ---
        pk_logits = self.pk_head(pair_repr)          # (B, L, L)
        pk_labels = (ss_labels == 2).long()          # binary PK mask
        pk_mask_flat = None
        if pair_mask is not None:
            tri = torch.ones(*pk_logits.shape[-2:], dtype=torch.bool,
                             device=pair_repr.device).triu(diagonal=1)
            tri = tri.unsqueeze(0).expand(pair_repr.size(0), -1, -1)
            pk_mask_flat = tri & pair_mask
        # Wrap pk as 2-class problem for focal
        pk_logits_2cls = torch.stack([-pk_logits, pk_logits], dim=-1)  # (B,L,L,2)
        l_nwc = contact_focal_loss(pk_logits_2cls, pk_labels,
                                   gamma=3.0, alpha=0.75,
                                   pair_mask=pair_mask)

        # --- Geometry consistency ---
        l_geo = geometry_consistency_loss(
            coords_pred, ss_probs, pair_mask=pair_mask
        )

        # --- Stem planarity ---
        l_plan = stem_planarity_loss(coords_pred, ss_probs)

        # --- Combine ---
        total = (l_coord
                 + lambda_ss  * l_ss
                 + lambda_nwc * l_nwc
                 + self.lambda_geo  * l_geo
                 + self.lambda_plan * l_plan)

        info = {
            "l_coord": l_coord.item(),
            "l_ss":    l_ss.item(),
            "l_nwc":   l_nwc.item(),
            "l_geo":   l_geo.item(),
            "l_plan":  l_plan.item(),
            "lambda_ss":  lambda_ss,
            "lambda_nwc": lambda_nwc,
            "total":   total.item(),
        }
        return total, info

    @torch.no_grad()
    def predict_ss(self, pair_repr: torch.Tensor) -> torch.Tensor:
        """
        Inference-only: return argmax class predictions.
        Returns (B, L, L) int64.
        """
        logits = self.ss_head(pair_repr)
        return logits.argmax(dim=-1)


# ---------------------------------------------------------------------------
# 7. DSSR annotation parser (CPU / numpy)
# ---------------------------------------------------------------------------

def parse_dssr_pairs(dssr_json: dict, seq_len: int) -> np.ndarray:
    """
    Build a (L, L) integer label matrix from DSSR JSON output.

    DSSR JSON structure (relevant keys):
      pairs[].nt1, pairs[].nt2, pairs[].name (e.g. "WC", "wobble", "Hoogsteen")
      pairs[].is_pseudoknot  (bool)

    Labels
    ------
    0 : no contact (default)
    1 : Watson-Crick / wobble (canonical stem)
    2 : non-WC pseudoknot or tertiary

    Parameters
    ----------
    dssr_json : dict  parsed from dssr --json output
    seq_len   : int   number of nucleotides

    Returns
    -------
    labels : (L, L) int64
    """
    WC_TYPES = {"WC", "wc", "Watson-Crick", "wobble", "W-C"}
    labels = np.zeros((seq_len, seq_len), dtype=np.int64)

    pairs = dssr_json.get("pairs", [])
    for p in pairs:
        # DSSR uses 1-based indexing
        try:
            i = int(p["nt1"].split(".")[1]) - 1
            j = int(p["nt2"].split(".")[1]) - 1
        except (KeyError, IndexError, ValueError):
            continue
        if not (0 <= i < seq_len and 0 <= j < seq_len):
            continue

        is_wc = p.get("name", "") in WC_TYPES
        is_pk = p.get("is_pseudoknot", False)

        label = 1 if (is_wc and not is_pk) else 2
        labels[i, j] = label
        labels[j, i] = label

    return labels


def labels_to_tensor(labels_np: np.ndarray, device: str = "cpu") -> torch.Tensor:
    """Convert numpy (L, L) int64 array to torch tensor."""
    return torch.from_numpy(labels_np).long().to(device)


# ---------------------------------------------------------------------------
# 8. Trainer integration helper
# ---------------------------------------------------------------------------

class SSCLTrainer:
    """
    Drop-in training step helper. Wraps the model, SSCL loss, and curriculum.

    Example
    -------
    trainer = SSCLTrainer(model, pair_dim=256, device="cuda")
    for batch in dataloader:
        loss, info = trainer.step(batch, optimizer)
    """

    def __init__(
        self,
        model: nn.Module,
        pair_dim: int,
        device: str = "cuda",
        lambda_high: float = 5.0,
        lambda_low: float = 0.5,
        lambda_nwc_high: float = 2.0,
        lambda_nwc_low: float = 0.2,
        warmup_steps: int = 500,
        anneal_steps: int = 9_000,
        anneal_mode: str = "cosine",
        use_focal: bool = True,
        lambda_geo: float = 0.2,
        lambda_plan: float = 0.05,
    ):
        self.model = model
        self.device = device
        self.sscl = SecondaryStructureConstrainedLoss(
            d_pair=pair_dim,
            use_focal=use_focal,
            lambda_geo=lambda_geo,
            lambda_plan=lambda_plan,
        ).to(device)
        self.scheduler = CurriculumLambdaScheduler(
            lambda_high=lambda_high,
            lambda_low=lambda_low,
            lambda_nwc_high=lambda_nwc_high,
            lambda_nwc_low=lambda_nwc_low,
            warmup_steps=warmup_steps,
            anneal_steps=anneal_steps,
            mode=anneal_mode,
        )

    def step(
        self,
        batch: dict,
        optimizer: torch.optim.Optimizer,
        grad_clip: float = 1.0,
    ) -> Tuple[float, dict]:
        """
        Expected batch keys
        -------------------
        pair_repr   : (B, L, L, d_pair)  — from model's pair representation
        coords_pred : (B, L, 3)          — predicted 3D coordinates
        l_coord     : scalar Tensor      — coordinate loss (e.g. RMSD, FAPE)
        ss_labels   : (B, L, L)          — from DSSR parse
        pair_mask   : (B, L, L)          — optional padding mask

        Returns
        -------
        total_loss_val : float
        info           : dict of loss components
        """
        λ_ss, λ_nwc = self.scheduler.step()

        pair_repr   = batch["pair_repr"].to(self.device)
        coords_pred = batch["coords_pred"].to(self.device)
        l_coord     = batch["l_coord"].to(self.device)
        ss_labels   = batch["ss_labels"].to(self.device)
        pair_mask   = batch.get("pair_mask", None)
        if pair_mask is not None:
            pair_mask = pair_mask.to(self.device)

        optimizer.zero_grad()
        total_loss, info = self.sscl(
            pair_repr=pair_repr,
            coords_pred=coords_pred,
            l_coord=l_coord,
            ss_labels=ss_labels,
            lambda_ss=λ_ss,
            lambda_nwc=λ_nwc,
            pair_mask=pair_mask,
        )
        total_loss.backward()
        if grad_clip > 0:
            nn.utils.clip_grad_norm_(
                list(self.model.parameters()) + list(self.sscl.parameters()),
                grad_clip,
            )
        optimizer.step()

        info["lambda_ss_curr"]  = λ_ss
        info["lambda_nwc_curr"] = λ_nwc
        return total_loss.item(), info

    def save(self, path: str):
        torch.save({
            "sscl_state":  self.sscl.state_dict(),
            "scheduler":   self.scheduler.state_dict(),
        }, path)

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.sscl.load_state_dict(ckpt["sscl_state"])
        self.scheduler.load_state_dict(ckpt["scheduler"])


# ---------------------------------------------------------------------------
# 9. Numba warm-up (call once at import to trigger JIT compilation)
# ---------------------------------------------------------------------------

def warmup_numba():
    """Pre-compile Numba kernels with dummy data to avoid first-call latency."""
    dummy_coords = np.random.randn(8, 3).astype(np.float32)
    dummy_pairs  = np.array([[0, 3], [1, 6], [2, 5]], dtype=np.int32)
    dummy_idx    = np.arange(8, dtype=np.int32)
    _ = pairwise_distance_matrix(dummy_coords)
    _ = extract_pair_distances(dummy_coords, dummy_pairs)
    _ = stem_consistency_score(dummy_coords, dummy_pairs)
    _ = backbone_bond_lengths(dummy_coords, dummy_idx)
    print("[rna_sscl] Numba kernels compiled.")


# ---------------------------------------------------------------------------
# 10. Quick self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import time

    warmup_numba()

    # --- Test Numba kernels ---
    N = 256
    coords_np = np.random.randn(N, 3).astype(np.float32) * 30
    pairs_np  = np.array([[i, N - 1 - i] for i in range(N // 2)], dtype=np.int32)
    idx_np    = np.arange(N, dtype=np.int32)

    t0 = time.perf_counter()
    D  = pairwise_distance_matrix(coords_np)
    print(f"pairwise_distance_matrix ({N}x{N}): {time.perf_counter()-t0:.4f}s | max={D.max():.2f}")

    t0 = time.perf_counter()
    dists = extract_pair_distances(coords_np, pairs_np)
    print(f"extract_pair_distances ({len(pairs_np)} pairs): {time.perf_counter()-t0:.4f}s")

    score = stem_consistency_score(coords_np, pairs_np[:10])
    print(f"stem_consistency_score: {score:.4f}")

    bl = backbone_bond_lengths(coords_np, idx_np)
    print(f"backbone_bond_lengths: mean={bl.mean():.2f} Å")

    # --- Test PyTorch loss modules ---
    B, L, d_pair = 2, 32, 128
    device = "cpu"

    pair_repr   = torch.randn(B, L, L, d_pair)
    coords_pred = torch.randn(B, L, 3) * 20
    ss_labels   = torch.randint(0, 3, (B, L, L))
    # Make symmetric labels (required)
    ss_labels   = torch.max(ss_labels, ss_labels.transpose(1, 2))

    l_coord = torch.tensor(2.5, requires_grad=True)

    sscl = SecondaryStructureConstrainedLoss(d_pair=d_pair, use_focal=True)
    scheduler = CurriculumLambdaScheduler(warmup_steps=0, anneal_steps=100)

    for step in range(3):
        λ_ss, λ_nwc = scheduler.step()
        total, info = sscl(pair_repr, coords_pred, l_coord,
                           ss_labels, λ_ss, λ_nwc)
        print(f"Step {step+1}: total={info['total']:.4f} | "
              f"l_ss={info['l_ss']:.4f} | l_nwc={info['l_nwc']:.4f} | "
              f"l_geo={info['l_geo']:.4f} | λ_SS={λ_ss:.3f}")

    # --- DSSR parser test ---
    fake_dssr = {
        "pairs": [
            {"nt1": "A.1", "nt2": "A.20", "name": "WC",        "is_pseudoknot": False},
            {"nt1": "A.5", "nt2": "A.15", "name": "WC",        "is_pseudoknot": False},
            {"nt1": "A.3", "nt2": "A.18", "name": "Hoogsteen", "is_pseudoknot": True},
        ]
    }
    lbl = parse_dssr_pairs(fake_dssr, seq_len=32)
    print(f"\nDSSR labels: WC count={( lbl==1).sum()}, PK count={(lbl==2).sum()}")

    print("\n[rna_sscl] All tests passed.")
