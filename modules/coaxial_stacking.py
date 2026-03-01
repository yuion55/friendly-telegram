"""
coaxial_stacking.py
===================
Coaxial stacking detection and scoring for RNA helices.

Identifies pairs of RNA helices that stack end-to-end (coaxial stacking) by
measuring the distance between their terminal base-pair midpoints, and provides
a learned scoring head that predicts coaxial stacking from pair representations.

Functions:
  label_coaxial_stacks — Numba-parallel labelling of helix pairs by end distance

Class:
  CoaxialStackingHead  — PyTorch head: pair features → stacking logits + BCE loss

Usage:
  from modules.coaxial_stacking import label_coaxial_stacks, CoaxialStackingHead
  labels = label_coaxial_stacks(helix_ends, coords)          # (H, H) bool
  head   = CoaxialStackingHead(d_pair=128)
  logits = head(pair_repr, helix_ends_list)                   # list of (H_b, H_b)
"""

from __future__ import annotations

import numpy as np
from numba import njit, prange
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# 1. LABEL COAXIAL STACKS  (parallel — each row written by one thread)
# ---------------------------------------------------------------------------

@njit(cache=True, parallel=True)
def label_coaxial_stacks(
    helix_ends: np.ndarray,
    coords: np.ndarray,
    dist_thresh: np.float32 = np.float32(3.5),
) -> np.ndarray:
    """
    Flag helix pairs whose stacking-end midpoints are within *dist_thresh*.

    Each helix has two terminal base-pair ends:

    * **End A** — midpoint of residues at ``5p_end`` and ``3p_start``
    * **End B** — midpoint of residues at ``5p_start`` and ``3p_end``

    For every ordered pair *(i, j)* the minimum distance across the four
    end-to-end combinations is compared against *dist_thresh*.

    Parameters
    ----------
    helix_ends : ndarray, shape (H, 4), int
        Columns are ``(h_5p_start, h_5p_end, h_3p_start, h_3p_end)``.
    coords : ndarray, shape (L, 3), float32
        Residue coordinates.
    dist_thresh : float32
        Distance cutoff in Å (default 3.5).

    Returns
    -------
    stacking : ndarray, shape (H, H), bool
        ``True`` where the helix pair is flagged as coaxially stacked.
    """
    H = helix_ends.shape[0]
    result = np.zeros((H, H), dtype=np.bool_)
    thresh_sq = np.float32(dist_thresh * dist_thresh)

    half = np.float32(0.5)

    for i in prange(H):
        # End A of helix i: midpoint of (5p_end, 3p_start)
        ai_x = (coords[helix_ends[i, 1], 0] + coords[helix_ends[i, 2], 0]) * half
        ai_y = (coords[helix_ends[i, 1], 1] + coords[helix_ends[i, 2], 1]) * half
        ai_z = (coords[helix_ends[i, 1], 2] + coords[helix_ends[i, 2], 2]) * half
        # End B of helix i: midpoint of (5p_start, 3p_end)
        bi_x = (coords[helix_ends[i, 0], 0] + coords[helix_ends[i, 3], 0]) * half
        bi_y = (coords[helix_ends[i, 0], 1] + coords[helix_ends[i, 3], 1]) * half
        bi_z = (coords[helix_ends[i, 0], 2] + coords[helix_ends[i, 3], 2]) * half

        for j in range(H):
            if i == j:
                continue
            # End A of helix j
            aj_x = (coords[helix_ends[j, 1], 0] + coords[helix_ends[j, 2], 0]) * half
            aj_y = (coords[helix_ends[j, 1], 1] + coords[helix_ends[j, 2], 1]) * half
            aj_z = (coords[helix_ends[j, 1], 2] + coords[helix_ends[j, 2], 2]) * half
            # End B of helix j
            bj_x = (coords[helix_ends[j, 0], 0] + coords[helix_ends[j, 3], 0]) * half
            bj_y = (coords[helix_ends[j, 0], 1] + coords[helix_ends[j, 3], 1]) * half
            bj_z = (coords[helix_ends[j, 0], 2] + coords[helix_ends[j, 3], 2]) * half

            # Squared distances for four end-end combinations
            d_aa = (ai_x - aj_x) ** 2 + (ai_y - aj_y) ** 2 + (ai_z - aj_z) ** 2
            min_sq = d_aa

            d_ab = (ai_x - bj_x) ** 2 + (ai_y - bj_y) ** 2 + (ai_z - bj_z) ** 2
            if d_ab < min_sq:
                min_sq = d_ab

            d_ba = (bi_x - aj_x) ** 2 + (bi_y - aj_y) ** 2 + (bi_z - aj_z) ** 2
            if d_ba < min_sq:
                min_sq = d_ba

            d_bb = (bi_x - bj_x) ** 2 + (bi_y - bj_y) ** 2 + (bi_z - bj_z) ** 2
            if d_bb < min_sq:
                min_sq = d_bb

            if min_sq < thresh_sq:
                result[i, j] = True

    return result


# ---------------------------------------------------------------------------
# 2. COAXIAL STACKING HEAD  (PyTorch)
# ---------------------------------------------------------------------------

class CoaxialStackingHead(nn.Module):
    """Predicts coaxial stacking between helix pairs from pair representations.

    For each helix pair *(i, j)* the pair representation is looked up at the
    two stacking-end index pairs (``5p_end``, ``3p_start``), concatenated, and
    fed through a small MLP.

    Args:
        d_pair: Dimension of the input pair representation.
    """

    def __init__(self, d_pair: int = 128):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(2 * d_pair, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(
        self,
        pair_repr: torch.Tensor,
        helix_ends_list: list[np.ndarray],
    ) -> list[torch.Tensor]:
        """Forward pass.

        Args:
            pair_repr: (B, L, L, d_pair) float32 pair representation.
            helix_ends_list: list of int arrays, each (H_b, 4), one per
                batch item.  Columns are
                ``(h_5p_start, h_5p_end, h_3p_start, h_3p_end)``.

        Returns:
            List of logit tensors, one per batch item, each (H_b, H_b).
        """
        device = pair_repr.device
        B = pair_repr.shape[0]
        logits_list: list[torch.Tensor] = []

        for b in range(B):
            ends = helix_ends_list[b]
            H_b = ends.shape[0]
            if H_b == 0:
                logits_list.append(
                    torch.zeros(0, 0, dtype=torch.float32, device=device),
                )
                continue

            ends_t = torch.as_tensor(ends, dtype=torch.long, device=device)
            # Stacking-end indices for each helix
            idx_5p_end = ends_t[:, 1]   # (H_b,)
            idx_3p_start = ends_t[:, 2]  # (H_b,)

            # Build all-pairs index grids
            ii, jj = torch.meshgrid(
                torch.arange(H_b, device=device),
                torch.arange(H_b, device=device),
                indexing="ij",
            )
            ii_flat = ii.reshape(-1)
            jj_flat = jj.reshape(-1)

            # feat1: pair_repr at (5p_end_i, 5p_end_j) — end-A cross-reference
            row1 = idx_5p_end[ii_flat]
            col1 = idx_5p_end[jj_flat]
            feat1 = pair_repr[b, row1, col1]  # (H_b*H_b, d_pair)

            # feat2: pair_repr at (3p_start_i, 3p_start_j) — end-A complement
            row2 = idx_3p_start[ii_flat]
            col2 = idx_3p_start[jj_flat]
            feat2 = pair_repr[b, row2, col2]  # (H_b*H_b, d_pair)

            feats = torch.cat([feat1, feat2], dim=-1)  # (H_b*H_b, 2*d_pair)
            logits_list.append(self.proj(feats).squeeze(-1).reshape(H_b, H_b))

        return logits_list

    def loss(
        self,
        logits_list: list[torch.Tensor],
        labels_list: list[torch.Tensor],
    ) -> torch.Tensor:
        """Binary cross-entropy loss averaged over batch items.

        Args:
            logits_list: list of (H_b, H_b) logit tensors.
            labels_list: list of (H_b, H_b) float32 binary label tensors.

        Returns:
            Scalar loss.
        """
        losses: list[torch.Tensor] = []
        for logits, labels in zip(logits_list, labels_list):
            if logits.numel() == 0:
                continue
            target = labels.to(dtype=torch.float32, device=logits.device)
            losses.append(
                nn.functional.binary_cross_entropy_with_logits(logits, target),
            )
        if not losses:
            return torch.tensor(0.0, requires_grad=True)
        return torch.stack(losses).mean()


# ---------------------------------------------------------------------------
# Self-test on synthetic data
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    rng = np.random.RandomState(42)
    L = 50

    coords = rng.randn(L, 3).astype(np.float32)

    # Place helix 0 end-A midpoint close to helix 1 end-B midpoint by
    # arranging their stacking-end residues within ~2 Å of each other.
    coords[4] = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    coords[45] = np.array([0.5, 0.0, 0.0], dtype=np.float32)
    coords[5] = np.array([0.2, 0.1, 0.0], dtype=np.float32)
    coords[44] = np.array([0.3, -0.1, 0.0], dtype=np.float32)

    helix_ends = np.array(
        [
            [0, 4, 45, 49],
            [5, 9, 40, 44],
            [10, 14, 35, 39],
            [15, 19, 30, 34],
        ],
        dtype=np.int64,
    )

    # --- Test label_coaxial_stacks ---
    labels = label_coaxial_stacks(helix_ends, coords, np.float32(3.5))
    assert labels.shape == (4, 4), f"Bad shape: {labels.shape}"
    assert labels.dtype == np.bool_, f"Bad dtype: {labels.dtype}"
    assert not np.any(np.diag(labels)), "Diagonal must be False"
    # Engineered pair (0, 1) and (1, 0) should be flagged
    assert labels[0, 1], "Expected helix 0-1 stacking"
    assert labels[1, 0], "Expected helix 1-0 stacking"
    print(f"label_coaxial_stacks: shape={labels.shape}, dtype={labels.dtype}")
    print(f"  stacking pairs: {np.argwhere(labels).tolist()}")

    # --- Test CoaxialStackingHead ---
    B, d_pair = 2, 128
    pair_repr = torch.randn(B, L, L, d_pair, dtype=torch.float32)
    helix_ends_list = [helix_ends, helix_ends[:3]]

    head = CoaxialStackingHead(d_pair=d_pair)
    logits = head(pair_repr, helix_ends_list)

    assert len(logits) == B
    assert logits[0].shape == (4, 4), f"Unexpected shape: {logits[0].shape}"
    assert logits[1].shape == (3, 3), f"Unexpected shape: {logits[1].shape}"
    assert logits[0].dtype == torch.float32

    # --- Test loss + backward ---
    labels_list = [
        torch.from_numpy(labels.astype(np.float32)),
        torch.zeros(3, 3, dtype=torch.float32),
    ]
    loss_val = head.loss(logits, labels_list)
    assert loss_val.shape == (), f"Loss should be scalar, got {loss_val.shape}"
    assert torch.isfinite(loss_val), f"Loss is not finite: {loss_val}"

    loss_val.backward()
    for name, p in head.named_parameters():
        assert p.grad is not None, f"No gradient for {name}"

    print(f"CoaxialStackingHead: logits shapes={[l.shape for l in logits]}")
    print(f"  loss={loss_val.item():.4f}")
    print("All checks passed.")
