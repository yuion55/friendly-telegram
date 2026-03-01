"""
ion_prediction.py
=================
Magnesium ion site prediction and geometry penalty for RNA structures.

Scores predicted 3-D coordinates against known Mg²⁺ binding sites using a
Numba-accelerated geometry penalty (ideal Mg–P distance 2.1 Å, coordination
number 6), and provides a learned per-residue head that predicts Mg²⁺ binding
probability from single representations.

Functions:
  mg_geometry_penalty — Numba-parallel penalty for Mg–P distance and coordination

Class:
  IonSiteHead         — PyTorch head: single features → per-residue Mg probability

Usage:
  from modules.ion_prediction import mg_geometry_penalty, IonSiteHead
  penalty = mg_geometry_penalty(pred_coords, mg_sites)
  head    = IonSiteHead(d_single=256)
  probs   = head(single_repr)                         # (B, L, 1)
"""

from __future__ import annotations

import numpy as np
from numba import njit, prange
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# 1. MG GEOMETRY PENALTY  (parallel — each Mg site written by one thread)
# ---------------------------------------------------------------------------

@njit(cache=True, parallel=True)
def mg_geometry_penalty(
    pred_coords: np.ndarray,
    mg_sites: np.ndarray,
) -> np.float32:
    """
    Harmonic geometry penalty for Mg²⁺ sites against predicted P-atom coords.

    For every Mg site the function finds all P atoms (residues in
    *pred_coords*) within 4 Å, applies a harmonic penalty toward the ideal
    Mg–P distance of 2.1 Å, and penalises deviation of the coordination
    number from the ideal value of 6.

    Parameters
    ----------
    pred_coords : ndarray, shape (L, 3), float32
        Predicted residue (P-atom) coordinates.
    mg_sites : ndarray, shape (M, 3), float32
        Known Mg²⁺ ion positions.

    Returns
    -------
    penalty : float32
        Total penalty summed over all Mg sites.
    """
    M = mg_sites.shape[0]
    L = pred_coords.shape[0]
    ideal_dist = np.float32(2.1)
    cutoff_sq = np.float32(16.0)  # 4.0 ** 2
    ideal_coord = np.float32(6.0)

    # Per-thread partial penalties accumulated into an array (no write conflicts)
    penalties = np.zeros(M, dtype=np.float32)

    for m in prange(M):
        site_penalty = np.float32(0.0)
        coord_count = np.float32(0.0)

        mx = mg_sites[m, 0]
        my = mg_sites[m, 1]
        mz = mg_sites[m, 2]

        for i in range(L):
            dx = pred_coords[i, 0] - mx
            dy = pred_coords[i, 1] - my
            dz = pred_coords[i, 2] - mz
            dist_sq = dx * dx + dy * dy + dz * dz

            if dist_sq < cutoff_sq:
                dist = np.sqrt(dist_sq)
                diff = dist - ideal_dist
                site_penalty += diff * diff
                coord_count += np.float32(1.0)

        # Penalise coordination number deviation from 6
        coord_diff = coord_count - ideal_coord
        site_penalty += coord_diff * coord_diff

        penalties[m] = site_penalty

    total = np.float32(0.0)
    for m in range(M):
        total += penalties[m]
    return total


# ---------------------------------------------------------------------------
# 2. ION SITE HEAD  (PyTorch)
# ---------------------------------------------------------------------------

class IonSiteHead(nn.Module):
    """Per-residue Mg²⁺ binding-site probability head.

    Architecture: linear(d_single → 64) → ReLU → linear(64 → 1) → sigmoid.

    Args:
        d_single: Dimension of the input single representation.
    """

    def __init__(self, d_single: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_single, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, single_repr: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            single_repr: (B, L, d_single) float32 single representation.

        Returns:
            probs: (B, L, 1) per-residue Mg binding probability in (0, 1).
        """
        return torch.sigmoid(self.net(single_repr))

    @staticmethod
    def loss(
        probs: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """Binary cross-entropy loss for ion-site prediction.

        Args:
            probs: (B, L, 1) predicted probabilities.
            targets: (B, L) float32 binary labels (1 = Mg site).

        Returns:
            Scalar loss.
        """
        return nn.functional.binary_cross_entropy(
            probs.squeeze(-1), targets.to(dtype=torch.float32),
        )


# ---------------------------------------------------------------------------
# Self-test on synthetic data
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    rng = np.random.RandomState(42)
    L = 50

    pred_coords = rng.randn(L, 3).astype(np.float32)

    # Place two Mg sites near some residues
    mg_sites = np.array([
        pred_coords[5] + np.array([2.0, 0.0, 0.0], dtype=np.float32),
        pred_coords[20] + np.array([0.0, 1.8, 0.0], dtype=np.float32),
    ], dtype=np.float32)
    M = mg_sites.shape[0]

    # --- Test mg_geometry_penalty ---
    penalty = mg_geometry_penalty(pred_coords, mg_sites)
    assert isinstance(penalty, (float, np.floating)), (
        f"Expected float, got {type(penalty)}"
    )
    assert np.isfinite(penalty), f"Penalty is not finite: {penalty}"
    assert penalty >= 0.0, f"Penalty must be non-negative, got {penalty}"
    print(f"mg_geometry_penalty: penalty={penalty:.4f}")

    # Verify penalty is zero when perfectly coordinated: 6 atoms at 2.1 Å
    center = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)
    ideal_coords = np.array([
        [2.1, 0.0, 0.0],
        [-2.1, 0.0, 0.0],
        [0.0, 2.1, 0.0],
        [0.0, -2.1, 0.0],
        [0.0, 0.0, 2.1],
        [0.0, 0.0, -2.1],
    ], dtype=np.float32)
    perfect_penalty = mg_geometry_penalty(ideal_coords, center)
    assert perfect_penalty < 1e-6, (
        f"Perfect octahedral should give ~0 penalty, got {perfect_penalty}"
    )
    print(f"  perfect octahedral penalty={perfect_penalty:.6f}")

    # --- Test IonSiteHead ---
    B, d_single = 2, 256
    single_repr = torch.randn(B, L, d_single, dtype=torch.float32)
    targets = torch.zeros(B, L, dtype=torch.float32)
    targets[0, 5] = 1.0
    targets[0, 20] = 1.0
    targets[1, 10] = 1.0

    head = IonSiteHead(d_single=d_single)
    probs = head(single_repr)

    assert probs.shape == (B, L, 1), f"Unexpected shape: {probs.shape}"
    assert probs.dtype == torch.float32
    assert (probs >= 0.0).all() and (probs <= 1.0).all(), "Probs out of [0, 1]"

    # --- Test loss + backward ---
    loss_val = head.loss(probs, targets)
    assert loss_val.shape == (), f"Loss should be scalar, got {loss_val.shape}"
    assert torch.isfinite(loss_val), f"Loss is not finite: {loss_val}"

    loss_val.backward()
    for name, p in head.named_parameters():
        assert p.grad is not None, f"No gradient for {name}"

    print(f"IonSiteHead: probs shape={probs.shape}, loss={loss_val.item():.4f}")
    print("All checks passed.")
