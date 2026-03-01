"""Secondary-structure base-pair probability supervision head."""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class SecondaryStructureHead(nn.Module):
    """Predicts a symmetric base-pair probability matrix from pair representations.

    Parameters
    ----------
    d_pair : int
        Dimension of the incoming pair representation (last axis).
    """

    def __init__(self, d_pair: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_pair, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(self, pair_repr: torch.Tensor) -> torch.Tensor:
        """Produce symmetric base-pair probabilities.

        Parameters
        ----------
        pair_repr : Tensor, shape (B, L, L, d_pair), float32

        Returns
        -------
        bpp : Tensor, shape (B, L, L), float32  – values in (0, 1)
        """
        logits = self.net(pair_repr).squeeze(-1)  # (B, L, L)
        # Symmetrise by averaging with transpose (no Python loops)
        logits = 0.5 * (logits + logits.transpose(-1, -2))
        return torch.sigmoid(logits)

    # ------------------------------------------------------------------
    # Loss
    # ------------------------------------------------------------------
    @staticmethod
    def loss(
        pred_bpp: torch.Tensor,
        true_bpp: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Masked binary cross-entropy loss.

        Parameters
        ----------
        pred_bpp : Tensor (B, L, L), float32 – predicted probabilities
        true_bpp : Tensor (B, L, L), float32 – ground-truth 0/1 matrix
        mask     : Tensor (B, L, L), float32, optional – 1 where valid

        Returns
        -------
        Scalar loss (mean over valid positions).
        """
        bce = F.binary_cross_entropy(pred_bpp, true_bpp, reduction="none")
        if mask is not None:
            bce = bce * mask
            return bce.sum() / mask.sum().clamp(min=1.0)
        return bce.mean()


# ======================================================================
# Self-tests
# ======================================================================
def _test_secondary_structure_head() -> None:
    """Validate SecondaryStructureHead on synthetic data."""
    B, L, D = 2, 50, 128
    torch.manual_seed(0)

    pair_repr = torch.randn(B, L, L, D, dtype=torch.float32)
    model = SecondaryStructureHead(d_pair=D)

    bpp = model(pair_repr)
    assert bpp.shape == (B, L, L), f"Expected shape {(B, L, L)}, got {bpp.shape}"
    assert bpp.dtype == torch.float32, f"Expected float32, got {bpp.dtype}"
    assert (bpp >= 0.0).all() and (bpp <= 1.0).all(), "Values must be in [0, 1]"

    # Symmetry check (no loops – pure broadcasting)
    diff = (bpp - bpp.transpose(-1, -2)).abs()
    assert diff.max().item() < 1e-6, f"Not symmetric: max diff = {diff.max().item()}"

    # Loss without mask
    true_bpp = torch.from_numpy(
        np.random.RandomState(42).rand(B, L, L).astype(np.float32)
    )
    true_bpp = 0.5 * (true_bpp + true_bpp.transpose(1, 2))  # symmetric target
    loss_val = SecondaryStructureHead.loss(bpp, true_bpp)
    assert loss_val.shape == (), "Loss must be scalar"
    assert loss_val.item() > 0.0, "Loss must be positive"

    # Loss with mask (broadcasting, no loops)
    mask = torch.from_numpy(
        (np.random.RandomState(7).rand(B, L, L) > 0.3).astype(np.float32)
    )
    loss_masked = SecondaryStructureHead.loss(bpp, true_bpp, mask=mask)
    assert loss_masked.shape == (), "Masked loss must be scalar"
    assert loss_masked.item() > 0.0, "Masked loss must be positive"

    # Backward pass
    loss_masked.backward()
    grad_norms = [
        p.grad.norm().item() for p in model.parameters() if p.grad is not None
    ]
    assert all(g > 0.0 for g in grad_norms), "All gradients must be non-zero"

    print("SecondaryStructureHead: all tests passed.")


if __name__ == "__main__":
    _test_secondary_structure_head()
