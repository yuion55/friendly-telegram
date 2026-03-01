"""Non-Watson-Crick contact classification head.

Six NWC classes: WC, Hoogsteen, Sugar-edge, A-minor, Ribose-zipper, Other.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

NWC_CLASSES = ["WC", "Hoogsteen", "Sugar-edge", "A-minor", "Ribose-zipper", "Other"]
NUM_NWC_CLASSES = len(NWC_CLASSES)


class NonWCContactHead(nn.Module):
    """Classifies pairwise RNA contacts into six NWC interaction types.

    Args:
        d_pair: Dimension of the input pair representation.
    """

    def __init__(self, d_pair: int = 128):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(d_pair, 64),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(64, NUM_NWC_CLASSES),
        )

    def forward(self, pair_repr: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            pair_repr: (B, L, L, d_pair) float32 pair representation.

        Returns:
            logits: (B, L, L, 6) classification logits.
        """
        return self.proj(pair_repr)

    @staticmethod
    def loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Cross-entropy loss over flattened spatial dims.

        Args:
            logits: (B, L, L, 6) predicted logits.
            targets: (B, L, L) integer class labels; -1 is ignored.

        Returns:
            Scalar loss.
        """
        flat_logits = logits.reshape(-1, NUM_NWC_CLASSES)
        flat_targets = targets.reshape(-1)
        return F.cross_entropy(flat_logits, flat_targets, ignore_index=-1)


if __name__ == "__main__":
    B, L, d_pair = 2, 50, 128

    pair_repr = torch.randn(B, L, L, d_pair, dtype=torch.float32)
    targets = torch.randint(0, NUM_NWC_CLASSES, (B, L, L))
    # Sprinkle some ignore tokens
    mask = torch.rand(B, L, L) < 0.1
    targets[mask] = -1

    head = NonWCContactHead(d_pair=d_pair)
    logits = head(pair_repr)
    assert logits.shape == (B, L, L, NUM_NWC_CLASSES), f"Unexpected shape: {logits.shape}"
    assert logits.dtype == torch.float32

    loss_val = head.loss(logits, targets)
    assert loss_val.shape == (), f"Loss should be scalar, got {loss_val.shape}"
    assert torch.isfinite(loss_val), f"Loss is not finite: {loss_val}"

    loss_val.backward()
    for name, p in head.named_parameters():
        assert p.grad is not None, f"No gradient for {name}"

    print(f"logits shape: {logits.shape}")
    print(f"loss: {loss_val.item():.4f}")
    print("All checks passed.")
