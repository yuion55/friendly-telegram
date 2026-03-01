"""Cross-domain attention module for domain assembly."""

import torch
import torch.nn as nn


class CrossDomainAttention(nn.Module):
    def __init__(self, d_pair: int = 128, n_heads: int = 4):
        super().__init__()
        self.d_pair = d_pair
        self.attn = nn.MultiheadAttention(embed_dim=d_pair, num_heads=n_heads, batch_first=False)
        self.layer_norm = nn.LayerNorm(d_pair)

    def forward(self, pair_repr: torch.Tensor, domain_mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pair_repr: (B, L, L, d_pair) pairwise representation.
            domain_mask: (B, L, L) bool mask where True indicates cross-domain entries.
        Returns:
            Updated pair_repr of shape (B, L, L, d_pair).
        """
        B, L, _, d = pair_repr.shape

        # Reshape to (B, L*L, d)
        x = pair_repr.reshape(B, L * L, d)

        # Zero out non-cross-domain entries via broadcasting: mask is (B, L*L, 1)
        flat_mask = domain_mask.reshape(B, L * L, 1).to(dtype=torch.float32)
        x = x * flat_mask

        # nn.MultiheadAttention expects (seq_len, batch, embed_dim)
        x = x.permute(1, 0, 2)  # (L*L, B, d)

        attn_out, _ = self.attn(x, x, x)  # (L*L, B, d)

        # Transpose back to (B, L*L, d)
        attn_out = attn_out.permute(1, 0, 2)

        # Residual connection + LayerNorm
        residual = pair_repr.reshape(B, L * L, d)
        out = self.layer_norm(residual + attn_out)

        return out.reshape(B, L, L, d)


if __name__ == "__main__":
    B, L, d_pair, n_heads = 2, 50, 128, 4

    torch.manual_seed(42)
    pair_repr = torch.randn(B, L, L, d_pair, dtype=torch.float32)
    domain_mask = torch.rand(B, L, L) > 0.5  # random bool mask

    model = CrossDomainAttention(d_pair=d_pair, n_heads=n_heads)
    out = model(pair_repr, domain_mask)

    assert out.shape == (B, L, L, d_pair), f"Expected shape {(B, L, L, d_pair)}, got {out.shape}"
    assert out.dtype == torch.float32, f"Expected float32, got {out.dtype}"
    assert torch.isfinite(out).all(), "Output contains non-finite values"
    print(f"Self-test passed: output shape {out.shape}, dtype {out.dtype}")
