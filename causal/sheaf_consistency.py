"""Sheaf GNN consistency layer for multi-scale RNA structure."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SheafGNNLayer(nn.Module):
    """Multi-scale sheaf GNN layer with learnable restriction maps."""

    def __init__(self, d_feat=64, n_scales=4):
        super().__init__()
        self.d_feat = d_feat
        self.n_scales = n_scales
        self.restriction_maps = nn.ModuleList(
            [nn.Linear(d_feat, d_feat, bias=False) for _ in range(n_scales - 1)]
        )
        self.node_update_mlps = nn.ModuleList(
            [nn.Sequential(
                nn.Linear(d_feat * 2, d_feat),
                nn.LayerNorm(d_feat),
                nn.GELU(),
            ) for _ in range(n_scales)]
        )

    def forward(self, scale_features):
        """Forward pass.

        Args:
            scale_features: list of (B, N_s, d_feat) tensors, coarsest first.

        Returns:
            Updated list of (B, N_s, d_feat) tensors.
        """
        out = [None] * self.n_scales
        out[0] = self.node_update_mlps[0](
            torch.cat([scale_features[0], scale_features[0]], dim=-1)
        )
        for s in range(1, self.n_scales):
            coarse = scale_features[s - 1]
            fine = scale_features[s]
            mapped = self.restriction_maps[s - 1](coarse)
            B, N_fine, d = fine.shape
            N_coarse = mapped.shape[1]
            idx = torch.linspace(0, N_coarse - 1, N_fine, device=fine.device)
            idx_lo = idx.long().clamp(max=N_coarse - 1)
            idx_hi = (idx_lo + 1).clamp(max=N_coarse - 1)
            w = (idx - idx_lo.float()).unsqueeze(0).unsqueeze(-1)
            interp = mapped[:, idx_lo] * (1 - w) + mapped[:, idx_hi] * w
            combined = torch.cat([fine, interp], dim=-1)
            out[s] = self.node_update_mlps[s](combined)
        return out

    @staticmethod
    def sheaf_cohomology_h1(scale_features):
        """Compute H1 sheaf cohomology measure.

        Returns:
            float: sum of Frobenius norms of (fine - interpolated_coarse)
                   across adjacent scale pairs. Zero means globally consistent.
        """
        total = 0.0
        for s in range(1, len(scale_features)):
            coarse = scale_features[s - 1]
            fine = scale_features[s]
            N_fine = fine.shape[1]
            N_coarse = coarse.shape[1]
            idx = torch.linspace(0, N_coarse - 1, N_fine, device=fine.device)
            idx_lo = idx.long().clamp(max=N_coarse - 1)
            idx_hi = (idx_lo + 1).clamp(max=N_coarse - 1)
            w = (idx - idx_lo.float()).unsqueeze(0).unsqueeze(-1)
            interp = coarse[:, idx_lo] * (1 - w) + coarse[:, idx_hi] * w
            total = total + torch.norm(fine - interp, p='fro').item()
        return total


if __name__ == "__main__":
    B, d = 2, 64
    scales = [torch.randn(B, n, d) for n in [10, 20, 30, 50]]
    layer = SheafGNNLayer(d_feat=d, n_scales=4)
    out = layer(scales)
    assert len(out) == 4
    for i, o in enumerate(out):
        assert o.shape == scales[i].shape, f"Scale {i} shape mismatch"
    h1 = SheafGNNLayer.sheaf_cohomology_h1(scales)
    assert isinstance(h1, float)
    loss = sum(o.sum() for o in out)
    loss.backward()
    print("SheafGNNLayer self-test PASSED")
