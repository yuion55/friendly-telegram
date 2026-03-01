"""Joint curriculum loss for RNA 3D structure prediction training."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class JointRNA3DLoss(nn.Module):
    """Joint loss with curriculum weighting between black-box and white-box terms."""

    def __init__(self):
        super().__init__()

    def forward(self, pred_coords, true_coords, pred_bpp, true_bpp,
                pred_genus, true_genus, pred_pd, true_pd,
                pred_ion, true_ion, epoch, max_epochs):
        """Compute joint loss with curriculum weighting.

        Args:
            pred_coords: (B, L, 3) predicted coordinates.
            true_coords: (B, L, 3) true coordinates.
            pred_bpp: (B, L, L) predicted base pair probabilities.
            true_bpp: (B, L, L) true base pair probabilities.
            pred_genus: (B,) predicted genus.
            true_genus: (B,) true genus.
            pred_pd: predicted persistence diagram features.
            true_pd: true persistence diagram features.
            pred_ion: (B, L) predicted ion site probabilities.
            true_ion: (B, L) true ion site labels.
            epoch: current epoch.
            max_epochs: total epochs.

        Returns:
            dict with total loss and all component values.
        """
        progress = epoch / max(max_epochs, 1)
        if progress < 0.2:
            lam_bb, lam_wb = 0.2, 0.8
        elif progress < 0.6:
            lam_bb, lam_wb = 0.5, 0.5
        else:
            lam_bb, lam_wb = 0.8, 0.2

        coord_loss = F.l1_loss(pred_coords, true_coords)

        ss_loss = F.binary_cross_entropy(
            pred_bpp.clamp(1e-7, 1 - 1e-7), true_bpp, reduction='mean'
        )

        lddt_proxy = (
            torch.cdist(pred_coords, pred_coords)
            - torch.cdist(true_coords, true_coords)
        ).abs().mean()

        ion_loss = F.binary_cross_entropy(
            pred_ion.clamp(1e-7, 1 - 1e-7), true_ion, reduction='mean'
        )

        genus_loss = (pred_genus.float() - true_genus.float()).abs().mean()

        tda_loss = (pred_pd - true_pd).pow(2).mean()

        bb_total = coord_loss + 0.3 * ss_loss + 0.2 * lddt_proxy + 0.1 * ion_loss
        wb_total = 0.5 * genus_loss + 0.5 * tda_loss
        total = lam_bb * bb_total + lam_wb * wb_total

        return {
            'total': total,
            'coord_loss': coord_loss.item(),
            'ss_loss': ss_loss.item(),
            'lddt_proxy': lddt_proxy.item(),
            'ion_loss': ion_loss.item(),
            'genus_loss': genus_loss.item(),
            'tda_loss': tda_loss.item(),
            'bb_total': bb_total.item(),
            'wb_total': wb_total.item(),
            'lam_bb': lam_bb,
            'lam_wb': lam_wb,
        }


if __name__ == "__main__":
    B, L = 2, 50
    pred_coords = torch.randn(B, L, 3)
    true_coords = torch.randn(B, L, 3)
    pred_bpp = torch.sigmoid(torch.randn(B, L, L))
    true_bpp = torch.zeros(B, L, L)
    pred_genus = torch.tensor([1.0, 2.0])
    true_genus = torch.tensor([1.0, 1.0])
    pred_pd = torch.randn(B, 10)
    true_pd = torch.randn(B, 10)
    pred_ion = torch.sigmoid(torch.randn(B, L))
    true_ion = torch.zeros(B, L)

    loss_fn = JointRNA3DLoss()

    for ep, expected_bb, expected_wb in [(0, 0.2, 0.8), (5, 0.5, 0.5), (9, 0.8, 0.2)]:
        result = loss_fn(pred_coords, true_coords, pred_bpp, true_bpp,
                         pred_genus, true_genus, pred_pd, true_pd,
                         pred_ion, true_ion, ep, 10)
        assert result['lam_bb'] == expected_bb
        assert result['lam_wb'] == expected_wb
        assert result['total'].requires_grad

    print("JointRNA3DLoss self-test PASSED")
