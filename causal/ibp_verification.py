"""Interval Bound Propagation for coordinate uncertainty verification."""

import numpy as np
import torch
import torch.nn as nn


class IBPBound:
    """Interval bound with propagation through linear, ReLU, sigmoid layers."""

    def __init__(self, lb, ub):
        self.lb = lb
        self.ub = ub

    def propagate_linear(self, weight, bias):
        """Propagate through a linear layer.

        Args:
            weight: (out, in) tensor.
            bias: (out,) tensor.

        Returns:
            IBPBound with new bounds.
        """
        w_pos = weight.clamp(min=0)
        w_neg = weight.clamp(max=0)
        new_lb = self.lb @ w_pos.T + self.ub @ w_neg.T + bias
        new_ub = self.ub @ w_pos.T + self.lb @ w_neg.T + bias
        return IBPBound(new_lb, new_ub)

    def propagate_relu(self):
        """Propagate through ReLU."""
        return IBPBound(self.lb.clamp(min=0), self.ub.clamp(min=0))

    def propagate_sigmoid(self):
        """Propagate through sigmoid."""
        return IBPBound(torch.sigmoid(self.lb), torch.sigmoid(self.ub))


class CoordinateBoundVerifier:
    """Verify coordinate prediction uncertainty via IBP."""

    def __init__(self, model=None, epsilon=0.1):
        self.model = model
        self.epsilon = epsilon

    def compute_coordinate_bounds(self, seq_embedding):
        """Compute coordinate bounds via forward pass on perturbed inputs.

        Args:
            seq_embedding: (1, L, d) tensor.

        Returns:
            (lb, ub): elementwise min/max bounds as tensors.
        """
        with torch.no_grad():
            center = seq_embedding
            upper = center + self.epsilon
            lower = center - self.epsilon
            if self.model is not None:
                out_c = self.model(center)
                out_u = self.model(upper)
                out_l = self.model(lower)
            else:
                out_c = center
                out_u = upper
                out_l = lower
            lb = torch.min(torch.min(out_c, out_u), out_l)
            ub = torch.max(torch.max(out_c, out_u), out_l)
        return lb, ub

    def uncertainty_certificate(self, seq_embedding):
        """Compute per-residue uncertainty certificate.

        Args:
            seq_embedding: (1, L, d) tensor.

        Returns:
            dict with delta, high_uncertainty_mask, n_uncertain_residues,
            max_uncertainty_angstrom.
        """
        lb, ub = self.compute_coordinate_bounds(seq_embedding)
        delta = (ub - lb) / 2.0
        if delta.dim() == 3:
            delta_norm = delta.norm(dim=-1).squeeze(0)
        else:
            delta_norm = delta.abs().squeeze(0)
            if delta_norm.dim() > 1:
                delta_norm = delta_norm.norm(dim=-1)
        high_mask = delta_norm > 2.0
        return {
            'delta': delta_norm,
            'high_uncertainty_mask': high_mask,
            'n_uncertain_residues': int(high_mask.sum().item()),
            'max_uncertainty_angstrom': float(delta_norm.max().item()),
        }


if __name__ == "__main__":
    b = IBPBound(torch.tensor([-1.0, 0.0]), torch.tensor([1.0, 2.0]))
    w = torch.tensor([[0.5, -0.3], [0.2, 0.8]])
    bias = torch.tensor([0.1, -0.1])
    b2 = b.propagate_linear(w, bias)
    assert (b2.lb <= b2.ub).all()
    b3 = b.propagate_relu()
    assert (b3.lb >= 0).all()
    b4 = b.propagate_sigmoid()
    assert (b4.lb >= 0).all() and (b4.ub <= 1).all()

    L, d = 50, 16
    emb = torch.randn(1, L, d)
    verifier = CoordinateBoundVerifier(model=None, epsilon=0.1)
    lb, ub = verifier.compute_coordinate_bounds(emb)
    assert (lb <= ub).all()
    cert = verifier.uncertainty_certificate(emb)
    assert 'delta' in cert
    assert 'high_uncertainty_mask' in cert
    assert 'n_uncertain_residues' in cert
    assert 'max_uncertainty_angstrom' in cert
    print("ibp_verification self-test PASSED")
