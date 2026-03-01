"""SE(3)-equivariant tensor field layer and equivariance verification."""

import torch
import torch.nn as nn


class SE3EquivarianceVerifier:
    """Verifies SE(3) equivariance of a model function on 3D point clouds."""

    def __init__(self, tolerance: float = 1e-4):
        self.tolerance = tolerance

    def verify(self, model_fn, x: torch.Tensor) -> bool:
        """Check that model_fn commutes with a random SO(3) rotation.

        Args:
            model_fn: callable mapping (B, L, 3) -> (B, L, 3) tensors.
            x: input coordinates of shape (B, L, 3), torch.float32.

        Returns:
            True if model_fn(x @ Q.T) is approximately model_fn(x) @ Q.T.
        """
        M = torch.randn(3, 3, dtype=torch.float32, device=x.device)
        Q, R = torch.linalg.qr(M)
        # Ensure proper rotation (det = +1) by flipping sign if needed
        Q = Q * torch.sign(torch.det(Q))

        x_rotated = x @ Q.T
        out_original = model_fn(x) @ Q.T
        out_rotated = model_fn(x_rotated)

        return bool(torch.allclose(out_rotated, out_original, atol=self.tolerance))


class TensorFieldLayer(nn.Module):
    """Equivariant tensor field layer with scalar and vector channels.

    Scalar path uses a linear projection augmented by vector norms.
    Vector path linearly combines input vectors and gates them via a
    sigmoid-activated scalar projection, preserving SO(3) equivariance.
    """

    def __init__(self, d_in: int, d_out: int):
        super().__init__()
        self.scalar_linear = nn.Linear(d_in, d_out)
        self.norm_linear = nn.Linear(d_in, d_out)
        self.vector_weight = nn.Parameter(
            torch.randn(d_out, d_in, dtype=torch.float32)
        )
        self.gate_linear = nn.Linear(d_in, d_out)

    def forward(
        self, scalars: torch.Tensor, vectors: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            scalars: (B, L, d_in) scalar features.
            vectors: (B, L, d_in, 3) vector features.

        Returns:
            new_scalars: (B, L, d_out)
            new_vectors: (B, L, d_out, 3)
        """
        # Vector norms as extra scalar input (rotation-invariant)
        v_norms = vectors.norm(dim=-1)  # (B, L, d_in)

        # Scalar path: base scalars + vector norm contribution
        new_scalars = self.scalar_linear(scalars) + self.norm_linear(v_norms)

        # Vector path: linearly combine input vector channels
        new_vectors = torch.einsum("oi,blid->blod", self.vector_weight, vectors)

        # Gate vectors with sigmoid of scalar projection (rotation-invariant gate)
        gate = torch.sigmoid(self.gate_linear(scalars))  # (B, L, d_out)
        new_vectors = new_vectors * gate.unsqueeze(-1)

        return new_scalars, new_vectors


if __name__ == "__main__":
    torch.manual_seed(42)

    B, L, d_in, d_out = 2, 50, 16, 32

    # --- Test TensorFieldLayer output shapes ---
    layer = TensorFieldLayer(d_in, d_out)
    scalars = torch.randn(B, L, d_in, dtype=torch.float32)
    vectors = torch.randn(B, L, d_in, 3, dtype=torch.float32)

    new_s, new_v = layer(scalars, vectors)
    assert new_s.shape == (B, L, d_out), f"Scalar shape mismatch: {new_s.shape}"
    assert new_v.shape == (B, L, d_out, 3), f"Vector shape mismatch: {new_v.shape}"
    print(f"TensorFieldLayer shapes OK: scalars {new_s.shape}, vectors {new_v.shape}")

    # --- Test SE(3) equivariance of vector output ---
    def make_equivariant_model(tfl, fixed_scalars):
        """Wrap TensorFieldLayer into a (B,L,3)->(B,L,3) function."""

        def model_fn(coords):
            # Broadcast coords into per-channel vectors: (B,L,3) -> (B,L,d_in,3)
            vecs = coords.unsqueeze(2).expand(-1, -1, fixed_scalars.shape[-1], -1)
            _, out_vectors = tfl(fixed_scalars, vecs)
            return out_vectors.sum(dim=2)  # (B, L, 3)

        return model_fn

    coords = torch.randn(B, L, 3, dtype=torch.float32)
    model_fn = make_equivariant_model(layer, scalars)

    verifier = SE3EquivarianceVerifier(tolerance=1e-4)
    is_equivariant = verifier.verify(model_fn, coords)
    print(f"SE3 equivariance test: {'PASSED' if is_equivariant else 'FAILED'}")
    assert is_equivariant, "Equivariance check failed!"

    print("All tests passed.")
