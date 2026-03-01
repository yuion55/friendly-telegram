"""
rna_topology_penalty.py
========================
Topology Enforcement — Knot/Clash Penalty for RNA 3D Structure Prediction Training

Implements:
  1. Coarse-grained backbone extraction (P / C4' atoms)
  2. Differentiable excluded-volume / clash penalty  (Gaussian repulsion)
  3. Differentiable writhe / crossing-number penalty (Gauss linking integral approximation)
  4. PyTorch auxiliary loss wrapper for training-loop injection
  5. Post-processing inference filter (reject structures with >N crossings)

Performance strategy
---------------------
- Numba @njit + @vectorize for the inner-loop geometry kernels (clash, writhe)
- numpy vectorised array operations everywhere else
- PyTorch custom autograd Function bridges the Numba kernels to gradient flow

Dependencies
------------
  pip install numpy numba torch
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from numba import njit, vectorize, float64, prange


# ─────────────────────────────────────────────────────────────────────────────
# Section 1 ▸ Numba-accelerated geometry kernels
# ─────────────────────────────────────────────────────────────────────────────

@njit(cache=True, parallel=True, fastmath=True)
def _clash_loss_and_grad(
    coords: np.ndarray,   # (N, 3)  backbone atom positions
    sigma: float,         # Gaussian width (Å)
    seq_gap: int,         # minimum sequence separation to count as clash
) -> tuple[float, np.ndarray]:
    """
    Compute excluded-volume clash loss and its gradient w.r.t. coords.

    Loss = Σ_{|i-j| > seq_gap}  exp( -||r_i - r_j||² / (2σ²) )

    The Gaussian repulsion is:
      • maximal when atoms overlap (dist→0)
      • ~0 when atoms are far apart (dist >> σ)
    Returns (scalar loss, gradient array same shape as coords).
    """
    N = coords.shape[0]
    inv2s2 = 1.0 / (2.0 * sigma * sigma)
    loss = 0.0
    grad = np.zeros_like(coords)

    for i in prange(N):
        for j in range(i + seq_gap + 1, N):
            dx = coords[i, 0] - coords[j, 0]
            dy = coords[i, 1] - coords[j, 1]
            dz = coords[i, 2] - coords[j, 2]
            d2 = dx * dx + dy * dy + dz * dz
            g = math.exp(-d2 * inv2s2)
            loss += g
            # ∂g/∂r_i = -g * (r_i - r_j) / σ²
            factor = -2.0 * g * inv2s2
            grad[i, 0] += factor * dx
            grad[i, 1] += factor * dy
            grad[i, 2] += factor * dz
            grad[j, 0] -= factor * dx
            grad[j, 1] -= factor * dy
            grad[j, 2] -= factor * dz

    return loss, grad


@njit(cache=True, fastmath=True)
def _gauss_linking_segment_pair(
    a1: np.ndarray, a2: np.ndarray,   # segment A endpoints (3,)
    b1: np.ndarray, b2: np.ndarray,   # segment B endpoints (3,)
) -> float:
    """
    Contribution to the Gauss linking integral from one pair of backbone
    bond vectors.

    Gauss integral (unnormalised):
      dΩ = ( (r12 × r13) · r14 ) / |r12|³  +  permutations
    
    Here we use the solid-angle / signed-area formula for two line segments
    as described in Klenin & Langowski (2000), returning the signed
    crossing contribution. The full sum over all (i,j) pairs approximates
    the writhe of the open chain.
    """
    # midpoint vectors
    da = a2 - a1
    db = b2 - b1
    r = a1 - b1

    # four corner displacement vectors
    r11 = a1 - b1
    r12 = a1 - b2
    r21 = a2 - b1
    r22 = a2 - b2

    n11 = math.sqrt(r11[0]**2 + r11[1]**2 + r11[2]**2)
    n12 = math.sqrt(r12[0]**2 + r12[1]**2 + r12[2]**2)
    n21 = math.sqrt(r21[0]**2 + r21[1]**2 + r21[2]**2)
    n22 = math.sqrt(r22[0]**2 + r22[1]**2 + r22[2]**2)

    if n11 < 1e-8 or n12 < 1e-8 or n21 < 1e-8 or n22 < 1e-8:
        return 0.0

    # normalised unit vectors
    u11 = r11 / n11;  u12 = r12 / n12
    u21 = r21 / n21;  u22 = r22 / n22

    # signed solid angle of the quadrilateral (a1,a2,b1,b2) seen from origin
    # approximated by sum of two triangles via the cross-product area formula
    def _tri_solid_angle(v1, v2, v3):
        c = (v1[0]*(v2[1]*v3[2]-v2[2]*v3[1])
           - v1[1]*(v2[0]*v3[2]-v2[2]*v3[0])
           + v1[2]*(v2[0]*v3[1]-v2[1]*v3[0]))
        d = 1.0 + v1[0]*v2[0]+v1[1]*v2[1]+v1[2]*v2[2] \
               + v2[0]*v3[0]+v2[1]*v3[1]+v2[2]*v3[2] \
               + v1[0]*v3[0]+v1[1]*v3[1]+v1[2]*v3[2]
        if abs(d) < 1e-12:
            return 0.0
        return 2.0 * math.atan2(c, d)

    omega = _tri_solid_angle(u11, u21, u22) + _tri_solid_angle(u11, u22, u12)
    return omega


@njit(cache=True, parallel=True, fastmath=True)
def _writhe_and_grad(
    coords: np.ndarray,   # (N, 3)
    seq_gap: int,         # minimum bond-index separation
) -> tuple[float, np.ndarray]:
    """
    Compute an approximate writhe (Gauss self-linking integral) and a
    numerical gradient via central differences (step = 1e-4 Å).

    Writhe W = (1/4π) Σ_{|i-j|>seq_gap} Ω(seg_i, seg_j)

    Returns (writhe, gradient).
    """
    N = coords.shape[0]
    inv4pi = 1.0 / (4.0 * math.pi)
    n_segs = N - 1

    # Forward pass: accumulate writhe
    writhe = 0.0
    for i in prange(n_segs):
        for j in range(i + seq_gap + 1, n_segs):
            omega = _gauss_linking_segment_pair(
                coords[i], coords[i+1], coords[j], coords[j+1]
            )
            writhe += omega
    writhe *= inv4pi

    # Numerical gradient (central differences)
    h = 1e-4
    grad = np.zeros_like(coords)
    for i in range(N):
        for k in range(3):
            coords[i, k] += h
            w_plus = 0.0
            for p in range(n_segs):
                for q in range(p + seq_gap + 1, n_segs):
                    w_plus += _gauss_linking_segment_pair(
                        coords[p], coords[p+1], coords[q], coords[q+1]
                    )
            w_plus *= inv4pi

            coords[i, k] -= 2 * h
            w_minus = 0.0
            for p in range(n_segs):
                for q in range(p + seq_gap + 1, n_segs):
                    w_minus += _gauss_linking_segment_pair(
                        coords[p], coords[p+1], coords[q], coords[q+1]
                    )
            w_minus *= inv4pi

            coords[i, k] += h  # restore
            grad[i, k] = (w_plus - w_minus) / (2 * h)

    return writhe, grad


@vectorize([float64(float64, float64)], cache=True, target="parallel")
def _soft_abs(x: float, eps: float) -> float:
    """Smooth absolute value: sqrt(x² + eps²) — differentiable at 0."""
    return math.sqrt(x * x + eps * eps)


# ─────────────────────────────────────────────────────────────────────────────
# Section 2 ▸ Differentiable writhe approximation for autograd
# ─────────────────────────────────────────────────────────────────────────────

def _compute_crossing_number_soft(
    coords: np.ndarray,
    seq_gap: int = 4,
) -> tuple[float, float]:
    """
    Returns (writhe, |writhe| as approximate crossing count).
    Pure-numpy path — used when we only need inference-time counting.
    """
    writhe, _ = _writhe_and_grad(coords, seq_gap)
    return writhe, abs(writhe)


# ─────────────────────────────────────────────────────────────────────────────
# Section 3 ▸ PyTorch autograd bridge
# ─────────────────────────────────────────────────────────────────────────────

class _ClashLossFunction(torch.autograd.Function):
    """
    Custom autograd Function wrapping the Numba clash kernel.
    Supports full backward pass for training-loop integration.
    """

    @staticmethod
    def forward(
        ctx,
        coords_tensor: torch.Tensor,   # (N, 3)
        sigma: float,
        seq_gap: int,
    ) -> torch.Tensor:
        coords_np = coords_tensor.detach().cpu().numpy().astype(np.float64)
        loss_val, grad_np = _clash_loss_and_grad(coords_np, sigma, seq_gap)
        ctx.save_for_backward(torch.from_numpy(grad_np))
        return torch.tensor(loss_val, dtype=coords_tensor.dtype,
                            device=coords_tensor.device)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        (grad_np,) = ctx.saved_tensors
        grad = grad_np.to(grad_output.device) * grad_output
        return grad, None, None


class _WritheLossFunction(torch.autograd.Function):
    """
    Custom autograd Function wrapping the Numba writhe kernel.
    Loss = writhe²  (penalises both positive and negative writhe equally).
    """

    @staticmethod
    def forward(
        ctx,
        coords_tensor: torch.Tensor,   # (N, 3)
        seq_gap: int,
    ) -> torch.Tensor:
        coords_np = coords_tensor.detach().cpu().numpy().astype(np.float64)
        writhe, grad_np = _writhe_and_grad(coords_np, seq_gap)
        # chain rule: d(w²)/d(coords) = 2w * dw/d(coords)
        scaled_grad = torch.from_numpy(2.0 * writhe * grad_np)
        ctx.save_for_backward(scaled_grad)
        return torch.tensor(
            writhe ** 2, dtype=coords_tensor.dtype, device=coords_tensor.device
        )

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        (scaled_grad,) = ctx.saved_tensors
        grad = scaled_grad.to(grad_output.device) * grad_output
        return grad, None


# ─────────────────────────────────────────────────────────────────────────────
# Section 4 ▸ High-level loss module (drop-in training-loop component)
# ─────────────────────────────────────────────────────────────────────────────

class TopologyPenalty(nn.Module):
    """
    Auxiliary loss module for RNA 3D backbone topology enforcement.

    Usage
    -----
    ::
        topo_loss_fn = TopologyPenalty(
            sigma=4.0,          # Gaussian repulsion width (Å)
            seq_gap=4,          # ignore bonded neighbours within this range
            clash_weight=1.0,   # λ for excluded-volume term
            writhe_weight=0.5,  # λ for writhe term
        )

        # Inside training loop:
        coords = model(batch)                     # (B, N, 3) predicted coords
        topo_loss = topo_loss_fn(coords)
        total_loss = primary_loss + topo_loss
        total_loss.backward()

    Parameters
    ----------
    sigma : float
        Gaussian repulsion width in Ångströms. Typical: 3.5–5.0 for P atoms.
    seq_gap : int
        Sequence separation threshold; pairs |i-j| <= seq_gap are ignored.
        Use 4 for P-atom backbone, 3 for C4'.
    clash_weight : float
        Weighting coefficient λ_clash for the excluded-volume term.
    writhe_weight : float
        Weighting coefficient λ_writhe for the knot/entanglement term.
    """

    def __init__(
        self,
        sigma: float = 4.0,
        seq_gap: int = 4,
        clash_weight: float = 1.0,
        writhe_weight: float = 0.5,
    ) -> None:
        super().__init__()
        self.sigma = sigma
        self.seq_gap = seq_gap
        self.clash_weight = clash_weight
        self.writhe_weight = writhe_weight

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        coords : torch.Tensor, shape (B, N, 3) or (N, 3)
            Coarse-grained backbone coordinates (P or C4' atoms).

        Returns
        -------
        torch.Tensor
            Scalar auxiliary loss to be added to the primary training loss.
        """
        if coords.dim() == 2:
            coords = coords.unsqueeze(0)  # add batch dim

        B = coords.shape[0]
        total_clash = coords.new_zeros(1)
        total_writhe = coords.new_zeros(1)

        for b in range(B):
            c = coords[b]  # (N, 3)
            clash = _ClashLossFunction.apply(c, self.sigma, self.seq_gap)
            writhe_sq = _WritheLossFunction.apply(c, self.seq_gap)
            total_clash = total_clash + clash
            total_writhe = total_writhe + writhe_sq

        loss = (
            self.clash_weight * total_clash / B
            + self.writhe_weight * total_writhe / B
        )
        return loss

    def extra_repr(self) -> str:
        return (
            f"sigma={self.sigma}, seq_gap={self.seq_gap}, "
            f"clash_weight={self.clash_weight}, writhe_weight={self.writhe_weight}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Section 5 ▸ Inference-time post-processing filter
# ─────────────────────────────────────────────────────────────────────────────

class TopologyFilter:
    """
    Post-processing filter that ranks candidate predictions and rejects
    those with too many topological crossings.

    Usage
    -----
    ::
        filt = TopologyFilter(max_crossings=2, seq_gap=4)

        # candidates: list of (N,3) numpy arrays or (K,N,3) tensor
        best_coords, meta = filt.select_best(candidates)
        print(meta)   # crossing counts and acceptance status per candidate
    """

    def __init__(
        self,
        max_crossings: float = 2.0,
        seq_gap: int = 4,
    ) -> None:
        self.max_crossings = max_crossings
        self.seq_gap = seq_gap

    def count_crossings(self, coords: np.ndarray) -> float:
        """Return approximate crossing number (|writhe|) for one structure."""
        coords_f64 = np.asarray(coords, dtype=np.float64)
        writhe, cross = _compute_crossing_number_soft(coords_f64, self.seq_gap)
        return cross

    def select_best(
        self,
        candidates: list[np.ndarray],
        scores: Optional[list[float]] = None,
    ) -> tuple[np.ndarray, list[dict]]:
        """
        Select the best topologically valid candidate.

        Parameters
        ----------
        candidates : list of (N, 3) numpy arrays
            Ranked candidate structures (best-first by primary metric).
        scores : list of float, optional
            Primary ranking scores (higher = better).  When provided,
            candidates are first sorted by score descending, then filtered.

        Returns
        -------
        best_coords : np.ndarray  (N, 3)
            First accepted candidate, or the least-knotted one if all fail.
        metadata : list of dict
            Per-candidate dict with keys:
              ``rank``, ``crossings``, ``accepted``, ``score``.
        """
        if scores is not None:
            order = np.argsort(scores)[::-1]
            candidates = [candidates[i] for i in order]
            scores = [scores[i] for i in order]

        metadata = []
        for rank, coords in enumerate(candidates):
            cross = self.count_crossings(coords)
            accepted = cross <= self.max_crossings
            metadata.append({
                "rank": rank,
                "crossings": round(cross, 4),
                "accepted": accepted,
                "score": scores[rank] if scores else None,
            })

        # Return first accepted
        for i, meta in enumerate(metadata):
            if meta["accepted"]:
                return candidates[i], metadata

        # Fallback: return least-knotted candidate
        least_knotted = min(range(len(metadata)),
                            key=lambda i: metadata[i]["crossings"])
        return candidates[least_knotted], metadata


# ─────────────────────────────────────────────────────────────────────────────
# Section 6 ▸ Utility helpers
# ─────────────────────────────────────────────────────────────────────────────

def extract_backbone_coords(
    atom_positions: np.ndarray,   # (N_atoms, 3)
    atom_names: list[str],
    atom_type: str = "P",
) -> np.ndarray:
    """
    Extract coarse-grained backbone positions from a full-atom coordinate array.

    Parameters
    ----------
    atom_positions : (N_atoms, 3) float array
    atom_names : list of str, length N_atoms
    atom_type : str
        ``"P"`` for phosphate backbone or ``"C4'"`` for sugar backbone.

    Returns
    -------
    (M, 3) float array of selected backbone positions.
    """
    mask = [i for i, name in enumerate(atom_names) if name.strip() == atom_type]
    if not mask:
        raise ValueError(
            f"No atoms named '{atom_type}' found.  "
            f"Unique names: {sorted(set(atom_names))[:10]}"
        )
    return atom_positions[np.array(mask)]


def writhe_from_coords(coords: np.ndarray, seq_gap: int = 4) -> float:
    """Convenience function: return writhe of a backbone as a plain float."""
    w, _ = _writhe_and_grad(np.asarray(coords, np.float64), seq_gap)
    return w


def clash_score_from_coords(
    coords: np.ndarray,
    sigma: float = 4.0,
    seq_gap: int = 4,
) -> float:
    """Convenience function: return Gaussian clash score as a plain float."""
    loss, _ = _clash_loss_and_grad(np.asarray(coords, np.float64), sigma, seq_gap)
    return loss


# ─────────────────────────────────────────────────────────────────────────────
# Section 7 ▸ Quick integration example / smoke-test
# ─────────────────────────────────────────────────────────────────────────────

def _demo():
    """
    Minimal end-to-end demo:
      • Simulated RNA backbone (60 residues, P-atoms)
      • Single forward + backward pass through TopologyPenalty
      • Inference-time TopologyFilter on 5 random candidates
    """
    print("=" * 62)
    print("  RNA Topology Penalty — demo / smoke-test")
    print("=" * 62)

    rng = np.random.default_rng(42)
    N = 60   # residues

    # --- Training-loop demo ---
    # Intentionally inject a knot-like coil in residues 20-40
    t = np.linspace(0, 4 * np.pi, N)
    coords_np = np.column_stack([
        6.0 * np.cos(t) + rng.normal(0, 0.3, N),
        6.0 * np.sin(t) + rng.normal(0, 0.3, N),
        3.0 * t / (2 * np.pi) + rng.normal(0, 0.3, N),
    ]).astype(np.float64)

    coords_torch = torch.tensor(
        coords_np, dtype=torch.float32, requires_grad=True
    )

    penalty_fn = TopologyPenalty(
        sigma=4.0, seq_gap=4,
        clash_weight=1.0, writhe_weight=0.5,
    )
    print(f"\nTopologyPenalty module: {penalty_fn}")

    loss = penalty_fn(coords_torch)
    print(f"\nTopology loss (forward):  {loss.item():.6f}")

    loss.backward()
    grad_norm = coords_torch.grad.norm().item()
    print(f"Gradient norm (backward): {grad_norm:.6f}")

    # --- Inference-time filtering demo ---
    print("\n--- Inference filter demo ---")
    candidates = []
    true_crossings = []
    for _ in range(5):
        c_np = coords_np + rng.normal(0, 1.5, coords_np.shape)
        candidates.append(c_np)

    scores = [rng.uniform(0.4, 0.9) for _ in range(5)]

    filt = TopologyFilter(max_crossings=2.0, seq_gap=4)
    best, meta = filt.select_best(candidates, scores=scores)

    print(f"{'Rank':<6} {'Score':<8} {'Crossings':<12} {'Accepted'}")
    print("-" * 40)
    for m in meta:
        print(
            f"{m['rank']:<6} {m['score']:<8.4f} "
            f"{m['crossings']:<12.4f} {m['accepted']}"
        )

    best_cross = filt.count_crossings(best)
    print(f"\nSelected structure crossing count: {best_cross:.4f}")
    print("=" * 62)


if __name__ == "__main__":
    _demo()
