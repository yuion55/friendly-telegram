"""Riemannian torsion-angle backbone: manifold penalties and diffusion on S¹."""

import numpy as np
from numba import njit, prange

# Seven backbone torsion angles per RNA residue.
TORSION_NAMES = ("alpha", "beta", "gamma", "delta", "epsilon", "zeta", "chi")

# RNA-appropriate allowed ranges (radians) for each torsion, shape (7, 2).
# Ranges are approximate consensus regions from crystallographic surveys.
ALLOWED_TORSION_RANGES = np.array(
    [
        [-1.40, 1.40],   # alpha   ~ ±80°
        [2.44, 4.89],    # beta    ~ 140°–280°
        [0.17, 1.22],    # gamma   ~ 10°–70°  (gauche+)
        [1.05, 2.79],    # delta   ~ 60°–160°
        [-2.79, -1.05],  # epsilon ~ -160°–-60°
        [-1.22, -0.17],  # zeta    ~ -70°–-10° (gauche-)
        [-2.97, -1.40],  # chi     ~ -170°–-80° (anti)
    ],
    dtype=np.float32,
)


@njit(cache=True)
def wrap_angle(angle):
    """Wrap *angle* into the interval [-π, π].

    Args:
        angle: scalar angle in radians.

    Returns:
        Wrapped angle as float32.
    """
    two_pi = np.float32(2.0 * np.pi)
    a = np.float32(angle) % two_pi
    if a > np.float32(np.pi):
        a -= two_pi
    elif a < np.float32(-np.pi):
        a += two_pi
    return a


@njit(cache=True, parallel=True)
def torsion_manifold_penalty(torsions, allowed_ranges):
    """Harmonic penalty for torsion angles outside their allowed ranges.

    For each residue and each of its 7 torsion angles the wrapped angle is
    compared against the corresponding allowed interval.  If it falls outside,
    the squared distance to the nearest boundary is accumulated.

    Args:
        torsions: float32 array of shape (L, 7).
        allowed_ranges: float32 array of shape (7, 2) with [lo, hi] per angle.

    Returns:
        Total penalty as float32 scalar.
    """
    L = torsions.shape[0]
    buf = np.zeros(L, dtype=np.float32)
    for i in prange(L):
        local = np.float32(0.0)
        for j in range(7):
            a = wrap_angle(torsions[i, j])
            lo = allowed_ranges[j, 0]
            hi = allowed_ranges[j, 1]
            if a < lo:
                d = lo - a
                local += d * d
            elif a > hi:
                d = a - hi
                local += d * d
        buf[i] = local
    total = np.float32(0.0)
    for i in range(L):
        total += buf[i]
    return total


@njit(cache=True)
def von_mises_sample(mu, kappa):
    """Sample from the von Mises distribution on S¹ using Best's algorithm.

    Args:
        mu: mean direction (radians).
        kappa: concentration parameter (>= 0).

    Returns:
        A single sample as float32, wrapped to [-π, π].
    """
    mu = np.float32(mu)
    kappa = np.float32(kappa)
    if kappa < np.float32(1e-8):
        # Nearly uniform – draw uniformly on the circle.
        return wrap_angle(np.float32(np.random.random() * 2.0 * np.pi - np.pi))
    # Best's algorithm parameters.
    tau = np.float32(1.0 + np.sqrt(1.0 + 4.0 * kappa * kappa))
    rho = np.float32((tau - np.sqrt(2.0 * tau)) / (2.0 * kappa))
    r = np.float32((1.0 + rho * rho) / (2.0 * rho))
    while True:
        u1 = np.float32(np.random.random())
        z = np.float32(np.cos(np.pi * u1))
        f = np.float32((1.0 + r * z) / (r + z))
        c = np.float32(kappa * (r - f))
        u2 = np.float32(np.random.random())
        if c * (np.float32(2.0) - c) > u2:
            break
        if np.log(c / u2) + np.float32(1.0) >= c:
            break
    u3 = np.float32(np.random.random())
    theta = mu + np.float32(np.sign(u3 - np.float32(0.5))) * np.float32(np.arccos(f))
    return wrap_angle(theta)


@njit(cache=True, parallel=True)
def torsion_diffusion_step(torsions, noise_scale, kappa=np.float32(10.0)):
    """One diffusion step on the torsion-angle manifold.

    Each torsion is perturbed by sampling from a von Mises distribution
    centred at the current value with effective concentration
    kappa_eff = kappa / (noise_scale + 1e-8).

    Args:
        torsions: float32 array of shape (L, 7).
        noise_scale: scalar noise level (>= 0).
        kappa: base concentration parameter (default 10).

    Returns:
        New torsion array of shape (L, 7), float32.
    """
    L = torsions.shape[0]
    out = np.empty((L, 7), dtype=np.float32)
    kappa_eff = np.float32(np.float32(kappa) / (np.float32(noise_scale) + np.float32(1e-8)))
    for i in prange(L):
        for j in range(7):
            out[i, j] = von_mises_sample(torsions[i, j], kappa_eff)
    return out


if __name__ == "__main__":
    np.random.seed(42)
    L = 50

    # Synthetic torsion angles uniformly in [-π, π].
    torsions = np.random.uniform(-np.pi, np.pi, size=(L, 7)).astype(np.float32)

    # Test wrap_angle.
    assert abs(wrap_angle(np.float32(0.0))) < 1e-6
    assert abs(wrap_angle(np.float32(4.0)) - (4.0 - 2.0 * np.pi)) < 1e-5
    assert abs(wrap_angle(np.float32(-4.0)) - (-4.0 + 2.0 * np.pi)) < 1e-5
    print("wrap_angle          OK")

    # Test torsion_manifold_penalty.
    penalty = torsion_manifold_penalty(torsions, ALLOWED_TORSION_RANGES)
    assert penalty >= np.float32(0.0)
    print(f"manifold_penalty     OK  (penalty={penalty:.4f})")

    # Test von_mises_sample.
    sample = von_mises_sample(np.float32(0.0), np.float32(10.0))
    assert -np.pi <= sample <= np.pi
    print(f"von_mises_sample     OK  (sample={sample:.4f})")

    # Test torsion_diffusion_step.
    stepped = torsion_diffusion_step(torsions, np.float32(0.5))
    assert stepped.shape == (L, 7)
    assert stepped.dtype == np.float32
    for idx in range(L):
        for jdx in range(7):
            assert -np.pi <= stepped[idx, jdx] <= np.pi
    print(f"torsion_diffusion    OK  (shape={stepped.shape})")

    print("All self-tests passed.")
