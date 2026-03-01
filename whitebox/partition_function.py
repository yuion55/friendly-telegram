"""Linear-approximation partition function and thermodynamic certificates for RNA."""

import numpy as np
from numba import njit, prange


@njit(cache=True, parallel=True)
def compute_bpp_linear_approx(sequence, L):
    """Compute base-pair probability matrix via a simplified partition function.

    Encodes A=0, C=1, G=2, U=3. Canonical pairs are AU, UA, GU, UG, CG, GC.
    Minimum gap between paired positions is 4.

    Args:
        sequence: int8 array of shape (L,) with nucleotide encoding.
        L: sequence length.

    Returns:
        Symmetric float32 array of shape (L, L) with base-pair probabilities.
    """
    # Build can_pair boolean matrix
    can_pair = np.zeros((L, L), dtype=np.bool_)
    for i in range(L):
        for j in range(i + 5, L):
            a = sequence[i]
            b = sequence[j]
            # AU / UA
            if (a == 0 and b == 3) or (a == 3 and b == 0):
                can_pair[i, j] = True
            # GU / UG
            elif (a == 2 and b == 3) or (a == 3 and b == 2):
                can_pair[i, j] = True
            # CG / GC
            elif (a == 1 and b == 2) or (a == 2 and b == 1):
                can_pair[i, j] = True

    # Forward partition function Q(i,j) -- sequential dependencies
    # Q[i,j] = Q[i,j-1] (j unpaired) + boltzmann*Q[i+1,j-1] (i,j pair)
    # This ensures Q[0,L-1] >= Q[i,j] for all i,j.
    kT = np.float64(0.6156)
    pair_energy = np.float64(2.0)
    boltzmann = np.exp(pair_energy / kT)

    Q = np.ones((L, L), dtype=np.float64)
    for span in range(1, L):
        for i in range(L - span):
            j = i + span
            Q[i, j] = Q[i, j - 1]
            if span >= 5 and can_pair[i, j]:
                Q[i, j] = Q[i, j] + boltzmann * Q[i + 1, j - 1]

    # Normalize by max partition value to ensure valid probabilities
    q_max = np.float64(0.0)
    for i in range(L):
        for j in range(L):
            if Q[i, j] > q_max:
                q_max = Q[i, j]
    total = q_max + np.float64(1.0)

    # BPP computation and symmetrization -- safe to parallelize
    bpp = np.zeros((L, L), dtype=np.float32)
    for i in prange(L):
        for j in range(i + 5, L):
            if can_pair[i, j]:
                pair_val = boltzmann * Q[i + 1, j - 1]
                bpp[i, j] = np.float32(pair_val / total)
                bpp[j, i] = bpp[i, j]

    return bpp


@njit(cache=True)
def thermodynamic_impossibility_certificate(pred_coords, sequence,
                                            delta_g_threshold=10.0):
    """Check if predicted structure is thermodynamically impossible.

    Counts contacts within 8 Angstroms, estimates structural energy as
    -0.5 * n_contacts, and compares to an MFE approximation derived from
    the base-pair probability matrix.

    Args:
        pred_coords: float32 array of shape (L, 3) with predicted positions.
        sequence: int8 array of shape (L,) with nucleotide encoding.
        delta_g_threshold: energy difference threshold for impossibility.

    Returns:
        True if the structure is thermodynamically impossible.
    """
    L = pred_coords.shape[0]
    contact_cutoff = np.float32(8.0)

    # Count contacts within cutoff
    n_contacts = 0
    for i in range(L):
        for j in range(i + 1, L):
            dx = pred_coords[i, 0] - pred_coords[j, 0]
            dy = pred_coords[i, 1] - pred_coords[j, 1]
            dz = pred_coords[i, 2] - pred_coords[j, 2]
            d = np.sqrt(dx * dx + dy * dy + dz * dz)
            if d < contact_cutoff:
                n_contacts += 1

    struct_energy = np.float32(-0.5) * np.float32(n_contacts)

    # MFE approximation from BPP
    bpp = compute_bpp_linear_approx(sequence, L)
    mfe_approx = np.float32(0.0)
    for i in range(L):
        for j in range(i + 5, L):
            if bpp[i, j] > np.float32(0.0):
                mfe_approx = mfe_approx - bpp[i, j]

    diff = struct_energy - mfe_approx
    if diff < np.float32(0.0):
        diff = -diff

    return diff > np.float32(delta_g_threshold)


if __name__ == "__main__":
    L = 50
    np.random.seed(42)

    # Synthetic sequence (A=0, C=1, G=2, U=3)
    sequence = np.random.randint(0, 4, size=L).astype(np.int8)

    # Test BPP computation
    bpp = compute_bpp_linear_approx(sequence, L)
    assert bpp.shape == (L, L), "BPP shape mismatch"
    assert bpp.dtype == np.float32, "BPP dtype must be float32"
    assert np.allclose(bpp, bpp.T), "BPP must be symmetric"
    assert np.all(bpp >= 0.0), "BPP must be non-negative"
    assert np.all(bpp <= 1.0), "BPP must be <= 1"
    print(f"BPP matrix: shape={bpp.shape}, dtype={bpp.dtype}")
    print(f"  max={bpp.max():.6f}, nonzero={np.count_nonzero(bpp)}")

    # Test thermodynamic certificate
    coords = np.random.randn(L, 3).astype(np.float32) * 3.0
    result = thermodynamic_impossibility_certificate(coords, sequence)
    assert isinstance(result, (bool, np.bool_)), "Certificate must return bool"
    print(f"Thermodynamic impossibility certificate: {result}")

    # Verify with a high threshold (should not flag)
    result_high = thermodynamic_impossibility_certificate(
        coords, sequence, delta_g_threshold=1000.0
    )
    assert not result_high, "Very high threshold should not flag impossibility"
    print(f"High-threshold certificate: {result_high}")

    print("Self-test passed.")
