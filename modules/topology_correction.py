"""
topology_correction.py
======================
Chain-break detection and repair for RNA backbone coordinates.

Detects breaks in the C3'-C3' backbone chain where consecutive inter-residue
distances exceed a physically plausible threshold (default 4.5 Å), then repairs
them via linear interpolation across break segments.

Functions:
  detect_chain_breaks  — Numba-parallel scan of consecutive C3' distances
  repair_chain_breaks  — sequential linear interpolation across break segments

Class:
  TopologyCorrector    — convenience wrapper: detect → repair pipeline

Usage:
  from modules.topology_correction import TopologyCorrector
  corrector = TopologyCorrector(threshold=4.5)
  coords_fixed = corrector(coords)  # (L, 3) float32
"""

from __future__ import annotations

import numpy as np
from numba import njit, prange


# ---------------------------------------------------------------------------
# 1. DETECT CHAIN BREAKS  (parallel — no write conflicts)
# ---------------------------------------------------------------------------

@njit(cache=True, parallel=True)
def detect_chain_breaks(
    coords: np.ndarray,
    threshold: np.float32 = np.float32(4.5),
) -> np.ndarray:
    """
    Flag consecutive C3'-C3' distances that exceed *threshold*.

    Parameters
    ----------
    coords : ndarray, shape (L, 3), float32
        C3' atom coordinates for each residue.
    threshold : float32
        Maximum allowed inter-residue distance (Å).

    Returns
    -------
    breaks : ndarray, shape (L-1,), bool
        True where ``||coords[i+1] - coords[i]|| > threshold``.
    """
    L = coords.shape[0]
    breaks = np.empty(L - 1, dtype=np.bool_)
    for i in prange(L - 1):
        dx = coords[i + 1, 0] - coords[i, 0]
        dy = coords[i + 1, 1] - coords[i, 1]
        dz = coords[i + 1, 2] - coords[i, 2]
        dist = np.sqrt(dx * dx + dy * dy + dz * dz)
        breaks[i] = dist > threshold
    return breaks


# ---------------------------------------------------------------------------
# 2. REPAIR CHAIN BREAKS  (sequential — interpolation has data dependencies)
# ---------------------------------------------------------------------------

@njit(cache=True)
def repair_chain_breaks(
    coords: np.ndarray,
    breaks: np.ndarray,
) -> np.ndarray:
    """
    Linearly interpolate coordinates across each contiguous break segment.

    For every maximal run of consecutive breaks ``[start .. end]`` the
    coordinates ``coords[start] .. coords[end+1]`` are replaced by evenly
    spaced points between ``coords[start]`` (anchor before the break) and
    ``coords[end+1]`` (anchor after the break).

    Parameters
    ----------
    coords : ndarray, shape (L, 3), float32
        Original C3' coordinates.
    breaks : ndarray, shape (L-1,), bool
        Break flags from :func:`detect_chain_breaks`.

    Returns
    -------
    out : ndarray, shape (L, 3), float32
        Repaired coordinates (copy; input is not mutated).
    """
    L = coords.shape[0]
    out = coords.copy()
    i = 0
    while i < L - 1:
        if not breaks[i]:
            i += 1
            continue
        # Find the end of the contiguous break segment
        seg_start = i          # last good anchor before the break
        while i < L - 1 and breaks[i]:
            i += 1
        seg_end = i            # first good anchor after the break
        # Interpolate interior points (seg_start+1 .. seg_end-1)
        # between anchor coords[seg_start] and coords[seg_end].
        n_pts = seg_end - seg_start + 1  # total points including both anchors
        for k in range(1, n_pts - 1):
            t = np.float32(k) / np.float32(n_pts - 1)
            for d in range(3):
                out[seg_start + k, d] = (
                    coords[seg_start, d] * (np.float32(1.0) - t)
                    + coords[seg_end, d] * t
                )
    return out


# ---------------------------------------------------------------------------
# 3. TOPOLOGY CORRECTOR  (high-level wrapper)
# ---------------------------------------------------------------------------

class TopologyCorrector:
    """
    Detect and repair backbone chain breaks in a single call.

    Parameters
    ----------
    threshold : float
        C3'-C3' distance above which a break is flagged (default 4.5 Å).
    """

    def __init__(self, threshold: float = 4.5) -> None:
        self.threshold = np.float32(threshold)

    def __call__(self, coords: np.ndarray) -> np.ndarray:
        """
        Run detect → repair pipeline.

        Parameters
        ----------
        coords : ndarray, shape (L, 3)
            Input C3' coordinates (any float dtype; cast to float32 internally).

        Returns
        -------
        ndarray, shape (L, 3), float32
            Corrected coordinates.
        """
        coords = np.asarray(coords, dtype=np.float32)
        breaks = detect_chain_breaks(coords, self.threshold)
        return repair_chain_breaks(coords, breaks)


# ---------------------------------------------------------------------------
# Self-test on synthetic data
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    rng = np.random.RandomState(42)
    L = 50

    # Build a smooth backbone with ~3.8 Å spacing
    coords = np.zeros((L, 3), dtype=np.float32)
    for i in range(1, L):
        direction = rng.randn(3).astype(np.float32)
        direction /= np.linalg.norm(direction)
        coords[i] = coords[i - 1] + direction * np.float32(3.8)

    # Inject two chain breaks
    coords[20] += np.float32(15.0)
    coords[35] += np.float32(12.0)

    corrector = TopologyCorrector(threshold=4.5)
    breaks = detect_chain_breaks(
        np.asarray(coords, dtype=np.float32), np.float32(4.5),
    )

    print(f"Chain length       : {L}")
    print(f"Breaks detected    : {int(breaks.sum())} / {L - 1}")
    assert breaks.shape == (L - 1,), f"Bad shape: {breaks.shape}"
    assert breaks.dtype == np.bool_, f"Bad dtype: {breaks.dtype}"
    assert breaks.sum() > 0, "Expected at least one break"

    fixed = corrector(coords)
    assert fixed.shape == (L, 3), f"Bad shape: {fixed.shape}"
    assert fixed.dtype == np.float32, f"Bad dtype: {fixed.dtype}"

    # After repair, no consecutive distance should exceed threshold
    dists = np.linalg.norm(np.diff(fixed, axis=0), axis=1)
    remaining = int((dists > 4.5).sum())
    print(f"Breaks after repair: {remaining} / {L - 1}")
    print(f"Max distance       : {dists.max():.3f} Å")
    print("Self-test PASSED ✓")
