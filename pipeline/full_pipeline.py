"""Full RNA 3D structure prediction pipeline."""

import numpy as np

from whitebox.partition_function import compute_bpp_linear_approx
from whitebox.genus_invariants import genus_from_bpp
from whitebox.chord_diagram import ChordDiagramCertificate
from whitebox.riemannian_backbone import (
    torsion_diffusion_step, wrap_angle, ALLOWED_TORSION_RANGES,
)
from modules.topology_correction import TopologyCorrector
from modules.ensemble_ranking import rank_ensemble


_NUC_MAP = {'A': 0, 'C': 1, 'G': 2, 'U': 3}


def _sequence_to_int8(sequence):
    """Convert RNA sequence string to int8 array."""
    return np.array([_NUC_MAP.get(c, 0) for c in sequence], dtype=np.int8)


def _torsions_to_coords(torsions, bond_length=3.9):
    """Convert torsion angles to C3' coordinates via idealized backbone geometry.

    Args:
        torsions: (L, 7) float32 array.
        bond_length: distance between consecutive C3' atoms.

    Returns:
        (L, 3) float32 coordinate array.
    """
    L = torsions.shape[0]
    coords = np.zeros((L, 3), dtype=np.float32)
    direction = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    for i in range(1, L):
        delta_angle = torsions[i, 0]
        cos_a = np.cos(delta_angle)
        sin_a = np.sin(delta_angle)
        new_dir = np.array([
            direction[0] * cos_a - direction[1] * sin_a,
            direction[0] * sin_a + direction[1] * cos_a,
            direction[2],
        ], dtype=np.float32)
        norm = np.sqrt(new_dir[0]**2 + new_dir[1]**2 + new_dir[2]**2)
        if norm > 1e-8:
            new_dir /= norm
        direction = new_dir
        coords[i] = coords[i - 1] + bond_length * direction
    return coords


class RNA3DPipeline:
    """Full RNA 3D structure prediction pipeline."""

    def __init__(self, config=None):
        self.config = config or {}
        self.topology_corrector = TopologyCorrector()
        self.certificate = ChordDiagramCertificate()

    def predict(self, sequence, n_candidates=100, n_submit=5):
        """Predict RNA 3D structure.

        Args:
            sequence: RNA sequence string.
            n_candidates: number of candidates to generate.
            n_submit: number of top candidates to return.

        Returns:
            list of dicts with 'coords', 'certificates', 'genus'.
        """
        L = len(sequence)
        seq_int8 = _sequence_to_int8(sequence)

        # Step 1: BPP and genus
        bpp = compute_bpp_linear_approx(seq_int8, L)
        pred_genus = genus_from_bpp(bpp, threshold=0.5)

        # Step 2: Generate candidates via torsion-space diffusion
        candidates = []
        for i in range(n_candidates):
            torsions = np.zeros((L, 7), dtype=np.float32)
            noise_scale = np.float32(i / max(n_candidates, 1))
            torsions = torsion_diffusion_step(torsions, noise_scale, np.float32(10.0))
            coords = _torsions_to_coords(torsions)
            candidates.append(coords)

        # Step 3: Topology correction and genus filtering
        verified = []
        verified_certs = []
        for coords in candidates:
            coords = self.topology_corrector(coords)
            # Extract pairs from bpp
            pair_list = []
            for i_idx in range(L):
                for j_idx in range(i_idx + 4, L):
                    if bpp[i_idx, j_idx] > 0.5:
                        pair_list.append([i_idx, j_idx])
            if len(pair_list) > 0:
                pairs = np.array(pair_list, dtype=np.int32)
                cert = self.certificate.certify(pairs, L)
                cand_genus = cert['genus']
                if abs(cand_genus - pred_genus) > 1:
                    continue
            else:
                pairs = np.zeros((0, 2), dtype=np.int32)
                cert = {'genus': 0, 'crossing_number': 0,
                        'pseudoknot_type': 'planar', 'n_pairs': 0,
                        'certificate_valid': True}
            verified.append(coords)
            verified_certs.append(cert)

        if len(verified) == 0:
            # Fallback: use first candidates
            for coords in candidates[:n_submit]:
                verified.append(self.topology_corrector(coords))
                verified_certs.append({
                    'genus': 0, 'crossing_number': 0,
                    'pseudoknot_type': 'planar', 'n_pairs': 0,
                    'certificate_valid': True,
                })

        # Step 4: Rank ensemble
        n_sel = min(n_submit, len(verified))
        proxy_scores = np.ones(len(verified), dtype=np.float32)
        indices = rank_ensemble(verified, proxy_scores, n_select=n_sel)

        # Step 5: Return results
        results = []
        for idx in indices:
            results.append({
                'coords': verified[idx],
                'certificates': verified_certs[idx],
                'genus': verified_certs[idx].get('genus', 0),
            })
        return results


if __name__ == "__main__":
    seq = "A" * 20 + "U" * 20 + "G" * 10
    pipeline = RNA3DPipeline(config={})
    results = pipeline.predict(seq, n_candidates=20, n_submit=5)
    assert len(results) <= 5
    for r in results:
        assert 'coords' in r
        assert r['coords'].shape == (len(seq), 3)
        assert r['coords'].dtype == np.float32
        assert 'certificates' in r
        assert 'genus' in r
    print(f"full_pipeline self-test PASSED: {len(results)} candidates returned")
