"""Kolmogorov Normalized Compression Distance for distribution shift detection."""

import zlib
import numpy as np


def serialize_structure(torsions, precision=2):
    """Serialize torsion angles to bytes.

    Args:
        torsions: (L, 7) float32 array of torsion angles.
        precision: decimal precision for quantization.

    Returns:
        bytes representation.
    """
    scale = 10 ** precision
    quantized = np.round(torsions * scale).astype(np.int16)
    return quantized.tobytes()


def kolmogorov_complexity_approx(data):
    """Approximate Kolmogorov complexity via zlib compression.

    Args:
        data: bytes object.

    Returns:
        int: length of compressed data.
    """
    return len(zlib.compress(data, level=9))


def normalized_compression_distance(x, y):
    """Compute NCD between two byte strings.

    Args:
        x: bytes object.
        y: bytes object.

    Returns:
        float: NCD value in [0, ~1].
    """
    kx = kolmogorov_complexity_approx(x)
    ky = kolmogorov_complexity_approx(y)
    kxy = kolmogorov_complexity_approx(x + y)
    denom = max(kx, ky)
    if denom == 0:
        return 0.0
    return (kxy - min(kx, ky)) / denom


class DistributionShiftDetector:
    """Detect OOD structures via NCD to training distribution."""

    def __init__(self, ncd_threshold=0.7):
        self.ncd_threshold = ncd_threshold
        self._training_serialized = []

    def fit(self, training_torsions):
        """Serialize all training structures.

        Args:
            training_torsions: list of (L_i, 7) float32 arrays.
        """
        self._training_serialized = [
            serialize_structure(t) for t in training_torsions
        ]

    def is_ood(self, pred_torsions, n_reference=20):
        """Check if a predicted structure is out-of-distribution.

        Args:
            pred_torsions: (L, 7) float32 array.
            n_reference: number of reference structures to sample.

        Returns:
            (bool, float): (is_ood, mean_ncd).
        """
        pred_bytes = serialize_structure(pred_torsions)
        n_ref = min(n_reference, len(self._training_serialized))
        if n_ref == 0:
            return True, 1.0
        indices = np.random.choice(
            len(self._training_serialized), size=n_ref, replace=False
        )
        ncds = [
            normalized_compression_distance(pred_bytes, self._training_serialized[i])
            for i in indices
        ]
        mean_ncd = float(np.mean(ncds))
        return mean_ncd > self.ncd_threshold, mean_ncd


if __name__ == "__main__":
    L = 50
    rng = np.random.RandomState(42)
    train_data = [rng.randn(L, 7).astype(np.float32) for _ in range(30)]
    s = serialize_structure(train_data[0])
    assert isinstance(s, bytes)
    kc = kolmogorov_complexity_approx(s)
    assert kc > 0
    ncd = normalized_compression_distance(s, s)
    assert 0.0 <= ncd <= 1.0 + 1e-6
    detector = DistributionShiftDetector(ncd_threshold=0.7)
    detector.fit(train_data)
    ood_flag, mean_ncd = detector.is_ood(train_data[0], n_reference=10)
    assert isinstance(ood_flag, bool)
    assert isinstance(mean_ncd, float)
    outlier = np.ones((L, 7), dtype=np.float32) * 100.0
    ood_flag2, mean_ncd2 = detector.is_ood(outlier, n_reference=10)
    assert isinstance(ood_flag2, bool)
    print("kolmogorov_ncd self-test PASSED")
