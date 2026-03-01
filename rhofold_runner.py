"""RhoFold+ model download, loading, and GPU inference for RNA 3D structure prediction."""

import os
import numpy as np
import torch

# Graceful import for RhoFold — may not be installed in all environments
try:
    from rhofold.rhofold import RhoFold as _RhoFoldModel
    _RHOFOLD_AVAILABLE = True
except ImportError:
    _RhoFoldModel = None
    _RHOFOLD_AVAILABLE = False

try:
    from huggingface_hub import hf_hub_download
    _HF_HUB_AVAILABLE = True
except ImportError:
    hf_hub_download = None
    _HF_HUB_AVAILABLE = False


_NUC_MAP = {'A': 0, 'C': 1, 'G': 2, 'U': 3, 'N': 4}
_PAD_TOKEN = 4
_CACHE_DIR = os.path.expanduser("~/.cache/rhofold/")
_HF_MODEL_ID = "ml4bio/RhoFold"
_HF_FILENAME = "rhofold_pretrained.pt"


def _download_weights(cache_dir=_CACHE_DIR, weights_path=None):
    """Download or locate pretrained RhoFold weights.

    Parameters
    ----------
    cache_dir : str
        Directory for caching downloaded weights.
    weights_path : str or None
        Explicit path to a local weights file.  When provided, the file is
        used directly and no download is attempted (offline-friendly).

    Returns
    -------
    str
        Path to the weights file.
    """
    # 1. Explicit local path (Kaggle offline dataset, etc.)
    if weights_path is not None:
        if os.path.isfile(weights_path):
            return weights_path
        raise FileNotFoundError(f"Weights file not found: {weights_path}")

    # 2. Cached copy
    os.makedirs(cache_dir, exist_ok=True)
    local_path = os.path.join(cache_dir, _HF_FILENAME)
    if os.path.isfile(local_path):
        return local_path

    # 3. Well-known Kaggle dataset paths (pre-downloaded during internet-on)
    for kaggle_path in [
        "/kaggle/input/rhofold-weights/rhofold_pretrained.pt",
        "/kaggle/input/rhofold-weights/models--ml4bio--RhoFold/rhofold_pretrained.pt",
    ]:
        if os.path.isfile(kaggle_path):
            return kaggle_path

    # 4. Download from HuggingFace Hub
    if not _HF_HUB_AVAILABLE:
        raise ImportError(
            "huggingface_hub is required to download RhoFold weights. "
            "Install with: pip install huggingface_hub"
        )
    path = hf_hub_download(
        repo_id=_HF_MODEL_ID,
        filename=_HF_FILENAME,
        cache_dir=cache_dir,
    )
    return path


class RhoFoldRunner:
    """Wrapper for RhoFold+ model inference on GPU or CPU.

    Parameters
    ----------
    device : str or torch.device
        Target device ('cuda' or 'cpu'). Falls back to CPU if CUDA unavailable.
    weights_path : str or None
        Explicit path to pretrained weights file.  When provided, no download
        is attempted — useful for Kaggle offline submissions.
    """

    def __init__(self, device=None, weights_path=None):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        if not _RHOFOLD_AVAILABLE:
            self.model = None
            return

        ckpt_path = _download_weights(weights_path=weights_path)
        self.model = _RhoFoldModel.from_pretrained(ckpt_path)
        self.model.eval()
        self.model = torch.compile(
            self.model, backend="inductor", mode="reduce-overhead"
        )
        self.model.to(self.device)

    @property
    def available(self):
        """Return True if RhoFold model is loaded."""
        return self.model is not None

    def encode_sequence(self, sequence):
        """Convert RNA sequence to RhoFold input dict.

        Parameters
        ----------
        sequence : str
            RNA sequence string (A, C, G, U, N).

        Returns
        -------
        dict
            Input dict with keys 'seq', 'seq_mask', 'msa', 'msa_mask'.
        """
        L = len(sequence)
        seq_ids = [_NUC_MAP.get(c, _PAD_TOKEN) for c in sequence]
        seq_tensor = torch.tensor([seq_ids], dtype=torch.long)
        seq_mask = torch.ones(1, L, dtype=torch.float32)
        msa = torch.zeros(1, 1, L, 23, dtype=torch.float32)
        msa_mask = torch.ones(1, 1, L, dtype=torch.float32)
        return {
            'seq': seq_tensor,
            'seq_mask': seq_mask,
            'msa': msa,
            'msa_mask': msa_mask,
        }

    def predict(self, sequence, n_recycles=4, return_plddt=False):
        """Run single-sequence inference and return C3' coordinates.

        Parameters
        ----------
        sequence : str
            RNA sequence string.
        n_recycles : int
            Number of recycling iterations.
        return_plddt : bool
            If True, also return per-residue pLDDT confidence scores.

        Returns
        -------
        np.ndarray or tuple
            Shape (L, 3) float32 predicted C3' coordinates.
            If return_plddt=True, returns (coords, plddt) where plddt is
            shape (L,) float32 in [0, 1].
        """
        if not self.available:
            raise RuntimeError(
                "RhoFold model not loaded. Install with: "
                "pip install 'rhofold @ git+https://github.com/ml4bio/RhoFold.git'"
            )
        inputs = self.encode_sequence(sequence)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            output = self.model(**inputs, num_recycles=n_recycles)
        # cord_tns_pred: (1, L, 23, 3) — C3' is atom index 4
        coords = output['cord_tns_pred'][0, :, 4, :].cpu().numpy()
        coords = coords.astype(np.float32)

        if not return_plddt:
            return coords

        # Extract per-residue pLDDT confidence scores from model output
        plddt = self._extract_plddt(output, len(sequence))
        return coords, plddt

    def _extract_plddt(self, output, L):
        """Extract per-residue pLDDT confidence scores from RhoFold+ output.

        Parameters
        ----------
        output : dict
            RhoFold+ model output dict.
        L : int
            Sequence length.

        Returns
        -------
        np.ndarray
            Shape (L,) float32 pLDDT scores in [0, 1].
        """
        # RhoFold+ stores pLDDT in 'plddt' key; fall back to coordinate-based
        # proxy if key is absent (varies by RhoFold version)
        if 'plddt' in output:
            plddt = output['plddt'][0, :L].cpu().numpy().astype(np.float32)
            return np.clip(plddt, 0.0, 1.0)

        # Fallback: compute a proxy pLDDT from per-residue coordinate deviation
        # across atom types (lower deviation → higher confidence)
        all_atoms = output['cord_tns_pred'][0, :L, :, :]  # (L, 23, 3)
        centroid = all_atoms.mean(dim=1, keepdim=True)  # (L, 1, 3)
        mean_dist = torch.norm(all_atoms - centroid, dim=-1).mean(dim=-1)  # (L,)
        # Normalise to [0, 1]: low deviation → high confidence
        dist_np = mean_dist.cpu().numpy().astype(np.float32)
        max_dist = dist_np.max() + 1e-8
        plddt = 1.0 - (dist_np / max_dist)
        return np.clip(plddt, 0.0, 1.0).astype(np.float32)

    def predict_batch(self, sequences, n_recycles=4):
        """Run batched inference on multiple sequences.

        Sequences are padded to the same length within the batch.

        Parameters
        ----------
        sequences : list of str
            RNA sequences.
        n_recycles : int
            Number of recycling iterations.

        Returns
        -------
        list of np.ndarray
            List of (L_i, 3) float32 coordinate arrays.
        """
        if not self.available:
            raise RuntimeError("RhoFold model not loaded.")

        lengths = [len(s) for s in sequences]
        max_len = max(lengths)
        batch_size = len(sequences)

        # Pad and stack inputs
        seq_batch = torch.full((batch_size, max_len), _PAD_TOKEN, dtype=torch.long)
        mask_batch = torch.zeros(batch_size, max_len, dtype=torch.float32)
        msa_batch = torch.zeros(batch_size, 1, max_len, 23, dtype=torch.float32)
        msa_mask_batch = torch.zeros(batch_size, 1, max_len, dtype=torch.float32)

        for i, seq in enumerate(sequences):
            enc = self.encode_sequence(seq)
            L = lengths[i]
            seq_batch[i, :L] = enc['seq'][0, :L]
            mask_batch[i, :L] = 1.0
            msa_batch[i, :, :L, :] = enc['msa'][0, :, :L, :]
            msa_mask_batch[i, :, :L] = 1.0

        inputs = {
            'seq': seq_batch.to(self.device),
            'seq_mask': mask_batch.to(self.device),
            'msa': msa_batch.to(self.device),
            'msa_mask': msa_mask_batch.to(self.device),
        }
        with torch.no_grad():
            output = self.model(**inputs, num_recycles=n_recycles)

        # Unpad outputs
        all_coords = output['cord_tns_pred']  # (B, max_L, 23, 3)
        results = []
        for i, L in enumerate(lengths):
            coords = all_coords[i, :L, 4, :].cpu().numpy().astype(np.float32)
            results.append(coords)
        return results

    def generate_ensemble(self, sequence, n_candidates=50, n_recycles_list=None,
                          return_plddt=False):
        """Generate an ensemble of structure predictions via varied recycling.

        Uses dropout-active passes for diversity, deterministic passes for accuracy.
        Pipelines GPU inference with CPU preprocessing using CUDA streams.

        Parameters
        ----------
        sequence : str
            RNA sequence string.
        n_candidates : int
            Number of candidate structures to generate.
        n_recycles_list : list of int or None
            Per-candidate recycle counts. If None, cycles [1,2,3,4,4,4].
        return_plddt : bool
            If True, also return per-residue pLDDT scores for each candidate.

        Returns
        -------
        list of np.ndarray or tuple
            List of n_candidates (L, 3) float32 coordinate arrays.
            If return_plddt=True, returns (candidates, plddts) where plddts
            is a list of (L,) float32 arrays.
        """
        if not self.available:
            raise RuntimeError("RhoFold model not loaded.")

        if n_recycles_list is None:
            base_recycles = [1, 2, 3, 4, 4, 4]
            n_recycles_list = [
                base_recycles[i % len(base_recycles)]
                for i in range(n_candidates)
            ]

        inputs = self.encode_sequence(sequence)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        candidates = []
        plddts = []
        use_streams = self.device.type == 'cuda'

        if use_streams:
            stream_infer = torch.cuda.Stream()
            stream_prep = torch.cuda.Stream()

        for idx, n_rec in enumerate(n_recycles_list):
            # Enable dropout for diversity when recycles < 4
            if n_rec < 4:
                self.model.train()
            else:
                self.model.eval()

            if use_streams:
                with torch.cuda.stream(stream_infer):
                    with torch.no_grad():
                        output = self.model(**inputs, num_recycles=n_rec)
                    coords = output['cord_tns_pred'][0, :, 4, :].cpu().numpy()
                torch.cuda.current_stream().wait_stream(stream_infer)
            else:
                with torch.no_grad():
                    output = self.model(**inputs, num_recycles=n_rec)
                coords = output['cord_tns_pred'][0, :, 4, :].cpu().numpy()

            candidates.append(coords.astype(np.float32))
            if return_plddt:
                plddts.append(self._extract_plddt(output, len(sequence)))

        # Restore eval mode
        self.model.eval()

        if return_plddt:
            return candidates, plddts
        return candidates


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    runner = RhoFoldRunner(device)

    # Test encode_sequence always works
    seq = "AUGCAUGC"
    inputs = runner.encode_sequence(seq)
    assert inputs['seq'].shape == (1, 8)
    assert inputs['seq'].dtype == torch.long
    assert inputs['seq_mask'].shape == (1, 8)
    assert inputs['seq_mask'].dtype == torch.float32
    assert inputs['msa'].shape == (1, 1, 8, 23)
    assert inputs['msa_mask'].shape == (1, 1, 8)
    # Verify encoding
    expected = [0, 3, 2, 1, 0, 3, 2, 1]  # A=0, U=3, G=2, C=1
    assert inputs['seq'][0].tolist() == expected

    # Test offline weights_path rejection for nonexistent file
    try:
        _download_weights(weights_path="/nonexistent/weights.pt")
        assert False, "Should have raised FileNotFoundError"
    except FileNotFoundError:
        pass

    if runner.available:
        coords = runner.predict(seq)
        print(f"predict shape: {coords.shape}")
        assert coords.shape == (8, 3)
        assert coords.dtype == np.float32
    else:
        print("RhoFold not installed — encode_sequence test passed, "
              "predict/generate_ensemble skipped.")

    print("rhofold_runner self-test PASSED")
