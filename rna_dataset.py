"""
rna_dataset.py — Stanford RNA 3D Folding 2 dataset loader
==========================================================
Provides :class:`StanfordRNA3DDataset`, a lightweight wrapper around the
competition CSV files that maps every ``target_id`` to its nucleotide
sequence and (L, 5, 3) float32 coordinate array (five conformers, three
spatial dimensions).

Coordinate columns in the labels CSV follow the naming convention
``x_1 … x_5``, ``y_1 … y_5``, ``z_1 … z_5``.  Rows are in long format
(one row per residue) and are sorted by ``resid`` before stacking.

Supported data roots (checked in order):
  1. ``data_root`` passed to the constructor
  2. ``./data/stanford-rna-3d-folding/``
  3. ``/kaggle/input/stanford-rna-3d-folding/``

Usage
-----
>>> ds = StanfordRNA3DDataset(split="train")
>>> seq = ds.get_sequence("R1107")
>>> coords = ds.get_coords("R1107")   # (L, 5, 3) float32
>>> all_ids = ds.target_ids()
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

_DEFAULT_ROOTS = [
    os.path.join(".", "data", "stanford-rna-3d-folding"),
    os.path.join("/", "kaggle", "input", "stanford-rna-3d-folding"),
]

_COORD_X_COLS = [f"x_{i}" for i in range(1, 6)]
_COORD_Y_COLS = [f"y_{i}" for i in range(1, 6)]
_COORD_Z_COLS = [f"z_{i}" for i in range(1, 6)]
_N_CONFORMERS = 5

_SPLIT_FILES = {
    "train":      {"sequences": "train_sequences.csv",      "labels": "train_labels.csv"},
    "validation": {"sequences": "train_sequences.csv",      "labels": "validation_labels.csv"},
    "test":       {"sequences": "test_sequences.csv",       "labels": None},
}

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────


def _resolve_data_root(data_root: Optional[str]) -> str:
    """
    Return the first existing data directory from *data_root* or the
    built-in search list.

    Parameters
    ----------
    data_root : str or None
        Explicit path.  When ``None``, the default search paths are tried.

    Returns
    -------
    str
        Resolved directory path.

    Raises
    ------
    FileNotFoundError
        If no candidate directory exists.
    """
    candidates = [data_root] if data_root else _DEFAULT_ROOTS
    for path in candidates:
        if path and os.path.isdir(path):
            return path
    raise FileNotFoundError(
        f"No data directory found.  Searched: {candidates}"
    )


def _detect_id_column(df: pd.DataFrame) -> str:
    """
    Auto-detect the identifier column in a DataFrame.

    Looks for ``target_id``, ``ID``, ``id`` (in that order), then falls
    back to the first column.
    """
    for name in ("target_id", "ID", "id"):
        if name in df.columns:
            return name
    for col in df.columns:
        if "id" in col.lower():
            return col
    return df.columns[0]


def _detect_resid_column(df: pd.DataFrame) -> Optional[str]:
    """
    Auto-detect the residue-index column (``resid``, ``res_id``, etc.).

    Returns ``None`` when no such column is found — the caller will then
    parse residue indices from the ``ID`` column instead.
    """
    for name in ("resid", "res_id", "residue_id"):
        if name in df.columns:
            return name
    for col in df.columns:
        if "resid" in col.lower():
            return col
    return None


def _extract_target_and_resid(raw_id: str):
    """
    Split a long-format ID ``'{target_id}_{resid}'`` into its parts.

    Returns
    -------
    tuple[str, int]
        ``(target_id, resid)``
    """
    raw = str(raw_id)
    parts = raw.rsplit("_", 1)
    if len(parts) == 2 and parts[1].isdigit():
        return parts[0], int(parts[1])
    return raw, 0


def _parse_labels(
    labels_df: pd.DataFrame,
) -> Dict[str, np.ndarray]:
    """
    Parse a long-format labels DataFrame into per-target coordinate arrays.

    Parameters
    ----------
    labels_df : pd.DataFrame
        Must contain an ID/target_id column, coordinate columns
        ``x_1 … x_5``, ``y_1 … y_5``, ``z_1 … z_5``, and optionally
        ``resid`` and ``resname`` columns.

    Returns
    -------
    dict[str, np.ndarray]
        Mapping from target_id to ``(L, 5, 3)`` float32 coordinate array,
        with rows sorted by resid.
    """
    id_col = _detect_id_column(labels_df)
    resid_col = _detect_resid_column(labels_df)

    # Detect coordinate columns flexibly
    x_cols = [c for c in _COORD_X_COLS if c in labels_df.columns]
    y_cols = [c for c in _COORD_Y_COLS if c in labels_df.columns]
    z_cols = [c for c in _COORD_Z_COLS if c in labels_df.columns]

    if not x_cols or not y_cols or not z_cols:
        raise ValueError(
            f"Could not find expected coordinate columns (x_1..x_5, y_1..y_5, "
            f"z_1..z_5).  Columns present: {list(labels_df.columns)}"
        )

    n_conf = min(len(x_cols), len(y_cols), len(z_cols))

    # Detect explicit target_id column vs composite ID column
    has_target_col = "target_id" in labels_df.columns
    labels_df = labels_df.copy()

    if has_target_col and resid_col is not None:
        # Both target_id and resid columns present — use directly
        labels_df["_target"] = labels_df["target_id"].astype(str)
        labels_df["_resid"] = labels_df[resid_col].astype(int)
    elif resid_col is not None:
        # resid column present but ID is composite — parse target from ID
        labels_df["_target"] = labels_df[id_col].apply(
            lambda x: _extract_target_and_resid(x)[0]
        )
        labels_df["_resid"] = labels_df[resid_col].astype(int)
    else:
        # No resid column — parse both from composite ID
        parsed = labels_df[id_col].apply(_extract_target_and_resid)
        labels_df["_target"] = parsed.apply(lambda t: t[0])
        labels_df["_resid"] = parsed.apply(lambda t: t[1])

    target_coords: Dict[str, np.ndarray] = {}

    for tid, grp in labels_df.groupby("_target"):
        grp = grp.sort_values("_resid")
        L = len(grp)
        # Always (L, 5, 3) per spec; unfilled conformers remain NaN.
        coords = np.full((L, _N_CONFORMERS, 3), np.nan, dtype=np.float32)
        for ci in range(n_conf):
            xc, yc, zc = x_cols[ci], y_cols[ci], z_cols[ci]
            coords[:, ci, 0] = grp[xc].values.astype(np.float32)
            coords[:, ci, 1] = grp[yc].values.astype(np.float32)
            coords[:, ci, 2] = grp[zc].values.astype(np.float32)
        target_coords[str(tid)] = coords

    return target_coords


def _parse_sequences(seq_df: pd.DataFrame) -> Dict[str, str]:
    """
    Parse a sequences DataFrame into a ``{target_id: sequence}`` mapping.
    """
    id_col = _detect_id_column(seq_df)
    seq_col = "sequence"
    if seq_col not in seq_df.columns:
        for c in seq_df.columns:
            if "seq" in c.lower():
                seq_col = c
                break
    return {
        str(tid): str(seq)
        for tid, seq in zip(seq_df[id_col], seq_df[seq_col])
    }


# ─────────────────────────────────────────────────────────────────────────────
# Dataset class
# ─────────────────────────────────────────────────────────────────────────────


class StanfordRNA3DDataset:
    """
    Lightweight dataset for the Stanford RNA 3D Folding 2 competition.

    Parameters
    ----------
    split : str
        One of ``'train'``, ``'validation'``, or ``'test'``.
    data_root : str or None
        Path to the directory containing the CSV files.  When ``None``,
        the class searches ``./data/stanford-rna-3d-folding/`` and
        ``/kaggle/input/stanford-rna-3d-folding/`` in order.

    Attributes
    ----------
    split : str
        Active split name.
    data_root : str
        Resolved data directory.
    """

    def __init__(self, split: str = "train", data_root: Optional[str] = None):
        if split not in _SPLIT_FILES:
            raise ValueError(
                f"Unknown split '{split}'.  Expected one of {list(_SPLIT_FILES)}"
            )
        self.split = split
        self.data_root = _resolve_data_root(data_root)

        files = _SPLIT_FILES[split]

        # ── Load sequences ────────────────────────────────────────────────
        seq_path = os.path.join(self.data_root, files["sequences"])
        seq_df = pd.read_csv(seq_path)
        self._sequences: Dict[str, str] = _parse_sequences(seq_df)

        # ── Load labels (coordinates) ─────────────────────────────────────
        self._coords: Dict[str, np.ndarray] = {}
        if files["labels"] is not None:
            lab_path = os.path.join(self.data_root, files["labels"])
            if os.path.isfile(lab_path):
                lab_df = pd.read_csv(lab_path)
                self._coords = _parse_labels(lab_df)

    # ── Public API ────────────────────────────────────────────────────────

    def target_ids(self) -> List[str]:
        """
        Return the list of all target IDs available in this split.

        Returns
        -------
        list[str]
            Sorted list of target identifiers.
        """
        ids = set(self._sequences.keys())
        if self._coords:
            ids |= set(self._coords.keys())
        return sorted(ids)

    def get_sequence(self, target_id: str) -> str:
        """
        Return the nucleotide sequence for *target_id*.

        Parameters
        ----------
        target_id : str
            Unique identifier, e.g. ``'R1107'``.

        Returns
        -------
        str
            RNA sequence string (e.g. ``'AUGC…'``).

        Raises
        ------
        KeyError
            If *target_id* is not found.
        """
        if target_id not in self._sequences:
            raise KeyError(f"target_id '{target_id}' not found in sequences")
        return self._sequences[target_id]

    def get_coords(self, target_id: str) -> np.ndarray:
        """
        Return 3D coordinates for *target_id*.

        Parameters
        ----------
        target_id : str
            Unique identifier.

        Returns
        -------
        np.ndarray
            Shape ``(L, 5, 3)`` float32 array — *L* residues, 5 conformers,
            3 spatial dimensions (x, y, z).  Missing conformers are ``NaN``.

        Raises
        ------
        KeyError
            If *target_id* has no associated coordinates.
        """
        if target_id not in self._coords:
            raise KeyError(
                f"target_id '{target_id}' not found in coordinate labels"
            )
        return self._coords[target_id]


# ─────────────────────────────────────────────────────────────────────────────
# Self-test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import time
    import tempfile

    print("=" * 70)
    print("  rna_dataset.py — self-test on L=50 synthetic data")
    print("=" * 70)

    L = 50
    N_TARGETS = 4
    rng = np.random.default_rng(42)

    # ── Build synthetic CSV files in a temp directory ─────────────────────
    tmpdir = tempfile.mkdtemp(prefix="rna_dataset_test_")
    print(f"\n  Temp dir: {tmpdir}")

    # Synthetic sequences
    targets = [f"SYNTH_{i:03d}" for i in range(N_TARGETS)]
    sequences = [
        "".join(rng.choice(list("AUGC"), size=L)) for _ in range(N_TARGETS)
    ]
    seq_df = pd.DataFrame({"target_id": targets, "sequence": sequences})
    seq_df.to_csv(os.path.join(tmpdir, "train_sequences.csv"), index=False)
    seq_df.to_csv(os.path.join(tmpdir, "test_sequences.csv"), index=False)

    # Synthetic labels (long format)
    rows = []
    for tid, seq in zip(targets, sequences):
        for r in range(L):
            row = {"ID": f"{tid}_{r}", "resid": r, "resname": seq[r]}
            for ci in range(1, 6):
                xyz = rng.normal(0.0, 15.0, 3).astype(np.float32)
                row[f"x_{ci}"] = float(xyz[0])
                row[f"y_{ci}"] = float(xyz[1])
                row[f"z_{ci}"] = float(xyz[2])
            rows.append(row)
    lab_df = pd.DataFrame(rows)
    lab_df.to_csv(os.path.join(tmpdir, "train_labels.csv"), index=False)
    lab_df.to_csv(os.path.join(tmpdir, "validation_labels.csv"), index=False)

    # ── Test 1: load train split ──────────────────────────────────────────
    print("\n" + "─" * 70)
    print("  Test 1: load train split")
    t0 = time.perf_counter()
    ds = StanfordRNA3DDataset(split="train", data_root=tmpdir)
    dt = time.perf_counter() - t0
    ids = ds.target_ids()
    print(f"    Loaded {len(ids)} targets in {dt:.4f}s")
    assert len(ids) == N_TARGETS, f"Expected {N_TARGETS}, got {len(ids)}"
    print("    ✓ target count correct")

    # ── Test 2: get_sequence ──────────────────────────────────────────────
    print("\n" + "─" * 70)
    print("  Test 2: get_sequence")
    for tid, expected_seq in zip(targets, sequences):
        seq = ds.get_sequence(tid)
        assert seq == expected_seq, f"Sequence mismatch for {tid}"
    print(f"    ✓ all {N_TARGETS} sequences match")

    # ── Test 3: get_coords shape and dtype ────────────────────────────────
    print("\n" + "─" * 70)
    print("  Test 3: get_coords shape + dtype")
    for tid in targets:
        coords = ds.get_coords(tid)
        assert coords.shape == (L, 5, 3), (
            f"Expected shape ({L}, 5, 3), got {coords.shape}"
        )
        assert coords.dtype == np.float32, (
            f"Expected float32, got {coords.dtype}"
        )
    print(f"    ✓ shape=({L}, 5, 3) float32 for all targets")

    # ── Test 4: coords are finite ─────────────────────────────────────────
    print("\n" + "─" * 70)
    print("  Test 4: coords finite (no NaN / Inf)")
    for tid in targets:
        coords = ds.get_coords(tid)
        assert np.all(np.isfinite(coords)), f"Non-finite values in {tid}"
    print("    ✓ all coordinates finite")

    # ── Test 5: resid sorting ─────────────────────────────────────────────
    print("\n" + "─" * 70)
    print("  Test 5: resid sorting preserved")
    # Shuffle rows and reload — should still produce same coords
    shuffled = lab_df.sample(frac=1.0, random_state=7)
    shuffled.to_csv(os.path.join(tmpdir, "train_labels.csv"), index=False)
    ds2 = StanfordRNA3DDataset(split="train", data_root=tmpdir)
    for tid in targets:
        c1 = ds.get_coords(tid)
        c2 = ds2.get_coords(tid)
        assert np.allclose(c1, c2), f"Sorting mismatch for {tid}"
    print("    ✓ shuffled CSV reloads identically after sort-by-resid")

    # ── Test 6: validation split ──────────────────────────────────────────
    print("\n" + "─" * 70)
    print("  Test 6: validation split loads")
    ds_val = StanfordRNA3DDataset(split="validation", data_root=tmpdir)
    assert len(ds_val.target_ids()) == N_TARGETS
    print("    ✓ validation split OK")

    # ── Test 7: test split (no labels) ────────────────────────────────────
    print("\n" + "─" * 70)
    print("  Test 7: test split (sequences only)")
    ds_test = StanfordRNA3DDataset(split="test", data_root=tmpdir)
    assert len(ds_test.target_ids()) == N_TARGETS
    try:
        ds_test.get_coords(targets[0])
        assert False, "Should have raised KeyError"
    except KeyError:
        pass
    print("    ✓ test split: sequences present, get_coords raises KeyError")

    # ── Test 8: KeyError for unknown target ───────────────────────────────
    print("\n" + "─" * 70)
    print("  Test 8: KeyError for unknown target_id")
    try:
        ds.get_sequence("NONEXISTENT_999")
        assert False, "Should have raised KeyError"
    except KeyError:
        pass
    try:
        ds.get_coords("NONEXISTENT_999")
        assert False, "Should have raised KeyError"
    except KeyError:
        pass
    print("    ✓ KeyError raised correctly")

    # ── Cleanup ───────────────────────────────────────────────────────────
    import shutil
    shutil.rmtree(tmpdir)

    print("\n" + "=" * 70)
    print("  [rna_dataset] All tests passed.")
    print("=" * 70)
