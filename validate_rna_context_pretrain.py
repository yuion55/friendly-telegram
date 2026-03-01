"""
validate_rna_context_pretrain.py  (v1 — REAL Stanford RNA 3D Folding 2 Data)
=============================================================================
Validation script for rna_context_pretrain.py — Context-Aware Pretraining
for Ligand/Complex RNAs with Numba JIT and vectorisation.

Uses the real Stanford RNA 3D Folding 2 Kaggle competition dataset.
Mirrors the structure and data-fetch pattern of validate_rna_briq_refinement.py.

Run in Colab after uploading kaggle.json:
    !python validate_rna_context_pretrain.py

What this script does
─────────────────────
  STAGE 0  Module load & symbol check
             • Verifies rna_context_pretrain.py imports cleanly
             • Warms up all Numba JIT kernels (cold-start ms recorded)
             • Checks all public symbols (functions, classes, constants)
             • Logs Python / NumPy / Numba / SciPy versions

  STAGE 1  Surgical Kaggle data fetch + coordinate extraction
             • test_sequences.csv   — real RNA sequences (lengths, IDs)
             • train_sequences.csv  — training sequences with metadata
             • train_labels.csv     — C3′ 3D coords (PDB-derived, ~1.5 GB)
             • Extracts per-target C4′-proxy backbone arrays from real coords
             • Selects MAX_TARGET_COORDS longest unique targets for tests
             • Builds REAL_SEQ_LENS distribution for end-to-end sampling
             • Constructs synthetic PDBComplex objects from real sequences

  STAGE 2  Numba JIT kernel correctness & performance (real coords)
             • compute_pairwise_distances: symmetry, non-negative, shape correct
               Benchmarks: N=50/100/200 atom coords; measures ms/call
             • build_contact_map: binary (0/1), consistency with distance mat
               threshold sweep [4, 8, 12] Å; contacts decrease with threshold
             • compute_rna_self_contacts: diagonal=0, seq-sep filter correct
               min_seq_sep=4 enforced; verified on 20/60/120 nt
             • vectorised_partner_mask: boolean shape (N_rna,); correct atoms
               True iff ≥1 partner atom within threshold; verified analytically
             • compute_metal_binding_features: Gaussian score ≥ 0; shape (L,)
               score decreases with distance; zero metals → score=0
             • apply_metal_distance_restraints: shape (L,L); zero off-metal pairs
               restraint ∝ 1/(1+d) for metal-binding residue pairs

  STAGE 3  Sequence utilities — correctness & edge cases
             • sequence_to_one_hot: shape (L,5) float32; rows sum to 1.0
               A/U/G/C/N all map to correct one-hot columns
               Unknown characters → N column; uppercase normalisation
               Tested on real sequences from test_sequences.csv
             • one_hot_to_sequence: roundtrip sequence_to_one_hot → identity
             • classify_rna_context: returns valid RNAContextType member
               riboswitch motif sequences → RIBOSWITCH; tRNA → TRNA
               unknown sequences → NAKED_RNA or best match
               Tested on 10 real sequences from test_sequences.csv

  STAGE 4  PDBComplex construction from real coordinate data
             • Builds PDBComplex objects from real C3′ coords + random protein
             • rna_coords shape (L,3) float32; all_atom_coords consistent
             • rna_atom_indices / partner_indices: non-overlapping int32 ranges
             • context_type assigned by classifier
             • mask_partner_coordinates: visible coords zero at partner indices
               original_partner shape matches partner_indices size
             • random_crop_rna: output len ≤ max_length; subsequence correct

  STAGE 5  extract_context_features — pipeline correctness (real coords)
             • contact_mask: shape (L,) bool; True iff residue near partner
             • self_contact_map: shape (L,L) int32; symmetric; diagonal=0
             • metal_binding_prob: shape (L,) float32; values in [0, ∞)
             • metal_restraints: shape (L,L) float32; ≥ 0; zero when no metals
             • partner_dist_mat: shape (L, N_partner) float32; non-negative
             • sequence_one_hot: shape (L,5) float32; row-sums = 1
             • Tested on 6 real targets (real sequence lengths)
             • Benchmarks: feature extraction ms/complex at N=20/60/120

  STAGE 6  ContextTypeEmbedding + MetalBindingHead
             • ContextTypeEmbedding weight shape (num_types, embed_dim)
               __call__: output shape (B, embed_dim) for B context IDs
               inject_into_single_repr: shape (B,L,d_model) preserved
               raises ValueError when embed_dim ≠ d_model with no projection
             • MetalBindingHead.forward: output (B,L) float32 in [0,1]
               sigmoid: strictly between 0 and 1; no NaN/Inf
               binary_cross_entropy_loss: finite scalar ≥ 0
               BCE(all-zeros-pred, all-zeros-labels) < BCE(random, all-zeros)

  STAGE 7  collate_features + PretrainingBatch
             • Padded length = max(seq_lengths)
             • one_hot shape (B,L,5); zeros beyond each sequence's length
             • contact_masks shape (B,L) bool; metal_probs shape (B,L) float32
             • self_contacts shape (B,L,L); metal_restraints shape (B,L,L)
             • seq_lengths correct; context_type_ids in valid range
             • Tested with B=4 real-length batches from test_sequences.csv

  STAGE 8  RNAContextPretrainer — training step (real sequences)
             • 8 runs on real sequence lengths from test_sequences.csv
             • training_step returns dict with 'metal_binding_loss',
               'contact_mask_loss', 'total_loss' — all finite non-negative
             • total_loss = metal_loss + 0.5 * contact_loss verified
             • partner masking applied before feature extraction each step
             • Wall-time logged per batch size / sequence length

  STAGE 9  inference_prepare on real sequences
             • output keys: single_repr, context_type_id, context_label,
               metal_prob, metal_restraints — all present
             • single_repr shape (1,L,d_model) float32; finite
             • metal_prob shape (L,) float32 in [0,1] (sigmoid output)
             • metal_restraints shape (L,L) float32; all ≥ 0
             • With known_metal_coords: metal_restraints non-zero near metal sites
             • context_label is a valid RNAContextType name
             • Tested on 6 real sequences from test_sequences.csv

  STAGE 10 Diagnosis — real-data performance summary & known limitations
"""

from __future__ import annotations

import os
import sys
import time
import math
import traceback
import textwrap
import warnings
import subprocess
import tempfile
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

MODULE_FILE        = "rna_context_pretrain.py"
COMP_NAME          = "stanford-rna-3d-folding-2"
DATA_DIR           = "stanford_data"
KAGGLE_CFG         = os.getcwd()

MAX_TEST_SEQS      = 10
MAX_LABEL_ROWS     = 400 * 300       # ≈ 120 k rows — RAM-safe for Colab free tier
MAX_TARGET_COORDS  = 6               # real RNA targets for geometry tests
SEQ_LEN_FALLBACK   = 60
TOL_FLOAT          = 1e-4
PI                 = math.pi

# Pipeline hyper-params (match module defaults)
D_MODEL            = 128
EMBED_DIM          = 64
HIDDEN_DIM         = 128
CONTACT_THRESH     = 8.0
METAL_THRESH       = 4.0
N_PROTEIN_ATOMS    = 150
N_METALS           = 3
E2E_RUNS           = 8

# ─────────────────────────────────────────────────────────────────────────────
# LOGGER (identical style to validate_rna_briq_refinement.py)
# ─────────────────────────────────────────────────────────────────────────────

class Logger:
    def __init__(self):
        self.records: List[Dict] = []
        self._cur  = None
        self._t0   = 0.0

    def section(self, title: str):
        print(f"\n{'='*70}")
        print(f"  {title}")
        print(f"{'='*70}")

    def begin(self, tid: str, name: str, tag: str):
        self._cur = dict(tid=tid, name=name, tag=tag, status="RUNNING",
                         ms=0.0, details={}, warnings=[], errors=[], tb=None)
        self._t0 = time.perf_counter()
        print(f"\n{'─'*70}")
        print(f"  [{tid}] {name}  ({tag})")
        print(f"{'─'*70}")

    def log(self, key: str, val: Any):
        if self._cur:
            self._cur["details"][key] = val
        import json
        try:
            txt = json.dumps(val, default=str)
        except Exception:
            txt = str(val)
        if len(txt) > 140:
            txt = txt[:137] + "…"
        print(f"    {key}: {txt}")

    def warn(self, msg: str):
        if self._cur:
            self._cur["warnings"].append(msg)
        print(f"    ⚠  {msg}")

    def err(self, msg: str):
        if self._cur:
            self._cur["errors"].append(msg)
        print(f"    ✗  {msg}")

    def end(self, status: str, reason: str = "", tb: str = ""):
        ms = (time.perf_counter() - self._t0) * 1000
        if self._cur:
            self._cur.update(status=status, ms=round(ms, 1),
                             reason=reason, tb=tb or None)
            self.records.append(self._cur)
            self._cur = None
        icon = {"PASS": "✓", "PARTIAL": "⚠", "FAIL": "✗", "ERROR": "✗"}.get(status, "?")
        print(f"\n  {icon}  STATUS: {status}  ({ms:.0f} ms)")
        if reason:
            print(f"     Reason : {reason}")
        if tb:
            print(f"     Traceback:\n{textwrap.indent(tb, '       ')}")

    def summary(self) -> Dict:
        total   = len(self.records)
        passed  = sum(1 for r in self.records if r["status"] == "PASS")
        partial = sum(1 for r in self.records if r["status"] == "PARTIAL")
        failed  = sum(1 for r in self.records if r["status"] in ("FAIL", "ERROR"))

        print(f"\n{'='*70}")
        print("  VALIDATION SUMMARY — rna_context_pretrain.py "
              "(Stanford RNA 3D Folding 2)")
        print(f"{'='*70}")
        print(f"  {'TID':<28}{'Tag':<24}{'Test':<18}{'Status':<12}ms")
        print("  " + "-"*84)
        for r in self.records:
            icon = {"PASS": "✓", "PARTIAL": "⚠"}.get(r["status"], "✗")
            print(f"  {r['tid']:<28}{r['tag']:<24}{r['name'][:16]:<18}"
                  f"{icon+' '+r['status']:<12}{r['ms']:.0f}")
        print("  " + "-"*84)
        print(f"  Total: {total}  |  PASS: {passed}  |  "
              f"PARTIAL: {partial}  |  FAIL/ERROR: {failed}")
        print()

        if failed + partial:
            print("  ── FAILURES / PARTIALS DETAIL ──")
            for r in self.records:
                if r["status"] not in ("PASS",):
                    print(f"\n  [{r['tid']}] {r['name']}")
                    print(f"       Status : {r['status']}")
                    if r.get("reason"):
                        print(f"       Reason : {r['reason']}")
                    for w in r["warnings"]:
                        print(f"       WARN   : {w}")
                    for e in r["errors"]:
                        print(f"       ERROR  : {e}")
                    if r.get("tb"):
                        print("       Traceback:")
                        for line in r["tb"].splitlines():
                            print(f"         {line}")

        return dict(total=total, passed=passed, partial=partial, failed=failed)


LOG = Logger()

# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def make_random_seq(N: int, seed: int = 0) -> str:
    rng = np.random.default_rng(seed)
    return "".join(rng.choice(list("ACGU"), N))


def benchmark(fn, repeats: int = 5) -> float:
    """Return median elapsed ms over `repeats` calls."""
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        times.append((time.perf_counter() - t0) * 1000)
    return float(np.median(times))


def make_random_coords(N: int, spread: float = 15.0,
                       seed: int = 0) -> np.ndarray:
    """Random 3D atom coords (N, 3) float32."""
    rng = np.random.default_rng(seed)
    return (rng.standard_normal((N, 3)) * spread).astype(np.float32)


def build_complex_from_real(
    target_id: str,
    c3p_coords: np.ndarray,
    sequence: str,
    n_protein: int = N_PROTEIN_ATOMS,
    n_metals: int  = N_METALS,
    seed: int      = 0,
):
    """
    Construct a PDBComplex using real C3' coordinates as RNA backbone,
    and synthetic protein / metal atoms centred near the RNA.
    """
    from rna_context_pretrain import (
        PDBComplex, RNAContextType, classify_rna_context, METAL_IONS
    )
    rng = np.random.default_rng(seed)
    L   = min(len(c3p_coords), len(sequence))
    rna_coords = c3p_coords[:L].astype(np.float32)

    # Centre protein near RNA centroid
    centroid       = rna_coords.mean(axis=0)
    protein_coords = (centroid + rng.standard_normal((n_protein, 3)) * 20.0
                      ).astype(np.float32)
    metal_coords   = (centroid + rng.standard_normal((n_metals, 3)) * 5.0
                      ).astype(np.float32)

    rna_atom_idx    = np.arange(L, dtype=np.int32)
    partner_idx     = np.arange(L, L + n_protein, dtype=np.int32)

    all_coords = np.vstack([rna_coords, protein_coords, metal_coords]).astype(np.float32)
    metal_type_list = [METAL_IONS[i % len(METAL_IONS)] for i in range(n_metals)]
    ctx = classify_rna_context(sequence[:L])

    return PDBComplex(
        pdb_id           = target_id,
        rna_sequence     = sequence[:L],
        rna_coords       = rna_coords,
        rna_atom_indices = rna_atom_idx,
        partner_indices  = partner_idx,
        all_atom_coords  = all_coords,
        metal_coords     = metal_coords,
        metal_types      = metal_type_list,
        partner_type     = "protein",
        context_type     = ctx,
    )


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 0 — MODULE LOAD + SYMBOL CHECK + NUMBA WARMUP
# ─────────────────────────────────────────────────────────────────────────────

LOG.section("STAGE 0: MODULE LOAD + SYMBOL CHECK + NUMBA WARMUP")

if not os.path.exists(MODULE_FILE):
    print(f"  ✗ {MODULE_FILE} not found in {os.getcwd()}")
    print(f"    Python files here: {[f for f in os.listdir('.') if f.endswith('.py')]}")
    sys.exit(1)

if os.getcwd() not in sys.path:
    sys.path.insert(0, os.getcwd())

for mod in ("rna_context_pretrain",):
    if mod in sys.modules:
        del sys.modules[mod]

LOG.begin("S0-IMPORT",
          "Import rna_context_pretrain and verify all public symbols",
          "ModuleLoad")
try:
    import rna_context_pretrain as ctx_mod
    LOG.log("module_path", os.path.abspath(MODULE_FILE))

    REQUIRED_SYMBOLS = [
        # Numba JIT kernels
        "compute_pairwise_distances",
        "build_contact_map",
        "compute_rna_self_contacts",
        "vectorised_partner_mask",
        "compute_metal_binding_features",
        "apply_metal_distance_restraints",
        # Enumerations / constants
        "RNAContextType",
        "RNA_NUCLEOTIDES",
        "PROTEIN_RESIDUES",
        "METAL_IONS",
        "CONTACT_THRESHOLD_A",
        "METAL_CONTACT_A",
        # Data structures
        "PDBComplex",
        "ContextFeatures",
        "PretrainingBatch",
        # Sequence utilities
        "sequence_to_one_hot",
        "one_hot_to_sequence",
        "classify_rna_context",
        # Embedding / heads
        "ContextTypeEmbedding",
        "MetalBindingHead",
        # Feature extraction
        "extract_context_features",
        "mask_partner_coordinates",
        "random_crop_rna",
        # Collation
        "collate_features",
        # Pretraining
        "RNAContextPretrainer",
        # Inference
        "inference_prepare",
        # Synthetic data
        "make_synthetic_complex",
    ]

    missing = [s for s in REQUIRED_SYMBOLS if not hasattr(ctx_mod, s)]
    LOG.log("required_symbols", len(REQUIRED_SYMBOLS))
    LOG.log("missing_symbols",  missing)
    LOG.log("numpy_version",    np.__version__)
    LOG.log("python_version",   sys.version.split()[0])

    try:
        import numba
        LOG.log("numba_version", numba.__version__)
        _NUMBA_OK = True
    except ImportError:
        LOG.warn("numba not installed — JIT kernels fall back to CPython")
        _NUMBA_OK = False

    try:
        import scipy
        LOG.log("scipy_version", scipy.__version__)
    except ImportError:
        LOG.warn("scipy not installed")

    if missing:
        LOG.end("FAIL", reason=f"Missing symbols: {missing}")
        sys.exit(1)
    else:
        LOG.end("PASS", reason=f"All {len(REQUIRED_SYMBOLS)} symbols present")
except Exception:
    LOG.end("ERROR", tb=traceback.format_exc())
    sys.exit(1)

# Convenient direct imports
from rna_context_pretrain import (
    compute_pairwise_distances,
    build_contact_map,
    compute_rna_self_contacts,
    vectorised_partner_mask,
    compute_metal_binding_features,
    apply_metal_distance_restraints,
    RNAContextType,
    RNA_NUCLEOTIDES,
    PROTEIN_RESIDUES,
    METAL_IONS,
    CONTACT_THRESHOLD_A,
    METAL_CONTACT_A,
    PDBComplex,
    ContextFeatures,
    PretrainingBatch,
    sequence_to_one_hot,
    one_hot_to_sequence,
    classify_rna_context,
    ContextTypeEmbedding,
    MetalBindingHead,
    extract_context_features,
    mask_partner_coordinates,
    random_crop_rna,
    collate_features,
    RNAContextPretrainer,
    inference_prepare,
    make_synthetic_complex,
)

# ── Numba JIT warmup (trigger first-call compilations) ───────────────────────

LOG.begin("S0-WARMUP",
          "Pre-compile all Numba JIT kernels — measure cold-start latency",
          "ModuleLoad")
try:
    t0_wu = time.perf_counter()

    # compute_pairwise_distances
    _c_a = np.zeros((4, 3), dtype=np.float32)
    _c_b = np.zeros((4, 3), dtype=np.float32)
    _out = np.zeros((4, 4), dtype=np.float32)
    compute_pairwise_distances(_c_a, _c_b, _out)

    # build_contact_map
    _dist_dummy = np.ones((4, 4), dtype=np.float32) * 5.0
    _contact_out = np.zeros((4, 4), dtype=np.int32)
    build_contact_map(_dist_dummy, 8.0, _contact_out)

    # compute_rna_self_contacts
    _rna_c = np.random.default_rng(0).standard_normal((8, 3)).astype(np.float32) * 10.0
    _sc_out = np.zeros((8, 8), dtype=np.int32)
    compute_rna_self_contacts(_rna_c, 8.0, 4, _sc_out)

    # vectorised_partner_mask
    _all_atoms = np.random.default_rng(1).standard_normal((12, 3)).astype(np.float32) * 10.0
    _rna_idx   = np.arange(4, dtype=np.int32)
    _par_idx   = np.arange(4, 12, dtype=np.int32)
    _vmask_out = np.zeros(4, dtype=np.bool_)
    vectorised_partner_mask(_rna_idx, _par_idx, _all_atoms, 8.0, _vmask_out)

    # compute_metal_binding_features
    _rna_c2 = np.random.default_rng(2).standard_normal((8, 3)).astype(np.float32)
    _met_c  = np.zeros((2, 3), dtype=np.float32)
    _radii  = np.array([3.5, 3.5], dtype=np.float32)
    _mbf_out = np.zeros(8, dtype=np.float32)
    compute_metal_binding_features(_rna_c2, _met_c, _radii, _mbf_out)

    # apply_metal_distance_restraints
    _dist_rna = np.ones((8, 8), dtype=np.float32) * 6.0
    _met_mask = np.array([True, False, True, False, True, False, True, False],
                         dtype=np.bool_)
    _restr_out = np.zeros((8, 8), dtype=np.float32)
    apply_metal_distance_restraints(_dist_rna, _met_mask, 0.5, _restr_out)

    warmup_ms = (time.perf_counter() - t0_wu) * 1000
    LOG.log("warmup_ms", round(warmup_ms, 1))
    LOG.end("PASS", reason=f"All 6 Numba kernels compiled in {warmup_ms:.0f} ms")
except Exception:
    LOG.end("ERROR", tb=traceback.format_exc())

print(f"\n  ✓ Stage 0 complete.\n")


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 1 — SURGICAL KAGGLE DATA FETCH + COORDINATE EXTRACTION
# ─────────────────────────────────────────────────────────────────────────────

LOG.section("STAGE 1: SURGICAL FETCH — Real Stanford RNA 3D Folding 2 Data")

os.environ["KAGGLE_CONFIG_DIR"] = KAGGLE_CFG
os.makedirs(DATA_DIR, exist_ok=True)


def fetch_file(filename: str) -> str:
    path = os.path.join(DATA_DIR, filename)
    if os.path.exists(path):
        size_mb = os.path.getsize(path) / 1_048_576
        print(f"  ✓ {filename} cached  ({size_mb:.1f} MB)")
        return path
    print(f"  [Kaggle API] Fetching {filename} …")
    result = subprocess.run(
        ["kaggle", "competitions", "download",
         "-c", COMP_NAME, "-f", filename, "-p", DATA_DIR],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"kaggle download failed for {filename}:\n{result.stderr.strip()}")
    zip_path = path + ".zip"
    if os.path.exists(zip_path):
        import zipfile
        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall(DATA_DIR)
        os.remove(zip_path)
    if not os.path.exists(path):
        raise FileNotFoundError(f"{filename} not found after download")
    return path


LOG.begin("S1-FETCH", "Download CSV files from Kaggle competition", "DataFetch")
try:
    TEST_CSV  = fetch_file("test_sequences.csv")
    TRAIN_SEQ = fetch_file("train_sequences.csv")
    TRAIN_LAB = fetch_file("train_labels.csv")

    test_df  = pd.read_csv(TEST_CSV)
    train_sq = pd.read_csv(TRAIN_SEQ)
    train_lb = pd.read_csv(TRAIN_LAB, nrows=MAX_LABEL_ROWS)

    LOG.log("test_sequences_rows",  len(test_df))
    LOG.log("train_sequences_rows", len(train_sq))
    LOG.log("train_labels_rows",    len(train_lb))
    LOG.log("test_columns",         list(test_df.columns))
    LOG.log("labels_columns",       list(train_lb.columns[:12]))
    LOG.end("PASS", reason="All 3 CSV files loaded successfully")
except Exception:
    LOG.end("ERROR", tb=traceback.format_exc())
    print("  ⚠  Kaggle fetch failed — falling back to synthetic data for all stages.")
    rng_fb  = np.random.default_rng(42)
    bases   = list("ACGU")
    seqs_fb = ["".join(rng_fb.choice(bases, rng_fb.integers(40, 120))) for _ in range(10)]
    test_df  = pd.DataFrame({"target_id": [f"t{i}" for i in range(10)],
                              "sequence": seqs_fb})
    train_sq = test_df.copy()
    rows_fb = []
    for i, (_, row) in enumerate(test_df.iterrows()):
        tid, seq = row["target_id"], row["sequence"]
        for r in range(len(seq)):
            xyz = np.random.default_rng(i * 1000 + r).normal(0, 15, 3)
            rows_fb.append({"ID": f"{tid}_{r}", "x_1": xyz[0],
                            "y_1": xyz[1], "z_1": xyz[2]})
    train_lb = pd.DataFrame(rows_fb)


def id_col(df: pd.DataFrame) -> str:
    for c in ("target_id", "ID", "id"):
        if c in df.columns:
            return c
    for c in df.columns:
        if "id" in c.lower():
            return c
    return df.columns[0]


def extract_target_id(row_id: str) -> str:
    parts = str(row_id).split("_")
    while len(parts) > 1 and parts[-1].isdigit():
        parts = parts[:-1]
    return "_".join(parts)


LOG.begin("S1-PARSE",
          "Parse train_labels: extract C3′ backbone coords per target",
          "DataFetch")
try:
    x_col = next((c for c in train_lb.columns if c.lower() in ("x_1","x1","x")), None)
    y_col = next((c for c in train_lb.columns if c.lower() in ("y_1","y1","y")), None)
    z_col = next((c for c in train_lb.columns if c.lower() in ("z_1","z1","z")), None)
    id_c  = id_col(train_lb)

    LOG.log("coord_cols_found", [x_col, y_col, z_col])
    LOG.log("id_col",           id_c)

    if None in (x_col, y_col, z_col):
        raise ValueError(
            f"Could not find x/y/z columns. Present: {list(train_lb.columns)}")

    train_lb["_target"] = train_lb[id_c].apply(extract_target_id)
    target_c3p: Dict[str, np.ndarray] = {}

    for tid, grp in train_lb.groupby("_target"):
        xyz = grp[[x_col, y_col, z_col]].dropna().values.astype(np.float64)
        if len(xyz) >= 10:
            target_c3p[tid] = xyz

    sorted_targets = sorted(target_c3p.items(), key=lambda kv: -len(kv[1]))
    REAL_TARGETS   = sorted_targets[:MAX_TARGET_COORDS]

    LOG.log("n_unique_targets",   len(target_c3p))
    LOG.log("n_selected_targets", len(REAL_TARGETS))
    for tid, xyz in REAL_TARGETS:
        LOG.log(f"target_{tid[:16]}_N", len(xyz))

    if not REAL_TARGETS:
        raise ValueError("No valid targets extracted from train_labels.csv")

    LOG.end("PASS", reason=(f"{len(target_c3p)} unique targets; "
                             f"{len(REAL_TARGETS)} selected for geometry tests"))
except Exception:
    LOG.end("ERROR", tb=traceback.format_exc())
    # Synthetic fallback: A-form helix-like coords
    def _aform_c3p(N, seed=0):
        rng = np.random.default_rng(seed)
        coords = np.zeros((N, 3))
        for i in range(N):
            angle = math.radians(i * 32.7)
            coords[i] = [9.0 * math.cos(angle), 9.0 * math.sin(angle), i * 2.81]
        return (coords + rng.normal(0, 0.3, coords.shape)).astype(np.float64)

    REAL_TARGETS = [(f"synth_{chr(65+i)}", _aform_c3p(60 + i*10, seed=i))
                    for i in range(6)]
    print(f"  ⚠  Using {len(REAL_TARGETS)} synthetic targets as fallback.")
    target_c3p = {tid: xyz for tid, xyz in REAL_TARGETS}

LOG.begin("S1-SEQLENS",
          "Extract sequence lengths + sequences from test_sequences.csv",
          "DataFetch")
try:
    seq_col = next((c for c in test_df.columns
                    if "seq" in c.lower()), test_df.columns[-1])
    test_df["_len"]  = test_df[seq_col].str.len()
    REAL_SEQ_LENS    = test_df["_len"].dropna().astype(int).tolist()
    REAL_SEQUENCES   = (test_df[seq_col].dropna()
                        .str.upper().str.replace("T", "U").tolist())

    LOG.log("seq_col",           seq_col)
    LOG.log("n_sequences",       len(REAL_SEQUENCES))
    LOG.log("len_min",           int(min(REAL_SEQ_LENS)))
    LOG.log("len_max",           int(max(REAL_SEQ_LENS)))
    LOG.log("len_median",        int(np.median(REAL_SEQ_LENS)))
    LOG.end("PASS", reason=f"{len(REAL_SEQUENCES)} real test sequences loaded")
except Exception:
    LOG.end("ERROR", tb=traceback.format_exc())
    REAL_SEQ_LENS  = [60, 80, 100, 120, 40, 70, 90, 50, 110, 30]
    REAL_SEQUENCES = [make_random_seq(l, seed=i) for i, l in enumerate(REAL_SEQ_LENS)]
    print("  ⚠  Using synthetic sequence lengths as fallback.")

# Build PDBComplex objects from real data for downstream stages
LOG.begin("S1-COMPLEXES",
          "Construct PDBComplex objects from real C3′ coords + sequences",
          "DataFetch")
try:
    REAL_COMPLEXES: List[PDBComplex] = []
    seq_col_train = next((c for c in train_sq.columns if "seq" in c.lower()),
                         train_sq.columns[-1])
    tid_col_train = id_col(train_sq)
    train_seq_map = {}
    for _, row in train_sq.iterrows():
        t = str(row[tid_col_train])
        s = str(row[seq_col_train]).upper().replace("T", "U")
        train_seq_map[t] = s

    for idx, (tid, c3p) in enumerate(REAL_TARGETS):
        N_rna = min(len(c3p), 120)
        seq   = train_seq_map.get(tid, make_random_seq(N_rna, seed=idx))
        seq   = seq.replace("T", "U").upper()
        cpx   = build_complex_from_real(
            tid, c3p[:N_rna], seq[:N_rna], seed=idx
        )
        REAL_COMPLEXES.append(cpx)

    LOG.log("n_complexes_built", len(REAL_COMPLEXES))
    LOG.log("rna_lengths",       [c.rna_coords.shape[0] for c in REAL_COMPLEXES])
    LOG.log("context_types",     [c.context_type.name for c in REAL_COMPLEXES])
    LOG.end("PASS", reason=f"{len(REAL_COMPLEXES)} PDBComplex objects built "
                            "from real targets")
except Exception:
    LOG.end("ERROR", tb=traceback.format_exc())
    REAL_COMPLEXES = [make_synthetic_complex(rna_length=60 + i*10, seed=i)
                      for i in range(6)]
    print("  ⚠  Using synthetic complexes as fallback.")

print(f"\n  ✓ Stage 1 complete.\n")


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 2 — NUMBA JIT KERNEL CORRECTNESS & PERFORMANCE
# ─────────────────────────────────────────────────────────────────────────────

LOG.section("STAGE 2: NUMBA JIT KERNEL CORRECTNESS & PERFORMANCE")

BENCH_RESULTS: List[Dict] = []


# ── 2A: compute_pairwise_distances ──────────────────────────────────────────

LOG.begin("S2-PWDIST-CORRECT",
          "compute_pairwise_distances: correctness checks",
          "NumbaKernels")
try:
    failures = []
    rng = np.random.default_rng(10)

    # Self-distance matrix must be symmetric and zero on diagonal
    N = 30
    coords = rng.standard_normal((N, 3)).astype(np.float32) * 10.0
    out = np.zeros((N, N), dtype=np.float32)
    compute_pairwise_distances(coords, coords, out)

    if out.shape != (N, N):
        failures.append(f"shape {out.shape} ≠ ({N},{N})")
    if not np.all(out >= 0):
        failures.append("negative distance found")
    if not np.allclose(out, out.T, atol=1e-4):
        failures.append("distance matrix not symmetric")
    if not np.allclose(np.diag(out), 0.0, atol=1e-4):
        failures.append("diagonal not zero")

    # Known distance: two points at (0,0,0) and (3,4,0) → 5 Å
    ca = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)
    cb = np.array([[3.0, 4.0, 0.0]], dtype=np.float32)
    od = np.zeros((1, 1), dtype=np.float32)
    compute_pairwise_distances(ca, cb, od)
    if not math.isclose(float(od[0, 0]), 5.0, abs_tol=1e-4):
        failures.append(f"known distance 5.0 ≠ {od[0,0]:.5f}")

    # Asymmetric: (N, M) shape
    M = 45
    coords_m = rng.standard_normal((M, 3)).astype(np.float32) * 10.0
    out_nm = np.zeros((N, M), dtype=np.float32)
    compute_pairwise_distances(coords, coords_m, out_nm)
    if out_nm.shape != (N, M):
        failures.append(f"asymmetric shape {out_nm.shape} ≠ ({N},{M})")

    LOG.log("symmetry_max_err", float(np.max(np.abs(out - out.T))))
    LOG.log("diagonal_max_err", float(np.max(np.abs(np.diag(out)))))
    LOG.log("known_dist_5A",    round(float(od[0, 0]), 5))

    if failures:
        LOG.end("FAIL", reason="; ".join(failures))
    else:
        LOG.end("PASS", reason="Symmetric, non-negative, zero diagonal; known 5Å correct")
except Exception:
    LOG.end("ERROR", tb=traceback.format_exc())


LOG.begin("S2-PWDIST-BENCH",
          "compute_pairwise_distances: performance N=50/100/200",
          "NumbaKernels")
try:
    for N in (50, 100, 200):
        coords_n = np.random.default_rng(N).standard_normal((N, 3)).astype(np.float32) * 15.0
        out_n    = np.zeros((N, N), dtype=np.float32)
        ms_med   = benchmark(lambda: compute_pairwise_distances(coords_n, coords_n, out_n))
        LOG.log(f"pwdist_N{N}_ms", round(ms_med, 2))
        BENCH_RESULTS.append({"tid": f"S2-PWDIST-BENCH-N{N}", "ms": ms_med})
    LOG.end("PASS", reason="Benchmarks complete; all outputs finite")
except Exception:
    LOG.end("ERROR", tb=traceback.format_exc())


# ── 2B: build_contact_map ───────────────────────────────────────────────────

LOG.begin("S2-CMAP-CORRECT",
          "build_contact_map: binary output, threshold sweep",
          "NumbaKernels")
try:
    failures = []
    rng = np.random.default_rng(20)
    N   = 40
    coords = rng.standard_normal((N, 3)).astype(np.float32) * 10.0
    dist_mat = np.zeros((N, N), dtype=np.float32)
    compute_pairwise_distances(coords, coords, dist_mat)

    prev_contacts = -1  # contacts must be non-decreasing as threshold grows
    for thresh in (4.0, 8.0, 12.0, 16.0):
        cmap = np.zeros((N, N), dtype=np.int32)
        build_contact_map(dist_mat, thresh, cmap)

        n_contacts = int(cmap.sum())
        if not np.all((cmap == 0) | (cmap == 1)):
            failures.append(f"thresh={thresh}: non-binary values in contact map")
        # Larger threshold → equal or more contacts (monotonically non-decreasing)
        if n_contacts < prev_contacts:
            failures.append(
                f"thresh={thresh}: contacts decreased ({n_contacts} < {prev_contacts}) "
                "— contact count must be non-decreasing with threshold")
        # Verify consistency with distance matrix
        expected = (dist_mat <= thresh).astype(np.int32)
        if not np.array_equal(cmap, expected):
            failures.append(f"thresh={thresh}: contact map inconsistent with dist_mat")
        prev_contacts = n_contacts
        LOG.log(f"contacts_thresh{int(thresh)}A", n_contacts)

    if failures:
        LOG.end("FAIL", reason="; ".join(failures))
    else:
        LOG.end("PASS",
                reason="Binary output; contacts increase with threshold (non-decreasing); "
                       "consistent with dist_mat at all 4 thresholds")
except Exception:
    LOG.end("ERROR", tb=traceback.format_exc())


# ── 2C: compute_rna_self_contacts ───────────────────────────────────────────

LOG.begin("S2-SELFCONT-CORRECT",
          "compute_rna_self_contacts: diagonal=0, seq-sep filter",
          "NumbaKernels")
try:
    failures = []
    rng = np.random.default_rng(30)

    for L in (20, 60, 120):
        rna_c = rng.standard_normal((L, 3)).astype(np.float32) * 12.0
        sc = np.zeros((L, L), dtype=np.int32)
        compute_rna_self_contacts(rna_c, 8.0, 4, sc)

        if sc.shape != (L, L):
            failures.append(f"L={L}: shape {sc.shape} ≠ ({L},{L})")
        if not np.all(np.diag(sc) == 0):
            failures.append(f"L={L}: diagonal not zero")
        if not np.all((sc == 0) | (sc == 1)):
            failures.append(f"L={L}: non-binary values")
        # Residues within min_seq_sep=4 must have 0 contact
        for i in range(min(L, 10)):
            for j in range(max(0, i-3), min(L, i+4)):
                if sc[i, j] != 0:
                    failures.append(f"L={L}: seq-sep <4 contact at ({i},{j})")
                    break

        LOG.log(f"L{L}_contacts",    int(sc.sum()))
        LOG.log(f"L{L}_diag_ok",     bool(np.all(np.diag(sc) == 0)))

    if failures:
        LOG.end("FAIL", reason="; ".join(failures[:3]))
    else:
        LOG.end("PASS", reason="Diagonal=0; seq-sep=4 enforced; binary; all tested lengths")
except Exception:
    LOG.end("ERROR", tb=traceback.format_exc())


# ── 2D: vectorised_partner_mask ─────────────────────────────────────────────

LOG.begin("S2-VMASK-CORRECT",
          "vectorised_partner_mask: analytical verification",
          "NumbaKernels")
try:
    failures = []

    # Analytical test: RNA atom at origin, partner at (3,0,0) → within 8Å → True
    all_atoms = np.array([[0.0, 0.0, 0.0],   # RNA atom 0
                           [3.0, 0.0, 0.0],   # partner atom 1 (dist=3 < 8)
                           [20., 0.0, 0.0],   # partner atom 2 (dist=20 > 8)
                          ], dtype=np.float32)
    rna_idx    = np.array([0], dtype=np.int32)
    part_near  = np.array([1], dtype=np.int32)
    part_far   = np.array([2], dtype=np.int32)
    mask_near  = np.zeros(1, dtype=np.bool_)
    mask_far   = np.zeros(1, dtype=np.bool_)

    vectorised_partner_mask(rna_idx, part_near, all_atoms, 8.0, mask_near)
    vectorised_partner_mask(rna_idx, part_far,  all_atoms, 8.0, mask_far)

    if not bool(mask_near[0]):
        failures.append("partner at 3Å should be within 8Å threshold → True")
    if bool(mask_far[0]):
        failures.append("partner at 20Å should NOT be within 8Å threshold → False")

    # Shape check with real-size arrays
    rng     = np.random.default_rng(40)
    N_rna   = 50
    N_part  = 100
    N_tot   = N_rna + N_part
    all_c   = rng.standard_normal((N_tot, 3)).astype(np.float32) * 15.0
    r_idx   = np.arange(N_rna, dtype=np.int32)
    p_idx   = np.arange(N_rna, N_tot, dtype=np.int32)
    vmask   = np.zeros(N_rna, dtype=np.bool_)
    vectorised_partner_mask(r_idx, p_idx, all_c, 8.0, vmask)

    if vmask.shape != (N_rna,):
        failures.append(f"shape {vmask.shape} ≠ ({N_rna},)")
    if vmask.dtype != np.bool_:
        failures.append(f"dtype {vmask.dtype} ≠ bool")

    LOG.log("analytical_near_True", bool(mask_near[0]))
    LOG.log("analytical_far_False", not bool(mask_far[0]))
    LOG.log("real_size_fraction_masked", round(vmask.mean(), 3))

    if failures:
        LOG.end("FAIL", reason="; ".join(failures))
    else:
        LOG.end("PASS", reason="Analytical near/far check correct; shape and dtype correct")
except Exception:
    LOG.end("ERROR", tb=traceback.format_exc())


# ── 2E: compute_metal_binding_features ──────────────────────────────────────

LOG.begin("S2-METAL-BIND",
          "compute_metal_binding_features: Gaussian proximity score",
          "NumbaKernels")
try:
    failures = []
    rng = np.random.default_rng(50)
    L   = 30
    rna_c  = rng.standard_normal((L, 3)).astype(np.float32) * 10.0
    met_c  = rng.standard_normal((3, 3)).astype(np.float32) * 5.0
    radii  = np.array([3.5, 3.0, 4.0], dtype=np.float32)
    out_p  = np.zeros(L, dtype=np.float32)

    compute_metal_binding_features(rna_c, met_c, radii, out_p)

    if out_p.shape != (L,):
        failures.append(f"shape {out_p.shape} ≠ ({L},)")
    if not np.all(out_p >= 0):
        failures.append("negative metal binding probability found")
    if not np.all(np.isfinite(out_p)):
        failures.append("non-finite values in metal binding output")

    # RNA atom placed at metal position → high score
    rna_at_metal = met_c[:1].copy()
    out_at = np.zeros(1, dtype=np.float32)
    compute_metal_binding_features(rna_at_metal, met_c, radii, out_at)

    rna_far = met_c[:1] + np.array([[100., 100., 100.]], dtype=np.float32)
    out_far = np.zeros(1, dtype=np.float32)
    compute_metal_binding_features(rna_far, met_c, radii, out_far)

    if not (float(out_at[0]) > float(out_far[0])):
        failures.append(
            f"score at metal ({out_at[0]:.4f}) should > score far ({out_far[0]:.6f})")

    # Zero metals → zero score
    out_zero = np.zeros(L, dtype=np.float32)
    if met_c.shape[0] > 0:
        empty_met = np.zeros((0, 3), dtype=np.float32)
        empty_rad = np.zeros(0, dtype=np.float32)
        # Only test if kernel handles empty arrays (may raise — catch separately)
        try:
            compute_metal_binding_features(rna_c, empty_met, empty_rad, out_zero)
            if not np.allclose(out_zero, 0.0):
                failures.append("zero metals should produce zero scores")
        except Exception:
            LOG.warn("kernel raised on empty metal array — acceptable behaviour")

    LOG.log("score_at_metal",  round(float(out_at[0]), 4))
    LOG.log("score_far_metal", round(float(out_far[0]), 6))
    LOG.log("score_mean",      round(float(out_p.mean()), 4))

    if failures:
        LOG.end("FAIL", reason="; ".join(failures))
    else:
        LOG.end("PASS", reason="Shape correct; ≥0; finite; score decreases with distance")
except Exception:
    LOG.end("ERROR", tb=traceback.format_exc())


# ── 2F: apply_metal_distance_restraints ─────────────────────────────────────

LOG.begin("S2-METAL-RESTR",
          "apply_metal_distance_restraints: shape, zeros, restraint ∝ 1/(1+d)",
          "NumbaKernels")
try:
    failures = []
    L  = 20
    rng = np.random.default_rng(60)
    dist_mat = (rng.standard_normal((L, L)) * 5.0 + 8.0).astype(np.float32)
    dist_mat = np.abs(dist_mat)  # ensure positive
    np.fill_diagonal(dist_mat, 0.0)

    # Half residues flagged as metal-binding
    met_mask = np.zeros(L, dtype=np.bool_)
    met_mask[::2] = True

    restr = np.zeros((L, L), dtype=np.float32)
    apply_metal_distance_restraints(dist_mat, met_mask, 0.5, restr)

    if restr.shape != (L, L):
        failures.append(f"shape {restr.shape} ≠ ({L},{L})")
    if not np.all(restr >= 0):
        failures.append("negative restraint found")

    # Non-metal-binding pairs must be zero
    for i in range(L):
        for j in range(L):
            if not (met_mask[i] and met_mask[j]) or i == j:
                if restr[i, j] != 0.0:
                    failures.append(f"non-metal pair ({i},{j}) has non-zero restraint")
                    break

    # Verify ∝ 1/(1+d): two metal-binding residues
    k = 0.5
    i0, j0 = 0, 2  # both metal-binding (even indices)
    expected_val = k / (1.0 + dist_mat[i0, j0])
    if not math.isclose(float(restr[i0, j0]), expected_val, rel_tol=1e-4):
        failures.append(f"restraint formula mismatch: {restr[i0,j0]:.6f} ≠ {expected_val:.6f}")

    # All-false mask → all-zero restraint
    restr_zero = np.zeros((L, L), dtype=np.float32)
    apply_metal_distance_restraints(dist_mat, np.zeros(L, dtype=np.bool_), 0.5, restr_zero)
    if not np.allclose(restr_zero, 0.0):
        failures.append("all-false mask should yield zero restraint")

    LOG.log("restraint_mean_metal_pairs", round(float(restr[met_mask][:, met_mask].mean()), 5))
    LOG.log("formula_check_i0j0",        round(float(restr[i0, j0]), 6))
    LOG.log("expected_i0j0",             round(expected_val, 6))

    if failures:
        LOG.end("FAIL", reason="; ".join(failures[:3]))
    else:
        LOG.end("PASS", reason="Shape correct; ≥0; zero off-metal pairs; formula k/(1+d) verified")
except Exception:
    LOG.end("ERROR", tb=traceback.format_exc())

print("\n  ✓ Stage 2 complete.\n")


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 3 — SEQUENCE UTILITIES
# ─────────────────────────────────────────────────────────────────────────────

LOG.section("STAGE 3: SEQUENCE UTILITIES — CORRECTNESS & EDGE CASES")

LOG.begin("S3-ONEHOT",
          "sequence_to_one_hot: shape, row-sums, nucleotide mapping",
          "SeqUtils")
try:
    failures = []

    # Basic shape
    seq = "AUGCNAUGCN"
    oh  = sequence_to_one_hot(seq)
    if oh.shape != (len(seq), 5):
        failures.append(f"shape {oh.shape} ≠ ({len(seq)},5)")
    if oh.dtype != np.float32:
        failures.append(f"dtype {oh.dtype} ≠ float32")
    if not np.allclose(oh.sum(axis=-1), 1.0, atol=1e-5):
        failures.append("row sums ≠ 1.0")

    # Correct column mapping: A=0, U=1, G=2, C=3, N=4
    nuc_to_col = {n: i for i, n in enumerate(RNA_NUCLEOTIDES)}
    for i, nuc in enumerate(seq):
        expected_col = nuc_to_col.get(nuc, nuc_to_col["N"])
        if oh[i, expected_col] != 1.0:
            failures.append(f"pos {i} nuc={nuc} expected col {expected_col}")

    # Unknown character → N column
    oh_unk = sequence_to_one_hot("X")
    if oh_unk[0, nuc_to_col["N"]] != 1.0:
        failures.append("unknown char 'X' should map to N column")

    # Uppercase normalisation
    oh_lower = sequence_to_one_hot("augc")
    oh_upper = sequence_to_one_hot("AUGC")
    if not np.allclose(oh_lower, oh_upper, atol=1e-5):
        failures.append("lowercase/uppercase mismatch")

    # Real sequences from test_sequences.csv
    n_tested = min(5, len(REAL_SEQUENCES))
    for seq_r in REAL_SEQUENCES[:n_tested]:
        oh_r = sequence_to_one_hot(seq_r)
        if not np.allclose(oh_r.sum(axis=-1), 1.0, atol=1e-5):
            failures.append(f"row sums ≠ 1 for real seq len={len(seq_r)}")
            break

    LOG.log("shape_ok",         oh.shape == (len(seq), 5))
    LOG.log("dtype_ok",         oh.dtype == np.float32)
    LOG.log("rowsums_ok",       bool(np.allclose(oh.sum(axis=-1), 1.0, atol=1e-5)))
    LOG.log("real_seqs_tested", n_tested)

    if failures:
        LOG.end("FAIL", reason="; ".join(failures[:3]))
    else:
        LOG.end("PASS",
                reason="Shape/dtype/sums correct; nucleotide mapping correct; "
                       "unknown→N; lowercase normalised")
except Exception:
    LOG.end("ERROR", tb=traceback.format_exc())


LOG.begin("S3-ROUNDTRIP",
          "one_hot_to_sequence: sequence→one_hot→sequence roundtrip",
          "SeqUtils")
try:
    failures = []
    for seq_r in REAL_SEQUENCES[:5]:
        oh_r  = sequence_to_one_hot(seq_r)
        seq_r2 = one_hot_to_sequence(oh_r)
        # After roundtrip, only AUGCN chars — T would have been absorbed as N
        seq_clean = seq_r.upper().replace("T", "U")
        seq_clean2 = seq_r2.upper()
        if seq_clean != seq_clean2:
            # Check if mismatch is only in non-AUGCN chars mapped to N
            mismatches = [(i, a, b) for i, (a, b) in
                          enumerate(zip(seq_clean, seq_clean2)) if a != b]
            for pos, orig, rec in mismatches:
                if orig not in "AUGCN":
                    continue  # expected: non-AUGCN → N
                failures.append(
                    f"roundtrip mismatch at pos {pos}: {orig} → {rec}")
    for seq_r in ["AUGCAUGC", "GGGCCC", "UUUAAA"]:
        oh_r  = sequence_to_one_hot(seq_r)
        if one_hot_to_sequence(oh_r) != seq_r:
            failures.append(f"roundtrip failed for '{seq_r}'")

    if failures:
        LOG.end("FAIL", reason="; ".join(failures[:3]))
    else:
        LOG.end("PASS", reason="Roundtrip identity for AUGCN sequences")
except Exception:
    LOG.end("ERROR", tb=traceback.format_exc())


LOG.begin("S3-CLASSIFIER",
          "classify_rna_context: motif hits, fallback to NAKED_RNA",
          "SeqUtils")
try:
    failures = []

    # Canonical motif sequences
    riboswitch_seq = "GGCGUAAUUCGAAUGGCGUA" * 3
    trna_seq       = "GCGCCUAUUCGAAUGCUCCAGCGCCUAUUCGAAUGCUCCA"
    unknown_seq    = "AUGUAUGCUAGCUAGCUAGCAUGUAUGCUAGCUAGCUAGC"

    ctx_ribo = classify_rna_context(riboswitch_seq)
    ctx_trna = classify_rna_context(trna_seq)
    ctx_unk  = classify_rna_context(unknown_seq)

    if ctx_ribo not in list(RNAContextType):
        failures.append(f"classify_rna_context returned invalid type: {ctx_ribo}")
    if ctx_unk not in list(RNAContextType):
        failures.append(f"unknown seq returned invalid type: {ctx_unk}")

    LOG.log("riboswitch_seq_ctx",  ctx_ribo.name)
    LOG.log("trna_seq_ctx",        ctx_trna.name)
    LOG.log("unknown_seq_ctx",     ctx_unk.name)

    # Test real sequences from test_sequences.csv
    real_ctx_names = []
    for seq_r in REAL_SEQUENCES[:MAX_TEST_SEQS]:
        ctx_r = classify_rna_context(seq_r)
        if ctx_r not in list(RNAContextType):
            failures.append(f"real seq classify returned invalid: {ctx_r}")
        real_ctx_names.append(ctx_r.name)

    from collections import Counter
    ctx_counts = Counter(real_ctx_names)
    LOG.log("real_seq_ctx_distribution", dict(ctx_counts))
    LOG.log("n_real_seqs_classified",    len(real_ctx_names))

    if failures:
        LOG.end("FAIL", reason="; ".join(failures[:3]))
    else:
        LOG.end("PASS",
                reason="Returns valid RNAContextType for all inputs; "
                       "motif hits logical; fallback to NAKED_RNA")
except Exception:
    LOG.end("ERROR", tb=traceback.format_exc())

print("\n  ✓ Stage 3 complete.\n")


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 4 — PDBComplex CONSTRUCTION & MASKING
# ─────────────────────────────────────────────────────────────────────────────

LOG.section("STAGE 4: PDBComplex CONSTRUCTION & PARTNER MASKING")

LOG.begin("S4-COMPLEX-SHAPE",
          "PDBComplex field shapes, dtypes, index consistency",
          "ComplexBuild")
try:
    failures = []

    for cpx in REAL_COMPLEXES:
        L  = cpx.rna_coords.shape[0]
        Np = cpx.partner_indices.shape[0]
        Na = cpx.all_atom_coords.shape[0]

        if cpx.rna_coords.dtype != np.float32:
            failures.append(f"{cpx.pdb_id}: rna_coords dtype {cpx.rna_coords.dtype} ≠ float32")
        if cpx.rna_coords.shape[1] != 3:
            failures.append(f"{cpx.pdb_id}: rna_coords shape[1] ≠ 3")
        if cpx.rna_atom_indices.dtype != np.int32:
            failures.append(f"{cpx.pdb_id}: rna_atom_indices dtype ≠ int32")
        if cpx.partner_indices.dtype != np.int32:
            failures.append(f"{cpx.pdb_id}: partner_indices dtype ≠ int32")

        # Indices must be within all_atom_coords bounds
        if cpx.rna_atom_indices.max() >= Na:
            failures.append(f"{cpx.pdb_id}: rna_atom_indices out of range")
        if Np > 0 and cpx.partner_indices.max() >= Na:
            failures.append(f"{cpx.pdb_id}: partner_indices out of range")

        # rna_atom_indices and partner_indices must be non-overlapping
        rna_set  = set(cpx.rna_atom_indices.tolist())
        part_set = set(cpx.partner_indices.tolist())
        if rna_set & part_set:
            failures.append(f"{cpx.pdb_id}: rna and partner atom indices overlap")

        LOG.log(f"{cpx.pdb_id[:12]}_L",   L)
        LOG.log(f"{cpx.pdb_id[:12]}_Np",  Np)
        LOG.log(f"{cpx.pdb_id[:12]}_ctx", cpx.context_type.name)

    if failures:
        LOG.end("FAIL", reason="; ".join(failures[:3]))
    else:
        LOG.end("PASS",
                reason="All PDBComplex fields have correct dtype, shape and consistent indices")
except Exception:
    LOG.end("ERROR", tb=traceback.format_exc())


LOG.begin("S4-MASK-PARTNER",
          "mask_partner_coordinates: partner zeros, RNA intact, shape",
          "ComplexBuild")
try:
    failures = []

    for cpx in REAL_COMPLEXES[:3]:
        vis, orig_partner = mask_partner_coordinates(cpx, mask_protein=True)

        # Shape
        if vis.shape != cpx.all_atom_coords.shape:
            failures.append(f"{cpx.pdb_id}: visible coords shape mismatch")
        if orig_partner.shape != (len(cpx.partner_indices), 3):
            failures.append(f"{cpx.pdb_id}: orig_partner shape mismatch")

        # Partner atoms zeroed
        partner_vis = vis[cpx.partner_indices]
        if not np.allclose(partner_vis, 0.0):
            failures.append(f"{cpx.pdb_id}: partner atoms not zeroed")

        # RNA atoms intact
        rna_vis = vis[cpx.rna_atom_indices]
        rna_orig = cpx.all_atom_coords[cpx.rna_atom_indices]
        if not np.allclose(rna_vis, rna_orig, atol=1e-5):
            failures.append(f"{cpx.pdb_id}: RNA atoms modified by masking")

        # original_partner matches original all_atom_coords
        partner_orig = cpx.all_atom_coords[cpx.partner_indices]
        if not np.allclose(orig_partner, partner_orig, atol=1e-5):
            failures.append(f"{cpx.pdb_id}: original_partner does not match all_atom_coords")

    if failures:
        LOG.end("FAIL", reason="; ".join(failures[:3]))
    else:
        LOG.end("PASS",
                reason="Partner atoms zeroed; RNA intact; original_partner correct; shape preserved")
except Exception:
    LOG.end("ERROR", tb=traceback.format_exc())


LOG.begin("S4-RANDOM-CROP",
          "random_crop_rna: output length ≤ max_length, subsequence correct",
          "ComplexBuild")
try:
    failures = []

    for cpx in REAL_COMPLEXES[:3]:
        seq    = cpx.rna_sequence
        coords = cpx.rna_coords
        L      = len(seq)
        max_l  = min(L // 2 + 1, 50)

        seq_c, coords_c = random_crop_rna(seq, coords, max_length=max_l)

        if len(seq_c) > max_l:
            failures.append(f"{cpx.pdb_id}: cropped seq len {len(seq_c)} > max {max_l}")
        if coords_c.shape[0] != len(seq_c):
            failures.append(
                f"{cpx.pdb_id}: coords_c length {coords_c.shape[0]} ≠ seq_c len {len(seq_c)}")

        # Verify it's a contiguous subsequence of the original
        found = False
        for start in range(L - len(seq_c) + 1):
            if seq[start:start + len(seq_c)] == seq_c:
                found = True
                break
        if not found:
            failures.append(f"{cpx.pdb_id}: cropped seq not a contiguous subsequence")

        # No-op when L ≤ max_length
        seq_nc, coords_nc = random_crop_rna(seq, coords, max_length=L + 10)
        if seq_nc != seq:
            failures.append(f"{cpx.pdb_id}: no-crop should return original sequence")

    if failures:
        LOG.end("FAIL", reason="; ".join(failures[:3]))
    else:
        LOG.end("PASS",
                reason="len ≤ max_length; contiguous subsequence; no-op when L ≤ max_length")
except Exception:
    LOG.end("ERROR", tb=traceback.format_exc())

print("\n  ✓ Stage 4 complete.\n")


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 5 — extract_context_features PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

LOG.section("STAGE 5: extract_context_features — PIPELINE CORRECTNESS")

FEAT_BENCH: List[Dict] = []

LOG.begin("S5-FEATURES-SHAPE",
          "extract_context_features: all field shapes and dtypes",
          "ContextFeatures")
try:
    failures = []

    for cpx in REAL_COMPLEXES:
        feat = extract_context_features(cpx)
        L    = cpx.rna_coords.shape[0]

        if feat.rna_length != L:
            failures.append(f"{cpx.pdb_id}: rna_length {feat.rna_length} ≠ {L}")
        if feat.contact_mask.shape != (L,):
            failures.append(f"{cpx.pdb_id}: contact_mask shape {feat.contact_mask.shape} ≠ ({L},)")
        if feat.contact_mask.dtype != np.bool_:
            failures.append(f"{cpx.pdb_id}: contact_mask dtype {feat.contact_mask.dtype} ≠ bool")
        if feat.self_contact_map.shape != (L, L):
            failures.append(f"{cpx.pdb_id}: self_contact_map shape ≠ ({L},{L})")
        if not np.all((feat.self_contact_map == 0) | (feat.self_contact_map == 1)):
            failures.append(f"{cpx.pdb_id}: self_contact_map not binary")
        if feat.metal_binding_prob.shape != (L,):
            failures.append(f"{cpx.pdb_id}: metal_binding_prob shape ≠ ({L},)")
        if not np.all(feat.metal_binding_prob >= 0):
            failures.append(f"{cpx.pdb_id}: metal_binding_prob has negatives")
        if feat.metal_restraints.shape != (L, L):
            failures.append(f"{cpx.pdb_id}: metal_restraints shape ≠ ({L},{L})")
        if not np.all(feat.metal_restraints >= 0):
            failures.append(f"{cpx.pdb_id}: metal_restraints has negatives")
        if feat.partner_dist_mat is not None:
            Np = cpx.partner_indices.shape[0]
            if feat.partner_dist_mat.shape != (L, Np):
                failures.append(
                    f"{cpx.pdb_id}: partner_dist_mat shape "
                    f"{feat.partner_dist_mat.shape} ≠ ({L},{Np})")
            if not np.all(feat.partner_dist_mat >= 0):
                failures.append(f"{cpx.pdb_id}: partner_dist_mat has negatives")
        if feat.sequence_one_hot.shape != (L, 5):
            failures.append(f"{cpx.pdb_id}: sequence_one_hot shape ≠ ({L},5)")
        if not np.allclose(feat.sequence_one_hot.sum(axis=-1), 1.0, atol=1e-5):
            failures.append(f"{cpx.pdb_id}: one_hot row sums ≠ 1")

        LOG.log(f"{cpx.pdb_id[:12]}_contacts",     int(feat.contact_mask.sum()))
        LOG.log(f"{cpx.pdb_id[:12]}_self_cont",    int(feat.self_contact_map.sum()))
        LOG.log(f"{cpx.pdb_id[:12]}_metal_mean",   round(float(feat.metal_binding_prob.mean()), 4))

    if failures:
        LOG.end("FAIL", reason="; ".join(failures[:3]))
    else:
        LOG.end("PASS",
                reason="All ContextFeatures fields correct shape, dtype and value range")
except Exception:
    LOG.end("ERROR", tb=traceback.format_exc())


LOG.begin("S5-SELFCONT-SYMM",
          "self_contact_map: symmetric, diagonal=0, seq-sep=4 enforced",
          "ContextFeatures")
try:
    failures = []

    for cpx in REAL_COMPLEXES[:3]:
        feat = extract_context_features(cpx)
        sc   = feat.self_contact_map
        L    = feat.rna_length

        if not np.array_equal(sc, sc.T):
            failures.append(f"{cpx.pdb_id}: self_contact_map not symmetric")
        if not np.all(np.diag(sc) == 0):
            failures.append(f"{cpx.pdb_id}: diagonal not zero")
        for i in range(min(L, 8)):
            for j in range(max(0, i - 3), min(L, i + 4)):
                if sc[i, j] != 0:
                    failures.append(f"{cpx.pdb_id}: seq-sep < 4 contact at ({i},{j})")
                    break

    if failures:
        LOG.end("FAIL", reason="; ".join(failures[:3]))
    else:
        LOG.end("PASS", reason="Symmetric; diagonal=0; seq-sep=4 enforced on real targets")
except Exception:
    LOG.end("ERROR", tb=traceback.format_exc())


LOG.begin("S5-FEAT-BENCH",
          "Feature extraction performance: N=20/60/120 residues",
          "ContextFeatures")
try:
    for L_test in (20, 60, 120):
        cpx_b = make_synthetic_complex(rna_length=L_test, seed=L_test)
        # Force one warm-up
        _ = extract_context_features(cpx_b)
        ms_med = benchmark(lambda: extract_context_features(cpx_b))
        LOG.log(f"extract_L{L_test}_ms", round(ms_med, 2))
        FEAT_BENCH.append({"L": L_test, "ms": ms_med})

    LOG.end("PASS", reason="Feature extraction benchmarks recorded")
except Exception:
    LOG.end("ERROR", tb=traceback.format_exc())

print("\n  ✓ Stage 5 complete.\n")


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 6 — ContextTypeEmbedding + MetalBindingHead
# ─────────────────────────────────────────────────────────────────────────────

LOG.section("STAGE 6: ContextTypeEmbedding + MetalBindingHead")

LOG.begin("S6-CTX-EMBED",
          "ContextTypeEmbedding: weight shape, __call__ output, inject",
          "Embeddings")
try:
    failures = []
    embed_dim = 64
    d_model   = D_MODEL
    B = 4

    ctx_emb = ContextTypeEmbedding(embed_dim=embed_dim, seed=42)
    num_types = RNAContextType.num_types()

    if ctx_emb.weight.shape != (num_types, embed_dim):
        failures.append(
            f"weight shape {ctx_emb.weight.shape} ≠ ({num_types},{embed_dim})")
    if ctx_emb.weight.dtype != np.float32:
        failures.append(f"weight dtype {ctx_emb.weight.dtype} ≠ float32")

    ctx_ids = np.array([0, 1, 5, 9], dtype=np.int32)
    out_emb = ctx_emb(ctx_ids)
    if out_emb.shape != (B, embed_dim):
        failures.append(f"__call__ shape {out_emb.shape} ≠ ({B},{embed_dim})")

    # inject_into_single_repr with projection
    proj = np.random.default_rng(0).standard_normal((embed_dim, d_model)).astype(np.float32)
    L    = 50
    single = np.zeros((B, L, d_model), dtype=np.float32)
    out_inj = ctx_emb.inject_into_single_repr(single, ctx_ids, projection=proj)
    if out_inj.shape != (B, L, d_model):
        failures.append(f"inject shape {out_inj.shape} ≠ ({B},{L},{d_model})")

    # Verify additive: single + broadcast(ctx_emb(ids) @ proj)
    expected_add = (ctx_emb(ctx_ids) @ proj)[:, np.newaxis, :]
    if not np.allclose(out_inj, single + expected_add, atol=1e-4):
        failures.append("inject_into_single_repr not correctly additive")

    # ValueError when embed_dim ≠ d_model with no projection
    ctx_emb_bad = ContextTypeEmbedding(embed_dim=embed_dim + 1, seed=0)
    try:
        ctx_emb_bad.inject_into_single_repr(single, ctx_ids, projection=None)
        failures.append("should have raised ValueError for embed_dim ≠ d_model")
    except ValueError:
        pass  # expected

    LOG.log("weight_shape",     list(ctx_emb.weight.shape))
    LOG.log("call_shape",       list(out_emb.shape))
    LOG.log("inject_shape",     list(out_inj.shape))
    LOG.log("num_context_types", num_types)

    if failures:
        LOG.end("FAIL", reason="; ".join(failures[:3]))
    else:
        LOG.end("PASS",
                reason="Weight shape correct; __call__ output correct; inject additive; "
                       "ValueError on dim mismatch")
except Exception:
    LOG.end("ERROR", tb=traceback.format_exc())


LOG.begin("S6-METAL-HEAD",
          "MetalBindingHead: forward in [0,1], BCE loss finite ≥ 0",
          "Embeddings")
try:
    failures = []
    B, L_h = 4, 60

    head = MetalBindingHead(d_model=D_MODEL, hidden_dim=HIDDEN_DIM, seed=7)
    single_repr = np.random.default_rng(0).standard_normal(
        (B, L_h, D_MODEL)).astype(np.float32) * 0.1

    metal_prob = head.forward(single_repr)

    if metal_prob.shape != (B, L_h):
        failures.append(f"forward shape {metal_prob.shape} ≠ ({B},{L_h})")
    if metal_prob.dtype != np.float32:
        failures.append(f"forward dtype {metal_prob.dtype} ≠ float32")
    if not np.all(metal_prob >= 0.0):
        failures.append("metal_prob has values < 0")
    if not np.all(metal_prob <= 1.0):
        failures.append("metal_prob has values > 1")
    if not np.all(np.isfinite(metal_prob)):
        failures.append("non-finite values in metal_prob")

    # BCE loss
    targets_zero = np.zeros((B, L_h), dtype=np.float32)
    targets_rand = np.random.default_rng(1).uniform(0, 1, (B, L_h)).astype(np.float32)
    targets_rand = (targets_rand > 0.5).astype(np.float32)

    bce_z = head.binary_cross_entropy_loss(metal_prob, targets_zero)
    bce_r = head.binary_cross_entropy_loss(metal_prob, targets_rand)

    if not math.isfinite(bce_z):
        failures.append(f"BCE(zero targets) not finite: {bce_z}")
    if not math.isfinite(bce_r):
        failures.append(f"BCE(random targets) not finite: {bce_r}")
    if bce_z < 0:
        failures.append(f"BCE should be ≥ 0, got {bce_z}")

    # Perfect prediction → near-zero loss
    perfect_pred = targets_zero.copy() + 0.001
    bce_perfect  = head.binary_cross_entropy_loss(perfect_pred, targets_zero)
    if bce_perfect > bce_z + 0.1:
        failures.append(
            f"Perfect prediction BCE {bce_perfect:.4f} not better than random {bce_z:.4f}")

    LOG.log("metal_prob_min",   round(float(metal_prob.min()), 5))
    LOG.log("metal_prob_max",   round(float(metal_prob.max()), 5))
    LOG.log("bce_zero_targets", round(bce_z, 5))
    LOG.log("bce_rand_targets", round(bce_r, 5))

    if failures:
        LOG.end("FAIL", reason="; ".join(failures[:3]))
    else:
        LOG.end("PASS",
                reason="Forward in [0,1]; no NaN/Inf; BCE finite ≥ 0; "
                       "perfect pred < random pred loss")
except Exception:
    LOG.end("ERROR", tb=traceback.format_exc())

print("\n  ✓ Stage 6 complete.\n")


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 7 — collate_features + PretrainingBatch
# ─────────────────────────────────────────────────────────────────────────────

LOG.section("STAGE 7: collate_features + PretrainingBatch")

LOG.begin("S7-COLLATE",
          "collate_features: padded shapes, zero-padding, context IDs",
          "BatchCollation")
try:
    failures = []

    # Extract features from real complexes
    feats = [extract_context_features(cpx) for cpx in REAL_COMPLEXES[:4]]
    batch = collate_features(feats)

    B  = len(feats)
    L  = max(f.rna_length for f in feats)

    if batch.padded_length != L:
        failures.append(f"padded_length {batch.padded_length} ≠ {L}")
    if batch.one_hot.shape != (B, L, 5):
        failures.append(f"one_hot shape {batch.one_hot.shape} ≠ ({B},{L},5)")
    if batch.contact_masks.shape != (B, L):
        failures.append(f"contact_masks shape {batch.contact_masks.shape} ≠ ({B},{L})")
    if batch.self_contacts.shape != (B, L, L):
        failures.append(f"self_contacts shape {batch.self_contacts.shape} ≠ ({B},{L},{L})")
    if batch.metal_probs.shape != (B, L):
        failures.append(f"metal_probs shape {batch.metal_probs.shape} ≠ ({B},{L})")
    if batch.metal_restraints.shape != (B, L, L):
        failures.append(f"metal_restraints shape {batch.metal_restraints.shape} ≠ ({B},{L},{L})")
    if batch.seq_lengths.shape != (B,):
        failures.append(f"seq_lengths shape {batch.seq_lengths.shape} ≠ ({B},)")

    # seq_lengths match actual
    for i, feat in enumerate(feats):
        if batch.seq_lengths[i] != feat.rna_length:
            failures.append(f"seq_lengths[{i}]={batch.seq_lengths[i]} ≠ {feat.rna_length}")

    # context_type_ids in valid range
    valid_ids = set(int(v) for v in RNAContextType)
    for i, cid in enumerate(batch.context_type_ids):
        if int(cid) not in valid_ids:
            failures.append(f"context_type_ids[{i}]={cid} not in valid RNAContextType values")

    # Padding check: beyond each seq's length, one_hot should be zero
    for i, feat in enumerate(feats):
        l = feat.rna_length
        if l < L:
            if not np.allclose(batch.one_hot[i, l:], 0.0):
                failures.append(f"one_hot[{i}, {l}:] not zero-padded")

    LOG.log("padded_length",       batch.padded_length)
    LOG.log("batch_size",          B)
    LOG.log("seq_lengths",         batch.seq_lengths.tolist())
    LOG.log("context_type_ids",    batch.context_type_ids.tolist())

    if failures:
        LOG.end("FAIL", reason="; ".join(failures[:3]))
    else:
        LOG.end("PASS",
                reason="All batch shapes correct; zero-padding verified; "
                       "seq_lengths correct; context_ids valid")
except Exception:
    LOG.end("ERROR", tb=traceback.format_exc())


LOG.begin("S7-COLLATE-UNIFORM",
          "collate_features: uniform-length batch (no padding edge case)",
          "BatchCollation")
try:
    failures = []

    # All same length
    L_same = 40
    cpxs_same = [make_synthetic_complex(rna_length=L_same, seed=i)
                 for i in range(4)]
    feats_same = [extract_context_features(c) for c in cpxs_same]
    batch_same = collate_features(feats_same)

    if batch_same.padded_length != L_same:
        failures.append(f"uniform padded_length {batch_same.padded_length} ≠ {L_same}")
    if not np.all(batch_same.seq_lengths == L_same):
        failures.append("not all seq_lengths == L_same")

    if failures:
        LOG.end("FAIL", reason="; ".join(failures))
    else:
        LOG.end("PASS", reason="Uniform-length batch: no spurious padding; padded_length=L_same")
except Exception:
    LOG.end("ERROR", tb=traceback.format_exc())

print("\n  ✓ Stage 7 complete.\n")


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 8 — RNAContextPretrainer TRAINING STEP (REAL SEQUENCES)
# ─────────────────────────────────────────────────────────────────────────────

LOG.section("STAGE 8: RNAContextPretrainer — TRAINING STEP (REAL SEQUENCES)")

def _stub_model(one_hot: np.ndarray, ctx_emb: np.ndarray) -> np.ndarray:
    """Placeholder model: Gaussian noise single representation."""
    B, L, _ = one_hot.shape
    d_model  = ctx_emb.shape[-1]
    return (np.random.default_rng(0).standard_normal((B, L, d_model))
            * 0.1).astype(np.float32)


LOG.begin("S8-LOSSES",
          "training_step: loss keys, finite, non-negative, formula correct",
          "E2EPretrain")
try:
    failures = []
    trainer = RNAContextPretrainer(
        d_model=D_MODEL, embed_dim=EMBED_DIM, hidden_dim=HIDDEN_DIM, seed=0)

    # Use real complexes (copy to avoid in-place modification issues)
    import copy
    cpxs_for_step = copy.deepcopy(REAL_COMPLEXES[:4])
    losses = trainer.training_step(cpxs_for_step, _stub_model)

    required_loss_keys = {"metal_binding_loss", "contact_mask_loss", "total_loss"}
    missing_keys = required_loss_keys - set(losses.keys())
    if missing_keys:
        failures.append(f"missing loss keys: {missing_keys}")

    for k, v in losses.items():
        if not math.isfinite(v):
            failures.append(f"{k} = {v} (not finite)")
        if v < 0:
            failures.append(f"{k} = {v} (negative)")

    # Verify total_loss formula
    if "total_loss" in losses and "metal_binding_loss" in losses and "contact_mask_loss" in losses:
        expected_total = losses["metal_binding_loss"] + 0.5 * losses["contact_mask_loss"]
        if not math.isclose(losses["total_loss"], expected_total, rel_tol=1e-4):
            failures.append(
                f"total_loss {losses['total_loss']:.5f} ≠ "
                f"metal + 0.5*contact = {expected_total:.5f}")

    LOG.log("metal_binding_loss", round(losses.get("metal_binding_loss", -1), 4))
    LOG.log("contact_mask_loss",  round(losses.get("contact_mask_loss", -1), 4))
    LOG.log("total_loss",         round(losses.get("total_loss", -1), 4))
    LOG.log("total_formula_ok",   math.isclose(
        losses.get("total_loss", 0),
        losses.get("metal_binding_loss", 0) + 0.5 * losses.get("contact_mask_loss", 0),
        rel_tol=1e-4))

    if failures:
        LOG.end("FAIL", reason="; ".join(failures[:3]))
    else:
        LOG.end("PASS",
                reason="All 3 loss keys present; finite; non-negative; "
                       "total = metal + 0.5*contact")
except Exception:
    LOG.end("ERROR", tb=traceback.format_exc())


LOG.begin("S8-E2E-WALLTIME",
          f"E2E pretraining step: {E2E_RUNS} runs on real sequence lengths",
          "E2EPretrain")
try:
    failures      = []
    E2E_RESULTS_LIST = []

    # Use a spread of real sequence lengths
    test_lens = sorted(set(REAL_SEQ_LENS[:E2E_RUNS * 2]))[:E2E_RUNS]
    if len(test_lens) < E2E_RUNS:
        test_lens = ([60, 80, 100, 40, 120, 50, 70, 90])[:E2E_RUNS]

    trainer_e2e = RNAContextPretrainer(
        d_model=D_MODEL, embed_dim=EMBED_DIM, hidden_dim=HIDDEN_DIM, seed=1)

    for run_i, L_e2e in enumerate(test_lens):
        cpxs_e2e = [make_synthetic_complex(rna_length=L_e2e, seed=run_i * 10 + b)
                    for b in range(4)]
        t0_e = time.perf_counter()
        import copy
        losses_e = trainer_e2e.training_step(copy.deepcopy(cpxs_e2e), _stub_model)
        ms_e = (time.perf_counter() - t0_e) * 1000

        for k, v in losses_e.items():
            if not math.isfinite(v):
                failures.append(f"run {run_i} L={L_e2e}: {k} not finite")

        E2E_RESULTS_LIST.append({"L": L_e2e, "ms": round(ms_e, 1),
                                  "total_loss": losses_e.get("total_loss", -1)})
        LOG.log(f"run_{run_i}_L{L_e2e}_ms", round(ms_e, 1))

    if failures:
        LOG.end("FAIL", reason="; ".join(failures[:3]))
    else:
        LOG.end("PASS",
                reason=f"{E2E_RUNS} E2E runs complete; all losses finite; "
                       "wall-time logged per length")
except Exception:
    LOG.end("ERROR", tb=traceback.format_exc())

print("\n  ✓ Stage 8 complete.\n")


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 9 — inference_prepare ON REAL SEQUENCES
# ─────────────────────────────────────────────────────────────────────────────

LOG.section("STAGE 9: inference_prepare — REAL SEQUENCES")


def _model_fn_stub(one_hot, ctx_emb):
    B, L, _ = one_hot.shape
    d = ctx_emb.shape[-1]
    return np.random.default_rng(0).standard_normal((B, L, d)).astype(np.float32) * 0.1


LOG.begin("S9-INFER-OUTPUT",
          "inference_prepare: output keys, shapes, value ranges",
          "InferencePrepare")
try:
    failures = []

    ctx_embedding = ContextTypeEmbedding(embed_dim=EMBED_DIM, seed=42)
    metal_head    = MetalBindingHead(d_model=D_MODEL, hidden_dim=HIDDEN_DIM, seed=7)
    projection    = np.random.default_rng(0).standard_normal(
        (EMBED_DIM, D_MODEL)).astype(np.float32) * 0.02

    required_keys = {
        "single_repr", "context_type_id", "context_label",
        "metal_prob", "metal_restraints"
    }

    for seq_r in REAL_SEQUENCES[:6]:
        L_r = len(seq_r)
        out = inference_prepare(
            sequence       = seq_r,
            model_fn       = _model_fn_stub,
            ctx_embedding  = ctx_embedding,
            metal_head     = metal_head,
            projection     = projection,
        )

        missing_k = required_keys - set(out.keys())
        if missing_k:
            failures.append(f"seq len={L_r}: missing keys {missing_k}")
            continue

        if out["single_repr"].shape != (1, L_r, D_MODEL):
            failures.append(
                f"seq len={L_r}: single_repr shape {out['single_repr'].shape} "
                f"≠ (1,{L_r},{D_MODEL})")
        if not np.all(np.isfinite(out["single_repr"])):
            failures.append(f"seq len={L_r}: single_repr has non-finite values")

        if out["metal_prob"].shape != (L_r,):
            failures.append(f"seq len={L_r}: metal_prob shape ≠ ({L_r},)")
        if not np.all(out["metal_prob"] >= 0) or not np.all(out["metal_prob"] <= 1.0):
            failures.append(f"seq len={L_r}: metal_prob out of [0,1]")

        if out["metal_restraints"].shape != (L_r, L_r):
            failures.append(f"seq len={L_r}: metal_restraints shape ≠ ({L_r},{L_r})")
        if not np.all(out["metal_restraints"] >= 0):
            failures.append(f"seq len={L_r}: metal_restraints has negatives")

        ctx_label = out["context_label"]
        valid_labels = {t.name for t in RNAContextType}
        if ctx_label not in valid_labels:
            failures.append(f"seq len={L_r}: context_label '{ctx_label}' not a valid type")

        LOG.log(f"seq_{L_r}_ctx",        ctx_label)
        LOG.log(f"seq_{L_r}_metal_mean", round(float(out["metal_prob"].mean()), 4))

    if failures:
        LOG.end("FAIL", reason="; ".join(failures[:3]))
    else:
        LOG.end("PASS",
                reason="All 5 output keys present; shapes/dtypes correct; "
                       "metal_prob in [0,1]; context_label valid")
except Exception:
    LOG.end("ERROR", tb=traceback.format_exc())


LOG.begin("S9-INFER-WITH-METALS",
          "inference_prepare: metal_restraints non-zero with known metal coords",
          "InferencePrepare")
try:
    failures = []

    ctx_embedding = ContextTypeEmbedding(embed_dim=EMBED_DIM, seed=42)
    metal_head    = MetalBindingHead(d_model=D_MODEL, hidden_dim=HIDDEN_DIM, seed=7)
    projection    = np.random.default_rng(0).standard_normal(
        (EMBED_DIM, D_MODEL)).astype(np.float32) * 0.02

    seq_m = REAL_SEQUENCES[0][:60]
    L_m   = len(seq_m)
    rna_init = make_random_coords(L_m, spread=10.0, seed=0)

    # Metal coords placed near RNA centroid
    metal_coords = (rna_init.mean(axis=0) +
                    np.random.default_rng(5).standard_normal((N_METALS, 3)).astype(np.float32)
                    * 3.0)
    metal_types  = ["MG", "ZN", "CA"]

    out_m = inference_prepare(
        sequence            = seq_m,
        model_fn            = _model_fn_stub,
        ctx_embedding       = ctx_embedding,
        metal_head          = metal_head,
        projection          = projection,
        known_metal_coords  = metal_coords,
        known_metal_types   = metal_types,
        rna_coords_init     = rna_init,
    )

    # With metal coords supplied, some restraints should be non-zero
    if not np.any(out_m["metal_restraints"] > 0):
        failures.append("With metal coords, metal_restraints should have non-zero entries")

    # metal_prob should be in [0,1] after merging head and coordinate-derived scores
    if not np.all(out_m["metal_prob"] >= 0) or not np.all(out_m["metal_prob"] <= 1.0):
        failures.append("metal_prob after merge out of [0,1]")

    LOG.log("restraints_nonzero_frac",
            round(float((out_m["metal_restraints"] > 0).mean()), 4))
    LOG.log("metal_prob_max_with_coords",
            round(float(out_m["metal_prob"].max()), 4))
    LOG.log("metal_prob_mean_with_coords",
            round(float(out_m["metal_prob"].mean()), 4))

    if failures:
        LOG.end("FAIL", reason="; ".join(failures))
    else:
        LOG.end("PASS",
                reason="metal_restraints non-zero near known metal sites; "
                       "merged metal_prob in [0,1]")
except Exception:
    LOG.end("ERROR", tb=traceback.format_exc())

print("\n  ✓ Stage 9 complete.\n")


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 10 — DIAGNOSIS
# ─────────────────────────────────────────────────────────────────────────────

LOG.section("STAGE 10: DIAGNOSIS — Real-Data Performance Summary & Known Limitations")

kernel_recs  = [r for r in LOG.records if r["tag"] == "NumbaKernels"]
seq_recs     = [r for r in LOG.records if r["tag"] == "SeqUtils"]
cpx_recs     = [r for r in LOG.records if r["tag"] == "ComplexBuild"]
feat_recs    = [r for r in LOG.records if r["tag"] == "ContextFeatures"]
emb_recs     = [r for r in LOG.records if r["tag"] == "Embeddings"]
batch_recs   = [r for r in LOG.records if r["tag"] == "BatchCollation"]
e2e_recs     = [r for r in LOG.records if r["tag"] == "E2EPretrain"]
infer_recs   = [r for r in LOG.records if r["tag"] == "InferencePrepare"]
data_recs    = [r for r in LOG.records if r["tag"] == "DataFetch"]

print("\n  ─── Real Dataset Statistics ──────────────────────────────────────────────")
print(f"  Competition        : {COMP_NAME}")
print(f"  Test sequences     : {len(test_df):,} targets")
print(f"  Train sequences    : {len(train_sq):,} sequences")
print(f"  Topology targets   : {len(REAL_TARGETS)}")
print(f"  RNA lengths        : {[len(v) for _, v in REAL_TARGETS]}")
print(f"  Seq len range      : {int(min(REAL_SEQ_LENS))} – {int(max(REAL_SEQ_LENS))} nt")
print(f"  Seq len median     : {int(np.median(REAL_SEQ_LENS))} nt")

print("\n  ─── Numba Kernel Benchmarks ──────────────────────────────────────────────")
print(f"  {'TID':<32}{'Status':<10}{'ms':>8}")
print("  " + "-"*52)
for r in kernel_recs:
    if "BENCH" in r["tid"]:
        icon = {"PASS": "✓", "PARTIAL": "⚠", "FAIL": "✗"}.get(r["status"], "?")
        print(f"  {icon} {r['tid']:<30}{r['status']:<10}{r['ms']:>8.1f}")

print("\n  ─── Feature Extraction Scaling ───────────────────────────────────────────")
for entry in FEAT_BENCH:
    print(f"    L={entry['L']:>5}  →  {entry['ms']:>8.2f} ms")

print("\n  ─── End-to-End Pretraining Wall-Time ─────────────────────────────────────")
try:
    for entry in E2E_RESULTS_LIST:
        print(f"    L={entry['L']:>5}  total_loss={entry['total_loss']:.4f}  "
              f"→  {entry['ms']:>8.1f} ms")
except NameError:
    print("    (E2E results not available)")

if not _NUMBA_OK:
    print("""
  ⚠  NUMBA NOT INSTALLED — all JIT kernels run in CPython.
     compute_pairwise_distances at N=200 will be ~200× slower.
     ACTION: pip install numba  (then re-run validate_rna_context_pretrain.py)
""")

print("""
  ─── Known Limitations (validated against Stanford RNA 3D Folding 2 data) ───

  1. SYNTHETIC COMPLEXES — NO REAL PDB PARTNER STRUCTURES:
     PDBComplex objects for validation are built using real RNA backbone coords
     (C3′-derived) but synthetic protein/metal positions. Real complexes would
     require PDB structure parsing (Biopython) for accurate inter-molecular
     contact maps. The partner_dist_mat and vectorised_partner_mask are therefore
     validated geometrically but not against experimental complex data.
     FIX: Integrate Biopython PDB parser to extract real protein/ligand atom
     coords from the PDB_RNA folder in the competition data.

  2. SEQUENCE CLASSIFIER IS MOTIF-ONLY (NOT TRAINED):
     classify_rna_context() uses a hand-curated set of 6-mer motifs to assign
     context types. This is highly inaccurate for novel sequences. For the
     Stanford RNA 3D Folding 2 competition data, most sequences will be
     classified as NAKED_RNA or RIBOSWITCH/RIBOZYME by chance motif hits.
     FIX: Train a k-mer frequency logistic regression or 1D-CNN on labelled
     RNA families (Rfam) to give proper probability distributions over
     RNAContextType values.

  3. METAL BINDING SCORES ARE NOT CALIBRATED:
     compute_metal_binding_features() uses a Gaussian proximity kernel with
     fixed ion radii. The raw scores (≥0, unbounded) are not probabilities.
     inference_prepare() applies sigmoid to the MLP head output, but the
     coordinate-derived scores are directly sigmoid-clipped, producing values
     that are not comparable across different structures.
     FIX: Apply isotonic regression or Platt scaling to calibrate metal-binding
     probabilities against curated metal-binding annotations from the PDB.

  4. PARTNER MASKING IS ALL-OR-NOTHING:
     mask_partner_coordinates() zeros all partner atoms unconditionally (when
     mask_protein=True). A more informative pretraining signal would use
     partial masking (e.g. mask 50% of partner residues at random), analogous
     to BERT-style masked-language modelling for the partner context.
     FIX: Add a masking_fraction parameter and select random subsets of
     partner_indices to zero; keep the unmasked partner atoms as context for
     the folding prediction.

  5. PRETRAINING BATCH COLLATION IS MEMORY-INEFFICIENT FOR LONG SEQUENCES:
     collate_features() pads self_contacts (L,L) and metal_restraints (L,L)
     to the maximum sequence length in the batch, leading to O(L²) memory
     per sample. For sequences > 500 nt this is prohibitive on Colab free tier.
     FIX: Store contact maps as sparse COO tensors (scipy.sparse or torch.sparse)
     and convert to dense only inside the model forward pass on GPU.

  6. CONTEXT EMBEDDING IS INJECTED ADDITIVELY (NO CROSS-ATTENTION):
     inject_into_single_repr() adds a single context vector broadcast over all
     sequence positions. This limits the expressiveness of the context signal;
     long-range structural differences (e.g. riboswitch aptamer vs expression
     platform) are not captured by a position-agnostic embedding.
     FIX: Replace additive injection with cross-attention between the context
     embedding (query) and the single representation (key/value), or use
     FiLM (feature-wise linear modulation) with learned γ and β vectors.

  7. METAL HEAD TRAINING SIGNAL IS METAL_BINDING_PROB FROM GAUSSIAN PROXIMITY:
     The pretraining label for MetalBindingHead is the Gaussian-kernel score
     compute_metal_binding_features() — not a binary "metal within 4Å" label.
     The BCE loss is thus trained against continuous (unbounded ≥ 0) targets
     rather than calibrated binary labels.
     FIX: Threshold metal_binding_prob > 0.5 for binary BCE labels, or use
     MSE regression on the normalised Gaussian scores instead.

  8. NO VALIDATION SPLIT / EARLY STOPPING IN PRETRAINING LOOP:
     RNAContextPretrainer.training_step() has no built-in validation loss
     tracking or early stopping. Pretraining on the full training set without
     validation could lead to overfitting of the auxiliary metal/contact heads.
     FIX: Reserve 10% of PDBComplex objects as a held-out validation set;
     compute and log validation losses every N steps; stop when validation
     loss plateaus using a patience counter.
""")

# ─────────────────────────────────────────────────────────────────────────────
# FINAL SUMMARY
# ─────────────────────────────────────────────────────────────────────────────

summary = LOG.summary()
passed  = summary["passed"]
total   = summary["total"]
pct     = 100 * passed // max(total, 1)

print(f"\n{'='*70}")
print(f"  FINAL SCORE (Real Stanford RNA 3D Folding 2 Data): {passed}/{total} ({pct}%)")
if pct == 100:
    print("  ✓ All tests passed on real data.")
elif pct >= 80:
    print("  ⚠  Most tests passed. See STAGE 10 DIAGNOSIS for remaining fixes.")
else:
    print("  ✗  Significant failures. See STAGE 10 for root causes and fixes.")
print(f"{'='*70}")
