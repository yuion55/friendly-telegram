"""
validate_rna_extensions.py  (v3 — REAL Kaggle Data)
====================================================
Validation script for rna_extensions.py using the real
Stanford RNA 3D Folding 2 Kaggle competition dataset.

Run in Colab after uploading kaggle.json:
    !python validate_rna_extensions.py

What this script does
─────────────────────
  STAGE 0  Auto-patches rna_extensions.py to fix the silent import bug
           (_WC_BASE_W / _WC_IS_GC / _TURNER_ARR missing from rna_topology)
           then reloads both modules cleanly.

  STAGE 1  Surgically fetches only the three CSVs needed from Kaggle:
             test_sequences.csv  — real test targets (incl. 9MME N=4640)
             train_sequences.csv — all training sequences
             train_labels.csv    — 3D C1′ coordinates

  STAGE 2  P3 + P5 on REAL sequences
             • Hardest test target (longest N) from test_sequences.csv
             • 5 more real test targets (longest → shortest)
             • HierarchicalFolder.fold()  — domain decomp + coaxial stacks
             • IonContactPredictor        — GNRA/UNCG, Mg²⁺ sites
             • Records: fold time, n_domains, n_pairs, n_ion_sites, n_gnra

  STAGE 3  P4 on REAL training structures (500 entries, RAM-safe)
             • Loads 500 real 3D-coordinate structures into TrainingTemplateDB
             • Queries template DB for every test target
             • Reports: top template TID, topology score, size ratio

  STAGE 4  Integration pipeline on REAL data
             • fold → find_templates → augment_pairs → annotate_motifs
             • Checks: all WC pairs preserved, ion pairs added,
               template returned, folding speed within budget

  STAGE 5  Full diagnostic: where the module is lacking on REAL inputs
"""

import os, sys, time, importlib, subprocess, math, traceback, textwrap, warnings
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
COMP_NAME     = "stanford-rna-3d-folding-2"
DATA_DIR      = "stanford_data"
KAGGLE_CFG    = os.getcwd()
MAX_TEMPLATES = 500
MAX_TEST_SEQS = 6
# train_labels.csv is long-format (one row per residue).  To load MAX_TEMPLATES
# complete structures we need MAX_TEMPLATES x avg_seq_length rows.  Training
# sequences range from ~20-4640 nt; 300 is a conservative average giving a
# generous safety margin without pulling the entire 317 MB file into RAM.
MAX_LABEL_ROWS = MAX_TEMPLATES * 300  # = 150,000 rows ~ 35 MB

# ─────────────────────────────────────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────────────────────────────────────

class Logger:
    def __init__(self):
        self.records: List[Dict] = []
        self._cur = None
        self._t0  = 0.0

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
        icon = {"PASS":"✓","PARTIAL":"⚠","FAIL":"✗","ERROR":"✗"}.get(status,"?")
        print(f"\n  {icon}  STATUS: {status}  ({ms:.0f} ms)")
        if reason:
            print(f"     Reason : {reason}")
        if tb:
            print(f"     Traceback:\n{textwrap.indent(tb, '       ')}")

    def summary(self) -> Dict:
        total   = len(self.records)
        passed  = sum(1 for r in self.records if r["status"] == "PASS")
        partial = sum(1 for r in self.records if r["status"] == "PARTIAL")
        failed  = sum(1 for r in self.records if r["status"] in ("FAIL","ERROR"))

        print(f"\n{'='*70}")
        print("  VALIDATION SUMMARY — Real Stanford RNA 3D Folding 2 Data")
        print(f"{'='*70}")
        print(f"  {'TID':<10}{'Tag':<22}{'Test':<28}{'Status':<10}ms")
        print("  " + "-"*70)
        for r in self.records:
            icon = {"PASS":"✓","PARTIAL":"⚠"}.get(r["status"],"✗")
            print(f"  {r['tid']:<10}{r['tag']:<22}{r['name'][:26]:<28}"
                  f"{icon+' '+r['status']:<10}{r['ms']:.0f}")
        print("  " + "-"*70)
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
# STAGE 0 — AUTO-PATCH + MODULE RELOAD
# ─────────────────────────────────────────────────────────────────────────────

LOG.section("STAGE 0: AUTO-PATCH SILENT BUG + MODULE RELOAD")

EXT_FILE  = "rna_extensions.py"

if os.path.exists(EXT_FILE):
    with open(EXT_FILE, "r") as fh:
        code = fh.read()
    dirty = False

    # Patch 1: remove constants that rna_topology never exported
    bad = "_WC_BASE_W, _WC_IS_GC, _TURNER_ARR,"
    if bad in code:
        code = code.replace(bad, "")
        dirty = True

    # Patch 2: _NUC_ENC is also not in rna_topology — remove from import AND
    # inject an unconditional module-level definition so it is always available.
    if "_NUC_ENC," in code or ", _NUC_ENC" in code:
        code = code.replace(", _NUC_ENC,", ",").replace("_NUC_ENC, ", "").replace(", _NUC_ENC", "")
        dirty = True
    _NUC_SENTINEL = "_NUC_ENC = {'A': 0, 'U': 1, 'T': 1, 'G': 2, 'C': 3}  # auto-patched"
    if _NUC_SENTINEL not in code and "# Coaxial stacking parameters" in code:
        code = code.replace(
            "# Coaxial stacking parameters",
            _NUC_SENTINEL + "\n\n# Coaxial stacking parameters",
            1
        )
        dirty = True

    if dirty:
        with open(EXT_FILE, "w") as fh:
            fh.write(code)
        print("  ✓ Patched rna_extensions.py — fixed missing constants")
    else:
        print("  ✓ rna_extensions.py already clean")
else:
    print(f"  ✗ {EXT_FILE} not found in {os.getcwd()}")
    sys.exit(1)

if os.getcwd() not in sys.path:
    sys.path.insert(0, os.getcwd())

# Fresh reload
for mod in ("rna_topology", "rna_extensions"):
    if mod in sys.modules:
        del sys.modules[mod]

try:
    import rna_topology
    print(f"  ✓ rna_topology loaded  (Numba={rna_topology._NUMBA_AVAILABLE})")
except Exception as e:
    print(f"  ✗ Cannot import rna_topology: {e}")
    sys.exit(1)

try:
    import rna_extensions
    print(f"  ✓ rna_extensions loaded")
    print(f"    Numba available        : {rna_extensions._NUMBA_AVAILABLE}")
    print(f"    rna_topology available : {rna_extensions._TOPOLOGY_AVAILABLE}")
except Exception as e:
    print(f"  ✗ Cannot import rna_extensions: {e}")
    sys.exit(1)

if not rna_extensions._TOPOLOGY_AVAILABLE:
    print("\n  ✗ CRITICAL: rna_topology still not available.")
    print("    Restart Colab runtime (Runtime → Restart) and re-run.")
    print(f"    CWD: {os.getcwd()}  |  Files: {[f for f in os.listdir('.') if f.endswith('.py')]}")
    sys.exit(1)

from rna_extensions import HierarchicalFolder, TrainingTemplateDB, IonContactPredictor
folder   = HierarchicalFolder()
tdb      = TrainingTemplateDB()
ion_pred = IonContactPredictor()
print("\n  ✓ All modules ready. Starting real-data validation.\n")


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 1 — SURGICAL KAGGLE FETCH
# ─────────────────────────────────────────────────────────────────────────────

LOG.section("STAGE 1: SURGICAL FETCH — Real Kaggle Competition Data")

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
        print(f"  ✗ Download failed:\n    {result.stderr.strip()}")
        sys.exit(1)
    zip_path = path + ".zip"
    if os.path.exists(zip_path):
        subprocess.run(["unzip", "-o", "-q", zip_path, "-d", DATA_DIR])
        os.remove(zip_path)
    size_mb = os.path.getsize(path) / 1_048_576 if os.path.exists(path) else 0
    print(f"  ✓ Downloaded {filename}  ({size_mb:.1f} MB)")
    return path


TEST_CSV  = fetch_file("test_sequences.csv")
TRAIN_SEQ = fetch_file("train_sequences.csv")
TRAIN_LAB = fetch_file("train_labels.csv")

test_df  = pd.read_csv(TEST_CSV)
train_sq = pd.read_csv(TRAIN_SEQ)
train_lb = pd.read_csv(TRAIN_LAB, nrows=MAX_LABEL_ROWS)

def id_col(df):
    """Return the name of the ID column in df, case-insensitively."""
    for candidate in ("target_id", "ID", "id"):
        if candidate in df.columns:
            return candidate
    # Last resort: first column whose name contains 'id'
    for c in df.columns:
        if "id" in c.lower():
            return c
    return df.columns[0]

test_df["_len"] = test_df["sequence"].str.len()
test_df = test_df.sort_values("_len", ascending=False).reset_index(drop=True)
test_targets = test_df.head(MAX_TEST_SEQS)

print(f"\n  Dataset overview:")
print(f"    test_sequences  : {len(test_df):,} targets  "
      f"(longest={test_df['_len'].max()}, shortest={test_df['_len'].min()})")
print(f"    train_sequences : {len(train_sq):,} sequences")
print(f"    train_labels    : {len(train_lb):,} rows loaded (cap={MAX_LABEL_ROWS:,} rows → {MAX_TEMPLATES} structures)")
print(f"\n  Top {MAX_TEST_SEQS} targets by length:")
for _, row in test_targets.iterrows():
    print(f"    {row.get(id_col(test_df), '?')}   N={row['_len']}")


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 2 — P3 + P5 ON REAL TEST SEQUENCES
# ─────────────────────────────────────────────────────────────────────────────

LOG.section("STAGE 2: P3 (HierarchicalFolder) + P5 (IonContactPredictor) — Real Sequences")

# Store fold results for reuse in Stages 3 & 4
fold_results_cache: Dict[str, Dict] = {}

for rank, (_, test_row) in enumerate(test_targets.iterrows(), start=1):
    tid = str(test_row.get(id_col(test_df), f"IDX{rank}"))
    seq = str(test_row["sequence"])
    N   = len(seq)

    # ── P3: Fold ──────────────────────────────────────────────────────────
    LOG.begin(f"P3-{rank:02d}", f"Fold {tid} (N={N})", "P3-HierarchicalFolder")
    try:
        LOG.log("target_id",    tid)
        LOG.log("seq_length_N", N)
        LOG.log("seq_prefix",   seq[:50] + ("…" if N > 50 else ""))
        LOG.log("gc_content",   round(sum(c in "GCgc" for c in seq) / N, 3))

        max_loop = 300 if N > 4000 else (400 if N > 2000 else None)
        LOG.log("max_loop",     max_loop)

        t0   = time.perf_counter()
        fold = folder.fold(seq, max_loop=max_loop)
        ms   = (time.perf_counter() - t0) * 1000

        fold_results_cache[tid] = fold

        err     = fold.get("error", "")
        n_dom   = fold.get("n_domains", 0)
        n_pairs = len(fold.get("pairs", []))
        n_stems = len(fold.get("stems", []))
        n_coax  = len(fold.get("coaxial_stacks", []))

        LOG.log("fold_time_ms",     round(ms, 1))
        LOG.log("error_flag",       err)
        LOG.log("n_domains",        n_dom)
        LOG.log("n_base_pairs",     n_pairs)
        LOG.log("n_stems",          n_stems)
        LOG.log("n_coaxial_stacks", n_coax)
        if fold.get("domains"):
            sizes = [d["end"] - d["start"] + 1 for d in fold["domains"]]
            LOG.log("domain_sizes", sizes[:8])
        if fold.get("coaxial_stacks"):
            gaps = [cs.get("gap") for cs in fold["coaxial_stacks"][:5]]
            scores = [round(cs.get("score",0),3) for cs in fold["coaxial_stacks"][:5]]
            LOG.log("coax_gaps",   gaps)
            LOG.log("coax_scores", scores)

        if err:
            LOG.warn(f"fold() error: {err}")
            LOG.end("PARTIAL", reason=err)
        elif ms > 30_000:
            LOG.warn(f"Slow: {ms/1000:.1f}s for N={N}")
            LOG.end("PARTIAL", reason=f"Time={ms/1000:.1f}s exceeds 30s budget")
        elif n_dom == 0 and N > 5:
            LOG.warn("0 domains for non-trivial sequence")
            LOG.end("PARTIAL", reason="0 domains returned")
        else:
            LOG.end("PASS",
                    reason=f"N={N} → {n_dom} domains, {n_pairs} pairs, "
                           f"{n_coax} coaxial in {ms:.0f}ms")
    except Exception:
        LOG.end("ERROR", tb=traceback.format_exc())

    # ── P5: Ion predictor ─────────────────────────────────────────────────
    LOG.begin(f"P5-{rank:02d}", f"Ion {tid} (N={N})", "P5-IonPredictor")
    try:
        cached = fold_results_cache.get(tid, {})
        pairs_in = cached.get("pairs", [])
        fold_err = cached.get("error", "")
        LOG.log("n_pairs_input", len(pairs_in))
        LOG.log("fold_had_error", bool(fold_err))

        t0 = time.perf_counter()
        motifs = ion_pred.annotate_motifs(seq, pairs_in)
        ms_m   = (time.perf_counter() - t0) * 1000

        LOG.log("motif_time_ms",      round(ms_m, 1))
        LOG.log("n_gnra_tetraloops",  motifs.get("n_gnra"))
        LOG.log("n_uncg_tetraloops",  motifs.get("n_uncg"))
        LOG.log("n_aminor_motifs",    motifs.get("n_aminor"))
        LOG.log("n_mg2_sites",        motifs.get("n_ion_sites"))
        LOG.log("top5_ion_scores",    sorted(motifs.get("ion_scores",[]), reverse=True)[:5])

        t0 = time.perf_counter()
        aug = ion_pred.augment_pairs(seq, pairs_in)
        ms_a = (time.perf_counter() - t0) * 1000
        n_new = len(aug) - len(pairs_in)
        pres  = set(pairs_in).issubset(set(aug))

        LOG.log("augment_time_ms",    round(ms_a, 1))
        LOG.log("n_ion_pairs_added",  n_new)
        LOG.log("wc_pairs_preserved", pres)

        if not pres:
            LOG.err("Original WC pairs dropped by augment_pairs!")
            LOG.end("FAIL", reason="augment_pairs removed WC pairs")
        elif ms_m > 60_000:
            LOG.warn(f"Ion annotation took {ms_m/1000:.1f}s")
            LOG.end("PARTIAL", reason=f"Too slow: {ms_m/1000:.1f}s for N={N}")
        else:
            LOG.end("PASS",
                    reason=(f"GNRA={motifs['n_gnra']}, UNCG={motifs['n_uncg']}, "
                            f"Mg²⁺={motifs['n_ion_sites']}, +{n_new} ion pairs"))
    except Exception:
        LOG.end("ERROR", tb=traceback.format_exc())


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 3 — P4: TEMPLATE DB ON REAL TRAINING DATA
# ─────────────────────────────────────────────────────────────────────────────

LOG.section(f"STAGE 3: P4 (TrainingTemplateDB) — {MAX_TEMPLATES} Real Training Structures")

LOG.begin("P4-BUILD", "Build DB from real train_labels.csv", "P4-TemplateDB")
try:
    id_sq  = id_col(train_sq)
    id_lb  = id_col(train_lb)
    seq_map = dict(zip(train_sq[id_sq].astype(str), train_sq["sequence"]))
    # Also index by base target_id stripped of residue-index suffix e.g. R1107_1 → R1107
    for raw_k, v in list(seq_map.items()):
        parts = str(raw_k).rsplit("_", 1)
        if len(parts) == 2 and parts[1].isdigit():
            seq_map.setdefault(parts[0], v)
    LOG.log("seq_map_size", len(seq_map))

    x_cols = [c for c in train_lb.columns if c.startswith("x_")]
    y_cols = [c for c in train_lb.columns if c.startswith("y_")]
    z_cols = [c for c in train_lb.columns if c.startswith("z_")]
    LOG.log("n_coord_cols",    len(x_cols))
    LOG.log("first_3_x_cols",  x_cols[:3])
    LOG.log("sample_train_ids", train_lb[id_lb].head(3).tolist())
    LOG.log("sample_seq_ids",   train_sq[id_sq].head(3).tolist())

    # ── Format detection ──────────────────────────────────────────────────────
    # LONG format (actual competition CSV): one row per residue.
    #   id_lb column holds "{target_id}_{resnum}" (e.g. "R1107_1").
    #   x_cols = ["x_1"], y_cols = ["y_1"], z_cols = ["z_1"].
    # WIDE format: one row per structure with x_1…x_N columns.
    is_long = len(x_cols) <= 5

    t0 = time.perf_counter()
    n_added, n_skip = 0, 0
    len_dist = []

    if is_long:
        # ── LONG FORMAT ───────────────────────────────────────────────────────
        LOG.log("csv_format", "long (one-row-per-residue)")
        xc = x_cols[0] if x_cols else None
        yc = y_cols[0] if y_cols else None
        zc = z_cols[0] if z_cols else None
        if not (xc and yc and zc):
            LOG.err("Long-format CSV missing x/y/z columns")
            LOG.end("FAIL", reason="Missing coordinate columns")
            raise RuntimeError("no coords")

        # Group residues by target_id
        from collections import OrderedDict
        targets_buf: dict = OrderedDict()  # target_id → list of (resnum, x, y, z)

        for _, row in train_lb.iterrows():
            raw_id = str(row[id_lb])
            parts  = raw_id.rsplit("_", 1)
            if len(parts) == 2 and parts[1].isdigit():
                target_id = parts[0]
                resnum    = int(parts[1])
            else:
                target_id = raw_id
                resnum    = 0
            try:
                x, y, z = float(row[xc]), float(row[yc]), float(row[zc])
                if target_id not in targets_buf:
                    targets_buf[target_id] = []
                targets_buf[target_id].append((resnum, x, y, z))
            except (ValueError, KeyError):
                n_skip += 1

        # Build one template per target
        for target_id, residues in targets_buf.items():
            if n_added >= MAX_TEMPLATES:
                break
            seq_t = seq_map.get(target_id, "")
            if not seq_t:
                n_skip += 1
                continue
            residues.sort(key=lambda r: r[0])
            N_c = len(residues)
            if N_c < 5:
                n_skip += 1
                continue
            coords = np.array([[r[1], r[2], r[3]] for r in residues],
                               dtype=np.float64)
            tdb.add_from_coords(target_id, seq_t[:N_c], coords)
            n_added += 1
            len_dist.append(N_c)

    else:
        # ── WIDE FORMAT ───────────────────────────────────────────────────────
        LOG.log("csv_format", "wide (one-row-per-structure)")
        for _, row in train_lb.iterrows():
            if n_added >= MAX_TEMPLATES:
                break
            tid_t = str(row.get(id_lb, ""))
            # Normalise: strip residue-index suffix
            base_tid = tid_t.rsplit("_", 1)[0] \
                if "_" in tid_t and tid_t.rsplit("_", 1)[1].isdigit() \
                else tid_t
            seq_t = seq_map.get(tid_t, "") or seq_map.get(base_tid, "")
            if not seq_t:
                n_skip += 1
                continue
            try:
                xs = np.array([row[c] for c in x_cols if pd.notna(row[c])], dtype=np.float32)
                ys = np.array([row[c] for c in y_cols if pd.notna(row[c])], dtype=np.float32)
                zs = np.array([row[c] for c in z_cols if pd.notna(row[c])], dtype=np.float32)
                N_c = min(len(xs), len(ys), len(zs), len(seq_t))
                if N_c < 5:
                    n_skip += 1
                    continue
                coords = np.stack([xs[:N_c], ys[:N_c], zs[:N_c]], axis=1).astype(np.float64)
                tdb.add_from_coords(tid_t, seq_t[:N_c], coords)
                n_added += 1
                len_dist.append(N_c)
            except Exception:
                n_skip += 1

    ms = (time.perf_counter() - t0) * 1000
    LOG.log("n_added",     n_added)
    LOG.log("n_skipped",   n_skip)
    LOG.log("build_ms",    round(ms, 1))
    LOG.log("tdb_size",    tdb.n_templates)
    if len_dist:
        LOG.log("N_min",   min(len_dist))
        LOG.log("N_max",   max(len_dist))
        LOG.log("N_mean",  round(float(np.mean(len_dist)), 1))
    if tdb._all:
        s = tdb._all[0]
        LOG.log("sample_entry_tid",    s.get("tid"))
        LOG.log("sample_entry_N",      s.get("N"))
        LOG.log("sample_entry_pairs",  len(s.get("pairs",[])))
        LOG.log("sample_fingerprint",  s.get("fingerprint"))

    if n_added == 0:
        LOG.err("0 templates added — ID mismatch between CSVs or coord parsing failed")
        LOG.end("FAIL", reason="0 templates from real data")
    elif n_added < 10:
        LOG.warn(f"Only {n_added} templates loaded")
        LOG.end("PARTIAL", reason=f"Low count: {n_added}")
    else:
        LOG.end("PASS",
                reason=f"{n_added} structures indexed in {ms:.0f}ms "
                       f"(N={min(len_dist)}–{max(len_dist)})")
except Exception:
    LOG.end("ERROR", tb=traceback.format_exc())


# Template retrieval for each test target
for rank, (_, test_row) in enumerate(test_targets.iterrows(), start=1):
    tid = str(test_row.get(id_col(test_df), f"IDX{rank}"))
    seq = str(test_row["sequence"])
    N   = len(seq)

    LOG.begin(f"P4-{rank:02d}", f"find_templates {tid} (N={N})", "P4-TemplateDB")
    try:
        cached_pairs = fold_results_cache.get(tid, {}).get("pairs", [])
        fold_err     = fold_results_cache.get(tid, {}).get("error", "")
        LOG.log("n_query_pairs", len(cached_pairs))
        LOG.log("fold_had_error", bool(fold_err))

        t0 = time.perf_counter()
        templates = tdb.find_templates(seq, cached_pairs, k=5)
        ms = (time.perf_counter() - t0) * 1000

        LOG.log("query_ms",        round(ms, 1))
        LOG.log("n_found",         len(templates))
        if templates:
            LOG.log("top_tid",       templates[0].get("tid"))
            LOG.log("top_score",     round(templates[0].get("score",0), 4))
            LOG.log("top_size_ratio",round(templates[0].get("size_ratio",0), 3))
            LOG.log("top_N",         templates[0].get("N"))
            LOG.log("top_fp",        templates[0].get("fingerprint"))
            LOG.log("all_tids",      [t["tid"] for t in templates])
            LOG.log("all_scores",    [round(t["score"],4) for t in templates])

        if fold_err:
            LOG.warn(f"Fold error upstream: {fold_err} — querying on empty pairs")
        if len(templates) == 0:
            LOG.warn("0 templates returned")
            if tdb.n_templates > 0:
                # Show why: size mismatch
                sizes = [e["N"] for e in tdb._all[:20]]
                LOG.warn(f"DB N range: {min(sizes)}–{max(sizes)}  query N={N}")
                LOG.warn("size_tolerance=0.5 may filter all candidates for large N")
            LOG.end("PARTIAL", reason="0 templates found — size_tolerance filtering")
        else:
            LOG.end("PASS",
                    reason=f"Top: {templates[0]['tid']} "
                           f"(score={templates[0]['score']:.4f}, "
                           f"N_tmpl={templates[0].get('N')}) in {ms:.0f}ms")
    except Exception:
        LOG.end("ERROR", tb=traceback.format_exc())


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 4 — FULL INTEGRATION ON HARDEST REAL TARGET
# ─────────────────────────────────────────────────────────────────────────────

LOG.section("STAGE 4: Full Integration — Hardest Real Target")

LOG.begin("INTEG-01", "fold→templates→augment→motifs", "Integration")
try:
    hard = test_targets.iloc[0]
    tid_h = str(hard.get(id_col(test_df), "IDX0"))
    seq_h = str(hard["sequence"])
    N_h   = len(seq_h)

    LOG.log("target",  tid_h)
    LOG.log("N",       N_h)
    LOG.log("preview", seq_h[:60] + "…")

    print("\n    [1/4] Hierarchical fold …")
    t0 = time.perf_counter()
    fold_h = folder.fold(seq_h, max_loop=300 if N_h > 4000 else 400)
    ms1    = (time.perf_counter() - t0) * 1000
    pairs_h = fold_h.get("pairs", [])
    LOG.log("step1_fold_ms",    round(ms1,1))
    LOG.log("step1_n_pairs",    len(pairs_h))
    LOG.log("step1_n_domains",  fold_h.get("n_domains",0))
    LOG.log("step1_fold_error", fold_h.get("error",""))

    print("    [2/4] Template retrieval …")
    t0 = time.perf_counter()
    tmpl_h = tdb.find_templates(seq_h, pairs_h, k=3)
    ms2    = (time.perf_counter() - t0) * 1000
    LOG.log("step2_ms",      round(ms2,1))
    LOG.log("step2_n_found", len(tmpl_h))
    if tmpl_h:
        LOG.log("step2_top",
                {"tid": tmpl_h[0]["tid"], "score": round(tmpl_h[0]["score"],4),
                 "N_tmpl": tmpl_h[0].get("N"), "size_ratio": round(tmpl_h[0].get("size_ratio",0),3)})

    print("    [3/4] Ion augmentation …")
    t0 = time.perf_counter()
    aug_h  = ion_pred.augment_pairs(seq_h, pairs_h)
    ms3    = (time.perf_counter() - t0) * 1000
    n_new  = len(aug_h) - len(pairs_h)
    pres   = set(pairs_h).issubset(set(aug_h))
    LOG.log("step3_ms",         round(ms3,1))
    LOG.log("step3_orig_pairs", len(pairs_h))
    LOG.log("step3_aug_pairs",  len(aug_h))
    LOG.log("step3_ion_added",  n_new)
    LOG.log("step3_wc_ok",      pres)

    print("    [4/4] Motif annotation …")
    t0 = time.perf_counter()
    mot_h  = ion_pred.annotate_motifs(seq_h, aug_h)
    ms4    = (time.perf_counter() - t0) * 1000
    LOG.log("step4_ms",       round(ms4,1))
    LOG.log("step4_gnra",     mot_h.get("n_gnra"))
    LOG.log("step4_uncg",     mot_h.get("n_uncg"))
    LOG.log("step4_mg2",      mot_h.get("n_ion_sites"))
    LOG.log("step4_aminor",   mot_h.get("n_aminor"))
    LOG.log("total_ms",       round(ms1+ms2+ms3+ms4,1))

    err_h = fold_h.get("error","")
    if not pres:
        LOG.err("WC pairs dropped!")
        LOG.end("FAIL", reason="augment_pairs removed WC pairs")
    elif err_h:
        LOG.warn(f"fold error: {err_h}")
        LOG.end("PARTIAL",
                reason=(f"P3 blocked by {err_h}; "
                        f"P4={len(tmpl_h)} templates, P5={n_new} ion pairs, "
                        f"Mg²⁺={mot_h['n_ion_sites']}"))
    else:
        LOG.end("PASS",
                reason=(f"N={N_h}: {ms1:.0f}ms fold, {ms2:.0f}ms templates, "
                        f"{ms3:.0f}ms ion, {ms4:.0f}ms motifs | "
                        f"{len(tmpl_h)} templates, {n_new} ion pairs, "
                        f"GNRA={mot_h['n_gnra']}, Mg²⁺={mot_h['n_ion_sites']}"))
except Exception:
    LOG.end("ERROR", tb=traceback.format_exc())


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 5 — DIAGNOSIS
# ─────────────────────────────────────────────────────────────────────────────

LOG.section("STAGE 5: DIAGNOSIS — Where the Module is Lacking on Real Data")

fold_recs = [r for r in LOG.records if r["tag"] == "P3-HierarchicalFolder"]
ion_recs  = [r for r in LOG.records if r["tag"] == "P5-IonPredictor"]
tmpl_recs = [r for r in LOG.records if r["tag"] == "P4-TemplateDB"
             and r["tid"] != "P4-BUILD"]
build_rec = next((r for r in LOG.records if r["tid"]=="P4-BUILD"), {})

print("\n  ─── P3: HierarchicalFolder ───────────────────────────────────────────")
for r in fold_recs:
    d  = r["details"]
    st = {"PASS":"✓","PARTIAL":"⚠","FAIL":"✗","ERROR":"✗"}.get(r["status"],"?")
    print(f"  {st} {r['tid']:10s} N={d.get('seq_length_N','?'):5}  "
          f"domains={d.get('n_domains','?'):4}  pairs={d.get('n_base_pairs','?'):5}  "
          f"coaxial={d.get('n_coaxial_stacks','?'):3}  "
          f"gc={d.get('gc_content','?')}  {d.get('fold_time_ms','?'):.0f}ms  "
          f"{r['status']}"
          if isinstance(d.get('fold_time_ms'), float) else
          f"  {st} {r['tid']}: {r.get('reason','')}")

fold_errs = [r for r in fold_recs if "rna_topology" in r.get("reason","")]
if fold_errs:
    print(f"""
  ✗ ROOT CAUSE: {len(fold_errs)} fold test(s) blocked by rna_topology import bug.
    Stage 0 patch did not take effect — module was already cached.
    ACTION: Runtime → Restart all, then re-run this script fresh.
""")

slow = [r for r in fold_recs if r["status"]=="PARTIAL" and "slow" in r.get("reason","").lower()]
if slow:
    print(f"  ⚠  {len(slow)} slow fold(s) detected.")
    for r in slow:
        d = r["details"]
        N  = d.get("seq_length_N","?")
        ml = d.get("max_loop","?")
        ms = d.get("fold_time_ms","?")
        print(f"     {r['tid']}: N={N}, max_loop={ml}, time={ms}ms")
    print("""
    Numba JIT compiles on first call (~5s), then ~100× faster.
    For N=4640, L=300: O(N·L²) = 4640×300² ≈ 4.2×10⁸ ops.
    With Numba: ~2s.  Without Numba: ~200s.
    ACTION: pip install numba  (restart runtime after install)
""")

print("\n  ─── P4: TrainingTemplateDB ───────────────────────────────────────────")
bd = build_rec.get("details", {})
print(f"  Templates built : {bd.get('n_added', bd.get('n_templates_added',0))}  "
      f"(skipped {bd.get('n_skipped',0)})")
if bd.get("n_added",0) == 0 or bd.get("n_templates_added",0) == 0:
    print("""
  ✗ 0 templates — likely ID mismatch between train_sequences.csv
    and train_labels.csv.  Check:
      print(train_sq[['target_id']].head(3))
      print(train_lb[['target_id']].head(3))
    If the columns differ (e.g. 'R1107' vs 'R1107_1'), update seq_map
    to normalise the IDs (strip suffixes).
""")

tmpl_pass = sum(1 for r in tmpl_recs if r["status"]=="PASS")
tmpl_part = sum(1 for r in tmpl_recs if r["status"]=="PARTIAL")
print(f"  Template queries: {tmpl_pass} PASS, {tmpl_part} PARTIAL/FAIL")
for r in tmpl_recs:
    d  = r["details"]
    st = {"PASS":"✓","PARTIAL":"⚠","FAIL":"✗","ERROR":"✗"}.get(r["status"],"?")
    print(f"  {st} {r['tid']:10s}  found={d.get('n_found',0):3}  "
          f"top_score={d.get('top_score','—'):7}  "
          f"top_N={d.get('top_N','—'):5}  {d.get('query_ms','?'):.0f}ms"
          if isinstance(d.get('query_ms'), float) else
          f"  {st} {r['tid']}: {r.get('reason','')}")

if tmpl_part > 0:
    print("""
  ⚠  SIZE MISMATCH: Training structures are short; test targets are long.
     find_templates filters by size_tolerance=0.5 → [0.5N, 1.5N].
     For N=4640 this means templates must be 2320–6960 nt — rare in DB.
     FIX in TrainingTemplateDB.find_templates():
         if not candidates and self._all:
             candidates = sorted(self._all,
                                 key=lambda e: abs(e['N']-N))[:10]
""")

print("\n  ─── P5: IonContactPredictor ──────────────────────────────────────────")
for r in ion_recs:
    d  = r["details"]
    st = {"PASS":"✓","PARTIAL":"⚠","FAIL":"✗","ERROR":"✗"}.get(r["status"],"?")
    print(f"  {st} {r['tid']:10s}  gnra={d.get('n_gnra_tetraloops','?'):3}  "
          f"uncg={d.get('n_uncg_tetraloops','?'):3}  "
          f"mg2={d.get('n_mg2_sites','?'):4}  "
          f"+ion={d.get('n_ion_pairs_added','?'):4}  "
          f"wc_ok={d.get('wc_pairs_preserved','?')}  "
          f"{d.get('motif_time_ms','?'):.0f}ms"
          if isinstance(d.get('motif_time_ms'), float) else
          f"  {st} {r['tid']}: {r.get('reason','')}")

# Pair density analysis
print("\n  ─── Pair density on real sequences ───────────────────────────────────")
print(f"  {'Target':<12}{'N':>6}{'pairs':>7}{'pairs/N':>8}{'domains':>8}{'GC':>6}")
print("  " + "-"*50)
for r in fold_recs:
    d = r["details"]
    N  = d.get("seq_length_N", 0)
    np_= d.get("n_base_pairs", 0)
    nd = d.get("n_domains", 0)
    gc = d.get("gc_content", "?")
    frac = round(np_/N, 3) if N else 0
    flag = " ← HIGH" if frac > 0.5 else ""
    print(f"  {r['tid']:<12}{N:>6}{np_:>7}{frac:>8.3f}{nd:>8}  {gc}{flag}")

print("""
  ─── Known limitations on real competition data ──────────────────────

  1. PSEUDOKNOT HANDLING (long sequences):
     On N>2000 sequences with genus>0 (real pseudoknots), domain
     decomposition finds no clean zero-boundary because crossing pairs
     span all positions.  This produces 1 large domain = no decomposition.
     FIX: Stratify by genus level: decompose level-0 nested structure
          first, then treat level-1 pseudoknot stems as inter-domain bridges.

  2. TEMPLATE SIZE GAP:
     Training structures (median ~50–200 nt) are much shorter than
     test targets (up to 4640 nt).  find_templates size_tolerance=0.5
     means no templates match for large targets.
     FIX: Add topology-only retrieval mode that ignores size and
          scales junction angles by the N ratio.

  3. ION SITE DENSITY ON LARGE SEQUENCES:
     For N=4640, ION_CONTACT_RADIUS=6 positions ≈ 20Å.
     _bridge_pairs() creates pairs within each ion zone independently.
     Pairs bridging two DIFFERENT distant ion sites are NOT generated.
     FIX: Add cross-site bridging: for ion sites sA, sB with |sA-sB|>12,
          generate (i,j) where i∈zone(sA), j∈zone(sB).

  4. COAXIAL STACK OVER-REPORTING:
     COAX_MAX_GAP=2 may be too liberal for real sequences with many stems.
     Real coaxial stacks require stacking geometry (parallel helical axes).
     FIX: Add a Turner coaxial stacking energy filter: only keep
          coaxial pairs where the junction Turner energy < -0.5 kcal/mol.
""")


# ─────────────────────────────────────────────────────────────────────────────
# FINAL SUMMARY
# ─────────────────────────────────────────────────────────────────────────────

summary = LOG.summary()
passed = summary["passed"]
total  = summary["total"]
pct    = 100 * passed // max(total, 1)

print(f"{'='*70}")
print(f"  FINAL SCORE (Real Stanford RNA 3D Folding 2 Data): {passed}/{total} ({pct}%)")
if pct == 100:
    print("  ✓ All tests passed on real data.")
elif pct >= 75:
    print("  ⚠  Most tests passed. See STAGE 5 for remaining fixes.")
else:
    print("  ✗ Many tests failed. See STAGE 5 DIAGNOSIS for root causes.")
print(f"{'='*70}")
