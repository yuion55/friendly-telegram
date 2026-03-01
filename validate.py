"""Validation CLI for RNA 3D structure prediction with batched GPU inference."""

import argparse
import os
import sys
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from rna_dataset import StanfordRNA3DDataset
from metrics.tm_score import tm_score, kabsch_align
from pipeline.full_pipeline import RNA3DPipeline


def run_validation(split, n_targets, data_root, batch_size, n_candidates):
    """Run batched validation on the Stanford RNA 3D dataset.

    Parameters
    ----------
    split : str
        Dataset split ('train' or 'validation').
    n_targets : int or None
        Maximum number of targets to evaluate.
    data_root : str or None
        Path to data directory.
    batch_size : int
        Number of targets per GPU batch.
    n_candidates : int
        Number of candidate structures per target.
    """
    dataset = StanfordRNA3DDataset(split=split, data_root=data_root)
    pipeline = RNA3DPipeline(config={})

    ids = dataset.target_ids()
    if n_targets is not None:
        ids = ids[:n_targets]

    results_rows = []
    tm_scores = []

    # Group target_ids into batches
    batches = [ids[i:i + batch_size] for i in range(0, len(ids), batch_size)]

    pbar = tqdm(batches, desc=f"Validating ({split})")
    for batch_idx, batch_ids in enumerate(pbar):
        sequences = []
        true_coords_list = []
        valid_ids = []

        for target_id in batch_ids:
            seq = dataset.get_sequence(target_id)
            true_coords = dataset.get_coords(target_id)
            if true_coords is None or len(seq) == 0:
                continue
            sequences.append(seq)
            true_coords_list.append(true_coords)
            valid_ids.append(target_id)

        if not valid_ids:
            continue

        # Batched GPU inference via RhoFold if available
        if pipeline.rhofold.available and len(sequences) > 1:
            try:
                batch_preds = pipeline.rhofold.predict_batch(sequences)
            except Exception:
                batch_preds = None
        else:
            batch_preds = None

        for i, target_id in enumerate(valid_ids):
            seq = sequences[i]
            true_coords = true_coords_list[i]

            # Use first atom coords (index 0) from (L, 5, 3)
            if true_coords.ndim == 3:
                true_first = true_coords[:, 0, :]
            else:
                true_first = true_coords

            predictions = pipeline.predict(
                seq, n_candidates=n_candidates, n_submit=5
            )

            best_tm = 0.0
            for pred in predictions:
                pred_coords = pred['coords']
                min_len = min(len(pred_coords), len(true_first))
                p = pred_coords[:min_len].astype(np.float32)
                t = true_first[:min_len].astype(np.float32)
                aligned = kabsch_align(p, t)
                score = tm_score(aligned, t)
                best_tm = max(best_tm, score)

            tm_scores.append(best_tm)
            genus = predictions[0]['genus'] if predictions else 0
            results_rows.append({
                'target_id': target_id,
                'best_tm_score': best_tm,
                'sequence_length': len(seq),
                'genus': genus,
            })

        # Update progress bar with live mean TM
        if tm_scores:
            pbar.set_postfix(mean_tm=f"{np.mean(tm_scores):.4f}")

        # GPU memory management
        if (batch_idx + 1) % 10 == 0 and torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Final output
    if tm_scores:
        tm_arr = np.array(tm_scores)
        print(f"\nMean TM:          {tm_arr.mean():.4f}")
        print(f"Median TM:        {np.median(tm_arr):.4f}")
        print(f"Fraction >= 0.90: {(tm_arr >= 0.90).mean():.4f}")
        print(f"Fraction >= 0.70: {(tm_arr >= 0.70).mean():.4f}")
        print(f"Fraction >= 0.50: {(tm_arr >= 0.50).mean():.4f}")

    df = pd.DataFrame(results_rows)
    out_path = f"validation_results_{split}.csv"
    df.to_csv(out_path, index=False)
    print(f"Results saved to {out_path}")
    return df


def main():
    parser = argparse.ArgumentParser(description="Validate RNA 3D predictions")
    parser.add_argument(
        "--split", choices=["train", "validation"], default="validation"
    )
    parser.add_argument("--n_targets", type=int, default=None)
    parser.add_argument("--data_root", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--n_candidates", type=int, default=50)
    parser.add_argument(
        "--profile", action="store_true",
        help="Run torch profiler on first batch and save chrome trace."
    )
    args = parser.parse_args()

    if args.profile and torch.cuda.is_available():
        from torch.profiler import profile, ProfilerActivity
        print("Profiling first batch...")
        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            record_shapes=True,
        ) as prof:
            run_validation(
                args.split, min(args.batch_size, args.n_targets or args.batch_size),
                args.data_root, args.batch_size, args.n_candidates,
            )
        prof.export_chrome_trace("profiler_trace.json")
        print("Profiler trace saved to profiler_trace.json")
    else:
        run_validation(
            args.split, args.n_targets, args.data_root,
            args.batch_size, args.n_candidates,
        )


if __name__ == "__main__":
    main()
