"""Validation CLI for RNA 3D structure prediction."""

import argparse
import os
import sys
import numpy as np
import pandas as pd

from rna_dataset import StanfordRNA3DDataset
from metrics.tm_score import tm_score, kabsch_align
from pipeline.full_pipeline import RNA3DPipeline


def main():
    parser = argparse.ArgumentParser(description="Validate RNA 3D predictions")
    parser.add_argument("--split", choices=["train", "validation"], default="validation")
    parser.add_argument("--n_targets", type=int, default=None)
    parser.add_argument("--data_root", type=str, default=None)
    args = parser.parse_args()

    dataset = StanfordRNA3DDataset(split=args.split, data_root=args.data_root)
    pipeline = RNA3DPipeline(config={})

    ids = dataset.target_ids()
    if args.n_targets is not None:
        ids = ids[:args.n_targets]

    results_rows = []
    tm_scores = []

    for target_id in ids:
        seq = dataset.get_sequence(target_id)
        true_coords = dataset.get_coords(target_id)

        if true_coords is None or len(seq) == 0:
            continue

        # Use first atom coords (index 0) from (L, 5, 3)
        if true_coords.ndim == 3:
            true_first = true_coords[:, 0, :]
        else:
            true_first = true_coords

        predictions = pipeline.predict(seq, n_candidates=20, n_submit=5)

        best_tm = 0.0
        for pred in predictions:
            pred_coords = pred['coords']
            # Trim to min length
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
        print(f"  {target_id}: TM={best_tm:.4f} L={len(seq)}")

    if tm_scores:
        tm_arr = np.array(tm_scores)
        print(f"\nMean TM: {tm_arr.mean():.4f}")
        print(f"Median TM: {np.median(tm_arr):.4f}")
        print(f"Fraction >= 0.90: {(tm_arr >= 0.90).mean():.4f}")

    df = pd.DataFrame(results_rows)
    out_path = f"validation_results_{args.split}.csv"
    df.to_csv(out_path, index=False)
    print(f"Results saved to {out_path}")


if __name__ == "__main__":
    main()
