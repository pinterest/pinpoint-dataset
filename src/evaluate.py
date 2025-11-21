#!/usr/bin/env python3
"""
Evaluate retrieval results against ground truth.

This script computes standard retrieval metrics including:
  - Precision@k, Recall@k, mAP@k (mean Average Precision)
  - Negative recall@k (how many negative examples were retrieved)
  - Metrics with negatives removed
  - Linguistic sensitivity (variance across different text instructions
    for same images)

Usage:
    # Evaluate a single result file
    python evaluate.py \
        --results results.json \
        --ground_truth ground_truth.parquet \
        --output metrics.csv

    # Evaluate all JSON files in a directory
    python evaluate.py \
        --results_dir ./results \
        --ground_truth ground_truth.parquet \
        --output all_metrics.csv

Input format:
    results.json: {
        "00001": {
            "retrieved_items": ["signature1", "signature2", ...]
        },
        ...
    }

    ground_truth.parquet: Must contain columns:
        - query_id: Query identifier (will be normalized to 5-digit format)
        - positive_candidates: List/array of relevant item identifiers
        - negative_candidates: List/array of negative item identifiers
            (optional)
        - query_image_signature: Image identifier (for linguistic sensitivity)
        - query_image_signature2: Optional second image
            (for linguistic sensitivity)
"""

import argparse
import json
import warnings
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

from utils.data_utils import filter_out_negatives, normalize_query_id
from utils.metrics import (calculate_metrics_at_k,
                           calculate_neg_recall_at_k)

warnings.filterwarnings("ignore")


def load_results(file_path: str) -> Dict:
    """Load a standardized JSON results file"""
    with open(file_path, "r") as f:
        return json.load(f)


def evaluate_model(
    model_name: str, results: Dict, gt_df: pd.DataFrame
) -> Dict:
    """
    Evaluate a single model using standardized format.

    Args:
        model_name: Name of the model being evaluated
        results: Dictionary mapping query_id to {"retrieved_items": [...]}
        gt_df: Ground truth DataFrame

    Returns:
        Dictionary of aggregated metrics
    """
    print(f"Evaluating {model_name}...")

    # Initialize metric accumulators
    ks = ["1", "5", "10", "50"]
    metrics = {k: [] for k in ks}  # baseline AP@k
    metrics_no_neg = {k: [] for k in ks}  # AP@k after removing negatives
    neg_recalls = {k: [] for k in ks}
    precisions = {k: [] for k in ks}
    recalls = {k: [] for k in ks}

    # For linguistic sensitivity
    query_variations = {}  # (img1, img2) -> list of precision@10 scores

    matched = 0
    for _, row in gt_df.iterrows():
        gt_query_id = row["query_id"]
        query_id = normalize_query_id(gt_query_id)

        if query_id not in results:
            continue

        matched += 1
        query_data = results[query_id]
        retrieved = query_data.get("retrieved_items", [])
        if not retrieved:
            continue

        # Ground truth lists
        relevant = row["positive_candidates"]
        if hasattr(relevant, "tolist"):
            relevant = relevant.tolist()
        elif not isinstance(relevant, list):
            relevant = []

        negative = row.get("negative_candidates", [])
        if hasattr(negative, "tolist"):
            negative = negative.tolist()
        elif not isinstance(negative, list):
            negative = []

        # Baseline metrics
        for k in [1, 5, 10, 50]:
            k_str = str(k)
            results_k = calculate_metrics_at_k(retrieved, relevant, k)
            metrics[k_str].append(results_k["ap"])
            precisions[k_str].append(results_k["precision"])
            recalls[k_str].append(results_k["recall"])
            neg_recalls[k_str].append(
                calculate_neg_recall_at_k(retrieved, negative, k))

        # "No negatives" variant: remove negatives from retrieved,
        # preserve order
        retrieved_no_neg = filter_out_negatives(retrieved, negative)
        for k in [1, 5, 10, 50]:
            k_str = str(k)
            results_k_no_neg = calculate_metrics_at_k(
                retrieved_no_neg, relevant, k)
            metrics_no_neg[k_str].append(results_k_no_neg["ap"])

        # Linguistic sensitivity (precision@10 baseline)
        img1 = row.get("query_image_signature")
        img2 = row.get("query_image_signature2", None)
        if img1:
            key = (img1, img2) if img2 else (img1, None)
            if key not in query_variations:
                query_variations[key] = []
            results_10 = calculate_metrics_at_k(retrieved, relevant, 10)
            query_variations[key].append(results_10["precision"])

    print(f"  Matched {matched}/{len(gt_df)} queries")
    if matched == 0:
        print("  ⚠️  No queries matched")
        return None

    # Linguistic sensitivity aggregates
    ling_sens_ranges, ling_sens_stds = [], []
    for _, scores in query_variations.items():
        if len(scores) > 1:
            ling_sens_ranges.append(max(scores) - min(scores))
            ling_sens_stds.append(np.std(scores))

    # Compile final metrics
    result = {
        "model": model_name,
        "total_queries": matched
    }
    for k in ks:
        result[f"mAP@{k}"] = np.mean(metrics[k]) if metrics[k] else 0.0
        result[f"mAP@{k}_noNeg"] = (
            np.mean(metrics_no_neg[k]) if metrics_no_neg[k] else 0.0)
        result[f"precision@{k}"] = (
            np.mean(precisions[k]) if precisions[k] else 0.0)
        result[f"recall@{k}"] = (
            np.mean(recalls[k]) if recalls[k] else 0.0)
        result[f"NegRecall@{k}"] = (
            np.mean(neg_recalls[k]) if neg_recalls[k] else 0.0)

    # Impact summaries focused on @10
    result["delta_mAP@10_noNeg"] = (
        result["mAP@10_noNeg"] - result["mAP@10"])
    result["pct_increase_mAP@10_noNeg"] = (
        (result["delta_mAP@10_noNeg"] / result["mAP@10"])
        if result["mAP@10"] > 0 else 0.0
    )

    # Linguistic sensitivity
    result["ling_sens_range"] = (
        np.mean(ling_sens_ranges) if ling_sens_ranges else 0.0)
    result["ling_sens_std"] = (
        np.mean(ling_sens_stds) if ling_sens_stds else 0.0)

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate retrieval results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        "--results",
        type=str,
        default=None,
        help="Path to single results JSON file"
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default=None,
        help="Directory containing multiple results JSON files"
    )
    parser.add_argument(
        "--ground_truth",
        type=str,
        default="pinpoint_metadata.parquet",
        help=("Path to ground truth parquet file. "
              "Default: pinpoint_metadata.parquet")
    )
    parser.add_argument(
        "--output",
        type=str,
        default="metrics.csv",
        help="Output CSV file for metrics. Default: metrics.csv"
    )

    args = parser.parse_args()

    if args.results is None and args.results_dir is None:
        parser.error("Must provide either --results or --results_dir")

    if args.results is not None and args.results_dir is not None:
        parser.error("Cannot provide both --results and --results_dir")

    # Load ground truth
    print("Loading ground truth...")
    gt_df = pd.read_parquet(args.ground_truth)
    print(f"Loaded {len(gt_df)} queries")

    # Find result files
    if args.results:
        result_files = [Path(args.results)]
    else:
        results_dir = Path(args.results_dir)
        result_files = sorted(results_dir.glob("*.json"))

    if not result_files:
        print("❌ No result files found!")
        return

    print(f"Found {len(result_files)} result file(s)")

    # Evaluate each model
    all_results = []
    for file_path in result_files:
        model_name = file_path.stem
        print(f"\nProcessing {model_name}...")

        try:
            results = load_results(str(file_path))
            result = evaluate_model(model_name, results, gt_df)
            if result:
                all_results.append(result)
        except Exception as e:
            print(f"  ❌ Error evaluating {model_name}: {e}")

    if not all_results:
        print("\n❌ No models could be evaluated")
        return

    # Create DataFrame and save
    df = pd.DataFrame(all_results)

    # Reorder columns
    column_order = [
        "model",
        "precision@1",
        "precision@10",
        "NegRecall@10",
        "mAP@10",
        "mAP@10_noNeg",
        "delta_mAP@10_noNeg",
        "pct_increase_mAP@10_noNeg",
        "ling_sens_range",
        "total_queries",
        "mAP@1",
        "mAP@5",
        "mAP@50",
        "mAP@1_noNeg",
        "mAP@5_noNeg",
        "mAP@50_noNeg",
        "NegRecall@1",
        "NegRecall@5",
        "NegRecall@50",
        "precision@5",
        "precision@50",
        "recall@1",
        "recall@5",
        "recall@10",
        "recall@50",
        "ling_sens_std"
    ]

    # Ensure all columns exist
    column_order = [col for col in column_order if col in df.columns]
    df = df[column_order]

    # Sort by mAP@10
    df = df.sort_values("mAP@10", ascending=False)

    # Save to CSV
    df.to_csv(args.output, index=False, float_format="%.4f")
    print(f"\n✅ Saved metrics to {args.output}")


if __name__ == "__main__":
    main()

