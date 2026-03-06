#!/usr/bin/env python3
"""
Run multimodal retrieval using MetaCLIP2 model.

This script performs retrieval on a query corpus using a pre-built FAISS index.
Supports three modes:
  - combined: Weighted average of image and text embeddings
  - image_only: Image-only retrieval
  - text_only: Text-only retrieval

The output is a standardized JSON format compatible with the evaluation script.

Usage:
    python run_retrieval.py \
        --query_file queries.parquet \
        --index_dir ./indices/metaclip2 \
        --output_file results.json \
        --mode combined \
        --top_k 50

Input format (queries.parquet):
    Must contain columns:
    - query_id: Unique identifier for each query
    - query_image_signature: Image identifier/path
        (required for combined/image_only)
    - query_image_signature2: Optional second image (for multi-image queries)
    - instruction: Text instruction (required for combined/text_only)

Output format (results.json):
    {
        "00001": {
            "retrieved_items": ["signature1", "signature2", ...]
        },
        ...
    }
"""

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from utils.data_utils import normalize_query_id
from utils.embeddings import get_query_embedding
from utils.faiss_utils import load_faiss_index
from utils.model_loader import load_model

warnings.filterwarnings("ignore")


def run_retrieval(
    query_file_path: str,
    index_dir: str,
    output_file: str,
    top_k: int = 50,
    mode: str = "combined",
    alpha: float = 0.8,
    batch_size: int = 1,
    checkpoint_interval: int = 100,
    resume: bool = True,
    model_name: str = "ViT-H-14-worldwide-quickgelu",
    pretrained: str = "metaclip2_worldwide"
):
    """Run MetaCLIP2 retrieval on query corpus"""
    if mode not in ["combined", "image_only", "text_only"]:
        raise ValueError(
            f"Invalid mode: {mode}. "
            f"Must be 'combined', 'image_only', or 'text_only'")

    print(f"Running retrieval in {mode} mode")
    if mode == "combined":
        text_pct = alpha * 100
        img_pct = (1 - alpha) * 100
        print(f"Using alpha={alpha} ({text_pct:.0f}% text, "
              f"{img_pct:.0f}% image)")

    # Load model
    model, preprocess, tokenizer, device = load_model(model_name, pretrained)

    # Load FAISS index
    index_dir = Path(index_dir)
    index, identifiers, metadata = load_faiss_index(index_dir)

    # Load queries
    print(f"Loading query corpus from {query_file_path}...")
    query_df = pd.read_parquet(query_file_path)
    print(f"Loaded {len(query_df)} queries")

    # Filter queries based on mode
    if mode == "image_only":
        if "query_image_signature" not in query_df.columns:
            raise ValueError(
                "No 'query_image_signature' column found for image_only mode")
        query_df = query_df[query_df["query_image_signature"].notna()]
        print(f"Filtered to {len(query_df)} queries with image signatures")
    elif mode == "text_only":
        if "instruction" not in query_df.columns:
            raise ValueError(
                "No 'instruction' column found for text_only mode")
        query_df = query_df[query_df["instruction"].notna()]
        print(f"Filtered to {len(query_df)} queries with instructions")
    elif mode == "combined":
        required_cols = ["query_image_signature", "instruction"]
        missing = [col for col in required_cols if col not in query_df.columns]
        if missing:
            raise ValueError(
                f"Missing required columns for combined mode: {missing}")
        query_df = query_df[
            query_df["query_image_signature"].notna() &
            query_df["instruction"].notna()
        ]
        print(f"Filtered to {len(query_df)} queries with both image and text")

    # Check for checkpoint
    output_path = Path(output_file)
    checkpoint_path = output_path.with_suffix(".checkpoint.json")

    if resume and checkpoint_path.exists():
        print(f"Resuming from checkpoint: {checkpoint_path}")
        with open(checkpoint_path, "r") as f:
            checkpoint_data = json.load(f)
        all_results = checkpoint_data["results"]
        start_idx = checkpoint_data["last_processed_idx"] + 1
        print(f"Resuming from query {start_idx}")
    else:
        all_results = {}
        start_idx = 0

    # Process queries
    print(f"Processing {len(query_df) - start_idx} queries...")

    total_queries = len(query_df) - start_idx
    with tqdm(total=total_queries, desc="Processing queries") as pbar:
        for i in range(start_idx, len(query_df), batch_size):
            end_idx = min(i + batch_size, len(query_df))
            batch_df = query_df.iloc[i:end_idx]

            for idx, row in batch_df.iterrows():
                query_id = row.get("query_id", f"query_{idx+1:05d}")
                normalized_id = normalize_query_id(query_id)

                query_data = {
                    "query_id": query_id,
                    "query_image_signature": row.get(
                        "query_image_signature"),
                    "query_image_signature2": row.get(
                        "query_image_signature2"),
                    "instruction": row.get("instruction", "")
                }

                # Get query embedding
                query_embedding = get_query_embedding(
                    query_data=query_data,
                    model=model,
                    preprocess=preprocess,
                    tokenizer=tokenizer,
                    device=device,
                    mode=mode,
                    alpha=alpha
                )

                if query_embedding is None:
                    all_results[normalized_id] = {
                        "retrieved_items": []
                    }
                    continue

                # Search in FAISS index
                if len(query_embedding.shape) == 1:
                    query_embedding = query_embedding.reshape(1, -1)

                _, indices = index.search(query_embedding, top_k)
                retrieved_identifiers = identifiers[indices[0]].tolist()

                # Store in standardized format
                all_results[normalized_id] = {
                    "retrieved_items": retrieved_identifiers
                }

            pbar.update(len(batch_df))

            # Save checkpoint
            if ((i + batch_size) % checkpoint_interval == 0 or
                    i + batch_size >= len(query_df)):
                checkpoint_data = {
                    "last_processed_idx": min(i + batch_size - 1,
                                               len(query_df) - 1),
                    "results": all_results
                }
                with open(checkpoint_path, "w") as f:
                    json.dump(checkpoint_data, f, indent=2)

    # Save final results in standardized format
    print(f"Saving results to {output_file}...")
    with open(output_file, "w") as f:
        json.dump(all_results, f, indent=2)

    # Remove checkpoint file
    if checkpoint_path.exists():
        checkpoint_path.unlink()

    successful = sum(1 for r in all_results.values()
                     if r.get("retrieved_items"))
    print(f"\n✅ Retrieval complete!")
    print(f"Total queries: {len(all_results)}")
    print(f"Successful: {successful}")
    print(f"Results saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Run MetaCLIP2 retrieval",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        "--query_file",
        type=str,
        default="pinpoint_licensed.parquet",
        help=("Path to parquet file containing query corpus. "
              "Default: pinpoint_licensed.parquet"))
    parser.add_argument(
        "--index_dir",
        type=str,
        default="./indices/metaclip2",
        help="Directory containing FAISS index. Default: ./indices/metaclip2")
    parser.add_argument(
        "--output_file",
        type=str,
        default="results.json",
        help="Output file for retrieval results (JSON). Default: results.json")
    parser.add_argument(
        "--top_k",
        type=int,
        default=50,
        help="Number of top results to retrieve per query (default: 50)")
    parser.add_argument(
        "--mode",
        type=str,
        default="combined",
        choices=["combined", "image_only", "text_only"],
        help="Retrieval mode (default: combined)")
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.8,
        help="Weight for text in combined mode (default: 0.8)")
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Batch size for processing queries (default: 1)")
    parser.add_argument(
        "--checkpoint_interval",
        type=int,
        default=100,
        help="Save checkpoint after this many queries (default: 100)")
    parser.add_argument(
        "--no_resume",
        action="store_true",
        help="Do not resume from checkpoint")
    parser.add_argument(
        "--model_name",
        type=str,
        default="ViT-H-14-worldwide-quickgelu",
        help="OpenCLIP model name")
    parser.add_argument(
        "--pretrained",
        type=str,
        default="metaclip2_worldwide",
        help="OpenCLIP pretrained weights")

    args = parser.parse_args()

    run_retrieval(
        query_file_path=args.query_file,
        index_dir=args.index_dir,
        output_file=args.output_file,
        top_k=args.top_k,
        mode=args.mode,
        alpha=args.alpha,
        batch_size=args.batch_size,
        checkpoint_interval=args.checkpoint_interval,
        resume=not args.no_resume,
        model_name=args.model_name,
        pretrained=args.pretrained
    )


if __name__ == "__main__":
    main()

