#!/usr/bin/env python3
"""
Build FAISS index for MetaCLIP2 embeddings.

This script processes a corpus of images, extracts embeddings using MetaCLIP2,
and builds a FAISS index for efficient similarity search.

Usage:
    python build_faiss_index.py \
        --image_list images.txt \
        --output_dir ./indices/metaclip2 \
        --batch_size 32

Input format (images.txt):
    One image path or URL per line. Can be:
    - Local file paths: /path/to/image.jpg
    - HTTP/HTTPS URLs: https://example.com/image.jpg
    - Pinterest-style signatures: abc123def456... (will be converted to URL)

Output:
    - index.faiss: FAISS index file
    - identifiers.npy: Array of image identifiers (same order as index)
    - metadata.json: Index metadata
"""

import argparse
import json
import os
from datetime import datetime
from pathlib import Path

import faiss
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils.dataset import ImageDataset
from utils.faiss_utils import save_faiss_index
from utils.model_loader import get_embedding_dimension, load_model


def build_faiss_index(
    image_list_path: str,
    output_dir: str,
    batch_size: int = 32,
    num_workers: int = 4,
    checkpoint_interval: int = 1000,
    model_name: str = "ViT-H-14-worldwide-quickgelu",
    pretrained: str = "metaclip2_worldwide"
):
    """
    Build FAISS index for MetaCLIP2 model

    Args:
        image_list_path: Path to file containing image paths/URLs
            (one per line)
        output_dir: Directory to save index and metadata
        batch_size: Batch size for processing
        num_workers: Number of data loading workers
        checkpoint_interval: Save checkpoint every N batches
        model_name: OpenCLIP model name
        pretrained: OpenCLIP pretrained weights name
    """
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}")

    # Load model
    model, preprocess, _, device = load_model(model_name, pretrained)
    embedding_dim = get_embedding_dimension(model, device)
    print(f"Embedding dimension: {embedding_dim}")

    # Load dataset
    print("Loading image dataset...")
    dataset = ImageDataset(image_list_path, preprocess)
    print(f"Dataset size: {len(dataset)} images")

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )

    # Check for existing checkpoint
    checkpoint_path = os.path.join(output_dir, "checkpoint.npz")
    start_batch = 0
    all_embeddings = []
    all_identifiers = []
    processed_identifiers = set()

    if os.path.exists(checkpoint_path):
        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = np.load(checkpoint_path, allow_pickle=True)
        all_embeddings = checkpoint["embeddings"].tolist()
        all_identifiers = checkpoint["identifiers"].tolist()
        start_batch = checkpoint["batch_idx"] + 1
        processed_identifiers = set(all_identifiers)
        print(f"Resuming from batch {start_batch}, "
              f"{len(all_identifiers)} images processed")

    # Process images
    print("Extracting embeddings...")

    with torch.no_grad():
        pbar = tqdm(dataloader, initial=start_batch,
                    desc="Processing batches")
        for batch_idx, batch in enumerate(pbar):
            if batch_idx < start_batch:
                continue

            images, identifiers, success_flags = batch
            valid_indices = [
                i for i, success in enumerate(success_flags) if success
            ]

            if valid_indices:
                valid_images = images[valid_indices].to(device)

                try:
                    with torch.cuda.amp.autocast():
                        image_features = model.encode_image(valid_images)
                        image_features = F.normalize(image_features, dim=-1)

                    image_features = image_features.cpu().numpy()

                    for i, idx in enumerate(valid_indices):
                        identifier = identifiers[idx]
                        if identifier not in processed_identifiers:
                            all_embeddings.append(image_features[i])
                            all_identifiers.append(identifier)
                            processed_identifiers.add(identifier)

                except Exception as e:
                    print(f"Error processing batch {batch_idx}: {e}")
                    continue

            # Save checkpoint periodically
            if (batch_idx + 1) % checkpoint_interval == 0:
                print(f"Saving checkpoint at batch {batch_idx + 1}")
                np.savez(
                    checkpoint_path,
                    embeddings=np.array(all_embeddings),
                    identifiers=np.array(all_identifiers),
                    batch_idx=batch_idx
                )

    # Convert to numpy arrays
    embeddings_array = np.array(all_embeddings).astype("float32")
    identifiers_array = np.array(all_identifiers)

    print(f"Successfully processed {len(embeddings_array)} images")

    # Build FAISS index
    print("Building FAISS index...")
    index = faiss.IndexFlatIP(embedding_dim)
    index.add(embeddings_array)

    # Save index and metadata
    metadata = {
        "model_name": model_name,
        "pretrained": pretrained,
        "embedding_dim": embedding_dim,
        "num_images": len(identifiers_array),
        "creation_date": datetime.now().isoformat(),
        "image_list_file": image_list_path
    }

    save_faiss_index(index, identifiers_array, Path(output_dir), metadata)

    # Clean up checkpoint
    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)
        print("Removed checkpoint file")

    print(f"\n✅ Index building complete!")
    print(f"Total images indexed: {len(identifiers_array)}")
    print(f"Index saved to: {output_dir}")

    return index, identifiers_array


def main():
    parser = argparse.ArgumentParser(
        description="Build FAISS index for MetaCLIP2 model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        "--image_list",
        type=str,
        default="index_signatures.txt",
        help=("Path to file containing image paths/URLs "
              "(one per line). Default: index_signatures.txt")
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./indices/metaclip2",
        help="Output directory for index. Default: ./indices/metaclip2"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for processing (default: 32)"
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of data loading workers (default: 4)"
    )
    parser.add_argument(
        "--checkpoint_interval",
        type=int,
        default=1000,
        help="Save checkpoint every N batches (default: 1000)"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="ViT-H-14-worldwide-quickgelu",
        help="OpenCLIP model name (default: ViT-H-14-worldwide-quickgelu)")
    parser.add_argument(
        "--pretrained",
        type=str,
        default="metaclip2_worldwide",
        help="OpenCLIP pretrained weights (default: metaclip2_worldwide)")

    args = parser.parse_args()

    build_faiss_index(
        image_list_path=args.image_list,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        checkpoint_interval=args.checkpoint_interval,
        model_name=args.model_name,
        pretrained=args.pretrained
    )


if __name__ == "__main__":
    main()

