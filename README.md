# PinPoint Dataset

Code + dataset for the CVPR 2026 paper "PinPoint: Evaluation of Composed Image Retrieval with Explicit Negatives, Multi-Image Queries, and Paraphrase Testing"

![image](image.png)

## Note on data ownership and licensing

The dataset is released under CC BY 4.0 [see DATA_LICENSE.TXT]. Note that although we verified that the images within the dataset were listed as having a CC BY 2.0 license, we make no representations or warranties regarding the license status of each image. You should verify your ability to use each image for yourself.

We re-host the images on the Pinterest CDNs to avoid missing links due to deletion/URL failures. All the attribution, ownership details and original license of the images in this dataset can be found in `image_attribution.json`

## Pinpoint Dataset Retrieval Framework

This repository provides a framework for evaluating retrieval methods on the Pinpoint dataset. The code includes a complete implementation using MetaCLIP2 as an example, but the framework is designed to be extensible to any retrieval method.

### Project Structure

```
pinpoint-dataset/
├── src/                    # Main source code
│   ├── build_faiss_index.py    # Build FAISS index from image corpus
│   ├── run_retrieval.py         # Run retrieval on queries
│   ├── evaluate.py              # Evaluate retrieval results
│   └── utils/                   # Utility modules
│       ├── image_loader.py       # Image loading/downloading
│       ├── model_loader.py       # Model loading utilities
│       ├── faiss_utils.py        # FAISS index operations
│       ├── metrics.py            # Evaluation metrics
│       ├── data_utils.py         # Data processing utilities
│       ├── dataset.py            # Dataset classes
│       └── embeddings.py         # Embedding generation
├── index_signatures.txt         # Image signatures for index
├── pinpoint_metadata.parquet    # Query corpus and ground truth
├── requirements.txt             # Python dependencies
├── setup.sh                     # Automated setup script
└── README.md                    # This file
```

## Overview

This repository provides a complete framework for building retrieval systems and evaluating them on the Pinpoint dataset. The code includes:

1. **`src/build_faiss_index.py`**: Builds a FAISS index from a corpus of images (MetaCLIP2 example)
2. **`src/run_retrieval.py`**: Performs retrieval on a query corpus using the FAISS index (MetaCLIP2 example)
3. **`src/evaluate.py`**: Evaluates retrieval results against ground truth (works with any method)

All utility functions are organized in the `src/utils/` directory for better code organization and reusability.

**Note**: The MetaCLIP2 implementation is provided as an example. The framework is designed to be extensible - you can replace the embedding model and retrieval logic to test any other retrieval method while using the same evaluation pipeline.

## Installation

### Quick Install

For basic installation, you can use the provided `requirements.txt`:

```bash
pip install -r requirements.txt
```

**Note**: Some packages require special installation:

1. **PyTorch**: Install based on your system (CUDA vs CPU)
   ```bash
   # For CUDA 11.8 (GPU support)
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
   
   # For CPU only
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
   ```

2. **OpenCLIP**: Install from GitHub main branch for MetaCLIP2 support
   ```bash
   pip install --upgrade "open-clip-torch @ git+https://github.com/mlfoundations/open_clip.git@main"
   ```

3. **FAISS**: Choose based on your system
   ```bash
   # For GPU support
   pip install faiss-gpu
   
   # For CPU only
   pip install faiss-cpu
   ```

### Automated Setup Script

For a complete automated setup, you can use the provided setup script:

```bash
bash setup.sh
```

This script will:
- Check Python and pip availability
- Create a virtual environment (optional)
- Install PyTorch (GPU or CPU based on system)
- Install OpenCLIP from GitHub main branch
- Install FAISS (GPU or CPU based on system)
- Install all remaining requirements
- Verify all installations

**Note**: The setup script works on Linux/macOS. For Windows, use the manual installation steps above.

### Manual Installation

If you prefer manual installation:

```bash
# Core dependencies
pip install torch torchvision  # See note above for CUDA/CPU
pip install --upgrade "open-clip-torch @ git+https://github.com/mlfoundations/open_clip.git@main"
pip install faiss-cpu  # or faiss-gpu for GPU support

# Additional packages
pip install numpy pandas pyarrow pillow requests tqdm
```

## Files

This repository contains:

**Code:**
- `src/build_faiss_index.py`: Script to build FAISS index from image corpus
- `src/run_retrieval.py`: Script to perform retrieval on queries
- `src/evaluate.py`: Script to evaluate retrieval results

**Dataset Files:**
- `index_signatures.txt`: Image signatures for building the FAISS index
- `pinpoint_metadata.parquet`: Query corpus and ground truth data

**Configuration:**
- `requirements.txt`: Python package dependencies
- `setup.sh`: Automated setup script
- `LICENSE`: Apache 2.0 license

## Dataset

This repository includes the Pinpoint dataset files:

- **`index_signatures.txt`**: Contains image signatures (one per line) for building the FAISS index. These are Pinterest-style hex signatures that will be automatically converted to image URLs.
- **`pinpoint_metadata.parquet`**: Contains the query corpus and ground truth data with the following columns:
  - `query_id`: Unique identifier for each query
  - `query_image_signature`: Image identifier (Pinterest signature)
  - `query_image_signature2`: Optional second image (for multi-image queries)
  - `instruction`: Text instruction for the query
  - `positive_candidates`: List of relevant item identifiers
  - `negative_candidates`: List of negative item identifiers
  - Additional metadata columns (token_count, length_category, query_category, etc.)

## Quick Start

With the provided dataset files, you can run the complete pipeline with minimal arguments:

```bash
# Step 1: Build FAISS index (uses index_signatures.txt by default)
python src/build_faiss_index.py --output_dir ./indices/metaclip2

# Step 2: Run retrieval (uses pinpoint_metadata.parquet by default)
python src/run_retrieval.py --index_dir ./indices/metaclip2 --output_file results.json

# Step 3: Evaluate results (uses pinpoint_metadata.parquet as ground truth by default)
python src/evaluate.py --results results.json --ground_truth pinpoint_metadata.parquet --output metrics.csv
```

## Usage

### Step 1: Build FAISS Index

Build the index using the provided `index_signatures.txt` file:

```bash
python src/build_faiss_index.py \
    --image_list index_signatures.txt \
    --output_dir ./indices/metaclip2 \
    --batch_size 32 \
    --num_workers 4
```

**Note**: You can also use your own image list file. The file should contain one image path/URL/signature per line. Supported formats:
- Local file paths: `/path/to/image.jpg`
- HTTP/HTTPS URLs: `https://example.com/image.jpg`
- Pinterest-style signatures: 32+ character hex strings (automatically converted to Pinterest CDN URLs)

This will create:
- `index.faiss`: FAISS index file
- `identifiers.npy`: Array of image identifiers
- `metadata.json`: Index metadata

### Step 2: Run Retrieval

Run retrieval using the provided `pinpoint_metadata.parquet` file:

```bash
python src/run_retrieval.py \
    --query_file pinpoint_metadata.parquet \
    --index_dir ./indices/metaclip2 \
    --output_file results.json \
    --mode combined \
    --top_k 50 \
    --alpha 0.8
```

**Note**: You can also use your own query file. The Parquet file must contain columns:
- `query_id`: Unique identifier for each query
- `query_image_signature`: Image identifier/path (required for `combined`/`image_only` modes)
- `query_image_signature2`: Optional second image (for multi-image queries)
- `instruction`: Text instruction (required for `combined`/`text_only` modes)

**Modes:**
- `combined`: Weighted average of image and text embeddings (default: 80% text, 20% image)
- `image_only`: Image-only retrieval
- `text_only`: Text-only retrieval

The output is a standardized JSON format:
```json
{
    "00001": {
        "retrieved_items": ["signature1", "signature2", ...]
    },
    ...
}
```

### Step 3: Evaluate Results

Evaluate results using the provided `pinpoint_metadata.parquet` as ground truth:

```bash
python src/evaluate.py \
    --results results.json \
    --ground_truth pinpoint_metadata.parquet \
    --output metrics.csv
```

Or evaluate all JSON files in a directory:

```bash
python src/evaluate.py \
    --results_dir ./results \
    --ground_truth pinpoint_metadata.parquet \
    --output all_metrics.csv
```

**Note**: The ground truth Parquet file must contain columns:
- `query_id`: Query identifier (will be normalized to 5-digit format)
- `positive_candidates`: List/array of relevant item identifiers
- `negative_candidates`: List/array of negative item identifiers (optional)
- `query_image_signature`: Image identifier (for linguistic sensitivity)
- `query_image_signature2`: Optional second image (for linguistic sensitivity)

## Output Metrics

The evaluation script computes the following metrics:

- **Precision@k**: Fraction of top-k results that are relevant
- **Recall@k**: Fraction of relevant items retrieved in top-k
- **mAP@k**: Mean Average Precision at k
- **NegRecall@k**: Fraction of negative examples retrieved in top-k
- **mAP@k_noNeg**: mAP@k after removing negatives from retrieved results
- **delta_mAP@10_noNeg**: Improvement in mAP@10 after removing negatives
- **ling_sens_range**: Linguistic sensitivity (range of precision@10 across different text instructions for same images)
- **ling_sens_std**: Linguistic sensitivity (standard deviation)

## Model Configuration

The provided MetaCLIP2 example uses:
- **Model**: `ViT-H-14-worldwide-quickgelu`
- **Pretrained**: `metaclip2_worldwide`
- **Embedding dimension**: 1024
- **Image resolution**: 224x224

You can override these with command-line arguments (see `--help` for each script).

## Extending to Other Retrieval Methods

This framework is designed to be extensible. To test a different retrieval method:

1. **Replace the embedding model**: Modify `src/utils/model_loader.py` to load your model instead of MetaCLIP2
2. **Update embedding generation**: Modify `src/utils/embeddings.py` to generate embeddings with your model
3. **Keep the evaluation pipeline**: The `src/evaluate.py` script works with any retrieval method that produces results in the standardized JSON format

The evaluation metrics and ground truth format remain the same regardless of the retrieval method used, making it easy to compare different approaches on the Pinpoint dataset.

## Image Loading

The scripts support multiple image input formats:

1. **Local file paths**: `/path/to/image.jpg`
2. **HTTP/HTTPS URLs**: `https://example.com/image.jpg`
3. **Pinterest-style signatures**: 32+ character hex strings (automatically converted to Pinterest CDN URLs)

## Checkpointing

Both `src/build_faiss_index.py` and `src/run_retrieval.py` support checkpointing to resume interrupted runs:

- Index building: Checkpoints are saved every N batches (default: 1000)
- Retrieval: Checkpoints are saved every N queries (default: 100)

To disable checkpoint resumption, use `--no_resume` flag in `src/run_retrieval.py`.

## Notes

- The scripts use L2-normalized embeddings and inner product (cosine similarity) for retrieval
- Multi-image queries are handled by averaging the image embeddings
- Combined mode uses weighted averaging: `alpha * text_emb + (1 - alpha) * image_emb`
- All embeddings are normalized before and after averaging
