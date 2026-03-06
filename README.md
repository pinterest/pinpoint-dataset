# PinPoint Dataset

Code + dataset for the CVPR 2026 paper ["PinPoint: Evaluation of Composed Image Retrieval with Explicit Negatives, Multi-Image Queries, and Paraphrase Testing"](https://arxiv.org/abs/2603.04598)

![image](image.png)

## Dataset Statistics

| Statistic | Value |
|-----------|-------|
| Total Queries | 7,635 |
| Corpus Images | 109,599 |
| Relevance Judgments | 329K |
| Query Categories | 23 |
| Avg. Positive Answers per Query | 9.1 |
| Instruction Paraphrases per Query | 6 |
| Multi-Image Queries | 13.4% |

**What makes PinPoint unique:**
- **Multiple Correct Answers**: Unlike single-answer benchmarks, each query has ~9 relevant results on average
- **Explicit Hard Negatives**: Each query includes challenging negatives that models commonly confuse with positives
- **Paraphrase Robustness**: 6 instruction variants per query to measure linguistic sensitivity (models show up to 25% variation!)
- **Multi-Image Queries**: 13.4% of queries use two reference images for complex composition
- **Demographic Metadata**: Supports fairness evaluation across demographic groups

## Quick Start: Load and Explore the Data

```python
import pandas as pd

# Load the dataset
df = pd.read_parquet("pinpoint_licensed.parquet")

# View a sample query
sample = df.iloc[0]
print(f"Query ID: {sample['query_id']}")
print(f"Instruction: {sample['instruction']}")
print(f"Query Image: {sample['query_image_signature']}")
print(f"Positive candidates: {sample['positive_candidates'][:3]}...")  # First 3
print(f"Negative candidates: {sample['negative_candidates'][:3]}...")  # First 3
```

## Viewing Images

Images are hosted on Pinterest CDN. To view any image signature:

```python
def signature_to_url(signature):
    """Convert a Pinterest signature to a viewable image URL."""
    return f"https://i.pinimg.com/736x/{signature[:2]}/{signature[2:4]}/{signature[4:6]}/{signature}.jpg"

# Example
sig = "afd9ded10f1efd368cd8294da0bb34ce"
url = signature_to_url(sig)
# https://i.pinimg.com/736x/af/d9/de/afd9ded10f1efd368cd8294da0bb34ce.jpg
```

---

## Evaluate Your Model

### Step 1: Generate Results in the Required Format

Your results file must be a JSON file with this structure:

```json
{
    "00001": {
        "retrieved_items": ["signature1", "signature2", "signature3", ...]
    },
    "00002": {
        "retrieved_items": ["signature_a", "signature_b", "signature_c", ...]
    }
}
```

**Format requirements:**
- Keys are query IDs (5-digit strings: `"00001"`, `"00042"`, etc.)
- `retrieved_items`: ranked list of image signatures (best match first)
- Image signatures must match those in `index_signatures.txt`
- Include up to 50 items per query for full metric calculation

See `standardized_results/` for example files.

### Step 2: Run Evaluation

```bash
# Install minimal dependencies
pip install -r requirements-eval.txt

# Evaluate your results
python evaluate.py --results your_results.json --output your_metrics.csv
```

Or compare multiple methods at once:

```bash
python evaluate.py --results_dir ./my_results/ --output comparison.csv
```

---

## Baseline Results

Results on PinPoint benchmark (sorted by mAP@10):

| Model | Precision@1 | Precision@10 | mAP@10 | NegRecall@10 | mAP@10 (no neg) |
|-------|-------------|--------------|--------|--------------|-----------------|
| GPT-5 Text (reranked) | 0.298 | 0.203 | 0.184 | 0.061 | 0.189 |
| GPT-5 Text (premerge) | 0.288 | 0.197 | 0.179 | 0.089 | 0.190 |
| BGE-VL MLLM S1 (reranked) | 0.296 | 0.176 | 0.170 | 0.057 | 0.174 |
| GPT-5 Text (postmerge) | 0.264 | 0.178 | 0.158 | 0.093 | 0.168 |
| BGE-VL MLLM S1 | 0.233 | 0.142 | 0.131 | 0.087 | 0.141 |
| BGE-VL MLLM S2 | 0.193 | 0.141 | 0.121 | 0.122 | 0.141 |
| BGE-VL CLIP Large | 0.184 | 0.127 | 0.110 | 0.101 | 0.120 |
| MetaCLIP2 (combined) | 0.092 | 0.102 | 0.076 | 0.141 | 0.103 |
| MetaCLIP2 (text only) | 0.112 | 0.076 | 0.064 | 0.066 | 0.068 |
| MetaCLIP2 (image only) | 0.009 | 0.052 | 0.033 | 0.219 | 0.058 |

**Key insight:** NegRecall@10 measures how often models retrieve explicit negatives. Higher mAP@10 (no neg) vs mAP@10 indicates sensitivity to hard negatives.

---

## Output Metrics

The evaluation script computes:

| Metric | Description |
|--------|-------------|
| Precision@k | Fraction of top-k results that are relevant |
| Recall@k | Fraction of relevant items found in top-k |
| mAP@k | Mean Average Precision at k |
| NegRecall@k | Fraction of hard negatives retrieved in top-k |
| mAP@k_noNeg | mAP@k after removing negatives from results |
| delta_mAP@k_noNeg | Improvement when negatives removed |
| ling_sens_range | Linguistic sensitivity (precision range across paraphrases) |
| ling_sens_std | Linguistic sensitivity (standard deviation) |

---

## Dataset Files

| File | Description |
|------|-------------|
| `pinpoint_licensed.parquet` | Query corpus with ground truth (7,635 queries) |
| `index_signatures.txt` | Corpus image signatures (109,599 images) |
| `image_attribution.json` | Image attribution and licensing info |
| `standardized_results/` | Example result files from baseline methods |
| `requirements-eval.txt` | Minimal dependencies for evaluation only |
| `requirements.txt` | Full dependencies for MetaCLIP2 pipeline |

### Data Schema (`pinpoint_licensed.parquet`)

| Column | Type | Description |
|--------|------|-------------|
| `query_id` | string | Unique query identifier |
| `query_image_signature` | string | Reference image signature |
| `query_image_signature2` | string | Optional second reference image |
| `instruction` | string | Text instruction for the query |
| `positive_candidates` | list | Ground truth relevant images |
| `negative_candidates` | list | Hard negative images |

---

## Full Pipeline (MetaCLIP2 Example)

If you want to run the complete retrieval pipeline using the provided MetaCLIP2 implementation:

### Installation

```bash
# Option 1: Automated setup
bash setup.sh

# Option 2: Manual installation
pip install torch torchvision  # Add --index-url for CUDA/CPU specific
pip install "open-clip-torch @ git+https://github.com/mlfoundations/open_clip.git@main"
pip install faiss-cpu  # or faiss-gpu
pip install numpy pandas pyarrow pillow requests tqdm
```

### Run Pipeline

```bash
# Step 1: Build FAISS index (~109k images)
python build_faiss_index.py --output_dir ./indices/metaclip2

# Step 2: Run retrieval
python run_retrieval.py \
    --index_dir ./indices/metaclip2 \
    --output_file results.json \
    --mode combined \
    --alpha 0.8

# Step 3: Evaluate
python evaluate.py --results results.json --output metrics.csv
```

**Retrieval modes:**
- `combined`: Weighted average of image + text embeddings (default: 80% text, 20% image)
- `text_only`: Text embedding only
- `image_only`: Image embedding only

### Checkpointing

Both scripts support checkpointing for interrupted runs:
- Index building: Saves every 1000 batches
- Retrieval: Saves every 100 queries

Use `--no_resume` to start fresh.

---

## Extending to Other Models

**Option A (Recommended):** Use your own retrieval pipeline, output results in the JSON format above, and run `evaluate.py`.

**Option B:** Modify the provided code:
- `utils/model_loader.py` - Replace embedding model
- `utils/embeddings.py` - Update embedding generation

---

## Project Structure

```
pinpoint-dataset/
├── evaluate.py              # Evaluation script (most users need only this)
├── build_faiss_index.py     # Build FAISS index (MetaCLIP2 example)
├── run_retrieval.py         # Run retrieval (MetaCLIP2 example)
├── utils/                   # Utility modules
├── pinpoint_licensed.parquet
├── index_signatures.txt
├── standardized_results/
├── requirements.txt         # Full dependencies (MetaCLIP2 pipeline)
├── requirements-eval.txt    # Minimal dependencies (evaluation only)
└── setup.sh
```

---

## License

- **Code**: Apache 2.0 (see CODE_LICENSE.TXT)
- **Data**: CC BY 4.0 (see DATA_LICENSE.TXT)

Note: Images are re-hosted on Pinterest CDN. Individual image licenses are documented in `image_attribution.json`.

The dataset is released under CC BY 4.0 [see DATA_LICENSE.TXT]. Note that although we verified that the images within the dataset were listed as having a CC BY 2.0 license, we make no representations or warranties regarding the license status of each image. You should verify your ability to use each image for yourself.

---

## Citation

```bibtex
@misc{mahadev2026pinpointevaluationcomposedimage,
  title={PinPoint: Evaluation of Composed Image Retrieval with Explicit Negatives, Multi-Image Queries, and Paraphrase Testing},
  author={Rohan Mahadev and Joyce Yuan and Patrick Poirson and David Xue and Hao-Yu Wu and Dmitry Kislyuk},
  year={2026},
  eprint={2603.04598},
  archivePrefix={arXiv},
  primaryClass={cs.CV},
  url={https://arxiv.org/abs/2603.04598},
}
```

---

## Contact

For questions or issues, please open a GitHub issue.
