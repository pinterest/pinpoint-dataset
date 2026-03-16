#!/usr/bin/env python3
"""
Rerank retrieval results using a VLM (Vision-Language Model).

This script implements the logit-based reranking method described in the
PinPoint paper. It takes initial retrieval results, scores each candidate
using a VLM's True/False logit probabilities, and produces reranked results.

The VLM is asked: "Is this candidate image a precise match for the reference
image given the modification instruction?" The logit probabilities of the
"True" and "False" tokens are extracted and converted to a relevance score:

    score = P(True) / (P(True) + P(False))

Candidates are then sorted by this score in descending order.

By default, this script calls an OpenAI-compatible API (works with vLLM,
Ollama, SGLang, etc.). To use a different VLM backend, subclass VLMClient
and override the `score_candidate` method.

Usage:
    python rerank.py \
        --results results.json \
        --output reranked_results.json \
        --api_base http://localhost:8000/v1 \
        --model Qwen/Qwen2.5-VL-7B-Instruct

    # Rerank only top-N candidates per query (faster)
    python rerank.py \
        --results results.json \
        --output reranked_results.json \
        --api_base http://localhost:8000/v1 \
        --model Qwen/Qwen2.5-VL-7B-Instruct \
        --top_n 20
"""

import argparse
import json
import math
import warnings
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple

import pandas as pd
from tqdm import tqdm

from utils.data_utils import normalize_query_id

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Prompt templates (from the paper's reranking method)
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are an expert visual search evaluator specializing in precise image matching. Your task is to make highly discriminative binary decisions about visual transformation requests.
Be extremely precise and discriminating:
- Answer "True" ONLY when the candidate image CLEARLY and ACCURATELY demonstrates the exact requested change
- Answer "False" when the change is absent, partial, unclear, or when core visual elements don't match
- Pay attention to subtle but important differences - small variations matter significantly
- Prioritize precision over generosity - when in doubt, lean toward "False" """

BASE_INSTRUCTIONS = """You will evaluate whether a candidate image is a PRECISE match for a reference image given a user's specific modification request.
**STRICT Criteria for "True" (be very selective):**
- The candidate image demonstrates the EXACT requested change with high fidelity
- ALL core visual elements (object type, style, composition, context) closely match the reference
- The requested modification is clearly visible and well-executed
- NO significant visual inconsistencies or irrelevant elements
- The result would PERFECTLY satisfy a discerning user's search intent

**Criteria for "False" (default when uncertain):**
- The requested change is missing, partial, or poorly executed
- Core visual elements differ noticeably from the reference image
- The modification creates visual inconsistency or introduces irrelevant elements
- Any ambiguity about whether the change was properly implemented
- The result would leave a user wanting a better match

**Evaluation Process:**
1. Identify the SPECIFIC change requested in the modification
2. Examine the reference image for key visual elements to preserve
3. Analyze the candidate image for the exact requested change
4. Compare visual consistency between reference and candidate
5. Make a binary decision: only "True" for near-perfect matches

Answer with exactly one word: "True" or "False"

Be highly discriminating - subtle differences matter significantly in visual search."""


def signature_to_url(signature: str) -> str:
    """Convert a Pinterest image signature to a CDN URL."""
    return (f"https://i.pinimg.com/736x/{signature[:2]}/"
            f"{signature[2:4]}/{signature[4:6]}/{signature}.jpg")


def build_messages(
    instruction: str,
    query_image_url: str,
    candidate_image_url: str,
) -> List[Dict]:
    """
    Build the VLM message payload for scoring a single candidate.

    The message structure places the system prompt, base instructions,
    query context, and reference image in a shared prefix, with only
    the candidate image varying. This layout is optimized for KV-cache
    reuse when scoring multiple candidates for the same query.
    """
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": BASE_INSTRUCTIONS},
                {
                    "type": "text",
                    "text": (
                        f'\n**Current Task:**\nDetermine if the candidate '
                        f'image is a PRECISE match for the reference image '
                        f'with this specific modification: "{instruction}"\n\n'
                        f'**What to look for:**\n'
                        f'- The EXACT change specified in the modification '
                        f'request\n'
                        f'- Preservation of key visual elements from the '
                        f'reference\n'
                        f'- High-quality execution of the requested change\n'
                        f'- Visual consistency and coherence\n\n'
                        f'**Reference Image:**'
                    ),
                },
                {
                    "type": "image_url",
                    "image_url": {"url": query_image_url},
                },
                {
                    "type": "image_url",
                    "image_url": {"url": candidate_image_url},
                },
                {
                    "type": "text",
                    "text": (
                        "**Decision:**\nIs this candidate image a PRECISE "
                        "match showing the exact requested modification while "
                        "preserving the reference image's key visual "
                        "elements?\n\nAnswer exactly one word: True or False"
                    ),
                },
            ],
        },
    ]


def compute_relevance_score(logprobs: List[Dict]) -> Optional[float]:
    """
    Compute a relevance score from VLM logprobs.

    Extracts the log-probabilities of the "True" and "False" tokens and
    computes:  score = P(True) / (P(True) + P(False))

    Args:
        logprobs: List of {token, logprob} dicts from the top_logprobs
            field of a chat completion response.

    Returns:
        Float score in [0, 1], or None if True/False tokens not found.
    """
    true_logit = None
    false_logit = None

    for entry in logprobs:
        token = entry.get("token", "")
        if token == "True":
            true_logit = entry["logprob"]
        elif token == "False":
            false_logit = entry["logprob"]
        if true_logit is not None and false_logit is not None:
            break

    if true_logit is None or false_logit is None:
        return None

    p_true = math.exp(true_logit)
    p_false = math.exp(false_logit)
    return p_true / (p_true + p_false)


# ---------------------------------------------------------------------------
# VLM client interface
# ---------------------------------------------------------------------------

class VLMClient(ABC):
    """
    Abstract base class for calling a VLM to score a candidate image.

    To use a custom VLM backend, subclass this and implement
    `score_candidate`. The method receives the full message payload and
    should return the relevance score (float in [0,1]) or None on failure.
    """

    @abstractmethod
    def score_candidate(self, messages: List[Dict]) -> Optional[float]:
        """
        Score a single candidate image given the VLM messages.

        Args:
            messages: Chat messages in OpenAI format, as built by
                `build_messages()`. Contains the system prompt, reference
                image, candidate image, and evaluation instructions.

        Returns:
            Relevance score in [0, 1], or None if scoring fails.
            The score is P(True) / (P(True) + P(False)) derived from
            the model's logit probabilities.
        """
        ...


class OpenAICompatibleClient(VLMClient):
    """
    VLM client for OpenAI-compatible APIs (vLLM, SGLang, Ollama, etc.).

    Sends a chat completion request with logprobs enabled and extracts
    the True/False token probabilities to compute a relevance score.
    """

    def __init__(self, api_base: str, model: str, timeout: int = 30):
        import requests
        from requests.adapters import HTTPAdapter
        from urllib3.util.retry import Retry

        self.api_base = api_base.rstrip("/")
        self.model = model
        self.timeout = timeout

        self.session = requests.Session()
        retry = Retry(
            total=2,
            backoff_factor=0.1,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["POST"],
        )
        adapter = HTTPAdapter(pool_connections=10, pool_maxsize=20,
                              max_retries=retry)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)

    def score_candidate(self, messages: List[Dict]) -> Optional[float]:
        try:
            resp = self.session.post(
                f"{self.api_base}/chat/completions",
                json={
                    "model": self.model,
                    "messages": messages,
                    "temperature": 0,
                    "max_tokens": 1,
                    "logprobs": True,
                    "top_logprobs": 5,
                },
                timeout=self.timeout,
            )
            if resp.status_code != 200:
                return None

            data = resp.json()
            top_logprobs = (
                data["choices"][0]["logprobs"]["content"][0]["top_logprobs"]
            )
            return compute_relevance_score(top_logprobs)
        except Exception:
            return None


# ---------------------------------------------------------------------------
# Reranking logic
# ---------------------------------------------------------------------------

def rerank_query(
    client: VLMClient,
    instruction: str,
    query_image_sig: str,
    candidate_sigs: List[str],
) -> Tuple[List[str], List[Optional[float]]]:
    """
    Rerank candidates for a single query using VLM logit scores.

    Returns:
        Tuple of (reranked_signatures, scores).
    """
    query_url = signature_to_url(query_image_sig)
    scores: List[Optional[float]] = []

    for sig in candidate_sigs:
        candidate_url = signature_to_url(sig)
        messages = build_messages(instruction, query_url, candidate_url)
        scores.append(client.score_candidate(messages))

    paired = list(zip(candidate_sigs, scores))
    paired.sort(key=lambda x: x[1] if x[1] is not None else -1.0,
                reverse=True)

    return [p[0] for p in paired], [p[1] for p in paired]


def rerank_results(
    client: VLMClient,
    results: Dict,
    gt_df: pd.DataFrame,
    top_n: int = 20,
) -> Dict:
    """
    Rerank all queries in a results file.

    Args:
        client: VLM client instance.
        results: Original results dict {query_id: {retrieved_items: [...]}}.
        gt_df: Ground truth DataFrame with query metadata.
        top_n: Number of top candidates to rerank per query. Remaining
            candidates are appended in their original order.

    Returns:
        Reranked results dict in the same format as the input.
    """
    # Build lookup: query_id -> first row (one paraphrase per query_id)
    query_lookup = {}
    for _, row in gt_df.iterrows():
        qid = normalize_query_id(row["query_id"])
        if qid not in query_lookup:
            query_lookup[qid] = row

    reranked = {}
    for qid in tqdm(results, desc="Reranking"):
        retrieved = results[qid].get("retrieved_items", [])
        row = query_lookup.get(qid)

        if not retrieved or row is None:
            reranked[qid] = {"retrieved_items": retrieved}
            continue

        instruction = row.get("instruction", "")
        query_sig = row.get("query_image_signature")

        if not instruction or not query_sig:
            reranked[qid] = {"retrieved_items": retrieved}
            continue

        to_rerank = retrieved[:top_n]
        remainder = retrieved[top_n:]

        reranked_sigs, _ = rerank_query(
            client, instruction, query_sig, to_rerank
        )

        reranked[qid] = {"retrieved_items": reranked_sigs + remainder}

    return reranked


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Rerank retrieval results using a VLM",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--results", type=str, required=True,
        help="Path to input results JSON file")
    parser.add_argument(
        "--output", type=str, required=True,
        help="Path to output reranked results JSON file")
    parser.add_argument(
        "--ground_truth", type=str, default="pinpoint_licensed.parquet",
        help="Path to ground truth parquet (default: pinpoint_licensed.parquet)")
    parser.add_argument(
        "--api_base", type=str, default="http://localhost:8000/v1",
        help="OpenAI-compatible API base URL (default: http://localhost:8000/v1)")
    parser.add_argument(
        "--model", type=str, default="Qwen/Qwen2.5-VL-7B-Instruct",
        help="Model name (default: Qwen/Qwen2.5-VL-7B-Instruct)")
    parser.add_argument(
        "--top_n", type=int, default=20,
        help="Number of top candidates to rerank per query (default: 20)")
    parser.add_argument(
        "--timeout", type=int, default=30,
        help="API request timeout in seconds (default: 30)")

    args = parser.parse_args()

    print(f"Loading results from {args.results}")
    with open(args.results) as f:
        results = json.load(f)
    print(f"  {len(results)} queries")

    print(f"Loading ground truth from {args.ground_truth}")
    gt_df = pd.read_parquet(args.ground_truth)

    client = OpenAICompatibleClient(
        api_base=args.api_base,
        model=args.model,
        timeout=args.timeout,
    )
    print(f"Using model: {args.model} at {args.api_base}")

    reranked = rerank_results(client, results, gt_df, top_n=args.top_n)

    print(f"Saving reranked results to {args.output}")
    with open(args.output, "w") as f:
        json.dump(reranked, f, indent=2)

    print(f"Done. {len(reranked)} queries reranked.")


if __name__ == "__main__":
    main()
