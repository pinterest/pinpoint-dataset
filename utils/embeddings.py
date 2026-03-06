"""Embedding generation utilities for queries."""

from typing import Dict, Optional

import numpy as np
import torch
import torch.nn.functional as F

from .image_loader import load_image


def get_query_embedding(
    query_data: Dict,
    model,
    preprocess,
    tokenizer,
    device,
    mode: str = "combined",
    alpha: float = 0.8,
) -> Optional[np.ndarray]:
    """
    Get query embedding based on mode (combined, image_only, or text_only).
    
    Args:
        query_data: Dictionary containing query information with keys:
            - query_image_signature: Image identifier/path
            - query_image_signature2: Optional second image
            - instruction: Text instruction
        model: MetaCLIP2 model
        preprocess: Image preprocessing function
        tokenizer: Text tokenizer
        device: Device to use
        mode: "combined" (weighted average), "image_only", or "text_only"
        alpha: Weight for text in combined mode
            (0=image only, 1=text only, 0.8=80% text)
        
    Returns:
        Query embedding as numpy array, or None if processing fails
    """
    image_signature = query_data.get("query_image_signature")
    image_signature2 = query_data.get("query_image_signature2")
    text_instruction = query_data.get("instruction", "")

    # Validate inputs based on mode
    if mode in ["combined", "image_only"] and not image_signature:
        return None
    if mode in ["combined", "text_only"] and not text_instruction:
        return None

    try:
        with torch.no_grad():
            embeddings_to_average = []

            # Get image embedding if needed
            if mode in ["combined", "image_only"]:
                image = load_image(image_signature)
                if image is None:
                    return None

                image_tensor = preprocess(image).unsqueeze(0).to(device)
                image_features = model.encode_image(image_tensor)
                image_features = F.normalize(image_features, dim=-1)

                # Handle second image if present
                if image_signature2:
                    image2 = load_image(image_signature2)
                    if image2 is not None:
                        image2_tensor = preprocess(image2).unsqueeze(0)
                        image2_tensor = image2_tensor.to(device)
                        image2_features = model.encode_image(image2_tensor)
                        image2_features = F.normalize(image2_features, dim=-1)
                        # Average the two image embeddings
                        image_features = (image_features + image2_features) / 2
                        image_features = F.normalize(image_features, dim=-1)

                if mode == "image_only":
                    query_embedding = image_features
                else:
                    embeddings_to_average.append(image_features)

            # Get text embedding if needed
            if mode in ["combined", "text_only"]:
                text_tokens = tokenizer([text_instruction]).to(device)
                text_features = model.encode_text(text_tokens)
                text_features = F.normalize(text_features, dim=-1)

                if mode == "text_only":
                    query_embedding = text_features
                else:
                    embeddings_to_average.append(text_features)

            # For combined mode, weighted average of image and text
            if mode == "combined":
                image_emb = embeddings_to_average[0]
                text_emb = embeddings_to_average[1]
                query_embedding = alpha * text_emb + (1 - alpha) * image_emb
                query_embedding = F.normalize(query_embedding, dim=-1)

            # Convert to numpy
            query_embedding = query_embedding.cpu().numpy().astype(np.float32)
            if query_embedding.shape[0] == 1:
                query_embedding = query_embedding.squeeze(0)

            return query_embedding

    except Exception as e:
        print(f"Error processing query: {e}")
        return None

