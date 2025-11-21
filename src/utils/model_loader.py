"""Model loading utilities for MetaCLIP2."""

import torch
import open_clip


def load_model(
    model_name: str = "ViT-H-14-worldwide-quickgelu",
    pretrained: str = "metaclip2_worldwide"
):
    """
    Load MetaCLIP2 model, preprocessing, and tokenizer.
    
    Args:
        model_name: OpenCLIP model name
        pretrained: OpenCLIP pretrained weights name
        
    Returns:
        Tuple of (model, preprocess, tokenizer, device)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print(f"Loading MetaCLIP2 model: {model_name} "
          f"with {pretrained} weights...")
    model, _, preprocess = open_clip.create_model_and_transforms(
        model_name,
        pretrained=pretrained,
        device=device
    )
    tokenizer = open_clip.get_tokenizer(model_name)
    model.eval()

    print("Model loaded successfully")
    return model, preprocess, tokenizer, device


def get_embedding_dimension(model, device) -> int:
    """
    Get the embedding dimension of the model.
    
    Args:
        model: The loaded model
        device: Device to use
        
    Returns:
        Embedding dimension (int)
    """
    dummy_image = torch.randn(1, 3, 224, 224).to(device)
    with torch.no_grad():
        dummy_features = model.encode_image(dummy_image)
        return dummy_features.shape[1]


