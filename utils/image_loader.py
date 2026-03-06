"""Image loading utilities for local files, URLs, and Pinterest signatures."""

import time
from io import BytesIO
from pathlib import Path
from typing import Optional

import requests
from PIL import Image


def load_image(img_path: str, max_retries: int = 3) -> Optional[Image.Image]:
    """
    Load image from path, URL, or Pinterest signature.
    
    Supports:
    - Local file paths: /path/to/image.jpg
    - HTTP/HTTPS URLs: https://example.com/image.jpg
    - Pinterest-style signatures: 32+ character hex strings
    
    Args:
        img_path: Image path, URL, or Pinterest signature
        max_retries: Maximum number of retry attempts for network requests
        
    Returns:
        PIL Image object or None if loading fails
    """
    # Check if it's a local file
    if Path(img_path).exists():
        try:
            return Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"Failed to load local image {img_path}: {e}")
            return None

    # Check if it's a URL
    if img_path.startswith("http://") or img_path.startswith("https://"):
        for attempt in range(max_retries):
            try:
                response = requests.get(img_path, timeout=10)
                response.raise_for_status()
                return Image.open(BytesIO(response.content)).convert("RGB")
            except Exception as e:
                if attempt == max_retries - 1:
                    print(f"Failed to load URL {img_path}: {e}")
                time.sleep(0.5 * (attempt + 1))
        return None

    # Try Pinterest-style signature format (32+ char hex string)
    if (len(img_path) >= 32 and
            all(c in "0123456789abcdef" for c in img_path.lower())):
        url = (f"https://i.pinimg.com/736x/{img_path[:2]}/"
               f"{img_path[2:4]}/{img_path[4:6]}/{img_path}.jpg")
        for attempt in range(max_retries):
            try:
                response = requests.get(url, timeout=10)
                if response.status_code == 200:
                    return Image.open(BytesIO(response.content)).convert("RGB")
            except Exception as e:
                if attempt == max_retries - 1:
                    print(f"Failed to load signature {img_path}: {e}")
                time.sleep(0.5 * (attempt + 1))
        return None

    print(f"Unknown image format: {img_path}")
    return None

