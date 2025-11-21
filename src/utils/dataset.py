"""Dataset classes for image loading."""

import torch
from torch.utils.data import Dataset

from .image_loader import load_image


class ImageDataset(Dataset):
    """Dataset for loading images from file paths, URLs, or Pinterest
    signatures."""

    def __init__(self, image_list_path: str, preprocess):
        """
        Initialize dataset from a file containing image paths/URLs.
        
        Args:
            image_list_path: Path to file containing image paths/URLs
                (one per line)
            preprocess: Image preprocessing function from OpenCLIP
        """
        self.images = []
        self.preprocess = preprocess

        with open(image_list_path, "r") as f:
            for line in f:
                img_path = line.strip()
                if img_path:
                    self.images.append(img_path)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        """
        Get a single image and its identifier.
        
        Returns:
            Tuple of (image_tensor, image_path, success_flag)
        """
        img_path = self.images[idx]
        image = load_image(img_path)

        if image is not None:
            try:
                image_tensor = self.preprocess(image)
                return image_tensor, img_path, True
            except Exception as e:
                print(f"Failed to preprocess {img_path}: {e}")
                return torch.zeros(3, 224, 224), img_path, False
        else:
            return torch.zeros(3, 224, 224), img_path, False

