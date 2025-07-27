from torch.utils.data import Dataset
import os
from PIL import Image
import numpy as np
import torch
from typing import Union


def load_image(image_path: Union[str, np.ndarray], target_size: int = 512, left: int = 0, right: int = 0, top: int = 0,
               bottom: int = 0):
    """
    Load and preprocess an image with cropping and resizing.
    
    Args:
        image_path: Path to image file or numpy array
        target_size: Target size for the output image (square)
        left, right, top, bottom: Crop margins
        
    Returns:
        Preprocessed image tensor with shape (C, H, W) and values in [-1, 1]
    """
    if isinstance(image_path, str):
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")
        image = np.array(Image.open(image_path).convert('RGB'))[:, :, :3]
    else:
        image = image_path

    h, w, c = image.shape

    # Validate and apply crop margins
    left = min(left, w - 1)
    right = min(right, w - left - 1)
    top = min(top, h - 1)  # Fixed: was h - left - 1
    bottom = min(bottom, h - top - 1)

    # Apply cropping
    image = image[top:h - bottom, left:w - right]
    h, w, c = image.shape

    # Center crop to square
    if h < w:
        offset = (w - h) // 2
        image = image[:, offset:offset + h]
    elif w < h:
        offset = (h - w) // 2
        image = image[offset:offset + w]

    # Resize to target size
    image = np.array(Image.fromarray(image).resize((target_size, target_size)))

    # Convert to tensor and normalize to [-1, 1]
    image = torch.from_numpy(image).float() / 127.5 - 1
    image = image.permute(2, 0, 1)

    return image


class ImagePairDataset(Dataset):
    """
    Dataset for loading source and target image pairs.
    
    Directory structure should be:
    - dir/
      - src/     (source images)
      - target/  (target images)
    """

    def __init__(self, data_dir: str, target_size: int = 512, transform=None):
        """
        Initialize the dataset.
        
        Args:
            data_dir: Root directory containing 'src' and 'target' folders
            target_size: Target size for loaded images
            transform: Optional transform to apply to images
        """
        super().__init__()

        self.data_dir = data_dir
        self.target_size = target_size
        self.transform = transform

        src_dir = os.path.join(data_dir, 'src')
        target_dir = os.path.join(data_dir, 'target')
        if not os.path.exists(src_dir):
            raise FileNotFoundError(f"Source directory not found: {src_dir}")
        if not os.path.exists(target_dir):
            raise FileNotFoundError(f"Target directory not found: {target_dir}")

        self.src_image_names = sorted(
            [f for f in os.listdir(src_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))])
        self.target_image_names = sorted(
            [f for f in os.listdir(target_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))])

        if len(self.src_image_names) == 0:
            raise ValueError(f"No valid images found in {src_dir}")
        if len(self.target_image_names) == 0:
            raise ValueError(f"No valid images found in {target_dir}")

    def __len__(self):
        return len(self.src_image_names)

    def __getitem__(self, index: int):
        """   
        Returns:
            Tuple of (source_image, target_image, source_filename, target_filename)
        """
        src_img_path = os.path.join(self.data_dir, 'src', self.src_image_names[index])
        src_img = load_image(src_img_path, target_size=self.target_size)

        target_img_path = os.path.join(self.data_dir, 'target', self.target_image_names[0])
        target_img = load_image(target_img_path, target_size=self.target_size)

        if self.transform:
            src_img = self.transform(src_img)
            target_img = self.transform(target_img)

        return src_img, target_img, self.src_image_names[index], self.target_image_names[0]
