"""
Week 2 Integration Tests
Verifies all data preparation components work together
"""
import pytest
from pathlib import Path
import sys
sys.path.insert(0, '.')

from src.preprocessing.data_loader import DocumentDataLoader
from src.preprocessing.advanced_augmentation import cutout, add_document_noise
import numpy as np


def test_dataset_exists():
    """Verify dataset folders exist"""
    required_folders = [
        "data/augmented/train/national_ids",
        "data/augmented/train/kcse_certificates",
        "data/augmented/train/passports",
        "data/augmented/train/fake/national_ids",
        "data/augmented/validation/national_ids",
        "data/augmented/validation/fake/national_ids",
    ]
    for folder in required_folders:
        assert Path(folder).exists(), f"Missing folder: {folder}"


def test_dataset_has_images():
    """Verify dataset has enough images"""
    train_folder = Path("data/augmented/train/national_ids")
    images = list(train_folder.glob("*.jpg")) + list(train_folder.glob("*.JPG"))
    assert len(images) >= 50, f"Too few training images: {len(images)}"


def test_data_loader_initializes():
    """Test DataLoader creates correctly"""
    loader = DocumentDataLoader(image_size=(224, 224))
    assert loader is not None
    assert loader.image_size == (224, 224)


def test_data_loader_loads_train():
    """Test loading training data"""
    loader = DocumentDataLoader()
    data = loader.load_split('train')
    assert len(data) > 0, "No training data loaded"
    assert len(data) > 100, f"Too few images: {len(data)}"


def test_data_loader_loads_validation():
    """Test loading validation data"""
    loader = DocumentDataLoader()
    data = loader.load_split('validation')
    assert len(data) > 0, "No validation data loaded"


def test_batch_shape():
    """Test batch output has correct shape"""
    loader = DocumentDataLoader(image_size=(224, 224))
    loader.load_split('train')
    images, doc_labels, auth_labels = loader.load_batch(batch_size=4)
    
    assert images.shape == (4, 224, 224, 3), f"Wrong shape: {images.shape}"
    assert len(doc_labels) == 4
    assert len(auth_labels) == 4


def test_pixel_normalization():
    """Test images are normalized to [0, 1]"""
    loader = DocumentDataLoader()
    loader.load_split('train')
    images, _, _ = loader.load_batch(batch_size=4)
    
    assert images.min() >= 0.0, "Pixels below 0"
    assert images.max() <= 1.0, "Pixels above 1"


def test_cutout_augmentation():
    """Test cutout augmentation"""
    img = np.ones((224, 224, 3), dtype=np.uint8) * 128
    result = cutout(img, n_holes=2, hole_size=20)
    assert result.shape == img.shape
    assert result.sum() < img.sum()  # Some pixels zeroed out


def test_document_noise():
    """Test document noise augmentation"""
    img = np.ones((224, 224, 3), dtype=np.uint8) * 128
    result = add_document_noise(img)
    assert result.shape == img.shape


if __name__ == "__main__":
    pytest.main([__file__, "-v"])