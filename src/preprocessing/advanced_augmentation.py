"""
Advanced augmentation techniques
Cutout, MixUp for more diverse training data
"""
import cv2
import numpy as np
import random
from pathlib import Path


def cutout(image, n_holes=3, hole_size=30):
    """
    Randomly mask out square regions to simulate occlusion
    Useful for teaching model to focus on multiple features
    """
    h, w = image.shape[:2]
    result = image.copy()
    
    for _ in range(n_holes):
        y = random.randint(0, h - hole_size)
        x = random.randint(0, w - hole_size)
        result[y:y+hole_size, x:x+hole_size] = 0
    
    return result


def add_document_noise(image):
    """Simulate real-world document scanning noise"""
    result = image.copy().astype(np.float32)
    
    # Random choice of noise type
    noise_type = random.choice(['scan_lines', 'jpeg_artifact', 'fold_mark'])
    
    if noise_type == 'scan_lines':
        # Horizontal scan line artifacts
        for y in range(0, image.shape[0], random.randint(20, 50)):
            result[y, :] *= random.uniform(0.85, 0.95)
    
    elif noise_type == 'jpeg_artifact':
        # Simulate JPEG compression
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), random.randint(40, 70)]
        _, encoded = cv2.imencode('.jpg', image, encode_param)
        result = cv2.imdecode(encoded, cv2.IMREAD_COLOR).astype(np.float32)
    
    elif noise_type == 'fold_mark':
        # Simulate fold line
        x = random.randint(image.shape[1]//4, 3*image.shape[1]//4)
        result[:, x-2:x+2] *= 0.7
    
    return np.clip(result, 0, 255).astype(np.uint8)


def test_advanced_augmentation():
    """Quick test"""
    print("Testing advanced augmentation...")
    
    # Create test image
    test_img = np.ones((224, 224, 3), dtype=np.uint8) * 128
    
    # Test cutout
    result = cutout(test_img)
    assert result.shape == test_img.shape
    print("✅ Cutout: OK")
    
    # Test document noise
    result = add_document_noise(test_img)
    assert result.shape == test_img.shape
    print("✅ Document noise: OK")
    
    print("\n✅ Advanced augmentation ready!")


if __name__ == "__main__":
    test_advanced_augmentation()