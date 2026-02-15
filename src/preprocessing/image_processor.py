"""
Image preprocessing for document verification
"""
import cv2
import numpy as np
from pathlib import Path


class ImageProcessor:
    """Preprocess document images for ML model"""
    
    def __init__(self, target_size=(224, 224)):
        """
        Initialize processor
        
        Args:
            target_size: (width, height) for resized images
        """
        self.target_size = target_size
    
    def load_image(self, image_path):
        """Load image from file"""
        img = cv2.imread(str(image_path))
        if img is None:
            raise ValueError(f"Could not load image: {image_path}")
        return img
    
    def resize(self, image):
        """Resize image to target size"""
        return cv2.resize(image, self.target_size)
    
    def normalize(self, image):
        """Normalize pixel values to [0, 1]"""
        return image.astype(np.float32) / 255.0
    
    def to_grayscale(self, image):
        """Convert to grayscale"""
        if len(image.shape) == 3:
            return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return image
    
    def denoise(self, image):
        """Remove noise from image"""
        if len(image.shape) == 3:
            return cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)
        else:
            return cv2.fastNlMeansDenoising(image, None, 10, 7, 21)
    
    def enhance_contrast(self, image):
        """Enhance image contrast using CLAHE"""
        if len(image.shape) == 2:
            # Grayscale image
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            return clahe.apply(image)
        else:
            # Color image - apply to LAB color space
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            l = clahe.apply(l)
            
            enhanced = cv2.merge([l, a, b])
            return cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
    
    def detect_edges(self, image):
        """Detect edges using Canny"""
        gray = self.to_grayscale(image)
        edges = cv2.Canny(gray, 50, 150)
        return edges
    
    def preprocess(self, image_path, denoise=True, enhance=True):
        """
        Complete preprocessing pipeline
        
        Args:
            image_path: Path to image file
            denoise: Whether to apply denoising
            enhance: Whether to enhance contrast
        
        Returns:
            Preprocessed image (normalized, resized)
        """
        # Load image
        img = self.load_image(image_path)
        
        # Optional: Denoise
        if denoise:
            img = self.denoise(img)
        
        # Optional: Enhance contrast
        if enhance:
            img = self.enhance_contrast(img)
        
        # Resize
        img = self.resize(img)
        
        # Normalize
        img = self.normalize(img)
        
        return img
    
    def preprocess_batch(self, image_folder, output_folder=None):
        """
        Preprocess all images in a folder
        
        Args:
            image_folder: Folder containing images
            output_folder: Optional folder to save preprocessed images
        
        Returns:
            List of preprocessed images
        """
        input_path = Path(image_folder)
        image_files = list(input_path.glob("*.jpg")) + list(input_path.glob("*.png"))
        
        preprocessed = []
        
        print(f"\nPreprocessing {len(image_files)} images...")
        
        for img_file in image_files:
            try:
                img = self.preprocess(str(img_file))
                preprocessed.append(img)
                print(f"✓ {img_file.name}")
                
                # Save if output folder specified
                if output_folder:
                    output_path = Path(output_folder)
                    output_path.mkdir(parents=True, exist_ok=True)
                    output_file = output_path / img_file.name
                    
                    # Convert back to uint8 for saving
                    save_img = (img * 255).astype(np.uint8)
                    cv2.imwrite(str(output_file), save_img)
                    
            except Exception as e:
                print(f"✗ Error processing {img_file.name}: {e}")
        
        print(f"\n✓ Preprocessed {len(preprocessed)} images")
        return preprocessed
    
    def get_image_stats(self, image_path):
        """Get basic statistics about an image"""
        img = self.load_image(image_path)
        
        stats = {
            'shape': img.shape,
            'size_mb': Path(image_path).stat().st_size / (1024 * 1024),
            'mean_pixel': np.mean(img),
            'std_pixel': np.std(img),
            'min_pixel': np.min(img),
            'max_pixel': np.max(img),
            'dtype': img.dtype
        }
        
        return stats


# Test script
if __name__ == "__main__":
    import sys
    
    processor = ImageProcessor()
    
    print("=" * 60)
    print("IMAGE PREPROCESSOR TEST")
    print("=" * 60)
    
    # Check if test image provided
    test_images = list(Path("data/raw/national_ids/genuine").glob("*.jpg"))
    
    if test_images:
        test_image = test_images[0]
        print(f"\nTest image: {test_image.name}")
        
        # Get stats
        stats = processor.get_image_stats(str(test_image))
        print(f"\nImage statistics:")
        print(f"  Shape: {stats['shape']}")
        print(f"  Size: {stats['size_mb']:.2f} MB")
        print(f"  Mean pixel: {stats['mean_pixel']:.2f}")
        print(f"  Std dev: {stats['std_pixel']:.2f}")
        
        # Preprocess
        print(f"\nPreprocessing...")
        preprocessed = processor.preprocess(str(test_image))
        print(f"  Output shape: {preprocessed.shape}")
        print(f"  Output range: [{preprocessed.min():.2f}, {preprocessed.max():.2f}]")
        
        print(f"\n✓ Preprocessing working correctly!")
    else:
        print("\n⚠️  No test images found in data/raw/national_ids/genuine/")
        print("Add some images and run again to test preprocessing.")
    
    print("=" * 60)