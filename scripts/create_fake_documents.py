"""
Create synthetic fake documents for training
Simulates common forgery techniques
"""
import cv2
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import random


class FakeDocumentGenerator:
    """Generate fake documents by altering genuine ones"""
    
    def __init__(self):
        self.methods = [
            'remove_hologram',
            'alter_text',
            'wrong_font',
            'low_quality',
            'missing_watermark',
            'color_shift'
        ]
    
    def remove_hologram(self, image):
        """Simulate hologram removal (white out top-right corner)"""
        img = image.copy()
        h, w = img.shape[:2]
        
        # White out hologram area (top-right)
        cv2.rectangle(img, 
                     (int(w*0.7), int(h*0.1)), 
                     (int(w*0.95), int(h*0.3)), 
                     (255, 255, 255), -1)
        
        return img
    
    def alter_text_region(self, image):
        """Alter a text region to simulate tampering"""
        img = image.copy()
        h, w = img.shape[:2]
        
        # Select random region (simulating altered ID number or name)
        x1 = int(w * random.uniform(0.1, 0.4))
        y1 = int(h * random.uniform(0.3, 0.6))
        x2 = int(x1 + w * 0.3)
        y2 = int(y1 + h * 0.08)
        
        # Add white rectangle (simulating whited-out text)
        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 255, 255), -1)
        
        # Add random text-like noise
        for _ in range(5):
            line_x = random.randint(x1 + 5, x2 - 5)
            line_y = random.randint(y1 + 5, y2 - 5)
            cv2.line(img, (line_x, line_y), (line_x + 20, line_y), (0, 0, 0), 1)
        
        return img
    
    def reduce_quality(self, image):
        """Reduce image quality (common in photocopies)"""
        img = image.copy()
        
        # Add compression artifacts
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), random.randint(20, 50)]
        _, enc_img = cv2.imencode('.jpg', img, encode_param)
        img = cv2.imdecode(enc_img, 1)
        
        # Add blur
        img = cv2.GaussianBlur(img, (5, 5), 0)
        
        # Add noise
        noise = np.random.normal(0, 15, img.shape).astype(np.uint8)
        img = cv2.add(img, noise)
        
        return img
    
    def shift_colors(self, image):
        """Shift colors (wrong printing or scanning)"""
        img = image.copy()
        
        # Random color shift
        shift = random.randint(-30, 30)
        img = cv2.convertScaleAbs(img, alpha=1.0, beta=shift)
        
        # Alter color balance
        img[:, :, random.randint(0, 2)] = cv2.add(
            img[:, :, random.randint(0, 2)], 
            random.randint(-20, 20)
        )
        
        return img
    
    def add_artifacts(self, image):
        """Add scanning/printing artifacts"""
        img = image.copy()
        h, w = img.shape[:2]
        
        # Add random lines (scanner artifacts)
        for _ in range(random.randint(1, 3)):
            y = random.randint(0, h)
            cv2.line(img, (0, y), (w, y), (200, 200, 200), 1)
        
        # Add random spots
        for _ in range(random.randint(5, 15)):
            x = random.randint(0, w-10)
            y = random.randint(0, h-10)
            cv2.circle(img, (x, y), random.randint(1, 3), (0, 0, 0), -1)
        
        return img
    
    def create_fake(self, image, method='random'):
        """
        Create a fake document
        
        Args:
            image: Original image
            method: Forgery method or 'random'
        
        Returns:
            Fake document image
        """
        if method == 'random':
            method = random.choice(self.methods)
        
        if method == 'remove_hologram':
            return self.remove_hologram(image)
        elif method == 'alter_text':
            return self.alter_text_region(image)
        elif method == 'low_quality':
            return self.reduce_quality(image)
        elif method == 'color_shift':
            return self.shift_colors(image)
        elif method == 'missing_watermark':
            # Similar to remove_hologram but different region
            return self.remove_hologram(image)
        else:
            # Combine multiple methods
            img = image.copy()
            img = self.reduce_quality(img)
            img = self.add_artifacts(img)
            return img
    
    def generate_fake_dataset(self, input_folder, output_folder, num_fakes_per_image=1):
        """
        Generate fake documents from genuine ones
        
        Args:
            input_folder: Folder with genuine documents
            output_folder: Folder to save fakes
            num_fakes_per_image: Number of fake versions per image
        """
        input_path = Path(input_folder)
        output_path = Path(output_folder)
        output_path.mkdir(parents=True, exist_ok=True)
        
        image_files = (
            list(input_path.glob("*.jpg")) + 
            list(input_path.glob("*.png"))
        )
        
        print(f"\n{'='*60}")
        print(f"FAKE DOCUMENT GENERATION")
        print(f"{'='*60}")
        print(f"Input folder: {input_folder}")
        print(f"Genuine images: {len(image_files)}")
        print(f"Fakes per image: {num_fakes_per_image}")
        print(f"Expected output: {len(image_files) * num_fakes_per_image} fake documents")
        print(f"{'='*60}\n")
        
        total_created = 0
        
        for img_file in image_files:
            image = cv2.imread(str(img_file))
            if image is None:
                continue
            
            for idx in range(num_fakes_per_image):
                fake_img = self.create_fake(image, method='random')
                
                output_filename = f"{img_file.stem}_fake_{idx+1:02d}{img_file.suffix}"
                output_file = output_path / output_filename
                
                cv2.imwrite(str(output_file), fake_img)
                total_created += 1
                
                print(f"✓ Created: {output_filename}")
        
        print(f"\n✅ Fake generation complete!")
        print(f"   Total fake documents created: {total_created}\n")


# Test/Demo script
if __name__ == "__main__":
    import sys
    
    generator = FakeDocumentGenerator()
    
    print("=" * 70)
    print("FAKE DOCUMENT GENERATOR")
    print("=" * 70)
    print("\nUsage:")
    print("  python scripts/create_fake_documents.py <input_folder> <output_folder> <num_fakes>")
    print("\nExample:")
    print("  python scripts/create_fake_documents.py data/raw/national_ids/genuine data/raw/national_ids/fake 1")
    print("\n" + "=" * 70)
    
    if len(sys.argv) >= 3:
        input_folder = sys.argv[1]
        output_folder = sys.argv[2]
        num_fakes = int(sys.argv[3]) if len(sys.argv) > 3 else 1
        
        generator.generate_fake_dataset(input_folder, output_folder, num_fakes)
    else:
        print("\n⚠️  No arguments provided. Tool is ready to use!\n")