"""
Image anonymization script
Blurs sensitive information from document images
"""
from PIL import Image, ImageDraw, ImageFilter
import os
from pathlib import Path


class DocumentAnonymizer:
    """Anonymize sensitive information in document images"""
    
    def __init__(self):
        self.blur_strength = 50
    
    def blur_region(self, image, x, y, width, height):
        """
        Blur a specific region of the image
        
        Args:
            image: PIL Image object
            x, y: Top-left corner coordinates (as ratio 0-1)
            width, height: Size of region (as ratio 0-1)
        """
        img_width, img_height = image.size
        
        # Convert ratios to pixels
        left = int(x * img_width)
        top = int(y * img_height)
        right = int((x + width) * img_width)
        bottom = int((y + height) * img_height)
        
        # Extract region
        region = image.crop((left, top, right, bottom))
        
        # Blur region
        blurred_region = region.filter(ImageFilter.GaussianBlur(self.blur_strength))
        
        # Paste back
        image.paste(blurred_region, (left, top))
        
        return image
    
    def black_out_region(self, image, x, y, width, height):
        """
        Black out a specific region
        
        Args:
            image: PIL Image object
            x, y: Top-left corner coordinates (as ratio 0-1)
            width, height: Size of region (as ratio 0-1)
        """
        draw = ImageDraw.Draw(image)
        img_width, img_height = image.size
        
        # Convert ratios to pixels
        left = int(x * img_width)
        top = int(y * img_height)
        right = int((x + width) * img_width)
        bottom = int((y + height) * img_height)
        
        # Draw black rectangle
        draw.rectangle([left, top, right, bottom], fill='black')
        
        return image
    
    def anonymize_national_id(self, image_path, output_path, method='blur'):
        """
        Anonymize Kenyan National ID
        
        Regions to anonymize (approximate positions):
        - Photo: top-left (0.05, 0.15, 0.25, 0.35)
        - ID Number: center-left (0.15, 0.35, 0.4, 0.08)
        - Name: center-left (0.15, 0.45, 0.5, 0.08)
        - DOB: center-left (0.15, 0.55, 0.3, 0.08)
        - Ghost Image: right side (0.65, 0.4, 0.2, 0.25)
        """
        image = Image.open(image_path)
        
        if method == 'blur':
            # Blur sensitive regions
            image = self.blur_region(image, 0.05, 0.15, 0.25, 0.35)  # Photo
            image = self.blur_region(image, 0.15, 0.35, 0.4, 0.08)   # ID Number
            image = self.blur_region(image, 0.15, 0.45, 0.5, 0.08)   # Name
            image = self.blur_region(image, 0.15, 0.55, 0.3, 0.08)   # DOB
            image = self.blur_region(image, 0.65, 0.4, 0.2, 0.25)    # Ghost image
        else:
            # Black out regions
            image = self.black_out_region(image, 0.05, 0.15, 0.25, 0.35)
            image = self.black_out_region(image, 0.15, 0.35, 0.4, 0.08)
            image = self.black_out_region(image, 0.15, 0.45, 0.5, 0.08)
            image = self.black_out_region(image, 0.15, 0.55, 0.3, 0.08)
            image = self.black_out_region(image, 0.65, 0.4, 0.2, 0.25)
        
        # Save anonymized image
        image.save(output_path)
        print(f"✓ Anonymized: {os.path.basename(output_path)}")
        
        return output_path
    
    def anonymize_certificate(self, image_path, output_path, method='blur'):
        """
        Anonymize KCSE Certificate
        
        Regions to anonymize:
        - Name: top center (0.3, 0.25, 0.4, 0.05)
        - Index Number: center (0.3, 0.35, 0.4, 0.05)
        - School: center (0.3, 0.45, 0.4, 0.05)
        """
        image = Image.open(image_path)
        
        if method == 'blur':
            image = self.blur_region(image, 0.3, 0.25, 0.4, 0.05)  # Name
            image = self.blur_region(image, 0.3, 0.35, 0.4, 0.05)  # Index
            image = self.blur_region(image, 0.3, 0.45, 0.4, 0.05)  # School
        else:
            image = self.black_out_region(image, 0.3, 0.25, 0.4, 0.05)
            image = self.black_out_region(image, 0.3, 0.35, 0.4, 0.05)
            image = self.black_out_region(image, 0.3, 0.45, 0.4, 0.05)
        
        image.save(output_path)
        print(f"✓ Anonymized: {os.path.basename(output_path)}")
        
        return output_path
    
    def anonymize_passport(self, image_path, output_path, method='blur'):
        """
        Anonymize Passport
        
        Regions to anonymize:
        - Photo: left side (0.05, 0.2, 0.3, 0.4)
        - Name: right side (0.4, 0.3, 0.5, 0.08)
        - Passport Number: right side (0.4, 0.4, 0.3, 0.06)
        - MRZ: bottom (0.05, 0.85, 0.9, 0.1)
        """
        image = Image.open(image_path)
        
        if method == 'blur':
            image = self.blur_region(image, 0.05, 0.2, 0.3, 0.4)   # Photo
            image = self.blur_region(image, 0.4, 0.3, 0.5, 0.08)   # Name
            image = self.blur_region(image, 0.4, 0.4, 0.3, 0.06)   # Passport #
            image = self.blur_region(image, 0.05, 0.85, 0.9, 0.1)  # MRZ
        else:
            image = self.black_out_region(image, 0.05, 0.2, 0.3, 0.4)
            image = self.black_out_region(image, 0.4, 0.3, 0.5, 0.08)
            image = self.black_out_region(image, 0.4, 0.4, 0.3, 0.06)
            image = self.black_out_region(image, 0.05, 0.85, 0.9, 0.1)
        
        image.save(output_path)
        print(f"✓ Anonymized: {os.path.basename(output_path)}")
        
        return output_path
    
    def batch_anonymize(self, input_folder, output_folder, doc_type='national_id', method='blur'):
        """
        Anonymize all images in a folder
        
        Args:
            input_folder: Folder containing original images
            output_folder: Folder to save anonymized images
            doc_type: 'national_id', 'certificate', or 'passport'
            method: 'blur' or 'blackout'
        """
        input_path = Path(input_folder)
        output_path = Path(output_folder)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Get all image files
        image_files = list(input_path.glob("*.jpg")) + list(input_path.glob("*.png")) + list(input_path.glob("*.jpeg"))
        
        print(f"\n{'='*50}")
        print(f"Anonymizing {len(image_files)} images...")
        print(f"Document type: {doc_type}")
        print(f"Method: {method}")
        print(f"{'='*50}\n")
        
        for img_file in image_files:
            output_file = output_path / f"anon_{img_file.name}"
            
            if doc_type == 'national_id':
                self.anonymize_national_id(str(img_file), str(output_file), method)
            elif doc_type == 'certificate':
                self.anonymize_certificate(str(img_file), str(output_file), method)
            elif doc_type == 'passport':
                self.anonymize_passport(str(img_file), str(output_file), method)
        
        print(f"\n✓ Completed! {len(image_files)} images anonymized")
        print(f"✓ Saved to: {output_folder}\n")


# Test/Demo function
if __name__ == "__main__":
    import sys
    
    anonymizer = DocumentAnonymizer()
    
    print("=" * 60)
    print("DOCUMENT ANONYMIZATION TOOL")
    print("=" * 60)
    print("\nUsage:")
    print("  python scripts/anonymize_images.py <input_folder> <output_folder> <doc_type> <method>")
    print("\nExample:")
    print("  python scripts/anonymize_images.py data/raw/collected_today data/raw/anonymized national_id blur")
    print("\nDocument types: national_id, certificate, passport")
    print("Methods: blur, blackout")
    print("\n" + "=" * 60)
    
    # If arguments provided, run batch anonymization
    if len(sys.argv) >= 4:
        input_folder = sys.argv[1]
        output_folder = sys.argv[2]
        doc_type = sys.argv[3]
        method = sys.argv[4] if len(sys.argv) > 4 else 'blur'
        
        anonymizer.batch_anonymize(input_folder, output_folder, doc_type, method)
    else:
        print("\n⚠️  No arguments provided. Tool is ready to use!")
        print("Place images in data/raw/collected_today/ then run with arguments.\n")