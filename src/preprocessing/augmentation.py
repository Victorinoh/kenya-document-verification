"""
Data augmentation for document images
Generates multiple variations from original images
"""
import cv2
import numpy as np
from pathlib import Path
import albumentations as A
from tqdm import tqdm


class DocumentAugmenter:
    """Augment document images for training"""
    
    def __init__(self, output_size=(224, 224)):
        """
        Initialize augmenter with transformation pipeline
        
        Args:
            output_size: Target size for augmented images
        """
        self.output_size = output_size
        
        # Define augmentation pipeline
        self.transform = A.Compose([
            # Geometric transformations
            A.Rotate(limit=5, border_mode=cv2.BORDER_CONSTANT, value=255, p=0.7),
            A.ShiftScaleRotate(
                shift_limit=0.05,
                scale_limit=0.05,
                rotate_limit=3,
                border_mode=cv2.BORDER_CONSTANT,
                value=255,
                p=0.7
            ),
            A.Perspective(scale=(0.02, 0.05), p=0.5),
            
            # Image quality variations
            A.GaussianBlur(blur_limit=(3, 5), p=0.3),
            A.GaussNoise(var_limit=(5.0, 20.0), p=0.3),
            A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.3), p=0.2),
            
            # Brightness and contrast
            A.RandomBrightnessContrast(
                brightness_limit=0.15,
                contrast_limit=0.15,
                p=0.6
            ),
            A.RandomGamma(gamma_limit=(90, 110), p=0.3),
            
            # Color variations
            A.HueSaturationValue(
                hue_shift_limit=5,
                sat_shift_limit=10,
                val_shift_limit=10,
                p=0.4
            ),
            
            # Compression artifacts
            A.ImageCompression(quality_lower=85, quality_upper=100, p=0.3),
            
            # Resize to target
            A.Resize(height=output_size[0], width=output_size[1])
        ])
        
        # Lighter augmentation (for validation set)
        self.light_transform = A.Compose([
            A.Rotate(limit=2, border_mode=cv2.BORDER_CONSTANT, value=255, p=0.5),
            A.RandomBrightnessContrast(
                brightness_limit=0.1,
                contrast_limit=0.1,
                p=0.4
            ),
            A.Resize(height=output_size[0], width=output_size[1])
        ])
    
    def augment_image(self, image, num_augmentations=10, light=False):
        """
        Generate multiple augmented versions of an image
        
        Args:
            image: Input image (numpy array)
            num_augmentations: Number of variations to create
            light: Use lighter augmentations
        
        Returns:
            List of augmented images
        """
        augmented_images = []
        transform = self.light_transform if light else self.transform
        
        for _ in range(num_augmentations):
            augmented = transform(image=image)
            augmented_images.append(augmented['image'])
        
        return augmented_images
    
    def augment_dataset(self, input_folder, output_folder, num_augmentations=10, light=False):
        """
        Augment all images in a folder
        
        Args:
            input_folder: Folder containing original images
            output_folder: Folder to save augmented images
            num_augmentations: Number of variations per image
            light: Use lighter augmentations
        """
        input_path = Path(input_folder)
        output_path = Path(output_folder)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Get all image files
        image_files = (
            list(input_path.glob("*.jpg")) + 
            list(input_path.glob("*.png")) + 
            list(input_path.glob("*.jpeg"))
        )
        
        print(f"\n{'='*60}")
        print(f"DATA AUGMENTATION")
        print(f"{'='*60}")
        print(f"Input folder: {input_folder}")
        print(f"Output folder: {output_folder}")
        print(f"Images to process: {len(image_files)}")
        print(f"Augmentations per image: {num_augmentations}")
        print(f"Mode: {'Light' if light else 'Full'}")
        print(f"Expected output: {len(image_files) * (num_augmentations + 1)} images")
        print(f"{'='*60}\n")
        
        total_created = 0
        
        # Process each image
        for img_file in tqdm(image_files, desc="Augmenting images"):
            # Load image
            image = cv2.imread(str(img_file))
            if image is None:
                print(f"⚠️  Could not load: {img_file.name}")
                continue
            
            # Save original (resized)
            original_resized = cv2.resize(image, self.output_size)
            original_output = output_path / f"{img_file.stem}_original{img_file.suffix}"
            cv2.imwrite(str(original_output), original_resized)
            total_created += 1
            
            # Generate augmentations
            augmented_images = self.augment_image(image, num_augmentations, light)
            
            # Save augmented images
            for idx, aug_img in enumerate(augmented_images, 1):
                output_filename = f"{img_file.stem}_aug_{idx:02d}{img_file.suffix}"
                output_file = output_path / output_filename
                cv2.imwrite(str(output_file), aug_img)
                total_created += 1
        
        print(f"\n✅ Augmentation complete!")
        print(f"   Original images: {len(image_files)}")
        print(f"   Total created: {total_created}")
        print(f"   Saved to: {output_folder}\n")
        
        return total_created
    
    def create_split_datasets(self, input_folder, output_base, train_aug=10, val_aug=3):
        """
        Create train/validation split with different augmentation levels
        
        Args:
            input_folder: Folder with original images
            output_base: Base output folder
            train_aug: Augmentations for training set
            val_aug: Augmentations for validation set
        """
        input_path = Path(input_folder)
        image_files = (
            list(input_path.glob("*.jpg")) + 
            list(input_path.glob("*.png"))
        )
        
        # Split 85/15
        train_split = int(len(image_files) * 0.85)
        train_files = image_files[:train_split]
        val_files = image_files[train_split:]
        
        print(f"\n{'='*60}")
        print(f"CREATING TRAIN/VAL SPLIT")
        print(f"{'='*60}")
        print(f"Total images: {len(image_files)}")
        print(f"Training images: {len(train_files)} ({train_aug} aug each)")
        print(f"Validation images: {len(val_files)} ({val_aug} aug each)")
        print(f"{'='*60}\n")
        
        # Create training set
        train_folder = Path(output_base) / "train"
        train_folder.mkdir(parents=True, exist_ok=True)
        
        for img_file in tqdm(train_files, desc="Creating training set"):
            image = cv2.imread(str(img_file))
            if image is None:
                continue
            
            # Original
            original_resized = cv2.resize(image, self.output_size)
            cv2.imwrite(str(train_folder / f"{img_file.stem}_original{img_file.suffix}"), original_resized)
            
            # Augmentations
            augmented = self.augment_image(image, train_aug, light=False)
            for idx, aug_img in enumerate(augmented, 1):
                output_file = train_folder / f"{img_file.stem}_aug_{idx:02d}{img_file.suffix}"
                cv2.imwrite(str(output_file), aug_img)
        
        # Create validation set (lighter augmentation)
        val_folder = Path(output_base) / "validation"
        val_folder.mkdir(parents=True, exist_ok=True)
        
        for img_file in tqdm(val_files, desc="Creating validation set"):
            image = cv2.imread(str(img_file))
            if image is None:
                continue
            
            # Original
            original_resized = cv2.resize(image, self.output_size)
            cv2.imwrite(str(val_folder / f"{img_file.stem}_original{img_file.suffix}"), original_resized)
            
            # Light augmentations
            augmented = self.augment_image(image, val_aug, light=True)
            for idx, aug_img in enumerate(augmented, 1):
                output_file = val_folder / f"{img_file.stem}_aug_{idx:02d}{img_file.suffix}"
                cv2.imwrite(str(output_file), aug_img)
        
        train_count = len(list(train_folder.glob("*.jpg"))) + len(list(train_folder.glob("*.png")))
        val_count = len(list(val_folder.glob("*.jpg"))) + len(list(val_folder.glob("*.png")))
        
        print(f"\n✅ Dataset split complete!")
        print(f"   Training images: {train_count}")
        print(f"   Validation images: {val_count}")
        print(f"   Total: {train_count + val_count}\n")


# Test/Demo script
if __name__ == "__main__":
    import sys
    
    augmenter = DocumentAugmenter()
    
    print("=" * 70)
    print("DOCUMENT AUGMENTATION TOOL")
    print("=" * 70)
    print("\nUsage:")
    print("  python src/preprocessing/augmentation.py <input_folder> <output_folder> <num_aug>")
    print("\nExample:")
    print("  python src/preprocessing/augmentation.py data/raw/national_ids/genuine data/augmented/national_ids 10")
    print("\n" + "=" * 70)
    
    # If arguments provided, run augmentation
    if len(sys.argv) >= 3:
        input_folder = sys.argv[1]
        output_folder = sys.argv[2]
        num_aug = int(sys.argv[3]) if len(sys.argv) > 3 else 10
        
        augmenter.augment_dataset(input_folder, output_folder, num_aug)
    else:
        print("\n⚠️  No arguments provided. Tool is ready to use!")
        print("Add images to data/raw/ folders, then run with arguments.\n")