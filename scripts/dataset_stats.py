"""
Calculate and display dataset statistics
"""
from pathlib import Path
import cv2
import numpy as np


def count_images(folder):
    """Count images in a folder"""
    path = Path(folder)
    if not path.exists():
        return 0
    jpg_count = len(list(path.glob("*.jpg")))
    png_count = len(list(path.glob("*.png")))
    return jpg_count + png_count


def get_image_stats(folder):
    """Get statistics about images in folder"""
    path = Path(folder)
    if not path.exists():
        return None
    
    images = list(path.glob("*.jpg")) + list(path.glob("*.png"))
    if not images:
        return None
    
    sizes = []
    for img_path in images[:10]:  # Sample first 10
        img = cv2.imread(str(img_path))
        if img is not None:
            sizes.append(img.shape)
    
    return {
        'count': len(images),
        'sample_size': sizes[0] if sizes else None
    }


print("=" * 70)
print("DATASET STATISTICS")
print("=" * 70)

# Original data
print("\n📁 ORIGINAL DATA (data/raw/)")
print("-" * 70)
nat_id_genuine = count_images("data/raw/national_ids/genuine")
nat_id_fake = count_images("data/raw/national_ids/fake")
kcse_genuine = count_images("data/raw/kcse_certificates/genuine")
kcse_fake = count_images("data/raw/kcse_certificates/fake")
passport_genuine = count_images("data/raw/passports/genuine")
passport_fake = count_images("data/raw/passports/fake")

print(f"National IDs:        Genuine: {nat_id_genuine:3d}  |  Fake: {nat_id_fake:3d}")
print(f"KCSE Certificates:   Genuine: {kcse_genuine:3d}  |  Fake: {kcse_fake:3d}")
print(f"Passports:           Genuine: {passport_genuine:3d}  |  Fake: {passport_fake:3d}")
print(f"{'':20} Total: {nat_id_genuine + kcse_genuine + passport_genuine:3d}  |  Total: {nat_id_fake + kcse_fake + passport_fake:3d}")

# Augmented training data
print("\n📁 TRAINING SET (data/augmented/train/)")
print("-" * 70)
train_nat_id = count_images("data/augmented/train/national_ids")
train_kcse = count_images("data/augmented/train/kcse_certificates")
train_passport = count_images("data/augmented/train/passports")

print(f"National IDs:        {train_nat_id:4d} images")
print(f"KCSE Certificates:   {train_kcse:4d} images")
print(f"Passports:           {train_passport:4d} images")
print(f"{'':20} Total: {train_nat_id + train_kcse + train_passport:4d} images")

# Validation data
print("\n📁 VALIDATION SET (data/augmented/validation/)")
print("-" * 70)
val_nat_id = count_images("data/augmented/validation/national_ids")
val_kcse = count_images("data/augmented/validation/kcse_certificates")
val_passport = count_images("data/augmented/validation/passports")

print(f"National IDs:        {val_nat_id:4d} images")
print(f"KCSE Certificates:   {val_kcse:4d} images")
print(f"Passports:           {val_passport:4d} images")
print(f"{'':20} Total: {val_nat_id + val_kcse + val_passport:4d} images")

# Grand total
print("\n" + "=" * 70)
grand_total = (train_nat_id + train_kcse + train_passport + 
               val_nat_id + val_kcse + val_passport)
print(f"GRAND TOTAL: {grand_total} images ready for model training")
print("=" * 70)

# Dataset split percentages
train_pct = ((train_nat_id + train_kcse + train_passport) / grand_total * 100)
val_pct = ((val_nat_id + val_kcse + val_passport) / grand_total * 100)

print(f"\nDataset Split:")
print(f"  Training:   {train_pct:.1f}%")
print(f"  Validation: {val_pct:.1f}%")

print("\n✅ Dataset ready for Week 3 model training!\n")