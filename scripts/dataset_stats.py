"""
Calculate and display dataset statistics
"""
from pathlib import Path


def count_images(folder):
    """Count images in a folder"""
    path = Path(folder)
    if not path.exists():
        return 0
    jpg_count = len(list(path.glob("*.jpg")))
    png_count = len(list(path.glob("*.png")))
    JPG_count = len(list(path.glob("*.JPG")))
    return jpg_count + png_count + JPG_count


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

# Training genuine data
print("\n📁 TRAINING SET - GENUINE (data/augmented/train/)")
print("-" * 70)
train_nat_id = count_images("data/augmented/train/national_ids")
train_kcse = count_images("data/augmented/train/kcse_certificates")
train_passport = count_images("data/augmented/train/passports")

print(f"National IDs:        {train_nat_id:4d} images")
print(f"KCSE Certificates:   {train_kcse:4d} images")
print(f"Passports:           {train_passport:4d} images")
print(f"{'':20} Total: {train_nat_id + train_kcse + train_passport:4d} images")

# Training fake data
print("\n📁 TRAINING SET - FAKE (data/augmented/train/fake/)")
print("-" * 70)
train_fake_nat_id = count_images("data/augmented/train/fake/national_ids")
train_fake_kcse = count_images("data/augmented/train/fake/kcse_certificates")
train_fake_passport = count_images("data/augmented/train/fake/passports")

print(f"National IDs:        {train_fake_nat_id:4d} images")
print(f"KCSE Certificates:   {train_fake_kcse:4d} images")
print(f"Passports:           {train_fake_passport:4d} images")
print(f"{'':20} Total: {train_fake_nat_id + train_fake_kcse + train_fake_passport:4d} images")

# Validation genuine data
print("\n📁 VALIDATION SET - GENUINE (data/augmented/validation/)")
print("-" * 70)
val_nat_id = count_images("data/augmented/validation/national_ids")
val_kcse = count_images("data/augmented/validation/kcse_certificates")
val_passport = count_images("data/augmented/validation/passports")

print(f"National IDs:        {val_nat_id:4d} images")
print(f"KCSE Certificates:   {val_kcse:4d} images")
print(f"Passports:           {val_passport:4d} images")
print(f"{'':20} Total: {val_nat_id + val_kcse + val_passport:4d} images")

# Validation fake data
print("\n📁 VALIDATION SET - FAKE (data/augmented/validation/fake/)")
print("-" * 70)
val_fake_nat_id = count_images("data/augmented/validation/fake/national_ids")
val_fake_kcse = count_images("data/augmented/validation/fake/kcse_certificates")
val_fake_passport = count_images("data/augmented/validation/fake/passports")

print(f"National IDs:        {val_fake_nat_id:4d} images")
print(f"KCSE Certificates:   {val_fake_kcse:4d} images")
print(f"Passports:           {val_fake_passport:4d} images")
print(f"{'':20} Total: {val_fake_nat_id + val_fake_kcse + val_fake_passport:4d} images")

# Grand totals
print("\n" + "=" * 70)
genuine_total = (train_nat_id + train_kcse + train_passport + 
                 val_nat_id + val_kcse + val_passport)
fake_total = (train_fake_nat_id + train_fake_kcse + train_fake_passport +
              val_fake_nat_id + val_fake_kcse + val_fake_passport)
grand_total = genuine_total + fake_total

print(f"GENUINE IMAGES:  {genuine_total:4d}")
print(f"FAKE IMAGES:     {fake_total:4d}")
print(f"GRAND TOTAL:     {grand_total:4d} images ready for model training")
print("=" * 70)

# Dataset split percentages
train_total = (train_nat_id + train_kcse + train_passport + 
               train_fake_nat_id + train_fake_kcse + train_fake_passport)
val_total = (val_nat_id + val_kcse + val_passport +
             val_fake_nat_id + val_fake_kcse + val_fake_passport)

train_pct = (train_total / grand_total * 100) if grand_total > 0 else 0
val_pct = (val_total / grand_total * 100) if grand_total > 0 else 0

print(f"\nDataset Split:")
print(f"  Training:   {train_pct:.1f}% ({train_total} images)")
print(f"  Validation: {val_pct:.1f}% ({val_total} images)")

print("\n✅ Dataset ready for Week 3 model training!\n")