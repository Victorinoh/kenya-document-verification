"""
Diagnose exact folder contents - full picture
"""
from pathlib import Path

doc_types = ["national_ids", "kcse_certificates", "passports"]

print("=" * 70)
print("FULL FOLDER DIAGNOSTIC")
print("=" * 70)

total_train_genuine = 0
total_train_fake    = 0
total_val_genuine   = 0
total_val_fake      = 0

for split in ["train", "validation"]:
    print(f"\n{'='*30} {split.upper()} {'='*30}")
    
    for doc in doc_types:
        # Genuine
        genuine_dir = Path(f"data/augmented/{split}/{doc}")
        genuine_count = 0
        if genuine_dir.exists():
            genuine_count = len(
                list(genuine_dir.glob("*.jpg")) +
                list(genuine_dir.glob("*.JPG")) +
                list(genuine_dir.glob("*.png"))
            )

        # Fake
        fake_dir = Path(f"data/augmented/{split}/fake/{doc}")
        fake_count = 0
        if fake_dir.exists():
            fake_count = len(
                list(fake_dir.glob("*.jpg")) +
                list(fake_dir.glob("*.JPG")) +
                list(fake_dir.glob("*.png"))
            )

        print(f"\n  {doc}:")
        print(f"    Genuine : {genuine_count}  ({genuine_dir})")
        print(f"    Fake    : {fake_count}  ({fake_dir})")

        if split == "train":
            total_train_genuine += genuine_count
            total_train_fake    += fake_count
        else:
            total_val_genuine   += genuine_count
            total_val_fake      += fake_count

print("\n" + "=" * 70)
print("TOTALS")
print("=" * 70)
print(f"  Train   → genuine: {total_train_genuine}  fake: {total_train_fake}  "
      f"total: {total_train_genuine + total_train_fake}")
print(f"  Val     → genuine: {total_val_genuine}  fake: {total_val_fake}  "
      f"total: {total_val_genuine + total_val_fake}")
print(f"  GRAND TOTAL: {total_train_genuine+total_train_fake+total_val_genuine+total_val_fake}")
print("=" * 70)