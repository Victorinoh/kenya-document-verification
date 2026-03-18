"""
Prepare Dataset
Splits anonymized augmented images into train/validation sets
and organises them into the structure dataset_builder.py expects.

Output structure:
data/processed/
├── train/
│   ├── national_ids/        ← genuine IDs for classifier + detector
│   ├── passports/           ← genuine passports
│   ├── kcse_certificates/   ← genuine KCSE
│   └── fake/
│       ├── national_ids/    ← fake IDs for detector
│       ├── passports/       ← fake passports
│       └── kcse_certificates/
└── validation/
    ├── national_ids/
    ├── passports/
    ├── kcse_certificates/
    └── fake/
        ├── national_ids/
        ├── passports/
        └── kcse_certificates/
"""

import shutil
import random
from pathlib import Path

random.seed(42)

# ── CONFIG ────────────────────────────────────────────────────────────────────
SOURCE_BASE   = Path('data/raw/anonymized')
OUTPUT_BASE   = Path('data/processed')
DOC_TYPES     = ['national_ids', 'passports', 'kcse_certificates']
TRAIN_RATIO   = 0.80   # 80% train, 20% validation

# ── HELPERS ───────────────────────────────────────────────────────────────────
def copy_images(images, dest_folder):
    dest_folder.mkdir(parents=True, exist_ok=True)
    for img_path in images:
        shutil.copy2(img_path, dest_folder / img_path.name)

def get_images(folder):
    if not folder.exists():
        return []
    return (list(folder.glob('*.jpg'))  +
            list(folder.glob('*.jpeg')) +
            list(folder.glob('*.png')))

def split_images(images, train_ratio):
    random.shuffle(images)
    split = int(len(images) * train_ratio)
    return images[:split], images[split:]

# ── MAIN ──────────────────────────────────────────────────────────────────────
def main():
    print("=" * 55)
    print("PREPARING DATASET")
    print("=" * 55)

    total_train = 0
    total_val   = 0

    for doc_type in DOC_TYPES:
        print(f"\n📄 Processing {doc_type}")

        # ── GENUINE IMAGES ────────────────────────────────────────
        genuine_folder = SOURCE_BASE / doc_type / 'genuine'
        genuine_images = get_images(genuine_folder)

        if not genuine_images:
            print(f"  ⚠️  No genuine images found in {genuine_folder}")
            continue

        train_genuine, val_genuine = split_images(genuine_images, TRAIN_RATIO)

        # Copy genuine to train
        copy_images(
            train_genuine,
            OUTPUT_BASE / 'train' / doc_type
        )
        # Copy genuine to validation
        copy_images(
            val_genuine,
            OUTPUT_BASE / 'validation' / doc_type
        )

        print(f"  Genuine  → train: {len(train_genuine)} | "
              f"val: {len(val_genuine)}")

        # ── FAKE IMAGES ───────────────────────────────────────────
        fake_folder = SOURCE_BASE / doc_type / 'fake'
        fake_images = get_images(fake_folder)

        if not fake_images:
            print(f"  ⚠️  No fake images found in {fake_folder}")
        else:
            train_fake, val_fake = split_images(fake_images, TRAIN_RATIO)

            # Copy fake to train/fake/{doc_type}
            copy_images(
                train_fake,
                OUTPUT_BASE / 'train' / 'fake' / doc_type
            )
            # Copy fake to validation/fake/{doc_type}
            copy_images(
                val_fake,
                OUTPUT_BASE / 'validation' / 'fake' / doc_type
            )

            print(f"  Fake     → train: {len(train_fake)} | "
                  f"val: {len(val_fake)}")

        total_train += len(train_genuine)
        total_val   += len(val_genuine)

    # ── SUMMARY ───────────────────────────────────────────────────
    print("\n" + "=" * 55)
    print("DATASET PREPARATION COMPLETE")
    print("=" * 55)
    print(f"  Train genuine images : {total_train}")
    print(f"  Val   genuine images : {total_val}")
    print(f"\n  Output saved to: data/processed/")
    print("\n  Next step: update dataset_builder.py paths and run train.py")


if __name__ == "__main__":
    main()