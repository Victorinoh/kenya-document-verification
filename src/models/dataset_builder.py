"""
Dataset Builder
Converts image folders into TensorFlow datasets for CNN training
"""
import tensorflow as tf
import numpy as np
from pathlib import Path


# ── CONFIGURATION ─────────────────────────────────────────────────────────────

IMAGE_SIZE = (224, 224)
AUTOTUNE   = tf.data.AUTOTUNE

DOC_TYPE_MAP = {
    "national_ids":       0,
    "kcse_certificates":  1,
    "passports":          2,
}


# ── IMAGE LOADING ──────────────────────────────────────────────────────────────

def load_and_preprocess(path, label):
    """Load a single image and normalise to [0, 1]"""
    raw   = tf.io.read_file(path)
    image = tf.image.decode_jpeg(raw, channels=3)
    image = tf.image.resize(image, IMAGE_SIZE)
    image = tf.cast(image, tf.float32) / 255.0
    return image, label


def augment_image(image, label):
    """Light online augmentation applied during training"""
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_brightness(image, 0.1)
    image = tf.image.random_contrast(image, 0.9, 1.1)
    image = tf.clip_by_value(image, 0.0, 1.0)
    return image, label


# ── DATASET BUILDERS ──────────────────────────────────────────────────────────

def build_classifier_dataset(split="train", batch_size=16, augment=True):
    """
    Build a dataset for the Document Type Classifier.
    Labels: 0=National ID, 1=KCSE Certificate, 2=Passport

    Args:
        split:      'train' or 'validation'
        batch_size: images per batch
        augment:    apply online augmentation (training only)

    Returns:
        tf.data.Dataset yielding (image, doc_type_label) pairs
    """
    paths, labels = [], []

    for doc_type, label in DOC_TYPE_MAP.items():

        # Genuine images
        if split == "train":
            genuine_dir = Path(f"data/augmented/train/{doc_type}")
        else:
            genuine_dir = Path(f"data/augmented/validation/{doc_type}")

        for ext in ["*.jpg", "*.JPG", "*.png"]:
            for img_path in genuine_dir.glob(ext):
                paths.append(str(img_path))
                labels.append(label)

        # Fake images — same doc_type label (classifier identifies TYPE not authenticity)
        if split == "train":
            fake_dir = Path(f"data/augmented/train/fake/{doc_type}")
        else:
            fake_dir = Path(f"data/augmented/validation/fake/{doc_type}")

        for ext in ["*.jpg", "*.JPG", "*.png"]:
            for img_path in fake_dir.glob(ext):
                paths.append(str(img_path))
                labels.append(label)

    print(f"  Classifier {split}: {len(paths)} images")

    dataset = tf.data.Dataset.from_tensor_slices(
        (paths, labels)
    )
    dataset = dataset.shuffle(len(paths), seed=42)
    dataset = dataset.map(load_and_preprocess, num_parallel_calls=AUTOTUNE)

    if augment and split == "train":
        dataset = dataset.map(augment_image, num_parallel_calls=AUTOTUNE)

    dataset = dataset.batch(batch_size).prefetch(AUTOTUNE)
    return dataset, len(paths)


def build_detector_dataset(split="train", batch_size=16, augment=True):
    """
    Build a dataset for the Authenticity Detector.
    Labels: 0=Genuine, 1=Fake

    Args:
        split:      'train' or 'validation'
        batch_size: images per batch
        augment:    apply online augmentation (training only)

    Returns:
        tf.data.Dataset yielding (image, auth_label) pairs
    """
    paths, labels = [], []

    for doc_type in DOC_TYPE_MAP:

        # Genuine images → label 0
        if split == "train":
            genuine_dir = Path(f"data/augmented/train/{doc_type}")
        else:
            genuine_dir = Path(f"data/augmented/validation/{doc_type}")

        for ext in ["*.jpg", "*.JPG", "*.png"]:
            for img_path in genuine_dir.glob(ext):
                paths.append(str(img_path))
                labels.append(0)   # Genuine

        # Fake images → label 1
        if split == "train":
            fake_dir = Path(f"data/augmented/train/fake/{doc_type}")
        else:
            fake_dir = Path(f"data/augmented/validation/fake/{doc_type}")

        for ext in ["*.jpg", "*.JPG", "*.png"]:
            for img_path in fake_dir.glob(ext):
                paths.append(str(img_path))
                labels.append(1)   # Fake

    print(f"  Detector  {split}: {len(paths)} images "
          f"(genuine: {labels.count(0)}, fake: {labels.count(1)})")

    dataset = tf.data.Dataset.from_tensor_slices(
        (paths, labels)
    )
    dataset = dataset.shuffle(len(paths), seed=42)
    dataset = dataset.map(load_and_preprocess, num_parallel_calls=AUTOTUNE)

    if augment and split == "train":
        dataset = dataset.map(augment_image, num_parallel_calls=AUTOTUNE)

    dataset = dataset.batch(batch_size).prefetch(AUTOTUNE)
    return dataset, len(paths)


# ── QUICK TEST ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("DATASET BUILDER TEST")
    print("=" * 60)

    BATCH = 16

    print("\nBuilding Classifier datasets...")
    train_cls, n_train = build_classifier_dataset("train",      BATCH)
    val_cls,   n_val   = build_classifier_dataset("validation", BATCH, augment=False)

    print("\nBuilding Detector datasets...")
    train_det, n_train2 = build_detector_dataset("train",      BATCH)
    val_det,   n_val2   = build_detector_dataset("validation", BATCH, augment=False)

    # Inspect one batch
    print("\nInspecting one batch from classifier dataset...")
    for images, labels in train_cls.take(1):
        print(f"  Image batch shape : {images.shape}")
        print(f"  Label batch shape : {labels.shape}")
        print(f"  Pixel range       : [{images.numpy().min():.2f}, "
              f"{images.numpy().max():.2f}]")
        print(f"  Labels in batch   : {labels.numpy()}")

    print("\n✅ Dataset builder working correctly!")
    print("=" * 60)