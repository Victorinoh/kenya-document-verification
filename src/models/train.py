"""
Training Script
Trains both CNN models and saves them to models/saved_models/
"""
import os
import sys
import tensorflow as tf
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.models.cnn_model       import build_document_classifier, build_authenticity_detector
from src.models.dataset_builder import build_classifier_dataset, build_detector_dataset

# ── SETTINGS ──────────────────────────────────────────────────────────────────

BATCH_SIZE = 16     # Reduce to 8 if you get memory errors
EPOCHS     = 25
os.makedirs("models/saved_models",  exist_ok=True)
os.makedirs("models/checkpoints",   exist_ok=True)
os.makedirs("models/training_logs", exist_ok=True)


# ── CALLBACKS ─────────────────────────────────────────────────────────────────

def get_callbacks(model_name):
    return [
        # Save best model automatically
        tf.keras.callbacks.ModelCheckpoint(
            filepath=f"models/checkpoints/{model_name}_best.h5",
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1
        ),
        # Stop early if no improvement for 7 epochs
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=7,
            restore_best_weights=True,
            verbose=1
        ),
        # Reduce learning rate when stuck
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            min_lr=1e-7,
            verbose=1
        ),
        # Training log CSV
        tf.keras.callbacks.CSVLogger(
            f"models/training_logs/{model_name}_log.csv"
        ),
    ]


# ── TRAIN DOCUMENT CLASSIFIER ─────────────────────────────────────────────────

def train_classifier():
    print("\n" + "=" * 60)
    print("TRAINING: DOCUMENT TYPE CLASSIFIER")
    print("=" * 60)

    # Build datasets
    print("\nLoading datasets...")
    train_ds, n_train = build_classifier_dataset("train",      BATCH_SIZE)
    val_ds,   n_val   = build_classifier_dataset("validation", BATCH_SIZE, augment=False)

    steps_per_epoch  = n_train // BATCH_SIZE
    validation_steps = n_val   // BATCH_SIZE
    print(f"Steps per epoch : {steps_per_epoch}")
    print(f"Validation steps: {validation_steps}")

    # Build model
    print("\nBuilding model...")
    model = build_document_classifier(use_transfer_learning=False)

    # Train
    print(f"\nTraining for up to {EPOCHS} epochs...")
    history = model.fit(
        train_ds,
        epochs=EPOCHS,
        validation_data=val_ds,
        callbacks=get_callbacks("document_classifier"),
        verbose=1
    )

    # Save final model
    save_path = "models/saved_models/document_classifier.h5"
    model.save(save_path)
    print(f"\n✅ Classifier saved to: {save_path}")

    # Print best results
    best_val_acc = max(history.history["val_accuracy"])
    best_train_acc = max(history.history["accuracy"])
    print(f"   Best train accuracy : {best_train_acc:.4f} ({best_train_acc*100:.1f}%)")
    print(f"   Best val accuracy   : {best_val_acc:.4f} ({best_val_acc*100:.1f}%)")

    return history


# ── TRAIN AUTHENTICITY DETECTOR ───────────────────────────────────────────────

def train_detector():
    print("\n" + "=" * 60)
    print("TRAINING: AUTHENTICITY DETECTOR")
    print("=" * 60)

    # Build datasets
    print("\nLoading datasets...")
    train_ds, n_train = build_detector_dataset("train",      BATCH_SIZE)
    val_ds,   n_val   = build_detector_dataset("validation", BATCH_SIZE, augment=False)

    steps_per_epoch  = n_train // BATCH_SIZE
    validation_steps = n_val   // BATCH_SIZE
    print(f"Steps per epoch : {steps_per_epoch}")
    print(f"Validation steps: {validation_steps}")

    # Build model
    print("\nBuilding model...")
    model = build_authenticity_detector(use_transfer_learning=False)

    # Train
    print(f"\nTraining for up to {EPOCHS} epochs...")
    history = model.fit(
        train_ds,
        epochs=EPOCHS,
        validation_data=val_ds,
        callbacks=get_callbacks("authenticity_detector"),
        verbose=1
    )

    # Save final model
    save_path = "models/saved_models/authenticity_detector.h5"
    model.save(save_path)
    print(f"\n✅ Detector saved to: {save_path}")

    # Print best results
    best_val_acc = max(history.history["val_accuracy"])
    best_train_acc = max(history.history["accuracy"])
    print(f"   Best train accuracy : {best_train_acc:.4f} ({best_train_acc*100:.1f}%)")
    print(f"   Best val accuracy   : {best_val_acc:.4f} ({best_val_acc*100:.1f}%)")

    return history


# ── MAIN ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("DOCUMENT VERIFICATION - MODEL TRAINING")
    print("=" * 60)

    # Train both models
    classifier_history = train_classifier()
    detector_history   = train_detector()

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print("Saved models:")
    print("  models/saved_models/document_classifier.h5")
    print("  models/saved_models/authenticity_detector.h5")
    print("Training logs:")
    print("  models/training_logs/document_classifier_log.csv")
    print("  models/training_logs/authenticity_detector_log.csv")