"""
Authenticity Detector V2 — Improved Training
Fixes the genuine/fake bias with class weights, better architecture,
and longer training
"""
import os
import sys
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.models.dataset_builder import build_detector_dataset

os.makedirs("models/saved_models",  exist_ok=True)
os.makedirs("models/checkpoints",   exist_ok=True)
os.makedirs("models/training_logs", exist_ok=True)

IMAGE_SHAPE = (224, 224, 3)
BATCH_SIZE  = 16
EPOCHS      = 40


# ── IMPROVED MODEL ────────────────────────────────────────────────────────────

def build_detector_v2():
    """
    Improved authenticity detector:
    - Deeper architecture to learn subtle forgery cues
    - More aggressive dropout to prevent overfitting
    - L2 regularization
    """
    reg = keras.regularizers.l2(1e-4)

    model = keras.Sequential([
        keras.Input(shape=IMAGE_SHAPE),

        # Block 1 — edges and basic textures
        layers.Conv2D(32, (3,3), activation="relu", padding="same", kernel_regularizer=reg),
        layers.BatchNormalization(),
        layers.Conv2D(32, (3,3), activation="relu", padding="same", kernel_regularizer=reg),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2,2)),
        layers.Dropout(0.25),

        # Block 2 — patterns and security feature areas
        layers.Conv2D(64, (3,3), activation="relu", padding="same", kernel_regularizer=reg),
        layers.BatchNormalization(),
        layers.Conv2D(64, (3,3), activation="relu", padding="same", kernel_regularizer=reg),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2,2)),
        layers.Dropout(0.25),

        # Block 3 — complex forgery indicators
        layers.Conv2D(128, (3,3), activation="relu", padding="same", kernel_regularizer=reg),
        layers.BatchNormalization(),
        layers.Conv2D(128, (3,3), activation="relu", padding="same", kernel_regularizer=reg),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2,2)),
        layers.Dropout(0.3),

        # Block 4 — high-level forgery features
        layers.Conv2D(256, (3,3), activation="relu", padding="same", kernel_regularizer=reg),
        layers.BatchNormalization(),
        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.4),

        # Classification head
        layers.Dense(512, activation="relu", kernel_regularizer=reg),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(256, activation="relu", kernel_regularizer=reg),
        layers.Dropout(0.4),
        layers.Dense(1, activation="sigmoid", name="auth_output")
    ], name="authenticity_detector_v2")

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=5e-4),
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            keras.metrics.Precision(name="precision"),
            keras.metrics.Recall(name="recall"),
            keras.metrics.AUC(name="auc")
        ]
    )
    return model


# ── CALLBACKS ─────────────────────────────────────────────────────────────────

callbacks = [
    keras.callbacks.ModelCheckpoint(
        "models/checkpoints/detector_v2_best.h5",
        monitor="val_auc",        # Monitor AUC not just accuracy
        save_best_only=True,
        mode="max",
        verbose=1
    ),
    keras.callbacks.EarlyStopping(
        monitor="val_auc",
        patience=10,              # More patience for harder task
        restore_best_weights=True,
        mode="max",
        verbose=1
    ),
    keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=4,
        min_lr=1e-7,
        verbose=1
    ),
    keras.callbacks.CSVLogger(
        "models/training_logs/authenticity_detector_v2_log.csv"
    ),
]


# ── TRAIN ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("TRAINING: AUTHENTICITY DETECTOR V2")
    print("=" * 60)

    # Load datasets
    print("\nLoading datasets...")
    train_ds, n_train = build_detector_dataset("train",      BATCH_SIZE)
    val_ds,   n_val   = build_detector_dataset("validation", BATCH_SIZE, augment=False)
    print(f"Train: {n_train} images | Val: {n_val} images")

    # Class weights — penalize missing genuine docs more
    # This fixes the bias toward predicting "Fake"
    class_weight = {
        0: 1.5,   # Genuine — up-weight to fix recall
        1: 1.0    # Fake
    }
    print(f"\nClass weights: Genuine={class_weight[0]}, Fake={class_weight[1]}")

    # Build and summarize model
    model = build_detector_v2()
    model.summary()

    # Train
    print(f"\nTraining for up to {EPOCHS} epochs...")
    history = model.fit(
        train_ds,
        epochs=EPOCHS,
        validation_data=val_ds,
        callbacks=callbacks,
        class_weight=class_weight,
        verbose=1
    )

    # Save
    save_path = "models/saved_models/authenticity_detector_v2.h5"
    model.save(save_path)
    print(f"\n✅ Detector V2 saved: {save_path}")

    # Results
    best_val_acc = max(history.history["val_accuracy"])
    best_val_auc = max(history.history["val_auc"])
    best_precision = max(history.history["val_precision"])
    best_recall    = max(history.history["val_recall"])

    print("\n" + "=" * 60)
    print("V2 TRAINING RESULTS")
    print("=" * 60)
    print(f"  Best val accuracy  : {best_val_acc*100:.1f}%")
    print(f"  Best val AUC       : {best_val_auc:.4f}")
    print(f"  Best val precision : {best_precision:.4f}")
    print(f"  Best val recall    : {best_recall:.4f}")
    print("=" * 60)

    # Compare with V1
    print("\nV1 vs V2 Comparison:")
    print(f"  V1 accuracy : 63.4%  →  V2: {best_val_acc*100:.1f}%")
    print(f"  V1 AUC      : 0.6787 →  V2: {best_val_auc:.4f}")