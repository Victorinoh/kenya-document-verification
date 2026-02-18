"""
CNN Model Architecture
Document Verification System - Week 3
Two models:
  1. Document Type Classifier  (3 classes)
  2. Authenticity Detector     (2 classes)
"""
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import EfficientNetB0


# ── CONFIGURATION ─────────────────────────────────────────────────────────────

IMAGE_SIZE   = (224, 224)
IMAGE_SHAPE  = (224, 224, 3)
NUM_DOC_TYPES = 3   # National ID, KCSE Certificate, Passport
NUM_AUTH      = 2   # Genuine, Fake

DOC_CLASSES  = {0: "National ID", 1: "KCSE Certificate", 2: "Passport"}
AUTH_CLASSES = {0: "Genuine",     1: "Fake"}


# ── MODEL 1: DOCUMENT TYPE CLASSIFIER ─────────────────────────────────────────

def build_document_classifier(use_transfer_learning=True):
    """
    Build CNN to classify document type:
    National ID (0) / KCSE Certificate (1) / Passport (2)

    Args:
        use_transfer_learning: If True, use EfficientNetB0 backbone (recommended)
                               If False, build simple CNN from scratch

    Returns:
        Compiled Keras model
    """
    if use_transfer_learning:
        return _build_transfer_classifier()
    else:
        return _build_simple_classifier()


def _build_transfer_classifier():
    """EfficientNetB0-based classifier (recommended - better accuracy)"""

    # Load pre-trained EfficientNetB0 WITHOUT the top classification layer
    base_model = EfficientNetB0(
        weights="imagenet",       # Pre-trained on 1.2M ImageNet images
        include_top=False,        # Remove final classification layer
        input_shape=IMAGE_SHAPE
    )

    # Freeze base model weights initially
    # We will unfreeze later for fine-tuning
    base_model.trainable = False

    # Build full model
    inputs = keras.Input(shape=IMAGE_SHAPE, name="image_input")

    # Preprocessing: EfficientNet expects [0, 255] not [0, 1]
    x = layers.Rescaling(255.0)(inputs)

    # Base model feature extraction
    x = base_model(x, training=False)

    # Global pooling: (7, 7, 1280) → (1280,)
    x = layers.GlobalAveragePooling2D(name="gap")(x)

    # Dropout for regularisation
    x = layers.Dropout(0.3, name="dropout_1")(x)

    # Dense layer
    x = layers.Dense(256, activation="relu", name="dense_1")(x)
    x = layers.BatchNormalization(name="bn_1")(x)
    x = layers.Dropout(0.2, name="dropout_2")(x)

    # Output: 3 document types
    outputs = layers.Dense(
        NUM_DOC_TYPES,
        activation="softmax",
        name="doc_type_output"
    )(x)

    model = keras.Model(inputs, outputs, name="document_classifier")

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model


def _build_simple_classifier():
    """Simple CNN from scratch (fallback if transfer learning fails)"""

    model = keras.Sequential([
        keras.Input(shape=IMAGE_SHAPE),

        # Block 1
        layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),

        # Block 2
        layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),

        # Block 3
        layers.Conv2D(128, (3, 3), activation="relu", padding="same"),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),

        # Block 4
        layers.Conv2D(256, (3, 3), activation="relu", padding="same"),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),

        # Classification head
        layers.GlobalAveragePooling2D(),
        layers.Dense(256, activation="relu"),
        layers.Dropout(0.4),
        layers.Dense(NUM_DOC_TYPES, activation="softmax", name="doc_type_output")

    ], name="simple_document_classifier")

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model


# ── MODEL 2: AUTHENTICITY DETECTOR ────────────────────────────────────────────

def build_authenticity_detector(use_transfer_learning=True):
    """
    Build CNN to detect Genuine (0) vs Fake (1) documents.

    Args:
        use_transfer_learning: If True, use EfficientNetB0 backbone

    Returns:
        Compiled Keras model
    """
    if use_transfer_learning:
        return _build_transfer_detector()
    else:
        return _build_simple_detector()


def _build_transfer_detector():
    """EfficientNetB0-based authenticity detector"""

    base_model = EfficientNetB0(
        weights="imagenet",
        include_top=False,
        input_shape=IMAGE_SHAPE
    )
    base_model.trainable = False

    inputs = keras.Input(shape=IMAGE_SHAPE, name="image_input")
    x = layers.Rescaling(255.0)(inputs)
    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dropout(0.4, name="dropout_1")(x)
    x = layers.Dense(256, activation="relu", name="dense_1")(x)
    x = layers.BatchNormalization(name="bn_1")(x)
    x = layers.Dropout(0.3, name="dropout_2")(x)
    x = layers.Dense(128, activation="relu", name="dense_2")(x)

    # Binary output: Genuine vs Fake
    outputs = layers.Dense(
        1,
        activation="sigmoid",
        name="auth_output"
    )(x)

    model = keras.Model(inputs, outputs, name="authenticity_detector")

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
        metrics=["accuracy",
                 keras.metrics.Precision(name="precision"),
                 keras.metrics.Recall(name="recall")]
    )

    return model


def _build_simple_detector():
    """Simple CNN authenticity detector (fallback)"""

    model = keras.Sequential([
        keras.Input(shape=IMAGE_SHAPE),

        layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(128, (3, 3), activation="relu", padding="same"),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(256, (3, 3), activation="relu", padding="same"),
        layers.BatchNormalization(),
        layers.GlobalAveragePooling2D(),

        layers.Dense(256, activation="relu"),
        layers.Dropout(0.5),
        layers.Dense(1, activation="sigmoid", name="auth_output")

    ], name="simple_authenticity_detector")

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
        metrics=["accuracy",
                 keras.metrics.Precision(name="precision"),
                 keras.metrics.Recall(name="recall")]
    )

    return model


# ── FINE-TUNING HELPER ─────────────────────────────────────────────────────────

def unfreeze_top_layers(model, num_layers=20):
    """
    Unfreeze the top N layers of the base model for fine-tuning.
    Call this after initial training converges.

    Args:
        model: trained Keras model
        num_layers: number of layers from the top to unfreeze
    """
    # Find the EfficientNet base model inside
    for layer in model.layers:
        if "efficientnet" in layer.name.lower():
            base = layer
            base.trainable = True
            # Freeze all except top N layers
            for l in base.layers[:-num_layers]:
                l.trainable = False
            print(f"✅ Unfroze top {num_layers} layers of {base.name}")
            break

    # Recompile with lower learning rate for fine-tuning
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-5),
        loss=model.loss,
        metrics=model.metrics
    )
    return model


# ── QUICK TEST ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import numpy as np

    print("=" * 60)
    print("CNN ARCHITECTURE TEST")
    print("=" * 60)

    # Test document classifier
    print("\n1. Building Document Classifier...")
    classifier = build_document_classifier(use_transfer_learning=False)
    classifier.summary()

    # Test with dummy data
    dummy = np.random.rand(2, 224, 224, 3).astype("float32")
    pred = classifier.predict(dummy, verbose=0)
    print(f"\nTest prediction shape: {pred.shape}")
    print(f"Probabilities sum to 1: {pred.sum(axis=1)}")

    # Test authenticity detector
    print("\n2. Building Authenticity Detector...")
    detector = build_authenticity_detector(use_transfer_learning=False)
    detector.summary()

    pred2 = detector.predict(dummy, verbose=0)
    print(f"\nTest prediction shape: {pred2.shape}")
    print(f"Sample output (0=Genuine, 1=Fake): {pred2.flatten()}")

    print("\n✅ Both models built successfully!")
    print("=" * 60)