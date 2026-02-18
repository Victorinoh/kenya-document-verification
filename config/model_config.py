"""
Model training configuration
Adjust these settings based on your hardware
"""

# ── IMAGE SETTINGS ─────────────────────────────────────────────────────
IMAGE_SIZE  = (224, 224)
IMAGE_SHAPE = (224, 224, 3)

# ── TRAINING SETTINGS ──────────────────────────────────────────────────
BATCH_SIZE      = 16     # Reduce to 8 if you get memory errors
EPOCHS_INITIAL  = 20     # Initial training (frozen base)
EPOCHS_FINETUNE = 10     # Fine-tuning (unfrozen top layers)
LEARNING_RATE   = 1e-3
FINETUNE_LR     = 1e-5

# ── PATHS ──────────────────────────────────────────────────────────────
TRAIN_GENUINE   = "data/augmented/train"
TRAIN_FAKE      = "data/augmented/train/fake"
VAL_GENUINE     = "data/augmented/validation"
VAL_FAKE        = "data/augmented/validation/fake"

MODEL_SAVE_DIR  = "models/saved_models"
CHECKPOINT_DIR  = "models/checkpoints"

CLASSIFIER_PATH = "models/saved_models/document_classifier.h5"
DETECTOR_PATH   = "models/saved_models/authenticity_detector.h5"

# ── LABELS ─────────────────────────────────────────────────────────────
DOC_CLASSES  = {0: "National ID", 1: "KCSE Certificate", 2: "Passport"}
AUTH_CLASSES = {0: "Genuine",     1: "Fake"}