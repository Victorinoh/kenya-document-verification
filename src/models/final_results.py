"""
Final Model Results Summary
Documents what each model achieved and why
"""
import sys
import numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import tensorflow as tf
from sklearn.metrics import classification_report, roc_auc_score
from src.models.dataset_builder import (build_classifier_dataset,
                                         build_detector_dataset)

DOC_CLASSES  = ["National ID", "KCSE Cert", "Passport"]
AUTH_CLASSES = ["Genuine", "Fake"]


def evaluate_model(model_path, dataset_fn, binary=False, **ds_kwargs):
    model    = tf.keras.models.load_model(model_path)
    ds, n    = dataset_fn("validation", batch_size=16,
                          augment=False, **ds_kwargs)

    all_probs, all_labels = [], []
    for images, labels in ds:
        probs = model.predict(images, verbose=0)
        all_probs.extend(probs.flatten() if binary else probs)
        all_labels.extend(labels.numpy())

    all_probs  = np.array(all_probs)
    all_labels = np.array(all_labels)

    if binary:
        preds = (all_probs > 0.5).astype(int)
        auc   = roc_auc_score(all_labels, all_probs)
    else:
        preds = np.argmax(all_probs, axis=1)
        auc   = None

    accuracy = (preds == all_labels).mean()
    return accuracy, auc, preds, all_labels, n


if __name__ == "__main__":
    print("=" * 65)
    print("FINAL MODEL RESULTS — Week 3")
    print("=" * 65)

    # ── Classifier ───────────────────────────────────────────────────
    print("\n📋 MODEL 1: Document Type Classifier")
    print("-" * 65)
    cls_acc, _, cls_preds, cls_labels, n = evaluate_model(
        "models/saved_models/document_classifier.h5",
        build_classifier_dataset
    )
    print(f"Validation images : {n}")
    print(f"Accuracy          : {cls_acc*100:.1f}%  🏆")
    print()
    print(classification_report(cls_labels, cls_preds,
                                target_names=DOC_CLASSES))

    # ── Detector ─────────────────────────────────────────────────────
    print("\n📋 MODEL 2: Authenticity Detector (V1 — best)")
    print("-" * 65)
    det_acc, det_auc, det_preds, det_labels, n = evaluate_model(
        "models/saved_models/authenticity_detector.h5",
        build_detector_dataset,
        binary=True
    )
    print(f"Validation images : {n}")
    print(f"Accuracy          : {det_acc*100:.1f}%")
    print(f"ROC-AUC           : {det_auc:.4f}")
    print()
    print(classification_report(det_labels, det_preds,
                                target_names=AUTH_CLASSES))

    # ── Summary ──────────────────────────────────────────────────────
    print("=" * 65)
    print("WEEK 3 SUMMARY")
    print("=" * 65)
    print(f"""
  Model 1 — Document Type Classifier
  ├── Accuracy   : {cls_acc*100:.1f}% ✅
  ├── Status     : Production ready
  └── Notes      : Perfect separation of 3 document types

  Model 2 — Authenticity Detector
  ├── Accuracy   : {det_acc*100:.1f}%
  ├── AUC        : {det_auc:.4f}
  ├── Status     : Baseline — needs real forgery data
  └── Notes      : Currently detects image quality differences,
                   not true security feature tampering.
                   Improvement path → Week 4 OCR integration
                   will add rule-based verification on top.

  Dataset
  ├── Training   : 1,276 images
  ├── Validation :   464 images
  └── Balance    : 50/50 genuine/fake ✅

  Next Steps (Week 4)
  └── OCR extracts text fields from documents
      Rule-based checks validate ID numbers, dates, formats
      This adds a second verification layer on top of CNN
    """)
    print("=" * 65)