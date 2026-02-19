"""
Model Evaluation Script
Generates confusion matrix, classification report and accuracy plots
"""
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import seaborn as sns
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import tensorflow as tf
from sklearn.metrics import (confusion_matrix, classification_report,
                             roc_auc_score, roc_curve)
from src.models.dataset_builder import build_classifier_dataset, build_detector_dataset

os.makedirs("models/evaluation", exist_ok=True)

DOC_CLASSES  = ["National ID", "KCSE Cert", "Passport"]
AUTH_CLASSES = ["Genuine", "Fake"]


# ── HELPERS ───────────────────────────────────────────────────────────────────

def get_predictions(model, dataset, binary=False):
    """Run model on dataset and collect predictions + true labels"""
    all_preds, all_labels = [], []

    for images, labels in dataset:
        preds = model.predict(images, verbose=0)
        if binary:
            preds = (preds > 0.5).astype(int).flatten()
        else:
            preds = np.argmax(preds, axis=1)
        all_preds.extend(preds)
        all_labels.extend(labels.numpy())

    return np.array(all_preds), np.array(all_labels)


def plot_confusion_matrix(cm, class_names, title, save_path, color="Blues"):
    """Plot and save a confusion matrix"""
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap=color,
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax, linewidths=0.5
    )
    ax.set_title(title, fontsize=14, fontweight="bold", pad=15)
    ax.set_ylabel("True Label",      fontsize=12)
    ax.set_xlabel("Predicted Label", fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


def plot_training_history(log_path, model_name, save_path):
    """Plot accuracy and loss curves from training CSV log"""
    import pandas as pd

    if not Path(log_path).exists():
        print(f"  ⚠️  Log not found: {log_path}")
        return

    df = pd.read_csv(log_path)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"{model_name} — Training History",
                 fontsize=14, fontweight="bold")

    # Accuracy
    axes[0].plot(df["epoch"] + 1, df["accuracy"],     label="Train", linewidth=2, color="#2196F3")
    axes[0].plot(df["epoch"] + 1, df["val_accuracy"], label="Val",   linewidth=2, color="#FF5722", linestyle="--")
    axes[0].set_title("Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim(0, 1)

    # Loss
    axes[1].plot(df["epoch"] + 1, df["loss"],     label="Train", linewidth=2, color="#2196F3")
    axes[1].plot(df["epoch"] + 1, df["val_loss"], label="Val",   linewidth=2, color="#FF5722", linestyle="--")
    axes[1].set_title("Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


# ── EVALUATE CLASSIFIER ───────────────────────────────────────────────────────

def evaluate_classifier():
    print("\n" + "=" * 60)
    print("EVALUATING: DOCUMENT TYPE CLASSIFIER")
    print("=" * 60)

    # Load model
    model_path = "models/saved_models/document_classifier.h5"
    if not Path(model_path).exists():
        print(f"❌  Model not found: {model_path}")
        return
    model = tf.keras.models.load_model(model_path)
    print(f"✅  Model loaded: {model_path}")

    # Load validation data
    val_ds, n_val = build_classifier_dataset("validation", batch_size=16, augment=False)
    print(f"✅  Validation set: {n_val} images")

    # Get predictions
    print("Running predictions...")
    preds, labels = get_predictions(model, val_ds, binary=False)

    # Accuracy
    accuracy = (preds == labels).mean()
    print(f"\n📊 Validation Accuracy: {accuracy*100:.1f}%")

    # Classification report
    print("\nClassification Report:")
    print(classification_report(labels, preds, target_names=DOC_CLASSES))

    # Confusion matrix
    cm = confusion_matrix(labels, preds)
    plot_confusion_matrix(
        cm, DOC_CLASSES,
        "Document Type Classifier — Confusion Matrix",
        "models/evaluation/classifier_confusion_matrix.png"
    )

    # Training history plot
    plot_training_history(
        "models/training_logs/document_classifier_log.csv",
        "Document Classifier",
        "models/evaluation/classifier_training_history.png"
    )

    return accuracy


# ── EVALUATE DETECTOR ─────────────────────────────────────────────────────────

def evaluate_detector():
    print("\n" + "=" * 60)
    print("EVALUATING: AUTHENTICITY DETECTOR")
    print("=" * 60)

    # Load model
    model_path = "models/saved_models/authenticity_detector.h5"
    if not Path(model_path).exists():
        print(f"❌  Model not found: {model_path}")
        return
    model = tf.keras.models.load_model(model_path)
    print(f"✅  Model loaded: {model_path}")

    # Load validation data
    val_ds, n_val = build_detector_dataset("validation", batch_size=16, augment=False)
    print(f"✅  Validation set: {n_val} images")

    # Get predictions and raw probabilities for ROC curve
    print("Running predictions...")
    all_probs, all_labels = [], []
    for images, labels in val_ds:
        probs = model.predict(images, verbose=0).flatten()
        all_probs.extend(probs)
        all_labels.extend(labels.numpy())

    all_probs  = np.array(all_probs)
    all_labels = np.array(all_labels)
    preds      = (all_probs > 0.5).astype(int)

    # Metrics
    accuracy = (preds == all_labels).mean()
    auc      = roc_auc_score(all_labels, all_probs)
    print(f"\n📊 Validation Accuracy : {accuracy*100:.1f}%")
    print(f"📊 ROC-AUC Score       : {auc:.4f}")

    # Classification report
    print("\nClassification Report:")
    print(classification_report(all_labels, preds, target_names=AUTH_CLASSES))

    # Confusion matrix
    cm = confusion_matrix(all_labels, preds)
    plot_confusion_matrix(
        cm, AUTH_CLASSES,
        "Authenticity Detector — Confusion Matrix",
        "models/evaluation/detector_confusion_matrix.png",
        color="Oranges"
    )

    # ROC Curve
    fpr, tpr, _ = roc_curve(all_labels, all_probs)
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(fpr, tpr, color="#FF5722", linewidth=2,
            label=f"ROC Curve (AUC = {auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Random Guess")
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate",  fontsize=12)
    ax.set_title("Authenticity Detector — ROC Curve",
                 fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("models/evaluation/detector_roc_curve.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved: models/evaluation/detector_roc_curve.png")

    # Training history plot
    plot_training_history(
        "models/training_logs/authenticity_detector_log.csv",
        "Authenticity Detector",
        "models/evaluation/detector_training_history.png"
    )

    return accuracy, auc


# ── MAIN ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("MODEL EVALUATION")
    print("=" * 60)

    cls_accuracy         = evaluate_classifier()
    det_accuracy, det_auc = evaluate_detector()

    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    print(f"  Document Classifier  accuracy : {cls_accuracy*100:.1f}%")
    print(f"  Authenticity Detector accuracy : {det_accuracy*100:.1f}%")
    print(f"  Authenticity Detector AUC      : {det_auc:.4f}")
    print("\nEvaluation charts saved to: models/evaluation/")
    print("  - classifier_confusion_matrix.png")
    print("  - classifier_training_history.png")
    print("  - detector_confusion_matrix.png")
    print("  - detector_roc_curve.png")
    print("  - detector_training_history.png")
    print("=" * 60)