"""
Confidence Scorer
Combines CNN model score + OCR validation score into one final verdict.

Three-layer scoring:
  Layer 1 — CNN Classifier    : What document type is this?
  Layer 2 — CNN Detector      : Does it look genuine visually?
  Layer 3 — OCR Validator     : Do the text fields check out?

Final score = weighted combination of all three layers.
"""
import sys
import numpy as np
from pathlib import Path
sys.path.insert(0, ".")


# ── WEIGHTS ───────────────────────────────────────────────────────────────────

# How much each layer contributes to the final score
WEIGHTS = {
    "cnn_classifier": 0.20,   # Document type confirmation
    "cnn_detector":   0.45,   # Visual authenticity (main signal)
    "ocr_validator":  0.35,   # Text field validation
}

# Verdict thresholds
THRESHOLD_GENUINE  = 65    # Score >= 65 → GENUINE
THRESHOLD_UNCERTAIN = 45   # Score >= 45 → UNCERTAIN
                            # Score <  45 → FAKE

DOC_TYPE_NAMES = {
    0: "national_id",
    1: "kcse_certificate",
    2: "passport"
}

DOC_DISPLAY_NAMES = {
    "national_id":      "National ID",
    "kcse_certificate": "KCSE Certificate",
    "passport":         "Passport"
}


# ── SCORER CLASS ──────────────────────────────────────────────────────────────

class ConfidenceScorer:
    """
    Combines CNN and OCR scores into a single verification verdict.
    """

    def __init__(self,
                 classifier_path="models/saved_models/document_classifier.h5",
                 detector_path="models/saved_models/authenticity_detector.h5"):
        self.classifier     = None
        self.detector       = None
        self.classifier_path = classifier_path
        self.detector_path   = detector_path
        self._load_models()

    def _load_models(self):
        """Load CNN models (lazy — only load when needed)"""
        try:
            import tensorflow as tf
            if Path(self.classifier_path).exists():
                self.classifier = tf.keras.models.load_model(
                    self.classifier_path)
                print(f"✅  Classifier loaded")
            else:
                print(f"⚠️   Classifier not found: {self.classifier_path}")

            if Path(self.detector_path).exists():
                self.detector = tf.keras.models.load_model(
                    self.detector_path)
                print(f"✅  Detector loaded")
            else:
                print(f"⚠️   Detector not found: {self.detector_path}")

        except Exception as e:
            print(f"⚠️   Could not load models: {e}")

    def _preprocess_image(self, image_path):
        """Load and preprocess image for CNN input"""
        import cv2
        img = cv2.imread(str(image_path))
        if img is None:
            return None
        img = cv2.resize(img, (224, 224))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype("float32") / 255.0
        return np.expand_dims(img, axis=0)   # Shape: (1, 224, 224, 3)

    # ── CNN SCORING ───────────────────────────────────────────────────────────

    def get_classifier_score(self, image_path, expected_doc_type=None):
        """
        Run document type classifier.

        Returns:
            dict with predicted_type, confidence, matches_expected
        """
        if self.classifier is None:
            return {
                "predicted_type":    "unknown",
                "confidence":        0.0,
                "matches_expected":  False,
                "score":             0.0,
                "available":         False
            }

        img = self._preprocess_image(image_path)
        if img is None:
            return {"score": 0.0, "available": False}

        probs          = self.classifier.predict(img, verbose=0)[0]
        predicted_idx  = int(np.argmax(probs))
        confidence     = float(probs[predicted_idx])
        predicted_type = DOC_TYPE_NAMES.get(predicted_idx, "unknown")

        # Check if predicted type matches what we expect
        matches = True
        if expected_doc_type:
            # Normalise names for comparison
            exp = expected_doc_type.replace("_", "").lower()
            pred = predicted_type.replace("_", "").lower()
            matches = exp in pred or pred in exp

        # Score: full confidence if type matches, half if mismatch
        score = confidence * 100 if matches else confidence * 50

        return {
            "predicted_type":   predicted_type,
            "display_name":     DOC_DISPLAY_NAMES.get(predicted_type, predicted_type),
            "confidence":       round(confidence * 100, 1),
            "matches_expected": matches,
            "score":            round(score, 1),
            "available":        True,
            "all_probs": {
                DOC_TYPE_NAMES[i]: round(float(p) * 100, 1)
                for i, p in enumerate(probs)
            }
        }

    def get_detector_score(self, image_path):
        """
        Run authenticity detector.

        Returns:
            dict with genuine_probability, fake_probability, score
        """
        if self.detector is None:
            return {
                "genuine_probability": 50.0,
                "fake_probability":    50.0,
                "score":               50.0,
                "available":           False
            }

        img = self._preprocess_image(image_path)
        if img is None:
            return {"score": 50.0, "available": False}

        # Detector outputs probability of being FAKE (sigmoid)
        fake_prob    = float(self.detector.predict(img, verbose=0)[0][0])
        genuine_prob = 1.0 - fake_prob

        # Score: 100 = definitely genuine, 0 = definitely fake
        score = genuine_prob * 100

        return {
            "genuine_probability": round(genuine_prob * 100, 1),
            "fake_probability":    round(fake_prob * 100, 1),
            "score":               round(score, 1),
            "available":           True
        }

    def get_ocr_score(self, validation_report):
        """
        Convert OCR validation report to a 0-100 score.

        Args:
            validation_report: dict from DocumentValidator.validate()

        Returns:
            dict with score and summary
        """
        if not validation_report or "error" in validation_report:
            return {
                "score":           50.0,
                "available":       False,
                "checks_passed":   0,
                "checks_total":    0
            }

        overall_score  = validation_report.get("overall_score", 0)
        checks_passed  = validation_report.get("checks_passed", 0)
        checks_total   = validation_report.get("checks_total", 1)
        critical_passed = validation_report.get("critical_passed", 0)
        critical_total  = validation_report.get("critical_total", 1)

        # Penalty if critical checks failed
        if critical_total > 0 and critical_passed < critical_total:
            # Missing critical fields — but could be due to blur/anonymization
            # Use a softer penalty so blur doesn't completely destroy the score
            penalty_factor = critical_passed / critical_total
            adjusted_score = overall_score * (0.4 + 0.6 * penalty_factor)
        else:
            adjusted_score = overall_score

        return {
            "score":            round(adjusted_score, 1),
            "raw_score":        overall_score,
            "checks_passed":    checks_passed,
            "checks_total":     checks_total,
            "critical_passed":  critical_passed,
            "critical_total":   critical_total,
            "available":        True
        }

    # ── COMBINED SCORING ──────────────────────────────────────────────────────

    def score(self, image_path, doc_type, validation_report=None):
        """
        Compute the final combined verification score.

        Args:
            image_path:         path to document image
            doc_type:           'national_id' | 'kcse_certificate' | 'passport'
            validation_report:  dict from DocumentValidator.validate() (optional)

        Returns:
            dict with full scoring breakdown and final verdict
        """
        image_path = Path(image_path)

        # Layer 1 — CNN Classifier
        classifier_result = self.get_classifier_score(image_path, doc_type)

        # Layer 2 — CNN Detector
        detector_result = self.get_detector_score(image_path)

        # Layer 3 — OCR Validator
        ocr_result = self.get_ocr_score(validation_report)

        # ── Weighted Final Score ─────────────────────────────────────────────
        w = WEIGHTS

        # If OCR not available, redistribute its weight to detector
        if not ocr_result["available"]:
            w = {
                "cnn_classifier": 0.25,
                "cnn_detector":   0.75,
                "ocr_validator":  0.00
            }

        final_score = (
            classifier_result.get("score", 50) * w["cnn_classifier"] +
            detector_result.get("score",    50) * w["cnn_detector"]   +
            ocr_result.get("score",         50) * w["ocr_validator"]
        )
        final_score = round(final_score, 1)

        # ── Verdict ──────────────────────────────────────────────────────────
        if final_score >= THRESHOLD_GENUINE:
            verdict      = "GENUINE"
            verdict_icon = "✅"
            confidence   = "HIGH" if final_score >= 80 else "MEDIUM"
        elif final_score >= THRESHOLD_UNCERTAIN:
            verdict      = "UNCERTAIN"
            verdict_icon = "⚠️"
            confidence   = "LOW"
        else:
            verdict      = "FAKE"
            verdict_icon = "❌"
            confidence   = "HIGH" if final_score < 30 else "MEDIUM"

        return {
            "image":        str(image_path),
            "doc_type":     doc_type,
            "display_name": DOC_DISPLAY_NAMES.get(doc_type, doc_type),
            "final_score":  final_score,
            "verdict":      verdict,
            "verdict_icon": verdict_icon,
            "confidence":   confidence,
            "layers": {
                "classifier": classifier_result,
                "detector":   detector_result,
                "ocr":        ocr_result
            },
            "weights": w
        }

    def print_result(self, result):
        """Pretty print the full scoring result"""
        print(f"\n{'='*60}")
        print(f"  DOCUMENT VERIFICATION RESULT")
        print(f"{'='*60}")
        print(f"  Document   : {result['display_name']}")
        print(f"  Image      : {Path(result['image']).name}")
        print(f"  Final Score: {result['final_score']}%")
        print(f"  Verdict    : {result['verdict_icon']}  "
              f"{result['verdict']}  ({result['confidence']} confidence)")

        print(f"\n  Score Breakdown:")
        print(f"  {'─'*50}")

        # Classifier
        cls = result["layers"]["classifier"]
        w   = result["weights"]
        if cls.get("available"):
            match = "✅" if cls["matches_expected"] else "⚠️"
            print(f"  Layer 1 — CNN Classifier  "
                  f"(weight {w['cnn_classifier']*100:.0f}%)")
            print(f"    {match} Predicted: {cls['display_name']}  "
                  f"({cls['confidence']}% confident)")
            print(f"    Layer score: {cls['score']}%")
        else:
            print(f"  Layer 1 — CNN Classifier  ⚠️  not available")

        print()

        # Detector
        det = result["layers"]["detector"]
        if det.get("available"):
            icon = "✅" if det["genuine_probability"] > 50 else "❌"
            print(f"  Layer 2 — CNN Detector    "
                  f"(weight {w['cnn_detector']*100:.0f}%)")
            print(f"    {icon} Genuine: {det['genuine_probability']}%  |  "
                  f"Fake: {det['fake_probability']}%")
            print(f"    Layer score: {det['score']}%")
        else:
            print(f"  Layer 2 — CNN Detector    ⚠️  not available")

        print()

        # OCR
        ocr = result["layers"]["ocr"]
        if ocr.get("available"):
            print(f"  Layer 3 — OCR Validator   "
                  f"(weight {w['ocr_validator']*100:.0f}%)")
            print(f"    Checks: {ocr['checks_passed']}/{ocr['checks_total']} passed  |  "
                  f"Critical: {ocr['critical_passed']}/{ocr['critical_total']}")
            print(f"    Layer score: {ocr['score']}%")
        else:
            print(f"  Layer 3 — OCR Validator   ⚠️  not available")

        print(f"\n{'='*60}")


# ── QUICK TEST ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from src.ocr.text_extractor import extract_text
    from src.ocr.field_parser   import FieldParser
    from src.ocr.validators     import DocumentValidator

    print("=" * 60)
    print("CONFIDENCE SCORER TEST")
    print("=" * 60)

    scorer    = ConfidenceScorer()
    parser    = FieldParser()
    validator = DocumentValidator()

    test_cases = [
        ("data/raw/national_ids/genuine",      "national_id"),
        ("data/raw/kcse_certificates/genuine",  "kcse_certificate"),
        ("data/raw/passports/genuine",          "passport"),
    ]

    for folder, doc_type in test_cases:
        path = Path(folder)
        if not path.exists():
            continue
        images = (list(path.glob("*.jpg")) +
                  list(path.glob("*.JPG")) +
                  list(path.glob("*.png")))
        if not images:
            continue

        image_path = images[0]
        print(f"\nTesting: {image_path.name} ({doc_type})")

        # Run full pipeline
        ocr_result        = extract_text(str(image_path), doc_type=doc_type)
        fields            = parser.parse(ocr_result, doc_type)
        validation_report = validator.validate(fields, doc_type, ocr_result)
        result            = scorer.score(image_path, doc_type, validation_report)

        scorer.print_result(result)

    print("\n✅ Confidence scorer working!")