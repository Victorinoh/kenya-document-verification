"""
End-to-End Document Verification Pipeline
Single entry point: give it an image → get back a verdict.

Usage:
    from src.pipeline import verify_document
    result = verify_document("path/to/document.jpg", doc_type="national_id")
    print(result["verdict"])
"""
import sys
import time
from pathlib import Path
sys.path.insert(0, ".")

from src.ocr.text_extractor    import extract_text
from src.ocr.field_parser      import FieldParser
from src.ocr.validators        import DocumentValidator
from src.ocr.confidence_scorer import ConfidenceScorer


# ── SINGLETON INSTANCES (loaded once, reused) ─────────────────────────────────

_parser    = None
_validator = None
_scorer    = None


def _get_components():
    """Load pipeline components once and reuse across calls"""
    global _parser, _validator, _scorer
    if _parser is None:
        _parser    = FieldParser()
        _validator = DocumentValidator()
        _scorer    = ConfidenceScorer()
    return _parser, _validator, _scorer


def _collect_images(folder):
    """
    Collect unique images from a folder.
    Deduplicates case-insensitive names (fixes Windows .jpg/.JPG double-counting).
    """
    folder = Path(folder)
    seen   = set()
    images = []
    for ext in ["*.jpg", "*.JPG", "*.jpeg", "*.JPEG", "*.png", "*.PNG"]:
        for img in folder.glob(ext):
            key = img.name.lower()
            if key not in seen:
                seen.add(key)
                images.append(img)
    return sorted(images)


# ── MAIN PIPELINE FUNCTION ────────────────────────────────────────────────────

def verify_document(image_path, doc_type=None, verbose=False):
    """
    Verify a document image end-to-end.

    Args:
        image_path: path to document image (str or Path)
        doc_type:   'national_id' | 'kcse_certificate' | 'passport'
                    If None, auto-detected by CNN classifier
        verbose:    print detailed breakdown if True

    Returns:
        dict with:
          verdict       — 'GENUINE' | 'UNCERTAIN' | 'FAKE'
          verdict_icon  — '✅' | '⚠️' | '❌'
          final_score   — 0-100
          confidence    — 'HIGH' | 'MEDIUM' | 'LOW'
          doc_type      — detected or provided document type
          display_name  — human readable document name
          duration_ms   — processing time in milliseconds
          layers        — breakdown per scoring layer
          ocr           — extracted text and fields
          validation    — rule check results
    """
    start_time = time.time()
    image_path = Path(image_path)

    if not image_path.exists():
        return {
            "verdict":      "ERROR",
            "verdict_icon": "❌",
            "error":        f"Image not found: {image_path}",
            "final_score":  0,
            "duration_ms":  0
        }

    parser, validator, scorer = _get_components()

    # ── Step 1: Auto-detect document type if not provided ─────────────────────
    if doc_type is None:
        cls_result = scorer.get_classifier_score(image_path)
        doc_type   = cls_result.get("predicted_type", "national_id")
        if verbose:
            print(f"  Step 1: Auto-detected type → {doc_type} "
                  f"({cls_result.get('confidence', 0)}%)")
    else:
        if verbose:
            print(f"  Step 1: Document type provided → {doc_type}")

    # ── Step 2: Extract text via OCR ──────────────────────────────────────────
    if verbose:
        print(f"  Step 2: Running OCR...")
    ocr_result = extract_text(str(image_path), doc_type=doc_type)
    if verbose:
        print(f"          Confidence: {ocr_result.get('confidence', 0)}%  |  "
              f"Words: {ocr_result.get('word_count', 0)}")

    # ── Step 3: Parse fields from OCR text ────────────────────────────────────
    if verbose:
        print(f"  Step 3: Parsing fields...")
    fields           = parser.parse(ocr_result, doc_type)
    extraction_score = fields.get("extraction_score", {})
    if verbose:
        print(f"          Fields found: "
              f"{extraction_score.get('fields_found', 0)}/"
              f"{extraction_score.get('total_fields', 0)}")

    # ── Step 4: Validate fields against Kenyan document rules ─────────────────
    if verbose:
        print(f"  Step 4: Validating fields...")
    validation_report = validator.validate(fields, doc_type, ocr_result)
    if verbose:
        print(f"          Checks passed: "
              f"{validation_report.get('checks_passed', 0)}/"
              f"{validation_report.get('checks_total', 0)}")

    # ── Step 5: Compute final confidence score ────────────────────────────────
    if verbose:
        print(f"  Step 5: Computing final score...")
    score_result = scorer.score(image_path, doc_type, validation_report)

    duration_ms = round((time.time() - start_time) * 1000, 1)

    # ── Build final result dict ───────────────────────────────────────────────
    result = {
        # Core verdict
        "verdict":      score_result["verdict"],
        "verdict_icon": score_result["verdict_icon"],
        "final_score":  score_result["final_score"],
        "confidence":   score_result["confidence"],

        # Document info
        "doc_type":     doc_type,
        "display_name": score_result["display_name"],
        "image_path":   str(image_path),

        # Performance
        "duration_ms":  duration_ms,

        # Layer breakdown
        "layers": score_result["layers"],

        # OCR output
        "ocr": {
            "text_confidence":  ocr_result.get("confidence", 0),
            "word_count":       ocr_result.get("word_count", 0),
            "lines":            ocr_result.get("lines", []),
            "fields":           fields,
            "extraction_score": extraction_score
        },

        # Validation output
        "validation": {
            "verdict":       validation_report.get("verdict", "UNKNOWN"),
            "overall_score": validation_report.get("overall_score", 0),
            "checks_passed": validation_report.get("checks_passed", 0),
            "checks_total":  validation_report.get("checks_total", 0),
            "failed_checks": validation_report.get("failed_checks", [])
        }
    }

    if verbose:
        scorer.print_result(score_result)

    return result


# ── BATCH VERIFICATION ────────────────────────────────────────────────────────

def verify_batch(image_folder, doc_type=None, verbose=False):
    """
    Verify all images in a folder.

    Args:
        image_folder: path to folder containing images
        doc_type:     document type for all images (or None for auto-detect)
        verbose:      print per-image details

    Returns:
        tuple of (results list, verdicts summary dict)
    """
    folder = Path(image_folder)
    images = _collect_images(folder)

    if not images:
        print(f"⚠️  No images found in: {folder}")
        return [], {}

    print(f"\nProcessing {len(images)} images from: {folder}")
    print("-" * 55)

    results  = []
    verdicts = {"GENUINE": 0, "UNCERTAIN": 0, "FAKE": 0, "ERROR": 0}

    for i, img_path in enumerate(images, 1):
        result = verify_document(img_path, doc_type=doc_type, verbose=False)
        results.append(result)
        verdict = result.get("verdict", "ERROR")
        verdicts[verdict] = verdicts.get(verdict, 0) + 1

        icon  = result.get("verdict_icon", "❓")
        score = result.get("final_score", 0)
        ms    = result.get("duration_ms", 0)
        print(f"  [{i:3d}/{len(images)}] {icon} {score:5.1f}%  "
              f"{verdict:10}  ({ms:.0f}ms)  {img_path.name}")

    # ── Summary ───────────────────────────────────────────────────────────────
    total     = len(results)
    avg_score = sum(r.get("final_score", 0) for r in results) / total
    avg_ms    = sum(r.get("duration_ms",  0) for r in results) / total

    print(f"\n{'='*55}")
    print(f"BATCH SUMMARY — {total} images")
    print(f"{'='*55}")
    print(f"  ✅ Genuine   : {verdicts['GENUINE']:4d}  "
          f"({verdicts['GENUINE']/total*100:.1f}%)")
    print(f"  ⚠️  Uncertain : {verdicts['UNCERTAIN']:4d}  "
          f"({verdicts['UNCERTAIN']/total*100:.1f}%)")
    print(f"  ❌ Fake      : {verdicts['FAKE']:4d}  "
          f"({verdicts['FAKE']/total*100:.1f}%)")
    print(f"  Avg score  : {avg_score:.1f}%")
    print(f"  Avg time   : {avg_ms:.0f}ms per image")
    print(f"{'='*55}")

    return results, verdicts


# ── QUICK TEST ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("END-TO-END PIPELINE TEST")
    print("=" * 60)

    test_cases = [
        ("data/raw/national_ids/genuine",      "national_id"),
        ("data/raw/kcse_certificates/genuine",  "kcse_certificate"),
        ("data/raw/passports/genuine",          "passport"),
    ]

    # ── Single image tests ────────────────────────────────────────────────────
    print("\n--- Single Image Tests ---")
    for folder, doc_type in test_cases:
        path   = Path(folder)
        images = _collect_images(path)
        if not images:
            print(f"⚠️  No images found in {folder}")
            continue

        print(f"\n{'─'*60}")
        print(f"Testing {doc_type}: {images[0].name}")
        result = verify_document(images[0], doc_type=doc_type, verbose=True)
        print(f"\nFinal Answer: {result['verdict_icon']} {result['verdict']} "
              f"({result['final_score']}%) in {result['duration_ms']}ms")

    # ── Batch test ────────────────────────────────────────────────────────────
    print(f"\n{'─'*60}")
    print("--- Batch Test: National IDs ---")
    results, summary = verify_batch(
        "data/raw/national_ids/genuine",
        doc_type="national_id"
    )

    print(f"\n{'─'*60}")
    print("--- Batch Test: KCSE Certificates ---")
    results, summary = verify_batch(
        "data/raw/kcse_certificates/genuine",
        doc_type="kcse_certificate"
    )

    print(f"\n{'─'*60}")
    print("--- Batch Test: Passports ---")
    results, summary = verify_batch(
        "data/raw/passports/genuine",
        doc_type="passport"
    )

    print("\n✅ End-to-end pipeline working!")
    print("=" * 60)