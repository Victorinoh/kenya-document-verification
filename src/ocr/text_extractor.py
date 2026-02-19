"""
OCR Text Extractor
Extracts text from document images using Tesseract
"""
import cv2
import numpy as np
import pytesseract
from PIL import Image
from pathlib import Path

# Point to Tesseract installation
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'


# ── IMAGE PREPROCESSING FOR OCR ───────────────────────────────────────────────

def preprocess_for_ocr(image_path):
    """
    Preprocess a document image to improve OCR accuracy.
    Returns multiple versions — OCR will try each one.

    Steps:
      1. Grayscale conversion
      2. Denoising
      3. Adaptive thresholding (handles uneven lighting)
      4. Deskewing (straightens tilted scans)
    """
    # Load image
    img = cv2.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(f"Cannot load image: {image_path}")

    # Resize if too small — OCR needs at least 300 DPI equivalent
    h, w = img.shape[:2]
    if w < 1000:
        scale = 1000 / w
        img = cv2.resize(img, None, fx=scale, fy=scale,
                         interpolation=cv2.INTER_CUBIC)

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Denoise
    denoised = cv2.fastNlMeansDenoising(gray, h=10)

    # Version 1: Adaptive threshold (best for uneven lighting)
    adaptive = cv2.adaptiveThreshold(
        denoised, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 11, 2
    )

    # Version 2: Otsu threshold (best for clean scans)
    _, otsu = cv2.threshold(
        denoised, 0, 255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    # Version 3: Simple grayscale (sometimes best as-is)
    return {
        "adaptive": adaptive,
        "otsu":     otsu,
        "gray":     denoised,
        "original": img
    }


def deskew(image):
    """Straighten a tilted document image"""
    coords = np.column_stack(np.where(image > 0))
    if len(coords) == 0:
        return image
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = 90 + angle
    if abs(angle) < 0.5:
        return image   # Already straight enough

    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(
        image, M, (w, h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE
    )


# ── OCR EXTRACTION ────────────────────────────────────────────────────────────

def extract_text(image_path, doc_type=None):
    """
    Extract all text from a document image.

    Args:
        image_path: path to document image
        doc_type:   'national_id' | 'kcse_certificate' | 'passport' | None

    Returns:
        dict with:
          raw_text    — full OCR output
          lines       — list of non-empty lines
          confidence  — average OCR confidence (0-100)
          method      — which preprocessing worked best
    """
    image_path = Path(image_path)
    if not image_path.exists():
        return {"error": f"File not found: {image_path}"}

    # Get preprocessed versions
    versions = preprocess_for_ocr(image_path)

    # Tesseract config — optimized for document text
    configs = {
        "default":    "--oem 3 --psm 6",   # Assume uniform block of text
        "sparse":     "--oem 3 --psm 11",  # Sparse text (IDs with scattered fields)
        "single_col": "--oem 3 --psm 4",   # Single column
    }

    # Choose config based on document type
    if doc_type == "national_id":
        primary_config = configs["sparse"]
    elif doc_type == "passport":
        primary_config = configs["single_col"]
    else:
        primary_config = configs["default"]

    best_text       = ""
    best_confidence = 0
    best_method     = ""

    # Try each preprocessed version and keep the best result
    for method_name, processed in versions.items():
        if method_name == "original":
            continue   # Skip color original for text extraction

        try:
            # Get detailed OCR data including confidence scores
            data = pytesseract.image_to_data(
                processed,
                config=primary_config,
                output_type=pytesseract.Output.DICT
            )

            # Calculate average confidence (ignore -1 values)
            confidences = [
                int(c) for c in data["conf"]
                if str(c).isdigit() and int(c) >= 0
            ]
            avg_conf = np.mean(confidences) if confidences else 0

            # Get raw text
            text = pytesseract.image_to_string(
                processed, config=primary_config
            )

            if avg_conf > best_confidence:
                best_confidence = avg_conf
                best_text       = text
                best_method     = method_name

        except Exception as e:
            continue

    # Clean up extracted text
    lines = [
        line.strip()
        for line in best_text.split("\n")
        if line.strip() and len(line.strip()) > 1
    ]

    return {
        "raw_text":   best_text,
        "lines":      lines,
        "confidence": round(best_confidence, 1),
        "method":     best_method,
        "word_count": len(best_text.split()),
        "line_count": len(lines)
    }


def extract_text_regions(image_path):
    """
    Extract text with bounding box locations.
    Useful for knowing WHERE on the document each word appears.

    Returns list of dicts: {text, x, y, width, height, confidence}
    """
    image_path = Path(image_path)
    versions   = preprocess_for_ocr(image_path)
    processed  = versions["adaptive"]

    data = pytesseract.image_to_data(
        processed,
        config="--oem 3 --psm 6",
        output_type=pytesseract.Output.DICT
    )

    regions = []
    for i in range(len(data["text"])):
        text = data["text"][i].strip()
        conf = int(data["conf"][i]) if str(data["conf"][i]).isdigit() else -1

        if text and conf > 30:   # Only keep confident detections
            regions.append({
                "text":       text,
                "x":          data["left"][i],
                "y":          data["top"][i],
                "width":      data["width"][i],
                "height":     data["height"][i],
                "confidence": conf
            })

    return regions


# ── QUICK TEST ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    print("=" * 60)
    print("OCR TEXT EXTRACTOR TEST")
    print("=" * 60)

    # Find a sample image to test on
    test_dirs = [
        "data/raw/national_ids/genuine",
        "data/raw/kcse_certificates/genuine",
        "data/raw/passports/genuine",
        "data/augmented/train/national_ids",
    ]

    test_image = None
    for d in test_dirs:
        folder = Path(d)
        if folder.exists():
            images = (list(folder.glob("*.jpg")) +
                      list(folder.glob("*.JPG")) +
                      list(folder.glob("*.png")))
            if images:
                test_image = images[0]
                break

    if not test_image:
        print("❌  No test images found.")
        print("    Place an image in data/raw/national_ids/genuine/")
        sys.exit(1)

    print(f"\nTest image: {test_image}")
    print("-" * 60)

    # Run OCR
    result = extract_text(str(test_image), doc_type="national_id")

    if "error" in result:
        print(f"❌  Error: {result['error']}")
    else:
        print(f"Method used    : {result['method']}")
        print(f"OCR confidence : {result['confidence']}%")
        print(f"Words found    : {result['word_count']}")
        print(f"Lines found    : {result['line_count']}")
        print(f"\nExtracted text:")
        print("-" * 40)
        for line in result["lines"]:
            print(f"  {line}")

        # Test region extraction
        print("\nText regions (first 5):")
        regions = extract_text_regions(str(test_image))
        for r in regions[:5]:
            print(f"  '{r['text']}' at ({r['x']},{r['y']}) "
                  f"conf={r['confidence']}%")

    print("\n✅ OCR extractor working!")
    print("=" * 60)