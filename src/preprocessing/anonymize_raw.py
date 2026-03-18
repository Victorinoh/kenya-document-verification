import cv2
import os
import shutil
from pathlib import Path

# ── CONFIG ────────────────────────────────────────────────────────────────────
# Input — your original unanonymized images
RAW_DIRS = [
    'data/raw/national_ids/genuine',
    'data/raw/passports/genuine',
    'data/raw/kcse_certificates/genuine',
]

# Output — anonymized copies go here
ANON_BASE = 'data/raw/anonymized'

# ── FACE + ID NUMBER ANONYMIZER ───────────────────────────────────────────────
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

def anonymize_image(img):
    result = img.copy()
    h, w = result.shape[:2]

    # Detect and blur faces
    gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4, minSize=(20, 20))
    for (x, y, fw, fh) in faces:
        # Add padding around face
        pad = 10
        x1 = max(0, x - pad)
        y1 = max(0, y - pad)
        x2 = min(w, x + fw + pad)
        y2 = min(h, y + fh + pad)
        result[y1:y2, x1:x2] = cv2.GaussianBlur(
            result[y1:y2, x1:x2], (51, 51), 30
        )

    # Blur bottom third — ID numbers, dates, addresses
    result[int(h * 0.65):, :] = cv2.GaussianBlur(
        result[int(h * 0.65):, :], (31, 31), 15
    )

    # Blur top-right corner — ID number on National ID
    result[:int(h * 0.15), int(w * 0.55):] = cv2.GaussianBlur(
        result[:int(h * 0.15), int(w * 0.55):], (31, 31), 15
    )

    return result, len(faces)


# ── MAIN ──────────────────────────────────────────────────────────────────────
def main():
    total_processed = 0
    total_faces_found = 0

    for raw_dir in RAW_DIRS:
        raw_path = Path(raw_dir)
        if not raw_path.exists():
            print(f"⚠️  Skipping {raw_dir} — folder not found")
            continue

        # Mirror the folder structure under anonymized/
        relative = raw_path.relative_to('data/raw')
        anon_path = Path(ANON_BASE) / relative
        anon_path.mkdir(parents=True, exist_ok=True)

        images = (list(raw_path.glob('*.jpg')) +
                  list(raw_path.glob('*.png')) +
                  list(raw_path.glob('*.jpeg')))

        if not images:
            print(f"⚠️  No images found in {raw_dir}")
            continue

        print(f"\n📄 Anonymizing {len(images)} images in {raw_dir}")

        for img_path in images:
            img = cv2.imread(str(img_path))
            if img is None:
                print(f"  ⚠️  Could not read {img_path.name} — skipping")
                continue

            anonymized, faces = anonymize_image(img)
            total_faces_found += faces

            # Save anonymized copy
            out_path = anon_path / img_path.name
            cv2.imwrite(str(out_path), anonymized)
            total_processed += 1

            status = f"({faces} face(s) blurred)" if faces > 0 else "(no face detected — ID number regions blurred)"
            print(f"  ✅ {img_path.name} {status}")

    # Summary
    print("\n" + "=" * 50)
    print("ANONYMIZATION COMPLETE")
    print("=" * 50)
    print(f"  Total images processed : {total_processed}")
    print(f"  Total faces blurred    : {total_faces_found}")
    print(f"  Anonymized images saved to: {ANON_BASE}/")
    print("\n  ⚠️  IMPORTANT:")
    print("  Your original raw images are still in data/raw/")
    print("  Use ONLY the anonymized copies for training.")
    print("  Consider moving originals to an encrypted offline drive.")


if __name__ == "__main__":
    main()