import cv2
import numpy as np
import random
from pathlib import Path

random.seed(42)
np.random.seed(42)

# ── CONFIG ────────────────────────────────────────────────────────────────────
BASE_DIR = 'data/raw/anonymized'
DOC_TYPES = ['national_ids', 'passports', 'kcse_certificates']
LABELS    = ['genuine', 'fake']
AUG_PER_IMAGE = 8

# ── AUGMENTATION FUNCTIONS ────────────────────────────────────────────────────
def rotate(img):
    angle = random.uniform(-15, 15)
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1.0)
    return cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)

def flip(img):
    return cv2.flip(img, 1)

def brightness(img):
    factor = random.uniform(0.6, 1.4)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] * factor, 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

def blur(img):
    k = random.choice([3, 5])
    return cv2.GaussianBlur(img, (k, k), 0)

def noise(img):
    n = np.random.normal(0, random.uniform(5, 20), img.shape).astype(np.int16)
    return np.clip(img.astype(np.int16) + n, 0, 255).astype(np.uint8)

def perspective(img):
    h, w = img.shape[:2]
    margin = int(min(h, w) * 0.03)
    pts1 = np.float32([[0,0],[w,0],[0,h],[w,h]])
    pts2 = np.float32([
        [random.randint(0, margin), random.randint(0, margin)],
        [w-random.randint(0, margin), random.randint(0, margin)],
        [random.randint(0, margin), h-random.randint(0, margin)],
        [w-random.randint(0, margin), h-random.randint(0, margin)]
    ])
    M = cv2.getPerspectiveTransform(pts1, pts2)
    return cv2.warpPerspective(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)

def cutout(img):
    result = img.copy()
    h, w = img.shape[:2]
    for _ in range(2):
        x = random.randint(0, max(1, w - 30))
        y = random.randint(0, max(1, h - 30))
        result[y:y+30, x:x+30] = 0
    return result

def sharpen(img):
    kernel = np.array([[0,-1,0],[-1,5,-1],[0,-1,0]])
    return cv2.filter2D(img, -1, kernel)

AUGMENT_FUNCS = [rotate, flip, brightness, blur, noise, perspective, cutout, sharpen]

# ── MAIN ──────────────────────────────────────────────────────────────────────
def main():
    total_created = 0
    summary = {}

    for doc_type in DOC_TYPES:
        for label in LABELS:
            folder = Path(BASE_DIR) / doc_type / label
            if not folder.exists():
                print(f"⚠️  Skipping {doc_type}/{label} — folder not found")
                continue

            # Only augment original images — skip already augmented ones
            originals = [
                p for p in
                list(folder.glob('*.jpg')) +
                list(folder.glob('*.png')) +
                list(folder.glob('*.jpeg'))
                if '_aug_' not in p.stem
            ]

            if not originals:
                print(f"⚠️  No original images in {doc_type}/{label}")
                continue

            print(f"\n📄 Augmenting {doc_type}/{label} "
                  f"({len(originals)} originals → "
                  f"{len(originals) * AUG_PER_IMAGE} augmented)")

            count = 0
            for img_path in originals:
                img = cv2.imread(str(img_path))
                if img is None:
                    print(f"  ⚠️  Could not read {img_path.name} — skipping")
                    continue

                img = cv2.resize(img, (224, 224))

                for i in range(AUG_PER_IMAGE):
                    # Apply 2-3 random augmentations per copy
                    augmented = img.copy()
                    funcs = random.sample(AUGMENT_FUNCS, k=random.randint(2, 3))
                    for func in funcs:
                        augmented = func(augmented)

                    out_name = f"{img_path.stem}_aug_{i+1:02d}.jpg"
                    out_path = folder / out_name
                    cv2.imwrite(str(out_path), augmented)
                    count += 1

            total_created += count
            key = f"{doc_type}/{label}"
            summary[key] = {
                'originals': len(originals),
                'augmented': count,
                'total':     len(originals) + count
            }
            print(f"  ✅ Created {count} augmented images")

    # Final summary
    print("\n" + "=" * 55)
    print("AUGMENTATION COMPLETE")
    print("=" * 55)
    print(f"  {'Folder':<35} {'Orig':>5} {'Aug':>6} {'Total':>7}")
    print(f"  {'-'*35} {'-'*5} {'-'*6} {'-'*7}")
    for key, v in summary.items():
        print(f"  {key:<35} {v['originals']:>5} "
              f"{v['augmented']:>6} {v['total']:>7}")
    print(f"\n  Total new images created: {total_created}")
    print(f"\n  Next step: run train_classifier.py and train_detector.py")


if __name__ == "__main__":
    main()