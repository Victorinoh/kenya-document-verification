import cv2
import numpy as np
import random
import os
from pathlib import Path

random.seed(42)
np.random.seed(42)

# ── CONFIG ────────────────────────────────────────────────────────────────────
GENUINE_DIRS = {
    'national_ids':      'data/raw/anonymized/national_ids/genuine',
    'passports':         'data/raw/anonymized/passports/genuine',
    'kcse_certificates': 'data/raw/anonymized/kcse_certificates/genuine'
}
OUTPUT_DIRS = {
    'national_ids':      'data/raw/anonymized/national_ids/fake',
    'passports':         'data/raw/anonymized/passports/fake',
    'kcse_certificates': 'data/raw/anonymized/kcse_certificates/fake'
}

# ── FORGERY TYPE 1 — TEXT TAMPERING ──────────────────────────────────────────
def text_tampering(img):
    result = img.copy()
    h, w = result.shape[:2]
    num_regions = random.randint(2, 5)
    for _ in range(num_regions):
        rw = random.randint(40, 120)
        rh = random.randint(8, 18)
        rx = random.randint(0, max(1, w - rw))
        ry = random.randint(0, max(1, h - rh))
        region = result[ry:ry+rh, rx:rx+rw]
        mean_colour = region.mean(axis=(0, 1)).astype(np.uint8)
        noise = np.random.randint(-15, 15, 3)
        fill = np.clip(
            mean_colour.astype(np.int16) + noise, 0, 255
        ).astype(np.uint8)
        result[ry:ry+rh, rx:rx+rw] = fill
        num_lines = random.randint(1, 3)
        for _ in range(num_lines):
            lx1 = rx + random.randint(0, 10)
            lx2 = rx + rw - random.randint(0, 10)
            ly  = ry + random.randint(3, max(4, rh - 3))
            lthick = random.randint(1, 2)
            colour = tuple(int(c) for c in (mean_colour * 0.3).astype(np.uint8))
            cv2.line(result, (lx1, ly), (lx2, ly), colour, lthick)
    return result


# ── FORGERY TYPE 2 — PHOTO SUBSTITUTION ──────────────────────────────────────
def photo_substitution(img):
    result = img.copy()
    h, w = result.shape[:2]
    ph = int(h * random.uniform(0.25, 0.40))
    pw = int(w * random.uniform(0.18, 0.28))
    px = random.randint(5, 20)
    py = random.randint(10, 30)
    ph = min(ph, h - py)
    pw = min(pw, w - px)
    if ph <= 0 or pw <= 0:
        return result
    photo = result[py:py+ph, px:px+pw].copy()
    shift = np.random.randint(-20, 20, 3)
    photo = np.clip(
        photo.astype(np.int16) + shift, 0, 255
    ).astype(np.uint8)
    photo = cv2.GaussianBlur(photo, (3, 3), 0)
    result[py:py+ph, px:px+pw] = photo
    cv2.rectangle(
        result, (px, py), (px+pw, py+ph),
        (random.randint(80, 140),) * 3, 1
    )
    return result


# ── FORGERY TYPE 3 — COLOUR/CONTRAST MANIPULATION ────────────────────────────
def colour_manipulation(img):
    result = img.copy()
    h, w = result.shape[:2]
    hsv = cv2.cvtColor(result, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 0] = (hsv[:, :, 0] + random.uniform(-12, 12)) % 180
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] * random.uniform(0.80, 1.20), 0, 255)
    result = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    num_patches = random.randint(1, 3)
    for _ in range(num_patches):
        pw = random.randint(40, 100)
        ph = random.randint(20, 60)
        px = random.randint(0, max(1, w - pw))
        py = random.randint(0, max(1, h - ph))
        factor = random.uniform(0.65, 1.35)
        patch = result[py:py+ph, px:px+pw].astype(np.float32)
        result[py:py+ph, px:px+pw] = np.clip(
            patch * factor, 0, 255
        ).astype(np.uint8)
    return result


# ── FORGERY TYPE 4 — SCAN ARTIFACTS ──────────────────────────────────────────
def scan_artifacts(img):
    result = img.copy().astype(np.float32)
    h, w = result.shape[:2]
    num_lines = random.randint(3, 10)
    for _ in range(num_lines):
        ly = random.randint(0, h - 1)
        intensity = random.uniform(0.6, 0.95)
        result[ly, :] *= intensity
    gradient = np.linspace(
        random.uniform(0.80, 1.0),
        random.uniform(0.80, 1.0),
        w
    ).astype(np.float32)
    result *= gradient[np.newaxis, :, np.newaxis]
    result = np.clip(result, 0, 255).astype(np.uint8)
    num_dots = random.randint(20, 80)
    for _ in range(num_dots):
        dx = random.randint(0, w - 1)
        dy = random.randint(0, h - 1)
        result[dy, dx] = random.choice([0, 255])
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), random.randint(45, 70)]
    _, encoded = cv2.imencode('.jpg', result, encode_param)
    result = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    return result


# ── FORGERY TYPE 5 — WATERMARK REMOVAL ───────────────────────────────────────
def watermark_removal(img):
    result = img.copy()
    h, w = result.shape[:2]
    num_stripes = random.randint(2, 4)
    for i in range(num_stripes):
        offset = int(h * (0.2 + i * 0.2))
        pts = np.array([
            [0,     min(offset + 15, h - 1)],
            [w,     min(offset + 15 + w, h - 1)],
            [w,     min(offset + 30 + w, h - 1)],
            [0,     min(offset + 30, h - 1)]
        ], dtype=np.int32)
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(mask, [pts], 255)
        blurred = cv2.GaussianBlur(result, (15, 15), 0)
        result = np.where(
            mask[:, :, np.newaxis] > 0, blurred, result
        )
    noise = np.random.normal(0, 8, result.shape).astype(np.int16)
    result = np.clip(
        result.astype(np.int16) + noise, 0, 255
    ).astype(np.uint8)
    return result


# ── COMBINE ALL FORGERY TYPES ─────────────────────────────────────────────────
FORGERY_FUNCS = [
    ('text_tampering',     text_tampering),
    ('photo_substitution', photo_substitution),
    ('colour_manipulation',colour_manipulation),
    ('scan_artifacts',     scan_artifacts),
    ('watermark_removal',  watermark_removal),
]

def generate_fake(img):
    result = img.copy()
    selected = random.sample(FORGERY_FUNCS, k=random.randint(2, 4))
    applied = []
    for name, func in selected:
        result = func(result)
        applied.append(name)
    return result, applied


# ── MAIN ──────────────────────────────────────────────────────────────────────
def main():
    total_generated = 0
    summary = {}

    for doc_type, genuine_dir in GENUINE_DIRS.items():
        genuine_path = Path(genuine_dir)
        output_path  = Path(OUTPUT_DIRS[doc_type])
        output_path.mkdir(parents=True, exist_ok=True)

        if not genuine_path.exists():
            print(f"⚠️  Skipping {doc_type} — folder not found: {genuine_dir}")
            continue

        images = (list(genuine_path.glob('*.jpg'))  +
                  list(genuine_path.glob('*.png'))  +
                  list(genuine_path.glob('*.jpeg')))

        if not images:
            print(f"⚠️  No images found in {genuine_dir}")
            continue

        print(f"\n📄 Generating fakes for {doc_type} "
              f"({len(images)} genuine images)")

        count = 0
        for img_path in images:
            img = cv2.imread(str(img_path))
            if img is None:
                print(f"  ⚠️  Could not read {img_path.name} — skipping")
                continue

            img = cv2.resize(img, (224, 224))
            fake, applied = generate_fake(img)

            short = [a[:4] for a in applied]
            out_name = f"fake_{img_path.stem}_{'_'.join(short)}.jpg"
            out_path = output_path / out_name
            cv2.imwrite(str(out_path), fake)

            count += 1
            total_generated += 1
            print(f"  ✅ {img_path.name} → {out_name}")
            print(f"     Forgeries: {', '.join(applied)}")

        summary[doc_type] = count
        print(f"\n  Generated {count} fakes for {doc_type}")

    # Final summary
    print("\n" + "=" * 50)
    print("GENERATION COMPLETE")
    print("=" * 50)
    for doc_type, count in summary.items():
        genuine_count = len(
            list(Path(GENUINE_DIRS[doc_type]).glob('*.jpg')) +
            list(Path(GENUINE_DIRS[doc_type]).glob('*.png')) +
            list(Path(GENUINE_DIRS[doc_type]).glob('*.jpeg'))
        )
        print(f"  {doc_type:20s}: "
              f"{genuine_count} genuine → {count} fakes generated")
    print(f"\n  Total fakes generated: {total_generated}")
    print(f"  Saved to: data/raw/anonymized/*/fake/")
    print("\n  Next step: run augmentation to expand dataset 8x")


if __name__ == "__main__":
    main()