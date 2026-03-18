from pathlib import Path

base = Path('data/raw/anonymized')

print("=" * 45)
print("IMAGE COUNT — ANONYMIZED DATA")
print("=" * 45)

total = 0
for doc_type in ['national_ids', 'passports', 'kcse_certificates']:
    for label in ['genuine', 'fake']:
        folder = base / doc_type / label
        if not folder.exists():
            print(f"  {doc_type}/{label}: folder not found")
            continue
        count = len(list(folder.glob('*.jpg')) +
                    list(folder.glob('*.png')) +
                    list(folder.glob('*.jpeg')))
        total += count
        status = "✅" if count > 0 else "⚠️  EMPTY"
        print(f"  {doc_type}/{label}: {count} images  {status}")

print(f"\n  Total images: {total}")
print("=" * 45)