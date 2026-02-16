# Dataset Summary - Day 4

## Original Collection (Day 3)
- National IDs: 12 genuine samples
- KCSE Certificates: 10 genuine samples
- Passports: 8 genuine samples
- **Total Original: 30 images**

## Augmented Dataset (Day 4)

### Training Set
- National IDs: 132 images (12 × 11)
- KCSE Certificates: 110 images (10 × 11)
- Passports: 88 images (8 × 11)
- **Total Training: ~330 images**

### Validation Set
- National IDs: 48 images (12 × 4)
- KCSE Certificates: 40 images (10 × 4)
- Passports: 32 images (8 × 4)
- **Total Validation: ~120 images**

### Fake Documents
- Fake National IDs: 12
- Fake KCSE Certificates: 10
- Fake Passports: 8
- **Total Fakes: 30 images**

## Grand Total: ~480 images

### Dataset Split
- Training (genuine + augmented): 330 images (69%)
- Validation: 120 images (25%)
- Test (fake documents): 30 images (6%)

## Augmentation Techniques Applied
1. Rotation (±5 degrees)
2. Scaling (±5%)
3. Perspective transformation
4. Gaussian blur
5. Gaussian noise
6. Brightness/contrast adjustment
7. Color variations
8. JPEG compression artifacts

## Fake Document Methods
1. Hologram removal
2. Text alteration
3. Quality reduction
4. Color shifting
5. Artifact addition

## Ready for Model Training: ✅ YES