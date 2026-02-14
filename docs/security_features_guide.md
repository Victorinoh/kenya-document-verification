# Security Features Guide
## Kenyan Government Documents
Last Updated: February 2026

---

## 1. Kenyan National ID Card

### Physical Description
- Size: 85.6mm × 53.98mm (standard credit card size)
- Material: Polycarbonate (hard plastic, not paper)
- Color: Green and gold theme with coat of arms

### Security Features

#### Feature 1: Laser Engraving
- Personal data is laser-engraved (not printed)
- Cannot be erased, scratched or chemically altered
- Has a tactile feel when touched
- Detection: Check for raised/engraved text vs flat printed text

#### Feature 2: Hologram
- Located: Top right corner of the card
- Shows Kenya coat of arms
- Color shifts when tilted (gold to green)
- Contains microtext within hologram
- Detection: Color shift analysis at multiple angles

#### Feature 3: Ghost Image
- Secondary smaller photo of the holder
- Laser engraved in grayscale
- Located: Right side of card
- Appears when card is tilted
- Detection: Grayscale intensity analysis

#### Feature 4: Microprinting
- Extremely tiny text along card borders
- Text reads "REPUBLIC OF KENYA" repeatedly
- Only visible under magnification (10x+)
- Detection: High resolution scan + zoom analysis

#### Feature 5: Guilloche Pattern
- Complex interlaced background pattern
- Fine lines that are impossible to photocopy clearly
- Located: Card background
- Detection: Line continuity and pattern analysis

#### Feature 6: UV Reactive Features
- Invisible patterns visible only under UV light
- Kenya coat of arms glows under UV
- Detection: Requires UV light source

#### Feature 7: Serial Number
- Unique alphanumeric code
- Format: 8 digits (e.g., 12345678)
- Located: Bottom of card
- Detection: OCR + format validation regex: ^\d{8}$

### Data Fields to Extract (OCR)
| Field | Format | Validation Rule |
|-------|--------|-----------------|
| ID Number | 8 digits | ^\d{8}$ |
| Full Name | UPPERCASE letters | ^[A-Z\s]+$ |
| Date of Birth | DD/MM/YYYY | Valid date, age 18+ |
| Sex | M or F | ^[MF]$ |
| Place of Birth | Text | Non-empty string |
| Date of Issue | DD/MM/YYYY | After DOB, not future |

### Common Forgery Signs
- Flat/printed hologram (no color shift)
- Missing ghost image
- Wrong font type or size
- Blurry microprinting
- Inconsistent card thickness
- Photo edges visible (pasted photo)
- Text looks printed not engraved

---

## 2. KCSE Certificate (Kenya National Examinations Council)

### Physical Description
- Size: A4 (210mm × 297mm)
- Material: High quality security paper
- Color: Blue and gold theme with KNEC logo

### Security Features

#### Feature 1: Watermark
- KNEC logo/lion watermark embedded in paper
- Visible when held against light
- Cannot be photocopied
- Detection: Backlight image analysis, grayscale

#### Feature 2: Security Thread
- Metallic thread embedded in paper
- Runs vertically through certificate
- Contains microtext "KNEC"
- Detection: Thread continuity detection

#### Feature 3: Microprinting
- "KNEC" repeated in tiny text on borders
- Only visible under magnification
- Detection: High resolution scan + edge detection

#### Feature 4: Embossed Seal
- KNEC official seal physically pressed into paper
- Creates raised impression
- Located: Bottom left corner
- Detection: Shadow analysis from angled lighting

#### Feature 5: Certificate Serial Number
- Unique identifier for each certificate
- Format: YEAR/SEQUENTIAL (e.g., 2023/001234)
- Detection: OCR + format regex: ^\d{4}/\d{6}$

#### Feature 6: Security Background Pattern
- Complex guilloche pattern background
- Color gradient design
- Impossible to reproduce with standard printer
- Detection: Pattern matching + color analysis

#### Feature 7: QR Code (Recent certificates 2020+)
- Scannable QR code for online verification
- Links to KNEC verification portal
- Detection: QR decode + URL validation

### Data Fields to Extract (OCR)
| Field | Format | Validation Rule |
|-------|--------|-----------------|
| Candidate Name | UPPERCASE | ^[A-Z\s]+$ |
| Index Number | XX-XXXXX-XXX | ^\d{2}-\d{5}-\d{3}$ |
| School Name | Text | Non-empty |
| Year | 4 digits | 1990-present |
| Certificate No. | YYYY/XXXXXX | Valid format |
| Mean Grade | A to E | ^[A-E][+-]?$ |
| Subjects | List | 7+ subjects |
| Grades | A to E per subject | Valid grade |

### Mandatory Subjects (Must be present)
- English
- Kiswahili
- Mathematics
- One Science subject
- One Humanities subject

### Common Forgery Signs
- No watermark visible against light
- Missing embossed seal
- Wrong font (not official KNEC font)
- Incorrect grade format
- Subject not in official KNEC list
- Index number wrong format
- Mean grade doesn't match subject grades
- QR code doesn't link to KNEC portal

---

## 3. Kenyan Passport

### Physical Description
- Size: 125mm × 88mm (standard passport)
- Material: Polycarbonate data page + paper pages
- Color: Dark blue cover with gold coat of arms

### Security Features

#### Feature 1: Machine Readable Zone (MRZ)
- Two lines of text at bottom of data page
- Contains encoded personal information
- Format: ICAO standard (international)
- Detection: MRZ parsing library (mrz package)

#### Feature 2: Biometric Chip (e-Passport)
- RFID chip embedded in cover
- Contains digital copy of photo and data
- Detection: NFC reader (hardware required)

#### Feature 3: Laser Perforations
- Tiny laser holes forming passport number
- Visible when held against light
- Detection: Backlight image analysis

#### Feature 4: Color Shifting Ink
- Passport number printed with color-shifting ink
- Changes from gold to green when tilted
- Detection: Multi-angle color analysis

#### Feature 5: UV Features
- Hidden patterns visible under UV light
- Kenya map and patterns glow
- Detection: UV camera required

#### Feature 6: Holographic Laminate
- Clear holographic overlay on data page
- Contains microtext and patterns
- Detection: Light reflection analysis

### MRZ Format (Passport)
Line 1: P<KENYASURNAME<<FIRSTNAME<MIDDLE<<<<<<<<<
Line 2: PASSPORT_NO CHECK DOB CHECK SEX EXPIRY CHECK

### Data Fields to Extract (OCR)
| Field | Format | Validation Rule |
|-------|--------|-----------------|
| Passport Number | AA000000 | ^[A-Z]{2}\d{6}$ |
| Surname | UPPERCASE | ^[A-Z\s]+$ |
| Given Names | UPPERCASE | ^[A-Z\s]+$ |
| Nationality | KENYAN | Fixed value |
| Date of Birth | DD MMM YYYY | Valid date |
| Sex | M or F | ^[MF]$ |
| Place of Birth | Text | Non-empty |
| Date of Issue | DD MMM YYYY | Before expiry |
| Date of Expiry | DD MMM YYYY | After issue date |

### Common Forgery Signs
- MRZ doesn't match printed data
- Wrong MRZ format or checksum
- Missing holographic overlay
- Photo doesn't match MRZ data
- Incorrect passport number format
- Expiry date before issue date

---

## 4. Driving License

### Physical Description
- Size: 85.6mm × 53.98mm (credit card size)
- Material: Polycarbonate
- Color: Blue theme

### Security Features
- Hologram overlay
- Ghost image
- Microprinting
- Laser engraved data
- Barcode on back

### Data Fields
| Field | Format |
|-------|--------|
| License Number | Format: AXXXXXXXX |
| Full Name | UPPERCASE |
| Date of Birth | DD/MM/YYYY |
| Issue Date | DD/MM/YYYY |
| Expiry Date | DD/MM/YYYY |
| Vehicle Classes | A, B, C, etc. |

---

## 5. Birth Certificate

### Physical Description
- Size: A4
- Material: Security paper
- Issued by: Civil Registration Department

### Security Features
- Watermark (Government of Kenya)
- Serial number
- Official stamp
- Registrar signature

### Data Fields
| Field | Format |
|-------|--------|
| Registration Number | Unique ID |
| Child's Name | UPPERCASE |
| Date of Birth | DD/MM/YYYY |
| Place of Birth | Text |
| Father's Name | UPPERCASE |
| Mother's Name | UPPERCASE |
| Date of Registration | DD/MM/YYYY |

---

## 6. AI Detection Strategy

### Priority Matrix

| Feature | Detection Method | Difficulty | Priority |
|---------|-----------------|------------|----------|
| Hologram | Color shift analysis | Medium | HIGH |
| Watermark | Grayscale backlight | Medium | HIGH |
| MRZ | OCR + checksum | Low | HIGH |
| QR/Barcode | Decode + validate | Low | HIGH |
| Microprint | High-res zoom | High | MEDIUM |
| Ghost image | Grayscale compare | Medium | MEDIUM |
| UV features | UV hardware | Very High | LOW |
| Embossed seal | Shadow analysis | High | LOW |

### Implementation Order (Weeks 3-6)
Week 3: Hologram + Watermark detection
Week 4: MRZ parsing + QR/Barcode reading
Week 5: OCR text extraction all fields
Week 6: NLP validation rules

### Detection Accuracy Targets
- Hologram detection: 90%+
- OCR extraction: 95%+
- Format validation: 99%+
- Overall authenticity: 90%+