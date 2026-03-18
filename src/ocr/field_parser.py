"""
Field Parser
Extracts specific fields from raw OCR text using regex patterns.
Handles messy OCR output with fuzzy matching.
"""
import re
from pathlib import Path


# ── KENYAN DOCUMENT PATTERNS ──────────────────────────────────────────────────

ID_NUMBER_PATTERNS = [
    r'\b(\d{8})\b',
    r'(?:ID|No|Number|Namba)[:\s#]*(\d{7,9})',
    r'(?:NAMBA|NAMBARI)[:\s]*(\d{7,9})',
]

DATE_PATTERNS = [
    # Labelled first (most reliable)
    r'(?:DOB|D\.O\.B|Date of Birth|DATE OF BIRTH|TAREHE YA KUZALIWA|TAREHE)[:\s]*'
    r'(\d{1,2}[/\-\.\s]\d{1,2}[/\-\.\s]\d{4})',
    # Standard separators
    r'\b(\d{1,2}[/\-\.]\d{1,2}[/\-\.]\d{4})\b',
    # Spaces as separator
    r'\b(\d{1,2}\s+\d{1,2}\s+\d{4})\b',
    # Full month name
    r'\b(\d{1,2}\s+(?:January|February|March|April|May|June|July|August|'
    r'September|October|November|December)\s+\d{4})\b',
    # Short month name any case
    r'\b(\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{4})\b',
    # Uppercase short month (Tesseract ALL CAPS)
    r'\b(\d{1,2}\s+(?:JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC)\s+\d{4})\b',
    # No separator: 15031990
    r'\b(\d{2}\d{2}\d{4})\b',
]

NAME_PATTERNS = [
    r'(?:Name|Jina|JINA)[:\s]+([A-Z][A-Z\s]{3,40})',
    r'(?:Surname|SURNAME)[:\s]+([A-Z][A-Z\s]{2,25})',
    r'^([A-Z]{2,}\s+[A-Z]{2,}(?:\s+[A-Z]{2,})?)$',
]

SEX_PATTERNS = [
    r'(?:Sex|Gender|JINSIA)[:\s]*(MALE|FEMALE)',
    r'\b(MALE|FEMALE)\b',
]

DISTRICT_PATTERNS = [
    r'(?:District|WILAYA|Place of Birth)[:\s]+([A-Z][A-Za-z\s]{3,25})',
    r'(?:County|KAUNTI)[:\s]+([A-Z][A-Za-z\s]{3,25})',
]

KCSE_INDEX_PATTERNS = [
    r'\b(\d{11})\b',
    r'(?:Index|INDEX)[:\s#]*(\d{10,12})',
]

KCSE_YEAR_PATTERNS = [
    r'\b((?:19|20)\d{2})\b',
    r'(?:Year|MWAKA)[:\s]*(20\d{2})',
]

KCSE_GRADE_PATTERNS = [
    r'(?:Mean Grade|WASTANI)[:\s]*(A-|A|B\+|B-|B|C\+|C-|C|D\+|D-|D|E)\b',
    r'(?:Grade|DARAJA)[:\s]*(A-|A|B\+|B-|B|C\+|C-|C|D\+|D-|D|E)\b',
    r'\b(A-|B\+|B-|C\+|C-|D\+|D-|A|B|C|D|E)\b',
]

PASSPORT_NUMBER_PATTERNS = [
    r'\b([A-Z]{1,2}\d{7})\b',
    r'(?:Passport|PASIPOTI)[:\s#]*([A-Z]{1,2}\d{6,8})',
]

MRZ_PATTERNS = [
    r'(P<KEN[A-Z<]+)',
    r'([A-Z0-9<]{20,})',
]

VALID_GRADES = {"A", "A-", "B+", "B", "B-", "C+", "C", "C-", "D+", "D", "D-", "E"}

MONTH_MAP = {
    'JANUARY': '01', 'FEBRUARY': '02', 'MARCH': '03', 'APRIL': '04',
    'JUNE': '06', 'JULY': '07', 'AUGUST': '08', 'SEPTEMBER': '09',
    'OCTOBER': '10', 'NOVEMBER': '11', 'DECEMBER': '12',
    'JAN': '01', 'FEB': '02', 'MAR': '03', 'APR': '04',
    'MAY': '05', 'JUN': '06', 'JUL': '07', 'AUG': '08',
    'SEP': '09', 'OCT': '10', 'NOV': '11', 'DEC': '12',
}


# ── DATE CLEANER ──────────────────────────────────────────────────────────────

def clean_date(raw):
    """
    Normalise a raw OCR date string into DD/MM/YYYY.
    Handles OCR misreads, mixed separators, month names, and label prefixes.
    Returns None if the result is not a plausible date.
    """
    if not raw:
        return None

    d = raw.strip().upper()

    # 1. Strip label prefixes: "DOB:", "Date of Birth:", "TAREHE:" etc.
    d = re.sub(
        r'^(?:D\.O\.B|DOB|DATE OF BIRTH|TAREHE YA KUZALIWA|TAREHE)\s*[:\-]?\s*',
        '', d
    ).strip()

    # 2. Replace month names with numbers (longest first to avoid partial matches)
    for name in sorted(MONTH_MAP, key=len, reverse=True):
        if name in d:
            d = d.replace(name, MONTH_MAP[name])
            break   # only replace one month name

    # 3. Normalise separators to /
    d = re.sub(r'[\-\.\s]+', '/', d)

    # 4. Handle 8-digit no-separator: 15031990 → 15/03/1990
    if re.match(r'^\d{8}$', d):
        d = f"{d[:2]}/{d[2:4]}/{d[4:]}"

    # 5. Now safe to fix OCR digit misreads — string should be digits + slashes only
    if re.match(r'^[\d/OoSsIiLl]+$', d):
        d = (d.replace('O', '0').replace('o', '0')
              .replace('S', '5').replace('s', '5')
              .replace('I', '1').replace('i', '1')
              .replace('L', '1').replace('l', '1'))

    # 6. Clean up double slashes
    d = re.sub(r'/+', '/', d).strip('/')

    # 7. Must match DD/MM/YYYY
    if not re.match(r'^\d{1,2}/\d{1,2}/\d{4}$', d):
        return None

    # 8. Range check
    try:
        parts = d.split('/')
        day, month, year = int(parts[0]), int(parts[1]), int(parts[2])
        if not (1 <= day <= 31 and 1 <= month <= 12 and 1900 <= year <= 2030):
            return None
    except (ValueError, IndexError):
        return None

    return d


def is_valid_dob(date_str):
    """Return True if date_str (DD/MM/YYYY) is a plausible date of birth."""
    if not date_str:
        return False
    try:
        from datetime import date
        parts = date_str.split('/')
        day, month, year = int(parts[0]), int(parts[1]), int(parts[2])
        dob = date(year, month, day)
        age = (date.today() - dob).days / 365.25
        return 0 < age < 120
    except Exception:
        return False


# ── PARSER CLASS ──────────────────────────────────────────────────────────────

class FieldParser:
    """
    Parses specific fields from raw OCR text for each document type.
    Uses regex with fallback patterns for noisy OCR output.
    """

    def _clean_text(self, text):
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'[^\w\s/\-\.\,\:\#<]', ' ', text)
        return text.strip()

    def _find_first_match(self, text, patterns):
        for pattern in patterns:
            try:
                matches = re.findall(pattern, text, re.IGNORECASE | re.MULTILINE)
                if matches:
                    for m in matches:
                        val = m.strip() if isinstance(m, str) else m[0].strip()
                        if val:
                            return val
            except re.error:
                continue
        return None

    def _find_all_matches(self, text, patterns):
        results = []
        for pattern in patterns:
            try:
                matches = re.findall(pattern, text, re.IGNORECASE | re.MULTILINE)
                for m in matches:
                    val = m.strip() if isinstance(m, str) else m[0].strip()
                    if val and val not in results:
                        results.append(val)
            except re.error:
                continue
        return results

    def _extract_dates(self, text):
        """Extract, clean, and deduplicate all dates from text."""
        raw_dates = self._find_all_matches(text, DATE_PATTERNS)
        cleaned = []
        seen = set()
        for raw in raw_dates:
            c = clean_date(raw)
            if c and c not in seen:
                cleaned.append(c)
                seen.add(c)
        return cleaned

    def _validate_id_number(self, id_num):
        if not id_num:
            return False
        return bool(re.match(r'^\d{7,9}$', str(id_num).strip()))

    def _validate_kcse_year(self, year):
        if not year:
            return False
        try:
            year_str = str(year).strip()
            if len(year_str) != 4:
                return False
            return 1989 <= int(year_str) <= 2030
        except ValueError:
            return False

    def _validate_passport_number(self, num):
        if not num:
            return False
        return bool(re.match(r'^[A-Z]{1,2}\d{7}$', str(num).strip()))

    # ── NATIONAL ID ───────────────────────────────────────────────────────────

    def parse_national_id(self, ocr_result):
        text      = ocr_result.get("raw_text", "")
        lines     = ocr_result.get("lines", [])
        full_text = "\n".join(lines) + "\n" + text
        fields    = {}

        fields["is_kenyan_document"] = bool(
            re.search(r'KENYA|JAMHURI|REPUBLIC', full_text, re.IGNORECASE)
        )

        id_num = self._find_first_match(full_text, ID_NUMBER_PATTERNS)
        fields["id_number"] = {
            "value": id_num,
            "found": id_num is not None,
            "valid": self._validate_id_number(id_num)
        }

        dates       = self._extract_dates(full_text)
        likely_dob  = next((d for d in dates if is_valid_dob(d)), None)
        likely_issue = dates[-1] if len(dates) > 1 else None
        fields["dates"] = {
            "all_found":    dates,
            "count":        len(dates),
            "found":        len(dates) > 0,
            "likely_dob":   likely_dob,
            "likely_issue": likely_issue,
            "dob_valid":    is_valid_dob(likely_dob),
        }

        sex = self._find_first_match(full_text, SEX_PATTERNS)
        fields["sex"] = {"value": sex, "found": sex is not None}

        district = self._find_first_match(full_text, DISTRICT_PATTERNS)
        fields["district"] = {"value": district, "found": district is not None}

        key_fields  = ["id_number", "dates", "sex"]
        found_count = sum(1 for f in key_fields if fields[f]["found"])
        fields["extraction_score"] = {
            "fields_found": found_count,
            "total_fields": len(key_fields),
            "percentage":   round(found_count / len(key_fields) * 100)
        }
        return fields

    # ── KCSE ──────────────────────────────────────────────────────────────────

    def parse_kcse_certificate(self, ocr_result):
        text      = ocr_result.get("raw_text", "")
        lines     = ocr_result.get("lines", [])
        full_text = "\n".join(lines) + "\n" + text
        fields    = {}

        fields["is_kcse_document"] = bool(
            re.search(
                r'KCSE|KENYA CERTIFICATE|SECONDARY EDUCATION|KNEC',
                full_text, re.IGNORECASE
            )
        )

        index = self._find_first_match(full_text, KCSE_INDEX_PATTERNS)
        fields["index_number"] = {
            "value": index,
            "found": index is not None,
            "valid": len(str(index)) >= 10 if index else False
        }

        year       = self._find_first_match(full_text, KCSE_YEAR_PATTERNS)
        year_valid = self._validate_kcse_year(year)
        fields["year"] = {
            "value": year,
            "found": year is not None and year_valid,
            "valid": year_valid
        }

        mean_grade = self._find_first_match(full_text, KCSE_GRADE_PATTERNS[:2])
        if not mean_grade:
            mean_grade = self._find_first_match(full_text, KCSE_GRADE_PATTERNS[2:])
        grade_valid = mean_grade in VALID_GRADES if mean_grade else False
        fields["mean_grade"] = {
            "value": mean_grade,
            "found": mean_grade is not None and grade_valid,
            "valid": grade_valid
        }

        all_grades = self._find_all_matches(full_text, KCSE_GRADE_PATTERNS)
        all_grades = [g for g in all_grades if g in VALID_GRADES]
        fields["all_grades"] = {
            "values": all_grades,
            "count":  len(all_grades),
            "found":  len(all_grades) > 0
        }

        key_fields  = ["index_number", "year", "mean_grade"]
        found_count = sum(1 for f in key_fields if fields[f]["found"])
        fields["extraction_score"] = {
            "fields_found": found_count,
            "total_fields": len(key_fields),
            "percentage":   round(found_count / len(key_fields) * 100)
        }
        return fields

    # ── PASSPORT ──────────────────────────────────────────────────────────────

    def parse_passport(self, ocr_result):
        text      = ocr_result.get("raw_text", "")
        lines     = ocr_result.get("lines", [])
        full_text = "\n".join(lines) + "\n" + text
        fields    = {}

        fields["is_passport_document"] = bool(
            re.search(r'PASSPORT|PASIPOTI|KEN|KENYA', full_text, re.IGNORECASE)
        )

        passport_num = self._find_first_match(full_text, PASSPORT_NUMBER_PATTERNS)
        fields["passport_number"] = {
            "value": passport_num,
            "found": passport_num is not None,
            "valid": self._validate_passport_number(passport_num)
        }

        mrz = self._find_all_matches(full_text, MRZ_PATTERNS)
        fields["mrz"] = {"lines": mrz, "found": len(mrz) > 0, "count": len(mrz)}

        dates        = self._extract_dates(full_text)
        likely_dob   = next((d for d in dates if is_valid_dob(d)), None)
        likely_expiry = dates[-1] if len(dates) > 1 else None
        fields["dates"] = {
            "all_found":     dates,
            "found":         len(dates) > 0,
            "count":         len(dates),
            "likely_dob":    likely_dob,
            "likely_expiry": likely_expiry,
            "dob_valid":     is_valid_dob(likely_dob),
        }

        mrz_text = " ".join(mrz)
        combined = full_text + " " + mrz_text
        fields["nationality"] = {
            "found": bool(
                re.search(r'KENYAN|KENYA|P<KEN|KEN', combined, re.IGNORECASE)
            )
        }

        key_fields  = ["passport_number", "dates", "nationality"]
        found_count = sum(1 for f in key_fields if fields[f]["found"])
        fields["extraction_score"] = {
            "fields_found": found_count,
            "total_fields": len(key_fields),
            "percentage":   round(found_count / len(key_fields) * 100)
        }
        return fields

    # ── AUTO ROUTER ───────────────────────────────────────────────────────────

    def parse(self, ocr_result, doc_type):
        parsers = {
            "national_id":       self.parse_national_id,
            "national_ids":      self.parse_national_id,
            "kcse_certificate":  self.parse_kcse_certificate,
            "kcse_certificates": self.parse_kcse_certificate,
            "passport":          self.parse_passport,
            "passports":         self.parse_passport,
        }
        parser_fn = parsers.get(doc_type)
        if not parser_fn:
            return {"error": f"Unknown doc_type: {doc_type}"}
        return parser_fn(ocr_result)


# ── QUICK TEST ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")

    print("=" * 60)
    print("clean_date() UNIT TESTS")
    print("=" * 60)

    date_tests = [
        ("15/03/1990",       "15/03/1990"),
        ("15-03-1990",       "15/03/1990"),
        ("15.03.1990",       "15/03/1990"),
        ("15 03 1990",       "15/03/1990"),
        ("15/O3/199O",       "15/03/1990"),
        ("1S/03/1990",       None),
        ("15 MAR 1990",      "15/03/1990"),
        ("15 Mar 1990",      "15/03/1990"),
        ("15 March 1990",    "15/03/1990"),
        ("15031990",         "15/03/1990"),
        ("99/99/1990",       None),
        ("DOB: 15/03/1990",  "15/03/1990"),
        ("DOB 15/03/1990",   "15/03/1990"),
        ("Date of Birth: 15/03/1990", "15/03/1990"),
    ]

    passed = 0
    for raw, expected in date_tests:
        result = clean_date(raw)
        ok = result == expected
        passed += ok
        status = "✅" if ok else "❌"
        print(f"  {status} {raw!r:35} → {str(result):15} (expected {str(expected)})")

    print(f"\n  {passed}/{len(date_tests)} tests passed")

    print("\n" + "=" * 60)
    print("is_valid_dob() UNIT TESTS")
    print("=" * 60)
    for d, expected in [("15/03/1990", True), ("01/01/2030", False),
                        ("01/01/1850", False), (None, False)]:
        result = is_valid_dob(d)
        print(f"  {'✅' if result == expected else '❌'} is_valid_dob({str(d)!r:15}) → {result}")

    print("\n" + "=" * 60)
    print("FIELD PARSER TEST ON IMAGES")
    print("=" * 60)
    try:
        from src.ocr.text_extractor import extract_text
        parser = FieldParser()
        for folder, doc_type in [
            ("data/raw/national_ids/genuine",     "national_id"),
            ("data/raw/kcse_certificates/genuine", "kcse_certificate"),
            ("data/raw/passports/genuine",         "passport"),
        ]:
            path = Path(folder)
            if not path.exists():
                continue
            images = list(path.glob("*.jpg")) + list(path.glob("*.JPG")) + list(path.glob("*.png"))
            if not images:
                continue
            print(f"\n--- {doc_type.upper()} ---")
            ocr_result = extract_text(str(images[0]), doc_type=doc_type)
            print(f"OCR confidence: {ocr_result.get('confidence', 0)}%")
            fields = parser.parse(ocr_result, doc_type)
            for field, value in fields.items():
                if field == "extraction_score":
                    s = value
                    print(f"  SCORE: {s['fields_found']}/{s['total_fields']} ({s['percentage']}%)")
                elif isinstance(value, dict):
                    found   = value.get("found", False)
                    display = next((value.get(k) for k in ["value","values","lines","all_found"] if value.get(k)), None)
                    print(f"  {'✅' if (found and display) else '❌'} {field}: {display}")
                else:
                    print(f"  {'✅' if value else '❌'} {field}: {value}")
    except ImportError:
        print("  (Run from project root to test on real images)")

    print("\n✅ Field parser ready!")
    print("=" * 60)