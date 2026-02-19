"""
Field Parser
Extracts specific fields from raw OCR text using regex patterns.
Handles messy OCR output with fuzzy matching.
"""
import re
from pathlib import Path


# ── KENYAN DOCUMENT PATTERNS ──────────────────────────────────────────────────

# National ID patterns
ID_NUMBER_PATTERNS = [
    r'\b(\d{8})\b',
    r'(?:ID|No|Number|Namba)[:\s#]*(\d{7,9})',
    r'(?:NAMBA|NAMBARI)[:\s]*(\d{7,9})',
]

# Date patterns
DATE_PATTERNS = [
    r'\b(\d{1,2}[/\-\.]\d{1,2}[/\-\.]\d{4})\b',
    r'\b(\d{1,2}\s+(?:January|February|March|April|May|June|July|August|'
    r'September|October|November|December)\s+\d{4})\b',
    r'\b(\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)'
    r'\s+\d{4})\b',
    r'(?:DOB|Date of Birth|TAREHE)[:\s]*(\d{1,2}[/\-\.]\d{1,2}[/\-\.]\d{4})',
]

# Name patterns
NAME_PATTERNS = [
    r'(?:Name|Jina|JINA)[:\s]+([A-Z][A-Z\s]{3,40})',
    r'(?:Surname|SURNAME)[:\s]+([A-Z][A-Z\s]{2,25})',
    r'^([A-Z]{2,}\s+[A-Z]{2,}(?:\s+[A-Z]{2,})?)$',
]

# Sex patterns — full words only to avoid false positives
SEX_PATTERNS = [
    r'(?:Sex|Gender|JINSIA)[:\s]*(MALE|FEMALE)',
    r'\b(MALE|FEMALE)\b',
]

# District patterns
DISTRICT_PATTERNS = [
    r'(?:District|WILAYA|Place of Birth)[:\s]+([A-Z][A-Za-z\s]{3,25})',
    r'(?:County|KAUNTI)[:\s]+([A-Z][A-Za-z\s]{3,25})',
]

# KCSE patterns
KCSE_INDEX_PATTERNS = [
    r'\b(\d{11})\b',
    r'(?:Index|INDEX)[:\s#]*(\d{10,12})',
]

# Year must be full 4-digit year starting with 19 or 20
KCSE_YEAR_PATTERNS = [
    r'\b((?:19|20)\d{2})\b',
    r'(?:Year|MWAKA)[:\s]*(20\d{2})',
]

# Grade patterns — most specific first
KCSE_GRADE_PATTERNS = [
    r'(?:Mean Grade|WASTANI)[:\s]*(A-|A|B\+|B-|B|C\+|C-|C|D\+|D-|D|E)\b',
    r'(?:Grade|DARAJA)[:\s]*(A-|A|B\+|B-|B|C\+|C-|C|D\+|D-|D|E)\b',
    r'\b(A-|B\+|B-|C\+|C-|D\+|D-|A|B|C|D|E)\b',
]

# Passport patterns
PASSPORT_NUMBER_PATTERNS = [
    r'\b([A-Z]{1,2}\d{7})\b',
    r'(?:Passport|PASIPOTI)[:\s#]*([A-Z]{1,2}\d{6,8})',
]

# MRZ patterns
MRZ_PATTERNS = [
    r'(P<KEN[A-Z<]+)',
    r'([A-Z0-9<]{20,})',
]

# Valid KCSE grades set
VALID_GRADES = {"A", "A-", "B+", "B", "B-", "C+", "C", "C-", "D+", "D", "D-", "E"}


# ── PARSER CLASS ──────────────────────────────────────────────────────────────

class FieldParser:
    """
    Parses specific fields from raw OCR text for each document type.
    Uses regex with fallback patterns for noisy OCR output.
    """

    def _clean_text(self, text):
        """Basic cleanup of OCR output"""
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'[^\w\s/\-\.\,\:\#<]', ' ', text)
        return text.strip()

    def _find_first_match(self, text, patterns):
        """Try patterns in order, return first match found"""
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
        """Return all unique matches across all patterns"""
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

    def _get_display_value(self, field_dict):
        """Get the best display value from a field dict"""
        for key in ["value", "values", "lines", "all_found"]:
            val = field_dict.get(key)
            if val:
                return val
        return None

    def _validate_id_number(self, id_num):
        """Kenyan National ID is 7-9 digits"""
        if not id_num:
            return False
        return bool(re.match(r'^\d{7,9}$', str(id_num).strip()))

    def _validate_kcse_year(self, year):
        """KCSE started in 1989 — must be exactly 4 digits"""
        if not year:
            return False
        try:
            year_str = str(year).strip()
            if len(year_str) != 4:
                return False
            y = int(year_str)
            return 1989 <= y <= 2030
        except ValueError:
            return False

    def _validate_passport_number(self, num):
        """Kenyan passport: 1-2 letters + 7 digits"""
        if not num:
            return False
        return bool(re.match(r'^[A-Z]{1,2}\d{7}$', str(num).strip()))

    # ── NATIONAL ID PARSER ────────────────────────────────────────────────────

    def parse_national_id(self, ocr_result):
        """Extract fields from National ID OCR text"""
        text      = ocr_result.get("raw_text", "")
        lines     = ocr_result.get("lines", [])
        full_text = "\n".join(lines) + "\n" + text

        fields = {}

        # Kenya identifier
        fields["is_kenyan_document"] = bool(
            re.search(r'KENYA|JAMHURI|REPUBLIC', full_text, re.IGNORECASE)
        )

        # ID Number
        id_num = self._find_first_match(full_text, ID_NUMBER_PATTERNS)
        fields["id_number"] = {
            "value": id_num,
            "found": id_num is not None,
            "valid": self._validate_id_number(id_num)
        }

        # Dates
        dates    = self._find_all_matches(full_text, DATE_PATTERNS)
        has_dates = len(dates) > 0
        fields["dates"] = {
            "all_found":    dates,
            "count":        len(dates),
            "found":        has_dates,
            "likely_dob":   dates[0] if len(dates) > 0 else None,
            "likely_issue": dates[1] if len(dates) > 1 else None,
        }

        # Sex — full words only
        sex = self._find_first_match(full_text, SEX_PATTERNS)
        fields["sex"] = {
            "value": sex,
            "found": sex is not None
        }

        # District
        district = self._find_first_match(full_text, DISTRICT_PATTERNS)
        fields["district"] = {
            "value": district,
            "found": district is not None
        }

        # Extraction score
        key_fields  = ["id_number", "dates", "sex"]
        found_count = sum(1 for f in key_fields if fields[f]["found"])
        fields["extraction_score"] = {
            "fields_found": found_count,
            "total_fields": len(key_fields),
            "percentage":   round(found_count / len(key_fields) * 100)
        }

        return fields

    # ── KCSE PARSER ───────────────────────────────────────────────────────────

    def parse_kcse_certificate(self, ocr_result):
        """Extract fields from KCSE Certificate OCR text"""
        text      = ocr_result.get("raw_text", "")
        lines     = ocr_result.get("lines", [])
        full_text = "\n".join(lines) + "\n" + text

        fields = {}

        # KCSE identifier
        fields["is_kcse_document"] = bool(
            re.search(
                r'KCSE|KENYA CERTIFICATE|SECONDARY EDUCATION|KNEC',
                full_text, re.IGNORECASE
            )
        )

        # Index number
        index = self._find_first_match(full_text, KCSE_INDEX_PATTERNS)
        fields["index_number"] = {
            "value": index,
            "found": index is not None,
            "valid": len(str(index)) >= 10 if index else False
        }

        # Year — must pass validation to count as found
        year       = self._find_first_match(full_text, KCSE_YEAR_PATTERNS)
        year_valid = self._validate_kcse_year(year)
        fields["year"] = {
            "value": year,
            "found": year is not None and year_valid,
            "valid": year_valid
        }

        # Mean grade — try labelled patterns first, then any valid grade
        mean_grade = self._find_first_match(full_text, KCSE_GRADE_PATTERNS[:2])
        if not mean_grade:
            mean_grade = self._find_first_match(full_text, KCSE_GRADE_PATTERNS[2:])
        grade_valid = mean_grade in VALID_GRADES if mean_grade else False
        fields["mean_grade"] = {
            "value": mean_grade,
            "found": mean_grade is not None and grade_valid,
            "valid": grade_valid
        }

        # All subject grades — filter to valid grades only
        all_grades = self._find_all_matches(full_text, KCSE_GRADE_PATTERNS)
        all_grades = [g for g in all_grades if g in VALID_GRADES]
        fields["all_grades"] = {
            "values": all_grades,
            "count":  len(all_grades),
            "found":  len(all_grades) > 0
        }

        # Extraction score
        key_fields  = ["index_number", "year", "mean_grade"]
        found_count = sum(1 for f in key_fields if fields[f]["found"])
        fields["extraction_score"] = {
            "fields_found": found_count,
            "total_fields": len(key_fields),
            "percentage":   round(found_count / len(key_fields) * 100)
        }

        return fields

    # ── PASSPORT PARSER ───────────────────────────────────────────────────────

    def parse_passport(self, ocr_result):
        """Extract fields from Passport OCR text"""
        text      = ocr_result.get("raw_text", "")
        lines     = ocr_result.get("lines", [])
        full_text = "\n".join(lines) + "\n" + text

        fields = {}

        # Passport identifier
        fields["is_passport_document"] = bool(
            re.search(r'PASSPORT|PASIPOTI|KEN|KENYA', full_text, re.IGNORECASE)
        )

        # Passport number
        passport_num = self._find_first_match(full_text, PASSPORT_NUMBER_PATTERNS)
        fields["passport_number"] = {
            "value": passport_num,
            "found": passport_num is not None,
            "valid": self._validate_passport_number(passport_num)
        }

        # MRZ lines
        mrz = self._find_all_matches(full_text, MRZ_PATTERNS)
        fields["mrz"] = {
            "lines": mrz,
            "found": len(mrz) > 0,
            "count": len(mrz)
        }

        # Dates
        dates     = self._find_all_matches(full_text, DATE_PATTERNS)
        has_dates = len(dates) > 0
        fields["dates"] = {
            "all_found":     dates,
            "found":         has_dates,
            "count":         len(dates),
            "likely_dob":    dates[0]  if len(dates) > 0 else None,
            "likely_expiry": dates[-1] if len(dates) > 1 else None,
        }

        # Nationality — check full text AND MRZ lines
        mrz_text = " ".join(mrz)
        combined = full_text + " " + mrz_text
        fields["nationality"] = {
            "found": bool(
                re.search(r'KENYAN|KENYA|P<KEN|KEN', combined, re.IGNORECASE)
            )
        }

        # Extraction score
        key_fields  = ["passport_number", "dates", "nationality"]
        found_count = sum(1 for f in key_fields if fields[f]["found"])
        fields["extraction_score"] = {
            "fields_found": found_count,
            "total_fields": len(key_fields),
            "percentage":   round(found_count / len(key_fields) * 100)
        }

        return fields

    # ── AUTO PARSER ───────────────────────────────────────────────────────────

    def parse(self, ocr_result, doc_type):
        """Auto-route to correct parser based on doc_type"""
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
    from src.ocr.text_extractor import extract_text

    print("=" * 60)
    print("FIELD PARSER TEST")
    print("=" * 60)

    parser = FieldParser()

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

        print(f"\n--- {doc_type.upper()} ---")
        print(f"Image: {images[0].name}")

        ocr_result = extract_text(str(images[0]), doc_type=doc_type)
        print(f"OCR confidence : {ocr_result.get('confidence', 0)}%")
        print(f"Words found    : {ocr_result.get('word_count', 0)}")

        fields = parser.parse(ocr_result, doc_type)

        print("Fields extracted:")
        for field, value in fields.items():
            if field == "extraction_score":
                score = value
                print(f"  SCORE: {score['fields_found']}/{score['total_fields']} "
                      f"fields ({score['percentage']}%)")
            elif isinstance(value, dict):
                found = value.get("found", False)
                # Get best display value
                display = None
                for key in ["value", "values", "lines", "all_found"]:
                    candidate = value.get(key)
                    if candidate:
                        display = candidate
                        break
                # Only ✅ if truly found AND has a real value
                status = "✅" if (found and display) else "❌"
                print(f"  {status} {field}: {display}")
            else:
                print(f"  {'✅' if value else '❌'} {field}: {value}")

    print("\n✅ Field parser working!")
    print("=" * 60)