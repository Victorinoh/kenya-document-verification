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
    r'\b(\d{8})\b',                          # Standard 8-digit ID
    r'(?:ID|No|Number|Namba)[:\s#]*(\d{7,9})',  # With label
    r'(?:NAMBA|NAMBARI)[:\s]*(\d{7,9})',     # Swahili label
]

# Date patterns (DD/MM/YYYY, DD-MM-YYYY, DD.MM.YYYY)
DATE_PATTERNS = [
    r'\b(\d{1,2}[/\-\.]\d{1,2}[/\-\.]\d{4})\b',
    r'\b(\d{1,2}\s+\w+\s+\d{4})\b',         # DD Month YYYY
    r'(?:DOB|Date of Birth|TAREHE)[:\s]*(\d{1,2}[/\-\.]\d{1,2}[/\-\.]\d{4})',
]

# Name patterns
NAME_PATTERNS = [
    r'(?:Name|Jina|JINA)[:\s]+([A-Z][A-Z\s]{3,40})',
    r'(?:Surname|SURNAME)[:\s]+([A-Z][A-Z\s]{2,25})',
    r'^([A-Z]{2,}\s+[A-Z]{2,}(?:\s+[A-Z]{2,})?)$',  # ALL CAPS name line
]

# Sex/Gender patterns
SEX_PATTERNS = [
    r'\b(MALE|FEMALE|M|F)\b',
    r'(?:Sex|Gender|JINSIA)[:\s]*(MALE|FEMALE|M|F)',
]

# District/Place of birth
DISTRICT_PATTERNS = [
    r'(?:District|WILAYA|Place of Birth)[:\s]+([A-Z][A-Za-z\s]{3,25})',
    r'(?:County|KAUNTI)[:\s]+([A-Z][A-Za-z\s]{3,25})',
]

# KCSE patterns
KCSE_INDEX_PATTERNS = [
    r'\b(\d{11})\b',                         # 11-digit index number
    r'(?:Index|INDEX)[:\s#]*(\d{10,12})',
]

KCSE_YEAR_PATTERNS = [
    r'\b(19|20)(\d{2})\b',                   # Year 19xx or 20xx
    r'(?:Year|MWAKA)[:\s]*(20\d{2})',
]

KCSE_GRADE_PATTERNS = [
    r'\b(A|A-|B\+|B|B-|C\+|C|C-|D\+|D|D-|E)\b',
    r'(?:Grade|DARAJA)[:\s]*(A|A-|B\+|B|B-|C\+|C|C-|D\+|D|D-|E)',
    r'(?:Mean Grade|WASTANI)[:\s]*(A|A-|B\+|B|B-|C\+|C|C-|D\+|D|D-|E)',
]

# Passport patterns
PASSPORT_NUMBER_PATTERNS = [
    r'\b([A-Z]{1,2}\d{7})\b',               # Standard Kenyan passport A1234567
    r'(?:Passport|PASIPOTI)[:\s#]*([A-Z]{1,2}\d{6,8})',
]

MRZ_PATTERNS = [
    r'([A-Z0-9<]{44})',                      # MRZ line (44 chars)
    r'P<KEN([A-Z<]+)',                       # MRZ surname field
]


# ── PARSER CLASS ──────────────────────────────────────────────────────────────

class FieldParser:
    """
    Parses specific fields from raw OCR text for each document type.
    Uses regex with fallback patterns for noisy OCR output.
    """

    def __init__(self):
        self.ocr_corrections = {
            # Common OCR mistakes on Kenyan documents
            '0': ['O', 'o', 'Q'],
            'I': ['1', 'l', '|'],
            'S': ['5', '$'],
            'B': ['8', '6'],
            'G': ['6', '9'],
        }

    def _clean_text(self, text):
        """Basic cleanup of OCR output"""
        # Remove multiple spaces
        text = re.sub(r'\s+', ' ', text)
        # Remove common OCR artifacts
        text = re.sub(r'[^\w\s/\-\.\,\:\#<]', ' ', text)
        return text.strip()

    def _find_first_match(self, text, patterns):
        """Try patterns in order, return first match found"""
        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE | re.MULTILINE)
            if matches:
                # Return first non-empty match
                for m in matches:
                    val = m.strip() if isinstance(m, str) else m[0].strip()
                    if val:
                        return val
        return None

    def _find_all_matches(self, text, patterns):
        """Return all matches across all patterns"""
        results = []
        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE | re.MULTILINE)
            for m in matches:
                val = m.strip() if isinstance(m, str) else m[0].strip()
                if val and val not in results:
                    results.append(val)
        return results

    # ── NATIONAL ID PARSER ────────────────────────────────────────────────────

    def parse_national_id(self, ocr_result):
        """
        Extract fields from National ID OCR text.

        Args:
            ocr_result: dict from text_extractor.extract_text()

        Returns:
            dict of extracted fields with confidence indicators
        """
        text     = ocr_result.get("raw_text", "")
        lines    = ocr_result.get("lines", [])
        full_text = "\n".join(lines) + "\n" + text

        fields = {}

        # Check for Kenya identifier
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

        # Dates (collect all, label as DOB / issue / expiry by position)
        dates = self._find_all_matches(full_text, DATE_PATTERNS)
        fields["dates"] = {
            "all_found": dates,
            "count":     len(dates),
            "found":     len(dates) > 0
        }
        if dates:
            fields["dates"]["likely_dob"] = dates[0]
            if len(dates) > 1:
                fields["dates"]["likely_issue"] = dates[1]

        # Sex
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

        # Summary score — how many key fields were found
        key_fields     = ["id_number", "dates", "sex"]
        found_count    = sum(1 for f in key_fields if fields[f]["found"])
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

        # Check for KCSE identifier
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
            "valid": len(index) >= 10 if index else False
        }

        # Year
        year = self._find_first_match(full_text, KCSE_YEAR_PATTERNS)
        fields["year"] = {
            "value": year,
            "found": year is not None,
            "valid": self._validate_kcse_year(year)
        }

        # Mean grade
        grade = self._find_first_match(full_text, KCSE_GRADE_PATTERNS)
        fields["mean_grade"] = {
            "value": grade,
            "found": grade is not None,
            "valid": grade in ["A", "A-", "B+", "B", "B-", "C+",
                               "C", "C-", "D+", "D", "D-", "E"]
                     if grade else False
        }

        # All grades found (individual subjects)
        all_grades = self._find_all_matches(full_text, KCSE_GRADE_PATTERNS)
        fields["all_grades"] = {
            "values": all_grades,
            "count":  len(all_grades),
            "found":  len(all_grades) > 0
        }

        # Summary
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

        # Check for passport identifier
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
        dates = self._find_all_matches(full_text, DATE_PATTERNS)
        fields["dates"] = {
            "all_found": dates,
            "found":     len(dates) > 0,
            "count":     len(dates)
        }
        if dates:
            fields["dates"]["likely_dob"]    = dates[0]
        if len(dates) > 1:
            fields["dates"]["likely_expiry"] = dates[-1]

        # Nationality
        fields["nationality"] = {
            "found": bool(re.search(r'KENYAN|KENYA|KEN', full_text, re.IGNORECASE))
        }

        # Summary
        key_fields  = ["passport_number", "dates", "nationality"]
        found_count = sum(1 for f in key_fields if fields[f]["found"])
        fields["extraction_score"] = {
            "fields_found": found_count,
            "total_fields": len(key_fields),
            "percentage":   round(found_count / len(key_fields) * 100)
        }

        return fields

    # ── VALIDATORS ────────────────────────────────────────────────────────────

    def _validate_id_number(self, id_num):
        """Kenyan National ID is 7-9 digits"""
        if not id_num:
            return False
        return bool(re.match(r'^\d{7,9}$', str(id_num).strip()))

    def _validate_kcse_year(self, year):
        """KCSE started in 1989"""
        if not year:
            return False
        try:
            y = int(str(year).strip())
            return 1989 <= y <= 2030
        except ValueError:
            return False

    def _validate_passport_number(self, num):
        """Kenyan passport: 1-2 letters + 7 digits"""
        if not num:
            return False
        return bool(re.match(r'^[A-Z]{1,2}\d{7}$', str(num).strip()))

    # ── AUTO PARSER ───────────────────────────────────────────────────────────

    def parse(self, ocr_result, doc_type):
        """
        Auto-route to the correct parser based on doc_type.

        Args:
            ocr_result: dict from text_extractor.extract_text()
            doc_type:   'national_id' | 'kcse_certificate' | 'passport'

        Returns:
            dict of extracted and validated fields
        """
        parsers = {
            "national_id":       self.parse_national_id,
            "national_ids":      self.parse_national_id,
            "kcse_certificate":  self.parse_kcse_certificate,
            "kcse_certificates": self.parse_kcse_certificate,
            "passport":          self.parse_passport,
            "passports":         self.parse_passport,
        }
        parser = parsers.get(doc_type)
        if not parser:
            return {"error": f"Unknown doc_type: {doc_type}"}
        return parser(ocr_result)


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
        ("data/raw/national_ids/genuine",       "national_id"),
        ("data/raw/kcse_certificates/genuine",   "kcse_certificate"),
        ("data/raw/passports/genuine",           "passport"),
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

        # Extract text
        ocr_result = extract_text(str(images[0]), doc_type=doc_type)
        print(f"OCR confidence : {ocr_result.get('confidence', 0)}%")
        print(f"Words found    : {ocr_result.get('word_count', 0)}")

        # Parse fields
        fields = parser.parse(ocr_result, doc_type)

        print(f"Fields extracted:")
        for field, value in fields.items():
            if field == "extraction_score":
                score = value
                print(f"  SCORE: {score['fields_found']}/{score['total_fields']} "
                      f"fields ({score['percentage']}%)")
            elif isinstance(value, dict):
                found = value.get("found", False)
                val   = value.get("value") or value.get("values") or value.get("lines")
                status = "✅" if found else "❌"
                print(f"  {status} {field}: {val}")
            else:
                print(f"  {'✅' if value else '❌'} {field}: {value}")

    print("\n✅ Field parser working!")
    print("=" * 60)