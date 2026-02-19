"""
Rule-Based Document Validators
Validates extracted fields against Kenyan document rules.
Produces a validation score and list of failed checks.
"""
import re
from datetime import datetime, date
from pathlib import Path


# ── CONSTANTS ─────────────────────────────────────────────────────────────────

VALID_KCSE_GRADES = {
    "A", "A-", "B+", "B", "B-", "C+",
    "C", "C-", "D+", "D", "D-", "E"
}
VALID_SEXES = {"MALE", "FEMALE", "M", "F"}


# ── VALIDATOR CLASS ───────────────────────────────────────────────────────────

class DocumentValidator:
    """
    Validates extracted document fields against known Kenyan rules.
    Returns a validation report with pass/fail per check and overall score.
    """

    # ── NATIONAL ID VALIDATOR ─────────────────────────────────────────────────

    def validate_national_id(self, fields, ocr_result=None):
        """Validate National ID fields"""
        checks = []
        passed = 0

        # 1. Kenya identifier present
        r = self._check(
            "Kenya identifier present",
            fields.get("is_kenyan_document", False),
            "JAMHURI YA KENYA / REPUBLIC OF KENYA not detected",
            critical=True
        )
        checks.append(r)
        if r["passed"]: passed += 1

        # 2. ID number found
        id_field = fields.get("id_number", {})
        r = self._check(
            "ID number extracted",
            id_field.get("found", False),
            "Could not extract ID number from document",
            critical=True
        )
        checks.append(r)
        if r["passed"]: passed += 1

        # 3. ID number format valid
        r = self._check(
            "ID number format valid (7-9 digits)",
            id_field.get("valid", False),
            f"ID number '{id_field.get('value')}' is not 7-9 digits",
            critical=True
        )
        checks.append(r)
        if r["passed"]: passed += 1

        # 4. Date of birth found
        dates_field = fields.get("dates", {})
        r = self._check(
            "Date of birth found",
            dates_field.get("found", False),
            "No date found on document",
            critical=False
        )
        checks.append(r)
        if r["passed"]: passed += 1

        # 5. DOB is a valid past date
        dob = dates_field.get("likely_dob")
        r = self._check(
            "Date of birth is valid past date",
            self._validate_past_date(dob),
            f"Date '{dob}' is invalid or in the future",
            critical=False
        )
        checks.append(r)
        if r["passed"]: passed += 1

        # 6. Sex field found
        sex_field = fields.get("sex", {})
        r = self._check(
            "Sex field found",
            sex_field.get("found", False),
            "Sex/gender field not detected",
            critical=False
        )
        checks.append(r)
        if r["passed"]: passed += 1

        # 7. Sex value is valid
        sex_val = str(sex_field.get("value", "")).upper()
        r = self._check(
            "Sex value is valid (MALE/FEMALE)",
            sex_val in VALID_SEXES,
            f"'{sex_val}' is not a valid sex value",
            critical=False
        )
        checks.append(r)
        if r["passed"]: passed += 1

        # 8. OCR confidence acceptable
        ocr_conf = ocr_result.get("confidence", 0) if ocr_result else 0
        r = self._check(
            "OCR confidence acceptable (>30%)",
            ocr_conf > 30,
            f"Very low OCR confidence: {ocr_conf}% — image may be too blurry",
            critical=False
        )
        checks.append(r)
        if r["passed"]: passed += 1

        return self._build_report("National ID", checks, passed)

    # ── KCSE VALIDATOR ────────────────────────────────────────────────────────

    def validate_kcse_certificate(self, fields, ocr_result=None):
        """Validate KCSE Certificate fields"""
        checks = []
        passed = 0

        # 1. KCSE identifier present
        r = self._check(
            "KCSE identifier present",
            fields.get("is_kcse_document", False),
            "KCSE / KNEC / Kenya Certificate not detected",
            critical=True
        )
        checks.append(r)
        if r["passed"]: passed += 1

        # 2. Index number found
        index_field = fields.get("index_number", {})
        r = self._check(
            "Index number extracted",
            index_field.get("found", False),
            "Could not extract candidate index number",
            critical=True
        )
        checks.append(r)
        if r["passed"]: passed += 1

        # 3. Index number length valid
        r = self._check(
            "Index number length valid (10-12 digits)",
            index_field.get("valid", False),
            f"Index '{index_field.get('value')}' has wrong length",
            critical=True
        )
        checks.append(r)
        if r["passed"]: passed += 1

        # 4. Year found
        year_field = fields.get("year", {})
        r = self._check(
            "Examination year found",
            year_field.get("found", False),
            "Could not extract examination year",
            critical=False
        )
        checks.append(r)
        if r["passed"]: passed += 1

        # 5. Year is valid KCSE year
        r = self._check(
            "Year is valid KCSE year (1989-present)",
            year_field.get("valid", False),
            f"Year '{year_field.get('value')}' is outside valid KCSE range",
            critical=False
        )
        checks.append(r)
        if r["passed"]: passed += 1

        # 6. Mean grade found
        grade_field = fields.get("mean_grade", {})
        r = self._check(
            "Mean grade found",
            grade_field.get("found", False),
            "Could not extract mean grade",
            critical=False
        )
        checks.append(r)
        if r["passed"]: passed += 1

        # 7. Mean grade is valid
        r = self._check(
            "Mean grade is valid Kenyan grade",
            grade_field.get("valid", False),
            f"'{grade_field.get('value')}' is not a valid KCSE grade",
            critical=False
        )
        checks.append(r)
        if r["passed"]: passed += 1

        # 8. Minimum 7 subject grades found
        all_grades  = fields.get("all_grades", {})
        grade_count = all_grades.get("count", 0)
        r = self._check(
            "Minimum 7 subject grades found",
            grade_count >= 7,
            f"Only {grade_count} grades detected — expect 7+ subjects",
            critical=False
        )
        checks.append(r)
        if r["passed"]: passed += 1

        return self._build_report("KCSE Certificate", checks, passed)

    # ── PASSPORT VALIDATOR ────────────────────────────────────────────────────

    def validate_passport(self, fields, ocr_result=None):
        """Validate Passport fields"""
        checks = []
        passed = 0

        # 1. Passport identifier present
        r = self._check(
            "Passport identifier present",
            fields.get("is_passport_document", False),
            "PASSPORT / KENYA not detected on document",
            critical=True
        )
        checks.append(r)
        if r["passed"]: passed += 1

        # 2. Passport number found
        pnum_field = fields.get("passport_number", {})
        r = self._check(
            "Passport number extracted",
            pnum_field.get("found", False),
            "Could not extract passport number",
            critical=True
        )
        checks.append(r)
        if r["passed"]: passed += 1

        # 3. Passport number format valid
        r = self._check(
            "Passport number format valid (A1234567)",
            pnum_field.get("valid", False),
            f"'{pnum_field.get('value')}' does not match Kenyan passport format",
            critical=True
        )
        checks.append(r)
        if r["passed"]: passed += 1

        # 4. MRZ detected
        mrz_field = fields.get("mrz", {})
        r = self._check(
            "MRZ (Machine Readable Zone) detected",
            mrz_field.get("found", False),
            "No MRZ lines detected — may indicate forgery or poor scan",
            critical=True
        )
        checks.append(r)
        if r["passed"]: passed += 1

        # 5. Nationality is Kenyan
        nat_field = fields.get("nationality", {})
        r = self._check(
            "Nationality is Kenyan",
            nat_field.get("found", False),
            "Kenyan nationality not confirmed in document",
            critical=False
        )
        checks.append(r)
        if r["passed"]: passed += 1

        # 6. Dates found
        dates_field = fields.get("dates", {})
        r = self._check(
            "Dates found (DOB and/or expiry)",
            dates_field.get("found", False),
            "No dates detected on passport",
            critical=False
        )
        checks.append(r)
        if r["passed"]: passed += 1

        # 7. Expiry date is in the future
        expiry = dates_field.get("likely_expiry")
        r = self._check(
            "Expiry date is in the future",
            self._validate_future_date(expiry),
            f"Passport may be expired: '{expiry}'",
            critical=False
        )
        checks.append(r)
        if r["passed"]: passed += 1

        return self._build_report("Passport", checks, passed)

    # ── AUTO VALIDATOR ────────────────────────────────────────────────────────

    def validate(self, fields, doc_type, ocr_result=None):
        """Auto-route to correct validator based on doc_type"""
        validators = {
            "national_id":       self.validate_national_id,
            "national_ids":      self.validate_national_id,
            "kcse_certificate":  self.validate_kcse_certificate,
            "kcse_certificates": self.validate_kcse_certificate,
            "passport":          self.validate_passport,
            "passports":         self.validate_passport,
        }
        validator_fn = validators.get(doc_type)
        if not validator_fn:
            return {"error": f"Unknown doc_type: {doc_type}"}
        return validator_fn(fields, ocr_result)

    # ── HELPERS ───────────────────────────────────────────────────────────────

    def _check(self, name, condition, failure_msg, critical=False):
        return {
            "check":    name,
            "passed":   bool(condition),
            "critical": critical,
            "message":  "" if condition else failure_msg
        }

    def _build_report(self, doc_type, checks, passed):
        total           = len(checks)
        critical        = [c for c in checks if c["critical"]]
        critical_passed = sum(1 for c in critical if c["passed"])
        critical_total  = len(critical)

        # Weighted score: critical checks = 60%, regular = 40%
        critical_score = (critical_passed / critical_total * 60
                          if critical_total else 0)
        regular_passed = passed - critical_passed
        regular_total  = max(total - critical_total, 1)
        regular_score  = regular_passed / regular_total * 40
        overall_score  = round(critical_score + regular_score)

        # Verdict
        if critical_passed < critical_total:
            verdict = "LIKELY FAKE"
        elif overall_score >= 70:
            verdict = "LIKELY GENUINE"
        elif overall_score >= 50:
            verdict = "UNCERTAIN"
        else:
            verdict = "LIKELY FAKE"

        return {
            "document_type":   doc_type,
            "verdict":         verdict,
            "overall_score":   overall_score,
            "checks_passed":   passed,
            "checks_total":    total,
            "critical_passed": critical_passed,
            "critical_total":  critical_total,
            "failed_checks":   [c for c in checks if not c["passed"]],
            "all_checks":      checks
        }

    def _validate_past_date(self, date_str):
        """Check if date string is a valid past date"""
        if not date_str:
            return False
        formats = [
            "%d/%m/%Y", "%d-%m-%Y", "%d.%m.%Y",
            "%d %B %Y", "%d %b %Y"
        ]
        for fmt in formats:
            try:
                d = datetime.strptime(str(date_str).strip(), fmt).date()
                return d < date.today()
            except ValueError:
                continue
        return False

    def _validate_future_date(self, date_str):
        """Check if date string is a valid future date"""
        if not date_str:
            return False
        formats = [
            "%d/%m/%Y", "%d-%m-%Y", "%d.%m.%Y",
            "%d %B %Y", "%d %b %Y"
        ]
        for fmt in formats:
            try:
                d = datetime.strptime(str(date_str).strip(), fmt).date()
                return d > date.today()
            except ValueError:
                continue
        return False

    def print_report(self, report):
        """Pretty print a validation report"""
        verdict_icon = {
            "LIKELY GENUINE": "✅",
            "UNCERTAIN":      "⚠️",
            "LIKELY FAKE":    "❌"
        }.get(report["verdict"], "❓")

        print(f"\n{'='*55}")
        print(f"  {report['document_type']} — Validation Report")
        print(f"{'='*55}")
        print(f"  Verdict       : {verdict_icon}  {report['verdict']}")
        print(f"  Overall Score : {report['overall_score']}%")
        print(f"  Checks Passed : {report['checks_passed']}/{report['checks_total']}")
        print(f"  Critical      : {report['critical_passed']}/{report['critical_total']}")
        print(f"\n  Detail:")
        for check in report["all_checks"]:
            if check["passed"]:
                icon = "  ✅"
            elif check["critical"]:
                icon = "  ❌"
            else:
                icon = "  ⚠️"
            print(f"{icon} {check['check']}")
            if not check["passed"] and check["message"]:
                print(f"       → {check['message']}")
        print(f"{'='*55}")


# ── QUICK TEST ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    from src.ocr.text_extractor import extract_text
    from src.ocr.field_parser   import FieldParser

    print("=" * 55)
    print("RULE-BASED VALIDATOR TEST")
    print("=" * 55)

    parser    = FieldParser()
    validator = DocumentValidator()

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

        # Extract → Parse → Validate
        ocr_result = extract_text(str(images[0]), doc_type=doc_type)
        fields     = parser.parse(ocr_result, doc_type)
        report     = validator.validate(fields, doc_type, ocr_result)
        validator.print_report(report)

    print("\n✅ Validator working!")