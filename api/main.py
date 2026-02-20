"""
Document Verification API
FastAPI REST endpoint for the Kenya Document Verification System
"""
import sys
import shutil
import tempfile
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn


# ── APP SETUP ─────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Kenya Document Verification API",
    description="""
    AI-powered verification system for Kenyan government documents.

    ## Supported Documents
    - 🪪 **National ID** — Kenyan National Identity Card
    - 📜 **KCSE Certificate** — Kenya Certificate of Secondary Education
    - 📕 **Passport** — Kenyan Passport

    ## How It Works
    1. Upload a document image
    2. CNN classifies the document type
    3. OCR extracts text fields
    4. Rule-based validator checks field formats
    5. Combined score produces a GENUINE / UNCERTAIN / FAKE verdict
    """,
    version="1.0.0",
    contact={
        "name": "Victor Ino",
        "url":  "https://github.com/Victorinoh/kenya-document-verification"
    }
)

# Allow all origins for development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── CONSTANTS ─────────────────────────────────────────────────────────────────

SUPPORTED_DOC_TYPES = {
    "national_id",
    "kcse_certificate",
    "passport"
}

SUPPORTED_EXTENSIONS = {
    ".jpg", ".jpeg", ".png",
    ".JPG", ".JPEG", ".PNG"
}


# ── ROUTES ────────────────────────────────────────────────────────────────────

@app.get("/", tags=["General"])
def root():
    """API welcome message"""
    return {
        "name":    "Kenya Document Verification API",
        "version": "1.0.0",
        "status":  "running",
        "docs":    "/docs",
        "health":  "/health",
        "verify":  "/verify",
        "github":  "https://github.com/Victorinoh/kenya-document-verification"
    }


@app.get("/health", tags=["General"])
def health_check():
    """
    Health check endpoint.
    Returns API status and model availability.
    """
    classifier_exists = Path("models/saved_models/document_classifier.h5").exists()
    detector_exists   = Path("models/saved_models/authenticity_detector.h5").exists()
    both_ready        = classifier_exists and detector_exists

    return {
        "status": "healthy" if both_ready else "degraded",
        "models": {
            "document_classifier":   "available" if classifier_exists else "missing",
            "authenticity_detector": "available" if detector_exists   else "missing",
        },
        "supported_doc_types": sorted(list(SUPPORTED_DOC_TYPES)),
        "version": "1.0.0"
    }


@app.get("/supported", tags=["General"])
def get_supported_types():
    """List all supported document types with descriptions"""
    return {
        "supported_document_types": [
            {
                "id":          "national_id",
                "name":        "Kenyan National ID",
                "description": "National Identity Card issued by the Government of Kenya",
                "fields":      ["id_number", "full_name", "date_of_birth", "sex", "district"]
            },
            {
                "id":          "kcse_certificate",
                "name":        "KCSE Certificate",
                "description": "Kenya Certificate of Secondary Education issued by KNEC",
                "fields":      ["index_number", "candidate_name", "year", "mean_grade", "subjects"]
            },
            {
                "id":          "passport",
                "name":        "Kenyan Passport",
                "description": "Travel document issued by the Government of Kenya",
                "fields":      ["passport_number", "full_name", "date_of_birth", "expiry_date", "mrz"]
            }
        ]
    }


@app.post("/verify", tags=["Verification"])
async def verify_document_endpoint(
    file:     UploadFile = File(...,  description="Document image (JPG or PNG)"),
    doc_type: str        = Form(None, description="national_id | kcse_certificate | passport")
):
    """
    Verify a document image.

    Upload a photo of a Kenyan government document and receive:
    - **verdict**: GENUINE | UNCERTAIN | FAKE
    - **final_score**: 0-100 confidence score
    - **layers**: breakdown of CNN and OCR scores
    - **validation**: which rule checks passed or failed

    If **doc_type** is not provided it will be auto-detected by the CNN.
    """
    # ── Validate file extension ───────────────────────────────────────────────
    filename = file.filename or "document.jpg"
    file_ext = Path(filename).suffix
    if file_ext not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{file_ext}'. "
                   f"Please upload a JPG or PNG image."
        )

    # ── Validate doc_type ─────────────────────────────────────────────────────
    if doc_type and doc_type not in SUPPORTED_DOC_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported doc_type '{doc_type}'. "
                   f"Choose from: {sorted(list(SUPPORTED_DOC_TYPES))}"
        )

    # ── Save uploaded file to a temp location ─────────────────────────────────
    temp_dir  = tempfile.mkdtemp()
    temp_path = Path(temp_dir) / filename

    try:
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # ── Run the verification pipeline ─────────────────────────────────────
        from src.pipeline import verify_document

        result = verify_document(
            image_path=temp_path,
            doc_type=doc_type,
            verbose=False
        )

        # ── Check for pipeline errors ─────────────────────────────────────────
        if result.get("verdict") == "ERROR":
            raise HTTPException(
                status_code=500,
                detail=result.get("error", "Verification failed")
            )

        # ── Build clean API response ──────────────────────────────────────────
        response = {
            # Core verdict
            "verdict":      result["verdict"],
            "verdict_icon": result["verdict_icon"],
            "final_score":  result["final_score"],
            "confidence":   result["confidence"],

            # Document info
            "doc_type":     result["doc_type"],
            "display_name": result["display_name"],
            "duration_ms":  result["duration_ms"],

            # Layer breakdown
            "layers": {
                "classifier": {
                    "predicted_type": result["layers"]["classifier"].get("predicted_type"),
                    "confidence":     result["layers"]["classifier"].get("confidence"),
                    "score":          result["layers"]["classifier"].get("score"),
                },
                "detector": {
                    "genuine_probability": result["layers"]["detector"].get("genuine_probability"),
                    "fake_probability":    result["layers"]["detector"].get("fake_probability"),
                    "score":               result["layers"]["detector"].get("score"),
                },
                "ocr": {
                    "checks_passed": result["layers"]["ocr"].get("checks_passed"),
                    "checks_total":  result["layers"]["ocr"].get("checks_total"),
                    "score":         result["layers"]["ocr"].get("score"),
                }
            },

            # Validation summary
            "validation": {
                "verdict":       result["validation"]["verdict"],
                "overall_score": result["validation"]["overall_score"],
                "checks_passed": result["validation"]["checks_passed"],
                "checks_total":  result["validation"]["checks_total"],
                "failed_checks": [
                    c["check"] for c in result["validation"]["failed_checks"]
                ]
            },

            # OCR summary
            "ocr_summary": {
                "text_confidence": result["ocr"]["text_confidence"],
                "word_count":      result["ocr"]["word_count"],
                "fields_found":    result["ocr"]["extraction_score"].get("fields_found", 0),
                "fields_total":    result["ocr"]["extraction_score"].get("total_fields", 0),
            }
        }

        return JSONResponse(content=response)

    except HTTPException:
        raise   # Re-raise HTTP exceptions as-is

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Verification failed: {str(e)}"
        )

    finally:
        # Always clean up temp files
        shutil.rmtree(temp_dir, ignore_errors=True)


# ── SERVER ENTRY POINT ────────────────────────────────────────────────────────

if __name__ == "__main__":
    uvicorn.run(
        "api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )