"""
Unit tests for template loader
"""
import pytest
from src.utils.template_loader import TemplateLoader


def test_template_loader_initialization():
    """Test TemplateLoader initializes correctly"""
    loader = TemplateLoader()
    assert loader is not None
    assert isinstance(loader.templates, dict)


def test_templates_loaded():
    """Test that templates are loaded"""
    loader = TemplateLoader()
    assert len(loader.templates) >= 1, "No templates loaded"


def test_get_national_id_template():
    """Test getting National ID template"""
    loader = TemplateLoader()
    if "kenyan_national_id" in loader.templates:
        template = loader.get_template("kenyan_national_id")
        assert template is not None
        assert "document_type" in template
        assert template["document_type"] == "kenyan_national_id"


def test_get_security_features():
    """Test getting security features"""
    loader = TemplateLoader()
    if "kenyan_national_id" in loader.templates:
        features = loader.get_security_features("kenyan_national_id")
        assert isinstance(features, dict)
        assert len(features) > 0
        assert "hologram" in features


def test_get_data_fields():
    """Test getting data fields"""
    loader = TemplateLoader()
    if "kenyan_national_id" in loader.templates:
        fields = loader.get_data_fields("kenyan_national_id")
        assert isinstance(fields, dict)
        assert "id_number" in fields


def test_list_supported_documents():
    """Test listing supported documents"""
    loader = TemplateLoader()
    docs = loader.list_supported_documents()
    assert isinstance(docs, list)
    assert len(docs) > 0


def test_invalid_document_type():
    """Test error handling for invalid document type"""
    loader = TemplateLoader()
    with pytest.raises(ValueError):
        loader.get_template("invalid_document")