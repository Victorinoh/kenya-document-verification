"""
Document template loader and manager
Loads JSON templates for each document type
"""
import json
from pathlib import Path


class TemplateLoader:
    """Load and manage document verification templates"""

    def __init__(self, template_dir: str = "data/templates"):
        self.template_dir = Path(template_dir)
        self.templates = {}
        self._load_all_templates()

    def _load_all_templates(self):
        """Load all JSON templates from template directory"""
        if not self.template_dir.exists():
            print(f"Warning: Template directory not found: {self.template_dir}")
            return

        for template_file in self.template_dir.glob("*.json"):
            try:
                with open(template_file, "r") as f:
                    template = json.load(f)
                doc_type = template.get("document_type", template_file.stem)
                self.templates[doc_type] = template
                print(f"  Loaded template: {doc_type}")
            except Exception as e:
                print(f"  Error loading {template_file.name}: {e}")

    def get_template(self, doc_type: str) -> dict:
        """Get template for a document type"""
        if doc_type not in self.templates:
            raise ValueError(f"No template found for: {doc_type}")
        return self.templates[doc_type]

    def get_security_features(self, doc_type: str) -> dict:
        """Get security features for a document type"""
        return self.get_template(doc_type).get("security_features", {})

    def get_data_fields(self, doc_type: str) -> dict:
        """Get data fields for a document type"""
        return self.get_template(doc_type).get("data_fields", {})

    def get_forgery_indicators(self, doc_type: str) -> list:
        """Get forgery indicators for a document type"""
        return self.get_template(doc_type).get("forgery_indicators", [])

    def get_validation_rules(self, doc_type: str) -> dict:
        """Get validation rules for a document type"""
        return self.get_template(doc_type).get("validation_rules", {})

    def list_supported_documents(self) -> list:
        """List all supported document types"""
        return list(self.templates.keys())


# Test the loader
if __name__ == "__main__":
    print("=" * 50)
    print("TEMPLATE LOADER TEST")
    print("=" * 50)

    loader = TemplateLoader()

    print(f"\nSupported documents: {len(loader.templates)}")
    for doc in loader.list_supported_documents():
        print(f"  - {doc}")

    # Test National ID template
    if "kenyan_national_id" in loader.templates:
        print("\nNational ID Security Features:")
        features = loader.get_security_features("kenyan_national_id")
        for f in features:
            print(f"  - {f}: priority={features[f]['priority']}")

        print("\nNational ID Data Fields:")
        fields = loader.get_data_fields("kenyan_national_id")
        for field in fields:
            print(f"  - {field}: {fields[field]['format_regex']}")

    print("\n" + "=" * 50)
    print("Template loader working correctly!")
    print("=" * 50)