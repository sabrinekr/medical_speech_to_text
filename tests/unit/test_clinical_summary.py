"""Unit tests for ClinicalSummary model."""

import pytest
from pydantic import ValidationError

from medical_transcription.models.clinical_summary import ClinicalSummary


class TestClinicalSummary:
    """Test ClinicalSummary Pydantic model."""

    def test_valid_clinical_summary(self, sample_clinical_summary):
        """Test creation of valid clinical summary."""
        summary = ClinicalSummary(**sample_clinical_summary)

        assert summary.patient_complaint == sample_clinical_summary["patient_complaint"]
        assert summary.findings == sample_clinical_summary["findings"]
        assert summary.diagnosis == sample_clinical_summary["diagnosis"]
        assert summary.next_steps == sample_clinical_summary["next_steps"]
        assert summary.medications == sample_clinical_summary["medications"]
        assert summary.additional_notes == sample_clinical_summary["additional_notes"]

    def test_clinical_summary_missing_required_field(self):
        """Test that missing required fields raise validation error."""
        with pytest.raises(ValidationError):
            ClinicalSummary(
                patient_complaint="Test",
                findings="Test",
                diagnosis="Test"
                # Missing next_steps, medications, additional_notes
            )

    def test_clinical_summary_empty_lists_allowed(self):
        """Test that empty lists are allowed for optional fields."""
        summary = ClinicalSummary(
            patient_complaint="Kopfschmerzen",
            findings=["Keine besonderen Befunde"],
            diagnosis="Spannungskopfschmerz",
            next_steps=[],
            medications=[],
            additional_notes=""
        )

        assert summary.next_steps == []
        assert summary.medications == []
        assert summary.additional_notes == ""

    def test_clinical_summary_to_dict(self, sample_clinical_summary):
        """Test conversion to dictionary."""
        summary = ClinicalSummary(**sample_clinical_summary)
        summary_dict = summary.model_dump()

        assert isinstance(summary_dict, dict)
        assert summary_dict["patient_complaint"] == sample_clinical_summary["patient_complaint"]
        assert summary_dict["diagnosis"] == sample_clinical_summary["diagnosis"]

    def test_clinical_summary_json_serialization(self, sample_clinical_summary):
        """Test JSON serialization."""
        summary = ClinicalSummary(**sample_clinical_summary)
        json_str = summary.model_dump_json()

        assert isinstance(json_str, str)
        assert "patient_complaint" in json_str
        assert "diagnosis" in json_str

    def test_clinical_summary_german_characters(self):
        """Test handling of German special characters."""
        summary = ClinicalSummary(
            patient_complaint="Rückenschmerzen mit Ausstrahlung",
            findings=["Druckschmerz über L4/L5", "eingeschränkte Mobilität"],
            diagnosis="Verdacht auf Bandscheibenvorfall",
            next_steps=["MRT-Überweisung", "Schmerztherapie"],
            medications=["Ibuprofen 600mg 3x täglich"],
            additional_notes="Patient äußert starke Beschwerden"
        )

        # Verify German umlauts are preserved
        assert "ü" in summary.patient_complaint
        assert "ä" in summary.additional_notes
        assert any("ü" in f for f in summary.findings)
