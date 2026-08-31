from generation.answer import Citation
from generation.prompts import INSUFFICIENT_EVIDENCE_MESSAGE
from ui_format import format_source, is_abstention


def test_is_abstention_when_no_evidence():
    assert is_abstention("vad som helst", used_evidence=False) is True


def test_is_abstention_detects_canonical_message():
    assert is_abstention(INSUFFICIENT_EVIDENCE_MESSAGE, used_evidence=True) is True


def test_normal_answer_is_not_abstention():
    assert is_abstention("Karenstiden är 30 dagar.", used_evidence=True) is False


def test_format_source_full():
    c = Citation(
        title="Villkor Hundförsäkring 2026",
        document_id="NP-DOG-TERMS-2026",
        version="3.0",
        effective_date="2026-01-01",
        section="6. Karenstider",
        source="dog_insurance_terms_2026.md",
    )
    label = format_source(c)
    assert "Villkor Hundförsäkring 2026" in label
    assert "NP-DOG-TERMS-2026" in label
    assert "Avsnitt 6. Karenstider" in label
    assert "v3.0" in label
    assert ".md" not in label  # never expose raw file paths


def test_format_source_handles_missing_fields():
    c = Citation(title=None, document_id="NP-VET-2026", version=None,
                 effective_date=None, section=None, source=None)
    assert format_source(c) == "NP-VET-2026"
