import config
from langchain_core.documents import Document
from langchain_core.runnables import RunnableLambda

from generation.answer import build_citations, generate_answer


def _doc(section, content="text"):
    return Document(
        page_content=content,
        metadata={
            "document_id": "NP-DOG-TERMS-2026",
            "title": "Villkor Hundförsäkring 2026",
            "version": "3.0",
            "section": section,
            "source": "dog_insurance_terms_2026.md",
        },
    )


def test_abstains_without_evidence():
    res = generate_answer("Vad kostar hundförsäkring?", [], settings=config.get_settings())
    assert res.used_evidence is False
    assert res.citations == []
    assert "inte tillräcklig information" in res.answer


def test_build_citations_dedup_by_document_and_section():
    docs = [_doc("6. Karenstider"), _doc("6. Karenstider"), _doc("5. Självrisker")]
    citations = build_citations(docs)
    assert len(citations) == 2
    assert {c.section for c in citations} == {"6. Karenstider", "5. Självrisker"}
    # Citations expose structured metadata, not raw paths.
    assert citations[0].document_id == "NP-DOG-TERMS-2026"
    assert citations[0].title


def test_generates_answer_with_injected_llm():
    docs = [_doc("6. Karenstider", "Karenstiden för sjukdom är 30 dagar.")]
    fake_llm = RunnableLambda(lambda _prompt_value: "Karenstiden är 30 dagar.")
    res = generate_answer(
        "Hur lång är karenstiden?",
        docs,
        settings=config.get_settings(),
        llm=fake_llm,
    )
    assert res.used_evidence is True
    assert "30 dagar" in res.answer
    assert res.citations[0].document_id == "NP-DOG-TERMS-2026"
    assert res.citations[0].section == "6. Karenstider"
