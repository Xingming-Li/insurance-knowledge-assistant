from langchain_core.documents import Document

from retrieval.retriever import (
    CAT_TERMS_DOC,
    DOG_TERMS_DOC,
    MECHANICS_SECTION,
    is_calculation_question,
    supplement_deductible_mechanics,
)


def _doc(document_id, section):
    return Document(
        page_content=f"{document_id} {section}",
        metadata={"document_id": document_id, "section": section},
    )


class FakeStore:
    """Serves §5 chunks by metadata via .get(where=...) like Chroma."""

    def __init__(self, available):
        # available: list of (document_id, section)
        self._by_key = {(d, s): _doc(d, s) for d, s in available}
        self.get_calls = []

    def get(self, where=None):
        self.get_calls.append(where)
        conds = where["$and"]
        doc_id = conds[0]["document_id"]
        section = conds[1]["section"]
        hit = self._by_key.get((doc_id, section))
        if not hit:
            return {"documents": [], "metadatas": []}
        return {"documents": [hit.page_content], "metadatas": [hit.metadata]}


_BOTH_MECH = FakeStore([(DOG_TERMS_DOC, MECHANICS_SECTION), (CAT_TERMS_DOC, MECHANICS_SECTION)])

DOG_CALC = "Min hund har Premium, operationen kostar 30 000 SEK, hur mycket ersätts och vad blir självrisken?"
CAT_CALC = "Min katt: hur mycket ersätts och vad blir kundens självrisk?"


def _sections(docs):
    return [(d.metadata["document_id"], d.metadata["section"]) for d in docs]


# ---- calculation-question detection ---------------------------------------

def test_is_calculation_question():
    assert is_calculation_question(DOG_CALC) is True
    assert is_calculation_question("Hur lång är karenstiden för sjukdom?") is False
    assert is_calculation_question("Vilket är ersättningstaket för Premium?") is False


# ---- supplementation behaviour --------------------------------------------

def test_dog_calc_missing_dog5_is_supplemented():
    docs = [_doc(DOG_TERMS_DOC, "3. Omfattningsnivåer"), _doc("NP-VET-2026", "4. Förhandsgodkännande")]
    out = supplement_deductible_mechanics(DOG_CALC, docs, _BOTH_MECH)
    assert (DOG_TERMS_DOC, MECHANICS_SECTION) in _sections(out)


def test_cat_calc_missing_cat5_is_supplemented():
    docs = [_doc(CAT_TERMS_DOC, "3. Omfattningsnivåer")]
    out = supplement_deductible_mechanics(CAT_CALC, docs, _BOTH_MECH)
    assert (CAT_TERMS_DOC, MECHANICS_SECTION) in _sections(out)


def test_no_duplicate_when_already_present():
    docs = [_doc(DOG_TERMS_DOC, MECHANICS_SECTION), _doc("NP-VET-2026", "4. Förhandsgodkännande")]
    out = supplement_deductible_mechanics(DOG_CALC, docs, _BOTH_MECH)
    assert _sections(out).count((DOG_TERMS_DOC, MECHANICS_SECTION)) == 1
    assert _BOTH_MECH.get_calls[-1:] == _BOTH_MECH.get_calls[-1:]  # no crash


def test_non_calculation_question_not_supplemented():
    docs = [_doc(DOG_TERMS_DOC, "6. Karenstider")]
    out = supplement_deductible_mechanics("Hur lång är karenstiden?", docs, _BOTH_MECH)
    assert (DOG_TERMS_DOC, MECHANICS_SECTION) not in _sections(out)


def test_unknown_species_not_supplemented():
    docs = [_doc("NP-VET-2026", "4. Förhandsgodkännande")]
    out = supplement_deductible_mechanics(
        "Hur mycket ersätts och vad blir självrisken?", docs, _BOTH_MECH
    )
    assert not any(s == MECHANICS_SECTION for _, s in _sections(out))


def test_dog_calc_never_supplements_cat5_and_vice_versa():
    dog_out = supplement_deductible_mechanics(DOG_CALC, [], _BOTH_MECH)
    assert (DOG_TERMS_DOC, MECHANICS_SECTION) in _sections(dog_out)
    assert (CAT_TERMS_DOC, MECHANICS_SECTION) not in _sections(dog_out)

    cat_out = supplement_deductible_mechanics(CAT_CALC, [], _BOTH_MECH)
    assert (CAT_TERMS_DOC, MECHANICS_SECTION) in _sections(cat_out)
    assert (DOG_TERMS_DOC, MECHANICS_SECTION) not in _sections(cat_out)


def test_final_evidence_bounded_to_nine():
    docs = [_doc(DOG_TERMS_DOC, f"{i}. Sec{i}") for i in range(8)]  # full top-8, no §5
    out = supplement_deductible_mechanics(DOG_CALC, docs, _BOTH_MECH, max_chunks=8)
    assert len(out) == 9
    assert (DOG_TERMS_DOC, MECHANICS_SECTION) in _sections(out)  # §5 added
    