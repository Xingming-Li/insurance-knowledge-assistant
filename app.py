"""NordicPaws Försäkring — minimal Streamlit demo (Swedish, v1).

Thin UI over the existing baseline RAG pipeline: InsuranceRetriever (dense) +
generate_answer. No auth, no database, no conversation memory, no analytics.

Run:
    PYTHONPATH=src streamlit run app.py
"""
from __future__ import annotations

import sys
from pathlib import Path

# Make ``src`` importable when launched via ``streamlit run app.py``.
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import streamlit as st

from config import get_settings
from generation.answer import generate_answer
from retrieval.retriever import InsuranceRetriever
from ui_format import is_abstention

TITLE = "Pawli — NordicPaws AI-assistent"

# Clickable example/FAQ questions (Swedish). A spread of dog/cat, lookup and
# calculation, so the demo shows the assistant's range.
EXAMPLE_QUESTIONS = [
    "Hur lång är karenstiden för sjukdom för en hund?",
    "Vilka handlingar behöver kunden skicka in vid en skadeanmälan?",
    "Vad är det årliga ersättningstaket för kattförsäkring på nivå Premium?",
    "Täcks akut veterinärvård utanför ordinarie öppettider?",
    "En hund på nivå Premium ska genomgå en planerad operation som beräknas kosta 30 000 SEK. Krävs förhandsgodkännande, hur mycket ersätts och vad blir kundens självrisk?"
]


@st.cache_resource(show_spinner=False)
def _get_retriever() -> InsuranceRetriever:
    # Baseline dense retriever (the v1 default), cached across reruns.
    return InsuranceRetriever(settings=get_settings())


def main() -> None:
    st.set_page_config(page_title=TITLE, page_icon="🐾")
    st.title(TITLE)
    st.caption(
        "⚠️ Demo: NordicPaws Försäkring är ett **påhittat** bolag och allt innehåll "
        "är fiktivt — endast för demonstration. Detta är inte verklig rådgivning."
    )

    settings = get_settings()
    if not settings.openai_api_key:
        st.error(
            "OPENAI_API_KEY saknas. Lägg till nyckeln i miljön eller i en .env-fil "
            "för att köra demon."
        )
        return

    st.markdown("**Exempelfrågor** — klicka för att prova:")
    clicked = None
    for i, example in enumerate(EXAMPLE_QUESTIONS):
        if st.button(example, key=f"example_{i}", use_container_width=True):
            clicked = example

    typed = st.chat_input("Ställ en fråga om hund- eller kattförsäkring …")

    # Either a typed question or a clicked example drives the same answer flow.
    question = typed or clicked
    if not question:
        return

    st.chat_message("user").write(question)
    with st.chat_message("assistant"):
        with st.spinner("Söker i villkoren …"):
            retriever = _get_retriever()
            docs = retriever.retrieve_for_answer(question)
            result = generate_answer(question, docs, settings=settings)

        if is_abstention(result.answer, result.used_evidence):
            st.warning(result.answer)
        else:
            # The answer already carries inline [Källa N] and ends with its own
            # "Källor:" list of the sources actually used — render it as-is.
            st.markdown(result.answer)

        if docs:
            with st.expander("Visa hämtat underlag"):
                for i, doc in enumerate(docs, 1):
                    m = doc.metadata or {}
                    st.markdown(
                        f"**{i}. {m.get('title', '(okänd titel)')}** — "
                        f"{m.get('document_id', '?')} · "
                        f"Avsnitt {m.get('section', '–')} · "
                        f"v{m.get('version', '?')}"
                    )
                    st.caption(doc.page_content[:500])


if __name__ == "__main__":
    main()
