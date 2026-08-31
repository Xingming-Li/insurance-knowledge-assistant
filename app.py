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
from ui_format import format_source, is_abstention

TITLE = "Pawli — NordicPaws AI-assistent"


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

    question = st.chat_input("Ställ en fråga om hund- eller kattförsäkring …")
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
            st.markdown(result.answer)
            if result.citations:
                st.markdown("**Källor**")
                for c in result.citations:
                    st.markdown(f"- {format_source(c)}")

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
