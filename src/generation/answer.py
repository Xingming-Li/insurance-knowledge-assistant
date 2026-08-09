"""Answer generation: turn a question + retrieved evidence into a grounded,
Swedish answer with structured source citations.

Citations expose document title, document ID, version and section — never raw
file paths — so a future UI can render proper references.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List, Optional, Sequence

from config import Settings, get_settings
from generation.prompts import INSUFFICIENT_EVIDENCE_MESSAGE, build_prompt


@dataclass(frozen=True)
class Citation:
    """A human-presentable reference to a piece of evidence."""

    title: Optional[str]
    document_id: Optional[str]
    version: Optional[str]
    section: Optional[str]
    source: Optional[str]


@dataclass
class AnswerResult:
    """The assistant's response plus provenance."""

    answer: str
    citations: List[Citation] = field(default_factory=list)
    used_evidence: bool = False


def build_citations(docs: Sequence[Any]) -> List[Citation]:
    """Build a deduplicated citation list from retrieved documents."""
    seen = set()
    citations: List[Citation] = []
    for doc in docs:
        meta = getattr(doc, "metadata", {}) or {}
        key = (meta.get("document_id"), meta.get("section"))
        if key in seen:
            continue
        seen.add(key)
        citations.append(
            Citation(
                title=meta.get("title"),
                document_id=meta.get("document_id"),
                version=meta.get("version"),
                section=meta.get("section"),
                source=meta.get("source") or meta.get("filename"),
            )
        )
    return citations


def format_context(docs: Sequence[Any]) -> str:
    """Render retrieved chunks into a labelled context block for the prompt."""
    blocks = []
    for i, doc in enumerate(docs, 1):
        meta = getattr(doc, "metadata", {}) or {}
        header = (
            f"[Källa {i}] {meta.get('title', '')} "
            f"({meta.get('document_id', '')}, v{meta.get('version', '')}) "
            f"– Avsnitt: {meta.get('section', '')}"
        )
        blocks.append(f"{header}\n{doc.page_content}")
    return "\n\n---\n\n".join(blocks)


def _default_llm(settings: Settings):
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(
        model=settings.chat_model,
        temperature=0,
        api_key=settings.require_api_key(),
    )


def generate_answer(
    question: str,
    docs: Sequence[Any],
    settings: Optional[Settings] = None,
    llm: Any = None,
) -> AnswerResult:
    """Generate a grounded answer. Abstains if no evidence was retrieved.

    ``llm`` can be injected (any LCEL Runnable) for testing; otherwise a
    ChatOpenAI model is created lazily from settings.
    """
    settings = settings or get_settings()
    docs = list(docs or [])

    if not docs:
        return AnswerResult(
            answer=INSUFFICIENT_EVIDENCE_MESSAGE,
            citations=[],
            used_evidence=False,
        )

    from langchain_core.output_parsers import StrOutputParser

    prompt = build_prompt()
    llm = llm or _default_llm(settings)
    chain = prompt | llm | StrOutputParser()
    answer = chain.invoke({"context": format_context(docs), "question": question})

    return AnswerResult(
        answer=answer,
        citations=build_citations(docs),
        used_evidence=True,
    )
