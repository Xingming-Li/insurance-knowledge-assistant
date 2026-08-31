"""Pure presentation helpers for the Streamlit demo.

Deliberately Streamlit-free so they can be unit-tested without the UI
dependency. app.py imports these; it holds all Streamlit-specific code.
"""
from __future__ import annotations

import re
from typing import Any

# Markers indicating the assistant declared the evidence insufficient.
_ABSTENTION_MARKERS = (
    "inte tillräcklig",
    "räcker inte",
    "kan tyvärr inte besvara",
)


def is_abstention(answer_text: str, used_evidence: bool) -> bool:
    """True when the answer should be shown as an abstention, not a normal answer."""
    if not used_evidence:
        return True
    text = (answer_text or "").lower()
    return any(marker in text for marker in _ABSTENTION_MARKERS)


def format_source(citation: Any) -> str:
    """Human-readable Swedish source label from a Citation (never raw paths)."""
    parts = []
    title = getattr(citation, "title", None)
    document_id = getattr(citation, "document_id", None)
    section = getattr(citation, "section", None)
    version = getattr(citation, "version", None)
    if title:
        parts.append(str(title))
    if document_id:
        parts.append(str(document_id))
    if section:
        parts.append(f"Avsnitt {section}")
    if version:
        parts.append(f"v{version}")
    return " · ".join(parts) if parts else "Okänd källa"


def strip_generated_source_footer(answer: str) -> str:
    """Remove a trailing LLM-generated source list/footer.

    Keeps inline citations such as "(Källa 1)" intact.
    """
    if not answer:
        return answer

    patterns = [
        # Källor: [Källa 1], [Källa 8], [Källa 9].
        r"\n+\s*Källor\s*:\s*(?:\[?Källa\s+\d+\]?\s*[,;.]?\s*)+\s*$",

        # Markdown heading followed by a source list at the end.
        r"\n+\s*\*{0,2}Källor\*{0,2}\s*:?\s*\n(?:\s*[-*]\s+.*\n?)+\s*$",
    ]

    cleaned = answer.rstrip()

    for pattern in patterns:
        cleaned = re.sub(
            pattern,
            "",
            cleaned,
            flags=re.IGNORECASE | re.MULTILINE,
        ).rstrip()

    return cleaned
