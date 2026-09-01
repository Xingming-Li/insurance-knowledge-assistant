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
