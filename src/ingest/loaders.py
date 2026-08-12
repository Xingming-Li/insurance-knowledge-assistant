"""Document loading for the insurance knowledge base

This version supports the Markdown corpus under ``data/insurance_docs``.
Documents are loaded recursively and enriched with metadata parsed from
the standard document header (title, document ID, version, effective date)
so that citations can later reference the exact source.

The evaluation ground truth (``eval/golden_qa.jsonl`` and anything under
the ``eval`` directory) is excluded from the production knowledge base.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import List, Optional

from langchain_core.documents import Document

# Files/dirs that must never enter the knowledge base.
EXCLUDED_FILENAMES = {"golden_qa.jsonl"}
EXCLUDED_DIRS = {"eval"}


def _extract_title(text: str) -> Optional[str]:
    """Return the first level-1 Markdown heading (the document title)."""
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("# "):
            return stripped[2:].strip()
    return None


def _extract_table_value(text: str, label: str) -> Optional[str]:
    """Pull a value from the header metadata table row under ``| Fält | Värde |``."""
    match = re.search(rf"^\|\s*{re.escape(label)}\s*\|\s*(.+?)\s*\|", text, re.MULTILINE)
    return match.group(1).strip() if match else None


def parse_markdown(path: Path) -> Document:
    """Read a single Markdown file into a metadata-rich Document."""
    text = path.read_text(encoding="utf-8")
    metadata = {
        "source": str(path),
        "filename": path.name,
        "title": _extract_title(text),
        "document_id": _extract_table_value(text, "Dokument-ID"),
        "version": _extract_table_value(text, "Version"),
        "effective_date": _extract_table_value(text, "Ikraftträdandedatum"),
    }
    return Document(page_content=text, metadata=metadata)


def load_documents(docs_dir) -> List[Document]:
    """Recursively load all eligible Markdown documents under ``docs_dir``."""
    root = Path(docs_dir)
    documents: List[Document] = []
    for path in sorted(root.rglob("*.md")):
        if path.name in EXCLUDED_FILENAMES:
            continue
        if any(part in EXCLUDED_DIRS for part in path.parts):
            continue
        documents.append(parse_markdown(path))
    return documents
