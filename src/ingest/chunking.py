"""Structure-aware Markdown chunking

Documents are first split on their Markdown headings so that each piece
carries its section context (used later for citations), then size-bounded
with a small overlap.
"""
from __future__ import annotations

from typing import List

from langchain_core.documents import Document
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

# Split on document title (#) and section (##). The section header text
# is kept in metadata so a chunk can be cited as e.g. "6. Karenstider".
HEADERS_TO_SPLIT_ON = [("#", "h1"), ("##", "section")]


def _section_pieces(doc: Document) -> List[Document]:
    """Split one document into section-segmented pieces, carrying its metadata."""
    splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=HEADERS_TO_SPLIT_ON,
        strip_headers=False,
    )
    pieces = splitter.split_text(doc.page_content)
    out: List[Document] = []
    for piece in pieces:
        metadata = dict(doc.metadata)
        metadata["section"] = piece.metadata.get("section")
        out.append(Document(page_content=piece.page_content, metadata=metadata))
    return out


def chunk_documents(
    docs: List[Document],
    chunk_size: int = 1000,
    chunk_overlap: int = 120,
) -> List[Document]:
    """Turn loaded documents into metadata-rich, size-bounded chunks."""
    section_docs: List[Document] = []
    for doc in docs:
        section_docs.extend(_section_pieces(doc))

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        add_start_index=True,
    )
    return splitter.split_documents(section_docs)
