"""Build the Chroma vector index from the insurance corpus

Run directly:

    PYTHONPATH=src python -m ingest.build_index

Uses the modern ``langchain-chroma`` store, which auto-persists to
``persist_directory`` (no deprecated ``db.persist()`` call needed).
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

# Allow direct execution (``python src/ingest/build_index.py``) by putting the
# ``src`` root on the path when this file is not imported as part of a package.
if __package__ in (None, ""):  # pragma: no cover
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import Settings, get_settings  # noqa: E402
from ingest.chunking import chunk_documents  # noqa: E402
from ingest.loaders import load_documents  # noqa: E402


def build_index(settings: Settings | None = None, reset: bool = True) -> int:
    """Load, chunk, embed and persist the corpus. Returns the chunk count."""
    settings = settings or get_settings()

    documents = load_documents(settings.docs_path)
    chunks = chunk_documents(documents, settings.chunk_size, settings.chunk_overlap)

    # Heavy/optional deps imported lazily so the rest of the package stays
    # importable (and unit-testable) without them.
    from langchain_chroma import Chroma
    from langchain_openai import OpenAIEmbeddings

    if reset and Path(settings.chroma_path).exists():
        shutil.rmtree(settings.chroma_path)

    embeddings = OpenAIEmbeddings(
        model=settings.embedding_model,
        api_key=settings.require_api_key(),
    )
    Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=settings.collection_name,
        persist_directory=settings.chroma_path,
    )
    return len(chunks)


def main() -> None:
    settings = get_settings()
    count = build_index(settings)
    print(
        f"Indexed {count} chunks into '{settings.chroma_path}' "
        f"(collection: {settings.collection_name})."
    )


if __name__ == "__main__":
    main()
