import config
from ingest.chunking import chunk_documents
from ingest.loaders import load_documents


def test_chunks_preserve_metadata_and_sections():
    s = config.get_settings()
    docs = load_documents(s.docs_path)
    chunks = chunk_documents(docs, s.chunk_size, s.chunk_overlap)

    assert len(chunks) > len(docs)

    # Every chunk keeps document-level provenance and a start index.
    for c in chunks:
        assert c.metadata.get("document_id")
        assert c.metadata.get("title")
        assert "start_index" in c.metadata

    # Section metadata is preserved for citations.
    sections = {c.metadata.get("section") for c in chunks}
    assert "6. Karenstider" in sections
