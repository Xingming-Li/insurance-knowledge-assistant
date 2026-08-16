import config
from ingest.build_index import sanitize_metadata
from ingest.chunking import chunk_documents
from ingest.loaders import load_documents


def test_real_corpus_has_none_section_before_sanitizing():
    # Preamble blocks (before the first ## heading) have section=None.
    # This is exactly what Chroma rejects, so the sanitizer must remove it.
    s = config.get_settings()
    chunks = chunk_documents(load_documents(s.docs_path), s.chunk_size, s.chunk_overlap)
    assert any(c.metadata.get("section") is None for c in chunks)


def test_sanitize_metadata_drops_none_values():
    s = config.get_settings()
    chunks = chunk_documents(load_documents(s.docs_path), s.chunk_size, s.chunk_overlap)
    sanitize_metadata(chunks)

    for c in chunks:
        # No None values remain, and every remaining value is Chroma-safe.
        assert all(v is not None for v in c.metadata.values())
        assert all(isinstance(v, (str, int, float, bool)) for v in c.metadata.values())
