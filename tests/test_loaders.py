import config
from ingest.loaders import load_documents


def test_loads_six_documents_with_metadata():
    docs = load_documents(config.get_settings().docs_path)
    assert len(docs) == 6

    ids = {d.metadata["document_id"] for d in docs}
    assert "NP-DOG-TERMS-2026" in ids
    assert "NP-CS-2026" in ids

    for d in docs:
        assert d.metadata["title"], "missing title"
        assert d.metadata["document_id"], "missing document_id"
        assert d.metadata["version"], "missing version"
        assert d.metadata["effective_date"], "missing effective_date"
        assert d.metadata["filename"].endswith(".md")
        assert d.metadata["source"]
