from ingest.loaders import load_documents

_MD = (
    "# Titel\n\n"
    "| Fält | Värde |\n|---|---|\n"
    "| Dokument-ID | NP-TEST-1 |\n"
    "| Version | 1.0 |\n"
    "| Ikraftträdandedatum | 2026-01-01 |\n\n"
    "## 1. Avsnitt\nInnehåll.\n"
)


def test_eval_artifacts_are_excluded(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    (data / "doc.md").write_text(_MD, encoding="utf-8")

    eval_dir = tmp_path / "eval"
    eval_dir.mkdir()
    (eval_dir / "golden_qa.jsonl").write_text('{"id": "Q1"}', encoding="utf-8")
    # Even a stray .md under an eval directory must not be ingested.
    (eval_dir / "stray.md").write_text("# should be excluded", encoding="utf-8")

    docs = load_documents(tmp_path)
    names = {d.metadata["filename"] for d in docs}

    assert names == {"doc.md"}
    assert "golden_qa.jsonl" not in names
    assert "stray.md" not in names


def test_real_corpus_has_no_eval_file():
    import config

    docs = load_documents(config.get_settings().docs_path)
    assert all(d.metadata["filename"] != "golden_qa.jsonl" for d in docs)
