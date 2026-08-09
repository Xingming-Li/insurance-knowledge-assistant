import pytest

import config


def test_defaults_and_no_hardcoded_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    s = config.get_settings()
    assert s.openai_api_key is None  # never hard-coded
    assert s.embedding_model == "text-embedding-3-small"
    assert s.chat_model
    assert s.collection_name == "insurance_docs"
    assert 0 < s.chunk_overlap < s.chunk_size
    assert s.retrieval_k >= 1


def test_env_override(monkeypatch):
    monkeypatch.setenv("EMBEDDING_MODEL", "custom-emb")
    monkeypatch.setenv("RETRIEVAL_K", "7")
    monkeypatch.setenv("CHUNK_OVERLAP", "50")
    s = config.get_settings()
    assert s.embedding_model == "custom-emb"
    assert s.retrieval_k == 7
    assert s.chunk_overlap == 50


def test_require_api_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(RuntimeError):
        config.get_settings().require_api_key()

    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-123")
    assert config.get_settings().require_api_key() == "sk-test-123"
