# NordicPaws Försäkring — Insurance Knowledge Assistant

A synthetic RAG demo for pet insurance customer service and veterinary support, built with **LangChain, OpenAI embeddings, Chroma, and Python**.

> ⚠️ **Demo only:** NordicPaws Försäkring is fictional. The documents and figures are synthetic and must not be used for real insurance decisions.

## What it does

The assistant answers questions about:

- 🐕 Dog and 🐈 Cat insurance
- Coverage and reimbursement limits
- Deductibles and waiting periods
- Veterinary care and surgery
- Claims procedures
- Exclusions and limitations

It is also designed to **abstain when the documents do not contain sufficient evidence**, for example questions about premium prices or coverage abroad.

## Current status

- ✅ 6 synthetic Swedish insurance documents
- ✅ 58 indexed chunks in Chroma
- ✅ Structure-aware Markdown chunking
- ✅ Metadata and source citations
- ✅ Swedish grounded-generation prompt
- ✅ Abstention behavior
- ✅ 14 automated tests passing
- ✅ 11-question evaluation set
- ✅ 10/10 answerable questions retrieve expected evidence
- ⚠️ Simple similarity threshold is **not reliable enough for abstention**

## Architecture

```text
Insurance documents
        ↓
Markdown loader
        ↓
Structure-aware chunking
        ↓
OpenAI embeddings
        ↓
Chroma
        ↓
Retriever
        ↓
Grounded LLM
        ↓
Answer + citations / abstention
```

## Project structure

```text
insurance-knowledge-assistant/
├── data/insurance_docs/    # Synthetic insurance corpus
├── src/
│   ├── config.py
│   ├── ingest/
│   ├── retrieval/
│   └── generation/
├── eval/
│   └── golden_qa.jsonl
├── tests/
├── .env.example
└── requirements.txt
```

## Run

Create a virtual environment and install dependencies:

```bash
pip install -r requirements.txt
cp .env.example .env
```

Add your `OPENAI_API_KEY` to `.env`.

Run tests:

```bash
pytest -q
```

Build the vector index:

```bash
PYTHONPATH=src python -m ingest.build_index
```

## Next steps

1. Add a Streamlit chat UI
2. Improve grounded abstention
3. Run the full evaluation harness
4. Improve retrieval/ranking based on evaluation results
