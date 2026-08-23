# NordicPaws Försäkring — Insurance Knowledge Assistant

A RAG demo for pet insurance customer service and veterinary support, built with **LangChain, OpenAI embeddings, Chroma, and Python**.

> ⚠️ **Demo only:** NordicPaws Försäkring is fictional. The documents and figures are synthetic and must not be used for real decisions.

## What it does

The assistant answers questions about:

- 🐕 Dog and 🐈 Cat insurance
- Coverage and reimbursement
- Deductibles and waiting periods
- Veterinary care and surgery
- Claims procedures
- Exclusions and limitations

It also **abstains when the documents do not contain sufficient evidence**, for example questions about premium prices or coverage abroad.

## Current status

- ✅ 6 synthetic Swedish insurance documents
- ✅ 12-question evaluation set
- ✅ Metadata and source citation recording
- ✅ Structure-aware Markdown chunking
- ✅ 58 indexed chunks in Chroma
- ✅ Swedish grounded-generation prompt
- ✅ Abstention behavior
- ✅ 28 automated tests passed
- ⚠️ 6/10 answerable questions retrieve **intact**, expected evidence
- ⚠️ Retrieval/ranking to be improved
- ⚠️ A Streamlit/FastAPI chat UI to be added

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
Chroma database
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
├── data/insurance_docs/
├── src/
│   ├── config.py
│   ├── ingest/
│   ├── retrieval/
│   |── generation/
|   └── evaluation/
├── eval/
│   |── golden_qa.jsonl
|   └── run_eval.py
├── scripts/
│   └── inspect_retrieval.py
├── tests/
├── .env.example
└── requirements.txt
```

## Run

Create a virtual environment and install dependencies:

```bash
pip install -r requirements.txt
```

Set environment variables, including adding your `OPENAI_API_KEY` to `.env`.

```bash
cp .env.example .env
```

Run tests:

```bash
pytest -q
```

Build the vector index:

```bash
PYTHONPATH=src python -m ingest.build_index
```

Run evaluation harness:

```bash
PYTHONPATH=src python eval/run_eval.py
```
