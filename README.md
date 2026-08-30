# NordicPaws Försäkring --- Insurance Knowledge Assistant

A RAG demo for pet insurance customer service and veterinary support,
built with **LangChain, OpenAI embeddings, Chroma, and Python**.

> ⚠️ **Demo only:** NordicPaws Försäkring is fictional. The documents
> and figures are synthetic and must not be used for real insurance or
> veterinary decisions.

## What it does

The assistant answers Swedish-language questions about:

-   🐕 Dog and 🐈 cat insurance
-   Coverage and reimbursement
-   Deductibles and waiting periods
-   Veterinary care and surgery
-   Claims procedures
-   Exclusions and limitations

It also **abstains when the retrieved documents do not contain
sufficient evidence**, for example for questions about premium prices or
coverage abroad.

## Current status

-   ✅ 6 synthetic Swedish insurance documents
-   ✅ 12-question golden evaluation set
-   ✅ Structure-aware Markdown chunking
-   ✅ 58 indexed chunks in Chroma
-   ✅ Metadata and source citation recording
-   ✅ Swedish grounded-generation prompt
-   ✅ Evidence-based abstention behavior
-   ✅ Source-pair retrieval evaluation
-   ✅ Key-fact answer correctness evaluation
-   ✅ Structured calculation-output evaluation
-   ✅ Automatic detection of incorrect, missing, and ambiguous
    calculation outputs
-   ✅ Automated test suite
-   ⚠️ Current dense-retrieval baseline: 6/10 answerable questions
    retrieve all expected source evidence
-   ⚠️ Multi-part question retrieval and ranking need improvement
-   ⚠️ Streamlit/FastAPI chat UI to be added

## Evaluation

The evaluation harness separates **retrieval quality** from **answer
quality**.

Retrieval is evaluated using expected `(document_id, section)` source
pairs. Answer quality is evaluated using expected behavior, atomic key
facts, and structured calculation outputs where applicable.

Current dense-retrieval baseline at `k=4`:

``` text
Behavior match:           11/12
Correct abstentions:       2/2
Complete source retrieval: 6/10
Source-pair recall:       11/20
Correct answers:           6/10
Key-fact recall:          19/27
Correct calculations:      0/3
```

Current retrieval work focuses on multi-part questions where relevant
evidence is not consistently ranked in the top retrieved chunks.

## Architecture

``` text
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
        ↓
Evaluation
  ├── source-pair recall
  ├── key-fact correctness
  └── calculation correctness
```

## Project structure

``` text
insurance-knowledge-assistant/
├── data/
│   └── insurance_docs/
├── src/
│   ├── config.py
│   ├── ingest/
│   ├── retrieval/
│   ├── generation/
│   └── evaluation/
├── eval/
│   ├── golden_qa.jsonl
│   ├── run_eval.py
│   └── results/
├── scripts/
│   ├── inspect_retrieval.py
│   └── compare_retrieval.py
├── tests/
├── .env.example
└── requirements.txt
```

## Run

Create a virtual environment and install dependencies:

``` bash
pip install -r requirements.txt
```

Create `.env` from the example file and add your `OPENAI_API_KEY`:

``` bash
cp .env.example .env
```

Run tests:

``` bash
pytest -q
```

Build the vector index:

``` bash
PYTHONPATH=src python -m ingest.build_index
```

Run the evaluation harness:

``` bash
PYTHONPATH=src python eval/run_eval.py
```

Inspect retrieval for selected questions:

``` bash
PYTHONPATH=src python scripts/inspect_retrieval.py
```

Compare baseline dense vs decomposed retrieval:

``` bash
PYTHONPATH=src python scripts/compare_retrieval.py
```

> On Windows PowerShell, set `PYTHONPATH` separately before running the
> Python commands if the Unix-style inline syntax above is not
> supported.

## Next steps

1.  Diagnose missing evidence for multi-part questions.
2.  Evaluate query decomposition and retrieval improvements.
3.  Compare retrieval strategies using source-pair recall and retrieval
    noise.
4.  Re-evaluate answer and calculation correctness with the selected
    retriever.
5.  Add a lightweight Streamlit or FastAPI interface.
