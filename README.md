# NordicPaws Försäkring --- Insurance Knowledge Assistant

A Swedish-language RAG demo for **pet-insurance employees**, built with
**LangChain, OpenAI embeddings, Chroma, Streamlit, and Python**.

The v1 concept is an **internal insurance knowledge copilot**: it helps
customer-service and insurance staff find, interpret, and explain policy
information with traceable source evidence. A future customer-facing
assistant could reuse the same knowledge layer with stricter guardrails
and escalation workflows.

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
-   Policy-based deductible calculations

It provides grounded answers with source references and can expose the
retrieved evidence for employee verification.

It also **abstains when the available documents do not contain
sufficient evidence**, for example for questions about premium prices or
foreign/travel coverage.

## Intended user

The primary v1 user is an **insurance-company employee**, such as a
customer-service or claims-support agent.

A typical workflow is:

``` text
Customer question
      ↓
Insurance employee
      ↓
NordicPaws assistant
      ↓
Relevant policy evidence
+ grounded explanation
+ calculation when applicable
+ traceable sources
      ↓
Employee verifies
      ↓
Customer receives final answer
```

The employee remains responsible for the final customer communication or
insurance decision. NordicPaws is a decision-support and
knowledge-retrieval prototype, not an autonomous claims-decision system.

## Current status

-   ✅ 6 synthetic Swedish insurance documents
-   ✅ 58 indexed chunks in Chroma
-   ✅ Structure-aware Markdown chunking
-   ✅ Metadata and source citation recording
-   ✅ Swedish grounded-generation prompt
-   ✅ Evidence-based abstention behavior
-   ✅ Streamlit chat interface
-   ✅ Retrieved-evidence inspection in the UI
-   ✅ 12-question golden evaluation set
-   ✅ Source-pair retrieval evaluation
-   ✅ Atomic key-fact answer evaluation
-   ✅ Structured calculation-output evaluation
-   ✅ Automatic detection of incorrect, missing, and ambiguous
    calculation outputs
-   ✅ Deterministic dog/cat species filtering
-   ✅ Targeted deductible-mechanics evidence supplementation
-   ✅ Automated test suite
-   ⚠️ Complex multi-part questions can still miss relevant policy
    sections
-   ⚠️ The evaluation set is intentionally small and synthetic

## Retrieval and generation strategy

The v1 system intentionally uses a relatively simple retrieval
architecture rather than adding multiple LLM-based routing or reranking
stages.

The default pipeline uses:

-   OpenAI `text-embedding-3-small` embeddings
-   Chroma vector storage
-   Dense retrieval with `k=8`
-   Deterministic species filtering to reduce dog/cat
    cross-contamination
-   Targeted evidence supplementation for deductible calculations when
    the deductible-mechanics section is missing
-   Grounded Swedish answer generation
-   Evidence-based abstention when the documents do not support an
    answer

Query decomposition was explored experimentally but was **not selected
as the v1 default** because it did not consistently outperform the
simpler baseline under the evaluation setup.

## Evaluation

The evaluation harness deliberately separates **retrieval quality**,
**answer quality**, **behavior**, and **calculation correctness**.

Retrieval is evaluated using expected `(document_id, section)` source
pairs. Answer quality is evaluated using expected behavior and atomic
key facts. Numeric questions additionally use structured
calculation-output checks.

### Final v1 evaluation run

Model configuration:

``` text
Chat model:       gpt-4o-mini
Embeddings:       text-embedding-3-small
Temperature:      0.0
Retrieval k:      8
Golden questions: 12
```

  Metric                                            Result
  ------------------------------------ -------------------
  Behavior match                          **12/12 (100%)**
  Answerable questions answered           **10/10 (100%)**
  Correct abstentions                       **2/2 (100%)**
  Complete expected source retrieval        **8/10 (80%)**
  Expected source-pair recall              **17/20 (85%)**
  Answer correctness (`answer_ok`)          **8/10 (80%)**
  All expected key facts supported          **8/10 (80%)**
  Key-fact recall                        **22/26 (84.6%)**
  Calculation correctness                   **3/3 (100%)**

The remaining answer failures are complex multi-part questions where
dense retrieval does not consistently surface every policy section
needed to establish all expected facts.

These figures come from a **small synthetic development benchmark**.
They are useful for regression testing and engineering diagnostics, but
they should not be interpreted as estimates of production accuracy on
real insurance questions.

## Architecture

``` text
Synthetic Swedish insurance documents
                ↓
          Markdown loader
                ↓
     Structure-aware chunking
                ↓
        OpenAI embeddings
                ↓
          Chroma database
                ↓
       Dense retrieval (k=8)
                ↓
   Deterministic species filtering
                ↓
        ┌───────┴────────┐
        │                │
 ordinary question   deductible calculation
        │                ↓
        │        deductible-mechanics
        │          evidence check
        │                ↓
        │        supplement if missing
        └───────┬────────┘
                ↓
        Grounded Swedish LLM
                ↓
     answer + citations / abstention
                ↓
          Streamlit interface
                ↓
             Evaluation
       ├── behavior
       ├── source-pair recall
       ├── key-fact correctness
       └── calculation correctness
```

## Why deterministic safeguards?

Pure semantic retrieval can retrieve semantically similar but
operationally incomplete evidence, especially for multi-part insurance
questions.

The v1 prototype therefore uses narrow deterministic safeguards where
they are easy to justify:

1.  **Species filtering** prevents a dog-specific question from being
    answered using cat policy terms, and vice versa.
2.  **Deductible-mechanics supplementation** ensures that deductible
    calculations have access to both the applicable values and the rules
    describing how fixed and variable deductibles are applied.
3.  **Conditional grounding rules** instruct the model not to turn
    conditional policy rules into unsupported facts about a customer's
    situation.
4.  **Abstention** is preferred when the retrieved evidence is
    insufficient.

These safeguards are intentionally narrow rather than a collection of
question-specific rules.

## Project structure

``` text
insurance-knowledge-assistant/
├── data/
│   └── insurance_docs/
├── src/
│   ├── config.py
│   ├── ingest/
│   ├── retrieval/
│   │   ├── retriever.py
│   │   └── decomposed.py
│   ├── generation/
│   ├── evaluation/
│   └── ui_format.py
├── eval/
│   ├── golden_qa.jsonl
│   ├── run_eval.py
│   └── results/
├── scripts/
│   └── inspect_retrieval.py
├── tests/
├── app.py
├── .env.example
└── requirements.txt
```

## Run locally

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

Launch the Streamlit demo:

``` bash
streamlit run app.py
```

> On Windows PowerShell, set `PYTHONPATH` separately before running
> commands if the Unix-style inline syntax is not supported.

## Known limitations

-   The corpus and golden evaluation set are synthetic and small.
-   Dense retrieval can miss some evidence for questions spanning
    several independent policy concepts.
-   The assistant is not connected to customer, policy, claims, or
    pricing databases.
-   It does not make binding coverage or claims decisions.
-   It does not replace employee verification of policy terms.
-   The prototype has not undergone production security, privacy,
    regulatory, latency, or load testing.
-   Query decomposition exists as an experimental retrieval approach but
    is not used by the default v1 pipeline.

## Next steps

The next phase is **customer validation rather than further benchmark
optimization**.

Priority activities:

1.  Demonstrate the employee-facing prototype to pet-insurance
    stakeholders.
2.  Identify the highest-value workflow: customer service, claims
    support, internal policy search, onboarding/training, or another use
    case.
3.  Learn what document systems, auditability, permissions, security,
    and response-time requirements a real insurer would impose.
4.  Expand the evaluation set using realistic questions derived from
    validated workflows.
5.  Only then evaluate v2 improvements such as hybrid retrieval,
    reranking, deterministic policy-rule extraction, enterprise document
    ingestion, authentication, and human escalation.
6.  Explore a separate customer-facing mode only if customer discovery
    shows sufficient value and appropriate safety controls can be
    implemented.

## Disclaimer

NordicPaws Försäkring and all associated policy documents, limits,
deductibles, examples, and evaluation questions in this repository are
fictional and synthetic. The project is intended solely for software
engineering, RAG evaluation, and product-prototyping purposes.
