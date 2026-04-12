# Toronto Housing & Rental Policy Q&A

A production-grade RAG system for answering plain-language questions about Toronto's
short-term rental regulations, noise by-laws, and residential tenancy law.

Built as a portfolio project to demonstrate end-to-end NLP/RAG engineering — from
document ingestion and hierarchical chunking through hybrid retrieval, query expansion,
and citation-grounded generation.

**Live demo:** _[coming soon]_

---

## What it does

Residents, tenants, and landlords can ask questions like:

- _"Can I rent out my basement on Airbnb without a licence?"_
- _"What are the quiet hours for construction noise in Toronto?"_
- _"How much notice does my landlord have to give before entering my unit?"_

The system retrieves the relevant by-law passages and generates a grounded answer with
citations to specific section numbers — so you can verify the source directly.

---

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                   Offline Ingestion                  │
│                                                     │
│  PDF Parser → Hierarchical Chunker → BGE Embedder  │
│                          ↓                         │
│                     Qdrant Index                    │
└─────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────┐
│                    Online Query                      │
│                                                     │
│  User Query → HyDE Expansion → Hybrid Retrieval    │
│            → BGE Reranker → Claude Generation      │
└─────────────────────────────────────────────────────┘
```

_Full architecture diagram coming soon._

---

## Corpus

| Domain | Source |
|---|---|---|
| Short-term rentals | Toronto Municipal Code, Chapter 547 |
| Noise by-laws | Toronto Municipal Code, Chapter 591 |
| Residential tenancy | Ontario Residential Tenancies Act, 2006 |

---

## Technical decisions

### Chunking strategy

By-law documents are structured hierarchically: Chapter → Article → Section → Subsection.
Rather than fixed-size chunking, sections are detected using the document's own structural
markers (`§ 547-x.x` headings, `ARTICLE N` boundaries) and stored as discrete chunks with
metadata. This enables precise citation at the section level and supports metadata-filtered
retrieval by domain or article.

### Query expansion — HyDE

Before retrieval, the user's conversational query is expanded using Hypothetical Document
Embeddings (HyDE — Gao et al., 2023): Claude Haiku generates a hypothetical by-law passage
relevant to the query, which is embedded and used for retrieval alongside the original query.
This closes the vocabulary gap between natural language questions and legal prose.

### Hybrid retrieval

Dense vector search (BGE-large-en-v1.5) is combined with sparse BM25 keyword search in
Qdrant, then reranked with BGE-reranker-v2-m3. This outperforms dense-only retrieval on
exact legal term matching (e.g. "principal residence", "short-term rental company").

### Generation

Claude Sonnet is instructed to cite every factual claim with the specific section number
it came from, and to refuse to answer when no relevant passage is retrieved. This makes
hallucination auditable rather than invisible.

---

## Evaluation

_Results coming once benchmark is complete._

| Metric | Score |
|---|---|
| Recall@3 | _TBD_ |
| Faithfulness (RAGAS) | _TBD_ |
| Answer relevancy (RAGAS) | _TBD_ |
| Context precision (RAGAS) | _TBD_ |
| HyDE Recall@3 delta | _TBD_ |

Evaluated against a hand-curated benchmark of 40 question–answer pairs (≈13 per domain).
See `eval/benchmark.json`.

### HyDE ablation

The table below compares retrieval quality with and without query expansion:

| | Without HyDE | With HyDE | Delta |
|---|---|---|---|
| Recall@3 | _TBD_ | _TBD_ | _TBD_ |

---

## Known limitations & failure analysis

_To be completed after evaluation._

Examples where the system gets it wrong, and why:

1. _TBD_
2. _TBD_
3. _TBD_

---

## Stack

| Component | Tool |
|---|---|
| PDF parsing | pdfplumber |
| Embeddings | BAAI/bge-large-en-v1.5 |
| Vector store | Qdrant |
| Sparse retrieval | BM25 (rank-bm25) |
| Reranker | BAAI/bge-reranker-v2-m3 |
| Query expansion | Claude Haiku (HyDE) |
| Generation | Claude Sonnet |
| Demo UI | Streamlit |
| Evaluation | RAGAS |

---

## Setup

### Prerequisites

- Python 3.11+
- [uv](https://github.com/astral-sh/uv)
- Docker (for local Qdrant)

### Install

```bash
git clone https://github.com/Dante319/toronto-bylaw-rag
cd toronto-bylaw-rag
uv venv && source .venv/bin/activate
uv sync
```

### Environment variables

Copy `.env.example` to `.env` and fill in your keys:

```bash
cp .env.example .env
```

```
ANTHROPIC_API_KEY=your-key-here
QDRANT_URL=http://localhost:6333
QDRANT_API_KEY=                    # leave blank for local dev
```

### Source documents

_Download instructions coming soon._

Place PDFs in `data/raw/`:
```
data/raw/
  ch547_short_term_rental.pdf
  ch591_noise.pdf
```

### Run ingestion

```bash
# Start Qdrant
docker start qdrant

# Build the index
uv run ingestion/run_ingestion.py
```

### Run the demo

```bash
uv run streamlit run app/app.py
```

---

## Future work

- **Ontario Residential Tenancies Act** — Word doc ingestion via `python-docx`
- **LLM-based parser** — replace rule-based section detection with a structured
  extraction prompt for documents with non-standard formatting
- **Agentic layer** — tool-calling step to fetch live Toronto by-law amendments
  before generation
- **Multilingual support** — Toronto's tenant population is highly multilingual;
  query translation would significantly expand reach

---

## Related work

- Gao et al. (2023). [Precise Zero-Shot Dense Retrieval without Relevance Labels](https://arxiv.org/abs/2212.10496) — HyDE
- BAAI. [BGE Embedding Models](https://huggingface.co/BAAI/bge-large-en-v1.5)
- [City of Toronto Open Data Portal](https://open.toronto.ca)