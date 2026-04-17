# Toronto By-law Q&A

A production-grade RAG system for answering plain-language questions about Toronto's
short-term rental regulations and noise by-laws, grounded in the Toronto Municipal Code
with section-level citations.

**Live demo:** https://toronto-bylaw-rag.streamlit.app

---

## What it does

Residents, tenants, and landlords can ask questions like:

- _"Can I rent out my basement on Airbnb without a licence?"_
- _"What are the quiet hours for construction noise in Toronto?"_
- _"What happens if I operate a short-term rental without registering?"_
- _"Can a short-term rental company impose arbitration clauses on operators?"_

The system retrieves the relevant by-law passages and generates a grounded answer
with citations to specific section numbers — so you can verify the source directly.

---

## Demo

> _Screenshot coming soon_

---

## Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                      Offline Ingestion                        │
│                                                              │
│  Toronto Municipal Code PDFs                                 │
│       ↓                                                      │
│  pdfplumber → Article-bounded section parser                 │
│       ↓                                                      │
│  Hierarchical chunker (section-level, 512 token max)         │
│       ↓                                                      │
│  BGE-large-en-v1.5 embedder (local, batched)                │
│       ↓                                                      │
│  Qdrant Cloud (384-dim for deployment, 1024-dim locally)     │
└──────────────────────────────────────────────────────────────┘
                            ↓
┌──────────────────────────────────────────────────────────────┐
│                       Online Query                            │
│                                                              │
│  User query                                                  │
│       ↓                                                      │
│  Semantic domain detection (BGE cosine similarity)           │
│       ↓                                                      │
│  HyDE expansion (Claude Haiku → hypothetical by-law passage) │
│       ↓                                                      │
│  Hybrid retrieval: dense (BGE) + sparse (BM25) → RRF fusion  │
│       ↓                                                      │
│  Top-5 chunks → Claude Sonnet (citation-grounded generation) │
│       ↓                                                      │
│  Cited answer with § section references                      │
└──────────────────────────────────────────────────────────────┘
```

---

## Corpus

| Domain | Source | Chunks indexed |
|---|---|---|
| Short-term rentals | Toronto Municipal Code, Chapter 547 | ~50 |
| Noise by-laws | Toronto Municipal Code, Chapter 591 | ~32 |
| Residential tenancy | Ontario Residential Tenancies Act, 2006 | _coming soon_ |

Total: **82 chunks** across 2 domains.

---

## Technical decisions

### Document parsing — rule-based section detection

Toronto Municipal Code PDFs are structured hierarchically:
`Chapter → Article → § Section`. The parser detects section boundaries using
structural markers (`§ 547-x.x` headings, `ARTICLE N` labels) rather than
fixed-size chunking. This required:

- Suppressing repeating page headers (`TORONTO MUNICIPAL CODE`) and footers
  (`547-3 January 1, 2025`) which pdfplumber surfaces as content lines
- Treating ARTICLE headers as parent metadata rather than sections themselves
- Including table-of-contents entries as chunks — they are short and do not
  harm retrieval quality, and excluding them added fragile page-counting logic

The parser is document-specific by design. Different corpora (e.g. the Ontario RTA)
use different numbering conventions and require their own parser module. This is an
explicit tradeoff: reliability over generality for a known, stable corpus.

_See `notebooks/01_parser_verification.ipynb` for section extraction diagnostics._

### Chunking — section-level with overlap

Chunks map 1:1 to by-law sections where possible. Sections exceeding 512 tokens
are split at sentence boundaries with 64-token overlap. Each chunk carries metadata:

```json
{
  "chunk_id": "short_term_rental::547-1.2::0",
  "domain": "short_term_rental",
  "section_id": "547-1.2",
  "section_title": "Requirement for company licence and operator registration.",
  "parent_section": "ARTICLE 1",
  "page": 4
}
```

This enables section-level citation (`§ 547-1.2`) and domain-filtered retrieval.

_See `notebooks/02_chunker_debugger.ipynb` for length distribution analysis._

### Domain detection — semantic similarity

Rather than hardcoded keyword lists, domain classification uses cosine similarity
between the query embedding and a short description of each domain's content.
Queries above a confidence threshold (0.75) are routed to a single domain;
ambiguous queries search all domains. Adding a new domain requires only one
sentence in `config.py` — no code changes.

```python
DOMAIN_DESCRIPTIONS = {
    "short_term_rental": "Toronto short-term rental regulations, Airbnb licensing...",
    "noise": "Toronto noise by-laws, quiet hours, construction noise...",
}
```

### Query expansion — HyDE

Before retrieval, the user's conversational query is rewritten by Claude Haiku as a
hypothetical by-law passage (HyDE — Gao et al., 2023). This closes the vocabulary
gap between natural language questions and formal legal prose. Example:

> **Query:** "Can I rent out my basement on Airbnb without a licence?"
>
> **HyDE expansion:** "No person shall carry on the business of a short-term rental
> operator without first registering with Municipal Licensing and Standards and
> obtaining a valid registration number under Chapter 547..."

The expanded passage is embedded and used for dense retrieval alongside BM25
keyword search on the original query.

### Hybrid retrieval — dense + sparse + RRF

Dense vector search (BGE) and sparse BM25 keyword search are run in parallel,
then merged using Reciprocal Rank Fusion (RRF, k=60). RRF is rank-position-based
rather than score-based, making it robust to the different score scales of dense
and sparse systems. Top-5 results are passed to the generator.

BM25 is built from chunk payloads loaded from Qdrant on startup and cached
in-process — no separate keyword index to manage.

### Generation — Claude Sonnet with strict citation prompt

Claude Sonnet is instructed to:

1. Answer only from retrieved context — no outside knowledge
2. Cite every factual claim with its section number (`[§ 547-1.2]`)
3. Return a standard refusal if no relevant passage was retrieved
4. Flag amendments and recommend verification for time-sensitive questions

### Embedding model — two configs

| Environment | Model | Dimension | Reason |
|---|---|---|---|
| Local dev + eval | BGE-large-en-v1.5 | 1024 | Best retrieval quality |
| Deployed (Streamlit) | BGE-small-en-v1.5 | 384 | Fits 1GB RAM free tier |

The Qdrant Cloud collection is indexed with BGE-small. Local dev uses a separate
local Qdrant instance indexed with BGE-large.

---

## Evaluation

_In progress — results coming soon._

| Metric | Without HyDE | With HyDE | Delta |
|---|---|---|---|
| Recall@3 | _TBD_ | _TBD_ | _TBD_ |
| Faithfulness (RAGAS) | _TBD_ | _TBD_ | _TBD_ |
| Answer relevancy (RAGAS) | _TBD_ | _TBD_ | _TBD_ |
| Context precision (RAGAS) | _TBD_ | _TBD_ | _TBD_ |

Benchmark: 35 synthetically generated questions (RAGAS testset generator) +
5 manually curated edge cases. See `eval/benchmark.json`.

---

## Known limitations & failure analysis

_Preliminary observations — to be expanded after evaluation:_

1. **Section-level chunking misses sub-section specificity.** `§ 547-1.2` covers
   both company licensing (A) and operator registration (B) in one chunk. Queries
   specifically about operators rank this chunk lower than expected because the
   company clause dominates the embedding. Sub-section chunking would improve this.

2. **BGE-small retrieval quality on deployment.** The deployed app uses BGE-small
   (384-dim) for RAM constraints. The correct section occasionally ranks 4th-5th
   rather than 1st-3rd compared to BGE-large. Mitigated by passing top-5 chunks
   to the generator rather than top-3.

3. **Amendment noise in chunks.** Editor's notes (`[Amended 2024-05-23 by By-law
   503-2024]`) are included in chunk text. They do not harm answer quality but
   dilute the embedding signal for the substantive content.

4. _Further examples to be added after evaluation._

---

## Stack

| Component | Tool | Notes |
|---|---|---|
| PDF parsing | pdfplumber | Article-bounded section detection |
| Embeddings (local) | BAAI/bge-large-en-v1.5 | 1024-dim, ~1.3GB |
| Embeddings (deployed) | BAAI/bge-small-en-v1.5 | 384-dim, ~130MB |
| Vector store | Qdrant Cloud | Free tier, 1GB |
| Sparse retrieval | rank-bm25 | Built from Qdrant payloads on startup |
| Query expansion | Claude Haiku (HyDE) | Hypothetical document generation |
| Generation | Claude Sonnet | Citation-grounded, refusal on no context |
| Domain detection | BGE cosine similarity | Threshold-based, config-driven |
| Demo UI | Streamlit Community Cloud | Free tier |
| Evaluation | RAGAS | Faithfulness, relevancy, context precision |
| Package manager | uv | Faster than pip, lock file included |

---

## Repo structure

```
toronto-bylaw-rag/
├── data/
│   ├── raw/               # Source PDFs (not committed — download links above)
│   └── processed/         # Parsed, chunked JSONL files
├── ingestion/
│   ├── parser.py           # Toronto Municipal Code PDF parser
│   ├── chunker.py           # Section-level chunker with overlap
│   ├── embedder_indexer.py     # BGE embedding + Qdrant upsert
│   └── runner.py   # Ingestion runner with --recreate flag
├── retrieval/
│   ├── retrieve.py        # Dense, sparse, and hybrid RRF retrieval
│   ├── query_expand.py    # HyDE query expansion via Claude Haiku
│   ├── domain_detect.py   # Semantic domain classifier
│   └── generate.py        # Citation-grounded generation via Claude Sonnet
├── eval/
│   ├── benchmark.json         # 40 Q&A pairs (35 synthetic + 5 manual)
│   ├── generate_benchmark.py  # RAGAS testset generator
│   └── run_eval.py            # RAGAS metrics + HyDE ablation
├── app/
│   └── app.py             # Streamlit demo
├── notebooks/
│   ├── 00_extraction_comparison.ipynb
│   ├── 01_parser_debugger.ipynb
│   ├── 02_chunker_debugger.ipynb
│   ├── 03_embedder_debugger.ipynb
│   ├── 04_ingestion_runner_debugger.ipynb
│   ├── 05_retriever_debugger.ipynb
│   ├── 06_query_expander_debugger.ipynb
│   └── 07_generator_debugger.ipynb
├── config.py              # Centralized config, env + Streamlit secrets
├── requirements.txt       # Lean deps for Streamlit deployment
├── pyproject.toml         # Full deps for local dev (uv)
└── .env.example           # Environment variable template
```

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

```bash
cp .env.example .env
# Fill in ANTHROPIC_API_KEY, QDRANT_URL, QDRANT_API_KEY
```

```
ANTHROPIC_API_KEY=your-key-here
QDRANT_URL=http://localhost:6333
QDRANT_API_KEY=                     # leave blank for local Docker
EMBED_MODEL=BAAI/bge-large-en-v1.5  # use bge-small-en-v1.5 for deployment
```

### Source documents

Download and place in `data/raw/`:

| File | Source |
|---|---|
| `ch547_short_term_rental.pdf` | [Toronto Municipal Code, Ch. 547](https://toronto.ca/municipal-code) |
| `ch591_noise.pdf` | [Toronto Municipal Code, Ch. 591](https://toronto.ca/municipal-code) |

### Run locally

```bash
# Start Qdrant
docker start qdrant

# Build the index
PYTHONPATH=. uv run ingestion/run_ingestion.py

# Run the app
PYTHONPATH=. uv run streamlit run app/app.py
```

### Run evaluation

```bash
# Generate synthetic benchmark (run once)
PYTHONPATH=. uv run eval/generate_benchmark.py

# Run RAGAS evaluation
PYTHONPATH=. uv run eval/run_eval.py
```

---

## Future work

- **Ontario Residential Tenancies Act** — add as a third domain via Word doc
  ingestion (`python-docx`). The parser is modular — new document types require
  only a new parser module, no changes to retrieval or generation.
- **Sub-section chunking** — split sections at lettered clauses (A, B, C) for
  finer retrieval granularity, particularly for multi-clause sections like `§ 547-1.2`.
- **LLM-based parser** — replace rule-based section detection with a structured
  extraction prompt for documents with non-standard formatting.
- **Agentic layer** — tool-calling step to fetch live Toronto by-law amendment
  dates before generation, flagging recently changed sections.
- **Multilingual support** — Toronto's tenant population is highly multilingual;
  query translation would significantly expand reach.

---

## Related work

- Gao et al. (2023). [Precise Zero-Shot Dense Retrieval without Relevance Labels](https://arxiv.org/abs/2212.10496) — HyDE
- BAAI. [BGE Embedding Models](https://huggingface.co/BAAI/bge-large-en-v1.5)
- Robertson & Zaragoza (2009). The Probabilistic Relevance Framework: BM25 and Beyond
- Cormack et al. (2009). Reciprocal Rank Fusion outperforms Condorcet and individual Rank Learning Methods
- [City of Toronto Open Data Portal](https://open.toronto.ca)
- [Toronto Municipal Code](https://toronto.ca/municipal-code)