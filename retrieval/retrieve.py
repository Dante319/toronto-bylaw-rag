# retrieval/retrieve.py
import logging
from dataclasses import dataclass
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue
from rank_bm25 import BM25Okapi
import config

logger = logging.getLogger(__name__)


@dataclass
class RetrievedChunk:
    chunk_id: str
    domain: str
    section_id: str
    section_title: str
    parent_section: str
    text: str
    page: int | None
    score: float


_embed_model = None
_qdrant_client = None
_bm25_index = None
_bm25_chunks = None   # parallel list to _bm25_index corpus


def get_embed_model() -> SentenceTransformer:
    global _embed_model
    if _embed_model is None:
        logger.info(f"Loading embedding model: {config.EMBED_MODEL}")
        _embed_model = SentenceTransformer(config.EMBED_MODEL)
    return _embed_model


def get_qdrant_client() -> QdrantClient:
    global _qdrant_client
    if _qdrant_client is None:
        _qdrant_client = QdrantClient(
            url=config.QDRANT_URL,
            api_key=config.QDRANT_API_KEY,
        )
    return _qdrant_client


def get_bm25_index(domain: str | None = None) -> tuple[BM25Okapi, list[dict]]:
    """
    Build BM25 index from Qdrant payloads on first call, cache thereafter.
    Optionally filter by domain.
    """
    global _bm25_index, _bm25_chunks

    if _bm25_index is None:
        logger.info("Building BM25 index from Qdrant...")
        client = get_qdrant_client()

        all_chunks = []
        offset = None

        while True:
            results, offset = client.scroll(
                collection_name=config.COLLECTION_NAME,
                limit=100,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            all_chunks.extend([r.payload for r in results])
            if offset is None:
                break

        _bm25_chunks = all_chunks
        corpus = [chunk["text"].lower().split() for chunk in all_chunks]
        _bm25_index = BM25Okapi(corpus)
        logger.info(f"BM25 index built over {len(all_chunks)} chunks")

    # Filter if domain specified
    if domain:
        filtered = [
            (i, c) for i, c in enumerate(_bm25_chunks)
            if c["domain"] == domain
        ]
        indices, chunks = zip(*filtered) if filtered else ([], [])
        corpus = [c["text"].lower().split() for c in chunks]
        return BM25Okapi(corpus), list(chunks), list(indices)

    return _bm25_index, _bm25_chunks, list(range(len(_bm25_chunks)))


# ── Retrieval functions ────────────────────────────────────────────────────────

def dense_search(
    query: str,
    top_k: int = config.TOP_K_DENSE,
    domain: str | None = None,
) -> list[RetrievedChunk]:
    """Embed query and search Qdrant by cosine similarity."""
    model = get_embed_model()
    client = get_qdrant_client()

    # BGE requires this prefix for queries (different from indexing prefix)
    prefixed = f"Represent this sentence for searching relevant passages: {query}"
    vector = model.encode(prefixed, normalize_embeddings=True).tolist()

    query_filter = None
    if domain:
        query_filter = Filter(
            must=[FieldCondition(
                key="domain",
                match=MatchValue(value=domain)
            )]
        )

    results = client.query_points(
        collection_name=config.COLLECTION_NAME,
        query=vector,
        limit=top_k,
        query_filter=query_filter,
        with_payload=True,
    ).points

    return [_to_chunk(r.payload, r.score) for r in results]


def sparse_search(
    query: str,
    top_k: int = config.TOP_K_DENSE,
    domain: str | None = None,
) -> list[RetrievedChunk]:
    """BM25 keyword search over all chunks."""
    bm25, chunks, _ = get_bm25_index(domain)
    tokenized_query = query.lower().split()
    scores = bm25.get_scores(tokenized_query)

    # Get top_k indices by score
    top_indices = sorted(
        range(len(scores)),
        key=lambda i: scores[i],
        reverse=True
    )[:top_k]

    return [
        _to_chunk(chunks[i], float(scores[i]))
        for i in top_indices
        if scores[i] > 0   # skip zero-score results
    ]


def hybrid_search(
    query: str,
    top_k: int = config.TOP_K_DENSE,
    domain: str | None = None,
    rrf_k: int = 60,
) -> list[RetrievedChunk]:
    """
    Combine dense and sparse results using Reciprocal Rank Fusion.
    rrf_k=60 is the standard constant from the original RRF paper.
    """
    dense_results = dense_search(query, top_k=top_k, domain=domain)
    sparse_results = sparse_search(query, top_k=top_k, domain=domain)

    # Build RRF score map keyed by chunk_id
    rrf_scores: dict[str, float] = {}
    chunk_map: dict[str, RetrievedChunk] = {}

    for rank, chunk in enumerate(dense_results):
        rrf_scores[chunk.chunk_id] = rrf_scores.get(chunk.chunk_id, 0) + \
            1 / (rrf_k + rank + 1)
        chunk_map[chunk.chunk_id] = chunk

    for rank, chunk in enumerate(sparse_results):
        rrf_scores[chunk.chunk_id] = rrf_scores.get(chunk.chunk_id, 0) + \
            1 / (rrf_k + rank + 1)
        chunk_map[chunk.chunk_id] = chunk

    # Sort by RRF score, return top_k
    ranked = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]

    results = []
    for chunk_id, score in ranked:
        chunk = chunk_map[chunk_id]
        chunk.score = score
        results.append(chunk)

    return results


def _to_chunk(payload: dict, score: float) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=payload["chunk_id"],
        domain=payload["domain"],
        section_id=payload["section_id"],
        section_title=payload["section_title"],
        parent_section=payload.get("parent_section"),
        text=payload["text"],
        page=payload.get("page"),
        score=score,
    )