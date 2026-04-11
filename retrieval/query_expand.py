# retrieval/query_expand.py
import anthropic
import config
import logging

logger = logging.getLogger(__name__)

_client = None


def get_client() -> anthropic.Anthropic:
    global _client
    if _client is None:
        _client = anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)
    return _client


HYDE_SYSTEM = """You are an expert in Toronto municipal by-laws and Ontario tenancy law.
Given a question from a Toronto resident, write a short passage (3-5 sentences) that
looks like it could appear in an official Toronto by-law or Ontario legislation document
and would directly answer the question.

Write only the passage itself — no preamble, no explanation, no section numbers.
Use formal legal language consistent with municipal by-laws."""


def hyde_expand(query: str) -> str:
    """
    Generate a hypothetical document passage for the query using HyDE.
    The passage is embedded and used for retrieval instead of the raw query,
    closing the vocabulary gap between conversational language and legal prose.
    """
    client = get_client()

    logger.info(f"HyDE expanding query: {query[:60]}...")

    response = client.messages.create(
        model=config.HYDE_MODEL,
        max_tokens=200,
        system=HYDE_SYSTEM,
        messages=[{"role": "user", "content": query}],
    )

    expanded = response.content[0].text.strip()
    logger.info(f"HyDE expansion: {expanded[:80]}...")
    return expanded


def expand_and_retrieve(
    query: str,
    top_k: int = config.TOP_K_DENSE,
    domain: str | None = None,
):
    """
    Full HyDE pipeline: expand query → hybrid search on expanded text.
    Falls back to original query if expansion fails.
    """
    from retrieval.retrieve import hybrid_search

    try:
        expanded = hyde_expand(query)
        # Search with expanded query, but keep original for display
        results = hybrid_search(expanded, top_k=top_k, domain=domain)
        return results, expanded
    except Exception as e:
        logger.warning(f"HyDE expansion failed, falling back to original query: {e}")
        from retrieval.retrieve import hybrid_search
        results = hybrid_search(query, top_k=top_k, domain=domain)
        return results, query