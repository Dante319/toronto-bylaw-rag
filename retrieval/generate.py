# retrieval/generate.py
import anthropic
from retrieval.retrieve import RetrievedChunk
import config
import logging

logger = logging.getLogger(__name__)

_client = None


def get_client() -> anthropic.Anthropic:
    global _client
    if _client is None:
        _client = anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)
    return _client


SYSTEM_PROMPT = """You are a helpful assistant that answers questions about Toronto
municipal by-laws and Ontario tenancy law.

Rules you must follow:
1. Answer ONLY using the provided context passages. Do not use outside knowledge.
2. Cite every factual claim with the section number it came from, like this: [§ 547-1.2]
3. If the context does not contain enough information to answer, say exactly:
   "I could not find a relevant passage in the by-laws to answer this question.
   Please consult the City of Toronto website or a legal professional."
4. Never speculate or infer beyond what the passages explicitly state.
5. If the answer may have changed due to recent amendments, note this explicitly."""


def format_context(chunks: list[RetrievedChunk]) -> str:
    """Format retrieved chunks into a numbered context block for the prompt."""
    parts = []
    for i, chunk in enumerate(chunks, 1):
        parts.append(
            f"[Passage {i}]\n"
            f"Section: § {chunk.section_id} — {chunk.section_title}\n"
            f"Domain: {chunk.domain}\n"
            f"Text: {chunk.text.strip()}\n"
        )
    return "\n---\n".join(parts)


def generate_answer(
    query: str,
    chunks: list[RetrievedChunk],
) -> dict:
    """
    Generate a cited, grounded answer from retrieved chunks.

    Returns a dict with:
      - answer: the generated text
      - sources: list of section IDs cited
      - chunks_used: number of context passages
    """
    if not chunks:
        return {
            "answer": (
                "I could not find a relevant passage in the by-laws to answer "
                "this question. Please consult the City of Toronto website or "
                "a legal professional."
            ),
            "sources": [],
            "chunks_used": 0,
        }

    client = get_client()
    context = format_context(chunks)

    user_message = f"""Context passages:

        {context}

        Question: {query}

        Answer the question using only the context passages above. Cite section numbers."""

    response = client.messages.create(
        model=config.GENERATION_MODEL,
        max_tokens=1000,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_message}],
    )

    answer = response.content[0].text.strip()

    # Extract cited section numbers from the answer
    import re
    sources = re.findall(r'§\s*([\d]+-[\d.]+)', answer)

    return {
        "answer": answer,
        "sources": list(set(sources)),
        "chunks_used": len(chunks),
    }