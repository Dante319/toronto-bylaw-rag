"""
Generates synthetic benchmark questions from indexed chunks using Claude.
Pulls chunks directly from Qdrant, generates 2 questions per chunk,
saves to eval/benchmark.json.

Usage:
    PYTHONPATH=. uv run eval/generate_benchmark.py
"""
import json
import logging
import re
import time
from pathlib import Path
import anthropic
from qdrant_client import QdrantClient
import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

GENERATION_PROMPT = """You are building an evaluation benchmark for a RAG system over
Toronto municipal by-laws.

Given the following by-law passage, generate 2 questions that:
1. Are answerable directly from this passage
2. Sound like something a Toronto resident would actually ask
3. Vary in type — one factual, one about a specific rule or requirement

Return a JSON array of exactly 2 objects with this schema:
[
  {{
    "question": "...",
    "answer": "...",
    "question_type": "factual" | "rule" | "process" | "edge_case"
  }}
]

Return only the JSON array — no preamble, no explanation, no markdown fences.

By-law passage:
Section: § {section_id} — {section_title}
Domain: {domain}
Text: {text}"""


def load_chunks_from_qdrant() -> list[dict]:
    client = QdrantClient(
        url=config.QDRANT_URL,
        api_key=config.QDRANT_API_KEY if config.QDRANT_API_KEY else None,
    )

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

    # Filter out chunks that are too short or are pure TOC entries
    filtered = [
        c for c in all_chunks
        if len(c["text"].strip()) >= 150
        and "§" not in c["text"][:50]  # skip TOC chunks
    ]

    logger.info(f"Loaded {len(all_chunks)} chunks, {len(filtered)} usable for generation")
    return filtered


def generate_questions_for_chunk(
    chunk: dict,
    client: anthropic.Anthropic,
) -> list[dict]:
    prompt = GENERATION_PROMPT.format(
        section_id=chunk["section_id"],
        section_title=chunk["section_title"],
        domain=chunk["domain"],
        text=chunk["text"][:800],  # cap to avoid token limits
    )

    try:
        response = client.messages.create(
            model=config.HYDE_MODEL,   # Haiku — fast and cheap for generation
            max_tokens=600,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = response.content[0].text.strip()

        # Strip markdown fences if Claude adds them despite instructions
        raw = re.sub(r"^```json\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)

        questions = json.loads(raw)
        return questions

    except (json.JSONDecodeError, Exception) as e:
        logger.warning(f"Failed for §{chunk['section_id']}: {e}")
        return []


def main():
    output_path = Path("eval/benchmark.json")
    output_path.parent.mkdir(exist_ok=True)

    chunks = load_chunks_from_qdrant()
    client = anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)

    benchmark = []
    counter = 1

    for i, chunk in enumerate(chunks):
        logger.info(f"[{i+1}/{len(chunks)}] Generating for §{chunk['section_id']}...")

        questions = generate_questions_for_chunk(chunk, client)

        for q in questions:
            benchmark.append({
                "id": f"synthetic_{counter:03d}",
                "domain": chunk["domain"],
                "question": q.get("question", ""),
                "answer": q.get("answer", ""),
                "question_type": q.get("question_type", "factual"),
                "relevant_sections": [chunk["section_id"]],
                "source": "synthetic",
            })
            counter += 1

        # Polite rate limiting — Haiku is fast but has TPM limits
        time.sleep(0.5)

    logger.info(f"Generated {len(benchmark)} synthetic questions")

    # Save synthetic questions first
    with open(output_path, "w") as f:
        json.dump(benchmark, f, indent=2)

    logger.info(f"Saved to {output_path}")


if __name__ == "__main__":
    main()