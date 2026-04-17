# ingestion/runner.py
"""
Run this script once to parse, chunk, and index all source documents.
Make sure Qdrant is running (docker start qdrant) before executing.

Usage:
    uv run ingestion/runner.py
    uv run ingestion/runner.py --recreate   # wipe and rebuild index
"""
import argparse
import logging
from pathlib import Path

from ingestion.parser import parse_pdf
from ingestion.chunker import chunk_document
from ingestion.embedder_indexer import embed_and_index
import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ── Document registry ──────────────────────────────────────────────────────────
# Add your source documents here as you acquire them.
# Each entry is (parser_fn, source, domain)

DOCUMENTS = [
    ("pdf", config.DATA_RAW / "ch547_short_term_rental.pdf", "short_term_rental"),
    ("pdf", config.DATA_RAW / "ch591_noise.pdf",             "noise")
]


def main(recreate: bool = False):
    all_chunks = []

    for parser_type, source, domain in DOCUMENTS:
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing: {domain}")

        if parser_type == "pdf":
            if not Path(source).exists():
                logger.warning(f"File not found, skipping: {source}")
                continue
            doc = parse_pdf(Path(source), domain)
        else:
            logger.warning(f"Unsupported parser type, skipping: {parser_type}")
            continue

        chunks = chunk_document(doc)
        all_chunks.extend(chunks)
        logger.info(f"  Total chunks so far: {len(all_chunks)}")

    logger.info(f"\n{'='*60}")
    logger.info(f"Total chunks to index: {len(all_chunks)}")
    embed_and_index(all_chunks, recreate=recreate)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--recreate", action="store_true",
                        help="Delete and rebuild the Qdrant collection from scratch")
    args = parser.parse_args()
    main(recreate=args.recreate)