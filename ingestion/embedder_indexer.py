# ingestion/embed_index.py
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, PointStruct, PayloadSchemaType
)
from ingestion.chunker import Chunk
import config
import logging
from tqdm import tqdm
import uuid

logger = logging.getLogger(__name__)


def get_qdrant_client() -> QdrantClient:
    return QdrantClient(
        url=config.QDRANT_URL,
        api_key=config.QDRANT_API_KEY,
    )


def setup_collection(client: QdrantClient, recreate: bool = False):
    """Create the Qdrant collection if it doesn't exist."""
    existing = [c.name for c in client.get_collections().collections]

    if config.COLLECTION_NAME in existing:
        if recreate:
            client.delete_collection(config.COLLECTION_NAME)
            logger.info(f"Deleted existing collection: {config.COLLECTION_NAME}")
        else:
            logger.info(f"Collection already exists: {config.COLLECTION_NAME}")
            return

    client.create_collection(
        collection_name=config.COLLECTION_NAME,
        vectors_config=VectorParams(
            size=config.EMBED_DIMENSION,
            distance=Distance.COSINE,
        ),
    )

    # Index payload fields for fast filtering
    for field_name in ("domain", "section_id", "parent_section"):
        client.create_payload_index(
            collection_name=config.COLLECTION_NAME,
            field_name=field_name,
            field_schema=PayloadSchemaType.KEYWORD,
        )

    logger.info(f"Created collection: {config.COLLECTION_NAME}")


def embed_and_index(chunks: list[Chunk], recreate: bool = False):
    """
    Embed all chunks with BGE-large and upsert to Qdrant.
    Runs in batches to avoid OOM.
    """
    client = get_qdrant_client()
    setup_collection(client, recreate=recreate)

    logger.info(f"Loading embedding model: {config.EMBED_MODEL}")
    model = SentenceTransformer(config.EMBED_MODEL)

    texts = [chunk.text for chunk in chunks]
    logger.info(f"Embedding {len(texts)} chunks in batches of {config.EMBED_BATCH_SIZE}...")

    # BGE-large needs this prefix for retrieval (from model card)
    prefixed = [f"Represent this sentence for searching relevant passages: {t}" for t in texts]

    all_embeddings = []
    for i in tqdm(range(0, len(prefixed), config.EMBED_BATCH_SIZE)):
        batch = prefixed[i:i + config.EMBED_BATCH_SIZE]
        embeddings = model.encode(batch, normalize_embeddings=True)
        all_embeddings.extend(embeddings)

    # Build Qdrant points
    points = []
    for chunk, embedding in zip(chunks, all_embeddings):
        points.append(PointStruct(
            id=str(uuid.uuid4()),
            vector=embedding.tolist(),
            payload={
                "chunk_id": chunk.chunk_id,
                "domain": chunk.domain,
                "section_id": chunk.section_id,
                "section_title": chunk.section_title,
                "parent_section": chunk.parent_section,
                "text": chunk.text,
                "source_file": chunk.source_file,
                "page": chunk.page,
            }
        ))

    # Upsert in batches of 100
    logger.info(f"Upserting {len(points)} points to Qdrant...")
    for i in tqdm(range(0, len(points), 100)):
        client.upsert(
            collection_name=config.COLLECTION_NAME,
            points=points[i:i + 100],
        )

    logger.info("Indexing complete.")