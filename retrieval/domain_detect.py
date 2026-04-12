import numpy as np
from sentence_transformers import SentenceTransformer
import config
import logging

logger = logging.getLogger(__name__)

_domain_embeddings: dict[str, np.ndarray] | None = None


def _get_domain_embeddings(model: SentenceTransformer) -> dict[str, np.ndarray]:
    """
    Embed all domain descriptions once and cache.
    Re-uses the same model instance passed in — no extra loading.
    """
    global _domain_embeddings
    if _domain_embeddings is None:
        logger.info("Computing domain description embeddings...")
        _domain_embeddings = {
            domain: model.encode(description, normalize_embeddings=True)
            for domain, description in config.DOMAIN_DESCRIPTIONS.items()
        }
    return _domain_embeddings


def detect_domain(
    query: str,
    model: SentenceTransformer,
    threshold: float = config.DOMAIN_DETECTION_THRESHOLD,
) -> tuple[str | None, dict[str, float]]:
    """
    Classify query into a domain by semantic similarity.

    Returns:
        domain: the matched domain key, or None if below threshold
        scores: similarity score per domain (useful for UI display)
    """
    query_embedding = model.encode(
        f"Represent this sentence for searching relevant passages: {query}",
        normalize_embeddings=True
    )

    domain_embeddings = _get_domain_embeddings(model)

    scores = {
        domain: float(np.dot(query_embedding, domain_emb))
        for domain, domain_emb in domain_embeddings.items()
    }

    best_domain = max(scores, key=scores.__getitem__)
    best_score = scores[best_domain]

    if best_score >= threshold:
        logger.info(f"Domain detected: {best_domain} (score={best_score:.3f})")
        return best_domain, scores
    else:
        logger.info(f"Domain ambiguous — best={best_domain} ({best_score:.3f}), searching all")
        return None, scores