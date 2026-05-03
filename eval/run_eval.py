"""
Evaluates the RAG pipeline against benchmark.json.

Metrics:
  - Recall@3: is the relevant section in the top 3 retrieved chunks?
  - Recall@5: same for top 5
  - HyDE delta: Recall@3 improvement from query expansion
  - RAGAS faithfulness, answer relevancy, context precision

Usage:
    PYTHONPATH=. uv run eval/run_eval.py
    PYTHONPATH=. uv run eval/run_eval.py --skip-ragas   # fast mode, no API calls for metrics
"""
import json
import logging
import argparse
from pathlib import Path
from dataclasses import dataclass, field
from pydantic import SecretStr
from anthropic import Anthropic
from ragas import evaluate
from ragas.metrics import Faithfulness, AnswerRelevancy, ContextPrecision
from langchain_anthropic import ChatAnthropic
from ragas.dataset_schema import SingleTurnSample, EvaluationDataset

from retrieval.retrieve import hybrid_search, dense_search
from retrieval.query_expand import hyde_expand, expand_and_retrieve
from retrieval.generate import generate_answer
from retrieval.retrieve import hybrid_search, get_embed_model, get_qdrant_client, get_bm25_index
import config

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)


@dataclass
class EvalResult:
    question_id: str
    question: str
    domain: str
    relevant_sections: list[str]
    retrieved_sections_no_hyde: list[str]
    retrieved_sections_hyde: list[str]
    recall_at_3_no_hyde: bool
    recall_at_3_hyde: bool
    recall_at_5_no_hyde: bool
    recall_at_5_hyde: bool
    answer: str
    retrieved_contexts: list[str]


def recall_at_k(retrieved: list[str], relevant: list[str], k: int) -> bool:
    """True if any relevant section appears in the top-k retrieved sections."""
    top_k = retrieved[:k]
    return any(r in top_k for r in relevant)


def run_retrieval_eval(benchmark: list[dict], skip_hyde: bool = False) -> list[EvalResult]:
    """Run retrieval evaluation — no RAGAS, no generation, fast and free."""
    # Pre-warm all singletons before the loop
    # Without this, BM25 rebuilds on every question
    logger.warning("Pre-warming retrieval singletons...")
    get_embed_model()
    get_qdrant_client()
    get_bm25_index()
    logger.warning("Ready — starting evaluation loop")

    results = []

    for i, item in enumerate(benchmark):
        print(f"[{i+1}/{len(benchmark)}] Starting: {item['question'][:55]}...", flush=True)
    
        if not item.get("question") or not item.get("relevant_sections"):
            print(f"  Skipping — missing question or relevant_sections", flush=True)
            continue

        logger.info(f"[{i+1}/{len(benchmark)}] {item['question'][:60]}...")

        query = item["question"]
        relevant = item["relevant_sections"]

        # Without HyDE — plain hybrid search
        no_hyde_chunks = hybrid_search(query, top_k=5, domain=None)
        no_hyde_sections = [c.section_id for c in no_hyde_chunks]

        if skip_hyde:
            hyde_chunks = no_hyde_chunks
            hyde_sections = no_hyde_sections
        else:
            try:
                hyde_chunks, _ = expand_and_retrieve(query, top_k=5, domain=None)
                hyde_sections = [c.section_id for c in hyde_chunks]
            except Exception as e:
                logger.warning(f"HyDE failed for question {item['id']}: {e}")
                hyde_sections = no_hyde_sections
                hyde_chunks = no_hyde_chunks

        # Generate answer from HyDE results for RAGAS eval
        result = generate_answer(query, hyde_chunks[:5])

        results.append(EvalResult(
            question_id=item["id"],
            question=query,
            domain=item["domain"],
            relevant_sections=relevant,
            retrieved_sections_no_hyde=no_hyde_sections,
            retrieved_sections_hyde=hyde_sections,
            recall_at_3_no_hyde=recall_at_k(no_hyde_sections, relevant, 3),
            recall_at_3_hyde=recall_at_k(hyde_sections, relevant, 3),
            recall_at_5_no_hyde=recall_at_k(no_hyde_sections, relevant, 5),
            recall_at_5_hyde=recall_at_k(hyde_sections, relevant, 5),
            answer=result["answer"],
            retrieved_contexts=[c.text for c in hyde_chunks[:5]],
        ))

    return results


def run_ragas_eval(results: list[EvalResult]) -> dict:
    """Run RAGAS metrics over the generated answers."""
    llm = ChatAnthropic(
        model_name="claude-haiku-4-5-20251001",
        api_key=SecretStr(config.ANTHROPIC_API_KEY),
        timeout=30.0,
        stop=["\n\n"],
    )

    samples = []
    for r in results:
        if not r.answer or not r.retrieved_contexts:
            continue
        samples.append(SingleTurnSample(
            user_input=r.question,
            response=r.answer,
            retrieved_contexts=r.retrieved_contexts,
            reference=r.answer,  # use generated answer as reference
        ))

    dataset = EvaluationDataset(samples=samples)

    logger.warning("Running RAGAS evaluation — this makes API calls...")
    scores = evaluate(
        dataset=dataset,
        metrics=[Faithfulness(), AnswerRelevancy(), ContextPrecision()],
        llm=llm,
    )

    return scores


def print_report(results: list[EvalResult], ragas_scores: dict | None = None):
    n = len(results)
    if n == 0:
        print("No results to report.")
        return

    recall3_no_hyde = sum(r.recall_at_3_no_hyde for r in results) / n
    recall3_hyde = sum(r.recall_at_3_hyde for r in results) / n
    recall5_no_hyde = sum(r.recall_at_5_no_hyde for r in results) / n
    recall5_hyde = sum(r.recall_at_5_hyde for r in results) / n

    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(f"Questions evaluated: {n}")
    print()
    print(f"{'Metric':<30} {'Without HyDE':>12} {'With HyDE':>10} {'Delta':>8}")
    print("-" * 62)
    print(f"{'Recall@3':<30} {recall3_no_hyde:>12.3f} {recall3_hyde:>10.3f} {recall3_hyde - recall3_no_hyde:>+8.3f}")
    print(f"{'Recall@5':<30} {recall5_no_hyde:>12.3f} {recall5_hyde:>10.3f} {recall5_hyde - recall5_no_hyde:>+8.3f}")

    if ragas_scores:
        print()
        print("RAGAS Metrics (with HyDE):")
        print("-" * 62)
        for metric, score in ragas_scores.items():
            if isinstance(score, float):
                print(f"  {metric:<28} {score:.3f}")

    print()
    print("Per-domain breakdown:")
    print("-" * 62)
    for domain in set(r.domain for r in results):
        domain_results = [r for r in results if r.domain == domain]
        n_d = len(domain_results)
        r3 = sum(r.recall_at_3_hyde for r in domain_results) / n_d
        print(f"  {domain:<28} Recall@3={r3:.3f}  (n={n_d})")

    print()
    print("Failures (Recall@3 with HyDE = False):")
    print("-" * 62)
    failures = [r for r in results if not r.recall_at_3_hyde]
    if failures:
        for r in failures[:5]:   # show first 5
            print(f"  [{r.question_id}] {r.question[:55]}")
            print(f"    Expected: {r.relevant_sections}")
            print(f"    Got:      {r.retrieved_sections_hyde[:3]}")
    else:
        print("  None — perfect Recall@3!")
    print("=" * 60)


def save_results(results: list[EvalResult], ragas_scores: dict | None):
    output = {
        "summary": {
            "n": len(results),
            "recall_at_3_no_hyde": sum(r.recall_at_3_no_hyde for r in results) / len(results),
            "recall_at_3_hyde": sum(r.recall_at_3_hyde for r in results) / len(results),
            "recall_at_5_no_hyde": sum(r.recall_at_5_no_hyde for r in results) / len(results),
            "recall_at_5_hyde": sum(r.recall_at_5_hyde for r in results) / len(results),
            "ragas": ragas_scores or {},
        },
        "per_question": [
            {
                "id": r.question_id,
                "question": r.question,
                "domain": r.domain,
                "recall_at_3_no_hyde": r.recall_at_3_no_hyde,
                "recall_at_3_hyde": r.recall_at_3_hyde,
                "retrieved_no_hyde": r.retrieved_sections_no_hyde[:5],
                "retrieved_hyde": r.retrieved_sections_hyde[:5],
                "relevant": r.relevant_sections,
            }
            for r in results
        ],
    }

    out_path = Path("eval/results.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    logger.warning(f"Full results saved to {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--skip-ragas",
        action="store_true",
        help="Skip RAGAS metrics (faster, no API calls for evaluation)"
    )
    parser.add_argument("--skip-hyde", action="store_true")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of questions for quick testing")
    args = parser.parse_args()

    benchmark_path = Path("eval/benchmark.json")
    if not benchmark_path.exists():
        print("benchmark.json not found. Run generate_benchmark.py first.")
        return

    with open(benchmark_path) as f:
        benchmark = json.load(f)
    
    if args.limit:
        benchmark = benchmark[:args.limit]

    print(f"Loaded {len(benchmark)} benchmark questions")

    results = run_retrieval_eval(benchmark, skip_hyde=args.skip_hyde)

    ragas_scores = None
    if not args.skip_ragas:
        ragas_scores = run_ragas_eval(results)

    print_report(results, ragas_scores)
    save_results(results, ragas_scores)


if __name__ == "__main__":
    main()