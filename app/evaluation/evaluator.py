import json
from pathlib import Path

from app.core.logging import logger
from app.evaluation.metrics import hit_at_k, reciprocal_rank
from app.generation.rag_pipeline import retrieve
from app.models.schemas import (
    EvaluationResponse,
    RetrievalStrategy,
    StrategyResult,
)


def load_dataset(dataset_path: str) -> list[dict]:
    """Load the evaluation dataset from a JSON file."""
    path = Path(dataset_path)
    if not path.exists():
        raise FileNotFoundError(f"Evaluation dataset not found: {dataset_path}")

    with open(path, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    logger.info(f"Loaded {len(dataset)} evaluation questions from {dataset_path}")
    return dataset


def evaluate_strategy(
    dataset: list[dict],
    strategy: RetrievalStrategy,
    top_k: int = 5,
) -> StrategyResult:
    """Evaluate a single retrieval strategy across the entire dataset."""
    hits = []
    mrrs = []
    details = []

    for item in dataset:
        question = item["question"]
        relevant_sources = item["relevant_sources"]

        results, _candidate_count, _retrieval_ms, _reranking_ms = retrieve(
            question, strategy, top_k=top_k
        )

        hit = hit_at_k(results, relevant_sources, k=top_k)
        mrr = reciprocal_rank(results, relevant_sources)

        hits.append(hit)
        mrrs.append(mrr)

        details.append({
            "id": item.get("id", ""),
            "question": question,
            "category": item.get("category", ""),
            "hit": hit,
            "mrr": mrr,
            "top_result": results[0].document_name if results else "",
            "top_page": results[0].page_number if results else None,
        })

    avg_hit = sum(hits) / len(hits) if hits else 0.0
    avg_mrr = sum(mrrs) / len(mrrs) if mrrs else 0.0

    logger.info(f"Strategy {strategy.value}: Hit@{top_k}={avg_hit:.3f}, MRR={avg_mrr:.3f}")

    return StrategyResult(
        strategy=strategy.value,
        hit_rate_at_5=round(avg_hit, 4),
        mrr=round(avg_mrr, 4),
        details=details,
    )


def run_evaluation(dataset_path: str = "evaluation/dataset.json") -> EvaluationResponse:
    """Run evaluation across all three retrieval strategies and return comparison."""
    dataset = load_dataset(dataset_path)

    strategies = [
        RetrievalStrategy.VECTOR,
        RetrievalStrategy.HYBRID,
        RetrievalStrategy.HYBRID_RERANK,
    ]

    results = []
    for strategy in strategies:
        result = evaluate_strategy(dataset, strategy)
        results.append(result)

    # Save results
    output_path = Path("evaluation/results.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(
            [r.model_dump() for r in results],
            f,
            indent=2,
        )
    logger.info(f"Evaluation results saved to {output_path}")

    return EvaluationResponse(results=results)
