#!/usr/bin/env python3
"""End-to-end retrieval evaluation runner.

This script:
1. Creates the evaluation corpus PDFs (if missing)
2. Resets and ingests the evaluation documents
3. Runs retrieval for all questions across vector / hybrid / hybrid_rerank
4. Computes Hit@5 and MRR from actual retrieved results
5. Saves real output to evaluation/results.json

No LLM calls are made — this evaluates retrieval only.

Usage:
    python scripts/run_evaluation.py
"""

import shutil
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from app.core.logging import logger


def ensure_corpus():
    """Create the evaluation corpus if PDFs don't exist."""
    corpus_dir = Path("evaluation/corpus")
    handbook = corpus_dir / "employee_handbook.pdf"
    manual = corpus_dir / "product_manual.pdf"

    if handbook.exists() and manual.exists():
        logger.info("Evaluation corpus already exists")
        return

    logger.info("Creating evaluation corpus...")
    from scripts.create_evaluation_corpus import main as create_corpus

    create_corpus()


def reset_data():
    """Remove existing data so evaluation starts clean."""
    data_dirs = [Path("data/qdrant"), Path("data/index"), Path("data/uploads")]

    for d in data_dirs:
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True, exist_ok=True)
        # Restore .gitkeep so git tracks empty directories
        (d / ".gitkeep").touch()

    logger.info("Data directories reset")


def ingest_corpus():
    """Ingest the evaluation corpus documents."""
    from app.ingestion.pipeline import ingest_document

    corpus_dir = Path("evaluation/corpus")
    files = sorted(corpus_dir.glob("*.pdf"))

    if not files:
        logger.error("No PDF files found in evaluation/corpus/")
        sys.exit(1)

    for file_path in files:
        logger.info(f"Ingesting: {file_path.name}")
        doc_id, chunks = ingest_document(file_path, file_path.name)
        logger.info(f"  → {len(chunks)} chunks (doc_id={doc_id})")


def run():
    """Run the full evaluation pipeline."""
    dataset_path = "evaluation/dataset.json"
    if len(sys.argv) > 1:
        dataset_path = sys.argv[1]

    # Step 1: Ensure corpus exists
    ensure_corpus()

    # Step 2: Reset data and ingest
    reset_data()

    # Re-initialize singleton state after data reset
    from app.ingestion.registry import chunk_registry

    chunk_registry.path = Path("data/index/chunks.jsonl")
    chunk_registry.chunks = []
    chunk_registry._doc_map = {}

    from app.vectorstore.qdrant_store import qdrant_store

    qdrant_store._client = None

    ingest_corpus()

    # Step 3: Run evaluation
    from app.evaluation.evaluator import run_evaluation

    logger.info(f"Running evaluation with dataset: {dataset_path}")
    response = run_evaluation(dataset_path)

    # Step 4: Print results
    print("\n" + "=" * 60)
    print(f"{'Strategy':<25} {'Hit@5':<12} {'MRR':<12}")
    print("-" * 60)
    for result in response.results:
        print(f"{result.strategy:<25} {result.hit_rate_at_5:<12.4f} {result.mrr:<12.4f}")
    print("=" * 60)

    for result in response.results:
        print(f"\n--- {result.strategy} ---")
        for detail in result.details:
            hit_mark = "✓" if detail["hit"] == 1.0 else "✗"
            print(f"  {hit_mark} [{detail['category']}] {detail['question']}")
            print(
                f"    MRR: {detail['mrr']:.4f} | "
                f"Top: {detail['top_result']} p.{detail['top_page']}"
            )

    print(f"\nResults saved to evaluation/results.json")


if __name__ == "__main__":
    run()
