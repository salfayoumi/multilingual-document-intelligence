"""Retrieval evaluation utilities and command-line benchmark."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path

from .encoders import create_encoder
from .engine import DocumentIntelligence
from .ingestion import load_directory


@dataclass(frozen=True, slots=True)
class EvaluationCase:
    query: str
    relevant_sources: tuple[str, ...]


def load_cases(path: str | Path) -> list[EvaluationCase]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return [
        EvaluationCase(item["query"], tuple(item["relevant_sources"])) for item in payload
    ]


def evaluate(
    engine: DocumentIntelligence, cases: list[EvaluationCase], *, top_k: int = 3
) -> dict[str, float | int | str]:
    reciprocal_ranks: list[float] = []
    hits = 0
    latencies: list[float] = []
    for case in cases:
        started = time.perf_counter()
        results = engine.search(case.query, top_k=top_k)
        latencies.append((time.perf_counter() - started) * 1000)
        relevant = set(case.relevant_sources)
        rank = next(
            (
                index
                for index, result in enumerate(results, start=1)
                if result.chunk.source in relevant
            ),
            None,
        )
        hits += int(rank is not None)
        reciprocal_ranks.append(1 / rank if rank else 0.0)
    count = max(len(cases), 1)
    return {
        "cases": len(cases),
        f"hit_rate@{top_k}": round(hits / count, 4),
        "mean_reciprocal_rank": round(sum(reciprocal_ranks) / count, 4),
        "mean_latency_ms": round(sum(latencies) / count, 2),
        "encoder": engine.encoder.name,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--documents", default="demo_docs")
    parser.add_argument("--cases", default="evaluation/queries.json")
    parser.add_argument("--encoder", choices=["hashing", "semantic"], default="hashing")
    parser.add_argument("--top-k", type=int, default=3)
    args = parser.parse_args()

    engine = DocumentIntelligence(encoder=create_encoder(args.encoder))
    engine.index_documents(load_directory(args.documents))
    print(json.dumps(evaluate(engine, load_cases(args.cases), top_k=args.top_k), indent=2))


if __name__ == "__main__":
    main()
