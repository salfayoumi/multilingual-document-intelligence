"""Run a local multilingual retrieval and citation demonstration."""

from __future__ import annotations

import argparse
from pathlib import Path

from .encoders import create_encoder
from .engine import DocumentIntelligence
from .ingestion import load_directory


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("query", nargs="?", default="When should the vibration alarm be escalated?")
    parser.add_argument("--documents", default="demo_docs")
    parser.add_argument("--encoder", choices=["hashing", "semantic"], default="hashing")
    args = parser.parse_args()

    documents_path = Path(args.documents)
    engine = DocumentIntelligence(encoder=create_encoder(args.encoder))
    engine.index_documents(load_directory(documents_path))
    answer = engine.ask(args.query)
    print(answer.text)
    for citation in answer.citations:
        page = f", p. {citation.page}" if citation.page else ""
        print(f"\n[{citation.number}] {citation.source}{page}\n{citation.snippet}")


if __name__ == "__main__":
    main()

