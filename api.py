"""FastAPI service for multilingual document indexing and grounded answers."""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Annotated

from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel, Field

from document_intelligence import DocumentIntelligence, document_from_text, load_document
from document_intelligence.ingestion import SUPPORTED_SUFFIXES, load_directory

load_dotenv()

app = FastAPI(
    title="Multilingual Document Intelligence",
    description="Hybrid retrieval and citation-first answers for Arabic, Turkish, and English.",
    version="0.1.0",
)
engine = DocumentIntelligence()


class TextDocumentRequest(BaseModel):
    name: str = Field(min_length=1, max_length=180)
    text: str = Field(min_length=1)
    language: str | None = Field(default=None, pattern="^(ar|tr|en|mixed|unknown)$")


class QueryRequest(BaseModel):
    query: str = Field(min_length=2, max_length=2_000)
    top_k: int = Field(default=5, ge=1, le=12)
    dense_weight: float = Field(default=0.60, ge=0, le=1)


def _result_payload(result) -> dict:
    return {
        "rank": result.rank,
        "score": round(result.score, 4),
        "dense_score": round(result.dense_score, 4),
        "lexical_score": round(result.lexical_score, 4),
        "source": result.chunk.source,
        "language": result.chunk.language,
        "chunk_id": result.chunk.id,
        "page": result.chunk.metadata.get("page"),
        "text": result.chunk.text,
    }


@app.get("/health")
def health() -> dict:
    return {"status": "ok", **engine.stats()}


@app.post("/documents/text", status_code=201)
def add_text_document(request: TextDocumentRequest) -> dict:
    document = document_from_text(
        request.name, request.text, language=request.language, metadata={"source": "api"}
    )
    return engine.add_documents([document])


@app.post("/documents/files", status_code=201)
async def add_files(files: Annotated[list[UploadFile], File(...)]) -> dict:
    documents = []
    for upload in files:
        suffix = Path(upload.filename or "").suffix.casefold()
        if suffix not in SUPPORTED_SUFFIXES:
            raise HTTPException(
                status_code=415, detail=f"Unsupported file type: {suffix or 'none'}"
            )
        content = await upload.read()
        if len(content) > 15 * 1024 * 1024:
            raise HTTPException(status_code=413, detail="Each file must be 15 MB or smaller")
        with tempfile.NamedTemporaryFile(suffix=suffix) as temporary:
            temporary.write(content)
            temporary.flush()
            document = load_document(temporary.name)
        documents.append(
            document_from_text(
                upload.filename or "uploaded-document",
                document.text,
                language=document.language,
                metadata=document.metadata,
            )
        )
    return engine.add_documents(documents)


@app.post("/demo", status_code=201)
def load_demo_documents() -> dict:
    demo_path = Path(__file__).parent / "demo_docs"
    return engine.add_documents(load_directory(demo_path))


@app.post("/search")
def search(request: QueryRequest) -> dict:
    try:
        results = engine.search(
            request.query, top_k=request.top_k, dense_weight=request.dense_weight
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return {"query": request.query, "results": [_result_payload(item) for item in results]}


@app.post("/ask")
def ask(request: QueryRequest) -> dict:
    try:
        answer = engine.ask(
            request.query, top_k=request.top_k, dense_weight=request.dense_weight
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return answer.to_dict()


@app.delete("/documents")
def clear_documents() -> dict:
    engine.clear()
    return {"status": "cleared", **engine.stats()}
