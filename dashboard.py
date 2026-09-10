"""Streamlit interface for the multilingual document-intelligence engine."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from document_intelligence import DocumentIntelligence, document_from_text, load_document
from document_intelligence.encoders import create_encoder
from document_intelligence.evaluation import evaluate, load_cases
from document_intelligence.ingestion import SUPPORTED_SUFFIXES, load_directory

ROOT = Path(__file__).parent
load_dotenv(ROOT / ".env")

st.set_page_config(
    page_title="Multilingual Document Intelligence",
    page_icon="◈",
    layout="wide",
)

st.markdown(
    """
    <style>
    :root { --violet: #6c4cff; --coral: #ff6b6b; --ink: #172033; }
    .block-container { max-width: 1180px; padding-top: 2.4rem; }
    [data-testid="stMetric"] { border: 1px solid rgba(108,76,255,.2); border-radius: 16px;
        padding: 1rem; background: linear-gradient(145deg,
        rgba(108,76,255,.08), rgba(255,107,107,.04)); }
    .hero { padding: 1.5rem 0 1rem; }
    .eyebrow { color: #6c4cff; font-weight: 750; letter-spacing: .12em; font-size: .75rem; }
    .hero h1 { font-size: clamp(2.2rem, 5vw, 4.2rem); line-height: .98; margin: .35rem 0 1rem; }
    .hero p { max-width: 760px; font-size: 1.05rem; opacity: .78; }
    .source-card { border-left: 4px solid #6c4cff; padding: .9rem 1rem; margin: .7rem 0;
        border-radius: 0 12px 12px 0; background: rgba(108,76,255,.06); }
    .lang-pill { display:inline-block; padding:.2rem .55rem; border-radius:999px;
        background:rgba(255,107,107,.13); margin-right:.35rem; font-size:.75rem; font-weight:700; }
    </style>
    <div class="hero">
      <div class="eyebrow">ARABIC · TÜRKÇE · ENGLISH</div>
      <h1>Ask your documents.<br/>See the evidence.</h1>
      <p>A multilingual retrieval system that answers from uploaded documents, cites every
      source, and refuses to guess when the evidence is not there.</p>
    </div>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="Loading multilingual retrieval model…")
def cached_encoder(mode: str):
    return create_encoder(mode)


def new_engine(mode: str) -> DocumentIntelligence:
    return DocumentIntelligence(encoder=cached_encoder(mode))


if "encoder_mode" not in st.session_state:
    st.session_state.encoder_mode = os.getenv("DOCUMENT_AI_ENCODER", "semantic").casefold()
if "engine" not in st.session_state:
    st.session_state.engine = new_engine(st.session_state.encoder_mode)

with st.sidebar:
    st.header("Workspace")
    selected_label = st.selectbox(
        "Retrieval model",
        ["Semantic · cross-language", "Offline · same-language"],
        index=0 if st.session_state.encoder_mode == "semantic" else 1,
        help=(
            "Semantic mode matches meaning across Arabic, Turkish, and English. "
            "Offline mode is a faster lexical fallback for same-language questions."
        ),
    )
    requested_mode = "semantic" if selected_label.startswith("Semantic") else "hashing"
    if requested_mode != st.session_state.encoder_mode:
        try:
            st.session_state.engine = new_engine(requested_mode)
            st.session_state.encoder_mode = requested_mode
            st.success("Retrieval model changed. Re-index your documents.")
        except (RuntimeError, ValueError) as exc:
            st.error(str(exc))

    if st.button("Load multilingual demo", use_container_width=True):
        st.session_state.engine.add_documents(load_directory(ROOT / "demo_docs"))
        detected = " · ".join(code.upper() for code in st.session_state.engine.stats()["languages"])
        st.success(f"Demo documents indexed · {detected}")

    uploads = st.file_uploader(
        "Add documents",
        type=[suffix.lstrip(".") for suffix in sorted(SUPPORTED_SUFFIXES)],
        accept_multiple_files=True,
    )
    if uploads and st.button("Index uploaded files", type="primary", use_container_width=True):
        documents = []
        for upload in uploads:
            suffix = Path(upload.name).suffix.casefold()
            if suffix in {".txt", ".md"}:
                documents.append(document_from_text(upload.name, upload.getvalue().decode("utf-8")))
                continue
            with tempfile.NamedTemporaryFile(suffix=suffix) as temporary:
                temporary.write(upload.getvalue())
                temporary.flush()
                loaded = load_document(temporary.name)
            documents.append(
                document_from_text(
                    upload.name,
                    loaded.text,
                    language=loaded.language,
                    metadata=loaded.metadata,
                )
            )
        st.session_state.engine.add_documents(documents)
        detected = " · ".join(code.upper() for code in st.session_state.engine.stats()["languages"])
        st.success(f"Indexed {len(documents)} document(s) · detected {detected}")

    if st.session_state.engine.documents:
        with st.expander("Indexed documents", expanded=False):
            for document in st.session_state.engine.documents.values():
                languages = " · ".join(code.upper() for code in document.languages) or "UNKNOWN"
                st.caption(f"{document.name} · {languages}")

    if st.button("Clear session", use_container_width=True):
        st.session_state.engine.clear()
        st.rerun()

engine = st.session_state.engine
stats = engine.stats()
metric_cols = st.columns(4)
metric_cols[0].metric("Documents", stats["documents"])
metric_cols[1].metric("Passages", stats["chunks"])
language_codes = tuple(stats["languages"])
answer_label = "Grounded" if "grounded" in str(stats["answer_mode"]) else "Extractive"
metric_cols[2].metric("Languages", len(language_codes))
metric_cols[3].metric("Answer mode", answer_label)
if language_codes:
    st.caption("Detected languages: " + " · ".join(code.upper() for code in language_codes))

ask_tab, search_tab, evaluate_tab = st.tabs(["Ask", "Inspect retrieval", "Evaluate"])

with ask_tab:
    st.subheader("Ask a question")
    query = st.text_area(
        "Question",
        placeholder="Try: When should the vibration alarm be escalated?",
        height=100,
        label_visibility="collapsed",
    )
    if st.button("Find a grounded answer", type="primary", disabled=not query.strip()):
        if not engine.documents:
            st.warning("Load the demo or upload documents first.")
        else:
            answer = engine.ask(query)
            if answer.supported:
                st.markdown(answer.text)
                chunk_languages = {chunk.id: chunk.language for chunk in engine.chunks}
                evidence_languages = tuple(
                    dict.fromkeys(
                        chunk_languages.get(citation.chunk_id, "unknown")
                        for citation in answer.citations
                    )
                )
                evidence_label = " · ".join(code.upper() for code in evidence_languages)
                st.caption(
                    f"Evidence confidence {answer.confidence:.0%} · {answer.mode} · "
                    f"query {answer.language.upper()} · evidence {evidence_label}"
                )
                st.subheader("Evidence")
                for citation in answer.citations:
                    page = f" · page {citation.page}" if citation.page else ""
                    passage_language = chunk_languages.get(citation.chunk_id, "unknown").upper()
                    st.markdown(
                        f"<div class='source-card'><strong>[{citation.number}] "
                        f"{citation.source} · {passage_language}</strong>{page}<br>"
                        f"{citation.snippet}</div>",
                        unsafe_allow_html=True,
                    )
            else:
                st.warning(answer.text)

with search_tab:
    st.subheader("See how retrieval was ranked")
    search_query = st.text_input("Search documents", key="search_query")
    dense_weight = st.slider("Semantic weight", 0.0, 1.0, 0.60, 0.05)
    if search_query and engine.documents:
        for result in engine.search(search_query, top_k=8, dense_weight=dense_weight):
            with st.expander(
                f"#{result.rank} · {result.chunk.source} · evidence {result.score:.0%}"
            ):
                st.markdown(
                    f"<span class='lang-pill'>{result.chunk.language.upper()}</span>",
                    unsafe_allow_html=True,
                )
                st.write(result.chunk.text)
                st.caption(
                    f"Dense {result.dense_score:.3f} · BM25 {result.lexical_score:.3f}"
                )

with evaluate_tab:
    st.subheader("Measure retrieval quality")
    st.write("The included benchmark reports hit rate, reciprocal rank, and query latency.")
    if st.button("Run demo benchmark", disabled=not engine.documents):
        cases = load_cases(ROOT / "evaluation" / "queries.json")
        metrics = evaluate(engine, cases, top_k=3)
        st.json(metrics)
