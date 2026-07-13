"""
RAG Corpus Ingestion Pipeline  (CUDA + MongoDB edition)
=========================================================
- Reads all PDFs from company sub-folders under --corpus
- Extracts text via pdfplumber (falls back to pypdf)
- Generates per-company summaries using Groq LLM
- Embeds chunks on GPU via sentence-transformers + CUDA
- Stores embeddings in a FAISS GPU index (moved to CPU for persistence)
- Logs every event, PDF, chunk, summary and run stats to MongoDB

MongoDB collections (all in the database set by MONGO_DB env var):
  ingestion_runs   — one doc per script execution (start/end time, totals)
  pdf_docs         — one doc per PDF (path, page count, char count, status)
  chunks           — one doc per text chunk (text, faiss_id, metadata)
  company_summaries— one doc per company (name, summary, pdf count)
  ingestion_logs   — structured log stream (mirrors console output)

Usage:
    python build_embeddings.py --corpus ./corpus --output ./rag_store

Requirements (GPU):
    pip install pdfplumber pypdf faiss-gpu sentence-transformers groq pymongo python-dotenv tqdm
    # faiss-gpu requires a CUDA-compatible build; if unavailable fall back to faiss-cpu

.env keys used:
    GROQ_API_KEY      (required unless --skip-summaries)
    GROQ_MODEL        (optional, default llama3-8b-8192)
    MONGO_URI         (required, e.g. mongodb://localhost:27017)
    MONGO_DB          (optional, default rag_ingestion)
"""

import os
import sys
import json
import uuid
import argparse
import logging
import traceback
from datetime import datetime, timezone
from pathlib import Path

import pdfplumber
from pypdf import PdfReader
from dotenv import load_dotenv
from tqdm import tqdm
import numpy as np

# ─── Lazy imports ─────────────────────────────────────────────────────────────

def _import_faiss():
    # Try GPU build first, fall back to CPU
    try:
        import faiss
        return faiss
    except ImportError:
        raise ImportError(
            "faiss not installed.\n"
            "  GPU:  pip install faiss-gpu\n"
            "  CPU:  pip install faiss-cpu"
        )

def _import_sentence_transformers():
    try:
        from sentence_transformers import SentenceTransformer
        return SentenceTransformer
    except ImportError:
        raise ImportError("Run: pip install sentence-transformers")

def _import_groq():
    try:
        from groq import Groq
        return Groq
    except ImportError:
        raise ImportError("Run: pip install groq")

def _import_pymongo():
    try:
        from pymongo import MongoClient
        return MongoClient
    except ImportError:
        raise ImportError("Run: pip install pymongo")


# ─── MongoDB logger ───────────────────────────────────────────────────────────

class MongoHandler(logging.Handler):
    """Streams log records into MongoDB ingestion_logs collection."""

    def __init__(self, collection, run_id: str):
        super().__init__()
        self.col = collection
        self.run_id = run_id

    def emit(self, record: logging.LogRecord):
        try:
            self.col.insert_one({
                "run_id":    self.run_id,
                "ts":        datetime.now(timezone.utc),
                "level":     record.levelname,
                "logger":    record.name,
                "message":   self.format(record),
            })
        except Exception:
            pass  # never crash the pipeline because of logging


def setup_logging(mongo_col, run_id: str) -> logging.Logger:
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", "%H:%M:%S")

    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(fmt)

    mongo_h = MongoHandler(mongo_col, run_id)
    mongo_h.setFormatter(fmt)

    log = logging.getLogger("rag_ingest")
    log.setLevel(logging.INFO)
    log.addHandler(console)
    log.addHandler(mongo_h)
    return log


# ─── CUDA detection ───────────────────────────────────────────────────────────

def detect_device() -> tuple[str, int]:
    """
    Returns (device_str, cuda_device_index).
    device_str is 'cuda' or 'cpu'.
    cuda_device_index is used for FAISS GPU resources (-1 if CPU).
    """
    try:
        import torch
        if torch.cuda.is_available():
            idx = torch.cuda.current_device()
            name = torch.cuda.get_device_name(idx)
            return "cuda", idx, name
    except ImportError:
        pass
    return "cpu", -1, "CPU (torch not found or CUDA unavailable)"


# ─── PDF extraction ───────────────────────────────────────────────────────────

def extract_text_pdfplumber(pdf_path: Path) -> tuple[str, int]:
    try:
        with pdfplumber.open(pdf_path) as pdf:
            pages = [p.extract_text() or "" for p in pdf.pages]
            text = "\n\n".join(t.strip() for t in pages if t.strip())
            return text, len(pdf.pages)
    except Exception:
        return "", 0

def extract_text_pypdf(pdf_path: Path) -> tuple[str, int]:
    try:
        reader = PdfReader(str(pdf_path))
        pages = [p.extract_text() or "" for p in reader.pages]
        text = "\n\n".join(t.strip() for t in pages if t.strip())
        return text, len(reader.pages)
    except Exception:
        return "", 0

def extract_text(pdf_path: Path) -> tuple[str, int, str]:
    """Returns (text, page_count, extractor_used)."""
    text, pages = extract_text_pdfplumber(pdf_path)
    if len(text.strip()) >= 50:
        return text.strip(), pages, "pdfplumber"
    text, pages = extract_text_pypdf(pdf_path)
    return text.strip(), pages, "pypdf_fallback"


# ─── Chunking ─────────────────────────────────────────────────────────────────

def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
    words = text.split()
    if not words:
        return []
    chunks, start = [], 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunks.append(" ".join(words[start:end]))
        if end == len(words):
            break
        start += chunk_size - overlap
    return chunks


# ─── Groq summary ────────────────────────────────────────────────────────────

def summarise_company(groq_client, company_name: str, combined_text: str,
                      model: str, log, max_chars: int = 12_000) -> str:
    snippet = combined_text[:max_chars]
    prompt = (
        f"You are an expert business analyst. Below is extracted text from multiple "
        f"documents belonging to the company/folder '{company_name}'.\n\n"
        f"Write a concise structured summary covering:\n"
        f"1. What this company/entity does\n"
        f"2. Key products, services, or topics mentioned\n"
        f"3. Notable facts, figures, or dates\n"
        f"4. Any risks, challenges, or highlights\n\n"
        f"Keep the summary under 300 words.\n\n"
        f"--- DOCUMENT TEXT ---\n{snippet}\n--- END ---"
    )
    try:
        resp = groq_client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=512,
            temperature=0.3,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        log.warning(f"  Groq summary failed for '{company_name}': {e}")
        return f"[Summary unavailable — LLM error: {e}]"


# ─── FAISS helpers ────────────────────────────────────────────────────────────

def build_faiss_index(embeddings: np.ndarray, faiss, cuda_idx: int, log):
    """Build an IndexFlatIP, using GPU resources if available."""
    dim = embeddings.shape[1]
    cpu_index = faiss.IndexFlatIP(dim)

    use_gpu = cuda_idx >= 0
    if use_gpu:
        try:
            res = faiss.StandardGpuResources()
            gpu_index = faiss.index_cpu_to_gpu(res, cuda_idx, cpu_index)
            gpu_index.add(embeddings)
            # Move back to CPU for serialisation
            final_index = faiss.index_gpu_to_cpu(gpu_index)
            log.info(f"FAISS index built on GPU:{cuda_idx} — {final_index.ntotal} vectors")
            return final_index
        except Exception as e:
            log.warning(f"GPU FAISS failed ({e}), falling back to CPU index")

    cpu_index.add(embeddings)
    log.info(f"FAISS index built on CPU — {cpu_index.ntotal} vectors")
    return cpu_index


# ─── Core pipeline ────────────────────────────────────────────────────────────

def discover_company_folders(corpus_root: Path) -> list[Path]:
    return [
        child for child in sorted(corpus_root.iterdir())
        if child.is_dir() and (
            list(child.rglob("*.pdf")) + list(child.rglob("*.PDF"))
        )
    ]


def process_corpus(
    corpus_root: Path,
    output_dir: Path,
    embed_model_name: str = "all-MiniLM-L6-v2",
    chunk_size: int = 500,
    overlap: int = 50,
    skip_summaries: bool = False,
):
    load_dotenv()

    # ── Env vars ──────────────────────────────────────────────────────────────
    groq_api_key = os.getenv("GROQ_API_KEY")
    groq_model   = os.getenv("GROQ_MODEL", "llama3-8b-8192")
    mongo_uri    = os.getenv("MONGO_URI")
    mongo_db     = os.getenv("MONGO_DB", "rag_ingestion")

    if not mongo_uri:
        raise EnvironmentError("MONGO_URI not set in environment / .env file.")
    if not groq_api_key and not skip_summaries:
        raise EnvironmentError(
            "GROQ_API_KEY not set. Set it or pass --skip-summaries."
        )

    # ── Lazy imports ──────────────────────────────────────────────────────────
    faiss            = _import_faiss()
    SentenceTransformer = _import_sentence_transformers()
    Groq             = _import_groq()
    MongoClient      = _import_pymongo()

    # ── MongoDB setup ─────────────────────────────────────────────────────────
    mongo_client = MongoClient(mongo_uri)
    db = mongo_client[mongo_db]

    col_runs      = db["ingestion_runs"]
    col_pdfs      = db["pdf_docs"]
    col_chunks    = db["chunks"]
    col_summaries = db["company_summaries"]
    col_logs      = db["ingestion_logs"]

    run_id = str(uuid.uuid4())
    run_doc = {
        "run_id":        run_id,
        "started_at":    datetime.now(timezone.utc),
        "corpus_root":   str(corpus_root.resolve()),
        "output_dir":    str(output_dir.resolve()),
        "embed_model":   embed_model_name,
        "chunk_size":    chunk_size,
        "overlap":       overlap,
        "skip_summaries":skip_summaries,
        "status":        "running",
    }
    col_runs.insert_one(run_doc)

    log = setup_logging(col_logs, run_id)
    log.info(f"Run ID: {run_id}")
    log.info(f"MongoDB: {mongo_uri}  db={mongo_db}")

    # ── Device detection ──────────────────────────────────────────────────────
    device_str, cuda_idx, device_name = detect_device()
    log.info(f"Compute device: {device_name}  (device_str={device_str}, cuda_idx={cuda_idx})")

    # ── Embedding model ───────────────────────────────────────────────────────
    log.info(f"Loading embedding model: {embed_model_name} on {device_str.upper()}")
    embedder = SentenceTransformer(embed_model_name, device=device_str)

    # ── Groq client ───────────────────────────────────────────────────────────
    groq_client = None if skip_summaries else Groq(api_key=groq_api_key)

    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Discover folders ──────────────────────────────────────────────────────
    company_folders = discover_company_folders(corpus_root)
    log.info(f"Found {len(company_folders)} company folder(s) with PDFs")

    all_chunks:   list[str]  = []
    all_metadata: list[dict] = []
    company_summaries: dict[str, str] = {}

    total_pdfs        = 0
    total_failed_pdfs = 0

    # ── Per-company loop ──────────────────────────────────────────────────────
    for folder in tqdm(company_folders, desc="Companies", unit="folder"):
        company_name = folder.name
        pdfs = sorted(list(folder.rglob("*.pdf")) + list(folder.rglob("*.PDF")))

        log.info(f"[{company_name}] — {len(pdfs)} PDF(s)")

        company_text_parts: list[str] = []
        company_pdf_count  = 0

        for pdf_path in tqdm(pdfs, desc=f"  {company_name}", leave=False, unit="pdf"):
            total_pdfs += 1
            pdf_doc = {
                "run_id":    run_id,
                "company":   company_name,
                "pdf_file":  pdf_path.name,
                "pdf_path":  str(pdf_path.relative_to(corpus_root)),
                "ingested_at": datetime.now(timezone.utc),
            }

            try:
                raw_text, page_count, extractor = extract_text(pdf_path)
                pdf_doc.update({
                    "page_count":  page_count,
                    "char_count":  len(raw_text),
                    "extractor":   extractor,
                })

                if not raw_text:
                    log.warning(f"  No text extracted: {pdf_path.name}")
                    pdf_doc["status"] = "empty"
                    total_failed_pdfs += 1
                    col_pdfs.insert_one(pdf_doc)
                    continue

                company_text_parts.append(raw_text)
                company_pdf_count += 1

                chunks = chunk_text(raw_text, chunk_size=chunk_size, overlap=overlap)
                faiss_id_start = len(all_chunks)

                chunk_docs = []
                for i, chunk in enumerate(chunks):
                    faiss_id = faiss_id_start + i
                    all_chunks.append(chunk)
                    meta = {
                        "run_id":      run_id,
                        "faiss_id":    faiss_id,
                        "company":     company_name,
                        "pdf_file":    pdf_path.name,
                        "pdf_path":    str(pdf_path.relative_to(corpus_root)),
                        "chunk_index": i,
                        "chunk_total": len(chunks),
                        "char_count":  len(chunk),
                        "text":        chunk,
                        "ingested_at": datetime.now(timezone.utc),
                    }
                    all_metadata.append(meta)
                    chunk_docs.append(meta)

                if chunk_docs:
                    col_chunks.insert_many(chunk_docs)

                pdf_doc.update({
                    "status":       "ok",
                    "chunk_count":  len(chunks),
                    "faiss_id_start": faiss_id_start,
                    "faiss_id_end":   faiss_id_start + len(chunks) - 1,
                })
                log.info(
                    f"  {pdf_path.name} — {page_count}p  "
                    f"{len(raw_text)} chars  {len(chunks)} chunks  [{extractor}]"
                )

            except Exception as exc:
                log.error(f"  ERROR processing {pdf_path.name}: {exc}")
                log.debug(traceback.format_exc())
                pdf_doc["status"] = "error"
                pdf_doc["error"]  = str(exc)
                total_failed_pdfs += 1

            col_pdfs.insert_one(pdf_doc)

        # ── Groq summary ──────────────────────────────────────────────────────
        if not skip_summaries:
            combined = "\n\n".join(company_text_parts)
            if combined.strip():
                log.info(f"  Generating Groq summary for '{company_name}'…")
                summary = summarise_company(
                    groq_client, company_name, combined, groq_model, log
                )
            else:
                summary = "[No extractable text in this folder]"
        else:
            summary = "[Summaries skipped]"

        company_summaries[company_name] = summary

        col_summaries.replace_one(
            {"run_id": run_id, "company": company_name},
            {
                "run_id":      run_id,
                "company":     company_name,
                "pdf_count":   company_pdf_count,
                "summary":     summary,
                "updated_at":  datetime.now(timezone.utc),
            },
            upsert=True,
        )
        log.info(f"  Summary stored for '{company_name}'")

    # ── Build embeddings ──────────────────────────────────────────────────────
    if not all_chunks:
        msg = "No text chunks produced — check your PDF files."
        log.error(msg)
        col_runs.update_one(
            {"run_id": run_id},
            {"$set": {"status": "failed", "error": msg,
                       "ended_at": datetime.now(timezone.utc)}}
        )
        return

    log.info(f"Embedding {len(all_chunks)} chunks on {device_str.upper()} (batch=128)…")
    embeddings = embedder.encode(
        all_chunks,
        batch_size=128,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,   # enables cosine via inner product
        device=device_str,
    )
    embeddings = embeddings.astype(np.float32)
    dim = embeddings.shape[1]
    log.info(f"Embeddings shape: {embeddings.shape}  dtype={embeddings.dtype}")

    # ── FAISS index ───────────────────────────────────────────────────────────
    index = build_faiss_index(embeddings, faiss, cuda_idx, log)

    faiss_path = output_dir / "corpus.faiss"
    faiss.write_index(index, str(faiss_path))
    log.info(f"FAISS index saved → {faiss_path}")

    # ── Save local metadata JSON (for fast retrieval without Mongo) ───────────
    metadata_path = output_dir / "chunk_metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "run_id":           run_id,
                "embed_model":      embed_model_name,
                "device":           device_str,
                "chunk_size_words": chunk_size,
                "overlap_words":    overlap,
                "total_chunks":     len(all_chunks),
                "chunks": [
                    {k: v for k, v in m.items() if k != "text"}
                    for m in all_metadata
                ],
            },
            f, indent=2, ensure_ascii=False, default=str
        )
    log.info(f"Chunk metadata JSON saved → {metadata_path}")

    # ── Markdown summaries ────────────────────────────────────────────────────
    md_path = output_dir / "company_summaries.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Company Summaries\n\n")
        f.write(f"*Run ID: `{run_id}`  |  Corpus: `{corpus_root}`*\n\n---\n\n")
        for co, summ in company_summaries.items():
            f.write(f"## {co}\n\n{summ}\n\n---\n\n")
    log.info(f"Markdown summaries saved → {md_path}")

    # ── Finalise run record ───────────────────────────────────────────────────
    col_runs.update_one(
        {"run_id": run_id},
        {"$set": {
            "status":            "completed",
            "ended_at":          datetime.now(timezone.utc),
            "total_companies":   len(company_folders),
            "total_pdfs":        total_pdfs,
            "failed_pdfs":       total_failed_pdfs,
            "total_chunks":      len(all_chunks),
            "embed_dim":         dim,
            "faiss_vectors":     index.ntotal,
            "faiss_path":        str(faiss_path.resolve()),
            "device":            device_str,
            "cuda_device_name":  device_name,
        }}
    )
    log.info("Run record updated in MongoDB.")

    # ── Console summary ───────────────────────────────────────────────────────
    print("\n" + "═" * 64)
    print("  INGESTION COMPLETE")
    print("═" * 64)
    print(f"  Run ID             : {run_id}")
    print(f"  Device             : {device_name}")
    print(f"  Companies          : {len(company_folders)}")
    print(f"  PDFs processed     : {total_pdfs - total_failed_pdfs} ok / {total_failed_pdfs} failed")
    print(f"  Chunks             : {len(all_chunks)}")
    print(f"  Embedding dim      : {dim}")
    print(f"  FAISS vectors      : {index.ntotal}")
    print(f"\n  Output → {output_dir.resolve()}")
    print(f"    corpus.faiss")
    print(f"    chunk_metadata.json")
    print(f"    company_summaries.md")
    print(f"\n  MongoDB  db={mongo_db}")
    print(f"    ingestion_runs     (1 run doc)")
    print(f"    pdf_docs           ({total_pdfs} docs)")
    print(f"    chunks             ({len(all_chunks)} docs)")
    print(f"    company_summaries  ({len(company_summaries)} docs)")
    print(f"    ingestion_logs     (full log stream)")
    print("═" * 64)

    mongo_client.close()

# ─── CLI ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Build FAISS embeddings + Groq summaries from a PDF corpus. "
                    "Logs everything to MongoDB."
    )
    p.add_argument("--corpus",        type=Path, default=Path("../nifty50documents"))
    p.add_argument("--output",        type=Path, default=Path("../embeddings"))
    p.add_argument("--embed-model",   default="all-MiniLM-L6-v2")
    p.add_argument("--chunk-size",    type=int,  default=500)
    p.add_argument("--overlap",       type=int,  default=50)
    p.add_argument("--skip-summaries",action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    process_corpus(
        corpus_root      = args.corpus,
        output_dir       = args.output,
        embed_model_name = args.embed_model,
        chunk_size       = args.chunk_size,
        overlap          = args.overlap,
        skip_summaries   = args.skip_summaries,
    )