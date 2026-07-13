# Nifty50 Data Pipeline

An automated data engineering pipeline that collects live market data from the National Stock Exchange (NSE), downloads corporate announcements, generates semantic embeddings, and prepares retrieval-ready artifacts for an AI-powered financial chatbot.

The pipeline is designed to run on a daily schedule, ensuring that the chatbot always has access to the latest market information and corporate disclosures.

---

## Overview

The pipeline performs the following tasks every day:

1. Scrapes the latest NIFTY 50 market snapshot.
2. Downloads the latest Option Chain data.
3. Retrieves newly published corporate announcement PDFs for all NIFTY 50 companies.
4. Processes new documents into semantic chunks.
5. Generates vector embeddings.
6. Builds a FAISS index for fast similarity search.
7. Produces retrieval-ready artifacts consumed by the chatbot.

---

## Project Structure

```text
.
├── pdf_downloader.py          # Downloads new corporate announcement PDFs
├── nifty_50_scraper.py        # Scrapes live NIFTY 50 market table
├── option_chain_scraper.py    # Downloads Option Chain CSV
├── build_embeddings.py        # Generates chunks, embeddings and FAISS index
├── run_scraper.py             # Runs the complete daily pipeline
│
├── requirements.txt
├── sample.env
└── README.md
```

---

## Pipeline Workflow

```text
                NSE Website
                     │
        ┌────────────┴────────────┐
        │                         │
        ▼                         ▼
NIFTY 50 Scraper          Option Chain Scraper
        │                         │
        └────────────┬────────────┘
                     │
                     ▼
          Corporate PDF Downloader
                     │
                     ▼
             PDF Processing & Chunking
                     │
                     ▼
          Embedding Generation (MPNet)
                     │
                     ▼
                FAISS Index Build
                     │
                     ▼
        Retrieval Artifacts Generated
                     │
                     ▼
          Hugging Face Dataset Update
```

---

## Generated Artifacts

The pipeline produces the following files for the chatbot:

* `corpus.faiss`
* `chunk_metadata.json`
* `company_summaries.md`
* `nifty50_snapshot.json`
* `option_chain.csv`

These artifacts are published to a Hugging Face Dataset and are consumed directly by the chatbot during inference.

---

## Technologies Used

* Python
* Playwright
* BeautifulSoup
* PyMuPDF
* Sentence Transformers
* FAISS
* MongoDB Atlas
* Hugging Face Hub

---

## Scheduling

The pipeline is intended to run automatically once every trading day after market close.

The `run_scraper.py` script orchestrates the complete workflow:

```bash
python run_scraper.py
```

---

## Future Improvements

* Incremental embedding updates instead of rebuilding the full index.
* Automatic upload of generated artifacts to Hugging Face Dataset.
* Historical market data archival.
* Retry mechanism and pipeline monitoring.
* Dockerized deployment with scheduled execution.

---

## Related Project

This repository generates the data consumed by the **Nifty50 Agentic Chatbot**, which uses LangGraph, Retrieval-Augmented Generation (RAG), and Groq LLMs to answer market-related questions using these generated artifacts.
