# Deterministic Ontology Generation

This project implements an enterprise-grade pipeline for generating a strictly consistent Knowledge Graph (KG) and Ontology from unstructured text. It uses a **2-pass deterministic architecture** to ensure high reproducibility and structural integrity.

## Key Features

- **2nd Pass Determinism**: Phase 2 builds a frozen "T-Box" blueprint (schema) which is then strictly enforced during the Phase 3 population.
- **Delta Extraction**: During schema extension, the LLM only returns new discoveries to handle long documents and save tokens.
- **Multi-Backend Support**: Configure your preferred backend (Google Gemini, local Ollama, or Hugging Face Inference API).
- **Automated JSON Repair**: Custom logic to fix truncated or malformed JSON outputs from weaker or restricted models.
- **Evolutionary Reporting**: Automatic sequential comparison of runs with Jaccard similarity metrics (via `final_report.py`).

## Setup Instructions

1.  **Activate Virtual Environment:**
    ```bash
    .\.venv\Scripts\activate
    ```

2.  **Configure Environment Variables:**
    Create a `.env` file in the root directory:
    ```env
    # Choose backend: gemini, ollama, or huggingface
    LLM_BACKEND=gemini
    
    # Gemini Config
    GOOGLE_API_KEY=your-key
    
    # Hugging Face Config
    HUGGINGFACE_API_KEY=your-key
    HUGGINGFACE_MODEL=Qwen/Qwen2.5-7B-Instruct
    HF_USE_API=True
    
    # Ollama Config
    OLLAMA_MODEL=qwen2.5:7b
    ```

3.  **Data Placement:**
    Place your source documents (PDF or TXT) in `data/input/`. (Default: `sample.pdf`).

4.  **Execution:**
    Run the main pipeline:
    ```bash
    python src/main.py --no-cache
    ```
    To only generate the schema (Phase 1 & 2):
    ```bash
    python src/main.py --blueprint-only
    ```

5.  **View Reports:**
    Generate and view the stability report:
    ```bash
    python src/final_report.py
    ```

## Core Components

- `src/main.py`: Orchestrates the 2nd pass logic and archives results.
- `src/extraction.py`: Handles LLM calls, JSON repair, and caching.
- `src/state_tracker.py`: Manages the cumulative ontology state and Turtle (.ttl) export.
- `src/prompts.py`: Optimized system prompts with anti-hallucination and delta-extraction rules.
- `src/final_report.py`: Compares sequential runs to measure stability and drift.
- `src/document_processor.py`: Deterministic PDF cleaning and recursive chunking.
