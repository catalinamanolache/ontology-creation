# Deterministic Ontology Generation

This project implements an enterprise-grade pipeline for generating a single, deterministic Knowledge Graph (KG) and Ontology from unstructured text using Large Language Models (LLMs), as described in algorithm "From Fragmentation to Cohesion".

## Features

- **Semantic Chunking:** Deterministic cleaning and splitting of documents.
- **Structured LLM Extractions:** Pydantic schemas enforce strict JSON generation using `gpt-4o-mini` (or configurable models).
- **Topological Refinement Loop (Phase 4):** Automatically bridges disconnected knowledge subgraphs using Semantic Search (ChromaDB) and NetworkX Degree Centrality.
- **Dual Instantiation Test (Phase 5):** Evaluates stability mathematically using Jaccard Graph Similarity ensuring 95%+ reproducibility ($\tau \ge 0.7$).

## Setup Instructions

1. **Activate Virtual Environment:**
   Run the pipeline inside the prepared virtual environment to ensure all dependencies match `requirements.txt`.
   ```bash
   .\.venv\Scripts\activate
   ```

2. **Configure Environment Variables:**
   A `.env` file must be present in the root directory. Update it with a valid API key. (Routing must be enabled if using OpenRouter).
   ```env
   OPENAI_API_KEY=your-api-key-here
   OPENAI_BASE_URL=https://api.openai.com/v1  # Or https://openrouter.ai/api/v1
   MODEL_NAME=gpt-4o-mini                     # Or openai/gpt-4o-mini for OpenRouter
   ```

3. **Data Placement:**
   Drop your PDF documents into `data/input`. (The demo uses `sample.pdf`).

4. **Execution:**
   Run the main pipeline. This will process the documents, extract the triples, resolve the graph topology, and finally verify its mathematical determinism by running the test twice (Graph A and Graph B).
   ```bash
   python src/main.py
   ```

## Design Architecture

- `src/schemas.py`: Core Pydantic data structures acting as strict guardrails.
- `src/prompts.py`: The specialized Agentic prompts for Schema Bootstrapping, Property Extraction, and Subgraph Bridging.
- `src/document_processor.py`: Loads and normalizes PDFs deterministically.
- `src/state_tracker.py`: Maintains the global approved classes and properties.
- `src/graph_builder.py`: Uses NetworkX to maintain topological state and calculate degree centralities.
- `src/semantic_search.py`: Local ChromaDB vector store used to retrieve exact textual evidence connecting two disjoint graph islands.
- `src/verifier.py`: Calculates mathematical Jaccard similarity across multiple LLM runs.
