# Enterprise Knowledge Graph Pipeline (AbA Architecture)

This project implements an enterprise-grade pipeline for generating a strictly consistent Knowledge Graph (KG) and Ontology from unstructured text. It avoids the systematic failures of standard LLM extraction (like hallucinated constraints, category errors, and cycle creation) by employing an **Axiom-by-Axiom (AbA)** decomposed architecture and **Neuro-Symbolic Validation**.

## Key Innovations

- **Axiom-by-Axiom (AbA) Decomposition**: Properties and classes are never queried simultaneously. The T-Box schema is built sequentially through competency questions, classes, and then properties bound strictly to established domains/ranges.
- **BFO Anchoring**: Uses a custom BFO 2.0 (Basic Formal Ontology) skeleton to anchor every dynamically discovered class to an upper ontology category (e.g., `bfo:MaterialEntity`, `bfo:Process`), naturally preventing category errors.
- **Two-Way Chain-of-Thought Validation**: Eliminates hierarchical cycles and inversions by forcing the LLM to justify `subClassOf` relationships bidirectionally.
- **Neuro-Symbolic Self-Correction**: Implements a 3-layer OWL validator (Python heuristics → RDFLib → HermiT Reasoner via Owlready2) that audits the schema and loops back to the LLM up to 3 times to self-correct logical inconsistencies.
- **Aristotelian Definitions**: Enforces strict `"A [Class] is a [Parent] that [Differentia]"` textual definitions for high semantic value.
- **SLURM Cluster Ready**: Native support for HPC environments (FEP / Apptainer / Job Chaining) with built-in cache resumption for massive documents.

## Setup Instructions

1.  **Activate Virtual Environment:**
    ```bash
    # Windows
    .\.venv\Scripts\activate
    
    # Unix / WSL
    source .venv/bin/activate
    ```

2.  **Configure Environment Variables:**
    Create a `.env` file in the root directory:
    ```env
    # Choose backend: gemini, ollama, or huggingface
    LLM_BACKEND=ollama
    OLLAMA_MODEL=qwen2.5:7b
    
    # Or Google
    # LLM_BACKEND=gemini
    # GOOGLE_API_KEY=your-key
    ```

3.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Data Placement:**
    Place your source documents (PDF or TXT) in `data/input/`. (Default: `sample.pdf`).

## Execution

**Run the full architectural pipeline locally:**
```bash
python src/main.py --input "sample.pdf" --no-cache
```

**Generate only the T-Box Blueprint (Schema construction and validation):**
```bash
python src/main.py --blueprint-only
```

**Cluster Execution (FEP / SLURM):**
Using the provided `fep/launch.sh`, you can chain multiple jobs for documents that exceed cluster timeouts, or run massive parallel tests. For complete instructions, see `TUTORIAL_FEP.md`.
```bash
./fep/launch.sh --input "document_mare.pdf" --chain 5
```

## Analytics & Reporting

The project includes custom evaluation tools to measure run-to-run drift and LLM hallucination variance across multiple executions.

**Standard Evaluation:**
Compares `data/runs/run_X` locally and generates interactive `knowledge_graph.html` representations via `pyvis`.
```bash
python src/final_report.py
```

**Cluster Evaluation (Nested SLURM Jobs):**
Automatically recurses through complex HPC outputs (`data/fep_results/job_X/...`) to benchmark internal Job runs and inter-job stability.
```bash
python src/final_report_fep.py
```

## Core Components

- `src/main.py`: Orchestrates the 5-phase logic (Chunking → T-Box → Blueprint Freeze → A-Box Population → Output).
- `src/extraction.py`: Handles all LLM communication patterns, batch delta-extractions, and JSON repair.
- `src/owl_validator.py`: The 3-layer neuro-symbolic engine validating ontology coherence.
- `src/state_tracker.py`: Manages the strictly typed `subClassOf` state and exports standard Turtle (.ttl) graphs.
- `src/bfo_skeleton.py`: Provides the foundational top-level BFO alignment.
- `src/prompts.py`: Optimized multi-stage prompts using the AbA methodology.
