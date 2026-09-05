# Enterprise Knowledge Graph Pipeline (AbA Architecture)

This project builds a Knowledge Graph (KG) and ontology from unstructured text using a decomposed **Axiom-by-Axiom (AbA)** pipeline and neuro-symbolic validation.

## Key Innovations

- **Axiom-by-Axiom decomposition**: competency questions, classes, and properties are extracted in separate steps.
- **BFO anchoring**: each discovered class is attached to an upper ontology category.
- **Two-way hierarchy validation**: subclass relations are checked forward and reverse.
- **Neuro-symbolic correction loop**: Python checks, RDF checks, and OWL reasoner feedback are used to fix schema issues.
- **Aristotelian definitions**: classes are forced into high-signal definition format.

## Setup

1. Activate your virtual environment.
2. Create a `.env` file in the repository root.
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Place source documents (`.pdf` or `.txt`) in `data/input/`.

Example `.env`:
```env
LLM_BACKEND=ollama
OLLAMA_MODEL=qwen2.5:7b
# or use gemini/huggingface with their API keys
```

## Execution

Run full pipeline:
```bash
python src/main.py --input "sample.pdf"
```

Run only schema construction (blueprint):
```bash
python src/main.py --blueprint-only
```

Evaluate local run-to-run stability:
```bash
python src/final_report.py
```

## Technical Flow

### Phase 1 — Deterministic Document Processing
- Input is loaded from PDF/TXT.
- Text is cleaned with deterministic rules (metadata/noise removal).
- Reference sections are trimmed when present.
- Text is split into stable overlapping chunks.
- Chunks are saved for reproducibility.

### Phase 2 — Decomposed T-Box Construction
- **2.1 Competency Questions**: generate ontology questions that guide extraction.
- **2.2 Class Extraction**: extract classes only, with BFO parent and Aristotelian definition.
- **2.4 Property Extraction**: extract properties only, constrained by already-approved classes for domain/range.
- **2.5 Hierarchy Validation**:
  - Python DFS cycle detection and auto-removal.
  - LLM two-way subclass validation (forward must hold, reverse must not).
- **2.6 OWL Validation + Self-Correction**:
  - Layer 1: Python heuristic checks.
  - Layer 2: RDFLib graph/serialization checks.
  - Layer 3: owlready2/HermiT (when available).
  - If inconsistent, correction actions are generated and applied (up to 3 rounds).
- A frozen schema blueprint is emitted at the end of phase 2.

### Phase 3 — A-Box Population
- Each chunk is processed against the frozen schema (closed world assumption).
- Entities and relations are validated against approved classes/properties.
- Fuzzy deduplication merges equivalent entity IDs.
- Accepted triples are accumulated in KG state.

### Phase 4 — Export and Persistence
- Export complete state as JSON.
- Export ontology as Turtle (`.ttl`) with class/property constraints.
- Archive each run under incremented `run_N` folders.

### Phase 5 — Stability Reporting
- Build graph views from saved runs.
- Compute Jaccard similarity for classes, properties, and triples.
- Report stable vs divergent schema evolution between runs.

## Core Components

- `src/main.py` — orchestration entrypoint.
- `src/extraction.py` — LLM extraction/caching/retry.
- `src/state_tracker.py` — ontology and KG state handling.
- `src/owl_validator.py` — layered validation.
- `src/document_processor.py` — deterministic preprocessing/chunking.
- `src/prompts.py` — prompt templates for each phase.
- `src/schemas.py` — structured output contracts.
