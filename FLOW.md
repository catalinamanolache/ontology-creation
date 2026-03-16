# Ontology Creation: Detailed Project Flow

This document provides a step-by-step breakdown of the deterministic ontology generation pipeline. The core orchestration happens in `src/main.py` which ties together document processing, LLM-based entity-relationship extraction, and graph analysis.

## High-Level Orchestration: `main()`
Execution begins in the `main()` function of `src/main.py`.
1. **Component Initialization**:
   - `DocumentProcessor`: Handles reading and splitting the source PDF file into manageable text chunks.
   - `OntologyExtractor`: Contains the LLM chains (via `ChatGoogleGenerativeAI`) to extract schemas and triples.
   - `SemanticSearch`: Acts as a vector database (ChromaDB) to map text chunks to embeddings for later targeted context retrieval.
   - `OntologyVerifier`: Formats/validates outputs.
2. **File Preparation**: Identifies the target document (e.g., `sample.pdf`) and ensures the output directory (`data/output`) exists.
3. **Trigger Core Pipeline**: Calls `generate_graph(processor, extractor, search_db, pdf_path, run_id="FINAL")`.

---

## Core Pipeline: `generate_graph()`

The `generate_graph` function encapsulates the entire iterative extraction, bootstrapping, and topological refinement process.

### Step 1: Processing the Document
- **`processor.process_pdf(pdf_path)`**: 
  - Reads the PDF.
  - Splits the text into semantic chunks governed by `CHUNK_SIZE` and `CHUNK_OVERLAP` defined in `config/settings.py`.
  - *Returns:* A list of text chunks.

### Step 2: Phase 3a - Ontology Bootstrapping
Before extracting every piece of data, the system builds an overarching foundational schema.
- **`select_bootstrap_chunks(chunks, k, distributed)`**: Extracts a representative subset of the chunks distributed uniformly across the entire document length.
- **`extractor.bootstrap_ontology(bootstrap_text)`**: Sends the combined representative text to the LLM. 
  - The LLM identifies the primary Entity Classes and Property types present in the document.
- **`state_tracker.ingest_bootstrap(bootstrap_result)`**: Stores these initial foundational boundaries in the state tracker.
- **`state_tracker.format_for_prompt()`**: Condenses the extracted ontology into a strict set of rules/formats for all upcoming extraction phases to follow.

### Step 3: Phase 3b - Batch Triple Extraction
The pipeline iterates through the rest of the chunks to extract knowledge using the approved bounds.
- **Batching**: Chunks are grouped (e.g., 3 per batch) to limit LLM API calls and allow for better context.
- **`extractor.extract_triples(combined_chunk, approved_ontology_str)`**: Sends each batch to the LLM to extract Entities and Relations. 
  - The prompt is strictly constrained by the previously populated `approved_ontology_str`, meaning the LLM should avoid inventing unsanctioned classes/relations unless necessary.
- **Graph Assembly (`graph_builder`)**:
  - `graph_builder.add_entity()`: Incorporates the returned nodes with properties (ID, URI, Labels).
  - `graph_builder.add_relation()`: Creates directed edges between the parsed nodes.

### Step 4: Phase 4 - Topological Refinement (Iterative Bridging)
Often, extracting chunks in isolation leads to disjointed knowledge subgraphs (i.e. "islands" of knowledge that don't connect).
- **`search_db.index_chunks(chunks)`**: The original raw text chunks are added to the ChromaDB Vector database to enable semantic search capabilities.
- **Refinement Loop (`while iteration <= MAX_REFINEMENT_ITERATIONS`)**:
  1. **Connectivity Check**: `graph_builder.get_components()` analyzes the knowledge graph mathematically and identifies the primary connected component (`c_main`) and smaller disjoint components (`c_small_list`). If there are no small components (everything is joined), the loop breaks.
  2. **Anchor Identification**:
     - `graph_builder.get_top_degree_nodes(c_main, top_k=5)` gets the most central nodes from the main graph.
     - For each disconnected sub-graph, `graph_builder.get_top_degree_nodes(c_j, top_k=3)` gets its central anchors.
  3. **Context Bridging (`search_db.retrieve_context`)**: Uses semantic search to locate any original source text chunks that simultaneously discuss elements from both the main graph anchors AND the disconnected subgraph anchors.
  4. **Targeted Extraction (`extractor.bridge_subgraphs`)**: Takes the highly targeted, newly found "bridging context" text and asks the LLM to explicitly seek out relationships linking the main nodes to the disjoint nodes.
  5. **Graph Update**: Incorporates newly discovered linking relation edges into the `graph_builder`.
  6. **Convergence Measurement**: Calculates `delta_o` (growth percentage of the ontology schema over this iteration). The loop stops if the graph size stabilises (`delta_o <= epsilon`).

### Step 5: Phase 5 - Export and Output
- Back in `main()`, the finalized NetworkX graph object is converted into node-link schema format using `nx.node_link_data(graph)`.
- It saves everything to `data/output/knowledge_graph.json` so it can be visualised or consumed by Downstream applications.