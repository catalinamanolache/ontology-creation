import os
import sys
import json
import networkx as nx

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config.settings import settings
from src.extraction import OntologyExtractor
from src.graph_builder import KnowledgeGraphBuilder
from src.document_processor import DocumentProcessor
from src.state_tracker import KnowledgeStateTracker
from src.semantic_search import SemanticSearch
from src.verifier import OntologyVerifier


def select_bootstrap_chunks(all_chunks, k, distributed=True):
    if not all_chunks:
        return []
    k = max(1, min(k, len(all_chunks)))
    if not distributed or len(all_chunks) <= k:
        return all_chunks[:k]
    idxs = []
    for i in range(k):
        idx = round(i * (len(all_chunks) - 1) / (k - 1)) if k > 1 else 0
        idxs.append(idx)
    seen = set()
    unique_idxs = []
    for x in idxs:
        if x not in seen:
            seen.add(x)
            unique_idxs.append(x)
    return [all_chunks[i] for i in unique_idxs]


def generate_graph(processor, extractor, search_db, pdf_path, run_id="A"):
    print(f"\n{'='*20} Starting Run {run_id} {'='*20}")

    graph_builder = KnowledgeGraphBuilder()
    state_tracker = KnowledgeStateTracker()

    chunks = processor.process_pdf(pdf_path)
    print(f"Total chunks in document: {len(chunks)}")

    # Phase 3a: Bootstrap ontology from distributed chunks
    bootstrap_chunks = select_bootstrap_chunks(
        chunks,
        settings.BOOTSTRAP_CHUNKS,
        settings.BOOTSTRAP_DISTRIBUTED
    )
    print(f"Bootstrapping ontology from {len(bootstrap_chunks)} distributed chunks...")
    bootstrap_text = "\n".join(bootstrap_chunks)
    bootstrap_result = extractor.bootstrap_ontology(bootstrap_text)
    state_tracker.ingest_bootstrap(bootstrap_result)

    approved_ontology_str = state_tracker.format_for_prompt()

    # Phase 3b: Extract triples in batches to reduce API calls
    BATCH_SIZE = 5  # Reduce back to 5 so we don't hit max-output-tokens or break JSON parsers

    extraction_chunks = chunks if settings.EXTRACT_ALL_CHUNKS else chunks[:settings.EXTRACT_MAX_CHUNKS or len(chunks)]

    print(f"Extracting triples from {len(extraction_chunks)} chunks (batch size: {BATCH_SIZE})...")
    for batch_start in range(0, len(extraction_chunks), BATCH_SIZE):
        batch = extraction_chunks[batch_start:batch_start + BATCH_SIZE]
        batch_num = batch_start // BATCH_SIZE + 1
        total_batches = (len(extraction_chunks) + BATCH_SIZE - 1) // BATCH_SIZE
        print(f"  Processing batch {batch_num}/{total_batches} (chunks {batch_start+1}-{batch_start+len(batch)})...")
        combined_chunk = "\n\n---\n\n".join(batch)
        extraction_result = extractor.extract_triples(combined_chunk, approved_ontology_str)
        for entity in extraction_result.entities:
            graph_builder.add_entity(entity.id, entity.class_uri, label=entity.evidence_span)
        for relation in extraction_result.relations:
            graph_builder.add_relation(relation.source_id, relation.target_id, relation.property_uri)

    # Phase 4: Topological Refinement (runs AFTER all chunks are extracted)
    search_db.index_chunks(chunks)
    epsilon = 0.1
    iteration = 1

    print("Starting topological refinement loop...")
    MAX_REFINEMENT_ITERATIONS = 3  # or from settings

    while iteration <= MAX_REFINEMENT_ITERATIONS:
        c_main, c_small_list = graph_builder.get_components()
        if not c_small_list:
            print("Graph is fully connected. Refinement complete.")
            break

        print(f"  Iteration {iteration}: {len(c_small_list)} disconnected component(s) found. Consolidating into 1 batch call...")
        state_size_before = state_tracker.get_ontology_size()
        main_anchors = graph_builder.get_top_degree_nodes(c_main, top_k=10) # take more anchors for better coverage

        # Collect anchors from up to 20 disconnected subgraphs
        all_disconnected_anchors = []
        for c_j in c_small_list[:20]:
            all_disconnected_anchors.extend(graph_builder.get_top_degree_nodes(c_j, top_k=3))
            
        if not all_disconnected_anchors:
            break

        bridging_context = search_db.retrieve_context(main_anchors, all_disconnected_anchors, top_k=10)

        if bridging_context:
            bridge_result = extractor.bridge_subgraphs(
                text_context=bridging_context,
                main_anchors=main_anchors,
                disconnected_anchors=all_disconnected_anchors,
                approved_ontology=state_tracker.format_for_prompt()
            )

            for entity in bridge_result.entities:
                graph_builder.add_entity(entity.id, entity.class_uri, label=entity.evidence_span)
            for relation in bridge_result.relations:
                graph_builder.add_relation(relation.source_id, relation.target_id, relation.property_uri)

        state_size_after = state_tracker.get_ontology_size()
        delta_o = (state_size_after - state_size_before) / state_size_before if state_size_before > 0 else 0
        print(f"Refinement stopped after {iteration-1} iteration(s).")

        if delta_o <= epsilon and len(c_small_list) == len(graph_builder.get_components()[1]):
            print(f"  Convergence reached (delta_O={delta_o:.3f} <= eps={epsilon}).")
            break
        iteration += 1

    print(f"Run {run_id} completed.")
    return graph_builder.graph


def main():
    print("Starting Deterministic Ontology Generation Algorithm...")
    print(f"Loaded Settings. Chunk Size: {settings.CHUNK_SIZE} | Temp: {settings.TEMPERATURE}")

    processor = DocumentProcessor(chunk_size=settings.CHUNK_SIZE, chunk_overlap=settings.CHUNK_OVERLAP)
    extractor = OntologyExtractor()
    search_db = SemanticSearch()
    verifier = OntologyVerifier()

    pdf_path = os.path.join(settings.INPUT_DIR, "sample.pdf")
    if not os.path.exists(pdf_path):
        pdf_path = "sample.pdf"

    os.makedirs(settings.OUTPUT_DIR, exist_ok=True)

    print("\n" + "="*50)
    print("Phase 5: Live Ontology Extraction & Export")

    graph = generate_graph(processor, extractor, search_db, pdf_path, run_id="FINAL")

    output_path = os.path.join(settings.OUTPUT_DIR, "knowledge_graph.json")
    data = nx.node_link_data(graph)
    with open(output_path, "w") as f:
        json.dump(data, f, indent=4)

    print(f"\n[SUCCESS] Ontology successfully generated!")
    print(f"Nodes: {graph.number_of_nodes()} | Edges: {graph.number_of_edges()}")
    print(f"Output saved to: {output_path}")
    print("\nPipeline Implementation Complete.")

if __name__ == "__main__":
    main()