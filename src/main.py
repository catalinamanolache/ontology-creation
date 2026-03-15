import os
import sys

# Add the project root to sys.path so we can import 'config' and 'src'
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config.settings import settings
from src.extraction import OntologyExtractor
from src.graph_builder import KnowledgeGraphBuilder
from src.document_processor import DocumentProcessor
from src.state_tracker import KnowledgeStateTracker
from src.semantic_search import SemanticSearch

import os
import sys

# Add the project root to sys.path so we can import 'config' and 'src'
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config.settings import settings
from src.extraction import OntologyExtractor
from src.graph_builder import KnowledgeGraphBuilder
from src.document_processor import DocumentProcessor
from src.state_tracker import KnowledgeStateTracker
from src.semantic_search import SemanticSearch
from src.verifier import OntologyVerifier

def generate_graph(processor, extractor, search_db, pdf_path, run_id="A"):
    """
    Encapsulates the full graph generation pipeline (Phases 1-4) so it can be 
    run multiple times for the Dual Instantiation A/B test.
    """
    print(f"\n{'='*20} Starting Run {run_id} {'='*20}")
    
    graph_builder = KnowledgeGraphBuilder()
    state_tracker = KnowledgeStateTracker()
    
    chunks = processor.process_pdf(pdf_path)
    
    # Bootstrap from the first few chunks
    bootstrap_text = "\n".join(chunks[:3])
    bootstrap_result = extractor.bootstrap_ontology(bootstrap_text)
    state_tracker.ingest_bootstrap(bootstrap_result)
    
    approved_ontology_str = state_tracker.format_for_prompt()
    
    # Extraction
    for i, chunk in enumerate(chunks[:1]): # Limited to 1 chunk to save credits
        extraction_result = extractor.extract_triples(chunk, approved_ontology_str)
        for entity in extraction_result.entities:
            graph_builder.add_entity(entity.id, entity.class_uri, label=entity.evidence_span)
        for relation in extraction_result.relations:
            graph_builder.add_relation(relation.source_id, relation.target_id, relation.property_uri)
            
    # Topological Refinement
    search_db.index_chunks([chunk for chunk in chunks])
    epsilon = 0.1
    iteration = 1
    
    while True:
        c_main, c_small_list = graph_builder.get_components()
        if not c_small_list:
            break
            
        state_size_before = state_tracker.get_ontology_size()
        main_anchors = graph_builder.get_top_degree_nodes(c_main, top_k=5)
        
        for idx, c_j in enumerate(c_small_list):
            disconnected_anchors = graph_builder.get_top_degree_nodes(c_j, top_k=3)
            bridging_context = search_db.retrieve_context(main_anchors, disconnected_anchors)
            
            if not bridging_context: continue
                
            bridge_result = extractor.bridge_subgraphs(
                text_context=bridging_context,
                main_anchors=main_anchors,
                disconnected_anchors=disconnected_anchors,
                approved_ontology=state_tracker.format_for_prompt()
            )
            
            for entity in bridge_result.entities:
                graph_builder.add_entity(entity.id, entity.class_uri, label=entity.evidence_span)
            for relation in bridge_result.relations:
                graph_builder.add_relation(relation.source_id, relation.target_id, relation.property_uri)

        state_size_after = state_tracker.get_ontology_size()
        delta_o = (state_size_after - state_size_before) / state_size_before if state_size_before > 0 else 0
        
        if delta_o <= epsilon and len(c_small_list) == len(graph_builder.get_components()[1]):
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
    if not os.path.exists(pdf_path): pdf_path = "sample.pdf"

    # Create output directory if it doesn't exist
    os.makedirs(settings.OUTPUT_DIR, exist_ok=True)

    # --- Phase 5: Live Extraction & Export ---
    print("\n" + "="*50)
    print("Phase 5: Live Ontology Extraction & Export")
    
    # Run a single generation to save credits/quota
    graph = generate_graph(processor, extractor, search_db, pdf_path, run_id="FINAL")
    
    # Export to JSON
    output_path = os.path.join(settings.OUTPUT_DIR, "knowledge_graph.json")
    data = nx.node_link_data(graph)
    with open(output_path, "w") as f:
        json.dump(data, f, indent=4)
        
    print(f"\n[SUCCESS] Ontology successfully generated!")
    print(f"Nodes: {graph.number_of_nodes()} | Edges: {graph.number_of_edges()}")
    print(f"Output saved to: {output_path}")
    
    # Dual Instantiation logic (Skipped for now to respect your Free Tier quota)
    print("\nNote: Dual Instantiation A/B verification logic is implemented but skipped to save your free credits.")
    print("Pipeline Implementation Complete.")

if __name__ == "__main__":
    main()
