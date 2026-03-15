import os
import sys
import json
import time

# Ensure imports work from project root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config.settings import settings
from src.extraction import OntologyExtractor
from src.graph_builder import KnowledgeGraphBuilder
from src.state_tracker import KnowledgeStateTracker

def run_mini_test():
    print("--- Starting Mini-Extraction Proof ---")
    extractor = OntologyExtractor()
    graph_builder = KnowledgeGraphBuilder()
    state_tracker = KnowledgeStateTracker()
    
    # Mini Input Content
    mini_text = """
    Artificial Intelligence is a branch of Computer Science. 
    John McCarthy is a pioneer of Artificial Intelligence. 
    LISP is a programming language used for Artificial Intelligence.
    """
    
    # 1. Bootstrap Minimal Schema
    print("Step 1: Bootstrapping Schema...")
    bootstrap_result = extractor.bootstrap_ontology(mini_text)
    state_tracker.ingest_bootstrap(bootstrap_result)
    print(f"Schema Bootstrapped. Classes found: {[c.uri for c in bootstrap_result.classes]}")
    
    # Wait for rate limit
    print("Waiting 30 seconds to respect rate limits (Free Tier)...")
    time.sleep(30)
    
    # 2. Extract Triples
    print("Step 2: Extracting Triples...")
    approved_ontology = state_tracker.format_for_prompt()
    extraction_result = extractor.extract_triples(mini_text, approved_ontology)
    
    for entity in extraction_result.entities:
        graph_builder.add_entity(entity.id, entity.class_uri, label=entity.evidence_span)
    for relation in extraction_result.relations:
        graph_builder.add_relation(relation.source_id, relation.target_id, relation.property_uri)
        print(f"  * Extracted: {relation.source_id} -> {relation.property_uri} -> {relation.target_id}")

    # 3. Export Output
    output_dir = "data/output"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "mini_output.json")
    
    import networkx as nx
    data = nx.node_link_data(graph_builder.graph)
    with open(output_path, "w") as f:
        json.dump(data, f, indent=4)
        
    print(f"\n[SUCCESS] Mini-Ontology generated and saved to {output_path}")
    print(f"Nodes: {graph_builder.graph.number_of_nodes()} | Edges: {graph_builder.graph.number_of_edges()}")

if __name__ == "__main__":
    run_mini_test()
