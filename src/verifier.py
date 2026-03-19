import networkx as nx

class OntologyVerifier:
    def __init__(self):
        pass

    def calculate_jaccard_similarity(self, graph_a: nx.DiGraph, graph_b: nx.DiGraph) -> float:
        """
        Phase 5: Calculates the structural Jaccard similarity of two graphs based on their edge sets.
        sim(Ga, Gb) = |Edges(Ga) intersect Edges(Gb)| / |Edges(Ga) union Edges(Gb)|
        Edges are considered identical if (Source, Target, Property) match exactly.
        """
        edges_a = set(graph_a.edges(data="property_uri"))
        edges_b = set(graph_b.edges(data="property_uri"))

        if not edges_a and not edges_b:
            return 1.0 # Both empty -> 100% similar

        intersection = edges_a.intersection(edges_b)
        union = edges_a.union(edges_b)

        similarity = len(intersection) / len(union)
        return similarity

    def verify_ontology_stability(self, graph_a: nx.DiGraph, graph_b: nx.DiGraph, threshold: float = 0.7) -> bool:
        """
        Verifies if the sim(Ga, Gb) > tau (0.7).
        """
        score = self.calculate_jaccard_similarity(graph_a, graph_b)
        print(f"Mathematical Similarity Score (Jaccard on Edges): {score:.4f} (Threshold: {threshold})")
        
        if score >= threshold:
            print("[SUCCESS] The generated graphs are mathematically stable and practically deterministic.")
            return True
        else:
            print("[FAILED] The ontology generated highly divergent graphs. Determinism failed.")
            return False
