import networkx as nx
from typing import List

class KnowledgeGraphBuilder:
    def __init__(self):
        self.graph = nx.DiGraph()

    def add_entity(self, entity_id: str, class_uri: str, label: str = ""):
        self.graph.add_node(entity_id, type="entity", class_uri=class_uri, label=label)

    def add_relation(self, source_id: str, target_id: str, property_uri: str):
        self.graph.add_edge(source_id, target_id, property_uri=property_uri)

    def get_components(self):
        """
        Phase 4: Separates the graph into C_main (largest) and C_small (list of disconnected).
        Returns: (main_component_nodes, list_of_small_component_nodes)
        """
        # We use weakly connected components since relations are directed
        components = sorted(nx.weakly_connected_components(self.graph), key=len, reverse=True)
        if not components:
            return set(), []
            
        c_main = components[0]
        c_small = components[1:]
        
        return c_main, c_small

    def get_top_degree_nodes(self, nodes: set, top_k: int = 5) -> List[str]:
        """
        Phase 4: Calculates degree centrality within the specific subgraph and returns the top K anchors.
        """
        subgraph = self.graph.subgraph(nodes)
        # Sort nodes by descending node degree (total edges in/out)
        sorted_nodes = sorted(subgraph.degree(), key=lambda x: x[1], reverse=True)
        return [n[0] for n in sorted_nodes[:top_k]]

    def print_graph_stats(self):
        components = list(nx.weakly_connected_components(self.graph))
        print(f"Graph Nodes: {self.graph.number_of_nodes()}")
        print(f"Graph Edges: {self.graph.number_of_edges()}")
        print(f"Connected Components: {len(components)}")

