import os
import json
import networkx as nx
from pyvis.network import Network

def build_graph_from_state(state_file):
    with open(state_file, 'r', encoding='utf-8') as f:
        state = json.load(f)
    
    G = nx.DiGraph()
    for inst in state.get('instances', []):
        G.add_node(inst['id'], label=inst['id'], title=f"Class: {inst['class_uri']}")
        
    for rel in state.get('relations', []):
        G.add_edge(rel['source_id'], rel['target_id'], title=rel['property_uri'], label=rel['property_uri'].split(':')[-1])
        
    return G, state

def jaccard(set1, set2):
    if not set1 and not set2:
        return 1.0
    return len(set1.intersection(set2)) / len(set1.union(set2))

def generate_kg_html(G, output_file):
    net = Network(height="800px", width="100%", bgcolor="#222222", font_color="white", directed=True)
    net.from_nx(G)
    net.set_options("""
    var options = {
      "physics": {
        "forceAtlas2Based": { "gravitationalConstant": -50, "centralGravity": 0.01, "springLength": 100, "springConstant": 0.08 },
        "minVelocity": 0.75, "solver": "forceAtlas2Based", "timestep": 0.4
      },
      "edges": { "color": {"inherit": true}, "smooth": {"type": "dynamic", "roundness": 0.5} }
    }
    """)
    net.save_graph(output_file)
    print(f"  [+] Saved Interactive KG to: {output_file}")

def run_final_pipeline_evaluation():
    runs_dir = "data/runs"
    if not os.path.exists(runs_dir):
        print("No runs directory found.")
        return

    runs = [d for d in os.listdir(runs_dir) if os.path.isdir(os.path.join(runs_dir, d)) and d.startswith("run_")]
    
    # Numerical sort: run_1, run_2, ..., run_10
    runs.sort(key=lambda x: int(x.split('_')[1]) if x.split('_')[1].isdigit() else 0)

    print(f"\n{'='*60}")
    print(f"📊 ONTOLOGY PIPELINE EVALUATION REPORT")
    print(f"{'='*60}")

    states = []

    for run in runs:
        state_file = os.path.join(runs_dir, run, "ontology_state.json")
        if os.path.exists(state_file):
            print(f"\n[Run {run}] Processing Knowledge Graph...")
            G, state = build_graph_from_state(state_file)
            
            output_html = os.path.join(runs_dir, run, "knowledge_graph.html")
            generate_kg_html(G, output_html)
            states.append((run, state))

    if len(states) < 2:
        print("\n[!] Not enough runs to compare.")
        return

    print(f"\n{'='*60}")
    print(f"🔄 SEQUENTIAL COMPARISON (Run-to-Run Evolution)")
    print(f"{'='*60}")

    for i in range(len(states) - 1):
        runA_name, stateA = states[i]
        runB_name, stateB = states[i+1]

        classesA = set(stateA.get('approved_classes', {}).keys())
        classesB = set(stateB.get('approved_classes', {}).keys())
        propsA = set(stateA.get('approved_properties', {}).keys())
        propsB = set(stateB.get('approved_properties', {}).keys())
        edgesA = set((r['source_id'], r['property_uri'], r['target_id']) for r in stateA.get('relations', []))
        edgesB = set((r['source_id'], r['property_uri'], r['target_id']) for r in stateB.get('relations', []))

        sim_classes = jaccard(classesA, classesB)
        sim_props = jaccard(propsA, propsB)
        sim_edges = jaccard(edgesA, edgesB)

        print(f"\n👉 {runA_name} vs {runB_name}:")
        print(f"   • Classes    : {len(classesA)} -> {len(classesB)} (Sim: {sim_classes*100:.1f}%)")
        print(f"   • Properties : {len(propsA)} -> {len(propsB)} (Sim: {sim_props*100:.1f}%)")
        print(f"   • Triples    : {len(edgesA)} -> {len(edgesB)} (Sim: {sim_edges*100:.1f}%)")
        
        if sim_edges >= 0.7 and sim_classes >= 0.7:
            print("   ✅ STABLE evolution.")
        else:
            print("   ⚠️ EVOLVING/DIVERGENT schema.")

    print(f"\n{'='*60}")
    print(f"✅ Report generation complete.")
    print(f"{'='*60}")

if __name__ == "__main__":
    run_final_pipeline_evaluation()
