import os
import json
import networkx as nx
from pyvis.network import Network

def build_graph_from_state(state_file):
    with open(state_file, 'r', encoding='utf-8') as f:
        state = json.load(f)
    
    G = nx.DiGraph()
    for inst in state.get('instances', []):
        G.add_node(inst['id'], label=inst['id'], title=f"Class: {inst.get('class_uri', 'Unknown')}")
        
    for rel in state.get('relations', []):
        prop_uri = rel.get('property_uri', 'Unknown')
        G.add_edge(rel['source_id'], rel['target_id'], title=prop_uri, label=prop_uri.split(':')[-1])
        
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

def natural_sort_key(s):
    import re
    return [int(text) if text.isdigit() else text.lower() for text in re.split('([0-9]+)', s)]

def extract_states_from_job(job_dir):
    valid_runs = []
    
    # Recursively find any folder containing our target .json files
    # BUT filter out "output" folders to strictly compare "runs"
    for root, dirs, files in os.walk(job_dir):
        if "output" in root.split(os.sep):
            continue # We ignore final output folders for sequential comparison
        if "ontology_state.json" in files or "ontology_blueprint.json" in files:
            valid_runs.append(root)

    valid_runs.sort(key=natural_sort_key)
    states = []

    for run_path in valid_runs:
        rel_path = os.path.relpath(run_path, job_dir)
        run_label = rel_path if rel_path != '.' else 'root'

        state_file = os.path.join(run_path, "ontology_state.json")
        blueprint_file = os.path.join(run_path, "ontology_blueprint.json")

        if os.path.exists(state_file):
            G, state = build_graph_from_state(state_file)
            output_html = os.path.join(run_path, "knowledge_graph.html")
            try:
                generate_kg_html(G, output_html)
            except Exception:
                pass
            states.append((run_label, state))
        elif os.path.exists(blueprint_file):
            with open(blueprint_file, 'r', encoding='utf-8') as f:
                state = json.load(f)
            states.append((run_label, state))

    return states

def compare_states(nameA, stateA, nameB, stateB):
    classesA = set(stateA.get('approved_classes', {}).keys())
    classesB = set(stateB.get('approved_classes', {}).keys())
    propsA = set(stateA.get('approved_properties', {}).keys())
    propsB = set(stateB.get('approved_properties', {}).keys())
    
    edgesA = set((r.get('source_id'), r.get('property_uri'), r.get('target_id')) for r in stateA.get('relations', []))
    edgesB = set((r.get('source_id'), r.get('property_uri'), r.get('target_id')) for r in stateB.get('relations', []))

    sim_classes = jaccard(classesA, classesB)
    sim_props = jaccard(propsA, propsB)
    
    print(f"\n   👉 {nameA.replace(chr(92), '/')} vs {nameB.replace(chr(92), '/')}:")
    print(f"      • Classes    : {len(classesA)} -> {len(classesB)} (Sim: {sim_classes*100:.1f}%)")
    print(f"      • Properties : {len(propsA)} -> {len(propsB)} (Sim: {sim_props*100:.1f}%)")
    
    if len(edgesA) > 0 or len(edgesB) > 0:
        sim_edges = jaccard(edgesA, edgesB)
        print(f"      • Triples    : {len(edgesA)} -> {len(edgesB)} (Sim: {sim_edges*100:.1f}%)")
        if sim_edges >= 0.7 and sim_classes >= 0.7:
            print("      ✅ STABLE evolution.")
        else:
            print("      ⚠️ DIVERGENT schema.")
    else:
        if sim_props >= 0.7 and sim_classes >= 0.7:
            print("      ✅ STABLE blueprint evolution.")
        else:
            print("      ⚠️ DIVERGENT blueprint.")

def run_fep_pipeline_evaluation():
    base_dir = "data/fep_results"
    if not os.path.exists(base_dir):
        print(f"Directory {base_dir} not found. Ensure results were downloaded from FEP.")
        return

    print(f"\n{'='*70}")
    print(f"📊 FEP ONTOLOGY PIPELINE EVALUATION REPORT")
    print(f"{'='*70}")

    jobs = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d)) and d.startswith("job_")]
    jobs.sort(key=natural_sort_key)

    job_states = {}

    # Phase 1: Internal job comparison
    for job in jobs:
        job_dir = os.path.join(base_dir, job)
        states = extract_states_from_job(job_dir)
        
        if not states:
            continue
            
        job_states[job] = states
        
        print(f"\n{'-'*60}")
        print(f"🔹 INTERNAL RUNS COMPARISON: {job} ({len(states)} valid runs found)")
        print(f"{'-'*60}")

        if len(states) >= 2:
            for i in range(len(states) - 1):
                runA_name, stateA = states[i]
                runB_name, stateB = states[i+1]
                compare_states(runA_name, stateA, runB_name, stateB)
        else:
            print("  [!] Only one run found internally. Skipping internal comparison.")

    # Phase 2: Cross-job sequential comparison
    print(f"\n\n{'='*70}")
    print(f"🔗 CROSS-JOB CHAIN COMPARISON")
    print(f"   (Comparing Last Run of Job [N] with First Run of Job [N+1])")
    print(f"{'='*70}")

    valid_jobs = list(job_states.keys())
    if len(valid_jobs) < 2:
        print("  [!] Not enough valid jobs with runs to perform cross-job comparison.")
        return

    for i in range(len(valid_jobs) - 1):
        jobA_name = valid_jobs[i]
        jobB_name = valid_jobs[i+1]
        
        statesA = job_states[jobA_name]
        statesB = job_states[jobB_name]
        
        # Last run of Job A
        last_run_A_path, last_state_A = statesA[-1]
        # First run of Job B
        first_run_B_path, first_state_B = statesB[0]

        labelA = f"{jobA_name}/{last_run_A_path}"
        labelB = f"{jobB_name}/{first_run_B_path}"
        
        compare_states(labelA, last_state_A, labelB, first_state_B)

    print(f"\n{'='*70}")
    print(f"✅ Evaluation fully completed.")
    print(f"{'='*70}")

if __name__ == "__main__":
    run_fep_pipeline_evaluation()
