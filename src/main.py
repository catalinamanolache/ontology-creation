import os
import sys
import re
import glob
import hashlib

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config.settings import settings
from src.extraction import OntologyExtractor
from src.document_processor import DocumentProcessor
from src.ontology_similarity import jaccard_similarity_from_ttl


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


def build_ontology(processor, extractor, pdf_path, run_id="A", chunk_limit=0, batch_size_override=0):
    print(f"\n{'='*20} Starting Run {run_id} {'='*20}")
    chunks = processor.process_pdf(pdf_path)
    print(f"Total chunks in document: {len(chunks)}")

    # Bootstrap ontology from representative chunks.
    bootstrap_chunks = select_bootstrap_chunks(
        chunks,
        settings.BOOTSTRAP_CHUNKS,
        settings.BOOTSTRAP_DISTRIBUTED
    )
    print(f"Bootstrapping ontology from {len(bootstrap_chunks)} distributed chunks...")
    bootstrap_text = "\n".join(bootstrap_chunks)
    ontology_ttl = extractor.bootstrap_ontology(bootstrap_text)

    # Ontology refinement over chunk batches (no graph construction).
    batch_size = batch_size_override if batch_size_override > 0 else 3
    target_chunks = chunks if settings.EXTRACT_ALL_CHUNKS else chunks[:settings.EXTRACT_MAX_CHUNKS or len(chunks)]
    if chunk_limit and chunk_limit > 0:
        target_chunks = target_chunks[:chunk_limit]
    total_batches = (len(target_chunks) + batch_size - 1) // batch_size if target_chunks else 0
    print(f"Refining ontology over {len(target_chunks)} chunks (batch size: {batch_size})...")

    for batch_start in range(0, len(target_chunks), batch_size):
        batch = target_chunks[batch_start:batch_start + batch_size]
        batch_num = batch_start // batch_size + 1
        print(f"  Refinement batch {batch_num}/{total_batches} (chunks {batch_start+1}-{batch_start+len(batch)})...")
        combined_chunk = "\n\n---\n\n".join(batch)
        ontology_ttl = extractor.refine_ontology(ontology_ttl, combined_chunk)

    print(f"Run {run_id} completed.")
    return ontology_ttl


def _list_numbered_ontologies(folder_path):
    pattern = re.compile(r"^ontology_(\d{4})\.ttl$")
    entries = []
    for name in os.listdir(folder_path):
        match = pattern.match(name)
        if match:
            entries.append((int(match.group(1)), name))
    return sorted(entries, key=lambda x: x[0])


def save_numbered_ontology(ontology_ttl, output_dir):
    runs_dir = os.path.join(output_dir, "ontologies")
    os.makedirs(runs_dir, exist_ok=True)

    numbered = _list_numbered_ontologies(runs_dir)
    next_index = 1 if not numbered else numbered[-1][0] + 1
    file_name = f"ontology_{next_index:04d}.ttl"
    run_path = os.path.join(runs_dir, file_name)

    with open(run_path, "w", encoding="utf-8") as f:
        f.write(ontology_ttl)

    prev_path = None
    if numbered:
        prev_path = os.path.join(runs_dir, numbered[-1][1])

    return run_path, prev_path


def compute_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def clear_ontology_cache():
    cache_dir = os.path.join("data", "cache")
    if not os.path.isdir(cache_dir):
        return
    for path in glob.glob(os.path.join(cache_dir, "bootstrap_ontology_*.txt")):
        os.remove(path)
    for path in glob.glob(os.path.join(cache_dir, "refine_ontology_*.txt")):
        os.remove(path)


def main():
    print("Starting Ontology-Only Generation Pipeline (Hugging Face)...")
    print(
        f"Loaded Settings. Chunk Size: {settings.CHUNK_SIZE} | Temp: {settings.TEMPERATURE} | "
        f"Use Cache: {settings.USE_CACHE} | Determinism Runs: {settings.DET_TEST_RUNS}"
    )

    processor = DocumentProcessor(chunk_size=settings.CHUNK_SIZE, chunk_overlap=settings.CHUNK_OVERLAP)
    extractor = OntologyExtractor()

    pdf_path = os.path.join(settings.INPUT_DIR, "sample.pdf")
    if not os.path.exists(pdf_path):
        pdf_path = "sample.pdf"

    os.makedirs(settings.OUTPUT_DIR, exist_ok=True)

    print("\n" + "="*50)
    print("Ontology Bootstrapping + Refinement")

    runs = max(1, settings.DET_TEST_RUNS)
    run_hashes = []

    fast_mode_active = runs > 1 and settings.DET_FAST_MODE
    chunk_limit = settings.DET_CHUNK_LIMIT if fast_mode_active else 0
    batch_size_override = settings.DET_BATCH_SIZE if fast_mode_active else 0
    if fast_mode_active:
        print(
            f"[INFO] Determinism fast mode active: chunk_limit={chunk_limit}, "
            f"batch_size={batch_size_override}"
        )

    for idx in range(runs):
        run_label = f"FINAL_{idx+1}" if runs > 1 else "FINAL"
        if settings.CLEAR_CACHE_EACH_RUN:
            clear_ontology_cache()
            print("[INFO] Cleared ontology cache before run.")

        ontology_ttl = build_ontology(
            processor,
            extractor,
            pdf_path,
            run_id=run_label,
            chunk_limit=chunk_limit,
            batch_size_override=batch_size_override,
        )
        run_sha = compute_sha256(ontology_ttl)
        run_hashes.append(run_sha)

        run_path, prev_path = save_numbered_ontology(ontology_ttl, settings.OUTPUT_DIR)

        output_path = os.path.join(settings.OUTPUT_DIR, settings.ONTOLOGY_OUTPUT_FILE)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(ontology_ttl)

        print("\n[SUCCESS] Ontology successfully generated!")
        print(f"Numbered ontology saved to: {run_path}")
        print(f"Output saved to: {output_path}")
        print(f"SHA256: {run_sha}")

        if prev_path:
            with open(prev_path, "r", encoding="utf-8") as f:
                prev_ttl = f.read()
            score = jaccard_similarity_from_ttl(prev_ttl, ontology_ttl)
            prev_sha = compute_sha256(prev_ttl)
            print(f"Similarity vs previous run ({os.path.basename(prev_path)}): {score:.4f}")
            print(f"Exact hash match vs previous run: {run_sha == prev_sha}")
        else:
            print("Similarity vs previous run: N/A (this is the first numbered ontology run)")

    if runs > 1:
        unique_hashes = len(set(run_hashes))
        identical = unique_hashes == 1
        print(f"\nDeterminism summary: all runs identical by hash = {identical}")
        print(f"Unique ontology hashes: {unique_hashes}")

    print("\nPipeline Implementation Complete.")

if __name__ == "__main__":
    main()