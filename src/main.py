"""
Ontology Pipeline — Decomposed Axiom-by-Axiom (AbA) Architecture

Entry point coordinating:
  Phase 1: Deterministic chunking of the document.
  Phase 2: Decomposed T-Box construction (CQ → Classes → Properties → Validation → Reasoner).
  Phase 3: Knowledge Graph Population (A-Box) using the frozen schema.
"""

import os
import sys
import json
import shutil
import argparse

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config.settings import settings
from src.document_processor import DocumentProcessor
from src.extraction import OntologyExtractor
from src.state_tracker import KnowledgeStateTracker
from src.rate_limiter import RateLimiter
from src.owl_validator import OWLValidator, format_errors_for_prompt


def _archive_run(state: KnowledgeStateTracker):
    """Save a timestamped copy of the run to data/runs/."""
    runs_dir = os.path.join("data", "runs")
    os.makedirs(runs_dir, exist_ok=True)

    next_idx = 1
    while True:
        run_folder = os.path.join(runs_dir, f"run_{next_idx}")
        try:
            os.makedirs(run_folder, exist_ok=False)
            break
        except FileExistsError:
            next_idx += 1

    blueprint_path = os.path.join(run_folder, "ontology_blueprint.json")
    state_path = os.path.join(run_folder, "ontology_state.json")
    ttl_path = os.path.join(run_folder, "ontology_final.ttl")

    pure_schema = {
        "approved_classes": state.approved_classes,
        "approved_properties": state.approved_properties,
        "canonical_terms": state.canonical_terms,
        "subclass_relations": state.subclass_relations,
    }
    with open(blueprint_path, 'w', encoding='utf-8') as f:
        json.dump(pure_schema, f, indent=2)

    state.save_state(state_path)
    state.export_turtle(ttl_path)
    print(f"\n[ARCHIVE] Saved complete results to: {run_folder}")


def _format_cqs_for_prompt(cq_result) -> str:
    """Format CQ result into a string for downstream prompts."""
    if not cq_result or not cq_result.questions:
        return "(no competency questions generated)"
    lines = []
    for i, q in enumerate(cq_result.questions, 1):
        lines.append(f"  {i}. [{q.scope}] {q.question}")
    return "\n".join(lines)


def main(fresh_run=False, blueprint_only=False):
    print("=" * 60)
    print("  DECOMPOSED AXIOM-BY-AXIOM (AbA) ONTOLOGY PIPELINE")
    print("=" * 60)

    if fresh_run:
        print("[!] Fresh run requested. Clearing old cache...")
        cache_dir = os.path.join("data", "cache")
        if os.path.exists(cache_dir):
            shutil.rmtree(cache_dir)
            print("  Cache cleared.")

    os.makedirs(settings.OUTPUT_DIR, exist_ok=True)

    processor = DocumentProcessor(
        chunk_size=settings.CHUNK_SIZE,
        chunk_overlap=settings.CHUNK_OVERLAP
    )
    rate_limiter = RateLimiter(
        max_rpm=settings.RATE_LIMIT_RPM,
        max_rpd=settings.RATE_LIMIT_RPD,
        min_delay=settings.MIN_DELAY_BETWEEN_REQUESTS
    )
    extractor = OntologyExtractor(rate_limiter)
    state = KnowledgeStateTracker()

    # ── Locate input document ──
    pdf_path = os.path.join(settings.INPUT_DIR, "sample.pdf")
    txt_path = os.path.join(settings.INPUT_DIR, "sample.txt")
    doc_path = pdf_path if os.path.exists(pdf_path) else txt_path
    if not os.path.exists(doc_path):
        print(f"Error: Insert a document into {settings.INPUT_DIR}/ (sample.pdf or sample.txt)")
        return

    # ================================================================
    # Phase 1: Document Chunking
    # ================================================================
    print(f"\n--- Phase 1: Processing Document ---")
    chunks = processor.process_pdf(doc_path) if doc_path.endswith('.pdf') else processor.process_txt(doc_path)

    if not chunks:
        print("No readable text found.")
        return

    print(f"  Total chunks generated: {len(chunks)}")

    chunks_dir = os.path.join("data", "chunks")
    os.makedirs(chunks_dir, exist_ok=True)
    for i, chunk in enumerate(chunks):
        with open(os.path.join(chunks_dir, f"chunk_{i:03d}.txt"), "w", encoding="utf-8") as f:
            f.write(chunk)

    # ================================================================
    # Phase 2: Decomposed T-Box Construction
    # ================================================================
    print("\n--- Phase 2: Decomposed T-Box Construction (AbA Pipeline) ---")
    print("  Sub-phases: CQ -> Classes+BFO -> Properties -> Validation -> Reasoner")

    BATCH_SIZE = 3
    batches = ["\n\n".join(chunks[i:i + BATCH_SIZE]) for i in range(0, len(chunks), BATCH_SIZE)]

    try:
        for i, batch_text in enumerate(batches):
            print(f"\n  === Batch {i+1}/{len(batches)} ===")

            # ── Phase 2.1: Competency Questions ──
            print(f"  [2.1] Generating Competency Questions...")
            cq_result = extractor.generate_competency_questions(batch_text)
            cqs_str = _format_cqs_for_prompt(cq_result)
            print(f"  [2.1] Generated {len(cq_result.questions)} CQs.")

            # ── Phase 2.2: Class Extraction with BFO anchoring ──
            print(f"  [2.2] Extracting Classes (BFO-anchored, Aristotelian definitions)...")
            if i == 0:
                class_result = extractor.extract_classes_initial(batch_text, cqs_str)
            else:
                class_result = extractor.extract_classes_extend(
                    text=batch_text,
                    existing_classes_str=state.format_classes_for_prompt(),
                    cqs_str=cqs_str,
                )
            state.ingest_classes(class_result)
            print(f"  [2.2] Total classes so far: {len(state.approved_classes)}")

            # ── Phase 2.4: Property Extraction ──
            print(f"  [2.4] Extracting Properties (domain/range constrained)...")
            if i == 0:
                prop_result = extractor.extract_properties_initial(
                    text=batch_text,
                    classes_str=state.format_classes_for_prompt(),
                    cqs_str=cqs_str,
                )
            else:
                prop_result = extractor.extract_properties_extend(
                    text=batch_text,
                    existing_properties_str=state.format_properties_for_prompt(),
                    classes_str=state.format_classes_for_prompt(),
                    cqs_str=cqs_str,
                )
            state.ingest_properties(prop_result)
            print(f"  [2.4] Total properties so far: {len(state.approved_properties)}")

    except Exception as e:
        print(f"\n[ERROR] T-Box construction failed: {e}")
        import traceback
        traceback.print_exc()
        return

    # ── Phase 2.5: Two-Way Hierarchy Validation ──
    print(f"\n  --- Phase 2.5: Two-Way Hierarchy Validation ---")
    subclass_rels = state.get_subclass_relations_for_validation()
    if subclass_rels:
        print(f"  Validating {len(subclass_rels)} subClassOf relations...")

        # First: Python-side cycle detection
        cycles = state.detect_hierarchy_cycles()
        if cycles:
            print(f"  [!] Detected {len(cycles)} hierarchy cycle(s)! Auto-removing...")
            for cycle in cycles:
                # Remove the last edge in the cycle
                child = cycle[-2]
                parent = cycle[-1]
                state.remove_subclass(child, parent)

        # Then: LLM-based Two-Way validation
        remaining_rels = state.get_subclass_relations_for_validation()
        if remaining_rels:
            try:
                validation_result = extractor.validate_hierarchy_batch(remaining_rels)
                removed_count = 0
                for v in validation_result.validations:
                    # Valid hierarchy: forward=True, reverse=False
                    if not v.forward_valid or v.reverse_valid:
                        state.remove_subclass(v.child_uri, v.parent_uri)
                        removed_count += 1
                        print(f"    [REJECTED] {v.child_uri} subClassOf {v.parent_uri}: {v.reasoning}")
                print(f"  [2.5] Validated. Removed {removed_count} invalid relations.")
            except Exception as e:
                print(f"  [WARN] Two-Way validation failed (non-fatal): {e}")
    else:
        print("  No subClassOf relations to validate.")

    # ── Phase 2.6: OWL Reasoner Validation + Self-Correction ──
    print(f"\n  --- Phase 2.6: OWL/RDF Validation (Neuro-Symbolic) ---")
    validator = OWLValidator()
    MAX_CORRECTION_ROUNDS = 3

    for round_num in range(1, MAX_CORRECTION_ROUNDS + 1):
        report = validator.validate(
            approved_classes=state.approved_classes,
            approved_properties=state.approved_properties,
            subclass_relations=state.subclass_relations,
        )

        if report.is_consistent:
            print(f"  [2.6] Ontology is logically consistent! (round {round_num})")
            break

        print(f"  [2.6] Round {round_num}: Found {len(report.errors)} error(s), {len(report.warnings)} warning(s).")
        for err in report.errors:
            print(f"    ERROR: {err}")
        for warn in report.warnings:
            print(f"    WARN:  {warn}")

        if round_num == MAX_CORRECTION_ROUNDS:
            print(f"  [2.6] Max correction rounds reached. Proceeding with current state.")
            break

        # Self-correction via LLM
        print(f"  [2.6] Requesting LLM self-correction...")
        try:
            error_log = format_errors_for_prompt(report)
            correction = extractor.self_correct(
                ontology_str=state.format_ontology_for_prompt(),
                error_log=error_log,
            )
            if correction.corrections:
                state.apply_corrections(correction)
                print(f"  [2.6] Applied {len(correction.corrections)} correction(s). Re-validating...")
            else:
                print(f"  [2.6] LLM returned no corrections. Proceeding.")
                break
        except Exception as e:
            print(f"  [WARN] Self-correction failed (non-fatal): {e}")
            break

    # Print warnings even if consistent
    if report.warnings:
        print(f"  Warnings ({len(report.warnings)}):")
        for w in report.warnings:
            print(f"    - {w}")

    # ── Save Blueprint ──
    blueprint_path = os.path.join(settings.OUTPUT_DIR, "ontology_blueprint.json")
    pure_schema = {
        "approved_classes": state.approved_classes,
        "approved_properties": state.approved_properties,
        "canonical_terms": state.canonical_terms,
        "subclass_relations": state.subclass_relations,
    }
    with open(blueprint_path, 'w', encoding='utf-8') as f:
        json.dump(pure_schema, f, indent=2)

    print(f"\n  Blueprint FROZEN!")
    print(f"  Classes: {len(state.approved_classes)}, Properties: {len(state.approved_properties)}")
    print(f"  Subclass relations: {len(state.subclass_relations)}")
    print(f"  Saved to: {blueprint_path}")

    if blueprint_only:
        print("\n[!] Blueprint-only flag enabled. Skipping KG population phase.")
        _archive_run(state)
        return

    # ================================================================
    # Phase 3: A-Box Population (UNCHANGED)
    # ================================================================
    print("\n--- Phase 3: Knowledge Graph Population (A-Box) ---")
    print("  Re-reading all chunks to extract entities using the FROZEN schema...")

    skipped = 0
    for i, chunk in enumerate(chunks):
        print(f"  [Chunk {i+1}/{len(chunks)}] Populating KG...")

        try:
            result = extractor.extract_triples(
                chunk_text=chunk,
                approved_ontology=state.format_ontology_for_prompt(),
                canonical_terms=state.format_canonical_terms_for_prompt(),
                existing_entities=state.format_entities_for_prompt(),
            )
        except Exception as e:
            print(f"\n  [ERROR] Crash during KG population at chunk {i}: {e}")
            print("  -> Saving current progress before exiting...")
            state.save_state(os.path.join(settings.OUTPUT_DIR, "ontology_state_partial.json"))
            state.export_turtle(os.path.join(settings.OUTPUT_DIR, "ontology_partial.ttl"))
            _archive_run(state)
            print("  Partial progress saved. Exiting gracefully.")
            return

        id_map = {}
        for entity in result.entities:
            if not state.validate_entity(entity):
                skipped += 1
                continue
            canonical_id = state.add_entity(entity)
            id_map[entity.id] = canonical_id

        for relation in result.relations:
            if not state.validate_relation(relation):
                skipped += 1
                continue
            state.add_relation(relation, id_map)

    print(f"\n--- KG Population Complete ---")
    print(f"  Entities: {len(state.entities)}, Relations: {len(state.relations)}, Skipped: {skipped}")

    # Save final state
    state.save_state(os.path.join(settings.OUTPUT_DIR, "ontology_state.json"))
    state.export_turtle(os.path.join(settings.OUTPUT_DIR, "ontology_final.ttl"))

    _archive_run(state)
    print("\n[DONE] Pipeline finished successfully.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Decomposed AbA Ontology Pipeline")
    parser.add_argument("--fresh", action="store_true", help="Clear cache and fully re-run LLM calls")
    parser.add_argument("--blueprint-only", action="store_true", help="Only run Phase 1 & 2 (schema generation)")
    parser.add_argument("--no-cache", action="store_true", help="Disable cache for this run")
    parser.add_argument(
        "--reset-rate-limit",
        action="store_true",
        dest="reset_rate_limit",
        help="Reset persisted rate limit state (daily count) and exit",
    )
    args = parser.parse_args()

    if args.no_cache:
        settings.USE_CACHE = False

    # If user requested a rate-limit reset, do it and exit immediately.
    if args.reset_rate_limit:
        rl = RateLimiter(
            max_rpm=settings.RATE_LIMIT_RPM,
            max_rpd=settings.RATE_LIMIT_RPD,
            min_delay=settings.MIN_DELAY_BETWEEN_REQUESTS,
        )
        rl.reset(remove_file=True)
        print("[MAIN] Rate limit state reset. Continuing with run.")

    main(fresh_run=args.fresh, blueprint_only=args.blueprint_only)
