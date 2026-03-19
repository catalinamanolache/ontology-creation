"""
KnowledgeStateTracker — Maintains cumulative ontology state across all pipeline phases.

Extended for the Decomposed AbA Pipeline:
  - BFO-anchored class storage (bfo_parent, aristotelian_definition, subclass_of)
  - Subclass relation tracking with cycle detection
  - Domain/range consistency checks
  - Self-correction application from OWL validator feedback
  - Fuzzy deduplication of entity IDs
  - Export to Turtle (.ttl) with BFO annotations
  - Save/load state to/from JSON
"""

import json
import os
import re
from typing import Dict, List, Optional, Set, Tuple

from src.schemas import (
    ClassExtractionResult,
    PropertyExtractionResult,
    CorrectionResult,
    ValidationReport,
    Entity,
    Relation,
    SeedOntologyResult,
)


class KnowledgeStateTracker:
    def __init__(self):
        # ── T-Box (schema-level) ──
        # uri -> {comment, bfo_parent, aristotelian_definition, subclass_of}
        self.approved_classes: Dict[str, dict] = {}
        # uri -> {domain, range, comment}
        self.approved_properties: Dict[str, dict] = {}
        # abbreviation -> canonical URI
        self.canonical_terms: Dict[str, str] = {}
        # child_uri -> parent_uri (domain subclass relations, not BFO)
        self.subclass_relations: Dict[str, str] = {}

        # ── A-Box (instance-level) ──
        self.entities: Dict[str, dict] = {}    # id -> {class_uri, evidence_span}
        self.relations: List[dict] = []        # [{source_id, property_uri, target_id, evidence_span}]

        # ── Tracking ──
        self._entity_normalized_index: Dict[str, str] = {}  # normalized_id -> canonical_id

    # ================================================================
    # Ingestion — Decomposed Pipeline (new)
    # ================================================================

    def ingest_classes(self, result: ClassExtractionResult):
        """Ingest classes from the decomposed class extraction phase."""
        for cls in result.classes:
            if cls.uri not in self.approved_classes:
                self.approved_classes[cls.uri] = {
                    "comment": cls.comment,
                    "bfo_parent": cls.bfo_parent,
                    "aristotelian_definition": cls.aristotelian_definition,
                    "subclass_of": cls.subclass_of,
                }
                # Track subclass relation separately for validation
                if cls.subclass_of:
                    self.subclass_relations[cls.uri] = cls.subclass_of

        for abbr, canonical in result.canonical_terms.items():
            if abbr not in self.canonical_terms:
                self.canonical_terms[abbr] = canonical

    def ingest_properties(self, result: PropertyExtractionResult):
        """Ingest properties from the decomposed property extraction phase."""
        class_uris = set(self.approved_classes.keys())
        xsd_types = {
            "xsd:string", "xsd:integer", "xsd:float", "xsd:boolean",
            "xsd:date", "xsd:dateTime", "xsd:decimal", "xsd:double",
        }
        allowed_special = {"owl:Thing", "rdfs:Literal", "ex:Thing"}

        for prop in result.properties:
            if prop.uri in self.approved_properties:
                continue

            # Validate domain references a known class
            if prop.domain not in class_uris and prop.domain not in allowed_special:
                print(f"  [WARN] Property '{prop.uri}' skipped: domain '{prop.domain}' not in approved classes.")
                continue

            # Validate range references a known class or XSD type
            if (prop.range not in class_uris
                    and prop.range not in xsd_types
                    and prop.range not in allowed_special):
                print(f"  [WARN] Property '{prop.uri}' skipped: range '{prop.range}' not in approved classes or XSD.")
                continue

            self.approved_properties[prop.uri] = {
                "domain": prop.domain,
                "range": prop.range,
                "comment": prop.comment,
            }

    # ================================================================
    # Ingestion — Legacy (backward-compatible)
    # ================================================================

    def ingest_seed(self, seed_result: SeedOntologyResult):
        """Ingest a legacy SeedOntologyResult (Phase 2 monolithic)."""
        for cls in seed_result.classes:
            if cls.uri not in self.approved_classes:
                self.approved_classes[cls.uri] = {
                    "comment": cls.comment,
                    "bfo_parent": "bfo:Entity",
                    "aristotelian_definition": f"A {cls.uri.replace('ex:', '')} is an Entity.",
                    "subclass_of": None,
                }

        for prop in seed_result.properties:
            if prop.uri not in self.approved_properties:
                self.approved_properties[prop.uri] = {
                    "domain": prop.domain,
                    "range": prop.range,
                    "comment": prop.comment,
                }

        for abbr, canonical in seed_result.canonical_terms.items():
            if abbr not in self.canonical_terms:
                self.canonical_terms[abbr] = canonical

    def ingest_bootstrap(self, result):
        """Legacy alias."""
        self.ingest_seed(result)

    # ================================================================
    # Hierarchy Validation & Correction
    # ================================================================

    def get_subclass_relations_for_validation(self) -> List[dict]:
        """Return subclass relations formatted for the Two-Way validation prompt."""
        relations = []
        for child, parent in self.subclass_relations.items():
            relations.append({"child": child, "parent": parent})
        return relations

    def remove_subclass(self, child_uri: str, parent_uri: str):
        """Remove a subclass relation that failed validation."""
        if self.subclass_relations.get(child_uri) == parent_uri:
            del self.subclass_relations[child_uri]
            # Also update the class record
            if child_uri in self.approved_classes:
                self.approved_classes[child_uri]["subclass_of"] = None
            print(f"  [HIERARCHY] Removed invalid subClassOf: {child_uri} -> {parent_uri}")

    def detect_hierarchy_cycles(self) -> List[List[str]]:
        """Detect cycles in the subClassOf graph. Returns list of cycles found."""
        WHITE, GRAY, BLACK = 0, 1, 2
        all_nodes = set(self.subclass_relations.keys()) | set(self.subclass_relations.values())
        color = {n: WHITE for n in all_nodes}
        cycles = []

        def dfs(node: str, path: List[str]):
            color[node] = GRAY
            path.append(node)
            parent = self.subclass_relations.get(node)
            if parent:
                if color.get(parent, WHITE) == GRAY:
                    cycle_start = path.index(parent)
                    cycles.append(path[cycle_start:] + [parent])
                elif color.get(parent, WHITE) == WHITE:
                    dfs(parent, path)
            path.pop()
            color[node] = BLACK

        for node in list(all_nodes):
            if color[node] == WHITE:
                dfs(node, [])

        return cycles

    def apply_corrections(self, correction_result: CorrectionResult):
        """Apply corrections from the self-correction LLM response."""
        for corr in correction_result.corrections:
            action = corr.action
            target = corr.target_uri

            if action == "remove_class":
                if target in self.approved_classes:
                    del self.approved_classes[target]
                    print(f"  [CORRECTION] Removed class: {target} — {corr.reasoning}")
                # Also remove any properties that reference this class
                props_to_remove = []
                for prop_uri, info in self.approved_properties.items():
                    if info["domain"] == target or info["range"] == target:
                        props_to_remove.append(prop_uri)
                for p in props_to_remove:
                    del self.approved_properties[p]
                    print(f"  [CORRECTION] Cascade-removed property: {p}")
                # Remove subclass relations involving this class
                if target in self.subclass_relations:
                    del self.subclass_relations[target]
                self.subclass_relations = {
                    k: v for k, v in self.subclass_relations.items() if v != target
                }

            elif action == "remove_property":
                if target in self.approved_properties:
                    del self.approved_properties[target]
                    print(f"  [CORRECTION] Removed property: {target} — {corr.reasoning}")

            elif action == "update_domain":
                if target in self.approved_properties and corr.new_value:
                    old = self.approved_properties[target]["domain"]
                    self.approved_properties[target]["domain"] = corr.new_value
                    print(f"  [CORRECTION] Updated domain of {target}: {old} -> {corr.new_value}")

            elif action == "update_range":
                if target in self.approved_properties and corr.new_value:
                    old = self.approved_properties[target]["range"]
                    self.approved_properties[target]["range"] = corr.new_value
                    print(f"  [CORRECTION] Updated range of {target}: {old} -> {corr.new_value}")

            elif action == "remove_subclass":
                child = corr.new_value or target
                parent = target if corr.new_value else ""
                # Try both interpretations
                if child in self.subclass_relations:
                    del self.subclass_relations[child]
                    if child in self.approved_classes:
                        self.approved_classes[child]["subclass_of"] = None
                    print(f"  [CORRECTION] Removed subClassOf: {child} — {corr.reasoning}")

    # ================================================================
    # A-Box Operations (unchanged)
    # ================================================================

    def add_entity(self, entity: Entity) -> str:
        """Add an entity with fuzzy dedup. Returns the canonical entity ID."""
        normalized = self._normalize_id(entity.id)

        if normalized in self._entity_normalized_index:
            return self._entity_normalized_index[normalized]

        for existing_norm, existing_id in self._entity_normalized_index.items():
            if self._is_similar(normalized, existing_norm):
                return existing_id

        self.entities[entity.id] = {
            "class_uri": entity.class_uri,
            "evidence_span": entity.evidence_span,
        }
        self._entity_normalized_index[normalized] = entity.id
        return entity.id

    def add_relation(self, relation: Relation, id_map: Dict[str, str] = None):
        """Add a relation, remapping entity IDs if needed."""
        source = id_map.get(relation.source_id, relation.source_id) if id_map else relation.source_id
        target = id_map.get(relation.target_id, relation.target_id) if id_map else relation.target_id

        rel_dict = {
            "source_id": source,
            "property_uri": relation.property_uri,
            "target_id": target,
            "evidence_span": relation.evidence_span,
        }

        if rel_dict not in self.relations:
            self.relations.append(rel_dict)

    def validate_entity(self, entity: Entity) -> bool:
        """Check if entity's class_uri exists in approved classes."""
        return entity.class_uri in self.approved_classes

    def validate_relation(self, relation: Relation) -> bool:
        """Check if relation's property_uri exists in approved properties."""
        return relation.property_uri in self.approved_properties

    # ================================================================
    # Formatting for Prompts
    # ================================================================

    def format_ontology_for_prompt(self) -> str:
        """Format the full T-Box (classes + properties) for prompt injection."""
        # Compact class format for A-Box prompts
        classes_compact = {}
        for uri, info in self.approved_classes.items():
            if isinstance(info, dict):
                classes_compact[uri] = info.get("comment", "")
            else:
                classes_compact[uri] = info  # legacy string format

        state = {
            "Classes": classes_compact,
            "Properties": self.approved_properties,
        }
        return json.dumps(state, indent=2, ensure_ascii=False)

    def format_classes_for_prompt(self) -> str:
        """Format classes with BFO info for T-Box prompts."""
        lines = []
        for uri, info in self.approved_classes.items():
            if isinstance(info, dict):
                bfo = info.get("bfo_parent", "bfo:Entity")
                defn = info.get("aristotelian_definition", info.get("comment", ""))
                sub = info.get("subclass_of", None)
                line = f"- {uri} (BFO: {bfo})"
                if sub:
                    line += f" subClassOf {sub}"
                line += f": {defn}"
                lines.append(line)
            else:
                lines.append(f"- {uri}: {info}")
        return "\n".join(lines) if lines else "(none)"

    def format_properties_for_prompt(self) -> str:
        """Format properties for T-Box extension prompts."""
        if not self.approved_properties:
            return "(none)"
        lines = []
        for uri, info in self.approved_properties.items():
            lines.append(
                f"- {uri} (domain: {info['domain']}, range: {info['range']}): {info['comment']}"
            )
        return "\n".join(lines)

    def format_canonical_terms_for_prompt(self) -> str:
        """Format canonical terms for prompt injection."""
        if not self.canonical_terms:
            return "{}"
        return json.dumps(self.canonical_terms, indent=2, ensure_ascii=False)

    def format_entities_for_prompt(self) -> str:
        """Format existing entities for A-Box prompts."""
        if not self.entities:
            return "None yet."
        lines = []
        for eid, info in self.entities.items():
            lines.append(f"- {eid} (type: {info['class_uri']})")
        return "\n".join(lines)

    def format_for_prompt(self) -> str:
        """Legacy alias."""
        return self.format_ontology_for_prompt()

    def get_class_uris(self) -> List[str]:
        """Return list of all approved class URIs."""
        return list(self.approved_classes.keys())

    def get_ontology_size(self) -> int:
        return len(self.approved_classes) + len(self.approved_properties)

    # ================================================================
    # Fuzzy Deduplication Helpers
    # ================================================================

    @staticmethod
    def _normalize_id(entity_id: str) -> str:
        s = entity_id.lower().strip()
        s = re.sub(r'[_\-\s]+', '', s)
        s = re.sub(r's$', '', s)
        return s

    @staticmethod
    def _is_similar(a: str, b: str, threshold: float = 0.85) -> bool:
        if a == b:
            return True
        if not a or not b:
            return False
        if a in b or b in a:
            return True
        def bigrams(s):
            return set(s[i:i+2] for i in range(len(s)-1))
        ba, bb = bigrams(a), bigrams(b)
        if not ba or not bb:
            return a == b
        intersection = len(ba & bb)
        union = len(ba | bb)
        return (intersection / union) >= threshold if union > 0 else False

    # ================================================================
    # Persistence
    # ================================================================

    def save_state(self, filepath: str):
        """Save full state to JSON."""
        os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
        state = {
            "approved_classes": self.approved_classes,
            "approved_properties": self.approved_properties,
            "canonical_terms": self.canonical_terms,
            "subclass_relations": self.subclass_relations,
            "entities": self.entities,
            "relations": self.relations,
        }
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2, ensure_ascii=False)

    def load_state(self, filepath: str):
        """Load state from JSON."""
        if not os.path.exists(filepath):
            return
        with open(filepath, "r", encoding="utf-8") as f:
            state = json.load(f)
        self.approved_classes = state.get("approved_classes", {})
        self.approved_properties = state.get("approved_properties", {})
        self.canonical_terms = state.get("canonical_terms", {})
        self.subclass_relations = state.get("subclass_relations", {})
        self.entities = state.get("entities", {})
        self.relations = state.get("relations", [])
        # Rebuild normalized index
        self._entity_normalized_index = {}
        for eid in self.entities:
            self._entity_normalized_index[self._normalize_id(eid)] = eid

    # ================================================================
    # Export — Turtle (.ttl) with BFO annotations
    # ================================================================

    @staticmethod
    def _ensure_prefix(uri: str) -> str:
        if uri.startswith("ex:") or uri.startswith("xsd:") or uri.startswith("rdfs:") or uri.startswith("owl:") or uri.startswith("bfo:"):
            return uri
        return f"ex:{uri}"

    def export_turtle(self, filepath: str):
        """Export the ontology as a valid Turtle (.ttl) file with BFO annotations."""
        os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
        lines = []

        # Prefixes
        lines.append("@prefix ex: <http://example.org/ontology#> .")
        lines.append("@prefix owl: <http://www.w3.org/2002/07/owl#> .")
        lines.append("@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .")
        lines.append("@prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .")
        lines.append("@prefix xsd: <http://www.w3.org/2001/XMLSchema#> .")
        lines.append("@prefix bfo: <http://purl.obolibrary.org/obo/BFO_> .")
        lines.append("@prefix skos: <http://www.w3.org/2004/02/skos/core#> .")
        lines.append("")

        # ── Classes ──
        lines.append("# ===== OWL Classes (BFO-anchored) =====")
        for uri, info in self.approved_classes.items():
            prefixed = self._ensure_prefix(uri)

            if isinstance(info, dict):
                comment = info.get("comment", "")
                bfo_parent = info.get("bfo_parent", "")
                aristotelian = info.get("aristotelian_definition", "")
                subclass_of = info.get("subclass_of", None)
            else:
                # Legacy: info is just a comment string
                comment = info
                bfo_parent = ""
                aristotelian = ""
                subclass_of = None

            lines.append(f"{prefixed} rdf:type owl:Class ;")

            # BFO parent as rdfs:subClassOf
            if bfo_parent:
                lines.append(f"    rdfs:subClassOf {bfo_parent} ;")

            # Domain subclass relation
            if subclass_of:
                lines.append(f"    rdfs:subClassOf {self._ensure_prefix(subclass_of)} ;")

            # Aristotelian definition as skos:definition
            if aristotelian:
                lines.append(f'    skos:definition "{self._escape_ttl(aristotelian)}" ;')

            lines.append(f'    rdfs:comment "{self._escape_ttl(comment)}" .')
            lines.append("")

        # ── Properties ──
        lines.append("# ===== Properties =====")
        for uri, info in self.approved_properties.items():
            prefixed = self._ensure_prefix(uri)
            domain = self._ensure_prefix(info['domain'])
            rng = self._ensure_prefix(info['range'])

            # Detect datatype vs object property
            if info['range'].startswith("xsd:"):
                prop_type = "owl:DatatypeProperty"
            else:
                prop_type = "owl:ObjectProperty"

            lines.append(f"{prefixed} rdf:type {prop_type} ;")
            lines.append(f"    rdfs:domain {domain} ;")
            lines.append(f"    rdfs:range {rng} ;")
            lines.append(f'    rdfs:comment "{self._escape_ttl(info["comment"])}" .')
            lines.append("")

        # ── Instances (A-Box) ──
        lines.append("# ===== Instances =====")
        for eid, info in self.entities.items():
            safe_id = re.sub(r'[^a-zA-Z0-9_]', '_', eid)
            if safe_id.startswith("ex_"):
                safe_id = safe_id[3:]
            class_uri = self._ensure_prefix(info['class_uri'])
            lines.append(f"ex:{safe_id} rdf:type {class_uri} ;")
            lines.append(f'    rdfs:label "{self._escape_ttl(eid)}" .')
            lines.append("")

        # ── Relations ──
        lines.append("# ===== Relations =====")
        for rel in self.relations:
            src = re.sub(r'[^a-zA-Z0-9_]', '_', rel["source_id"])
            tgt = re.sub(r'[^a-zA-Z0-9_]', '_', rel["target_id"])
            if src.startswith("ex_"):
                src = src[3:]
            if tgt.startswith("ex_"):
                tgt = tgt[3:]
            prop = self._ensure_prefix(rel['property_uri'])
            lines.append(f"ex:{src} {prop} ex:{tgt} .")

        lines.append("")
        with open(filepath, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

    @staticmethod
    def _escape_ttl(text: str) -> str:
        return text.replace("\\", "\\\\").replace('"', '\\"').replace("\n", " ")
