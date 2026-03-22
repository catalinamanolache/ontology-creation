"""
OWL Validator — Neuro-Symbolic Validation Layer.

Provides a layered validation approach:
  Layer 1: Python heuristics (always available)
           — cycle detection, domain/range consistency, naming conventions
  Layer 2: rdflib graph validation (always available, rdflib is a dependency)
           — RDF parse check, SPARQL-based consistency queries
  Layer 3: owlready2 + HermiT reasoner (optional, requires Java)
           — full OWL DL reasoning, unsatisfiable class detection

Usage:
    validator = OWLValidator()
    report = validator.validate(state_tracker)
    if not report.is_consistent:
        # feed report.errors back to LLM for self-correction
"""

import re
from typing import Dict, List, Set, Tuple

from src.schemas import ValidationReport


# ────────────────────────────────────────────────────────────────────
# Layer availability detection
# ────────────────────────────────────────────────────────────────────

try:
    import rdflib
    from rdflib import Graph, Namespace, RDF, RDFS, OWL, XSD, Literal, URIRef
    HAS_RDFLIB = True
except ImportError:
    HAS_RDFLIB = False

try:
    import owlready2
    HAS_OWLREADY2 = True
except ImportError:
    HAS_OWLREADY2 = False


# ────────────────────────────────────────────────────────────────────
# Namespace constants
# ────────────────────────────────────────────────────────────────────

EX_NS = "http://example.org/ontology#"
BFO_NS = "http://purl.obolibrary.org/obo/BFO_"

XSD_DATATYPES = {
    "xsd:string", "xsd:integer", "xsd:float", "xsd:boolean",
    "xsd:date", "xsd:dateTime", "xsd:decimal", "xsd:double",
    "xsd:int", "xsd:long", "xsd:nonNegativeInteger",
}


class OWLValidator:
    """Multi-layer OWL ontology validator."""

    def __init__(self):
        self._errors: List[str] = []
        self._warnings: List[str] = []

    def validate(
        self,
        approved_classes: Dict[str, dict],
        approved_properties: Dict[str, dict],
        subclass_relations: Dict[str, str],
    ) -> ValidationReport:
        """
        Run all available validation layers and return a consolidated report.

        Args:
            approved_classes: uri -> {comment, bfo_parent, aristotelian_definition, subclass_of}
            approved_properties: uri -> {domain, range, comment}
            subclass_relations: child_uri -> parent_uri
        """
        self._errors = []
        self._warnings = []

        # Layer 1: Python heuristics (always runs)
        self._validate_naming_conventions(approved_classes, approved_properties)
        self._validate_domain_range_references(approved_classes, approved_properties)
        self._detect_hierarchy_cycles(subclass_relations)
        self._validate_property_conflicts(approved_properties)

        # Layer 2: rdflib graph validation
        if HAS_RDFLIB:
            self._validate_with_rdflib(approved_classes, approved_properties, subclass_relations)

        # Layer 3: owlready2 reasoner (optional)
        if HAS_OWLREADY2:
            self._validate_with_owlready2(approved_classes, approved_properties, subclass_relations)

        return ValidationReport(
            is_consistent=len(self._errors) == 0,
            errors=self._errors,
            warnings=self._warnings,
        )

    # ================================================================
    # Layer 1: Python Heuristics
    # ================================================================

    def _validate_naming_conventions(
        self,
        classes: Dict[str, dict],
        properties: Dict[str, dict],
    ):
        """Check naming conventions: TitleCase classes, camelCase properties."""
        for uri in classes:
            local = uri.replace("ex:", "")
            if not local[0].isupper():
                self._warnings.append(
                    f"Class '{uri}' does not follow TitleCase convention."
                )
            if " " in local or "-" in local:
                self._errors.append(
                    f"Class '{uri}' contains spaces or hyphens — invalid OWL local name."
                )

        for uri in properties:
            local = uri.replace("ex:", "")
            if local[0].isupper():
                self._warnings.append(
                    f"Property '{uri}' starts with uppercase — should be camelCase."
                )

    def _validate_domain_range_references(
        self,
        classes: Dict[str, dict],
        properties: Dict[str, dict],
    ):
        """Ensure every property's domain/range references an existing class or XSD datatype."""
        class_uris = set(classes.keys())
        # Also allow owl:Thing and rdfs:Literal as valid targets
        allowed_special = {"owl:Thing", "rdfs:Literal", "ex:Thing"}

        for prop_uri, info in properties.items():
            domain = info.get("domain", "")
            rng = info.get("range", "")

            if domain not in class_uris and domain not in allowed_special:
                self._errors.append(
                    f"Property '{prop_uri}' has domain '{domain}' which is not a known class."
                )

            if rng not in class_uris and rng not in XSD_DATATYPES and rng not in allowed_special:
                self._errors.append(
                    f"Property '{prop_uri}' has range '{rng}' which is not a known class or XSD datatype."
                )

    def _detect_hierarchy_cycles(self, subclass_relations: Dict[str, str]):
        """Detect cycles in the subClassOf graph using DFS."""
        # Build adjacency list: child -> parent
        if not subclass_relations:
            return

        # DFS cycle detection
        WHITE, GRAY, BLACK = 0, 1, 2
        color = {node: WHITE for node in set(subclass_relations.keys()) | set(subclass_relations.values())}

        def dfs(node: str, path: List[str]):
            color[node] = GRAY
            path.append(node)
            parent = subclass_relations.get(node)
            if parent:
                if color.get(parent, WHITE) == GRAY:
                    cycle_start = path.index(parent)
                    cycle = path[cycle_start:] + [parent]
                    self._errors.append(
                        f"Hierarchy cycle detected: {' -> '.join(cycle)}"
                    )
                elif color.get(parent, WHITE) == WHITE:
                    dfs(parent, path)
            path.pop()
            color[node] = BLACK

        for node in list(color.keys()):
            if color[node] == WHITE:
                dfs(node, [])

    def _validate_property_conflicts(self, properties: Dict[str, dict]):
        """Check for properties with identical domain+range that may be redundant."""
        seen: Dict[Tuple[str, str], List[str]] = {}
        for uri, info in properties.items():
            key = (info.get("domain", ""), info.get("range", ""))
            seen.setdefault(key, []).append(uri)

        for (domain, rng), uris in seen.items():
            if len(uris) > 3:
                self._warnings.append(
                    f"Multiple properties ({len(uris)}) share domain='{domain}' range='{rng}': "
                    f"{', '.join(uris[:4])}... — possible redundancy."
                )

    # ================================================================
    # Layer 2: rdflib Graph Validation
    # ================================================================

    def _validate_with_rdflib(
        self,
        classes: Dict[str, dict],
        properties: Dict[str, dict],
        subclass_relations: Dict[str, str],
    ):
        """Build an rdflib graph and run SPARQL-based consistency checks."""
        g = Graph()
        EX = Namespace(EX_NS)
        g.bind("ex", EX)
        g.bind("owl", OWL)
        g.bind("rdfs", RDFS)

        # Add classes
        for uri in classes:
            local = uri.replace("ex:", "")
            g.add((EX[local], RDF.type, OWL.Class))

        # Add subclass relations
        for child, parent in subclass_relations.items():
            child_local = child.replace("ex:", "")
            parent_local = parent.replace("ex:", "")
            g.add((EX[child_local], RDFS.subClassOf, EX[parent_local]))

        # Add properties
        for uri, info in properties.items():
            local = uri.replace("ex:", "")
            domain_local = info.get("domain", "Thing").replace("ex:", "")
            range_val = info.get("range", "Thing")

            g.add((EX[local], RDF.type, OWL.ObjectProperty))
            g.add((EX[local], RDFS.domain, EX[domain_local]))

            if range_val.startswith("xsd:"):
                # Datatype property — mark accordingly
                g.remove((EX[local], RDF.type, OWL.ObjectProperty))
                g.add((EX[local], RDF.type, OWL.DatatypeProperty))
                xsd_type = getattr(XSD, range_val.replace("xsd:", ""), XSD.string)
                g.add((EX[local], RDFS.range, xsd_type))
            else:
                range_local = range_val.replace("ex:", "")
                g.add((EX[local], RDFS.range, EX[range_local]))

        # SPARQL: Check for classes that are both domain and range of same property
        # but are in a disjoint subclass branch (basic check)
        try:
            g.serialize(format="turtle")
        except Exception as e:
            self._errors.append(f"rdflib serialization error (malformed graph): {e}")

        # SPARQL: Find properties whose domain is a subclass of their range (suspicious)
        query = """
        PREFIX ex: <%s>
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
        PREFIX owl: <http://www.w3.org/2002/07/owl#>

        SELECT ?prop ?domain ?range WHERE {
            ?prop rdfs:domain ?domain .
            ?prop rdfs:range ?range .
            ?domain rdfs:subClassOf ?range .
            ?prop a owl:ObjectProperty .
        }
        """ % EX_NS

        try:
            results = g.query(query)
            for row in results:
                self._warnings.append(
                    f"Property <{row.prop}> has domain that is a subclass of its range — "
                    f"may indicate a modelling issue."
                )
        except Exception:
            pass  # SPARQL query failed; non-fatal

    # ================================================================
    # Layer 3: owlready2 + HermiT Reasoner
    # ================================================================

    def _validate_with_owlready2(
        self,
        classes: Dict[str, dict],
        properties: Dict[str, dict],
        subclass_relations: Dict[str, str],
    ):
        """Build an owlready2 ontology and run the HermiT reasoner."""
        try:
            onto = owlready2.get_ontology(EX_NS)

            with onto:
                # Create classes
                owl_classes = {}
                for uri in classes:
                    local = uri.replace("ex:", "")
                    cls = type(local, (owlready2.Thing,), {"namespace": onto})
                    owl_classes[uri] = cls

                # Apply subclass relations
                for child_uri, parent_uri in subclass_relations.items():
                    child_cls = owl_classes.get(child_uri)
                    parent_cls = owl_classes.get(parent_uri)
                    if child_cls and parent_cls:
                        child_cls.is_a.append(parent_cls)

                # Create properties
                for uri, info in properties.items():
                    local = uri.replace("ex:", "")
                    domain_cls = owl_classes.get(info.get("domain", ""))
                    range_cls = owl_classes.get(info.get("range", ""))

                    range_val = info.get("range", "")
                    if range_val.startswith("xsd:"):
                        # Datatype property
                        prop = type(local, (owlready2.DataProperty,), {"namespace": onto})
                        if domain_cls:
                            prop.domain = [domain_cls]
                    else:
                        prop = type(local, (owlready2.ObjectProperty,), {"namespace": onto})
                        if domain_cls:
                            prop.domain = [domain_cls]
                        if range_cls:
                            prop.range = [range_cls]

            # Run HermiT reasoner
            try:
                with onto:
                    owlready2.sync_reasoner(infer_property_values=False)

                # Check for unsatisfiable (Nothing) classes
                for cls in onto.classes():
                    if cls is owlready2.Nothing:
                        continue
                    if owlready2.Nothing in cls.equivalent_to:
                        self._errors.append(
                            f"Class '{cls.name}' is unsatisfiable (equivalent to owl:Nothing)."
                        )
                    # Check ancestors for Nothing
                    try:
                        ancestors = cls.ancestors()
                        if owlready2.Nothing in ancestors:
                            self._errors.append(
                                f"Class '{cls.name}' has owl:Nothing as ancestor — inconsistent."
                            )
                    except Exception:
                        pass

            except Exception as e:
                err_msg = str(e)
                if "java" in err_msg.lower() or "jvm" in err_msg.lower():
                    self._warnings.append(
                        "owlready2: Java/JVM not available — skipping HermiT reasoning. "
                        "Install Java to enable full OWL DL validation."
                    )
                else:
                    self._errors.append(f"owlready2 reasoner error: {err_msg}")

            # Cleanup: destroy the ontology to avoid conflicts on next run
            onto.destroy()

        except Exception as e:
            self._warnings.append(f"owlready2 validation skipped due to error: {e}")


def format_errors_for_prompt(report: ValidationReport) -> str:
    """Format validation errors into a string suitable for the self-correction prompt."""
    lines = []
    for i, err in enumerate(report.errors, 1):
        lines.append(f"ERROR {i}: {err}")
    for i, warn in enumerate(report.warnings, 1):
        lines.append(f"WARNING {i}: {warn}")
    return "\n".join(lines) if lines else "No errors detected."
