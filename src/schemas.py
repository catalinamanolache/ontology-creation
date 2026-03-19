"""
Pydantic schemas for the Decomposed Ontology Pipeline.

Organized by pipeline phase:
  Phase 2.1 — Competency Questions (CQ)
  Phase 2.2 — Class Extraction with BFO anchoring
  Phase 2.3 — Aristotelian Definitions (embedded in 2.2)
  Phase 2.4 — Property Extraction
  Phase 2.5 — Two-Way Hierarchy Validation
  Phase 2.6 — OWL Reasoner Self-Correction
  Phase 3   — A-Box Entity / Relation Extraction (unchanged)
"""

from pydantic import BaseModel, Field
from typing import Dict, List, Optional


# ============================================================
# Phase 2.1 — Competency Questions
# ============================================================

class CompetencyQuestion(BaseModel):
    question: str = Field(description="A question the ontology should answer, e.g., 'What proteins are involved in X?'")
    scope: str = Field(
        default="class",
        description="The ontological scope: 'class' (about types), 'property' (about relationships), or 'instance' (about individuals)"
    )


class CQResult(BaseModel):
    questions: List[CompetencyQuestion] = Field(description="Competency questions derived from the source text")


# ============================================================
# Phase 2.2 — Class Extraction with BFO Anchoring
# ============================================================

class OntologyClassBFO(BaseModel):
    uri: str = Field(description="Class URI with ex: prefix in TitleCase singular, e.g., ex:Protein")
    bfo_parent: str = Field(description="BFO leaf category, e.g., bfo:MaterialEntity, bfo:Process")
    subclass_of: Optional[str] = Field(
        default=None,
        description="URI of a domain superclass if this is a sub-type, e.g., ex:AminoAcid subclass_of ex:Molecule"
    )
    aristotelian_definition: str = Field(
        description=(
            "Aristotelian definition: 'A [ClassName] is a [BFO Parent / Genus] that [differentia]'. "
            "Example: 'A Document is an InformationContentEntity that records structured information.'"
        )
    )
    comment: str = Field(description="Concise 1-sentence rdfs:comment")


class ClassExtractionResult(BaseModel):
    reasoning_steps: List[str] = Field(description="1-2 short reasoning sentences")
    classes: List[OntologyClassBFO] = Field(description="Extracted OWL classes anchored to BFO")
    canonical_terms: Dict[str, str] = Field(
        default_factory=dict,
        description="Mapping of abbreviations/synonyms to canonical class URIs"
    )


# ============================================================
# Phase 2.4 — Property Extraction
# ============================================================

class OntologyProperty(BaseModel):
    uri: str = Field(description="Property URI in camelCase with verb prefix, e.g., ex:hasCapital")
    domain: str = Field(description="Domain class URI, e.g., ex:Country")
    range: str = Field(description="Range class URI or XSD datatype, e.g., ex:City or xsd:string")
    comment: str = Field(description="Concise 1-sentence rdfs:comment")


class PropertyExtractionResult(BaseModel):
    reasoning_steps: List[str] = Field(description="1-2 short reasoning sentences")
    properties: List[OntologyProperty] = Field(description="Extracted OWL properties with domain and range")


# ============================================================
# Phase 2.5 — Two-Way Hierarchy Validation
# ============================================================

class HierarchyValidationItem(BaseModel):
    child_uri: str = Field(description="The proposed subclass URI")
    parent_uri: str = Field(description="The proposed superclass URI")
    forward_valid: bool = Field(description="True if 'every [child] is necessarily a [parent]'")
    reverse_valid: bool = Field(description="True if 'every [parent] is necessarily a [child]' — should be False for valid hierarchy")
    reasoning: str = Field(description="1-sentence justification")


class HierarchyValidationResult(BaseModel):
    validations: List[HierarchyValidationItem] = Field(description="Validation results for each subclass relation")


# ============================================================
# Phase 2.6 — Self-Correction (OWL Reasoner feedback)
# ============================================================

class OntologyCorrection(BaseModel):
    action: str = Field(
        description="One of: 'remove_class', 'remove_property', 'update_domain', 'update_range', 'remove_subclass'"
    )
    target_uri: str = Field(description="URI of the class or property to correct")
    new_value: Optional[str] = Field(default=None, description="New value (for update actions) or null (for remove actions)")
    reasoning: str = Field(description="1-sentence explanation of why this correction is needed")


class CorrectionResult(BaseModel):
    reasoning_steps: List[str] = Field(description="1-2 sentences of reasoning")
    corrections: List[OntologyCorrection] = Field(description="List of corrections to apply")


# ============================================================
# OWL Validation Report (Python-side, not LLM-generated)
# ============================================================

class ValidationReport(BaseModel):
    is_consistent: bool = Field(description="True if no logical errors were found")
    errors: List[str] = Field(default_factory=list, description="List of error messages from the validator")
    warnings: List[str] = Field(default_factory=list, description="Non-fatal warnings")


# ============================================================
# Legacy — Seed Ontology (kept for backward compatibility)
# ============================================================

class OntologyClass(BaseModel):
    uri: str = Field(description="The URI of the class using 'ex:' prefix in TitleCase singular, e.g., ex:Country, ex:Person")
    comment: str = Field(description="A concise 1-sentence rdfs:comment describing the class")


class SeedOntologyResult(BaseModel):
    reasoning_steps: List[str] = Field(description="Step-by-step reasoning before defining the ontology")
    classes: List[OntologyClass] = Field(description="Core OWL classes identified in the text")
    properties: List[OntologyProperty] = Field(description="Core object/data properties with domain and range")
    canonical_terms: Dict[str, str] = Field(
        default_factory=dict,
        description="Dictionary mapping abbreviations and synonyms to canonical class/entity names"
    )


# ============================================================
# Phase 3 — A-Box Entity / Relation Extraction (unchanged)
# ============================================================

class Entity(BaseModel):
    id: str = Field(description="Snake_case unique identifier for the entity, e.g., 'paris', 'john_smith'")
    class_uri: str = Field(description="The URI of the class from the Approved Ontology, e.g., ex:City")
    evidence_span: str = Field(description="Exact substring from the source text proving this entity exists")


class Relation(BaseModel):
    source_id: str = Field(description="The snake_case ID of the source entity")
    property_uri: str = Field(description="The URI of the property from the Approved Ontology, e.g., ex:hasCapital")
    target_id: str = Field(description="The snake_case ID of the target entity or a literal value")
    evidence_span: str = Field(description="Exact substring from the source text proving this relation exists")


class ExtractionResult(BaseModel):
    reasoning_steps: List[str] = Field(description="Step-by-step reasoning before extracting entities and relations")
    entities: List[Entity] = Field(description="List of entities extracted from the text chunk")
    relations: List[Relation] = Field(description="List of relations extracted from the text chunk")


# ============================================================
# Legacy compatibility aliases
# ============================================================
BootstrapResult = SeedOntologyResult
