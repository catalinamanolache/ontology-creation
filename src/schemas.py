from pydantic import BaseModel, Field
from typing import List, Optional

class Entity(BaseModel):
    id: str = Field(description="Snake_case unique identifier for the entity")
    class_uri: str = Field(description="The URI of the class from the Approved Ontology, e.g., ex:Patient")
    evidence_span: str = Field(description="Exact substring from the text proving this entity exists")

class Relation(BaseModel):
    source_id: str = Field(description="The ID of the source entity")
    property_uri: str = Field(description="The URI of the property from the Approved Ontology, e.g., ex:hasAge")
    target_id: str = Field(description="The ID of the target entity or literal value")
    evidence_span: str = Field(description="Exact substring from the text proving this relation")

class ExtractionResult(BaseModel):
    reasoning_steps: List[str] = Field(description="Step-by-step reasoning before extracting")
    entities: List[Entity] = Field(description="List of extracted entities")
    relations: List[Relation] = Field(description="List of extracted relations")

class OntologyClass(BaseModel):
    uri: str = Field(description="The URI of the class, e.g., ex:Patient")
    comment: str = Field(description="A short description of the class")

class OntologyProperty(BaseModel):
    uri: str = Field(description="The URI of the property, e.g., ex:hasAge")
    domain: str = Field(description="The domain class URI")
    range: str = Field(description="The range class URI or datatype")
    comment: str = Field(description="A short description of the property")

class BootstrapResult(BaseModel):
    reasoning_steps: List[str] = Field(description="Step-by-step reasoning")
    classes: List[OntologyClass] = Field(description="Core classes")
    properties: List[OntologyProperty] = Field(description="Core properties")
