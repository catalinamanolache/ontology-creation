# ============================================================
# Decomposed Prompt Templates — Axiom-by-Axiom (AbA) Pipeline
#
# Replaces the monolithic T-Box prompt with serialized phases:
#   2.1  Competency Questions (CQ)
#   2.2  Class Extraction + BFO anchoring + Aristotelian definitions
#   2.2e Class Extension (delta-only)
#   2.4  Property Extraction
#   2.4e Property Extension (delta-only)
#   2.5  Two-Way Hierarchy Validation
#   2.6  Self-Correction from Reasoner errors
#   3.0  A-Box Extraction (unchanged)
# ============================================================


# ────────────────────────────────────────────────────────────────────
# Phase 2.1 — Competency Questions
# ────────────────────────────────────────────────────────────────────

PROMPT_CQ_GENERATION = """
### ROLE
You are a Knowledge Engineer. Generate Competency Questions (CQs) that an OWL ontology built from this text should be able to answer.

### INSTRUCTIONS
1. Read the source text carefully.
2. Generate 5-10 questions that capture the key knowledge in the text.
3. Each question must test whether the ontology can represent a specific fact or relationship.
4. Tag each question with a scope: "class" (about types of things), "property" (about relationships), or "instance" (about specific individuals).

### OUTPUT FORMAT
Return a single JSON object with one key "questions", containing a list of objects.
Each object has exactly 2 keys:
- "question": The competency question string
- "scope": One of "class", "property", or "instance"

CRITICAL: Start your response directly with {{ and end with }}. No markdown, no explanation.

### SOURCE TEXT
{document_text}
"""


# ────────────────────────────────────────────────────────────────────
# Phase 2.2 — Class Extraction (Initial) with BFO + Aristotelian Def
# ────────────────────────────────────────────────────────────────────

PROMPT_CLASS_EXTRACTION_INITIAL = """
### ROLE
You are a Knowledge Engineer. Extract ONLY the OWL classes from the text below.
Do NOT extract properties, relationships, or instances.

### BFO ONTOLOGY SKELETON
{bfo_skeleton}

### COMPETENCY QUESTIONS (use as guidance for what concepts matter)
{competency_questions}

### STRICT RULES
1. NAMESPACE: Use ONLY the ex: prefix. Every URI must start with ex:.
2. NAMING: Classes MUST be TitleCase singular nouns (ex:Protein, NOT ex:Proteins).
3. BFO ANCHORING: Every class MUST have a bfo_parent from the BFO leaf categories above.
4. ARISTOTELIAN DEFINITION: Every class MUST have a definition in this EXACT format:
   "A [ClassName] is a [BFO Parent name without prefix] that [distinguishing characteristic]."
   Example: "A Document is an InformationContentEntity that records structured information."
   Example: "A Reaction is a Process that transforms chemical substances."
5. NO INSTANCES: Do not create classes for specific named individuals (Paris is an instance, City is a class).
6. SUBCLASS: If class A is a more specific type of class B, set "subclass_of" to B's URI. Otherwise set it to null.
7. Keep classes general and reusable. Aim for 5-15 classes.

### OUTPUT FORMAT
Return a single JSON object with keys in this EXACT order:
1. "classes": List of objects, each with: "uri", "bfo_parent", "subclass_of", "aristotelian_definition", "comment"
2. "canonical_terms": Dict mapping abbreviations/synonyms to canonical URIs
3. "reasoning_steps": List of 1-2 short sentences

CRITICAL: Start your response directly with {{ and end with }}. No markdown, no explanation.

### SOURCE TEXT
{document_text}
"""


# ────────────────────────────────────────────────────────────────────
# Phase 2.2e — Class Extension (Delta-Only)
# ────────────────────────────────────────────────────────────────────

PROMPT_CLASS_EXTENSION = """
### ROLE
You are a Knowledge Engineer extending an existing set of OWL classes with NEW concepts only.

### BFO ONTOLOGY SKELETON
{bfo_skeleton}

### COMPETENCY QUESTIONS
{competency_questions}

### EXISTING CLASSES (do NOT duplicate — only ADD what is missing)
{existing_classes}

### STRICT EXTENSION RULES
1. NO REDUNDANCY: If a concept is already covered by an existing class, do NOT add it.
2. CANONICAL MAPPING: If the new text uses a different word for an existing class, add it to "canonical_terms" instead.
3. HIERARCHY: If a new concept is a specific sub-type, set "subclass_of" to the existing class URI.
4. DELTA ONLY: Return ONLY newly discovered classes. Do NOT repeat existing ones.
5. BFO + ARISTOTELIAN: Same rules as initial extraction (bfo_parent required, aristotelian definition required).
6. If the existing classes already cover everything, return empty "classes" list.

### OUTPUT FORMAT
Return a single JSON object with keys in this EXACT order:
1. "classes": List of NEW class objects only (uri, bfo_parent, subclass_of, aristotelian_definition, comment)
2. "canonical_terms": Dict mapping NEW abbreviations/synonyms to canonical URIs
3. "reasoning_steps": List of 1-2 short sentences

CRITICAL: Start your response directly with {{ and end with }}. No markdown, no explanation.

### NEW SOURCE TEXT
{document_text}
"""


# ────────────────────────────────────────────────────────────────────
# Phase 2.4 — Property Extraction (Initial)
# ────────────────────────────────────────────────────────────────────

PROMPT_PROPERTY_EXTRACTION_INITIAL = """
### ROLE
You are a Knowledge Engineer. Extract ONLY the OWL properties (relationships) from the text.
The classes have already been defined — use them as domain and range.

### AVAILABLE CLASSES
{existing_classes}

### COMPETENCY QUESTIONS (use as guidance for what relationships matter)
{competency_questions}

### STRICT RULES
1. NAMESPACE: Use ONLY the ex: prefix.
2. NAMING: Properties MUST be camelCase with a verb prefix (ex:hasCapital, ex:locatedIn, ex:writtenBy).
3. ANTI-REDUNDANCY: Never include the domain class name inside the property name. Wrong: ex:proteinHasFunction. Right: ex:hasFunction.
4. DOMAIN AND RANGE CONSTRAINT: Both domain and range MUST reference a class from AVAILABLE CLASSES above, OR an XSD datatype (xsd:string, xsd:integer, xsd:float, xsd:boolean, xsd:date).
5. If a property's domain or range class does not exist in AVAILABLE CLASSES, do NOT create that property.
6. Aim for 5-20 properties depending on domain complexity.

### OUTPUT FORMAT
Return a single JSON object with keys in this EXACT order:
1. "properties": List of objects, each with: "uri", "domain", "range", "comment"
2. "reasoning_steps": List of 1-2 short sentences

CRITICAL: Start your response directly with {{ and end with }}. No markdown, no explanation.

### SOURCE TEXT
{document_text}
"""


# ────────────────────────────────────────────────────────────────────
# Phase 2.4e — Property Extension (Delta-Only)
# ────────────────────────────────────────────────────────────────────

PROMPT_PROPERTY_EXTENSION = """
### ROLE
You are a Knowledge Engineer extending an existing set of OWL properties with NEW relationships only.

### AVAILABLE CLASSES
{existing_classes}

### EXISTING PROPERTIES (do NOT duplicate)
{existing_properties}

### COMPETENCY QUESTIONS
{competency_questions}

### STRICT EXTENSION RULES
1. DELTA ONLY: Return ONLY newly discovered properties. Do NOT repeat existing ones.
2. SEMANTIC CHECK: Before adding, verify the meaning is not already covered by an existing property.
3. DOMAIN/RANGE: Must reference classes from AVAILABLE CLASSES or XSD datatypes.
4. Same naming conventions as initial extraction.
5. If existing properties cover everything, return empty "properties" list.

### OUTPUT FORMAT
Return a single JSON object with keys in this EXACT order:
1. "properties": List of NEW property objects only (uri, domain, range, comment)
2. "reasoning_steps": List of 1-2 short sentences

CRITICAL: Start your response directly with {{ and end with }}. No markdown, no explanation.

### NEW SOURCE TEXT
{document_text}
"""


# ────────────────────────────────────────────────────────────────────
# Phase 2.5 — Two-Way Hierarchy Validation
# ────────────────────────────────────────────────────────────────────

PROMPT_TWO_WAY_VALIDATION = """
### TASK
Validate the following proposed rdfs:subClassOf relationships using bidirectional reasoning.

For each relationship "A rdfs:subClassOf B" (A is a sub-type of B), answer TWO questions:
1. FORWARD: "Is every instance of A necessarily also an instance of B?" → Must be YES for a valid subclass.
2. REVERSE: "Is every instance of B necessarily also an instance of A?" → Must be NO for a valid subclass (if YES, they are equivalent, not sub/super).

### RELATIONSHIPS TO VALIDATE
{relationships}

### OUTPUT FORMAT
Return a single JSON object with key "validations", a list of objects:
- "child_uri": the subclass URI
- "parent_uri": the superclass URI
- "forward_valid": true if every child is necessarily a parent (should be true)
- "reverse_valid": true if every parent is necessarily a child (should be false for valid hierarchy)
- "reasoning": 1-sentence explanation

CRITICAL: Start your response directly with {{ and end with }}. No markdown, no explanation.
"""


# ────────────────────────────────────────────────────────────────────
# Phase 2.6 — Self-Correction from Reasoner Errors
# ────────────────────────────────────────────────────────────────────

PROMPT_SELF_CORRECTION = """
### ROLE
You are a Knowledge Engineer fixing logical errors detected by an OWL reasoner.

### CURRENT ONTOLOGY
{current_ontology}

### ERRORS DETECTED BY REASONER
{error_log}

### INSTRUCTIONS
For each error, propose exactly one correction. Available actions:
- "remove_class": Remove a problematic class entirely
- "remove_property": Remove a problematic property entirely
- "update_domain": Change a property's domain to a valid class
- "update_range": Change a property's range to a valid class or datatype
- "remove_subclass": Remove a subClassOf relation (set new_value to the child URI, target_uri to parent URI)

### OUTPUT FORMAT
Return a single JSON object with keys:
1. "corrections": List of objects with "action", "target_uri", "new_value" (null if removing), "reasoning"
2. "reasoning_steps": List of 1-2 short sentences

CRITICAL: Start your response directly with {{ and end with }}. No markdown, no explanation.
"""


# ────────────────────────────────────────────────────────────────────
# Phase 3 — A-Box Extraction (UNCHANGED from original)
# ────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT_EXTRACTION = """
You are a high-precision extraction engine.
Task: Identify facts and format them as triples using ONLY the provided ontology.

### STRICT FILTRATION RULES
1. CLOSED WORLD ASSUMPTION: If a fact is in the text but its Class or Property is NOT in approved_ontology, you MUST ignore it. Do not invent schema elements.
2. EXACT MATCH: The 'evidence_span' must be a verbatim substring from the text.
3. IDENTIFIER CONSISTENCY: Use lowercase_snake_case for entity IDs.
4. CROSS-CHECK: Before outputting a triple, verify that the subject's type matches the property's domain.

APPROVED ONTOLOGY (you MUST use ONLY these classes and properties):
{approved_ontology}

CANONICAL TERMS (use these to resolve abbreviations and synonyms):
{canonical_terms}

ALREADY EXTRACTED ENTITIES (DO NOT create duplicates - reuse these IDs if the text refers to the same thing):
{existing_entities}

### OUTPUT STRUCTURE SPECIFICATION
Your output must be a single JSON object. Do not include markdown blocks.
The object must have exactly these 3 keys:
1. "reasoning_steps": List of 1-2 short strings explaining your extraction logic.
2. "entities": List of objects with keys "id", "class_uri", "evidence_span".
3. "relations": List of objects with keys "source_id", "property_uri", "target_id", "evidence_span".

CRITICAL: Start your response directly with {{ and end with }}. No markdown, no explanation.

### Text Chunk
{chunk_text}
"""
