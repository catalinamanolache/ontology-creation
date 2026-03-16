creation_prompt = """
    You are a knowledge engineer. Using OWL (Turtle syntax), generate an ontology for the domain
    described by the user. The ontology must:
 
    1) Use the 'ex:' prefix for classes and properties (e.g., ex:Substance, ex:hasConcentration).
       - You can assume @prefix ex: http://example.org/ .
    2) Identify key Classes (ex:SomeClass) for the major concepts in the text.
       - Keep class names concise (TitleCase), e.g. ex:Patient, ex:Condition.
       - Avoid instance-like naming or paragraph-length names.
    3) Create relevant object properties (ex:someRelationship) or data properties (ex:someDataProperty)
       with appropriate rdfs:domain and rdfs:range.
       - For example, ex:hasAge a owl:DatatypeProperty ; rdfs:domain ex:Patient ; rdfs:range xsd:integer .
       - Keep properties general enough to be reused, avoiding single-use specifics.
    4) Provide a short comment (rdfs:comment) for each class and property describing it in 1-2 lines.
    5) Ensure no self-referential properties (no subject=object forced).
    6) Output only the ontology in valid Turtle.
       - You can include prefix declarations for clarity: @prefix ex: http://example.org/ .
       - No additional commentary, no JSON, no code blocks.
 
    ### Example (for reference only) ###
    @prefix ex: http://example.org/ .
    @prefix xsd: http://www.w3.org/2001/XMLSchema# .
    @prefix rdfs: http://www.w3.org/2000/01/rdf-schema# .
    @prefix owl: http://www.w3.org/2002/07/owl# .
 
    ex:Patient a owl:Class ;
       rdfs:comment "Represents a patient receiving care." .
 
    ex:hasAge a owl:DatatypeProperty ;
       rdfs:domain ex:Patient ;
       rdfs:range xsd:integer ;
       rdfs:comment "Indicates the patient's age." .
 
    ### User-provided domain text ###
    {document_text}
 
    ### Generate your ontology now ###
    """

refinement_prompt = """

    You are a knowledge engineer refining an existing ontology in OWL Turtle syntax (using ex: prefix).

    Below is the current ontology (Turtle), followed by the current RDF triples (JSON), then a new text chunk.

 

    You may add or modify classes and properties if (and only if) new information in the text chunk requires it.

    Maintain these rules:

      1) Use the same 'ex:' prefix.

      2) Keep class names short and conceptual. Avoid instance-like names.

      3) Keep properties general. Avoid single-use or self-referential edges.

      4) If existing classes/properties suffice, do not add duplicates.

      5) Provide brief rdfs:comment for any newly added class/property.

      6) Output valid Turtle only. No code blocks, no JSON, no extraneous commentary.

 

    ### Current Ontology (Turtle) ###

    {current_ontology_text}

 

    ### Current RDF Triples (JSON) ###

    {existing_triples_json}

 

    ### New Chunk of Text ###

    {chunk_text}

 

    ### Task ###

    Refine or extend the ontology in Turtle to accommodate new concepts. Return the updated ontology now.

    """