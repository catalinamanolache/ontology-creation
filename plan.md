# Enterprise Knowledge Graph Pipeline — Project Plan

> Arhitectură Axiom-by-Axiom (AbA) cu validare neuro-simbolică.
> Versiunea curentă implementează toate cele 5 îmbunătățiri academice față de pipeline-ul 2-pass original.

---

## 1. Motivație și Problemă

Generarea de ontologii cu LLM-uri suferă de patru eșecuri sistematice:

| Problemă | Cauză | Impact |
|---|---|---|
| **Blind Extraction** | Promptul monolit cere clase + proprietăți + constrângeri simultan | LLM-ul halucinează domain/range pentru că nu are clasele definite când generează proprietățile |
| **Erori categoriale** | LLM-ul generează clase "de la zero" fără un sistem de referință | `ex:Author` poate deveni subclasă a `ex:Document` în loc de `bfo:Role` |
| **Cicluri ierarhice** | `A subClassOf B` și `B subClassOf A` coexistă | OWL reasoner-ul declară ontologia inconsistentă |
| **Definiții vagi** | `rdfs:comment: "O entitate care reprezintă ceva"` | Zero valoare semantică; nu poate fi folosit pentru inferență |

**Soluția**: Descompunere completă a fazei T-Box în pași serializați + validare neuro-simbolică.

---

## 2. Structura Proiectului

```
covaliu/
├── src/
│   ├── main.py                # Orchestrator principal (5 faze)
│   ├── extraction.py          # LLM calls: cache + retry + JSON repair
│   ├── state_tracker.py       # Starea cumulativă T-Box + A-Box
│   ├── prompts.py             # 8 prompturi decomposed (AbA)
│   ├── schemas.py             # Modele Pydantic pentru fiecare fază
│   ├── bfo_skeleton.py        # Mini-schelet BFO 2.0 (17 categorii)
│   ├── owl_validator.py       # Validator neuro-simbolic pe 3 straturi
│   ├── document_processor.py  # Chunking deterministic
│   ├── rate_limiter.py        # Sliding window RPM/RPD
│   ├── graph_builder.py       # Vizualizare Pyvis HTML
│   └── final_report.py        # Evaluare Jaccard run-to-run
├── config/
│   └── settings.py            # Pydantic-settings (.env)
├── data/
│   ├── input/                 # sample.pdf sau sample.txt
│   ├── chunks/                # chunk_000.txt, chunk_001.txt ...
│   ├── cache/                 # MD5 cache (prompt hash → response)
│   ├── runs/                  # run_1/, run_2/, ... (arhive complete)
│   └── output/                # ontology_blueprint.json, ontology_final.ttl
├── requirements.txt
├── plan.md                    # Acest fișier
├── FLOW.md                    # Detalii tehnice de flux
└── README.md
```

---

## 3. Stack Tehnologic

| Strat | Tehnologie | Rol |
|---|---|---|
| **Orchestrare** | Python 3.10+, LangChain | Coordonare pipeline faze |
| **LLM Backends** | Ollama (local), Google Gemini, Hugging Face | Flexibilitate deployment |
| **Structured Output** | LangChain `.with_structured_output()` + Pydantic | Forțare format JSON |
| **JSON Repair** | Algoritm custom cu backtracking | Fail-safe pentru trunchieri HF |
| **Ontologie** | RDFLib, owlready2 (opțional, Java) | Validare RDF/OWL |
| **BFO** | Basic Formal Ontology 2.0 (17 categorii) | Ancorare categorială top-level |
| **Caching** | MD5 hash pe prompt → disk JSON | Zero re-apeluri LLM |
| **Rate Limiting** | Sliding window deque | Respectare limite free-tier |
| **Export** | Turtle (.ttl), NetworkX, Pyvis HTML | Interoperabilitate RDF |

---

## 4. Cele 5 Îmbunătățiri Arhitecturale

### 4.1 Descompunere Axiom-by-Axiom (AbA)

**Problema**: Promptul `SYSTEM_PROMPT_SEED_INITIAL` cerea simultan clase, proprietăți și constrângeri de domain/range. LLM-ul nu putea garanta că proprietatea `ex:hasAuthor` cu `domain: ex:Document` era validă dacă `ex:Document` nu era încă definit.

**Soluția**: Pipeline seriallizat în 3 sub-pași per batch:

```
Batch N:
  2.1 → generate_competency_questions(text)     # Ce întrebări trebuie ontologia să răspundă?
  2.2 → extract_classes_initial/extend(text)    # NUMAI clase, ancorate BFO
  2.4 → extract_properties_initial/extend(text) # NUMAI proprietăți, domain/range din clase deja aprobate
```

**Fișiere**: `src/prompts.py` (6 prompturi noi), `src/extraction.py` (metode noi), `src/schemas.py` (modele noi)

---

### 4.2 Validare Ierarhie Two-Way Chain-of-Thought

**Problema**: LLM-urile produc cicluri (`A subClassOf B, B subClassOf A`) sau inversiuni (`ex:Animal subClassOf ex:Dog`).

**Soluția**: Două mecanisme în cascadă:

1. **Python DFS** (Layer 0, fără LLM): Detectează și elimină ciclurile înainte de validarea LLM.
2. **LLM batch validation** (Layer 1): Trimite toate relațiile `subClassOf` într-un singur prompt și cere răspuns bidirecțional:
   - Forward: "Este orice `A` în mod necesar un `B`?" → trebuie TRUE
   - Reverse: "Este orice `B` în mod necesar un `A`?" → trebuie FALSE (altfel sunt echivalente)

**Fișiere**: `src/state_tracker.py` (`detect_hierarchy_cycles`, `remove_subclass`), `src/extraction.py` (`validate_hierarchy_batch`), `src/prompts.py` (`PROMPT_TWO_WAY_VALIDATION`)

---

### 4.3 Ancorare BFO Top-Level Ontology

**Problema**: LLM-urile generează clase fără referință categorială — `ex:Author` poate fi tratată ca fizic, informațional sau social în funcție de context.

**Soluția**: Injectare mini-schelet BFO 2.0 în fiecare prompt de extracție de clase. Fiecare clasă trebuie să declare `bfo_parent` din lista de categorii frunză:

```
bfo:MaterialEntity      → obiecte fizice (proteine, molecule, artefacte)
bfo:InformationContentEntity → documente, date, specificații
bfo:Process             → activități, reacții, evenimente
bfo:Role                → context social/instituțional (autor, pacient)
bfo:Quality             → atribute inerente (culoare, masă)
bfo:Function            → capacități proiectate (a pompa, a tăia)
bfo:SpatialRegion       → zone geografice/spațiale
bfo:TemporalRegion      → intervale de timp
```

**Output în Turtle**:
```turtle
ex:Document rdfs:subClassOf bfo:InformationContentEntity .
ex:Author   rdfs:subClassOf bfo:Role .
```

**Fișiere**: `src/bfo_skeleton.py` (complet nou), `src/prompts.py` (injecție BFO), `src/state_tracker.py` (export TTL cu BFO)

---

### 4.4 Definiții Aristotelice

**Problema**: `rdfs:comment: "A class that represents a document"` nu are valoare semantică.

**Soluția**: Fiecare clasă primește `aristotelian_definition` în format strict:
```
"Un [Concept] este un [Genus / BFO Parent] care [Differentia]"
```

**Exemple**:
```
"A Document is an InformationContentEntity that records structured information."
"A Reaction is a Process that transforms chemical substances into products."
"An Author is a Role that is held by an agent who creates a work."
```

Definiția este exportată în Turtle ca `skos:definition`.

**Fișiere**: `src/schemas.py` (`OntologyClassBFO.aristotelian_definition`), `src/prompts.py` (constrângere format), `src/state_tracker.py` (export `skos:definition`)

---

### 4.5 Arhitectură Neuro-Simbolică (OWL Validator)

**Problema**: LLM-urile nu pot garanta consistența OWL/RDF — pot crea proprietăți cu domain greșit, clase orfane, sau definiții disjuncte suprapuse.

**Soluția**: Validator pe 3 straturi în `src/owl_validator.py`:

| Strat | Tehnologie | Ce verifică | Disponibilitate |
|---|---|---|---|
| **L1** | Python pur | Naming, domain/range refs, cicluri DFS, redundanță proprietăți | Mereu |
| **L2** | rdflib + SPARQL | Serializare RDF, domaine = subclase ale range-ului | Mereu (dep. existentă) |
| **L3** | owlready2 + HermiT | Clase nesatisfiabile (owl:Nothing), raționament OWL DL complet | Opțional (necesită Java) |

**Self-Correction Loop** (max 3 runde):
```
validate() → ValidationReport(errors=[...])
    ↓ dacă există erori
format_errors_for_prompt(report)
    ↓
extractor.self_correct(ontology, errors) → CorrectionResult
    ↓
state.apply_corrections(result)
    ↓
validate() din nou
```

**Acțiuni de corecție suportate**: `remove_class`, `remove_property`, `update_domain`, `update_range`, `remove_subclass`.

**Fișiere**: `src/owl_validator.py` (complet nou), `src/extraction.py` (`self_correct`), `src/state_tracker.py` (`apply_corrections`), `src/prompts.py` (`PROMPT_SELF_CORRECTION`)

---

## 5. Arhitectura Completă a Pipeline-ului

```
INPUT (PDF / TXT)
    │
    ▼
Phase 1: DocumentProcessor
    ├── PyPDFLoader
    ├── _clean_text() [regex: copyright, ISSN, numere pagini]
    ├── Tăiere la "REFERENCES" / "Acknowledgments"
    └── RecursiveCharacterTextSplitter(size=1500, overlap=250)
         → chunks[] salvate în data/chunks/
    │
    ▼
Phase 2: T-Box Construction [per batch de 3 chunks]
    │
    ├── 2.1 CQ Generation
    │     └── PROMPT_CQ_GENERATION → CQResult(questions=[])
    │
    ├── 2.2 Class Extraction
    │     ├── [batch=0] PROMPT_CLASS_EXTRACTION_INITIAL
    │     │     ├── Injecție BFO_PROMPT_SKELETON
    │     │     └── OntologyClassBFO(uri, bfo_parent, subclass_of, aristotelian_definition, comment)
    │     └── [batch>0] PROMPT_CLASS_EXTENSION (delta only)
    │           └── state.ingest_classes(result)
    │
    ├── 2.4 Property Extraction
    │     ├── [batch=0] PROMPT_PROPERTY_EXTRACTION_INITIAL
    │     │     └── domain/range CONSTRÂNS la clase deja aprobate
    │     └── [batch>0] PROMPT_PROPERTY_EXTENSION (delta only)
    │           └── state.ingest_properties(result) [cu validare domain/range]
    │
    ├── 2.5 Two-Way Hierarchy Validation [post-batching]
    │     ├── detect_hierarchy_cycles() [DFS Python]
    │     └── validate_hierarchy_batch() [LLM batch]
    │           └── remove_subclass(child, parent) pentru cele invalide
    │
    └── 2.6 OWL Validator + Self-Correction [max 3 runde]
          ├── OWLValidator.validate(classes, properties, subclass_relations)
          │     ├── Layer 1: Python heuristics
          │     ├── Layer 2: rdflib
          │     └── Layer 3: owlready2 (opțional)
          └── dacă !consistent:
                ├── extractor.self_correct(ontology, errors) → CorrectionResult
                └── state.apply_corrections(result)
    │
    ▼
BLUEPRINT FROZEN → ontology_blueprint.json
    │
    ▼
Phase 3: A-Box Population [per chunk]
    ├── SYSTEM_PROMPT_EXTRACTION (neschimbat)
    │     ├── Closed World Assumption: NUMAI clase/proprietăți din blueprint
    │     ├── ExtractionResult(entities=[], relations=[])
    │     └── Validare + fuzzy dedup per entitate
    │
    └── state.entities + state.relations acumulate
    │
    ▼
Phase 4: Output & Persistence
    ├── ontology_state.json      [stare completă]
    ├── ontology_final.ttl       [RDF/OWL cu BFO + skos:definition]
    └── data/runs/run_N/         [arhivă completă a run-ului]
    │
    ▼
Phase 5: Evaluation (final_report.py)
    ├── Generare KG HTML (Pyvis) per run
    └── Jaccard similarity run-to-run (clase, proprietăți, triple)
```

---

## 6. Modele Pydantic (Schemas)

```python
# Phase 2.1
CQResult(questions: List[CompetencyQuestion])
CompetencyQuestion(question: str, scope: "class"|"property"|"instance")

# Phase 2.2
ClassExtractionResult(reasoning_steps, classes: List[OntologyClassBFO], canonical_terms)
OntologyClassBFO(uri, bfo_parent, subclass_of, aristotelian_definition, comment)

# Phase 2.4
PropertyExtractionResult(reasoning_steps, properties: List[OntologyProperty])
OntologyProperty(uri, domain, range, comment)

# Phase 2.5
HierarchyValidationResult(validations: List[HierarchyValidationItem])
HierarchyValidationItem(child_uri, parent_uri, forward_valid, reverse_valid, reasoning)

# Phase 2.6
CorrectionResult(reasoning_steps, corrections: List[OntologyCorrection])
OntologyCorrection(action, target_uri, new_value, reasoning)

# Phase 3 (neschimbat)
ExtractionResult(reasoning_steps, entities: List[Entity], relations: List[Relation])
```

---

## 7. Multi-Backend Support

Toți cei 3 backends trec prin același `_call_with_retry_and_cache()`:

```python
# .env
LLM_BACKEND=ollama       # sau gemini, huggingface
OLLAMA_MODEL=qwen2.5:7b  # orice model Ollama
OLLAMA_BASE_URL=http://localhost:11434
GOOGLE_API_KEY=...
HUGGINGFACE_API_KEY=...
HUGGINGFACE_MODEL=mistralai/Mistral-7B-Instruct-v0.2
```

**Diferențe de tratament**:
- **Ollama / Gemini**: `llm.with_structured_output(ModelClass)` — output structurat nativ
- **HuggingFace**: text raw → extracție JSON → `_normalize_hf_json()` → Pydantic validate → JSON repair dacă eșuează

---

## 8. Comenzi de Rulare

```bash
# Instalare dependințe
pip install -r requirements.txt

# Pipeline complet
python src/main.py

# Rulare cu un document de intrare specific din folderul /data/input/
python src/main.py --input "document_mare.pdf"

# Fără cache (forțează re-apeluri LLM)
python src/main.py --no-cache

# Curăță tot cacheul și rulează din nou
python src/main.py --fresh

# Numai faza T-Box (fără populare A-Box)
python src/main.py --blueprint-only

# Specificare folder de date izolat (sau resume)
python src/main.py --data-dir "data/runs/custom_run"

# Evaluare comparativă a run-urilor
python src/final_report.py
```

---

## 9. Structura Fișierelor de Output

### `ontology_blueprint.json`
```json
{
  "approved_classes": {
    "ex:Document": {
      "comment": "...",
      "bfo_parent": "bfo:InformationContentEntity",
      "aristotelian_definition": "A Document is an InformationContentEntity that...",
      "subclass_of": null
    }
  },
  "approved_properties": {
    "ex:hasAuthor": { "domain": "ex:Document", "range": "ex:Author", "comment": "..." }
  },
  "canonical_terms": { "doc": "ex:Document", "paper": "ex:Document" },
  "subclass_relations": { "ex:ResearchPaper": "ex:Document" }
}
```

### `ontology_final.ttl`
```turtle
@prefix ex:   <http://example.org/ontology#> .
@prefix bfo:  <http://purl.obolibrary.org/obo/BFO_> .
@prefix skos: <http://www.w3.org/2004/02/skos/core#> .

ex:Document rdf:type owl:Class ;
    rdfs:subClassOf bfo:InformationContentEntity ;
    skos:definition "A Document is an InformationContentEntity that records structured information." ;
    rdfs:comment "A document artifact." .
```

---

## 10. Reziliență și Fail-Safe

| Mecanism | Implementare | Scop |
|---|---|---|
| **JSON Repair** | `_repair_json()` cu backtracking | HuggingFace trunchiază output-ul |
| **MD5 Cache** | hash(prompt) → `data/cache/*.json` | Zero costuri pentru re-rulări |
| **Retry exponential** | 5 încercări, sleep(5·attempt) | 429/503 recoverable |
| **Rate Limiter** | Sliding window deque RPM+RPD | Free tier Gemini/HF |
| **Domain/Range Guard** | `ingest_properties()` skip cu warning | Proprietăți cu clase inexistente |
| **Partial Save** | Crash → save_state + exit graceful | Nu se pierde progresul |
| **BFO Fuzzy Fix** | `_fix_bfo_parents()` normalizare | LLM scrie "MaterialEntity" în loc de "bfo:MaterialEntity" |
| **Cycle Auto-Remove** | DFS înainte de validare LLM | Eficiență: fără apeluri LLM pentru cicluri triviale |
