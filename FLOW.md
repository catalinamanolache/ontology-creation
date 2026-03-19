# Ontology Pipeline — Technical Flow

> Descriere pas-cu-pas a fiecărei decizii tehnice și a fiecărui apel de funcție în pipeline-ul AbA.

---

## Principii de Design

| Principiu | Decizie tehnică | Motivare |
|---|---|---|
| **T-Box / A-Box Separation** | Faza 2 generează schema, Faza 3 populează instanțele | Faza 3 nu poate inventa clase noi → halucination suppression prin Closed World |
| **Serialized Prompting** | CQ → Clase → Proprietăți (nu simultan) | LLM-ul nu poate evalua domain/range dacă clasele nu sunt încă definite |
| **BFO Anchoring** | Fiecare clasă declarată cu `bfo_parent` | Previne erori categoriale masive (ex: `ex:Author subClassOf ex:Document`) |
| **Two-Way Validation** | Forward AND reverse test pentru subClassOf | Un singur test direcțional nu detectează inversiunile |
| **Delta Extraction** | Batch-urile 2+ returnează NUMAI elemente noi | Economie tokeni, previne trunchieri HuggingFace |
| **Neuro-Symbolic Validation** | Python L1 → rdflib L2 → owlready2 L3 | Degradare grațioasă: funcționează și fără Java |
| **MD5 Cache** | hash(prompt_content) → fișier JSON pe disc | Re-rulări gratuite în development |

---

## Faza 1 — Chunking Deterministic

**Fișier**: `src/document_processor.py`

### Pasul 1.1 — Încărcare document
```python
loader = PyPDFLoader(file_path)
document_pages = loader.load()
full_text = "".join(page.page_content + "\n" for page in document_pages)
```
**Decizie**: PyPDFLoader extrage text pagină cu pagină. Concatenarea adaugă `\n` între pagini pentru a păstra separarea paragrafelor.

### Pasul 1.2 — Tăiere deterministă
```python
if "REFERENCES" in full_text:
    full_text = full_text.split("REFERENCES")[0]
if "Acknowledgments" in full_text:
    full_text = full_text.split("Acknowledgments")[0]
```
**Decizie**: Referințele bibliografice și mulțumirile nu au valoare semantică pentru ontologie. Tăierea deterministă asigură că fiecare run procesează exact același text.

### Pasul 1.3 — Curățare regex
```python
text = re.sub(r'STM Journals \d{4}\..*', '', text)
text = re.sub(r'©\s*\d+', '', text)
text = re.sub(r'Volume \d+, Issue \d+', '', text)
text = re.sub(r'\s+', ' ', text)
```
**Decizie**: Metadatele dinamice (copyright, ISSN, numere de volum) variază între run-uri și ar polua chunk-urile. Strip-uirea lor garantează chunks identice.

### Pasul 1.4 — Splitting
```python
RecursiveCharacterTextSplitter(
    chunk_size=1500,
    chunk_overlap=250,
    separators=["\n\n", "\n", ".", " ", ""],
)
```
**Decizie `chunk_size=1500`**: Suficient de mare pentru a conține un paragraf complet de text academic, suficient de mic pentru a rămâne în fereastra de context a modelelor 7B.

**Decizie `chunk_overlap=250`**: ~17% suprapunere. Asigură că faptele care trec între două chunk-uri nu se pierd. Prea mult overlap → costuri inutile; prea puțin → pierdere context la granițe.

**Decizie separatori**: Ierarhie descendentă `\n\n → \n → . → space → ""`. RCTextSplitter încearcă să taie la separatorul cel mai de sus din ierarhie care produce un chunk de dimensiunea dorită.

**Output**: `chunks[]` salvate în `data/chunks/chunk_NNN.txt` pentru reproducibilitate și debugging.

---

## Faza 2 — Construcția T-Box (Decomposed AbA)

**Orchestrare**: `src/main.py` → `src/extraction.py`

**Batching**:
```python
BATCH_SIZE = 3
batches = ["\n\n".join(chunks[i:i+3]) for i in range(0, len(chunks), 3)]
```
**Decizie `BATCH_SIZE=3`**: Trimitem 3 chunk-uri concatenate (~4500 caractere) per apel T-Box. Suficient context pentru a identifica clase majore fără a depăși fereastra de context sau a genera prea mult JSON de repararat.

---

### Sub-faza 2.1 — Generare Competency Questions

**Funcție**: `extractor.generate_competency_questions(batch_text)`

**Prompt**: `PROMPT_CQ_GENERATION`

**Schema**: `CQResult(questions: List[CompetencyQuestion(question, scope)])`

**Exemplu output**:
```json
{
  "questions": [
    {"question": "What proteins are involved in DNA replication?", "scope": "class"},
    {"question": "What is the substrate of Polymerase I?", "scope": "property"},
    {"question": "Which organisms express RecA?", "scope": "instance"}
  ]
}
```

**Decizie**: CQ-urile nu sunt folosite ca filtre hard — sunt injectate în prompturile de extracție ca "ghid de atenție" pentru LLM. Scopul (`class`/`property`/`instance`) ajută LLM-ul să știe dacă o întrebare indică o clasă nouă sau o proprietate nouă.

**Cache**: Cheie = MD5(`batch_text`). O re-rulare pe același document nu re-generează CQ-urile.

---

### Sub-faza 2.2 — Extracție Clase cu Ancorare BFO

**Funcții**:
- Batch 0: `extractor.extract_classes_initial(text, cqs_str)`
- Batch 1+: `extractor.extract_classes_extend(text, existing_classes_str, cqs_str)`

**Prompt** (injecții critice):
```
### BFO ONTOLOGY SKELETON
Every domain class MUST be anchored under exactly ONE BFO leaf:
  - bfo:MaterialEntity: Physical objects with matter
  - bfo:InformationContentEntity: Documents, data, records
  - bfo:Process: Events and activities over time
  - bfo:Role: Social/institutional context
  ...

### ARISTOTELIAN DEFINITION (format strict):
"A [ClassName] is a [BFO Parent] that [differentia]"

### DELTA RULE (batch extend):
Return ONLY NEW classes. If concept already covered, add to canonical_terms instead.
```

**Schema returnată**:
```python
ClassExtractionResult(
    classes=[
        OntologyClassBFO(
            uri="ex:Protein",
            bfo_parent="bfo:MaterialEntity",
            subclass_of=None,
            aristotelian_definition="A Protein is a MaterialEntity that is a macromolecule formed from amino acids.",
            comment="Biological macromolecule."
        )
    ],
    canonical_terms={"proteins": "ex:Protein", "polypeptide": "ex:Protein"},
    reasoning_steps=["Identified Protein as a physical molecule."]
)
```

**Post-processing `_fix_bfo_parents()`**:
```python
# LLM scrie "MaterialEntity" în loc de "bfo:MaterialEntity" → normalizare
bfo_lower = {c.lower(): c for c in BFO_LEAF_CATEGORIES}
normalized = cls.bfo_parent.lower().replace(" ", "")
if normalized in bfo_lower:
    cls.bfo_parent = bfo_lower[normalized]
else:
    cls.bfo_parent = "bfo:Entity"  # fallback sigur
```

**Ingestie**: `state.ingest_classes(result)`
- Stochează fiecare clasă ca `dict{comment, bfo_parent, aristotelian_definition, subclass_of}`
- Înregistrează `subclass_of` în `state.subclass_relations[child] = parent`

---

### Sub-faza 2.4 — Extracție Proprietăți (domain/range constrained)

**Funcții**:
- Batch 0: `extractor.extract_properties_initial(text, classes_str, cqs_str)`
- Batch 1+: `extractor.extract_properties_extend(text, existing_props_str, classes_str, cqs_str)`

**Constrângere critică în prompt**:
```
### DOMAIN AND RANGE CONSTRAINT:
Both domain and range MUST reference a class from AVAILABLE CLASSES above,
OR an XSD datatype. If a class doesn't exist, do NOT create that property.
```

**Decizie**: Prin injectarea listei de clase aprobate direct în prompt, LLM-ul nu poate inventa `domain: ex:UnknownClass`. Este cea mai importantă diferență față de abordarea monolith.

**Ingestie cu validare Python**:
```python
def ingest_properties(self, result):
    for prop in result.properties:
        if prop.domain not in class_uris and prop.domain not in allowed_special:
            print(f"  [WARN] Property '{prop.uri}' skipped: domain '{prop.domain}' not in approved classes.")
            continue  # ← Proprietatea e ignorată silențios
        # ... similar pentru range
        self.approved_properties[prop.uri] = {domain, range, comment}
```

**Decizie**: Validarea Python este un al doilea strat de apărare după constrângerea din prompt. Dacă LLM-ul tot halucinează un domain inexistent, proprietatea nu intră niciodată în blueprint.

---

### Sub-faza 2.5 — Validare Ierarhie Two-Way

**Se execută după toate batch-urile**, o singură dată pe totalitatea relațiilor `subClassOf`.

#### Etapa 2.5.A — Detecție cicluri Python (DFS)

```python
def detect_hierarchy_cycles(self) -> List[List[str]]:
    WHITE, GRAY, BLACK = 0, 1, 2
    color = {n: WHITE for n in all_nodes}

    def dfs(node, path):
        color[node] = GRAY
        path.append(node)
        parent = self.subclass_relations.get(node)
        if parent:
            if color.get(parent) == GRAY:  # ← ciclu detectat
                cycle_start = path.index(parent)
                cycles.append(path[cycle_start:] + [parent])
            elif color.get(parent) == WHITE:
                dfs(parent, path)
        path.pop()
        color[node] = BLACK
```

**Decizie**: DFS cu colorare tri-stare (white/gray/black). Nodurile gray sunt în recursie activă — dacă găsim un arc spre un nod gray, am găsit un ciclu. Complexitate O(V+E).

**Auto-remediere**: Ultimul arc din ciclu este eliminat automat (nu necesită LLM → zero costuri).

#### Etapa 2.5.B — Validare LLM Bidirecțională (batch)

```python
relations = state.get_subclass_relations_for_validation()
# → [{"child": "ex:ResearchPaper", "parent": "ex:Document"}, ...]

validation_result = extractor.validate_hierarchy_batch(relations)
```

**Prompt injectat**:
```
For "A rdfs:subClassOf B":
  FORWARD: "Is every A necessarily a B?" → must be YES
  REVERSE: "Is every B necessarily an A?" → must be NO (else they're equivalent)
```

**Logica de respingere**:
```python
for v in validation_result.validations:
    if not v.forward_valid or v.reverse_valid:
        state.remove_subclass(v.child_uri, v.parent_uri)
```

**Decizie batch vs. individual**: Trimitem TOATE relațiile într-un singur prompt în loc de câte un prompt per relație. Economisim N-1 apeluri API. Dezavantaj: dacă promptul e prea lung, LLM-ul poate trunchia. Soluție: pentru documente cu >50 relații subClassOf, se poate implementa chunking de relații.

---

### Sub-faza 2.6 — OWL Validator + Self-Correction Loop

#### `OWLValidator.validate()` — 3 straturi

**Layer 1 — Python Heuristics** (mereu rulează):

```python
def _validate_naming_conventions(classes, properties):
    # ex:protein → WARNING (nu TitleCase)
    # ex:Protein Name → ERROR (spațiu în URI)

def _validate_domain_range_references(classes, properties):
    # ex:hasAuthor domain=ex:UnknownClass → ERROR

def _detect_hierarchy_cycles(subclass_relations):
    # A→B→A → ERROR

def _validate_property_conflicts(properties):
    # >3 proprietăți cu același (domain, range) → WARNING
```

**Layer 2 — rdflib** (mereu rulează, dep. existentă):

```python
g = Graph()
# Construiește graf RDF din stare
g.serialize(format="turtle")  # dacă eșuează → ERROR de graf malformat

# SPARQL: proprietăți cu domain = subclasă a range-ului (inversiune logică)
query = """
SELECT ?prop ?domain ?range WHERE {
    ?prop rdfs:domain ?domain .
    ?prop rdfs:range ?range .
    ?domain rdfs:subClassOf ?range .  # ← domenul e mai specific decât range-ul
}
"""
```

**Layer 3 — owlready2 + HermiT** (opțional, necesită Java):

```python
onto = owlready2.get_ontology(EX_NS)
with onto:
    # Creează clase și proprietăți dinamice
    ...
owlready2.sync_reasoner(infer_property_values=False)
# Verifică clase echivalente cu owl:Nothing (nesatisfiabile)
for cls in onto.classes():
    if owlready2.Nothing in cls.equivalent_to:
        errors.append(f"Class '{cls.name}' is unsatisfiable.")
```

**Decizie fallback Java**: Dacă JVM nu e disponibil, eroarea e capturată, adăugată ca WARNING (nu ERROR), iar pipeline-ul continuă. HermiT este cel mai puternic dar nu blocant.

#### Self-Correction Loop

```python
for round_num in range(1, 4):  # max 3 runde
    report = validator.validate(classes, properties, subclass_relations)

    if report.is_consistent:
        break  # gata, ontologia e consistentă

    error_log = format_errors_for_prompt(report)
    correction = extractor.self_correct(ontology_str, error_log)
    state.apply_corrections(correction)
    # → re-validare în runda următoare
```

**Promptul de self-correction**:
```
### ERRORS DETECTED BY REASONER
ERROR 1: Property 'ex:hasCapital' has domain 'ex:UnknownCity' which is not a known class.
ERROR 2: Hierarchy cycle detected: ex:A -> ex:B -> ex:A

### INSTRUCTIONS
For each error, propose exactly one correction. Available actions:
- "remove_class", "remove_property", "update_domain", "update_range", "remove_subclass"
```

**Acțiuni aplicate în `state.apply_corrections()`**:
- `remove_class` → șterge clasa + toate proprietățile care o referențiază (cascade) + relațiile subClassOf
- `remove_property` → șterge proprietatea
- `update_domain` / `update_range` → modifică valoarea
- `remove_subclass` → elimină relația din `subclass_relations` și resetează `subclass_of=None` în clasă

---

## Faza 3 — Populare A-Box (Knowledge Graph)

**Orchestrare**: `src/main.py` → per chunk individual (nu batch)

**Decizie procesare chunk-cu-chunk**: Spre deosebire de Faza 2 (batch-uri de 3), Faza 3 procesează câte un chunk. Motivul: entitățile extrase din chunk-ul N sunt injectate în promptul chunk-ului N+1 ca `existing_entities`, oferind context pentru deduplicare. Dacă am batch-ui și Faza 3, pierdem această propagare.

### Pasul 3.1 — Construire prompt
```python
prompt = SYSTEM_PROMPT_EXTRACTION.format(
    approved_ontology=state.format_ontology_for_prompt(),   # clase + proprietăți
    canonical_terms=state.format_canonical_terms_for_prompt(),  # sinonime → URI
    existing_entities=state.format_entities_for_prompt(),   # entități deja extrase
    chunk_text=chunk,
)
```

### Pasul 3.2 — Extracție LLM → `ExtractionResult`
```python
ExtractionResult(
    reasoning_steps=["Found two entities: protein X and gene Y."],
    entities=[
        Entity(id="protein_x", class_uri="ex:Protein", evidence_span="protein X"),
        Entity(id="gene_y", class_uri="ex:Gene", evidence_span="gene Y"),
    ],
    relations=[
        Relation(
            source_id="gene_y",
            property_uri="ex:encodes",
            target_id="protein_x",
            evidence_span="gene Y encodes protein X"
        )
    ]
)
```

**Constrângere Closed World** în prompt:
```
CLOSED WORLD ASSUMPTION: If a fact is in the text but its Class or Property is NOT in
approved_ontology, you MUST ignore it. Do not invent schema elements.
```

### Pasul 3.3 — Validare și Deduplicare

```python
for entity in result.entities:
    if not state.validate_entity(entity):   # class_uri ∉ approved_classes
        skipped += 1
        continue

    canonical_id = state.add_entity(entity)
    id_map[entity.id] = canonical_id  # ← remap pentru relații
```

**Fuzzy deduplicare în `add_entity()`**:
```python
normalized = _normalize_id(entity.id)
# normalize: lowercase, remove _ și -, remove trailing s

# 1. Match exact pe normalized index
if normalized in self._entity_normalized_index:
    return self._entity_normalized_index[normalized]  # deja există

# 2. Jaccard bigrams (threshold 85%)
for existing_norm, existing_id in self._entity_normalized_index.items():
    if _is_similar(normalized, existing_norm, 0.85):
        return existing_id  # același concept, alt ID → reutilizăm

# 3. Entitate nouă
self.entities[entity.id] = {class_uri, evidence_span}
self._entity_normalized_index[normalized] = entity.id
```

**Decizie bigrams**: Comparăm perechile de caractere consecutive (bigramele). `protein_x` și `proteinx` au ~90% bigrams comune. Jaccard pe bigramele mulțimii este mai robust decât Levenshtein pentru ID-uri cu underscore/spații.

```python
def bigrams(s):
    return set(s[i:i+2] for i in range(len(s)-1))

intersection = len(ba & bb)
union = len(ba | bb)
similarity = intersection / union
```

**Decizie threshold 85%**: Pragul de 70% produce prea multe false pozitive (ex: `enzyme` și `enzyme_complex` sunt diferite dar share mulți bigrami). 90% e prea strict (nu unică variante de genul `john_smith` vs `johnsmith`). 85% e empiric optim pentru ID-uri snake_case.

---

## Faza 4 — Output și Persistență

### Export Turtle (.ttl)

```python
# Clasă cu BFO și definiție aristotelică
ex:Document rdf:type owl:Class ;
    rdfs:subClassOf bfo:InformationContentEntity ;        # ← BFO parent
    rdfs:subClassOf ex:Artifact ;                         # ← subclass_of domeniu
    skos:definition "A Document is an InformationContentEntity that records..." ;
    rdfs:comment "A document artifact." .

# Proprietate cu tip detectat automat
ex:hasAuthor rdf:type owl:ObjectProperty ;    # range e clasă → ObjectProperty
    rdfs:domain ex:Document ;
    rdfs:range ex:Author ;
    rdfs:comment "Links a document to its author." .

ex:pageCount rdf:type owl:DatatypeProperty ;  # range e xsd:integer → DatatypeProperty
    rdfs:domain ex:Document ;
    rdfs:range xsd:integer ;
    rdfs:comment "Number of pages." .
```

**Decizie ObjectProperty vs DatatypeProperty**: Detectăm automat — dacă `range.startswith("xsd:")` → `DatatypeProperty`, altfel `ObjectProperty`. Aceasta este o îmbunătățire față de versiunea anterioară care tipiza totul ca ObjectProperty.

### Archivare run

```python
data/runs/run_N/
    ├── ontology_blueprint.json   # schema pură (fără A-Box)
    ├── ontology_state.json       # stare completă (T-Box + A-Box)
    └── ontology_final.ttl        # export RDF complet
```

**Decizie index numeric**: `run_1`, `run_2` ... și nu timestamp, pentru că `final_report.py` face comparații secvențiale și trebuie să sorteze numeric corect (`run_10` > `run_9`, nu `run_10` < `run_9` cum ar da sort lexicografic).

---

## Faza 5 — Evaluare Jaccard

**Fișier**: `src/final_report.py`

### Sortare corectă

```python
runs.sort(key=lambda x: int(x.split('_')[1]) if x.split('_')[1].isdigit() else 0)
# → [run_1, run_2, run_3, ..., run_10]  (nu run_1, run_10, run_2)
```

### Calcul similaritate

```python
def jaccard(set1, set2):
    if not set1 and not set2:
        return 1.0  # două seturi goale sunt identice
    return len(set1 & set2) / len(set1 | set2)

sim_classes = jaccard(classesA, classesB)    # pe seturi de URI-uri
sim_props   = jaccard(propsA, propsB)
sim_edges   = jaccard(edgesA, edgesB)        # triple (source, prop, target)
```

### Prag de stabilitate

```python
if sim_edges >= 0.7 and sim_classes >= 0.7:
    print("✅ STABLE evolution.")
else:
    print("⚠️ EVOLVING/DIVERGENT schema.")
```

**Decizie τ=0.70**: Pragul de 70% este standard în literatura de ontology learning (Maedche & Staab 2001, OntoEval). Sub 70% → schema s-a schimbat semnificativ între run-uri, probabil din cauza instabilității LLM sau a modificării documentului.

---

## Core: `_call_with_retry_and_cache()` — Anatomie

```
_call_with_retry_and_cache(prefix, model_class, prompt, **cache_kwargs)
    │
    ├── 1. Construiește cheia de cache
    │     payload = {prompt, kwargs}
    │     h = MD5(json.dumps(payload))
    │     cache_path = data/cache/{prefix}_{h}.json
    │
    ├── 2. Cache hit?
    │     YES → model_class.model_validate_json(file_content) → return
    │
    ├── 3. Rate limiter (Gemini/HuggingFace)
    │     rate_limiter.wait_if_needed()
    │
    ├── 4. Apel LLM (retry loop, max 5 încercări)
    │     ├── HuggingFace:
    │     │     response = llm.invoke(prompt)
    │     │     raw_text = response.content
    │     │     json_str = extrage din ```json ... ``` sau direct
    │     │     parsed = json.loads(json_str) sau _repair_json(json_str)
    │     │     normalized = _normalize_hf_json(parsed, model_class)
    │     │     return model_class.model_validate(normalized)
    │     │
    │     └── Ollama / Gemini:
    │           structured_llm = llm.with_structured_output(model_class)
    │           return structured_llm.invoke(prompt)
    │
    ├── 5. Excepție?
    │     ├── 429/RESOURCE_EXHAUSTED → sleep(60s) → retry
    │     ├── 503/overloaded → sleep(30s) → retry
    │     └── altele → sleep(5·attempt) → retry → ultimul attempt → raise
    │
    └── 6. Salvare cache + return result
```

---

## `_normalize_hf_json()` — Normalizare HuggingFace

LLM-urile locale returnează adesea chei cu nume alternative. Normalizarea acoperă toate schemele:

```python
# Comun: reasoning_steps
if "reasoning_steps" not in parsed:
    for alt in ["thought_process", "reasoning", "thoughts"]:
        if alt in parsed:
            parsed["reasoning_steps"] = parsed.pop(alt)
if isinstance(parsed.get("reasoning_steps"), str):
    parsed["reasoning_steps"] = [parsed["reasoning_steps"]]

# ClassExtractionResult
"approved_classes" → "classes"
classes: {"ex:Protein": "comment"} → [{"uri": "ex:Protein", "comment": "...", "bfo_parent": "bfo:Entity", ...}]

# PropertyExtractionResult
"approved_properties" → "properties"
properties: {"ex:hasAuthor": {"domain": ..., "range": ...}} → [{"uri": ..., ...}]

# HierarchyValidationResult
"results" / "validation_results" → "validations"

# CorrectionResult
"fixes" / "changes" → "corrections"
```

**Decizie**: Normalizarea e mai bună decât să adăugăm exemple JSON în prompt (care consumă tokeni) sau să folosim `strict=False` la Pydantic (care acceptă orice și poate genera date incorecte silențios).

---

## JSON Repair — Algoritm Detaliat

```
Input: json_str (potențial trunchiat)
    │
    ├── Pasul 1: Scan caracter cu caracter
    │     Tracking: in_string, escaped, stack = []
    │     La '{' → push '}'
    │     La '[' → push ']'
    │     La '"' → toggle in_string
    │
    ├── Pasul 2: String deschis?
    │     in_string == True → append '"'
    │
    └── Pasul 3: Backtracking loop
          Încearcă să închidă structura și parseze:
          temp = current_json + "".join(reversed(stack))
          json.loads(temp) → SUCCESS → return
          FAIL → remove last char → repeat
          (Complexitate: O(N) în cazul cel mai rău)
```

**Decizie**: Backtracking-ul se oprește când găsește cel mai lung prefix valid. Aceasta înseamnă că un JSON trunchiat la jumătatea unei liste va fi reparat cu lista parțială, nu cu o listă goală — maximizăm datele recuperate.

---

## Rate Limiter — Sliding Window

```
deque = [t1, t2, t3, ..., t_N]  (timestamps ultimele 60s)

wait_if_needed():
    ├── RPD check: daily_count >= max_rpd → raise RuntimeError
    ├── RPM check: len(deque) >= max_rpm
    │     oldest = deque[0]
    │     wait = 60 - (now - oldest) + 1s safety margin
    │     sleep(wait)
    └── Min delay: now - deque[-1] < min_delay → sleep(min_delay - elapsed)

record_request():
    deque.append(now)
    daily_count += 1
    save_state()  # persistat pe disc → supraviețuiește restart-urilor
```

**Decizie persistență pe disc**: Counter-ul zilnic este salvat în `data/rate_limit_state.json`. Dacă pipeline-ul e oprit și restartat în aceeași zi, counter-ul nu se resetează și nu vom depăși limita RPD a free tier-ului.

---

## Diagrama Fluxului de Date

```
PDF/TXT
  │
  ▼ DocumentProcessor
chunks[] ─────────────────────────────────────────────────────┐
  │                                                            │
  ├─ [Faza 2, per batch de 3]                                 │
  │    │                                                       │
  │    ├─ CQResult ──────────────────────────┐                │
  │    │                                     │                │
  │    ├─ ClassExtractionResult ─► state     │                │
  │    │   (BFO, AristotelianDef, subclass)  │                │
  │    │                                     ▼                │
  │    └─ PropertyExtractionResult ──► state (uses classes)   │
  │                                                            │
  ├─ [Post-batch]                                             │
  │    ├─ detect_hierarchy_cycles() ─► remove_subclass()      │
  │    └─ validate_hierarchy_batch() ─► remove_subclass()     │
  │                                                            │
  ├─ [OWL Validation, max 3x]                                 │
  │    ├─ OWLValidator.validate() ─► ValidationReport         │
  │    └─ self_correct() ─► apply_corrections()               │
  │                                                            │
  ├─ BLUEPRINT FROZEN                                         │
  │                                                            │
  └─ [Faza 3, per chunk] ◄──────────────────────────────────┘
       │
       ├─ ExtractionResult ─► validate_entity()
       │                    ─► add_entity() [fuzzy dedup]
       │                    ─► add_relation() [id_map remap]
       │
       └─ state.entities + state.relations
                │
                ▼
         ontology_final.ttl
         ontology_state.json
         data/runs/run_N/
```
