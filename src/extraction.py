"""
OntologyExtractor — Decomposed Axiom-by-Axiom (AbA) LLM Pipeline.

Replaces monolithic T-Box extraction with serialized phases:
  2.1  Competency Questions
  2.2  Class Extraction + BFO anchoring + Aristotelian definitions
  2.4  Property Extraction (domain/range as separate step)
  2.5  Two-Way Hierarchy Validation (batch)
  2.6  Self-Correction from Reasoner errors

Supports three backends:
  - "ollama"      → local Ollama server
  - "gemini"      → Google Gemini API
  - "huggingface" → Hugging Face Inference API or local transformers

Preserves: MD5 caching, exponential backoff retry, JSON repair.
"""

import time
import os
import json
import re
import hashlib
from typing import List, Optional

from config.settings import settings
from src.schemas import (
    CQResult,
    ClassExtractionResult,
    PropertyExtractionResult,
    HierarchyValidationResult,
    CorrectionResult,
    ExtractionResult,
    SeedOntologyResult,
)
from src.prompts import (
    PROMPT_CQ_GENERATION,
    PROMPT_CLASS_EXTRACTION_INITIAL,
    PROMPT_CLASS_EXTENSION,
    PROMPT_PROPERTY_EXTRACTION_INITIAL,
    PROMPT_PROPERTY_EXTENSION,
    PROMPT_TWO_WAY_VALIDATION,
    PROMPT_SELF_CORRECTION,
    SYSTEM_PROMPT_EXTRACTION,
)
from src.bfo_skeleton import BFO_PROMPT_SKELETON, is_valid_bfo_parent, BFO_LEAF_CATEGORIES

CACHE_DIR = "data/cache"
os.makedirs(CACHE_DIR, exist_ok=True)


# ────────────────────────────────────────────────────────────────────
# JSON Repair (preserved from original — still needed as fail-safe)
# ────────────────────────────────────────────────────────────────────

def _repair_json(json_str: str) -> str:
    """
    Attempt to repair truncated JSON by:
    1. Closing an open string.
    2. Trimming back partial keys or values until a safe break point.
    3. Balancing all open braces and brackets.
    """
    json_str = json_str.strip()
    if not json_str:
        return ""

    # 1. Handle open string
    stack = []
    in_string = False
    escaped = False

    cleaned_chars = []
    for char in json_str:
        if char == '"' and not escaped:
            in_string = not in_string
        if in_string:
            if char == '\\':
                escaped = not escaped
            else:
                escaped = False
        else:
            if char == '{':
                stack.append('}')
            elif char == '[':
                stack.append(']')
            elif char == '}':
                if stack and stack[-1] == '}':
                    stack.pop()
            elif char == ']':
                if stack and stack[-1] == ']':
                    stack.pop()
        cleaned_chars.append(char)

    json_str = "".join(cleaned_chars)
    if in_string:
        json_str += '"'

    # 2. Trim trailing garbage (partial keys/fields) and try to close
    while len(json_str) > 0:
        try:
            temp_str = json_str
            # Recount brackets for closure
            s = []
            in_s = False
            for c in temp_str:
                if c == '"':
                    in_s = not in_s
                if not in_s:
                    if c == '{':
                        s.append('}')
                    elif c == '[':
                        s.append(']')
                    elif c == '}':
                        if s and s[-1] == '}':
                            s.pop()
                    elif c == ']':
                        if s and s[-1] == ']':
                            s.pop()
            while s:
                temp_str += s.pop()
            json.loads(temp_str)
            return temp_str
        except Exception:
            json_str = json_str[:-1].rstrip()
            if not json_str:
                break

    return json_str


# ────────────────────────────────────────────────────────────────────
# LLM Backend Factory (preserved — multi-backend support)
# ────────────────────────────────────────────────────────────────────

def _create_llm():
    """Create the appropriate LLM based on LLM_BACKEND setting."""
    backend = getattr(settings, "LLM_BACKEND", "ollama").lower()

    if backend == "ollama":
        try:
            from langchain_ollama import ChatOllama
        except ImportError:
            raise ImportError(
                "langchain-ollama not installed. Run: pip install langchain-ollama"
            )

        model = getattr(settings, "OLLAMA_MODEL", "qwen2.5:7b")
        base_url = getattr(settings, "OLLAMA_BASE_URL", "http://localhost:11434")
        print(f"  [LLM] Using Ollama local model: {model}")
        print(f"  [LLM] Base URL: {base_url}")

        return ChatOllama(
            model=model,
            base_url=base_url,
            temperature=settings.TEMPERATURE,
            top_p=settings.TOP_P,
            seed=settings.SEED,
            num_predict=4096,
        )

    elif backend == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI

        if not settings.GOOGLE_API_KEY:
            raise ValueError("Missing GOOGLE_API_KEY in .env")

        print(f"  [LLM] Using Google Gemini: {settings.MODEL_NAME}")
        return ChatGoogleGenerativeAI(
            model=settings.MODEL_NAME,
            google_api_key=settings.GOOGLE_API_KEY,
            temperature=settings.TEMPERATURE,
            top_p=settings.TOP_P,
            seed=settings.SEED,
            convert_system_message_to_human=True,
        )

    elif backend == "huggingface":
        from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

        if not settings.HUGGINGFACE_API_KEY:
            raise ValueError("Missing HUGGINGFACE_API_KEY in .env")

        print(f"  [LLM] Using Hugging Face model: {settings.HUGGINGFACE_MODEL}")

        if settings.HF_USE_API:
            llm = HuggingFaceEndpoint(
                repo_id=settings.HUGGINGFACE_MODEL,
                huggingfacehub_api_token=settings.HUGGINGFACE_API_KEY,
                temperature=settings.TEMPERATURE,
                top_p=settings.TOP_P,
                max_new_tokens=4096,
            )
        else:
            from langchain_huggingface import HuggingFacePipeline
            llm = HuggingFacePipeline.from_model_id(
                model_id=settings.HUGGINGFACE_MODEL,
                task="text-generation",
                pipeline_kwargs={
                    "max_new_tokens": 4096,
                    "temperature": settings.TEMPERATURE,
                    "top_p": settings.TOP_P,
                },
            )

        return ChatHuggingFace(llm=llm)

    else:
        raise ValueError(f"Unknown LLM_BACKEND: {backend}. Use 'ollama', 'gemini', or 'huggingface'.")


# ────────────────────────────────────────────────────────────────────
# HuggingFace JSON Normalization (generalized for all schema types)
# ────────────────────────────────────────────────────────────────────

def _normalize_hf_json(parsed_json: dict, model_class) -> dict:
    """
    Normalize raw JSON from HuggingFace responses for different schema types.
    Handles common LLM output variations (renamed keys, wrong types, etc.).
    """
    # ── Common: reasoning_steps normalization ──
    model_fields = set()
    if hasattr(model_class, "model_fields"):
        model_fields = set(model_class.model_fields.keys())
    elif hasattr(model_class, "__fields__"):
        model_fields = set(model_class.__fields__.keys())

    if "reasoning_steps" in model_fields:
        if "reasoning_steps" not in parsed_json:
            for alt_key in ["thought_process", "reasoning", "thoughts", "explanation"]:
                if alt_key in parsed_json:
                    parsed_json["reasoning_steps"] = parsed_json.pop(alt_key)
                    break
            else:
                parsed_json["reasoning_steps"] = ["Extracted via fallback."]
        if isinstance(parsed_json.get("reasoning_steps"), str):
            parsed_json["reasoning_steps"] = [parsed_json["reasoning_steps"]]

    # ── CQResult normalization ──
    if model_class == CQResult:
        if "questions" not in parsed_json:
            # Maybe the model returned a flat list
            if isinstance(parsed_json, list):
                parsed_json = {"questions": parsed_json}
            # Or used a different key
            for alt_key in ["competency_questions", "cqs", "cq_list"]:
                if alt_key in parsed_json:
                    parsed_json["questions"] = parsed_json.pop(alt_key)
                    break
        # Ensure each question has scope
        if "questions" in parsed_json:
            for q in parsed_json["questions"]:
                if isinstance(q, str):
                    q_idx = parsed_json["questions"].index(q)
                    parsed_json["questions"][q_idx] = {"question": q, "scope": "class"}
                elif isinstance(q, dict) and "scope" not in q:
                    q["scope"] = "class"

    # ── ClassExtractionResult normalization ──
    if model_class == ClassExtractionResult:
        if "approved_classes" in parsed_json and "classes" not in parsed_json:
            parsed_json["classes"] = parsed_json.pop("approved_classes")
        if "classes" in parsed_json and isinstance(parsed_json["classes"], dict):
            class_list = []
            for uri, comment in parsed_json["classes"].items():
                class_list.append({
                    "uri": uri,
                    "comment": str(comment),
                    "bfo_parent": "bfo:Entity",
                    "subclass_of": None,
                    "aristotelian_definition": f"A {uri.replace('ex:', '')} is an Entity.",
                })
            parsed_json["classes"] = class_list
        # Ensure BFO fields exist in each class
        if "classes" in parsed_json and isinstance(parsed_json["classes"], list):
            for cls in parsed_json["classes"]:
                if isinstance(cls, dict):
                    cls.setdefault("bfo_parent", "bfo:Entity")
                    cls.setdefault("subclass_of", None)
                    cls.setdefault("aristotelian_definition",
                                   f"A {cls.get('uri', '').replace('ex:', '')} is an Entity.")
                    cls.setdefault("comment", cls.get("aristotelian_definition", ""))
        if "canonical_terms" not in parsed_json:
            parsed_json["canonical_terms"] = {}

    # ── PropertyExtractionResult normalization ──
    if model_class == PropertyExtractionResult:
        if "approved_properties" in parsed_json and "properties" not in parsed_json:
            parsed_json["properties"] = parsed_json.pop("approved_properties")
        if "properties" in parsed_json and isinstance(parsed_json["properties"], dict):
            prop_list = []
            for uri, details in parsed_json["properties"].items():
                if isinstance(details, dict):
                    prop_list.append({
                        "uri": uri,
                        "domain": details.get("domain", "ex:Thing"),
                        "range": details.get("range", "ex:Thing"),
                        "comment": details.get("comment", ""),
                    })
            parsed_json["properties"] = prop_list

    # ── HierarchyValidationResult normalization ──
    if model_class == HierarchyValidationResult:
        if "validations" not in parsed_json:
            if isinstance(parsed_json, list):
                parsed_json = {"validations": parsed_json}
            for alt_key in ["results", "hierarchy_validations", "validation_results"]:
                if alt_key in parsed_json:
                    parsed_json["validations"] = parsed_json.pop(alt_key)
                    break

    # ── CorrectionResult normalization ──
    if model_class == CorrectionResult:
        if "corrections" not in parsed_json:
            for alt_key in ["fixes", "changes", "repair"]:
                if alt_key in parsed_json:
                    parsed_json["corrections"] = parsed_json.pop(alt_key)
                    break

    # ── Legacy SeedOntologyResult normalization ──
    if model_class == SeedOntologyResult:
        if "approved_classes" in parsed_json and "classes" not in parsed_json:
            parsed_json["classes"] = parsed_json.pop("approved_classes")
        if "approved_properties" in parsed_json and "properties" not in parsed_json:
            parsed_json["properties"] = parsed_json.pop("approved_properties")
        if "classes" in parsed_json and isinstance(parsed_json["classes"], dict):
            class_list = []
            for uri, comment in parsed_json["classes"].items():
                class_list.append({"uri": uri, "comment": str(comment)})
            parsed_json["classes"] = class_list
        if "properties" in parsed_json and isinstance(parsed_json["properties"], dict):
            prop_list = []
            for uri, details in parsed_json["properties"].items():
                if isinstance(details, dict):
                    prop_list.append({
                        "uri": uri,
                        "domain": details.get("domain", "ex:Thing"),
                        "range": details.get("range", "ex:Thing"),
                        "comment": details.get("comment", ""),
                    })
            parsed_json["properties"] = prop_list
        if "canonical_terms" not in parsed_json:
            parsed_json["canonical_terms"] = {}

    return parsed_json


# ────────────────────────────────────────────────────────────────────
# Main Extractor Class
# ────────────────────────────────────────────────────────────────────

class OntologyExtractor:
    def __init__(self, rate_limiter=None):
        self.llm = _create_llm()
        self.rate_limiter = rate_limiter
        self.backend = getattr(settings, "LLM_BACKEND", "ollama").lower()

    # ================================================================
    # Core LLM Call (cache + retry + JSON repair)
    # ================================================================

    def _call_with_retry_and_cache(self, prefix: str, model_class, prompt: str, **cache_kwargs):
        """
        Core LLM call with:
        1. MD5-based disk cache
        2. Rate limiting (cloud backends)
        3. Exponential backoff retry
        4. JSON repair for truncated responses
        5. Schema-aware normalization for HuggingFace
        """
        # 1. Check cache
        if settings.USE_CACHE:
            payload = {"prompt": prompt, "kwargs": cache_kwargs}
            cache_content = json.dumps(payload, sort_keys=True, ensure_ascii=False)
            h = hashlib.md5(cache_content.encode("utf-8")).hexdigest()
            cache_path = os.path.join(CACHE_DIR, f"{prefix}_{h}.json")

            if os.path.exists(cache_path):
                print(f"  [CACHE] Loading '{prefix}' result from cache...")
                with open(cache_path, "r", encoding="utf-8") as f:
                    content = f.read()
                try:
                    return model_class.model_validate_json(content)
                except AttributeError:
                    return model_class.parse_raw(content)
        else:
            print(f"  [LLM] Cache is disabled. Making live call...")

        # 2. Rate limit check (cloud APIs)
        if self.rate_limiter and self.backend in ("gemini", "huggingface"):
            self.rate_limiter.wait_if_needed()
            self.rate_limiter.print_status()

        # 3. API call with retry
        max_retries = 5
        for attempt in range(max_retries):
            try:
                print(f"\n  [API] Calling '{prefix}' (attempt {attempt+1}/{max_retries})...")

                if self.backend == "huggingface":
                    result = self._call_huggingface(prompt, model_class)
                else:
                    structured_llm = self.llm.with_structured_output(model_class)
                    result = structured_llm.invoke(prompt)

                # Record request for rate limiting
                if self.rate_limiter and self.backend in ("gemini", "huggingface"):
                    self.rate_limiter.record_request()

                # Save to cache
                if settings.USE_CACHE:
                    try:
                        data_str = result.model_dump_json(indent=2)
                    except AttributeError:
                        data_str = result.json(indent=2)

                    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
                    with open(cache_path, "w", encoding="utf-8") as f:
                        f.write(data_str)

                # Log success
                item_count = len(
                    getattr(result, "entities", [])
                    or getattr(result, "classes", [])
                    or getattr(result, "properties", [])
                    or getattr(result, "questions", [])
                    or getattr(result, "validations", [])
                    or getattr(result, "corrections", [])
                    or []
                )
                print(f"  [API] Success! Got {item_count} items.")
                return result

            except Exception as e:
                err_str = str(e)
                if "429" in err_str or "Quota" in err_str or "RESOURCE_EXHAUSTED" in err_str:
                    wait_time = 60
                    print(f"  [RATE LIMIT] 429 error. Waiting {wait_time}s...")
                    time.sleep(wait_time)
                elif "503" in err_str or "overloaded" in err_str.lower():
                    wait_time = 30
                    print(f"  [SERVER] 503 error. Waiting {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    print(f"  [ERROR] {type(e).__name__}: {err_str[:300]}")
                    if attempt == max_retries - 1:
                        raise
                    time.sleep(5 * (attempt + 1))

        raise Exception(f"API call '{prefix}' failed after {max_retries} retries.")

    def _call_huggingface(self, prompt: str, model_class):
        """
        Handle HuggingFace backend: raw text → JSON parse → normalization → validation.
        Extracted for clarity; contains the full JSON repair pipeline.
        """
        response = self.llm.invoke(prompt)
        raw_text = response.content if hasattr(response, "content") else str(response)

        # Extract JSON from response
        json_str = raw_text.strip()
        if "```json" in json_str:
            json_str = json_str.split("```json")[1].split("```")[0].strip()
        elif "```" in json_str:
            json_str = json_str.split("```")[1].split("```")[0].strip()

        # Try direct parse
        try:
            parsed_json = json.loads(json_str)
        except json.JSONDecodeError:
            print("  [WARN] JSON truncated. Attempting repair...")
            repaired = _repair_json(json_str)
            try:
                parsed_json = json.loads(repaired)
            except json.JSONDecodeError:
                # Last resort: find outermost { ... }
                match = re.search(r"(\{.*\})", json_str, re.DOTALL)
                if match:
                    parsed_json = json.loads(_repair_json(match.group(1)))
                else:
                    raise

        # Normalize for schema
        parsed_json = _normalize_hf_json(parsed_json, model_class)

        # Validate with Pydantic
        try:
            return model_class.model_validate(parsed_json)
        except Exception:
            # Final fallback: try model_validate_json on the raw string
            return model_class.model_validate_json(json.dumps(parsed_json))

    # ================================================================
    # Phase 2.1 — Competency Questions
    # ================================================================

    def generate_competency_questions(self, text: str) -> CQResult:
        """Generate Competency Questions from source text to guide extraction."""
        prompt = PROMPT_CQ_GENERATION.format(document_text=text)
        return self._call_with_retry_and_cache(
            "cq_gen", CQResult, prompt, text=text
        )

    # ================================================================
    # Phase 2.2 — Class Extraction with BFO Anchoring
    # ================================================================

    def extract_classes_initial(self, text: str, cqs_str: str) -> ClassExtractionResult:
        """Extract initial classes from text, anchored to BFO categories."""
        prompt = PROMPT_CLASS_EXTRACTION_INITIAL.format(
            document_text=text,
            bfo_skeleton=BFO_PROMPT_SKELETON,
            competency_questions=cqs_str,
        )
        result = self._call_with_retry_and_cache(
            "class_initial", ClassExtractionResult, prompt,
            text=text, cqs=cqs_str,
        )
        # Post-process: validate BFO parents
        self._fix_bfo_parents(result)
        return result

    def extract_classes_extend(
        self, text: str, existing_classes_str: str, cqs_str: str
    ) -> ClassExtractionResult:
        """Extend existing classes with delta from new text."""
        prompt = PROMPT_CLASS_EXTENSION.format(
            document_text=text,
            bfo_skeleton=BFO_PROMPT_SKELETON,
            existing_classes=existing_classes_str,
            competency_questions=cqs_str,
        )
        result = self._call_with_retry_and_cache(
            "class_extend", ClassExtractionResult, prompt,
            text=text, existing=existing_classes_str, cqs=cqs_str,
        )
        self._fix_bfo_parents(result)
        return result

    @staticmethod
    def _fix_bfo_parents(result: ClassExtractionResult):
        """Ensure all BFO parents are valid; default to bfo:Entity if not."""
        for cls in result.classes:
            if not is_valid_bfo_parent(cls.bfo_parent):
                # Try fuzzy match
                bfo_lower = {c.lower(): c for c in BFO_LEAF_CATEGORIES}
                normalized = cls.bfo_parent.lower().replace(" ", "").replace("_", "")
                if normalized in bfo_lower:
                    cls.bfo_parent = bfo_lower[normalized]
                else:
                    # Default fallback
                    cls.bfo_parent = "bfo:Entity"

    # ================================================================
    # Phase 2.4 — Property Extraction
    # ================================================================

    def extract_properties_initial(
        self, text: str, classes_str: str, cqs_str: str
    ) -> PropertyExtractionResult:
        """Extract initial properties, constrained to existing classes."""
        prompt = PROMPT_PROPERTY_EXTRACTION_INITIAL.format(
            document_text=text,
            existing_classes=classes_str,
            competency_questions=cqs_str,
        )
        return self._call_with_retry_and_cache(
            "prop_initial", PropertyExtractionResult, prompt,
            text=text, classes=classes_str, cqs=cqs_str,
        )

    def extract_properties_extend(
        self, text: str, existing_properties_str: str, classes_str: str, cqs_str: str
    ) -> PropertyExtractionResult:
        """Extend existing properties with delta from new text."""
        prompt = PROMPT_PROPERTY_EXTENSION.format(
            document_text=text,
            existing_properties=existing_properties_str,
            existing_classes=classes_str,
            competency_questions=cqs_str,
        )
        return self._call_with_retry_and_cache(
            "prop_extend", PropertyExtractionResult, prompt,
            text=text, existing_props=existing_properties_str,
            classes=classes_str, cqs=cqs_str,
        )

    # ================================================================
    # Phase 2.5 — Two-Way Hierarchy Validation (batch)
    # ================================================================

    def validate_hierarchy_batch(
        self, relations: List[dict]
    ) -> HierarchyValidationResult:
        """
        Batch-validate subClassOf relations using Two-Way Chain-of-Thought.

        Args:
            relations: List of {"child": "ex:X", "parent": "ex:Y"} dicts.

        Returns:
            HierarchyValidationResult with per-relation verdicts.
        """
        if not relations:
            return HierarchyValidationResult(validations=[])

        rel_str = json.dumps(relations, indent=2)
        prompt = PROMPT_TWO_WAY_VALIDATION.format(relationships=rel_str)
        return self._call_with_retry_and_cache(
            "hierarchy_validate", HierarchyValidationResult, prompt,
            relations=rel_str,
        )

    # ================================================================
    # Phase 2.6 — Self-Correction from Reasoner Errors
    # ================================================================

    def self_correct(self, ontology_str: str, error_log: str) -> CorrectionResult:
        """Ask the LLM to fix errors detected by the OWL reasoner."""
        prompt = PROMPT_SELF_CORRECTION.format(
            current_ontology=ontology_str,
            error_log=error_log,
        )
        return self._call_with_retry_and_cache(
            "self_correct", CorrectionResult, prompt,
            ontology=ontology_str, errors=error_log,
        )

    # ================================================================
    # Phase 3 — A-Box Extraction (UNCHANGED)
    # ================================================================

    def extract_triples(
        self,
        chunk_text: str,
        approved_ontology: str,
        canonical_terms: str,
        existing_entities: str,
    ) -> ExtractionResult:
        """Phase 3: Extract entities and relations from a single chunk."""
        prompt = SYSTEM_PROMPT_EXTRACTION.format(
            approved_ontology=approved_ontology,
            canonical_terms=canonical_terms,
            existing_entities=existing_entities,
            chunk_text=chunk_text,
        )
        return self._call_with_retry_and_cache(
            "extract",
            ExtractionResult,
            prompt,
            text=chunk_text,
            ontology=approved_ontology,
            canonical=canonical_terms,
            entities=existing_entities,
        )

    # ================================================================
    # Legacy Compatibility
    # ================================================================

    def seed_ontology_initial(self, text: str) -> SeedOntologyResult:
        """Legacy alias — use extract_classes_initial + extract_properties_initial instead."""
        from src.prompts import PROMPT_CLASS_EXTRACTION_INITIAL
        prompt = PROMPT_CLASS_EXTRACTION_INITIAL.format(
            document_text=text,
            bfo_skeleton=BFO_PROMPT_SKELETON,
            competency_questions="(none)",
        )
        # This won't return SeedOntologyResult directly — caller should migrate
        raise DeprecationWarning(
            "seed_ontology_initial is deprecated. Use the decomposed pipeline: "
            "generate_competency_questions → extract_classes_initial → extract_properties_initial"
        )

    def bootstrap_ontology(self, text: str) -> SeedOntologyResult:
        """Legacy alias."""
        return self.seed_ontology_initial(text)
