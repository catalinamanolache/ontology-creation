import time
import os
import json
import hashlib
from typing import List
from langchain_google_genai import ChatGoogleGenerativeAI
from config.settings import settings
from src.schemas import BootstrapResult, ExtractionResult
from src.prompts import SYSTEM_PROMPT_BOOTSTRAP, SYSTEM_PROMPT_EXTRACTION, SYSTEM_PROMPT_BRIDGING

CACHE_DIR = "data/cache"
os.makedirs(CACHE_DIR, exist_ok=True)

class OntologyExtractor:
    def __init__(self):
        if not settings.GOOGLE_API_KEY:
            raise ValueError(
                "Missing GOOGLE_API_KEY (or GEMINI_API_KEY) in environment/.env"
            )

        model_name = self._resolve_model_name(settings.MODEL_NAME)

        self.llm = ChatGoogleGenerativeAI(
            model=model_name,
            google_api_key=settings.GOOGLE_API_KEY,
            temperature=settings.TEMPERATURE,
            convert_system_message_to_human=True
        )

    @staticmethod
    def _resolve_model_name(model_name: str) -> str:
        # Standardize to a known, working model id
        return "gemini-2.5-flash"

    def _call_with_retry_and_cache(self, prefix: str, model_class, prompt: str, **cache_kwargs):
        # 1. Check Cache
        cache_content = json.dumps(cache_kwargs, sort_keys=True)
        h = hashlib.md5(cache_content.encode('utf-8')).hexdigest()
        cache_path = os.path.join(CACHE_DIR, f"{prefix}_{h}.json")
        
        if os.path.exists(cache_path):
            print(f"  [CACHE] Loading {prefix} result from local cache...")
            with open(cache_path, "r", encoding="utf-8") as f:
                content = f.read()
                try:
                    return model_class.model_validate_json(content)
                except AttributeError:
                    return model_class.parse_raw(content)
                    
        # 2. Add API Backoff and Retry
        max_retries = 5
        for attempt in range(max_retries):
            try:
                print(f"\n=======================================================")
                print(f"  [API CALL] INITIERE ACȚIUNE: '{prefix}'")
                print(f"  [API CALL] ÎNCERCAREA {attempt+1}/{max_retries}. Size: ~{len(str(cache_kwargs.get('text', '')))} chars.")
                print(f"=======================================================")
                
                time.sleep(2) # O scurtă pauză pentru a nu bloca imediat API-ul
                structured_llm = self.llm.with_structured_output(model_class)
                
                print("  -> Trimit request-ul catre Google... (Asteptam raspuns)")
                result = structured_llm.invoke(prompt)
                print("  -> RASPUNS PRIMIT DE LA GOOGLE!")
                
                # 3. Save to cache
                print(f"  [API SUCCESS] Succes pentru '{prefix}'! Am extras {len(getattr(result, 'entities', getattr(result, 'classes', [])))} entitati/clase.")
                print(f"  [CACHE] Salvez output-ul in: {cache_path}")
                try:
                    data_str = result.model_dump_json(indent=2)
                except AttributeError:
                    data_str = result.json(indent=2)
                    
                with open(cache_path, "w", encoding="utf-8") as f:
                    f.write(data_str)
                    
                return result
                
            except Exception as e:
                err_str = str(e)
                if "429" in err_str or "Quota" in err_str or "RESOURCE_EXHAUSTED" in err_str:
                    wait_time = 45 # Asteptam 45 secunde la 429
                    print(f"\n  [RATE LIMIT ERROR] EROARE 429: S-a atins limita de token-uri/request-uri (Free Tier)!")
                    print(f"  [RATE LIMIT ERROR] Google a refuzat cererea. Asteptam {wait_time} secunde inainte de a incerca din nou...")
                    time.sleep(wait_time)
                elif "503" in err_str or "overloaded" in err_str.lower():
                    wait_time = 20
                    print(f"\n  [SERVER ERROR] Eroare 503 Google. Asteptam {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    print(f"\n  [ERROR FATAL] O alta eroare s-a produs: {err_str}")
                    print(f"  [ERROR TYPE] Tip eroare: {type(e)}")
                    raise e
        raise Exception(f"API call failed after {max_retries} retries.")

    def bootstrap_ontology(self, text: str) -> BootstrapResult:
        """
        Phase 3: Bootstraps the initial schema from the text using Structured Outputs.
        """
        prompt = SYSTEM_PROMPT_BOOTSTRAP.format(document_text=text)
        return self._call_with_retry_and_cache("bootstrap", BootstrapResult, prompt, text=text)

    def extract_triples(self, text: str, approved_ontology: str) -> ExtractionResult:
        """
        Phase 3: Populates the graph based on the approved ontology.
        """
        prompt = SYSTEM_PROMPT_EXTRACTION.format(
            approved_ontology=approved_ontology,
            chunk_text=text
        )
        return self._call_with_retry_and_cache("extract", ExtractionResult, prompt, text=text, ontology=approved_ontology)

    def bridge_subgraphs(self, text_context: str, main_anchors: List[str], disconnected_anchors: List[str], approved_ontology: str) -> ExtractionResult:
        """
        Phase 4: Bridges subgraphs using topological context.
        """
        prompt = SYSTEM_PROMPT_BRIDGING.format(
            main_subgraph_anchors=", ".join(main_anchors),
            disconnected_subgraph_anchors=", ".join(disconnected_anchors),
            approved_ontology=approved_ontology,
            chunk_text=text_context
        )
        return self._call_with_retry_and_cache(
            "bridge",
            ExtractionResult,
            prompt,
            text=text_context,
            main=main_anchors,
            disc=disconnected_anchors,
            ontology=approved_ontology
        )
