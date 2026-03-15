import time
from typing import List
from langchain_google_genai import ChatGoogleGenerativeAI
from config.settings import settings
from src.schemas import BootstrapResult, ExtractionResult
from src.prompts import SYSTEM_PROMPT_BOOTSTRAP, SYSTEM_PROMPT_EXTRACTION, SYSTEM_PROMPT_BRIDGING

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
        # Accept both "models/<id>" and plain "<id>" formats.
        normalized = model_name.replace("models/", "", 1)

        # Legacy model IDs can 404 on newer API versions.
        legacy_map = {
            "gemini-1.5-flash": "gemini-2.5-flash",
            "gemini-1.5-pro": "gemini-2.5-pro",
            "gemini-2.0-flash-lite": "gemini-2.5-flash",
            "gemini-2.0-flash": "gemini-2.5-flash",
        }
        return legacy_map.get(normalized, normalized)

    def bootstrap_ontology(self, text: str) -> BootstrapResult:
        """
        Phase 3: Bootstraps the initial schema from the text using Structured Outputs.
        """
        time.sleep(2) # Protect against Free Tier Rate Limits (429)
        prompt = SYSTEM_PROMPT_BOOTSTRAP.format(document_text=text)
        structured_llm = self.llm.with_structured_output(BootstrapResult)
        return structured_llm.invoke(prompt)

    def extract_triples(self, text: str, approved_ontology: str) -> ExtractionResult:
        """
        Phase 3: Populates the graph based on the approved ontology.
        """
        time.sleep(2) # Protect against Free Tier Rate Limits (429)
        prompt = SYSTEM_PROMPT_EXTRACTION.format(
            approved_ontology=approved_ontology,
            chunk_text=text
        )
        structured_llm = self.llm.with_structured_output(ExtractionResult)
        return structured_llm.invoke(prompt)

    def bridge_subgraphs(self, text_context: str, main_anchors: List[str], disconnected_anchors: List[str], approved_ontology: str) -> ExtractionResult:
        """
        Phase 4: Bridges subgraphs using topological context.
        """
        time.sleep(2) # Protect against Free Tier Rate Limits (429)
        prompt = SYSTEM_PROMPT_BRIDGING.format(
            main_subgraph_anchors=", ".join(main_anchors),
            disconnected_subgraph_anchors=", ".join(disconnected_anchors),
            approved_ontology=approved_ontology,
            chunk_text=text_context
        )
        structured_llm = self.llm.with_structured_output(ExtractionResult)
        return structured_llm.invoke(prompt)
