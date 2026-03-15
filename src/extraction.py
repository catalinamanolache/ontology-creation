import time
from typing import List
from langchain_google_genai import ChatGoogleGenerativeAI
from config.settings import settings
from src.schemas import BootstrapResult, ExtractionResult
from src.prompts import SYSTEM_PROMPT_BOOTSTRAP, SYSTEM_PROMPT_EXTRACTION, SYSTEM_PROMPT_BRIDGING

class OntologyExtractor:
    def __init__(self):
        # We use the native Google GenAI SDK for reliability with the provided key
        self.llm = ChatGoogleGenerativeAI(
            model=settings.MODEL_NAME,
            google_api_key=settings.OPENAI_API_KEY,
            temperature=settings.TEMPERATURE,
            convert_system_message_to_human=True
        )

    def bootstrap_ontology(self, text: str) -> BootstrapResult:
        """
        Phase 3: Bootstraps the initial schema from the text using Structured Outputs.
        """
        time.sleep(10) # Protect against Free Tier Rate Limits (429)
        prompt = SYSTEM_PROMPT_BOOTSTRAP.format(document_text=text)
        structured_llm = self.llm.with_structured_output(BootstrapResult)
        return structured_llm.invoke(prompt)

    def extract_triples(self, text: str, approved_ontology: str) -> ExtractionResult:
        """
        Phase 3: Populates the graph based on the approved ontology.
        """
        time.sleep(10) # Protect against Free Tier Rate Limits (429)
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
        time.sleep(10) # Protect against Free Tier Rate Limits (429)
        prompt = SYSTEM_PROMPT_BRIDGING.format(
            main_subgraph_anchors=", ".join(main_anchors),
            disconnected_subgraph_anchors=", ".join(disconnected_anchors),
            approved_ontology=approved_ontology,
            chunk_text=text_context
        )
        structured_llm = self.llm.with_structured_output(ExtractionResult)
        return structured_llm.invoke(prompt)
