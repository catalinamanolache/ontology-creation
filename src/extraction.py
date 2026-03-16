import os
import json
import hashlib
from typing import List
from huggingface_hub import InferenceClient
from config.settings import settings
from src.prompts import creation_prompt, refinement_prompt

CACHE_DIR = "data/cache"
os.makedirs(CACHE_DIR, exist_ok=True)

class OntologyExtractor:
    def __init__(self):
        self.model_name = settings.MODEL_NAME
        self.fallback_model_name = settings.FALLBACK_MODEL_NAME
        self.client = InferenceClient(api_key=settings.HF_API_TOKEN or None)

    def _cache_key(self, prefix: str, payload: dict) -> str:
        digest = hashlib.md5(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()
        return os.path.join(CACHE_DIR, f"{prefix}_{digest}.txt")

    def _cleanup_turtle(self, text: str) -> str:
        cleaned = []
        for line in text.splitlines():
            line = line.strip()
            if not line or line.startswith("```"):
                continue
            cleaned.append(line)

        # Deduplicate identical lines while preserving order to reduce repetition.
        seen = set()
        unique_lines = []
        for line in cleaned:
            if line not in seen:
                seen.add(line)
                unique_lines.append(line)

        # Ensure the last Turtle statement is properly terminated with a dot.
        # LLM output is sometimes truncated mid-declaration which breaks rdflib.
        if unique_lines:
            last = unique_lines[-1]
            if (not last.endswith(".") and
                    not last.startswith("@prefix") and
                    not last.lower().startswith("prefix")):
                unique_lines[-1] = last + " ."

        return "\n".join(unique_lines).strip()

    def _generate_text(self, prompt: str) -> str:
        last_error = None
        for model in [self.model_name, self.fallback_model_name]:
            try:
                # Preferred path: many hosted providers support conversational for instruct models.
                response = self.client.chat_completion(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=settings.TEMPERATURE,
                    max_tokens=settings.MAX_NEW_TOKENS,
                )
                content = response.choices[0].message.content
                if content:
                    return content.strip()
            except Exception as exc:
                last_error = exc
                try:
                    # Secondary path: try classic text-generation endpoint.
                    output = self.client.text_generation(
                        prompt=prompt,
                        model=model,
                        max_new_tokens=settings.MAX_NEW_TOKENS,
                        temperature=settings.TEMPERATURE,
                        do_sample=False,
                        return_full_text=False,
                    )
                    return output.strip()
                except Exception as exc2:
                    last_error = exc2
                    print(f"[WARN] Model '{model}' failed, trying fallback...")
        raise RuntimeError(f"Hugging Face generation failed: {last_error}")

    def _cached_generate(self, prefix: str, prompt: str, payload: dict) -> str:
        cache_path = self._cache_key(prefix, payload)
        if settings.USE_CACHE and os.path.exists(cache_path):
            with open(cache_path, "r", encoding="utf-8") as f:
                return f.read()

        text = self._generate_text(prompt)
        normalized = self._cleanup_turtle(text)
        if settings.USE_CACHE:
            with open(cache_path, "w", encoding="utf-8") as f:
                f.write(normalized)
        return normalized

    def bootstrap_ontology(self, text: str) -> str:
        prompt = creation_prompt.format(document_text=text)
        return self._cached_generate("bootstrap_ontology", prompt, {"text": text, "model": self.model_name})

    def refine_ontology(self, current_ontology: str, chunk_text: str) -> str:
        prompt = refinement_prompt.format(
            current_ontology_text=current_ontology,
            existing_triples_json="{}",
            chunk_text=chunk_text,
        )
        return self._cached_generate(
            "refine_ontology",
            prompt,
            {"current_ontology": current_ontology, "chunk_text": chunk_text, "model": self.model_name},
        )
