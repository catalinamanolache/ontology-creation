import os
import re
from typing import List
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

class DocumentProcessor:
    def __init__(self, chunk_size: int = 1500, chunk_overlap: int = 250):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", ".", " ", ""],
            length_function=len,
            is_separator_regex=False,
        )

    def _clean_text(self, text: str) -> str:
        """
        Removes noise from text while keeping relevant info.
        This ensures determinism by stripping variable metadata.
        """
        # Remove copyright elements and page numbers (e.g., © 21)
        text = re.sub(r'STM Journals \d{4}\. All Rights Reserved', '', text)
        text = re.sub(r'©\s*\d+(?:\s*,\s*\d+)?', '', text)
        text = re.sub(r'Volume \d+, Issue \d+', '', text)
        text = re.sub(r'ISSN: \d{4}-\d{4}', '', text)
        
        # Specific cleaning for the sample documents
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

    def process_pdf(self, file_path: str) -> List[str]:
        """
        Loads a PDF, cleans it deterministically, and splits it into fixed chunks.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"The file {file_path} was not found.")

        loader = PyPDFLoader(file_path)
        document_pages = loader.load()

        full_text = ""
        for page in document_pages:
            full_text += page.page_content + "\n"

        # Deterministic cut to remove references and acknowledgments
        if "REFERENCES" in full_text:
            full_text = full_text.split("REFERENCES")[0]
        if "Acknowledgments" in full_text:
            full_text = full_text.split("Acknowledgments")[0]

        cleaned_text = self._clean_text(full_text)
        
        # Create chunks
        chunks = self.text_splitter.split_text(cleaned_text)
        return chunks
