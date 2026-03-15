import os
import chromadb
from typing import List

class SemanticSearch:
    def __init__(self, db_path: str = "./chroma_db", collection_name: str = "ontology_docs"):
        # We use a persistent client so we don't re-embed chunks continually
        self.client = chromadb.PersistentClient(path=db_path)
        
        # We can use Chroma's default embedding function, or explicitly configure one (e.g., text-embedding-ada-002)
        # Using default SentenceTransformer for local execution speed & ease.
        self.collection = self.client.get_or_create_collection(name=collection_name)

    def index_chunks(self, chunks: List[str]):
        """
        Takes raw string chunks and stores them in ChromaDB.
        """
        ids = [f"chunk_{i}" for i in range(len(chunks))]
        
        # Only add if the collection is empty to prevent duplicates on re-runs
        if self.collection.count() == 0:
            print(f"Indexing {len(chunks)} chunks into ChromaDB...")
            self.collection.add(
                documents=chunks,
                ids=ids
            )
        else:
            print("ChromaDB collection already contains documents, skipping indexing.")

    def retrieve_context(self, main_anchors: List[str], disconnected_anchors: List[str], top_k: int = 3) -> str:
        """
        Embeds a query combining both anchor sets and returns top-K relevant text chunks
        to feed to the LLM.
        """
        # A good embedding query bridges the two concepts intuitively
        query = f"Find the relationship between {' '.join(main_anchors)} and {' '.join(disconnected_anchors)}"
        
        results = self.collection.query(
            query_texts=[query],
            n_results=top_k
        )
        
        if not results['documents'] or not results['documents'][0]:
            return ""

        # Join the retrieved chunks into a single context string
        retrieved_context = "\n\n---\n\n".join(results['documents'][0])
        return retrieved_context
