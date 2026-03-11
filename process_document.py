import os
import re
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

def clean_text(text):
    """
    Function to remove noise from text while keeping authors and publication info.
    """
    # Remove only copyright elements and page numbers (e.g., © 21)
    text = re.sub(r'STM Journals \d{4}\. All Rights Reserved', '', text)
    text = re.sub(r'©\s*\d+(?:\s*,\s*\d+)?', '', text)
    text = re.sub(r'Volume \d+, Issue \d+', '', text)
    text = re.sub(r'ISSN: \d{4}-\d{4}', '', text)

    # Fix excessive whitespace and broken indentation
    # \s+ means any consecutive spaces, tabs, or new lines
    text = re.sub(r'\s+', ' ', text)
    
    return text.strip()

def process_document(file_path, chunk_size=1500, chunk_overlap=250):
    # Check if the file exists
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file {file_path} was not found.")

    print(f"Loading document: {file_path}")
    loader = PyPDFLoader(file_path)
    document_pages = loader.load()

    # Combine all pages into one large string for easier cleanup
    full_text = ""
    for page in document_pages:
        full_text += page.page_content + "\n"

    # Cut the text right before the references and acknowledgments sections
    # to avoid including them in the chunks
    if "REFERENCES" in full_text:
        full_text = full_text.split("REFERENCES")[0]
    if "Acknowledgments" in full_text:
        full_text = full_text.split("Acknowledgments")[0]

    # Apply the cleaning function
    full_text_cleaned = clean_text(full_text)

    # Define splitting rules
    # Use logical separators to avoid splitting in the middle of sentences or tables
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", " ", ""],
        length_function=len,
        is_separator_regex=False,
    )

    # Split cleaned text into chunks
    chunks = text_splitter.create_documents([full_text_cleaned])
    
    print(f"Document successfully split into {len(chunks)} cleaned sections.")
    return chunks

if __name__ == "__main__":
    file_path = "sample.pdf"
    chunk_size = 1500
    chunk_overlap = 250
    output_file = "chunks_output.txt"

    try:
        chunks = process_document(file_path, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        
        # Save chunks to a text file for inspection, with metadata about chunk size and overlap
        # Each chunk line will be capped at 100 characters for better readability,
        # although the actual chunk size is 1500 characters
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(f"Total chunks: {len(chunks)}\n")
            f.write(f"Configured chunk_size: {chunk_size} | Configured chunk_overlap: {chunk_overlap}\n")
            f.write("=" * 60 + "\n\n")
            for i, fragment in enumerate(chunks):
                content = fragment.page_content
                wrapped_content = "\n".join(
                    content[j:j + 100] for j in range(0, len(content), 100)
                )
                f.write(f"--- Chunk {i + 1} ---\n")
                f.write(f"Index: {i + 1} | Size: {len(content)} chars | Overlap: {chunk_overlap} chars\n")
                f.write(wrapped_content + "\n\n")
        print(f"Chunks saved to {output_file}")
            
    except Exception as e:
        print(f"An error occurred during processing: {e}")