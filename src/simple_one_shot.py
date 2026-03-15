import os
import sys
import json

# Ensure imports work from project root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.extraction import OntologyExtractor

def one_shot_test():
    print("--- Starting One-Shot API Verification ---")
    extractor = OntologyExtractor()
    
    # Simple input
    test_text = "The cat sat on the mat. The cat is named Whiskers."
    
    try:
        print("Sending one-shot request to Gemini...")
        result = extractor.bootstrap_ontology(test_text)
        
        print("\n[SUCCESS] API Key is verified and working!")
        print("Extracted Classes from prompt:")
        for cls in result.classes:
            print(f" - {cls.uri}: {cls.description}")
            
        print("\nExtracted Properties from prompt:")
        for prop in result.properties:
            print(f" - {prop.uri}: {prop.description}")
            
    except Exception as e:
        print(f"\n[ERROR] API call failed: {e}")

if __name__ == "__main__":
    one_shot_test()
