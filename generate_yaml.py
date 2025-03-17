import os
import yaml
import argparse
from typing import List, Dict, Any
import asyncio

# Import the chatbot from the existing implementation
# Assuming chatbot2.py contains these imports - you'll need to adjust based on your actual implementation
from chatbot2 import VitessFAQChatbot, initialize_llm, initialize_inference_llm

# On Windows, use this event loop policy
if os.name == 'nt':
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

def create_test_cases_from_questions(
    questions_file: str,
    output_yaml: str,
    doc_paths: List[str],
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    chunk_size: int = 500,
    chunk_overlap: int = 75,
    top_k: int = 2  # Set to 2 to get top 2 chunks
) -> None:
    """
    Read questions from a text file, query the chatbot for each question,
    and create a YAML file with test cases including expected chunks.
    
    Args:
        questions_file: Path to the text file containing questions (one per line)
        output_yaml: Path to save the generated YAML file
        doc_paths: List of paths to documents to ingest into the chatbot
        embedding_model: Embedding model to use
        chunk_size: Chunk size for document processing
        chunk_overlap: Chunk overlap for document processing
        top_k: Number of top chunks to retrieve (default: 2)
    """
    # Initialize LLM
    llm = initialize_inference_llm()
    
    # Initialize chatbot
    chatbot = VitessFAQChatbot(
        embedding_model=embedding_model,
        llm_model=llm,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        top_k=top_k
    )
    
    # Ingest documents
    print(f"Ingesting documents from: {doc_paths}")
    chatbot.ingest_documents(doc_paths)
    print("Document ingestion complete.")
    
    # Read questions from file
    print(f"Reading questions from: {questions_file}")
    with open(questions_file, 'r', encoding='utf-8') as file:
        questions = [line.strip() for line in file if line.strip()]
    
    print(f"Found {len(questions)} questions.")
    
    # Process each question and build test cases
    test_cases = []
    for i, question in enumerate(questions):
        print(f"Processing question {i+1}/{len(questions)}: {question[:50]}...")
        
        # Query the chatbot
        result = chatbot.query(question)
        
        # Extract documents
        retrieved_docs = result.get("documents", [])
        
        # Extract content from top chunks
        expected_chunks = []
        for doc in retrieved_docs:
            if hasattr(doc, "content"):
                expected_chunks.append(doc.content)
        
        # Create test case
        test_case = {
            "question": question,
            "expected_chunks": expected_chunks
        }
        
        test_cases.append(test_case)
    
    # Create YAML structure
    yaml_data = {"test_cases": test_cases}
    
    # Write to YAML file
    with open(output_yaml, 'w', encoding='utf-8') as file:
        yaml.dump(yaml_data, file, default_flow_style=False, allow_unicode=True, sort_keys=False)
    
    print(f"Generated YAML file with {len(test_cases)} test cases: {output_yaml}")

def main():
    """Main function to parse arguments and run the converter"""
    parser = argparse.ArgumentParser(
        description="Convert a list of questions into a YAML test cases file with retrieved chunks"
    )
    parser.add_argument(
        "--questions_file", 
        required=True, 
        help="Path to the text file containing questions (one per line)"
    )
    parser.add_argument(
        "--output_yaml", 
        default="generated_test_cases.yaml", 
        help="Path to save the generated YAML file"
    )
    parser.add_argument(
        "--doc_paths", 
        nargs="+", 
        default=["./data/troubleshoot.txt", "./data/faq.txt", "data/design-docs.txt", "data/flags.txt", "data/v22.txt"], 
        help="Paths to documents to ingest"
    )
    parser.add_argument(
        "--embedding_model", 
        default="sentence-transformers/all-MiniLM-L6-v2", 
        help="Embedding model to use"
    )
    parser.add_argument(
        "--chunk_size", 
        type=int, 
        default=500, 
        help="Chunk size for document processing"
    )
    parser.add_argument(
        "--chunk_overlap", 
        type=int, 
        default=75, 
        help="Chunk overlap for document processing"
    )
    parser.add_argument(
        "--top_k", 
        type=int, 
        default=2, 
        help="Number of top chunks to retrieve (default: 2)"
    )
    
    args = parser.parse_args()
    
    create_test_cases_from_questions(
        questions_file=args.questions_file,
        output_yaml=args.output_yaml,
        doc_paths=args.doc_paths,
        embedding_model=args.embedding_model,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        top_k=args.top_k
    )

if __name__ == "__main__":
    main()