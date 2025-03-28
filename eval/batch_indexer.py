# python batch_indexer.py ./vitess_docs --store-type chroma

import os
import argparse
import sys
from typing import List, Optional

# Import from the previous artifact
from document_store_factory import (
    AbstractDocumentStoreFactory, 
    DocumentIndexer, 
    BaseVitessDocumentStoreConfig
)

def index_vitess_docs(
    data_dir: str, 
    store_type: str = "chroma",
    embedding_model: str = "sentence-transformers/all-mpnet-base-v2",
    chunk_size: int = 200,
    chunk_overlap: int = 40,
    huggingface_token: Optional[str] = None,
    collection_name: str = "vitess_docs",
    persist_directory: Optional[str] = None
):
    """
    Batch index Vitess documentation from a specified directory
    
    Args:
        data_dir (str): Directory containing documentation files
        store_type (str): Type of document store to use
        embedding_model (str): Embedding model for document indexing
        chunk_size (int): Size of document chunks
        chunk_overlap (int): Overlap between chunks
        huggingface_token (Optional[str]): Hugging Face token for authentication
        collection_name (str): Name of the Chroma collection
        persist_directory (Optional[str]): Directory to persist Chroma index
    """
    # Validate data directory
    if not os.path.isdir(data_dir):
        print(f"Error: {data_dir} is not a valid directory")
        sys.exit(1)
    
    # Collect markdown and text files
    text_files = [
        os.path.join(data_dir, f) 
        for f in os.listdir(data_dir) 
        if f.endswith(('.txt', '.md'))
    ]
    
    if not text_files:
        print(f"No text files found in {data_dir}")
        sys.exit(1)
    
    # Configure document store
    config = BaseVitessDocumentStoreConfig(
        embedding_model=embedding_model,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        collection_name=collection_name,
        persist_directory=persist_directory
    )
    
    # Create document store
    document_store = AbstractDocumentStoreFactory.create_document_store(
        store_type=store_type,
        config=config,
        collection_name=collection_name,
        persist_directory=persist_directory
    )
    
    # Index documents
    DocumentIndexer.index_documents(
        document_store=document_store,
        file_paths=text_files,
        embedding_model=embedding_model,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        huggingface_token=huggingface_token
    )

def main():
    parser = argparse.ArgumentParser(description="Batch index Vitess documentation")
    parser.add_argument(
        "data_dir", 
        help="Directory containing documentation files"
    )
    parser.add_argument(
        "--store-type", 
        default="chroma", 
        choices=["inmemory", "chroma"],
        help="Type of document store to use"
    )
    parser.add_argument(
        "--embedding-model", 
        default="sentence-transformers/all-mpnet-base-v2",
        help="Embedding model for indexing"
    )
    parser.add_argument(
        "--chunk-size", 
        type=int, 
        default=200,
        help="Size of document chunks"
    )
    parser.add_argument(
        "--chunk-overlap", 
        type=int, 
        default=40,
        help="Overlap between document chunks"
    )
    parser.add_argument(
        "--huggingface-token", 
        help="Hugging Face authentication token"
    )
    parser.add_argument(
        "--collection-name", 
        default="vitess_docs",
        help="Name of the Chroma collection"
    )
    parser.add_argument(
        "--persist-directory", 
        help="Directory to persist Chroma index"
    )
    
    args = parser.parse_args()
    
    index_vitess_docs(
        data_dir=args.data_dir,
        store_type=args.store_type,
        embedding_model=args.embedding_model,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        huggingface_token=args.huggingface_token,
        collection_name=args.collection_name,
        persist_directory=args.persist_directory
    )

if __name__ == "__main__":
    main()