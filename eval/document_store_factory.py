# document_store_factory.py

import os
import torch
from typing import List, Dict, Any, Optional
import chromadb

from haystack import Pipeline
from haystack_integrations.document_stores.chroma import ChromaDocumentStore
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.document_stores.types import DocumentStore
from haystack.utils import ComponentDevice
from haystack.components.preprocessors import DocumentSplitter
from haystack.components.embedders import SentenceTransformersDocumentEmbedder
from haystack.components.converters import TextFileToDocument
from haystack.components.writers import DocumentWriter
from haystack.utils import Secret

from sentence_transformers import SentenceTransformer

from chatbot import RobustFrontmatterExtractor

class BaseVitessDocumentStoreConfig:
    """Base configuration for document stores"""
    def __init__(
        self, 
        embedding_model: str = "sentence-transformers/all-mpnet-base-v2",
        chunk_size: int = 200,
        chunk_overlap: int = 40,
        device: str = None,
        huggingface_token: Optional[str] = None,
        collection_name: str = "vitess_docs",
        persist_directory: Optional[str] = './chroma_db'
    ):
        self.embedding_model = embedding_model
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
        self.huggingface_token = huggingface_token
        self.collection_name = collection_name
        self.persist_directory = persist_directory or os.path.join(os.getcwd(), "chroma_storage")

class ChromaEmbeddingFunction:
    def __init__(self, model_name: str):
        """
        Custom embedding function for Chroma that matches the required interface
        
        Args:
            model_name (str): Name of the SentenceTransformers model to use
        """
        self.model = SentenceTransformer(model_name)
    
    def __call__(self, input: List[str]) -> List[List[float]]:
        """
        Embed the input texts
        
        Args:
            input (List[str]): List of texts to embed (note the parameter name is 'input')
        
        Returns:
            List[List[float]]: List of embeddings
        """
        return self.model.encode(input).tolist()

def get_chroma_embedding_function(model_name: str = "sentence-transformers/all-mpnet-base-v2"):
    """
    Create a Chroma-compatible embedding function
    
    Args:
        model_name (str): Name of the SentenceTransformers model
    
    Returns:
        ChromaEmbeddingFunction: Chroma-compatible embedding function
    """
    return ChromaEmbeddingFunction(model_name)

class AbstractDocumentStoreFactory:
    """Factory for creating document stores with flexible configuration"""
    @staticmethod
    def create_document_store(
        store_type: str = "chroma",
        config: BaseVitessDocumentStoreConfig = None,
        **kwargs
    ) -> DocumentStore:
        config = config or BaseVitessDocumentStoreConfig()

        if store_type.lower() == "chroma":
            # Determine persist directory
            persist_directory = (
                kwargs.get("persist_directory") or
                config.persist_directory or
                os.path.join(os.getcwd(), "chroma_storage")
            )

            # Ensure persist directory exists
            os.makedirs(persist_directory, exist_ok=True)

            # Create Chroma embedding function
            embedding_function = get_chroma_embedding_function(config.embedding_model)

            # Create Chroma client
            chroma_client = chromadb.PersistentClient(path=persist_directory)

            # Create Chroma collection
            collection_name = kwargs.get("collection_name", config.collection_name)
            collection = chroma_client.get_or_create_collection(
                name=collection_name,
                embedding_function=embedding_function
            )

            # Create Chroma document store
            return ChromaDocumentStore(
                collection_name=collection_name,
                client=chroma_client,
                collection=collection,
                embedding_function=embedding_function,
                persist_directory=persist_directory
            )
                                
class DocumentIndexer:
    """Utility class for indexing documents into various document stores"""
    @staticmethod
    def index_documents(
        document_store: DocumentStore, 
        file_paths: List[str], 
        embedding_model: str,
        chunk_size: int = 200,
        chunk_overlap: int = 40,
        device: str = None,
        huggingface_token: Optional[str] = None
    ):
        """
        Index documents into the specified document store
        
        Args:
            document_store (DocumentStore): Target document store
            file_paths (List[str]): List of file paths to index
            embedding_model (str): Embedding model to use
            chunk_size (int): Size of document chunks
            chunk_overlap (int): Overlap between chunks
            device (str): Device to run embedding on
            huggingface_token (Optional[str]): Hugging Face token for authentication
        """
        device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
        token_secret = Secret.from_token(huggingface_token) if huggingface_token else None
        
        # Setup components
        text_converter = TextFileToDocument()
        frontmatter_extractor = RobustFrontmatterExtractor()
        splitter = DocumentSplitter(
            split_by="word", 
            split_length=chunk_size, 
            split_overlap=chunk_overlap
        )
        doc_embedder = SentenceTransformersDocumentEmbedder(
            model=embedding_model,
            device=ComponentDevice.from_str(device),
            token=token_secret
        )
        writer = DocumentWriter(document_store=document_store)
        
        # Create pipeline
        indexing_pipeline = Pipeline()
        indexing_pipeline.add_component("text_converter", text_converter)
        indexing_pipeline.add_component("frontmatter_extractor", frontmatter_extractor)
        indexing_pipeline.add_component("splitter", splitter)
        indexing_pipeline.add_component("embedder", doc_embedder)
        indexing_pipeline.add_component("writer", writer)
        
        # Connect pipeline
        indexing_pipeline.connect("text_converter.documents", "frontmatter_extractor.documents")
        indexing_pipeline.connect("frontmatter_extractor.documents", "splitter.documents")
        indexing_pipeline.connect("splitter.documents", "embedder.documents")
        indexing_pipeline.connect("embedder.documents", "writer.documents")
        
        # Index documents
        for file_path in file_paths:
            print(f"Indexing document: {file_path}")
            indexing_pipeline.run({"text_converter": {"sources": [file_path]}})
        
        # For Chroma, the persist_directory is already set during initialization
        print(f"Indexed {len(document_store.filter_documents())} documents.")