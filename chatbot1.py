import os
import re
import time
from typing import List, Dict, Any, Optional, Union
import yaml
import markdown
import subprocess
import shutil

import torch
from dotenv import load_dotenv

import streamlit as st

# Load environment variables
load_dotenv()

# Retrieve Hugging Face token
HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACE_TOKEN")

# Haystack components
from haystack import Pipeline, Document
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.components.retrievers.in_memory import InMemoryBM25Retriever
from haystack.components.preprocessors import DocumentCleaner, DocumentSplitter
from haystack.components.embedders import (
    SentenceTransformersDocumentEmbedder,
    SentenceTransformersTextEmbedder,
)
from haystack.components.retrievers import InMemoryEmbeddingRetriever
from haystack.components.converters import TextFileToDocument
from haystack.components.writers import DocumentWriter
from haystack.components.joiners import DocumentJoiner
from haystack.components.rankers import TransformersSimilarityRanker
from haystack.utils import Secret, ComponentDevice

# Hugging Face & Transformers
from langchain_huggingface import HuggingFaceEndpoint
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, BitsAndBytesConfig

# Groq API
from groq import Groq


def initialize_inference_llm(model_name: str = "llama-3.3-70b-versatile", api_key: Optional[str] = None, temperature: float = 0.2):
    groq_api_key = os.environ.get("GROQ_API_KEY")
    if not groq_api_key:
        raise ValueError("Groq API key must be provided either as a parameter or as GROQ_API_KEY environment variable")
    
    client = Groq(api_key=groq_api_key)
    
    def llm_callable(prompt: Union[str, List[Dict[str, str]]], max_tokens: int = 3000) -> str:
        if isinstance(prompt, str):
            messages = [{"role": "user", "content": prompt}]
        else:
            messages = prompt
            
        chat_completion = client.chat.completions.create(
            messages=messages,
            model=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        
        return chat_completion.choices[0].message.content
    
    return llm_callable


def initialize_llm(model_name):
    # """Initialize and return the language model with AWQ quantization"""
    # start_time = time.time()

    # print(f"Loading model: {model_name}")

    # # Configure tokenizer settings based on model
    # if model_name == "mistralai/Mistral-7B-Instruct-v0.3":
    #     use_fast = True
    # elif model_name == "microsoft/phi-2":
    #     use_fast = False
    # else:
    #     use_fast = False

    # # Load AWQ quantized model
    # model = AutoAWQForCausalLM.from_quantized(
    #     model_name,
    #     device="cpu",  # AWQ is better for CPU usage
    #     torch_dtype=torch.float16
    # )

    # tokenizer = AutoTokenizer.from_pretrained(
    #     model_name,
    #     use_fast=use_fast,
    #     add_eos_token=True,
    #     padding_side="right"
    # )

    # print(f"Model and tokenizer loaded successfully (use_fast={use_fast}).")
    # end_time = time.time()
    # execution_time = end_time - start_time
    # print(f"Execution time: {execution_time} seconds")

    # tokenizer.pad_token_id = tokenizer.eos_token_id

    # pipe = pipeline(
    #     "text-generation",
    #     model=model,
    #     tokenizer=tokenizer,
    #     framework="pt",
    #     max_new_tokens=1000,
    #     temperature=0.3,
    #     top_p=0.9,
    #     repetition_penalty=1.2,
    #     do_sample=True,
    #     return_full_text=True
    # )

    # def llm_wrapper(prompt):
    #     try:
    #         response = pipe(prompt)[0]["generated_text"]

    #         if "[/INST]" in response:
    #             answer = response.split("[/INST]")[-1].strip()
    #         else:
    #             answer = response.replace(prompt, "").strip()

    #         return answer
    #     except Exception as e:
    #         print(f"Error in LLM wrapper: {e}")
    #         return "Error generating response."
    llm_wrapper = ''
    return llm_wrapper


class TextSplitter:
    """Custom text splitter with improved chunking strategy"""
    def __init__(
        self,
        chunk_size: int = 200,
        chunk_overlap: int = 40,
        separators: List[str] = None,
        keep_separator: bool = True,
        strip_whitespace: bool = True
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or ["\n## ", "\n### ", "\n#### ", "\n", ". ", " "]
        self.keep_separator = keep_separator
        self.strip_whitespace = strip_whitespace

    def _find_split_point(self, text: str, target_length: int) -> int:
        """Find the best point to split text based on hierarchy of separators"""
        if len(text) <= target_length:
            return len(text)

        for separator in self.separators:
            # Find all occurrences of the separator
            positions = [m.start() for m in re.finditer(re.escape(separator), text[:target_length + len(separator)])]

            if positions:
                # Get the last occurrence before target_length
                valid_positions = [pos for pos in positions if pos <= target_length]
                if valid_positions:
                    split_point = valid_positions[-1]
                    if self.keep_separator and separator != " ":
                        # Move split point after separator
                        split_point += len(separator)
                    return split_point

        # Fallback: split at target_length
        return target_length

    def create_documents(self, text: str, meta: Optional[Dict[str, Any]] = None) -> List[Document]:
        """Split text into documents with intelligent chunking"""
        chunks = []
        start = 0
        text = text.strip() if self.strip_whitespace else text

        while start < len(text):
            # Determine end point for current chunk
            end = min(start + self.chunk_size, len(text))
            if end < len(text):
                end = self._find_split_point(text, end)

            # Extract chunk
            chunk = text[start:end]
            if self.strip_whitespace:
                chunk = chunk.strip()

            # Skip empty chunks
            if chunk:
                doc_meta = meta.copy() if meta else {}
                doc_meta["start_idx"] = start
                doc_meta["end_idx"] = end

                chunks.append(Document(content=chunk, meta=doc_meta))

            # Set start position for next chunk with overlap
            if end == len(text):
                break
            start = max(0, end - self.chunk_overlap)

        return chunks

    def process_file(self, file_path: str, meta: Optional[Dict[str, Any]] = None) -> List[Document]:
        """Process a file and split into chunks"""
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()

            base_meta = {"source": os.path.basename(file_path)}
            if meta:
                base_meta.update(meta)

            return self.create_documents(content, base_meta)
        except Exception as e:
            print(f"Error processing file {file_path}: {e}")
            return []
        
class LocalLLMAnswerGenerator:
    """Generates answers using a pre-loaded language model"""
    def __init__(self, local_llm):
        self.llm = local_llm
        print(f"llm_model type: {type(self.llm)}")

        print("Using pre-loaded language model")

    def generate(self, context: str, question: str, history: List = None) -> str:
        """Generate an answer based on context and question"""
        print("Starting answer generation...")

        history_text = ""
        if history:
            print(f"Including {len(history)} previous exchanges in prompt")
            for q, a in history[-2:]:  # Last 2 exchanges
                history_text += f"Human: {q}\nAssistant: {a}\n\n"

        # Add context truncation
        max_context_length = 3000  # Safe limit to prevent exceeding token limit
        if len(context) > max_context_length:
            print(f"Context too long ({len(context)} chars), truncating to ~{max_context_length} chars")
            # Split context into sentences and rebuild up to the limit
            sentences = re.split(r'(?<=[.!?])\s+', context)
            truncated_context = ""
            for sentence in sentences:
                if len(truncated_context) + len(sentence) + 1 <= max_context_length:
                    truncated_context += sentence + " "
                else:
                    break
            context = truncated_context.strip()
            print(f"Truncated context length: {len(context)} chars")

        prompt = f"""<s>[INST] You are an expert Vitess support assistant. Vitess is an open-source database clustering system for horizontal scaling of MySQL.

Answer the question based ONLY on the following context. If you don't know the answer or the information is not in the context, say "I don't have enough information to answer this question" and suggest what information might help.

Be accurate, helpful, concise, and clear. Format your answer using markdown when appropriate.

Context:
{context}

{history_text}
Question: {question} [/INST]
"""

        try:
            print(f"Context length: {len(context)} characters")
            print(f"Prompt length: {len(prompt)} characters")
            print("Starting LLM inference...")

            approx_tokens = len(prompt.split())
            print(f"Approximate token count: {approx_tokens}")

            inference_start = time.time()
            print("Sending request to model...")

            answer = self.llm(prompt)
            print(f"Model response received in {time.time() - inference_start:.2f}s")

            if isinstance(answer, str):
                # Direct string output
                response = answer
            elif isinstance(answer, list) and len(answer) > 0:
                # For models that return a list of responses
                response = answer[0]
                # Check if response has a text attribute
                if hasattr(response, 'text'):
                    response = response.text
                elif isinstance(response, dict) and 'generated_text' in response:
                    response = response['generated_text']
                else:
                    response = str(response)
            else:
                response = str(answer)

            # Clean up response
            if "[/INST]" in response:
                response = response.split("[/INST]")[-1].strip()

            response = response.strip('"\'')

            print(f"Processed answer (length: {len(response)} chars)")

            # fallback message
            if not response.strip():
                print("Empty answer after extraction, using fallback")
                response = "Based on the available information, I couldn't find specific details about this query in Vitess. Please check the official Vitess documentation for more information."

            return response
        except Exception as e:
            print(f"Error generating answer: {e}")
            import traceback
            traceback.print_exc()
        return "I apologize, but I encountered an error generating a response. Please try rephrasing your question or ask about a different Vitess topic."


class VitessFAQChatbot:
    """Main chatbot class integrating all components with dynamic repository cloning"""
    def __init__(
        self,
        embedding_model,
        llm_model,
        chunk_size: int = 200,
        chunk_overlap: int = 40,
        top_k: int = 3,
        clone_repos: bool = True,
        cleanup_repos: bool = True
    ):
        print("Initializing VitessFAQChatbot...")
        self.embedding_model = embedding_model
        self.llm_model = llm_model
        self.top_k = top_k
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.clone_repos = clone_repos
        self.cleanup_repos = cleanup_repos
        
        print(f"Using device: {self.device}")

        # Repository settings
        self.website_repo = os.environ.get("WEBSITE_REPO", "https://github.com/vitessio/website")
        self.vitess_repo = os.environ.get("VITESS_REPO", "https://github.com/vitessio/vitess")
        self.website_dir = "website_repo"
        self.vitess_dir = "vitess_repo"
        
        print("Initializing text splitter...")
        # Initialize text splitter
        self.text_splitter = TextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n## ", "\n### ", "\n#### ", "\n", ". ", " "]
        )

        print("Initializing document store...")
        # Initialize document store using cosine similarity
        self.document_store = InMemoryDocumentStore(embedding_similarity_function="cosine")

        # Initialize embedding components before pipelines
        print("Initializing embedding components...")
        self.init_embedding_components()

        # Initialize indexing pipeline
        print("Initializing indexing pipeline...")
        self.init_indexing_pipeline()

        # Initialize retrieval pipeline
        print("Initializing retrieval pipeline...")
        self.init_retrieval_pipeline()

        # Initialize answer generator
        print("Initializing answer generator...")
        self.answer_generator = LocalLLMAnswerGenerator(self.llm_model)

        # Chat history
        self.chat_history = []
        
        # Initialize document collection
        if self.clone_repos:
            print("Starting document collection initialization...")
            self.init_document_collection()
            print("Document collection initialization completed.")

    def init_embedding_components(self):
        """Initialize embedding components that will be used across pipelines"""
        print("Creating document embedder...")
        # Create document embedder for semantic search - use device from initialization
        self.doc_embedder = SentenceTransformersDocumentEmbedder(
            model=self.embedding_model,
            device=ComponentDevice.from_str(self.device),
            token=Secret.from_token(HUGGINGFACE_TOKEN)
        )

        print("Creating text embedder...")
        # Text embedder with same model and device
        self.text_embedder = SentenceTransformersTextEmbedder(
            model=self.embedding_model,
            device=ComponentDevice.from_str(self.device),
            token=Secret.from_token(HUGGINGFACE_TOKEN)
        )

        # Warm up embedders
        print("Warming up embedding models...")
        try:
            self.doc_embedder.warm_up()
            print("Document embedder warmed up successfully")
        except Exception as e:
            print(f"Error warming up document embedder: {e}")
            
        try:
            self.text_embedder.warm_up()
            print("Text embedder warmed up successfully")
        except Exception as e:
            print(f"Error warming up text embedder: {e}")

    def init_indexing_pipeline(self):
        """Initialize indexing pipeline with file type routing"""
        # Create converter components
        print("Creating text converter...")
        self.text_converter = TextFileToDocument()

        # Create preprocessing components
        print("Creating document cleaner...")
        self.cleaner = DocumentCleaner(
            remove_empty_lines=True,
            remove_extra_whitespaces=True
        )

        print("Creating document splitter...")
        self.splitter = DocumentSplitter(
            split_by="word",
            split_length=self.text_splitter.chunk_size,
            split_overlap=self.text_splitter.chunk_overlap
        )

        # Create document writer
        print("Creating document writer...")
        self.writer = DocumentWriter(document_store=self.document_store)

        # Create indexing pipeline
        print("Building indexing pipeline...")
        self.indexing_pipeline = Pipeline()
        self.indexing_pipeline.add_component("text_converter", self.text_converter)
        self.indexing_pipeline.add_component("cleaner", self.cleaner)
        self.indexing_pipeline.add_component("splitter", self.splitter)
        self.indexing_pipeline.add_component("embedder", self.doc_embedder)
        self.indexing_pipeline.add_component("writer", self.writer)

        # Connect components in the pipeline
        print("Connecting indexing pipeline components...")
        self.indexing_pipeline.connect("text_converter.documents", "cleaner.documents")
        self.indexing_pipeline.connect("cleaner.documents", "splitter.documents")
        self.indexing_pipeline.connect("splitter.documents", "embedder.documents")
        self.indexing_pipeline.connect("embedder.documents", "writer.documents")
        print("Indexing pipeline ready")

    def init_retrieval_pipeline(self):
        """Initialize retrievers and retrieval pipeline"""
        # BM25 Retriever setup
        print("Creating BM25 retriever...")
        self.bm25_retriever = InMemoryBM25Retriever(document_store=self.document_store)

        # Embedding retriever with document store
        print("Creating embedding retriever...")
        self.embedding_retriever = InMemoryEmbeddingRetriever(
            document_store=self.document_store,
            top_k=self.top_k
        )

        # Create document joiner for hybrid retrieval
        print("Creating document joiner...")
        self.document_joiner = DocumentJoiner()

        # Create ranker
        try:
            print("Initializing transformer similarity ranker...")
            self.ranker = TransformersSimilarityRanker(
                model="BAAI/bge-reranker-base",
                device=ComponentDevice.from_str(self.device)
            )
            self.use_ranker = True
            print("Ranker initialized successfully")
        except Exception as e:
            print(f"Warning: Could not initialize ranker: {e}")
            self.use_ranker = False

        # Create hybrid retrieval pipeline
        print("Creating retrieval pipeline...")
        self.retrieval_pipeline = Pipeline()
        self.retrieval_pipeline.add_component("text_embedder", self.text_embedder)
        self.retrieval_pipeline.add_component("embedding_retriever", self.embedding_retriever)
        self.retrieval_pipeline.add_component("bm25_retriever", self.bm25_retriever)
        self.retrieval_pipeline.add_component("document_joiner", self.document_joiner)

        # Connect base components
        print("Connecting retrieval pipeline components...")
        self.retrieval_pipeline.connect("text_embedder", "embedding_retriever")
        self.retrieval_pipeline.connect("bm25_retriever", "document_joiner")
        self.retrieval_pipeline.connect("embedding_retriever", "document_joiner")

        if self.use_ranker:
            print("Adding ranker to retrieval pipeline...")
            self.retrieval_pipeline.add_component("ranker", self.ranker)
            self.retrieval_pipeline.connect("document_joiner", "ranker")
        print("Retrieval pipeline ready")
    
    def clone_repo(self, repo_url: str, dest_dir: str) -> None:
        """Clone a Git repository if it doesn't already exist."""
        if not os.path.exists(dest_dir):
            print(f"Cloning {repo_url} to {dest_dir}...")
            try:
                subprocess.run(["git", "clone", repo_url, dest_dir], check=True)
                print(f"Successfully cloned {repo_url}")
            except subprocess.CalledProcessError as e:
                print(f"Error cloning repository {repo_url}: {e}")
                raise
        else:
            print(f"Repository {repo_url} already exists at {dest_dir}.")
    
    def cleanup_repository(self, repo_dirs: List[str]) -> None:
        """Clean up cloned repositories with Windows-specific handling."""
        for repo_dir in repo_dirs:
            if os.path.exists(repo_dir):
                print(f"Attempting to clean up repository: {repo_dir}")
                try:
                    # On Windows, sometimes files are locked by other processes
                    # Try multiple approaches
                    try:
                        # First attempt: standard removal
                        shutil.rmtree(repo_dir)
                        print(f"Deleted {repo_dir}")
                    except PermissionError:
                        print(f"Permission error when deleting {repo_dir}, trying with permissions modification...")
                        # Second attempt: force with os.chmod
                        for root, dirs, files in os.walk(repo_dir, topdown=False):
                            for file in files:
                                try:
                                    file_path = os.path.join(root, file)
                                    os.chmod(file_path, 0o777)
                                except Exception as e:
                                    print(f"Could not change permissions for {file_path}: {e}")
                        
                        # Try again after changing permissions
                        shutil.rmtree(repo_dir)
                        print(f"Deleted {repo_dir} after changing permissions")
                except Exception as e:
                    print(f"Warning: Could not fully delete {repo_dir}: {e}")
                    print("You may need to manually delete this directory later.")
            else:
                print(f"Repository directory {repo_dir} does not exist, no cleanup needed")
    
    def parse_markdown(self, file_path: str) -> Dict[str, Any]:
        """Parse a markdown file, extracting front matter and content."""
        print(f"Parsing markdown file: {file_path}")
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            print(f"Read {len(content)} bytes from {file_path}")
            metadata = {}
            markdown_content = content
            
            # Extract YAML front matter if present
            if content.startswith('---'):
                print(f"YAML front matter detected in {file_path}")
                parts = content.split('---', 2)
                if len(parts) >= 3:
                    try:
                        metadata = yaml.safe_load(parts[1].strip())
                        print(f"Parsed metadata: {list(metadata.keys())}")
                        markdown_content = parts[2].strip()
                    except yaml.YAMLError as e:
                        print(f"Error parsing YAML front matter in {file_path}: {e}")
            
            # Create a document object with metadata and content
            relative_path = os.path.relpath(file_path)
            print(f"Completed parsing {file_path}")
            return {
                "file_path": relative_path,
                "metadata": metadata,
                "content": markdown_content,
                "html": markdown.markdown(markdown_content)
            }
        
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def process_markdown_files(self, repo_dir: str, subdir: str = None) -> List[Dict[str, Any]]:
        """Process all markdown files in a repository or subdirectory."""
        documents = []
        
        base_dir = repo_dir
        if subdir:
            base_dir = os.path.join(repo_dir, subdir)
        
        if not os.path.exists(base_dir):
            print(f"Directory {base_dir} does not exist.")
            return documents
        
        print(f"Processing markdown files in {base_dir}...")
        
        # First collect all markdown files
        md_files = []
        for root, _, files in os.walk(base_dir):
            for file in files:
                if file.endswith(".md"):
                    md_files.append(os.path.join(root, file))
        
        print(f"Found {len(md_files)} markdown files in {base_dir}")
        
        # Then process each file
        for i, file_path in enumerate(md_files):
            print(f"Processing file {i+1}/{len(md_files)}: {file_path}")
            doc = self.parse_markdown(file_path)
            if doc:
                documents.append(doc)
                print(f"Successfully processed {file_path}")
            else:
                print(f"Failed to process {file_path}")
        
        print(f"Completed processing {len(documents)} markdown files in {base_dir}")
        return documents
    
    def process_text_files(self, directory: str) -> List[Dict[str, Any]]:
        """Process text files in a directory as plain documents."""
        documents = []
        
        if not os.path.exists(directory):
            print(f"Directory {directory} does not exist.")
            return documents
        
        print(f"Looking for text files in {directory}...")
        text_files = [f for f in os.listdir(directory) if f.endswith(".txt")]
        print(f"Found {len(text_files)} text files in {directory}")
        
        for i, file in enumerate(text_files):
            file_path = os.path.join(directory, file)
            print(f"Processing text file {i+1}/{len(text_files)}: {file_path}")
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                print(f"Read {len(content)} bytes from {file_path}")
                
                doc = {
                    "file_path": file_path,
                    "metadata": {"type": "text", "filename": file},
                    "content": content
                }
                documents.append(doc)
                print(f"Successfully processed text file {file_path}")
            except Exception as e:
                print(f"Error processing {file_path}: {e}")
                import traceback
                traceback.print_exc()
        
        print(f"Completed processing {len(documents)} text files in {directory}")
        return documents
    
    def init_document_collection(self):
        """Initialize document collection by cloning repos and processing files"""
        try:
            # Clone repositories
            print("=== STARTING REPOSITORY CLONING ===")
            self.clone_repo(self.website_repo, self.website_dir)
            self.clone_repo(self.vitess_repo, self.vitess_dir)
            print("=== COMPLETED REPOSITORY CLONING ===")
            
            # Process Vitess website documentation
            print("\n=== STARTING WEBSITE DOCUMENTATION PROCESSING ===")
            website_docs = []
            docs_paths = [
                "content/en/docs/22.0",
                "content/en/docs/faq",
                "content/en/docs/troubleshoot",
                "content/en/docs/design-docs"
            ]
            
            for path in docs_paths:
                print(f"\nProcessing docs path: {path}")
                path_docs = self.process_markdown_files(self.website_dir, path)
                website_docs.extend(path_docs)
                print(f"Added {len(path_docs)} documents from {path}, total now: {len(website_docs)}")
            
            print(f"=== COMPLETED WEBSITE DOCUMENTATION PROCESSING: {len(website_docs)} documents ===")
            
            # Process flags from vitess repo
            print("\n=== STARTING FLAGS PROCESSING ===")
            flags_dir = os.path.join(self.vitess_dir, "go", "flags", "endtoend")
            print(f"Processing flags from directory: {flags_dir}")
            flag_docs = self.process_text_files(flags_dir)
            print(f"=== COMPLETED FLAGS PROCESSING: {len(flag_docs)} documents ===")
            
            # Ingest all documents
            print("\n=== STARTING DOCUMENT INGESTION ===")
            print(f"Ingesting {len(website_docs)} website documents...")
            self.ingest_documents_from_json(website_docs)
            
            print(f"Ingesting {len(flag_docs)} flag documents...")
            self.ingest_documents_from_json(flag_docs)
            
            print(f"=== COMPLETED DOCUMENT INGESTION ===")
            print(f"Successfully ingested {len(website_docs)} website docs and {len(flag_docs)} flag docs")
            
        except Exception as e:
            print(f"Error initializing document collection: {e}")
            import traceback
            traceback.print_exc()
        finally:
            # Clean up repositories if specified
            if self.cleanup_repos:
                print("\n=== STARTING REPOSITORY CLEANUP ===")
                self.cleanup_repository([self.website_dir, self.vitess_dir])
                print("=== COMPLETED REPOSITORY CLEANUP ===")
    
    def ingest_documents_from_json(self, doc_list: List[Dict[str, Any]]) -> None:
        """Ingest documents from JSON-like data structure using the text splitter"""
        if not doc_list:
            print("No documents to ingest")
            return
        
        try:
            print(f"Starting ingestion of {len(doc_list)} documents")
            all_chunks = []
            
            for i, doc_data in enumerate(doc_list):
                print(f"Processing document {i+1}/{len(doc_list)}: {doc_data.get('file_path', 'unknown')}")
                content = doc_data.get("content", "")
                if not content.strip():
                    print(f"Skipping document with empty content: {doc_data.get('file_path', 'unknown')}")
                    continue
                
                # Prepare metadata
                meta = {
                    "source": doc_data.get("file_path", "unknown"),
                    "doc_type": "markdown" if doc_data.get("file_path", "").endswith(".md") else "text"
                }
                
                # Add all metadata from the original document
                if "metadata" in doc_data and isinstance(doc_data["metadata"], dict):
                    meta.update(doc_data["metadata"])
                
                print(f"Splitting document {i+1}/{len(doc_list)} with {len(content)} characters")
                # Create chunks with metadata
                try:
                    chunks = self.text_splitter.create_documents(content, meta)
                    print(f"Created {len(chunks) if chunks else 0} chunks for document {i+1}/{len(doc_list)}")
                    if chunks:
                        all_chunks.extend(chunks)
                except Exception as e:
                    print(f"Error creating chunks for document {i+1}/{len(doc_list)}: {e}")
                    import traceback
                    traceback.print_exc()
            
            # Process through embedding pipeline
            if all_chunks:
                print(f"Created total of {len(all_chunks)} chunks from {len(doc_list)} documents")
                
                # Process chunks in batches to avoid memory issues
                batch_size = 50
                num_batches = (len(all_chunks) + batch_size - 1) // batch_size
                
                for i in range(0, len(all_chunks), batch_size):
                    print(f"Processing batch {i//batch_size + 1}/{num_batches}")
                    batch = all_chunks[i:i + batch_size]
                    
                    # Clean documents
                    print(f"Cleaning {len(batch)} documents in batch {i//batch_size + 1}")
                    try:
                        cleaned_docs = self.cleaner.run(documents=batch)["documents"]
                        print(f"Successfully cleaned {len(cleaned_docs)} documents")
                    except Exception as e:
                        print(f"Error cleaning documents in batch {i//batch_size + 1}: {e}")
                        import traceback
                        traceback.print_exc()
                        continue
                    
                    # Embed documents
                    print(f"Embedding {len(cleaned_docs)} documents in batch {i//batch_size + 1}")
                    try:
                        embedded_docs = self.doc_embedder.run(documents=cleaned_docs)["documents"]
                        print(f"Successfully embedded {len(embedded_docs)} documents")
                    except Exception as e:
                        print(f"Error embedding documents in batch {i//batch_size + 1}: {e}")
                        import traceback
                        traceback.print_exc()
                        continue
                    
                    # Write to document store
                    print(f"Writing {len(embedded_docs)} documents to store in batch {i//batch_size + 1}")
                    try:
                        self.writer.run(documents=embedded_docs)
                        print(f"Successfully wrote batch {i//batch_size + 1} to document store")
                    except Exception as e:
                        print(f"Error writing documents to store in batch {i//batch_size + 1}: {e}")
                        import traceback
                        traceback.print_exc()
                        continue
                
                # Verify documents in store
                try:
                    store_docs = self.document_store.filter_documents()
                    print(f"Document store now contains {len(store_docs)} documents")
                except Exception as e:
                    print(f"Error retrieving documents from store: {e}")
            else:
                print("No valid chunks created")
                
        except Exception as e:
            print(f"Error ingesting documents: {e}")
            import traceback
            traceback.print_exc()
    
    def get_fallback_response(self, question: str) -> str:
        """Generate a fallback response when retrieval fails"""
        return (
            "I don't have enough information in my knowledge base to answer that question about Vitess. "
            "You might want to check the official Vitess documentation at https://vitess.io/docs/ or "
            "ask your question in the Vitess community Slack channel."
        )

    def query(self, question: str) -> Dict[str, Any]:
        """Query the chatbot with a question"""
        print(f"Processing query: '{question}'")
        start_time = time.time()
        
        if not question.strip():
            print("Empty question received, returning default response")
            return {
                "answer": "Please ask a question about Vitess.",
                "documents": []
            }
        
        try:
            # Check if document store has documents
            print("Checking document store...")
            all_docs = self.document_store.filter_documents()
            if not all_docs:
                print("Warning: Document store is empty!")
                return {
                    "answer": "I don't have any knowledge base to search from. Please ingest some documentation first.",
                    "documents": []
                }
            
            print(f"Document store contains {len(all_docs)} documents")
            
            print("Running hybrid retrieval pipeline...")
            retrieval_start = time.time()
            
            query_inputs = {
                "text_embedder": {"text": question},
                "bm25_retriever": {"query": question}
            }
            
            # Add ranker input
            if self.use_ranker:
                query_inputs["ranker"] = {"query": question}
            
            # Run the pipeline
            try:
                print("Executing retrieval pipeline...")
                retrieval_result = self.retrieval_pipeline.run(query_inputs)
                print("Retrieval pipeline execution complete")
            except Exception as e:
                print(f"Error running retrieval pipeline: {e}")
                import traceback
                traceback.print_exc()
                return {
                    "answer": "I encountered an error retrieving information. " + self.get_fallback_response(question),
                    "documents": []
                }
            
            # Get documents from correct component
            if self.use_ranker:
                retrieved_docs = retrieval_result["ranker"]["documents"]
            else:
                retrieved_docs = retrieval_result["document_joiner"]["documents"]
            
            print(f"Retrieval completed in {time.time() - retrieval_start:.2f}s. Found {len(retrieved_docs)} documents")
            
            # Limit to top_k
            retrieved_docs = retrieved_docs[:self.top_k]
            
            if not retrieved_docs:
                # No relevant documents found, use fallback
                print("No relevant documents found, using fallback response")
                fallback_answer = self.get_fallback_response(question)
                self.chat_history.append((question, fallback_answer))
                return {
                    "answer": fallback_answer,
                    "documents": []
                }
            
            # Prepare context from documents with source information
            print("Preparing context from retrieved documents...")
            context_parts = []
            for i, doc in enumerate(retrieved_docs):
                source_info = f"\nSource: {doc.meta.get('source', 'Unknown')}"
                if 'title' in doc.meta:
                    source_info += f" (Title: {doc.meta['title']})"
                
                context_parts.append(f"{doc.content}{source_info}")
                print(f"Document {i+1}: {doc.meta.get('source', 'Unknown')}")
            
            context = "\n\n".join(context_parts)
            print(f"Context prepared. Total context length: {len(context)} characters")
            
            # Generate answer
            print("Generating answer using LLM...")
            generation_start = time.time()
            try:
                answer = self.answer_generator.generate(
                    context=context,
                    question=question,
                    history=self.chat_history
                )
                print("Answer generation successful")
            except Exception as e:
                print(f"Error generating answer: {e}")
                import traceback
                traceback.print_exc()
                answer = self.get_fallback_response(question)
            
            print(f"Answer generation completed in {time.time() - generation_start:.2f}s")
            
            # Check if we got a valid answer
            if not answer or answer.strip() == "":
                print("Empty answer received from generator, using fallback")
                answer = self.get_fallback_response(question)
            
            # Update chat history
            self.chat_history.append((question, answer))
            
            print(f"Total query processing time: {time.time() - start_time:.2f}s")
            return {
                "answer": answer,
                "documents": retrieved_docs,
                "metadata": {
                    "sources": [doc.meta.get('source', 'Unknown') for doc in retrieved_docs]
                }
            }
        except Exception as e:
            print(f"Error querying: {e}")
            import traceback
            traceback.print_exc()
            fallback_answer = "I encountered an error processing your question. " + self.get_fallback_response(question)
            return {
                "answer": fallback_answer,
                "documents": []
            }
            
    def __del__(self):
        """Clean up resources when object is deleted"""
        print("Cleaning up resources...")
        if self.cleanup_repos:
            try:
                self.cleanup_repository([self.website_dir, self.vitess_dir])
            except Exception as e:
                print(f"Error cleaning up repositories: {e}")