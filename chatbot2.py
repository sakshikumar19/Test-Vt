import os
import re
from typing import List, Dict, Any, Optional, Union
import torch
import numpy as np
import time
import yaml

import pydantic
from pydantic import ConfigDict

import streamlit as st

if not hasattr(pydantic.BaseModel, 'model_config'):
    pydantic.BaseModel.model_config = ConfigDict(arbitrary_types_allowed=True)
else:
    # Update existing model_config
    if isinstance(pydantic.BaseModel.model_config, dict):
        pydantic.BaseModel.model_config['arbitrary_types_allowed'] = True
    else:
        # If it's a ConfigDict, update it
        pydantic.BaseModel.model_config.arbitrary_types_allowed = True

from haystack import Pipeline, Document
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.components.retrievers.in_memory import InMemoryBM25Retriever
from haystack.components.preprocessors import DocumentCleaner, DocumentSplitter
from haystack.components.embedders import SentenceTransformersDocumentEmbedder, SentenceTransformersTextEmbedder
from haystack.components.retrievers import InMemoryEmbeddingRetriever
from haystack.components.converters import TextFileToDocument
from haystack.components.writers import DocumentWriter
from haystack.utils import Secret
from haystack.components.joiners import DocumentJoiner
from haystack.utils import ComponentDevice
from haystack.components.rankers import TransformersSimilarityRanker

from langchain_huggingface import HuggingFaceEndpoint
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, BitsAndBytesConfig

from pydantic import BaseModel, ConfigDict
import numpy as np
from groq import Groq

from dotenv import load_dotenv

load_dotenv()

huggingface_token = os.getenv("HUGGINGFACE_TOKEN")

class MyModel(BaseModel):
    array: np.ndarray
    
    model_config = ConfigDict(arbitrary_types_allowed=True)

# def initialize_inference_llm(model_name):
#     llm = HuggingFaceEndpoint(
#         repo_id="mistralai/Mistral-7B-Instruct-v0.3",
#         huggingfacehub_api_token=huggingface_token,
#         temperature=0.2,
#         max_new_tokens=3000,
#         task="text-generation"
#     )
#     return llm


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

# def initialize_llm(model_name):
#     """Initialize and return the language model with model-specific tokenizer settings"""
#     start_time = time.time()

#     print(f"Loading model: {model_name}")

#     # Configure tokenizer settings based on model
#     if model_name == "mistralai/Mistral-7B-Instruct-v0.3":
#         use_fast = True
#     elif model_name == "microsoft/phi-2":
#         use_fast = False
#     else:
#         # Default for other models
#         use_fast = False

#     bnb_config = BitsAndBytesConfig(
#         load_in_4bit=True,
#         bnb_4bit_use_double_quant=False,
#         bnb_4bit_quant_type="nf4",
#         bnb_4bit_compute_dtype=torch.float16
#     )

#     model = AutoModelForCausalLM.from_pretrained(
#         model_name,
#         quantization_config=bnb_config,
#         device_map="auto"
#     )

#     tokenizer = AutoTokenizer.from_pretrained(
#         model_name,
#         use_fast=use_fast,  # Now using model-specific setting
#         add_eos_token=True,
#         padding_side="right"
#     )

#     print(f"Model and tokenizer loaded successfully (use_fast={use_fast}).")
#     end_time = time.time()
#     execution_time = end_time - start_time
#     print(f"Execution time: {execution_time} seconds")

#     tokenizer.pad_token_id = tokenizer.eos_token_id

#     pipe = pipeline(
#         "text-generation",
#         model=model,
#         tokenizer=tokenizer,
#         framework='pt',
#         max_new_tokens=1000,
#         temperature=0.3,
#         top_p=0.9,
#         repetition_penalty=1.2,
#         do_sample=True,
#         return_full_text=True  # Get the full text including the prompt
#     )

#     def llm_wrapper(prompt):
#         try:
#             response = pipe(prompt)[0]['generated_text']

#             # Extracting answer that comes after the prompt
#             if "[/INST]" in response:
#                 answer = response.split("[/INST]")[-1].strip()
#             else:
#                 answer = response.replace(prompt, "").strip()

#             return answer
#         except Exception as e:
#             print(f"Error in LLM wrapper: {e}")
#             return "Error generating response."

#     return llm_wrapper

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
    """Custom text splitter with improved chunking strategy that preserves frontmatter metadata"""
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

    def _extract_frontmatter(self, text: str) -> tuple[Dict[str, Any], str]:
        """Extract YAML frontmatter from text and return both frontmatter and content"""
        frontmatter = {}
        content = text
        
        # Look for YAML frontmatter between --- markers
        frontmatter_match = re.match(r'^---\s*\n(.*?)\n---\s*\n(.*)$', text, re.DOTALL)
        if frontmatter_match:
            try:
                yaml_content = frontmatter_match.group(1)
                frontmatter = yaml.safe_load(yaml_content) or {}
                content = frontmatter_match.group(2)
            except Exception as e:
                print(f"Error parsing frontmatter: {e}")
        
        return frontmatter, content

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
        """Split text into documents with intelligent chunking and preserved metadata"""
        # Extract frontmatter if present
        frontmatter, content = self._extract_frontmatter(text)
        
        # Prepare base metadata
        base_meta = meta.copy() if meta else {}
        
        # Add frontmatter to metadata
        if frontmatter:
            base_meta.update(frontmatter)
        
        # Now proceed with chunking the content part
        chunks = []
        start = 0
        content = content.strip() if self.strip_whitespace else content

        while start < len(content):
            # Determine end point for current chunk
            end = min(start + self.chunk_size, len(content))
            if end < len(content):
                end = self._find_split_point(content, end)

            # Extract chunk
            chunk = content[start:end]
            if self.strip_whitespace:
                chunk = chunk.strip()

            # Skip empty chunks
            if chunk:
                # Create a new metadata dict for each chunk
                chunk_meta = base_meta.copy()
                chunk_meta["start_idx"] = start
                chunk_meta["end_idx"] = end
                
                # Store section/heading context if possible
                heading_match = re.search(r'^#+\s+(.+?)$', chunk, re.MULTILINE)
                if heading_match:
                    chunk_meta["heading"] = heading_match.group(1)
                
                # Look for parent heading if this chunk doesn't have one
                if "heading" not in chunk_meta:
                    # Find the nearest heading before this chunk
                    content_before = content[:start]
                    headings = re.findall(r'^(#+)\s+(.+?)$', content_before, re.MULTILINE)
                    if headings:
                        level, heading = headings[-1]
                        chunk_meta["parent_heading"] = heading
                        chunk_meta["heading_level"] = len(level)

                chunks.append(Document(content=chunk, meta=chunk_meta))

            # Set start position for next chunk with overlap
            if end == len(content):
                break
            start = max(0, end - self.chunk_overlap)

        return chunks

    def process_file(self, file_path: str, meta: Optional[Dict[str, Any]] = None) -> List[Document]:
        """Process a file and split into chunks while preserving metadata"""
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
            
    def split_text(self, text: str, meta: Optional[Dict[str, Any]] = None) -> List[str]:
        """Legacy method for compatibility - converts documents back to text"""
        docs = self.create_documents(text, meta)
        return [doc.content for doc in docs]
            
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
    """Main chatbot class integrating all components"""
    def __init__(
        self,
        embedding_model,
        llm_model,
        chunk_size: int = 200,
        chunk_overlap: int = 40,
        top_k: int = 3
    ):
        self.embedding_model = embedding_model
        self.llm_model = llm_model
        print(f"llm_model type: {type(self.llm_model)}")
        self.top_k = top_k
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")

        # Initialize text splitter
        self.text_splitter = TextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n## ", "\n### ", "\n#### ", "\n", ". ", " "]
        )

        # Initialize document store using cosine similarity
        self.document_store = InMemoryDocumentStore(embedding_similarity_function="cosine")

        # Initialize embedding components first before pipelines
        self.init_embedding_components()

        # Initialize indexing pipeline
        self.init_indexing_pipeline()

        # Initialize retrieval pipeline
        self.init_retrieval_pipeline()

        # Initialize answer generator
        # self.answer_generator = LocalLLMAnswerGenerator(llm_model, self.device)
        self.answer_generator = LocalLLMAnswerGenerator(self.llm_model)

        # Chat history
        self.chat_history = []

    def init_embedding_components(self):
        """Initialize embedding components that will be used across pipelines"""
        # Create document embedder for semantic search - use device from initialization
        self.doc_embedder = SentenceTransformersDocumentEmbedder(
            model=self.embedding_model,
            device=ComponentDevice.from_str(self.device),  # Use class device
            token=Secret.from_token(huggingface_token)
        )

        # Text embedder with same model and device
        self.text_embedder = SentenceTransformersTextEmbedder(
            model=self.embedding_model,
            device=ComponentDevice.from_str(self.device),  # Use class device
            token=Secret.from_token(huggingface_token)
        )

        # Warm up embedders
        print("Warming up embedding models...")
        self.doc_embedder.warm_up()
        self.text_embedder.warm_up()

    def init_indexing_pipeline(self):
        """Initialize indexing pipeline with file type routing"""
        # Create converter components
        self.text_converter = TextFileToDocument()

        # Create preprocessing components
        self.cleaner = DocumentCleaner(
            remove_empty_lines=True,
            remove_extra_whitespaces=True
        )

        self.splitter = DocumentSplitter(
            split_by="word",
            split_length=self.text_splitter.chunk_size,
            split_overlap=self.text_splitter.chunk_overlap
        )

        # Create document writer
        self.writer = DocumentWriter(document_store=self.document_store)

        # Create indexing pipeline
        self.indexing_pipeline = Pipeline()
        self.indexing_pipeline.add_component("text_converter", self.text_converter)
        self.indexing_pipeline.add_component("cleaner", self.cleaner)
        self.indexing_pipeline.add_component("splitter", self.splitter)
        self.indexing_pipeline.add_component("embedder", self.doc_embedder)
        self.indexing_pipeline.add_component("writer", self.writer)

        # Connect components in the pipeline
        self.indexing_pipeline.connect("text_converter.documents", "cleaner.documents")
        self.indexing_pipeline.connect("cleaner.documents", "splitter.documents")
        self.indexing_pipeline.connect("splitter.documents", "embedder.documents")
        self.indexing_pipeline.connect("embedder.documents", "writer.documents")

    def init_retrieval_pipeline(self):
        """Initialize retrievers and retrieval pipeline"""
        # BM25 Retriever setup
        self.bm25_retriever = InMemoryBM25Retriever(document_store=self.document_store)

        # Embedding retriever with document store
        self.embedding_retriever = InMemoryEmbeddingRetriever(
            document_store=self.document_store,
            top_k=self.top_k
        )

        # Create document joiner for hybrid retrieval
        self.document_joiner = DocumentJoiner()

        # Create ranker
        try:
            self.ranker = TransformersSimilarityRanker(
                model="BAAI/bge-reranker-base",
                device=ComponentDevice.from_str(self.device)
            )
            self.use_ranker = True
        except Exception as e:
            print(f"Warning: Could not initialize ranker: {e}")
            self.use_ranker = False

        # Create hybrid retrieval pipeline
        self.retrieval_pipeline = Pipeline()
        self.retrieval_pipeline.add_component("text_embedder", self.text_embedder)
        self.retrieval_pipeline.add_component("embedding_retriever", self.embedding_retriever)
        self.retrieval_pipeline.add_component("bm25_retriever", self.bm25_retriever)
        self.retrieval_pipeline.add_component("document_joiner", self.document_joiner)

        # Connect base components
        self.retrieval_pipeline.connect("text_embedder", "embedding_retriever")
        self.retrieval_pipeline.connect("bm25_retriever", "document_joiner")
        self.retrieval_pipeline.connect("embedding_retriever", "document_joiner")

        if self.use_ranker:
            self.retrieval_pipeline.add_component("ranker", self.ranker)
            self.retrieval_pipeline.connect("document_joiner", "ranker")

    def ingest_documents(self, file_paths: List[str]) -> None:
        """Ingest documents into the document store using the pipeline"""
        try:
            # Verify file existence and readability
            valid_file_paths = []
            for file_path in file_paths:
                if not os.path.exists(file_path):
                    print(f"File does not exist: {file_path}")
                    continue

                # read file content
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                        print(f"Successfully read file: {file_path}, content length: {len(content)} characters")
                        if len(content) < 10:  # Very short content is suspicious
                            print(f"File content is suspiciously short: {content}")
                        else:
                            valid_file_paths.append(file_path)
                except Exception as e:
                    print(f"Error reading file {file_path}: {e}")

            if not valid_file_paths:
                print("No valid files to ingest")
                return

            print(f"Starting document processing for {len(valid_file_paths)} files")
            
            # Process documents with our custom splitter first
            all_documents = []
            for file_path in valid_file_paths:
                docs = self.text_splitter.process_file(file_path)
                if docs:
                    print(f"Created {len(docs)} chunks from {file_path}")
                    # Print metadata sample for verification
                    if docs:
                        print(f"Sample metadata: {docs[0].meta}")
                    all_documents.extend(docs)
            
            if not all_documents:
                print("No documents created after splitting")
                return
                
            print(f"Total documents to index: {len(all_documents)}")
                
            # Clean the documents
            cleaned_docs = self.cleaner.run(documents=all_documents)["documents"]
            
            # Generate embeddings
            print("Generating embeddings...")
            embedded_docs = self.doc_embedder.run(documents=cleaned_docs)["documents"]
            
            # Write to document store
            print("Writing to document store...")
            self.writer.run(documents=embedded_docs)
            
            # Verify documents in store after ingestion
            store_docs = self.document_store.filter_documents()
            print(f"Document store now contains {len(store_docs)} documents")

            # Check embeddings
            if store_docs:
                has_embeddings = all(hasattr(doc, "embedding") and doc.embedding is not None for doc in store_docs)
                print(f"Documents have embeddings: {has_embeddings}")

                if has_embeddings and len(store_docs) > 0:
                    embedding_dim = len(store_docs[0].embedding) if hasattr(store_docs[0], "embedding") else 0
                    print(f"Embedding dimension: {embedding_dim}")

                # Show first document if available
                print(f"First document preview: {store_docs[0].content[:100]}...")
                print(f"First document metadata: {store_docs[0].meta}")

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
            retrieval_result = self.retrieval_pipeline.run(query_inputs)

            # Get documents from correct component
            if self.use_ranker:
                retrieved_docs = retrieval_result["ranker"]["documents"]
            else:
                retrieved_docs = retrieval_result["document_joiner"]["documents"]

            print(f"Retrieval completed in {time.time() - retrieval_start:.2f}s. Found {len(retrieved_docs)} documents")

            # Apply metadata-based scoring boost
            scored_docs = []
            for doc in retrieved_docs:
                base_score = doc.score if hasattr(doc, "score") else 1.0
                metadata_boost = 1.0
                
                # Boost documents with matching title or description in metadata
                if "title" in doc.meta and any(term in doc.meta["title"].lower() for term in question.lower().split()):
                    metadata_boost *= 1.3
                    
                if "description" in doc.meta and any(term in doc.meta["description"].lower() for term in question.lower().split()):
                    metadata_boost *= 1.2
                    
                # Boost documents that have headings matching query terms
                if "heading" in doc.meta and any(term in doc.meta["heading"].lower() for term in question.lower().split()):
                    metadata_boost *= 1.5
                    
                if "parent_heading" in doc.meta and any(term in doc.meta["parent_heading"].lower() for term in question.lower().split()):
                    metadata_boost *= 1.2
                    
                # Assign new score
                doc.score = base_score * metadata_boost
                scored_docs.append(doc)
                
            # Sort by new score
            scored_docs.sort(key=lambda x: x.score if hasattr(x, "score") else 0, reverse=True)
            
            # Limit to top_k
            retrieved_docs = scored_docs[:self.top_k]
            
            # Add metadata to context
            print("Document metadata for retrieved documents:")
            for i, doc in enumerate(retrieved_docs):
                print(f"Doc {i+1} score: {doc.score if hasattr(doc, 'score') else 'N/A'}")
                print(f"Doc {i+1} metadata: {doc.meta}")

            if not retrieved_docs:
                # No relevant documents found, use fallback
                print("No relevant documents found, using fallback response")
                fallback_answer = self.get_fallback_response(question)
                self.chat_history.append((question, fallback_answer))
                return {
                    "answer": fallback_answer,
                    "documents": []
                }

            # Prepare context from documents with metadata
            print("Preparing context from retrieved documents...")
            context_parts = []
            
            for doc in retrieved_docs:
                # Add section metadata if available
                section_info = ""
                if "heading" in doc.meta:
                    section_info = f"Section: {doc.meta['heading']}\n"
                elif "parent_heading" in doc.meta:
                    section_info = f"Section: {doc.meta['parent_heading']}\n"
                    
                # Add title if available
                title_info = ""
                if "title" in doc.meta:
                    title_info = f"Title: {doc.meta['title']}\n"
                    
                # Combine metadata with content
                context_parts.append(f"{title_info}{section_info}{doc.content}")
                
            context = "\n\n---\n\n".join(context_parts)
            print(f"Context prepared. Total context length: {len(context)} characters")

            # Generate answer
            print("Generating answer using LLM...")
            generation_start = time.time()
            answer = self.answer_generator.generate(
                context=context,
                question=question,
                history=self.chat_history
            )
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
                "documents": retrieved_docs
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