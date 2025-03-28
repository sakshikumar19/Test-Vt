from haystack import component
from haystack.dataclasses import Document
from typing import List, Union, Dict, Any

@component
class DebugComponent:
    def __init__(self, name="debug"):
        self.name = name
    
    @component.output_types(documents=List[Document])
    def run(self, documents: List[Document]) -> Dict[str, List[Document]]:
        print(f"[{self.name}] Processing {len(documents)} documents")
        for i, doc in enumerate(documents[:3]):  # Show first 3 docs for brevity
            print(f"[{self.name}] Document {i} metadata: {doc.meta}")
            print(f"[{self.name}] Document {i} content preview: {doc.content[:100]}...")
        return {"documents": documents}
    
@component
class CustomDocumentSplitter:
    def __init__(self, chunk_size=700, chunk_overlap=50, respect_frontmatter_boundaries=True):
        from chatbot5 import SimpleTextSplitter
        self.text_splitter = SimpleTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            respect_frontmatter_boundaries=respect_frontmatter_boundaries
        )
        print("CustomDocumentSplitter initialized")

    @component.output_types(documents=List[Document])
    def run(self, documents: List[Union[Document, str]]) -> Dict[str, List[Document]]:
        result_docs = []
        print(f"Processing {len(documents)} documents in CustomDocumentSplitter")
        
        for doc_index, doc in enumerate(documents):
            # Handle both Document objects and strings
            if isinstance(doc, str):
                doc_content = doc
                doc_meta = {}
                doc_id = f"doc_{doc_index}"
                print(f"Document {doc_index} is a string, not a Document object")
            else:
                print(f"Document {doc_index} type: {type(doc)}")
                doc_content = doc.content
                doc_meta = doc.meta.copy() if hasattr(doc, 'meta') and doc.meta is not None else {}
                doc_id = doc.id if hasattr(doc, 'id') else f"doc_{doc_index}"
                print(f"Document {doc_index} original metadata: {doc_meta}")
            
            # Debug the content for frontmatter
            print(f"Document {doc_index} content preview: {doc_content[:200]}...")
            
            # Split the document
            chunks = self.text_splitter.create_documents(doc_content, meta=doc_meta)
            print(f"Document {doc_index} split into {len(chunks)} chunks")
            
            # Debug the chunks
            if chunks:
                print(f"First chunk from document {doc_index}:")
                print(f"  Type: {type(chunks[0])}")
                print(f"  Metadata: {chunks[0].meta if hasattr(chunks[0], 'meta') else 'No meta attribute'}")
                print(f"  Content preview: {chunks[0].content[:100] if hasattr(chunks[0], 'content') else '?'}...")
            
            # Process each chunk
            for i, chunk in enumerate(chunks):
                if hasattr(chunk, 'meta'):
                    # Already a Haystack Document with metadata
                    chunk_id = f"{doc_id}_{i}"
                    if not hasattr(chunk, 'id') or not chunk.id:
                        chunk.id = chunk_id
                    result_docs.append(chunk)
                    print(f"Added Haystack Document chunk {i} with metadata: {chunk.meta}")
                else:
                    # Convert to Haystack Document
                    chunk_content = chunk.content if hasattr(chunk, 'content') else chunk.text if hasattr(chunk, 'text') else str(chunk)
                    chunk_meta = chunk.meta if hasattr(chunk, 'meta') else {}
                    # Ensure we have the original document metadata
                    combined_meta = doc_meta.copy()
                    combined_meta.update(chunk_meta)
                    result_docs.append(Document(
                        content=chunk_content,
                        meta=combined_meta,
                        id=f"{doc_id}_{i}"
                    ))
                    print(f"Converted chunk {i} to Haystack Document with metadata: {combined_meta}")
        
        print(f"Returning {len(result_docs)} documents")
        if result_docs:
            print(f"First result document metadata: {result_docs[0].meta}")
            
        return {"documents": result_docs}