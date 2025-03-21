# custom_components.py
from haystack import component
from haystack.dataclasses import Document
from typing import List

@component
class CustomDocumentSplitter:
    # No __haystack_input__ declaration - let Haystack infer it

    def __init__(self, chunk_size=700, chunk_overlap=50, respect_frontmatter_boundaries=True):
        # Import your class inside the method to avoid import errors
        from chatbot2 import SimpleTextSplitter
        self.text_splitter = SimpleTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            respect_frontmatter_boundaries=respect_frontmatter_boundaries
        )

    @component.output_types(documents=List[Document])
    def run(self, documents: List[Document]) -> dict:
        # Process documents with your custom splitter
        result_docs = []
        for doc in documents:
            # Assuming create_documents returns objects with a content attribute
            chunks = self.text_splitter.create_documents(doc.content)
            for i, chunk in enumerate(chunks):
                # Create a combined metadata dictionary that includes both
                # the original document metadata and the chunk metadata
                combined_meta = doc.meta.copy() if hasattr(doc, 'meta') else {}
                chunk_meta = chunk.meta if hasattr(chunk, 'meta') else {}
                combined_meta.update(chunk_meta)  # Add/override with chunk metadata
                
                # Create new Haystack Document for each chunk
                result_docs.append(Document(
                    content=chunk.content if hasattr(chunk, 'content') else chunk.text if hasattr(chunk, 'text') else str(chunk),
                    meta=combined_meta,  # Use the combined metadata
                    id=f"{doc.id}_{i}" if hasattr(doc, 'id') else f"doc_{i}"
                ))
        
        return {"documents": result_docs}