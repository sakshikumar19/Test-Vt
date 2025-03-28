from haystack import component
from haystack import Document
from typing import List, Dict

@component
class DocumentDebugger:
    """Component to debug document content at different pipeline stages"""
    def __init__(self, stage_name="unknown", sample_length=200):
        self.stage_name = stage_name
        self.sample_length = sample_length
    
    @component.output_types(documents=List[Document])
    def run(self, documents: List[Document]) -> Dict[str, List[Document]]:
        """Print document content and pass through documents unchanged"""
        print(f"\n==== DOCUMENT DEBUGGER: {self.stage_name} ====")
        for i, doc in enumerate(documents[:2]):  # Only show first 2 docs to avoid flooding logs
            content_sample = doc.content[:self.sample_length]
            print(f"Document {i}: {repr(content_sample)}...")
            print(f"Metadata: {doc.meta}")
            print("-" * 50)
        if len(documents) > 2:
            print(f"...and {len(documents)-2} more documents")
        print(f"==== END DEBUGGER: {self.stage_name} ====\n")
        
        return {"documents": documents}
