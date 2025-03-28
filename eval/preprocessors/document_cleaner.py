from typing import Protocol, List, Dict, runtime_checkable
from haystack.dataclasses import Document
from haystack.components.preprocessors import DocumentCleaner

# Define a protocol for a document cleaner that
# can be used in the HayStack Pipeline. The `DocumentCleaner`
# class from Haystack also implements this protocol.
@runtime_checkable
class DocumentCleanerProtocol(Protocol):
    def run(self, documents: List[Document]) -> Dict[str, List[Document]]:
        ...

# We can create our own custom DocumentCleaner by implementing the protocol
# This is a dummy implementation to show how to implement a custom logic.
class CustomDocumentCleaner(DocumentCleanerProtocol):
    def run(self, documents: List[Document]) -> Dict[str, List[Document]]:
        cleaned_docs = []
        for doc in documents:
            # Custom cleaning logic
            doc.content = doc.content.replace("foo", "bar")
            cleaned_docs.append(doc)
        return {"documents": cleaned_docs}

# GetDocumentCleaner gets the document cleaner to be used in the pipeline
# It asserts that the cleaner implements the DocumentCleanerProtocol.
def GetDocumentCleaner() -> DocumentCleanerProtocol:
    # Currently there is only one implementation of the protocol
    # worth using. But if we have more implementations, then a flag can be
    # added to select which implementation to use.
    cleaner = DocumentCleaner()
    assert isinstance(cleaner, DocumentCleanerProtocol), "Cleaner does not implement the protocol"
    return cleaner

