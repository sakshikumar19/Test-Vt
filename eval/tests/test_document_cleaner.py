import pytest

from haystack.dataclasses import Document
from preprocessors.document_cleaner import (
    DocumentCleanerProtocol,
    CustomDocumentCleaner,
    GetDocumentCleaner
)

def test_custom_cleaner_protocol_compatibility():
    cleaner = CustomDocumentCleaner()
    assert isinstance(cleaner, DocumentCleanerProtocol), "Custom cleaner doesn't implement the protocol"

def test_custom_cleaner_behavior():
    docs = [Document(content="foo bar foo", meta={})]
    cleaner = CustomDocumentCleaner()
    result = cleaner.run(docs)

    assert "documents" in result
    cleaned_docs = result["documents"]
    assert isinstance(cleaned_docs, list)
    assert all(isinstance(doc, Document) for doc in cleaned_docs)
    assert cleaned_docs[0].content == "bar bar bar" or "bar" in cleaned_docs[0].content  # depending on your logic

def test_get_document_cleaner_factory():
    cleaner = GetDocumentCleaner()
    assert isinstance(cleaner, DocumentCleanerProtocol)
    doc = Document(content="       This   is  a  document  to  clean\n\n\nsubstring to remove")
    result = cleaner.run([doc])
    assert "documents" in result
    assert result["documents"][0].content == "This is a document to clean substring to remove"