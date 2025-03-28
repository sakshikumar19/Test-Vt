# testtt.py
from custom_components import CustomDocumentSplitter
from haystack.dataclasses import Document

with open("data/v22.txt", "r", encoding="utf-8") as file:
    test_content = file.read()

test_doc = Document(content=test_content, meta={"file_path": "test.txt"})

# Process with your splitter
splitter = CustomDocumentSplitter()
# The component expects a list of Document objects directly, not in a dictionary
result = splitter.run(documents=[test_doc])

# Check results
for doc in result["documents"]:
    print(f"Result metadata: {doc.meta}")
    print(f"Result content: {doc.content}")