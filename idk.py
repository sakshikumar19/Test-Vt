from chatbot2 import TextSplitter

# Sample YAML document with frontmatter
test_text = """---
title: "AI Research"
author: "Sakshi Kumar"
category: "Machine Learning"
---

## Introduction
AI is transforming various industries. It is reshaping healthcare, finance, and more.

### Neural Networks
Neural networks mimic the brain by learning patterns from data.

## Future Trends
The future of AI is exciting, with advancements in deep learning.
"""

# Instantiate the updated TextSplitter class
splitter = TextSplitter(chunk_size=100, chunk_overlap=20)

print("lol first")

# Run the text splitting process
documents = splitter.create_documents(test_text)

print("lol")

# Print results for verification
for i, doc in enumerate(documents):
    print(f"Chunk {i+1}:")
    print(f"Content: {doc.content}")
    print(f"Metadata: {doc.meta}")
    print("-" * 80)