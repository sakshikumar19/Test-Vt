from chatbot import VitessFAQChatbot, initialize_llm

def main():
    """Main function to initialize and run the chatbot"""
    
    chatbot = VitessFAQChatbot(
        embedding_model="sentence-transformers/all-MiniLM-L6-v2",
        llm_model=initialize_llm('idk'),
        chunk_size=300,
        chunk_overlap=50,
        top_k=3
    )

    # Inspect document store to verify it's working
    stats = chatbot.inspect_document_store()
    print(f"Document store stats: {stats}")

    chatbot.run_cli()

if __name__ == "__main__":
    main()