try:
    from pydantic_patch import patch_pydantic
except ImportError:
    def patch_pydantic():
        """
        Patch Pydantic to allow arbitrary types by modifying internal modules.
        This needs to be done before importing haystack.
        """
        print("Applying Pydantic patch for arbitrary types...")
        
        # Import pydantic modules we need to patch
        import pydantic
        from pydantic import ConfigDict
        from pydantic._internal import _dataclasses
        from pydantic._internal import _generate_schema
        
        original_unknown_type_schema = _generate_schema.GenerateSchema._unknown_type_schema
        
        def patched_unknown_type_schema(self, obj):
            """
            Override the unknown type schema generator to handle pandas DataFrame.
            """
            if hasattr(obj, '__module__') and hasattr(obj, '__name__'):
                if obj.__module__ == 'pandas.core.frame' and obj.__name__ == 'DataFrame':
                    return {
                        'type': 'any',
                        'original_type': 'pandas.core.frame.DataFrame',
                        'arbitrary_type': True
                    }
            
            return original_unknown_type_schema(self, obj)
        
        _generate_schema.GenerateSchema._unknown_type_schema = patched_unknown_type_schema
        
        if not hasattr(pydantic.BaseModel, 'model_config'):
            pydantic.BaseModel.model_config = ConfigDict(arbitrary_types_allowed=True)
        else:
            if isinstance(pydantic.BaseModel.model_config, dict):
                pydantic.BaseModel.model_config['arbitrary_types_allowed'] = True
            else:
                pydantic.BaseModel.model_config.arbitrary_types_allowed = True
        
        print("Pydantic patch applied successfully")

    patch_pydantic()

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