def patch_pydantic():
    """
    Patch Pydantic to allow arbitrary types by modifying internal modules.
    This needs to be done before importing haystack.
    """
    print("Applying comprehensive Pydantic patch for arbitrary types...")
    
    import pydantic
    from pydantic import ConfigDict
    from pydantic._internal import _dataclasses
    from pydantic._internal import _generate_schema
    
    original_unknown_type_schema = _generate_schema.GenerateSchema._unknown_type_schema
    
    def patched_unknown_type_schema(self, obj):
        """
        Override the unknown type schema generator to handle problematic types.
        """
        special_types = [
            ('pandas.core.frame', 'DataFrame'),
            ('numpy', 'ndarray'),
            ('torch', 'Tensor'),
            ('scipy.sparse', 'spmatrix'),
        ]
        
        if hasattr(obj, '__module__') and hasattr(obj, '__name__'):
            for module_name, type_name in special_types:
                if obj.__module__.startswith(module_name) and obj.__name__ == type_name:
                    return {
                        'type': 'any',
                        'original_type': f"{obj.__module__}.{obj.__name__}",
                        'arbitrary_type': True
                    }
                    
        # For class objects of any kind that aren't in our list
        if isinstance(obj, type):
            return {
                'type': 'any',
                'original_type': f"{obj.__module__}.{obj.__name__}" if hasattr(obj, '__module__') and hasattr(obj, '__name__') else str(obj),
                'arbitrary_type': True
            }
        
        try:
            return original_unknown_type_schema(self, obj)
        except Exception as e:
            # If all else fails, treat it as any type
            print(f"Warning: Using fallback schema for type {obj}: {str(e)}")
            return {
                'type': 'any',
                'original_type': str(obj),
                'arbitrary_type': True
            }
    
    _generate_schema.GenerateSchema._unknown_type_schema = patched_unknown_type_schema
    
    # Set default config for all models
    if not hasattr(pydantic.BaseModel, 'model_config'):
        pydantic.BaseModel.model_config = ConfigDict(arbitrary_types_allowed=True)
    else:
        if isinstance(pydantic.BaseModel.model_config, dict):
            pydantic.BaseModel.model_config['arbitrary_types_allowed'] = True
        else:
            pydantic.BaseModel.model_config.arbitrary_types_allowed = True
    
    if hasattr(pydantic, 'BaseSettings'):
        if not hasattr(pydantic.BaseSettings, 'model_config'):
            pydantic.BaseSettings.model_config = ConfigDict(arbitrary_types_allowed=True)
        else:
            # Update existing model_config
            if isinstance(pydantic.BaseSettings.model_config, dict):
                pydantic.BaseSettings.model_config['arbitrary_types_allowed'] = True
            else:
                # If it's a ConfigDict, update it
                pydantic.BaseSettings.model_config.arbitrary_types_allowed = True
    
    try:
        _dataclasses._arbitrary_types_allowed = True
    except:
        pass
    
    print("Comprehensive Pydantic patch applied successfully")

patch_pydantic()