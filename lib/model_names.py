"""Compatibility names for predictor selection."""
MODEL_ALIASES = {'STPatchGPT': 'STPatchTransformer'}  # historical identifiers

def canonical_model_name(name):
    """Resolve historical model identifiers without changing other predictors."""
    return MODEL_ALIASES.get(name, name)
