
from typing import List
import numpy as np
from sentence_transformers import SentenceTransformer


model_cache = {}

def get_model(model_name: str):
    if model_name not in model_cache:
        if "mini" in model_name.lower() or "minilm" in model_name.lower():
            model_cache[model_name] = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        elif "bge" in model_name.lower():
            model_cache[model_name] = SentenceTransformer("BAAI/bge-base-en-v1.5")
        else:
            raise ValueError(f"Unsupported embedding model: {model_name}")
    return model_cache[model_name]

def get_embedding(text: str, model_name: str = "all-MiniLM-L6-v2") -> List[float]:
    """
    Generate real embedding using SentenceTransformer models.
    """
    model = get_model(model_name)
    embedding = model.encode([text])[0]
    return embedding.tolist()
