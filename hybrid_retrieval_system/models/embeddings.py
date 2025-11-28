from sentence_transformers import SentenceTransformer

# Lazily load the model to avoid heavy work at import time
_embedding_model = None

def _get_embedding_model():
    global _embedding_model
    if _embedding_model is None:
        # Load MiniLM model once
        _embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    return _embedding_model

def embed_text(text: str):
    model = _get_embedding_model()
    vector = model.encode(text, convert_to_numpy=True)
    return vector.tolist()
