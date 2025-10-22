import json
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
# from src.utils.config import Config

def build_embeddings(chunk_path, embedder=None):
    if embedder is None:
        embedder = SentenceTransformer("all-MiniLM-L6-v2")

    with open(chunk_path, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    print(f"Loaded {len(chunks)} chunks for embedding")
    texts = [c["text"] for c in chunks]
    embeddings = embedder.encode(texts, convert_to_numpy=True, show_progress_bar=True)

    embeddings = np.array(embeddings, dtype="float32")
    faiss.normalize_L2(embeddings)

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    faiss.write_index(index, "data/processed/faiss_index.bin")
    print("FAISS index saved at data/processed/faiss_index.bin")

