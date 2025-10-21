import json, os
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
from src.utils.config import Config

def build_embeddings(chunk_file="data/processed/chunks.json",
                     index_out="data/processed/faiss_index.bin"):
    with open(chunk_file, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    texts = [c["text"] for c in chunks]

    print(f"🧠 Loaded {len(texts)} chunks for embedding")

    model = SentenceTransformer("all-MiniLM-L6-v2")
    print("🔢 Encoding text chunks...")
    embs = model.encode(texts, convert_to_numpy=True, show_progress_bar=True)
    print("✅ Embeddings shape:", embs.shape)

    faiss.normalize_L2(embs)
    index = faiss.IndexFlatIP(embs.shape[1])
    index.add(embs)
    print("💾 Adding vectors to FAISS index...")

    faiss.write_index(index, index_out)
    print(f"✅ FAISS index saved at {index_out}")

    return index, chunks


if __name__ == "__main__":
    build_embeddings()
