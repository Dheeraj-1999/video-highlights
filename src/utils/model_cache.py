# src/utils/model_cache.py
import torch
from sentence_transformers import SentenceTransformer
import whisper

class ModelCache:
    whisper_model = None
    embed_model = None

    @classmethod
    def load_whisper(cls, model_name="tiny"):
        if cls.whisper_model is None:
            print(f"🔹 Loading Whisper model: {model_name}")
            cls.whisper_model = whisper.load_model(model_name)
            print("✅ Whisper model loaded and cached.")
        return cls.whisper_model

    @classmethod
    def load_embedder(cls, model_name="all-mpnet-base-v2"):
        if cls.embed_model is None:
            print(f"🔹 Loading SentenceTransformer: {model_name}")
            cls.embed_model = SentenceTransformer(model_name)
            print("✅ SentenceTransformer loaded and cached.")
        return cls.embed_model
