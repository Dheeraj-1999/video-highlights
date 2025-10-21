import os
import json
import tempfile
import gc
from functools import lru_cache

from fastapi import FastAPI, UploadFile, Form
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from src.utils.helpers import create_dirs
from src.utils.config import Config
from src.audio.transcriber import extract_audio
from src.text.chunker import merge_segments
from src.text.embedding_builder import build_embeddings
from src.text.highlight_selector import query_similar_chunks, rerank_with_llm
from src.video.cutter import create_highlight_reel, limit_highlight_duration

# ============================================================
# ⚙️ FastAPI setup
# ============================================================
app = FastAPI(title="🎬 GenAI Video Highlight API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],        # later restrict to your Streamlit domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================
# 🧠 Lazy-load models on demand  (kept cached thereafter)
# ============================================================
@lru_cache(maxsize=1)
def get_whisper():
    import whisper
    print("🧠 Loading Whisper (tiny)...")
    return whisper.load_model("tiny")

@lru_cache(maxsize=1)
def get_embedder():
    from sentence_transformers import SentenceTransformer
    print("🧠 Loading SentenceTransformer (MiniLM-L6-v2)...")
    return SentenceTransformer("all-MiniLM-L6-v2")

# ============================================================
# 🚀 API Route
# ============================================================
@app.post("/generate_highlight")
async def generate_highlight(
    video_file: UploadFile,
    user_prompt: str = Form(...),
    target_duration: int = Form(60)
):
    try:
        create_dirs()
        tmp_dir = tempfile.mkdtemp()
        video_path = os.path.join(tmp_dir, video_file.filename)

        with open(video_path, "wb") as f:
            f.write(await video_file.read())

        # Step 1 – Transcription
        print("🎧 Transcribing (lazy Whisper load if first run)...")
        whisper_model = get_whisper()
        audio_path = extract_audio(video_path)
        result = whisper_model.transcribe(audio_path)
        segments = result["segments"]
        del result
        gc.collect()

        # Step 2 – Chunk merge
        chunks = merge_segments(segments)
        chunk_path = os.path.join(Config.PROCESSED_DIR, "chunks.json")
        with open(chunk_path, "w", encoding="utf-8") as f:
            json.dump(chunks, f, indent=2)
        del segments
        gc.collect()

        # Step 3 – Embeddings
        print("🔢 Building embeddings (lazy embedder load if first run)...")
        build_embeddings(chunk_path, embedder=get_embedder())
        gc.collect()

        # Step 4 – Query + LLM rerank
        results = query_similar_chunks("interesting highlights", top_k=15)
        ranked = rerank_with_llm(results, user_prompt, target_duration)

        # Step 5 – Trim to target duration
        ranked = limit_highlight_duration(ranked, max_total_seconds=target_duration)

        # Step 6 – Save highlight metadata
        highlight_path = os.path.join(Config.PROCESSED_DIR, "highlight_candidates.json")
        with open(highlight_path, "w", encoding="utf-8") as f:
            json.dump(ranked, f, indent=2)

        # Step 7 – Assemble final reel
        output_video = create_highlight_reel(video_path)

        # Clean up temp files
        try:
            os.remove(audio_path)
        except Exception:
            pass
        gc.collect()

        return FileResponse(
            output_video,
            media_type="video/mp4",
            filename=os.path.basename(output_video),
        )

    except Exception as e:
        print("❌ Pipeline failed:", e)
        return JSONResponse({"error": str(e)}, status_code=500)
