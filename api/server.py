import os
import json
import tempfile
import asyncio
from fastapi import FastAPI, UploadFile, Form
from fastapi.responses import FileResponse, JSONResponse
from src.utils.helpers import create_dirs
from src.utils.config import Config
from src.audio.transcriber import extract_audio, transcribe_audio
from src.text.chunker import merge_segments
from src.text.embedding_builder import build_embeddings
from src.text.highlight_selector import query_similar_chunks, rerank_with_llm
from src.video.cutter import create_highlight_reel, limit_highlight_duration

# -------------------------------------------------------------
# GLOBAL MODEL CACHE
# -------------------------------------------------------------
from sentence_transformers import SentenceTransformer
import whisper
from fastapi.middleware.cors import CORSMiddleware
app = FastAPI(title="🎬 GenAI Video Highlight API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # lock this down later to your UI domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Cached models
CACHE = {
    "whisper": None,
    "embedder": None,
}

# -------------------------------------------------------------
# LOAD MODELS (runs only once)
# -------------------------------------------------------------
@app.on_event("startup")
async def load_models():
    create_dirs()
    if CACHE["whisper"] is None:
        CACHE["whisper"] = whisper.load_model("tiny")
        print("✅ Whisper model loaded into cache")

    if CACHE["embedder"] is None:
        CACHE["embedder"] = SentenceTransformer("all-MiniLM-L6-v2")
        print("✅ SentenceTransformer model loaded into cache")

# -------------------------------------------------------------
# API ROUTE: Generate Highlight
# -------------------------------------------------------------
@app.post("/generate_highlight")
async def generate_highlight(
    video_file: UploadFile,
    user_prompt: str = Form(...),
    target_duration: int = Form(60)
):
    try:
        # Save uploaded video to a temp path
        tmp_dir = tempfile.mkdtemp()
        video_path = os.path.join(tmp_dir, video_file.filename)
        with open(video_path, "wb") as f:
            f.write(await video_file.read())

        # Step 1: Transcription using cached whisper
        whisper_model = CACHE["whisper"]
        print("🎧 Transcribing with cached Whisper...")
        audio_path = extract_audio(video_path)
        result = whisper_model.transcribe(audio_path)
        segments = result["segments"]

        # Step 2: Chunking
        chunks = merge_segments(segments)
        chunk_path = os.path.join(Config.PROCESSED_DIR, "chunks.json")
        with open(chunk_path, "w", encoding="utf-8") as f:
            json.dump(chunks, f, indent=2)

        # Step 3: Build embeddings
        print("🔢 Building embeddings with cached model...")
        build_embeddings(chunk_path, embedder=CACHE["embedder"])

        # Step 4: Query + LLM rerank
        results = query_similar_chunks("interesting highlights", top_k=15)
        ranked = rerank_with_llm(results, user_prompt, target_duration)

        # Step 5: Enforce duration
        ranked = limit_highlight_duration(ranked, max_total_seconds=target_duration)

        # Step 6: Save highlight candidates
        highlight_path = os.path.join(Config.PROCESSED_DIR, "highlight_candidates.json")
        with open(highlight_path, "w", encoding="utf-8") as f:
            json.dump(ranked, f, indent=2)

        # Step 7: Create highlight reel
        output_video = create_highlight_reel(video_path)

        return FileResponse(
            output_video,
            media_type="video/mp4",
            filename=os.path.basename(output_video)
        )

    except Exception as e:
        print("❌ Pipeline failed:", e)
        return JSONResponse({"error": str(e)}, status_code=500)
