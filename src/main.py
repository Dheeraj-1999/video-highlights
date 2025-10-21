import os
import time
import json
from src.utils.config import Config
from src.utils.helpers import create_dirs

# Import pipeline modules
from src.audio.transcriber import extract_audio, transcribe_audio
from src.text.chunker import merge_segments
from src.text.embedding_builder import build_embeddings
from src.text.highlight_selector import query_similar_chunks, rerank_with_llm
from src.video.cutter import create_highlight_reel
from src.video.cutter import limit_highlight_duration

def run_pipeline(video_path, user_prompt, target_duration=60):
    """Run the entire end-to-end GenAI highlight creation pipeline."""
    start_time = time.time()
    create_dirs()

    print("\n🎧 Step 1: Audio Extraction & Transcription...")
    audio_path = extract_audio(video_path)
    segments = transcribe_audio(audio_path)
    transcript_path = os.path.join(Config.PROCESSED_DIR, "transcript_segments.json")
    with open(transcript_path, "w", encoding="utf-8") as f:
        json.dump(segments, f, indent=2)

    print("\n🧩 Step 2: Merging transcript segments...")
    chunks = merge_segments(segments)
    chunk_path = os.path.join(Config.PROCESSED_DIR, "chunks.json")
    with open(chunk_path, "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=2)

    print("\n🔢 Step 3: Building embeddings + FAISS index...")
    build_embeddings(chunk_path)

    print("\n🤖 Step 4: Selecting creative highlights via LLM...")
    results = query_similar_chunks("video highlight interesting moments", top_k=15)
    ranked = rerank_with_llm(results, user_prompt, target_duration)
    ranked = limit_highlight_duration(ranked, max_total_seconds=target_duration)
    highlight_path = os.path.join(Config.PROCESSED_DIR, "highlight_candidates.json")
    with open(highlight_path, "w", encoding="utf-8") as f:
        json.dump(ranked, f, indent=2)

    print("\n🎬 Step 5: Creating highlight reel video...")
    output_video = create_highlight_reel(video_path)
    print(f"✅ All steps complete! Final highlight video: {output_video}")
    end_time = time.time()
    print(f"🕒 Total pipeline time: {round(end_time - start_time, 2)} sec")

    return output_video  # ✅ ADD THIS LINE



if __name__ == "__main__":
    # ======= CONFIGURE HERE =======
    video_path = "data/raw/sample.mp4"

    # 🎨 Change this anytime for different creative output
    user_prompt = "Show only winning moment in the form of short reel"#fours, sixes, and big crowd reactions in a 1-minute highlight reel
    target_duration = 60  # seconds
    # results = query_similar_chunks("cricket commentary highlights", top_k=15)


    # ===============================
    run_pipeline(video_path, user_prompt, target_duration)
