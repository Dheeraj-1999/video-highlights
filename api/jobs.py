import os
import uuid
import json
import asyncio
import moviepy.editor as mp
from src.utils.config import Config
from src.audio.transcriber import extract_audio, transcribe_audio
from src.text.chunker import merge_segments
from src.text.embedding_builder import build_embeddings
from src.text.highlight_selector import rerank_with_llm
from src.video.cutter import create_highlight_reel, limit_highlight_duration
from src.text.highlight_selector import generate_candidate_highlights
from src.video.cutter import pad_and_merge_segments
# ------------------------------------------------------------------
# GLOBALS
# ------------------------------------------------------------------
Config.ensure_dirs()
JOBS = {}
JOB_DIR = os.path.join(Config.PROCESSED_DIR, "jobs")
os.makedirs(JOB_DIR, exist_ok=True)

# ------------------------------------------------------------------
# HELPERS
# ------------------------------------------------------------------
def save_job_state(job_id, job_data):
    """Persist job state to JSON."""
    path = os.path.join(JOB_DIR, f"{job_id}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(job_data, f, indent=2)


def load_job_state(job_id):
    """Load job state from disk."""
    path = os.path.join(JOB_DIR, f"{job_id}.json")
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return JOBS.get(job_id)


# ------------------------------------------------------------------
# JOB CREATION
# ------------------------------------------------------------------
def create_job(filename: str, file_bytes: bytes, target_duration: int = 60):
    job_id = str(uuid.uuid4())
    JOBS[job_id] = {
        "state": "queued",
        "progress": 0,
        "message": "Job created",
        "result_path": None,
        "error": None,
    }
    save_job_state(job_id, JOBS[job_id])

    loop = asyncio.get_event_loop()
    loop.create_task(process_video_job(job_id, filename, file_bytes, target_duration))
    return job_id


# ------------------------------------------------------------------
# MAIN ASYNC PIPELINE
# ------------------------------------------------------------------
async def process_video_job(job_id: str, filename: str, file_bytes: bytes, target_duration: int):
    job = JOBS[job_id]
    try:
        # Save uploaded file
        job.update({"state": "running", "progress": 5, "message": "Saving uploaded video"})
        save_job_state(job_id, job)
        os.makedirs(Config.RAW_DIR, exist_ok=True)
        video_path = os.path.join(Config.RAW_DIR, filename)
        with open(video_path, "wb") as f:
            f.write(file_bytes)

        job.update({"progress": 15, "message": "Extracting audio"})
        save_job_state(job_id, job)
        audio_path = extract_audio(video_path)

        job.update({"progress": 25, "message": "Transcribing"})
        save_job_state(job_id, job)
        segments = transcribe_audio(audio_path, model_name="tiny")

        job.update({"progress": 40, "message": "Merging transcript chunks"})
        save_job_state(job_id, job)
        chunks = merge_segments(segments)
        chunk_path = os.path.join(Config.PROCESSED_DIR, f"{job_id}_chunks.json")
        with open(chunk_path, "w", encoding="utf-8") as f:
            json.dump(chunks, f, indent=2)

        job.update({"progress": 55, "message": "Building embeddings"})
        save_job_state(job_id, job)
        build_embeddings(chunk_path)

        job.update({"progress": 70, "message": "Selecting highlights"})
        save_job_state(job_id, job)

        index_path = os.path.join(Config.PROCESSED_DIR, "faiss_index.bin")
        candidates = generate_candidate_highlights(index_path, chunk_path, top_k=30)
        ranked = rerank_with_llm(candidates[:12], "A Cricket Video Editor", target_duration)
        # results = query_similar_chunks(
        #     "video summary highlights",
        #     top_k=10,
        #     index_path=index_path,
        #     chunk_path=chunk_path
        # )
        # if not results:
        #     raise ValueError("No chunks retrieved from FAISS. Possibly corrupted index or chunk mismatch.")
        # results = query_similar_chunks("video summary highlights", top_k=10)
        # ranked = rerank_with_llm(results, "Create an engaging summary", target_duration)
        
        job.update({"progress": 75, "message": "Smoothing highlight segments"})
        save_job_state(job_id, job)
        video_clip = mp.VideoFileClip(video_path)
        video_duration = video_clip.duration
        video_clip.close()
        
        ranked = sorted(ranked, key=lambda x: x["start"])
        ranked = pad_and_merge_segments(
            ranked,
            pad=1.5,          # seconds of padding before & after each clip
            merge_gap=2.0,    # merge clips if they are within 2 seconds
            video_duration=video_duration
        )
        ranked = limit_highlight_duration(ranked, max_total_seconds=target_duration)
        if not ranked:
            raise ValueError("No highlight segments found after retrieval.")
        # Save ranked JSON for debugging
        ranked_path = os.path.join(Config.PROCESSED_DIR, f"{job_id}_ranked.json")
        with open(ranked_path, "w", encoding="utf-8") as f:
            json.dump(ranked, f, indent=2)

        job.update({"progress": 85, "message": "Creating highlight reel"})
        save_job_state(job_id, job)

        output_path = create_highlight_reel(video_path, ranked)  # returns real path
        abs_path = os.path.abspath(output_path)
        print(f"✅ Highlight reel created at: {abs_path}")

        # Job success
        job.update({
            "state": "done",
            "progress": 100,
            "message": "completed",
            "result_path": output_path,
            "error": None,
            "download_url": f"http://127.0.0.1:8000/result/{job_id}"
        })
        save_job_state(job_id, job)
        print(f"✅ Job {job_id} completed successfully.")

    except Exception as e:
        print(f"❌ Job {job_id} failed: {e}")
        job.update({
            "state": "failed",
            "message": str(e),
            "error": str(e)
        })
        save_job_state(job_id, job)


def get_job_status(job_id: str):
    return load_job_state(job_id) or {"error": "Job not found"}
