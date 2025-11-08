import os
import uvicorn
from fastapi import FastAPI, UploadFile, Form
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from src.utils.helpers import create_dirs
from api.jobs import create_job, get_job_status, load_job_state
from src.utils.model_cache import ModelCache

# ------------------------------------------------------------------
app = FastAPI(title="🎬 GenAI Video Highlight API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------------------------------------------------------
@app.on_event("startup")
async def startup_event():
    create_dirs()
    print("✅ Directories created. API Ready.")
    ModelCache.load_whisper("tiny")
    ModelCache.load_embedder("all-mpnet-base-v2")
    print("🔥 Models pre-loaded successfully.")


@app.get("/")
def root():
    return {"message": "Welcome to GenAI Video Highlights API"}


@app.post("/jobs")
async def start_job(video_file: UploadFile, target_duration: int = Form(60)):
    file_bytes = await video_file.read()
    job_id = create_job(video_file.filename, file_bytes, target_duration)
    return {"job_id": job_id}


@app.get("/status/{job_id}")
def check_job_status(job_id: str):
    job_info = load_job_state(job_id)
    if not job_info:
        return JSONResponse(status_code=200, content={"state": "queued", "progress": 0, "message": "Starting..."})
    print(f"📊 STATUS [{job_id[:6]}]: {job_info}")
    return job_info


@app.get("/result/{job_id}")
def download_result(job_id: str):
    job = load_job_state(job_id)
    if not job:
        return JSONResponse(status_code=404, content={"error": "Job not found"})

    result_path = job.get("result_path")
    if not result_path or not os.path.isfile(result_path):
        print(f"⚠️ Missing result file for {job_id}: {result_path}")
        return JSONResponse(status_code=404, content={"error": f"Result file missing: {result_path}"})

    return FileResponse(result_path, media_type="video/mp4", filename=os.path.basename(result_path))

@app.get("/warmup")
def warmup_models():
    """
    Pre-load Whisper + SentenceTransformer models in memory.
    Useful to avoid cold-start latency.
    """
    try:
        whisper_model = ModelCache.load_whisper("tiny")
        embed_model = ModelCache.load_embedder("all-mpnet-base-v2")
        return {
            "message": "✅ Models warmed up and cached.",
            "whisper_device": str(whisper_model.device),
            "embedder_loaded": embed_model is not None
        }
    except Exception as e:
        return {"error": str(e)}


if __name__ == "__main__":
    uvicorn.run("api.server:app", host="0.0.0.0", port=8000)
