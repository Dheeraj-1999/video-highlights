# 🎬 GenAI Video Highlights (v2-Summary)

Automatically generate short highlight reels from long videos using Generative AI.  
This system extracts, analyzes, and summarizes video content using **Whisper**, **Sentence-Transformers**, **FAISS**, **GPT-4o-mini**, and **MoviePy**.

---

## 🌟 What It Does

This app takes any long `.mp4` video (like a cricket match, lecture, meeting, or podcast) and produces an **AI-generated highlight reel**.

### 🧠 Pipeline Overview
1. 🎧 **Whisper (OpenAI)** → Transcribes video audio into text.  
2. 🧩 **Sentence-Transformers + FAISS** → Creates embeddings and finds meaningful segments.  
3. 🤖 **GPT-4o-mini** → Chooses the most interesting moments based on context.  
4. 🎞️ **MoviePy** → Cuts and merges those segments into a `summary.mp4` highlight video.  
5. ⚡ **FastAPI + Streamlit** → Backend handles processing; frontend shows progress, video, and download.

---

## 🧱 Folder Structure

video-highlights/
├── api/
│ ├── server.py # FastAPI backend routes
│ └── jobs.py # Asynchronous job management
├── app/
│ └── streamlit_app.py # Streamlit UI for uploads & progress
├── src/
│ ├── audio/transcriber.py
│ ├── text/
│ │ ├── chunker.py
│ │ ├── embedding_builder.py
│ │ ├── highlight_selector.py
│ ├── video/cutter.py
│ └── utils/
│ ├── config.py
│ └── helpers.py
├── data/
│ ├── raw/ # Uploaded videos
│ └── processed/ # Transcripts, FAISS index, highlights
├── Dockerfile.api
├── Dockerfile.ui
├── docker-compose.yml
└── requirements.txt


---

## ⚙️ Setup Guide (Local Installation)

### 1️⃣ Clone the repository
```bash
git clone https://github.com/Dheeraj-1999/video-highlights.git
cd video-highlights
```

## Using Conda (recommended):
```
conda create -n genai python=3.10 -y
conda activate genai
```
## Or using venv:
```
python -m venv genai
genai\Scripts\activate      # Windows
```

## Install dependencies
```
pip install -r requirements.txt
```


## Start the Streamlit Frontend
```
streamlit run app/streamlit_app.py
```
## Start the FastAPI Backend
```
uvicorn api.server:app --reload --port 8000
```