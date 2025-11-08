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

```
video-highlights/
├── api/
│   ├── server.py
│   └── jobs.py
├── app/
│   └── streamlit_app.py
├── src/
│   ├── audio/transcriber.py
│   ├── text/{chunker,embedding_builder,highlight_selector}.py
│   ├── video/cutter.py
│   └── utils/{config,helpers}.py
├── data/{raw,processed}/
├── Dockerfile.api
├── Dockerfile.ui
├── docker-compose.yml
└── requirements.txt
```

---

## ⚙️ Setup Guide (Local Installation)

### 1️⃣ Clone the repository
```bash
git clone https://github.com/Dheeraj-1999/video-highlights.git
cd video-highlights
```

### 2️⃣ Create environment
```bash
conda create -n genai python=3.10 -y
conda activate genai
```
or
```bash
python -m venv genai
source genai/bin/activate   # Mac/Linux
genai\Scripts\activate    # Windows
```

### 3️⃣ Install dependencies
```bash
pip install -r requirements.txt
```

---

## 🚀 Run the Application

### 1️⃣ Start Backend (FastAPI)
```bash
python -m uvicorn api.server:app --port 8000
```
→ Runs on **http://127.0.0.1:8000**

### 2️⃣ Start Frontend (Streamlit)
```bash
streamlit run app/streamlit_app.py
```
→ Runs on **http://localhost:8501**

---

## 🎬 Using the App

1. Upload a `.mp4` file  
2. Choose target highlight duration (e.g. 60–180 s)  
3. Click **Generate Highlights**  
4. Watch progress → download final highlight when complete  

---

## 🧠 Model Configuration

| Component | Default | Change in | Alternatives |
|------------|----------|------------|---------------|
| Whisper | `tiny` | `src/audio/transcriber.py` | `base`, `small` |
| Embeddings | `all-mpnet-base-v2` | `src/text/embedding_builder.py`, `highlight_selector.py` | `all-MiniLM-L6-v2` |
| LLM | `gpt-4o-mini` | `src/text/highlight_selector.py` | `gpt-4o`, `gpt-3.5-turbo` |

---

## ✍️ Modify Prompt Template

Edit file:
```
src/prompts/highlight_prompt.txt
```

Variables: `$target_duration`, `$custom_prompt`, `$results_json`

---

## 🧾 API Endpoints

| Method | Endpoint | Description |
|---------|-----------|-------------|
| `POST` | `/jobs` | Upload & start job |
| `GET` | `/status/{job_id}` | Check job progress |
| `GET` | `/result/{job_id}` | Download final video |

---

## 🧰 Technologies

| Layer | Tool |
|--------|------|
| Frontend | Streamlit |
| Backend | FastAPI |
| Transcription | Whisper |
| Embeddings | SentenceTransformers + FAISS |
| LLM | GPT-4o-mini |
| Video Editing | MoviePy |
| Deployment | Docker / AWS Lightsail |

---

## 🩺 Troubleshooting

| Issue | Fix |
|--------|-----|
| OOM | use Whisper tiny model |
| 404 result | wait few seconds before polling |
| list index out of range | re-run, re-check chunks |
| cosine threshold fail | lower threshold in `highlight_selector.py` |

---

## 🏁 Summary

GenAI Video Highlights is a modular pipeline that automates highlight generation using AI — ready for demos, interview showcases, and real-world projects 🚀
