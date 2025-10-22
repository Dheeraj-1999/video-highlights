import os
import tempfile
import requests
import streamlit as st

# ⚙️ FastAPI endpoint
# API_URL = "http://127.0.0.1:8000/generate_highlight"
import os
API_URL = os.getenv("VIDEO_API_URL", "http://127.0.0.1:8000/generate_highlight")
# API_URL = os.getenv("VIDEO_API_URL", "https://genai-video-api.onrender.com/generate_highlight")


st.set_page_config(page_title="🎬 GenAI Video Highlights", layout="wide")
st.title("🎥 Generative AI Video Highlights")
st.write("Upload a video and let AI create a creative, time-bound highlight reel.")

# Sidebar configuration
st.sidebar.header("🧠 Highlight Configuration")
user_prompt = st.sidebar.text_area(
    "Creative Prompt",
    "Show only boundaries, crowd reactions, and commentator excitement."
)
target_duration = st.sidebar.slider(
    "Target Highlight Duration (seconds)", 15, 180, 60, step=10
)
generate_button = st.sidebar.button("🚀 Generate Highlights")

# File upload
uploaded_file = st.file_uploader("Upload a .mp4 video", type=["mp4"])

if generate_button:
    if uploaded_file is None:
        st.error("Please upload a video file first.")
        st.stop()

    st.info("⚙️ Uploading to FastAPI and running pipeline...")
    st.video(uploaded_file)

    # Save temp file
    tmp_dir = tempfile.mkdtemp()
    video_path = os.path.join(tmp_dir, uploaded_file.name)
    with open(video_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Prepare payload for FastAPI
    files = {"video_file": open(video_path, "rb")}
    data = {
        "user_prompt": user_prompt,
        "target_duration": target_duration,
    }

    # Call FastAPI backend
    try:
        response = requests.post(API_URL, files=files, data=data, timeout=600)

        if response.status_code == 200:
            output_path = os.path.join(tmp_dir, "highlight_reel.mp4")
            with open(output_path, "wb") as f:
                f.write(response.content)
            st.success("✅ Highlight reel created successfully!")
            st.video(output_path)
        else:
            st.error(f"❌ Backend error: {response.status_code} - {response.text}")

    except Exception as e:
        st.error(f"❌ Connection failed: {e}")
