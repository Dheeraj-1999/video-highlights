import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import tempfile
import streamlit as st
from src.main import run_pipeline   # we’ll refactor main.py slightly below
from src.utils.helpers import create_dirs


# -------------------------------------------------------------
# STREAMLIT UI CONFIG
# -------------------------------------------------------------
st.set_page_config(page_title="🎬 GenAI Video Highlights", layout="wide")

st.title("🎥 Generative AI Video Highlights")
st.write("Upload a video and let AI create a creative, time-bound highlight reel.")

create_dirs()

# -------------------------------------------------------------
# SIDEBAR CONTROLS
# -------------------------------------------------------------
st.sidebar.header("🧠 Highlight Configuration")

user_prompt = st.sidebar.text_area(
    "Creative Prompt",
    "Show only boundaries, crowd reactions, and commentator excitement."
)

target_duration = st.sidebar.slider(
    "Target Highlight Duration (seconds)",
    30, 180, 60, step=10
)

generate_button = st.sidebar.button("🚀 Generate Highlights")

# -------------------------------------------------------------
# MAIN PANEL
# -------------------------------------------------------------
uploaded_file = st.file_uploader("Upload a .mp4 video", type=["mp4"])

if generate_button:
    if uploaded_file is None:
        st.error("Please upload a video file first.")
        st.stop()

    # Save uploaded file to a temporary location
    tmp_dir = tempfile.mkdtemp()
    video_path = os.path.join(tmp_dir, uploaded_file.name)
    with open(video_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.info("⚙️ Running AI highlight pipeline... this may take a few minutes ⏳")
    st.video(video_path)

    # Run the pipeline
    try:
        output_path = run_pipeline(video_path, user_prompt, target_duration)
        if output_path and os.path.exists(output_path):
            st.success("✅ Highlight reel created successfully!")
            st.video(output_path)
        else:
            st.warning("⚠️ No valid highlight clips found. Try a different prompt or duration.")
    except Exception as e:
        st.error(f"❌ Pipeline failed: {e}")
