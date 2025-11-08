import os
import time
import tempfile
import requests
import streamlit as st

API_URL = os.getenv("VIDEO_API_URL", "http://127.0.0.1:8000")

st.set_page_config(page_title="🎬 GenAI Video Highlights", layout="wide")
st.title("🎥 Generative AI Video Highlights")
st.caption("Upload a video and let AI create a creative, time-bound highlight reel.")

# Sidebar
st.sidebar.header("🧠 Highlight Configuration")
target_duration = st.sidebar.slider("Target Highlight Duration (seconds)", 15, 360, 60, step=10)
generate_button = st.sidebar.button("🚀 Generate Highlights")

# File upload
uploaded_file = st.file_uploader("Upload a .mp4 video", type=["mp4"])

if generate_button:
    if uploaded_file is None:
        st.error("Please upload a video file first.")
        st.stop()

    st.info("⚙️ Uploading video and starting processing job...")
    st.video(uploaded_file)

    tmp_dir = tempfile.mkdtemp()
    video_path = os.path.join(tmp_dir, uploaded_file.name)
    with open(video_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    try:
        files = {"video_file": open(video_path, "rb")}
        data = {"target_duration": target_duration}
        res = requests.post(f"{API_URL}/jobs", files=files, data=data)
        if res.status_code != 200:
            st.error(f"❌ Job creation failed: {res.text}")
            st.stop()

        job_id = res.json()["job_id"]
        st.success(f"✅ Job created! ID: `{job_id}`")

    except Exception as e:
        st.error(f"🚨 Failed to connect to backend: {e}")
        st.stop()

    # Poll job status
    progress_bar = st.progress(0)
    status_text = st.empty()
    download_shown = False

    while True:
        try:
            resp = requests.get(f"{API_URL}/status/{job_id}", timeout=10)
            if resp.status_code != 200:
                status_text.text("⏳ Waiting for backend...")
                time.sleep(3)
                continue
            status = resp.json()
        except Exception as e:
            status_text.text(f"⚠️ Waiting for job... {e}")
            time.sleep(3)
            continue

        state = status.get("state", "unknown")
        progress = int(status.get("progress", 0))
        message = status.get("message", "")
        status_text.text(f"🌀 {state.upper()} | {message}")
        progress_bar.progress(min(progress, 100))

        if state.lower() == "done":
            st.success("✅ Highlight generation completed!")
            download_url = status.get("download_url")

            if not download_shown and download_url:
                try:
                    st.info("⬇️ Fetching final video...")
                    r = requests.get(download_url, stream=True, timeout=30)
                    if r.status_code == 200:
                        tmp_output = os.path.join(tmp_dir, f"{job_id}_output.mp4")
                        with open(tmp_output, "wb") as f:
                            for chunk in r.iter_content(8192):
                                f.write(chunk)
                        st.video(tmp_output)
                        st.download_button(
                            "📥 Download MP4",
                            data=open(tmp_output, "rb").read(),
                            file_name=f"{job_id}_output.mp4",
                            mime="video/mp4",
                        )
                        download_shown = True
                    else:
                        st.warning(f"⚠️ Could not download result ({r.status_code})")
                except Exception as e:
                    st.error(f"❌ Download failed: {e}")
            break

        elif state.lower() == "failed":
            st.error(f"❌ Job failed: {status.get('error')}")
            break

        time.sleep(2)
