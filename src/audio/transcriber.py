import os, sys, json
import subprocess
import whisper
from src.utils.config import Config

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

def extract_audio(video_path: str) -> str:
    """
    Extracts audio from a video using ffmpeg and returns audio file path.
    """
    output_audio = os.path.join(Config.PROCESSED_DIR, "audio.wav")
    cmd = [
        "ffmpeg", "-y", "-i", video_path,
        "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1",
        output_audio
    ]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return output_audio


def transcribe_audio(audio_path: str, model_size: str = "tiny") -> list:
    """
    Transcribes an audio file using Whisper and returns a list of segments
    with timestamps and text.
    """
    print(f"Loading Whisper model: {model_size}")
    model = whisper.load_model(model_size)
    print("Transcribing...")
    result = model.transcribe(audio_path, fp16=False, verbose=False, chunk_length=15)
    return result["segments"]
