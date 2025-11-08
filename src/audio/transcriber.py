import os, sys, json, ffmpeg
import subprocess
import whisper
from src.utils.config import Config
from src.utils.model_cache import ModelCache


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

def extract_audio(video_path: str, out_audio: str = None) -> str:
    """
    Extract mono 16kHz WAV for Whisper.
    """
    if out_audio is None:
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        out_audio = os.path.join(base_dir, "data", "processed", "audio.wav")

    os.makedirs(os.path.dirname(out_audio), exist_ok=True)

    (
        ffmpeg
        .input(video_path)
        .output(out_audio, ac=1, ar=16000)  # mono, 16kHz
        .overwrite_output()
        .run(quiet=True)
    )
    return out_audio


def transcribe_audio(audio_path: str, model_name: str = "tiny") -> list:
    """
    Transcribes an audio file using Whisper and returns a list of segments
    with timestamps and text.
    """
    print(f"Loading Whisper model: {model_name}")
    # model = whisper.load_model(model_name)
    model = ModelCache.load_whisper(model_name)
    print("Transcribing...")
    result = model.transcribe(audio_path, fp16=False, # CPU must be False
                            verbose=False,# keep logs clean
                            word_timestamps=False, # optional; set True if you need per-word timing
                            temperature=0.0,         # deterministic
                            condition_on_previous_text=False  # helps with segment drift on long files
                            )
    segments = []
    for seg in result.get("segments", []):
        segments.append({
            "text": seg.get("text", "").strip(),
            "start": float(seg.get("start", 0.0)),
            "end": float(seg.get("end", 0.0)),
        })
    return segments
    # return result["segments"]
