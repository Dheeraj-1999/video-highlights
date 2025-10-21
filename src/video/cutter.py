import os
import json
from moviepy.editor import VideoFileClip, concatenate_videoclips
from src.utils.config import Config

def load_highlight_candidates(path="data/processed/highlight_candidates.json"):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["segments"] if isinstance(data, dict) and "segments" in data else data


# from moviepy.editor import VideoFileClip, concatenate_videoclips

def extract_clips(video_path: str, highlights, fade_duration=0.3):
    base_video = VideoFileClip(video_path)
    video_duration = base_video.duration
    clips = []

    for h in highlights:
        try:
            start, end = h["start"], h["end"]
            # clamp within bounds
            start = max(0, start)
            end = min(video_duration, end)
            if end - start < 2:
                continue

            clip = base_video.subclip(start, end)

            if clip.audio is None and base_video.audio is not None:
                clip = clip.set_audio(base_video.audio.subclip(start, end))

            clip = clip.fadein(fade_duration).fadeout(fade_duration)
            clips.append(clip)

        except Exception as e:
            print(f"⚠️ Skipping invalid segment {h}: {e}")
    # base_video.close()
    return clips, base_video

def limit_highlight_duration(segments, max_total_seconds):
    """
    Trim or drop segments so total duration ≈ max_total_seconds.
    Keeps segments in ranked order.
    """
    selected = []
    total = 0.0

    for seg in segments:
        seg_duration = seg["end"] - seg["start"]
        if total + seg_duration <= max_total_seconds:
            selected.append(seg)
            total += seg_duration
        else:
            remaining = max_total_seconds - total
            if remaining > 2:  # keep at least 2-sec fragment
                seg["end"] = seg["start"] + remaining
                selected.append(seg)
                total += remaining
            break

    print(f"🎯 Trimmed highlight duration to {round(total,1)} s "
          f"(target {max_total_seconds} s, {len(selected)} segments)")
    return selected


def create_highlight_reel(video_path, highlight_file="data/processed/highlight_candidates.json"):
    highlights = load_highlight_candidates(highlight_file)
    print(f"🎯 Loaded {len(highlights)} highlight candidates")

    clips, base_video = extract_clips(video_path, highlights)
    if not clips:
        print("⚠️ No valid highlight clips found.")
        return None

    final = concatenate_videoclips(clips, method="compose")
    output_path = os.path.join(Config.PROCESSED_DIR, "highlight_reel.mp4")

    final.write_videofile(
        output_path,
        codec="libx264",
        audio_codec="aac",
        temp_audiofile="temp-audio.m4a",
        remove_temp=True,
        verbose=True
    )
    print(f"✅ Highlight reel created: {output_path}")
    final.close()
    base_video.close()
    return output_path

if __name__ == "__main__":
    video_path = "data/raw/sample.mp4"
    create_highlight_reel(video_path)
