import os
import json
from moviepy.editor import VideoFileClip, concatenate_videoclips
from src.utils.config import Config
import numpy as np

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
            print(f"Skipping invalid segment {h}: {e}")
    # base_video.close()
    return clips, base_video

def limit_highlight_duration(segments, max_total_seconds=360.0):
    """
    Trim or drop segments so total duration ≈ max_total_seconds.
    Keeps segments in ranked order.
    """

    if not segments:
        return []

    # Keep chronological order
    ranked = sorted(segments, key=lambda x: x["start"])
    selected, total = [], 0.0
    refined = []
    # --- 1️⃣ Split long clips only
    for seg in ranked:
        start, end = float(seg["start"]), float(seg["end"])
        dur = end - start
        if dur > 40:
            step = 20
            for s in np.arange(start, end, step):
                e = min(s + step, end)
                refined.append({"start": s, "end": e, "score": seg["score"], "text": seg["text"]})
        else:
            refined.append(seg)

    # --- 2️⃣ Pick sequentially until max_total_seconds
    for seg in refined:
        dur = seg["end"] - seg["start"]
        if total + dur <= max_total_seconds:
            selected.append(seg)
            total += dur
        else:
            break

    print(f"🎯 Trimmed highlight duration to {total:.1f}s (target {max_total_seconds}s, {len(selected)} segments)")
    return selected
################# old Code #####################
    # selected = []
    # total = 0.0

    # for seg in segments:
    #     seg_duration = seg["end"] - seg["start"]
    #     if total + seg_duration <= max_total_seconds:
    #         selected.append(seg)
    #         total += seg_duration
    #     else:
    #         remaining = max_total_seconds - total
    #         if remaining > 2:  # keep at least 2-sec fragment
    #             seg["end"] = seg["start"] + remaining
    #             selected.append(seg)
    #             total += remaining
    #         break

    # print(f"🎯 Trimmed highlight duration to {round(total,1)} s "
    #       f"(target {max_total_seconds} s, {len(selected)} segments)")
    # return selected


def pad_and_merge_segments(segments, pad=1.5, merge_gap=2.0, video_duration=None):
    """
    Smooth segments by adding padding and merging near-adjacent ones,
    while preserving metadata fields like 'score' and 'text'.
    """
    if not segments:
        return []

    segs = sorted(segments, key=lambda x: x["start"])
    merged = []

    for seg in segs:
        s = max(0, seg["start"] - pad)
        e = seg["end"] + pad
        if video_duration:
            e = min(e, video_duration)

        # Preserve meta
        new_seg = {
            "start": s,
            "end": e,
            "text": seg.get("text", ""),
            "score": seg.get("score", 0.0)
        }

        if not merged:
            merged.append(new_seg)
        else:
            last = merged[-1]
            if s - last["end"] <= merge_gap:
                # Merge: extend the last segment and keep max score/text concatenation
                last["end"] = max(last["end"], e)
                last["score"] = max(last.get("score", 0.0), new_seg.get("score", 0.0))
                if seg.get("text"):
                    last["text"] = (last.get("text", "") + " " + seg["text"]).strip()
            else:
                merged.append(new_seg)

    print(f"🪄 Padded {len(segs)} → {len(merged)} merged segments (pad={pad}s, gap={merge_gap}s)")
    return merged



def create_highlight_reel(video_path, highlights=None, highlight_file="data/processed/highlight_candidates.json"):
    if highlights is None:
        highlights = load_highlight_candidates(highlight_file)
    else:
        print(f"🎯 Using in-memory highlights ({len(highlights)})")

    if not highlights:
        print("⚠️ No highlight segments found.")
        return None    
    # highlights = load_highlight_candidates(highlight_file)
    highlights = sorted(highlights, key=lambda x: x["start"])
    print(f"🎯 Loaded {len(highlights)} highlight candidates")

    clips, base_video = extract_clips(video_path, highlights)
    if not clips:
        print("No valid highlight clips found.")
        return None
    for i, clip in enumerate(clips):
        clips[i] = clip.crossfadein(0.3).crossfadeout(0.3)
    final = concatenate_videoclips(clips, method="compose")
    output_path = os.path.join(Config.PROCESSED_DIR, "highlight_reel.mp4")

    final.write_videofile(
        output_path,
        codec="libx264",
        audio_codec="aac",
        temp_audiofile="temp-audio.m4a",
        remove_temp=True,
        threads=2,
        write_logfile=False,
        verbose=True
    )
    print(f"✅ Highlight reel created: {output_path}")
    final.close()
    base_video.close()
    return output_path
