import json
from typing import List, Dict

def merge_segments(segments: List[Dict], chunk_seconds: float = 30.0) -> List[Dict]:
    """
    Merge small Whisper segments into ~chunk_seconds windows.
    Keeps start/end timestamps for later video cutting.
    """
    chunks, buf, start_t = [], [], None
    for seg in segments:
        s, e, txt = seg["start"], seg["end"], seg["text"].strip()
        if start_t is None:
            start_t = s
        buf.append(txt)
        # when buffer duration passes threshold, finalize chunk
        if e - start_t >= chunk_seconds:
            chunks.append({
                "start": start_t,
                "end": e,
                "text": " ".join(buf)
            })
            buf, start_t = [], None
    # leftovers
    if buf:
        chunks.append({
            "start": start_t or 0.0,
            "end": segments[-1]["end"],
            "text": " ".join(buf)
        })
    return chunks


if __name__ == "__main__":
    # quick standalone test
    with open("data/processed/transcript_segments.json") as f:
        segs = json.load(f)
    merged = merge_segments(segs)
    with open("data/processed/chunks.json", "w") as f:
        json.dump(merged, f, indent=2)
    print(f"Created {len(merged)} chunks")
