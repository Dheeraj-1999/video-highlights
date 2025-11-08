from typing import List, Dict


def merge_segments(segments, max_chunk=25.0, merge_gap=2.0):
    chunks = []
    cur = []
    cur_start, cur_end = None, None

    for seg in segments:
        s, e, text = seg["start"], seg["end"], seg["text"]
        if cur and (s - cur_end > merge_gap or e - cur_start > max_chunk):
            chunks.append({"start": cur_start, "end": cur_end, "text": " ".join(cur)})
            cur, cur_start = [], None
        if cur_start is None:
            cur_start = s
        cur_end = e
        cur.append(text)

    if cur:
        chunks.append({"start": cur_start, "end": cur_end, "text": " ".join(cur)})
    return chunks


# def merge_segments(segments: List[Dict], chunk_seconds: float = 15.0) -> List[Dict]:
#     """
#     Merge small Whisper segments into ~chunk_seconds windows.
#     Keeps start/end timestamps for later video cutting.
#     """
#     chunks, buf, start_t = [], [], None
#     for seg in segments:
#         s, e, txt = seg["start"], seg["end"], seg["text"].strip()
#         if start_t is None:
#             start_t = s
#         buf.append(txt)
#         # when buffer duration passes threshold, finalize chunk
#         if e - start_t >= chunk_seconds:
#             chunks.append({
#                 "start": start_t,
#                 "end": e,
#                 "text": " ".join(buf)
#             })
#             buf, start_t = [], None
#     # leftovers
#     if buf:
#         chunks.append({
#             "start": start_t or 0.0,
#             "end": segments[-1]["end"],
#             "text": " ".join(buf)
#         })
#     return chunks
