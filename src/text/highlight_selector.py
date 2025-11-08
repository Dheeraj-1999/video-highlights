import os
import json
import faiss
from string import Template
from openai import OpenAI
from sentence_transformers import SentenceTransformer
from src.utils.config import Config
import numpy as np
from datetime import datetime


client = OpenAI(api_key=Config.OPENAI_API_KEY)
# -----------------------------------------------------------
# Utility functions
# -----------------------------------------------------------

def load_index(index_path):
    """Load FAISS index from disk."""
    if not os.path.exists(index_path):
        raise FileNotFoundError(f"FAISS index not found: {index_path}")
    return faiss.read_index(index_path)


def load_chunks(chunk_path):
    """Load pre-computed text chunks."""
    if not os.path.exists(chunk_path):
        raise FileNotFoundError(f"Chunk file not found: {chunk_path}")
    with open(chunk_path, "r", encoding="utf-8") as f:
        return json.load(f)


# -----------------------------------------------------------
# Step 1 - Semantic retrieval
# -----------------------------------------------------------

def query_similar_chunks(query, top_k=10, index_path=None, chunk_path=None,
                         min_cosine=0.15, dynamic_topk=True, model_name: str = "all-mpnet-base-v2"):#"all-MiniLM-L6-v2"
    """
    Get top_k most semantically similar transcript chunks.
    Dynamically loads the correct FAISS index and chunk file.
    """
    if index_path is None:
        index_path = "data/processed/faiss_index.bin"
    if chunk_path is None:
        chunk_path = "data/processed/chunks.json"

    print(f"Querying top {top_k} relevant transcript chunks...")
    model = SentenceTransformer(model_name)

    index = load_index(index_path)
    chunks = load_chunks(chunk_path)
    q_emb = model.encode([query])
    q_emb = np.array(q_emb, dtype="float32")
    # q_emb = model.encode([query], convert_to_numpy=True, normalize_embeddings=False)
    faiss.normalize_L2(q_emb)

    if dynamic_topk:
        top_k = max(8, min(50, int((len(chunks) ** 0.5) * 2)))
    
    D, I = index.search(q_emb, top_k)
    print(f"🔍 Cosine score sample: {D[0][:10]}")
    results = []
    # print(f"⚠️ No results above cosine threshold {min_cosine}. Returning top_k fallback.")
    for idx, score in zip(I[0], D[0]):
        if idx < 0 or idx >= len(chunks):
            continue
        # FAISS returns *similarity* if index is normalized; else convert to cosine-ish
        cosine = float(score)
        if cosine >= min_cosine:
            c = chunks[idx]
            results.append({
                "text": c["text"],
                "start": float(c["start"]),
                "end": float(c["end"]),
                "score": cosine
            })
    if len(results) == 0:
        print(f"⚠️ No results above cosine threshold {min_cosine}. Returning top_k fallback.")
        # ✅ Always append fallback even if threshold logic misfires
        for idx, score in zip(I[0], D[0]):
            if 0 <= idx < len(chunks):
                c = chunks[idx]
                results.append({
                    "text": c["text"],
                    "start": float(c["start"]),
                    "end": float(c["end"]),
                    "score": float(score)
                })
                # results.append({
            #     "text": chunks[idx]["text"],
            #     "start": chunks[idx]["start"],
            #     "end": chunks[idx]["end"],
            #     "score": float(score)
            # })
    print(f"Retrieved {len(results)} candidate segments.")
    return results

def mmr_diversify(candidates, embedder=None, lambda_=0.7, max_items=12):
    """Re-rank candidates using Maximal Marginal Relevance (diversity)."""
    if not candidates:
        return []
    texts = [c["text"] for c in candidates]
    if embedder is None:
        embedder = SentenceTransformer("all-mpnet-base-v2")
    # print("🔍 Type of embedder:", type(embedder))
    E = embedder.encode(texts)#, convert_to_numpy=True, normalize_embeddings=True
    faiss.normalize_L2(E)

    selected_idx = []
    remaining = set(range(len(candidates)))
    best = int(np.argmax([c["score"] for c in candidates]))
    selected_idx.append(best)
    remaining.remove(best)

    while remaining and len(selected_idx) < max_items:
        rem_mat = E[list(remaining)]
        sel_mat = E[selected_idx]
        rel = np.array([candidates[i]["score"] for i in remaining])
        div = (rem_mat @ sel_mat.T).max(axis=1)
        mmr = lambda_ * rel - (1 - lambda_) * div
        pick_local = int(np.argmax(mmr))
        pick_global = list(remaining)[pick_local]
        selected_idx.append(pick_global)
        remaining.remove(pick_global)

    selected = [candidates[i] for i in selected_idx]
    return sorted(selected, key=lambda x: x["start"])
# ---------------------------------------------------------------------
# Keyword boost
# ---------------------------------------------------------------------
def keyword_boost(text, keywords):
    t = text.lower()
    boost = sum(1.0 for k in keywords if k in t)
    return 0.05 * boost  # 5% per keyword hit

def apply_keyword_boost(results, keywords):
    if not results:
        return []
    for r in results:
        r["score"] += keyword_boost(r["text"], keywords)
    results.sort(key=lambda x: x["score"], reverse=True)
    return results

# Multi-query retrieval (E)
def multi_query_union(queries, top_k, index_path, chunk_path,
                      min_cosine=0.15, embed_model="all-mpnet-base-v2"):
    all_cands = []
    for q in queries:
        all_cands += query_similar_chunks(
            q, top_k=top_k,
            index_path=index_path,
            chunk_path=chunk_path,
            min_cosine=min_cosine,
            model_name=embed_model
        )
    print(f"🔹 Before dedup: {len(all_cands)} total candidates")
    # de-duplicate by temporal overlap (~1s gap)
    all_cands.sort(key=lambda x: (x["start"]))#-x["score"],
    dedup = []
    for c in all_cands:
        # if not dedup or c["start"] - dedup[-1]["end"] > -1.0:
        #     dedup.append(c)
        if not dedup:
            dedup.append(c)
            continue

        # Merge only if there is a large overlap (so we don’t collapse time regions)
        if c["start"] <= dedup[-1]["end"]:
            # If they overlap, extend the last one’s end time
            dedup[-1]["end"] = max(dedup[-1]["end"], c["end"])
            dedup[-1]["score"] = max(dedup[-1]["score"], c["score"])
        else:
            dedup.append(c)
    print(f"🔹 After dedup: {len(dedup)} kept (gap threshold -1.0s)")
    return dedup


# Quality guardrails (H)
# ---------------------------------------------------------------------
def clean_segments(highlights, min_duration=1.0, merge_short=True):
    """
    Clean highlight segments:
    - Drops invalid ranges.
    - Keeps even short ones (>1s by default).
    - Optionally merges adjacent short clips.
    """
    if not highlights:
        return []

    # 1️⃣ Basic validation
    cleaned = []
    for s in highlights:
        start = float(s.get("start", 0))
        end = float(s.get("end", 0))
        if end > start and (end - start) >= min_duration:
            cleaned.append({
                "start": start,
                "end": end,
                "text": s.get("text", ""),
                "score": s.get("score", 0)
            })

    if not cleaned:
        print("⚠️ No valid segments after cleaning.")
        return []

    # 2️⃣ Sort chronologically
    cleaned.sort(key=lambda x: x["start"])

    # 3️⃣ Merge very close short clips (optional)
    if merge_short:
        merged = [cleaned[0]]
        for seg in cleaned[1:]:
            # If gap between clips is small (<1s), merge them
            if seg["start"] - merged[-1]["end"] < 1.0:
                merged[-1]["end"] = max(merged[-1]["end"], seg["end"])
                merged[-1]["score"] = max(merged[-1]["score"], seg["score"])
                merged[-1]["text"] += " " + seg.get("text", "")
            else:
                merged.append(seg)
        cleaned = merged

    print(f"🧹 Cleaned {len(cleaned)} segments (min_dur={min_duration}s, merged={merge_short})")
    return cleaned



# Unified pipeline helper for jobs.py
# ---------------------------------------------------------------------
def generate_candidate_highlights(
    index_path,
    chunk_path,
    embed_model="all-mpnet-base-v2",#"all-MiniLM-L6-v2",
    top_k=30,
    target_duration=60
):
    """
    High-level pipeline combining multi-query, keyword boost, and MMR.
    Returns clean, diverse candidate highlights.
    """

    print("🚀 Starting semantic highlight candidate generation...")

    # ---- Step 1: Multi-query retrieval ----
    # "video summary highlights",
    #     "most interesting moments",
    #     "key exciting parts",
    #     "important segments for summary",
    queries = [
        "fours and sixes",
        "wickets and catches",
        "loud voices and cheers",
    ]
    results = multi_query_union(
        queries, top_k=top_k,
        index_path=index_path,
        chunk_path=chunk_path,
        min_cosine=0.15,
        embed_model=embed_model
    )
    print(f"🔸 Total retrieved (multi-query): {len(results)}")

    # ---- Step 2: Keyword boosting ----
    KEYWORDS = [
        "exicement", "amazing", "incredible", "unbelievable", "dramatic",
        "goal", "score", "touchdown", "home run", "emotional", "missed",
        "highlight", "clutch", "comeback", "record", "championship", "centuary"
        "six", "four", "wicket", "appeal", "catch", "review",
        "out", "boundary", "milestone", "hundred", "fifty",
        "target", "win", "pressure", "decision", "goal",
        "achievement", "success", "deadline", "important", "dropped", "unplayable"
    ]
    boosted = apply_keyword_boost(results, KEYWORDS)
    print(f"🔸 After keyword boost: {len(boosted)}")

    # ---- Step 3: MMR diversification ----
    diverse = mmr_diversify(boosted, lambda_=0.7, max_items=15)
    print(f"🔸 After MMR diversification: {len(diverse)}")

    # ---- Step 4: Clean up segments ----
    cleaned = clean_segments(diverse)
    print(f"✅ Final candidate highlights: {len(cleaned)}")
    return cleaned


# Evaluation Logging (I)
# ---------------------------------------------------------------------
def log_highlight_summary(job_id, stage_counts, output_dir="data/processed"):
    """Save a lightweight CSV of summary stats per job."""
    import csv
    path = os.path.join(output_dir, "highlight_summary.csv")
    file_exists = os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["timestamp", "job_id", "chunks", "retrieved", "boosted", "diverse", "final"])
        row = [datetime.now().isoformat(), job_id] + stage_counts
        writer.writerow(row)

# Example integration stub (for jobs.py)
# ---------------------------------------------------------------------
def get_highlight_candidates(job_id, index_path, chunk_path, target_duration):
    """
    Example callable for jobs.py.
    Combines retrieval + cleaning. Handles LLM rerank externally.
    """
    candidates = generate_candidate_highlights(index_path, chunk_path)
    log_highlight_summary(job_id, [
        "?",  # chunks (you can fill from upstream)
        len(candidates),  # retrieved
        len(candidates),  # boosted
        len(candidates),  # diverse
        len(candidates)   # final
    ])
    return candidates

# -----------------------------------------------------------
# Step 2 - Load prompt template
# -----------------------------------------------------------

def load_prompt(template_name="highlight_prompt.txt"):
    """Load reusable prompt template from src/prompts folder."""
    prompt_path = os.path.join("src", "prompts", template_name)
    if not os.path.exists(prompt_path):
        raise FileNotFoundError(f"Prompt template not found: {prompt_path}")

    with open(prompt_path, "r", encoding="utf-8") as f:
        return Template(f.read())


# -----------------------------------------------------------
# Step 3 - Re-rank with LLM
# -----------------------------------------------------------

def rerank_with_llm(results, custom_prompt, target_duration=60, template_name="highlight_prompt.txt"):
    """
    Uses LLM to re-select highlight-worthy segments
    based on user's creative instructions.
    """
    template = load_prompt(template_name)
    results_json = json.dumps(results, indent=2)

    filled_prompt = template.substitute(
        target_duration=target_duration,
        custom_prompt=custom_prompt,
        results_json=results_json
    )

    print(f"🧠 Calling LLM for creative highlight selection...")
    # print(filled_prompt)
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": filled_prompt}],
        response_format={"type": "json_object"}
    )

    try:
        raw = response.choices[0].message.content
        data = json.loads(raw)

        # unwrap possible nesting keys
        if isinstance(data, dict):
            for key in ["segments", "selected_segments", "highlights"]:
                if key in data:
                    data = data[key]
                    break
        # if isinstance(data, dict):
        #     for key in ["segments", "selected_segments", "highlights"]:
        #         if key in data and isinstance(data[key], list):
        #             data = data[key]
        #             break
        else:
            print("⚠️ Unexpected format from LLM, using fallback results.")
            data = results

        return data

    except Exception as e:
        print(f"⚠️ Parsing fallback due to error: {e}")
        return results
