import os
import json
import faiss
from string import Template
from openai import OpenAI
from sentence_transformers import SentenceTransformer
from src.utils.config import Config


client = OpenAI(api_key=Config.OPENAI_API_KEY)
# -----------------------------------------------------------
# Utility functions
# -----------------------------------------------------------

def load_index(index_path="data/processed/faiss_index.bin"):
    """Load FAISS index from disk."""
    return faiss.read_index(index_path)


def load_chunks(chunk_path="data/processed/chunks.json"):
    """Load pre-computed text chunks."""
    with open(chunk_path, "r", encoding="utf-8") as f:
        return json.load(f)


# -----------------------------------------------------------
# Step 1 - Semantic retrieval
# -----------------------------------------------------------

def query_similar_chunks(query, top_k=8):
    """Get top_k most semantically similar transcript chunks."""
    print(f"Querying top {top_k} relevant transcript chunks...")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    index = load_index()
    chunks = load_chunks()

    q_emb = model.encode([query], convert_to_numpy=True)
    faiss.normalize_L2(q_emb)

    D, I = index.search(q_emb, top_k)
    results = []
    for idx, score in zip(I[0], D[0]):
        if idx >= 0:
            results.append({
                "text": chunks[idx]["text"],
                "start": chunks[idx]["start"],
                "end": chunks[idx]["end"],
                "score": float(score)
            })

    print(f"Retrieved {len(results)} candidate segments.")
    return results


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
    print(filled_prompt)
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
