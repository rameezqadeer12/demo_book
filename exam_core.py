# exam_core.py

import os, pickle, faiss, re
import numpy as np
from sentence_transformers import SentenceTransformer
from llama_cpp import Llama
from huggingface_hub import snapshot_download

# ⬇ AUTO DOWNLOAD FROM HUGGING FACE
snapshot_download(
    repo_id="rameezqadeer19/punjab-exam-rag",
    repo_type="model",
    local_dir=".",
    local_dir_use_symlinks=False
)

STORE = "rag_store"
MODEL_PATH = "model/mistral.gguf"

# ---- load rag ----
with open(os.path.join(STORE, "chunks.pkl"), "rb") as f:
    chunks = pickle.load(f)

index = faiss.read_index(os.path.join(STORE, "index.faiss"))
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# ---- load model ----
llm = Llama(
    model_path=MODEL_PATH,
    n_ctx=2048,
    n_threads=4,
    n_gpu_layers=0
)

# ---- retrieval ----
def retrieve(query, top_k=5):
    q_emb = embedder.encode([query], normalize_embeddings=True)
    scores, ids = index.search(q_emb.astype("float32"), top_k)
    return [chunks[i]["text"] for i in ids[0] if i != -1]

# ---- book answer ----
def build_book_answer(retrieved, max_chars=600):
    parts, total = [], 0
    for t in retrieved:
        t = re.sub(r"\s+", " ", t).strip()
        if len(t) < 40: continue
        if total + len(t) > max_chars:
            t = t[:max_chars-total]
        parts.append(t)
        total += len(t)
        if total >= max_chars: break
    return " ".join(parts)

# ---- main ask ----
def ask(question):

    if question.count("?") > 1:
        return "❌ Please ask only ONE question."

    retrieved = retrieve(question)
    book = build_book_answer(retrieved)

    if len(book) < 50:
        return "❌ NOT FOUND IN BOOK."

    prompt = f"""<s>[SYSTEM]
You are a strict school teacher. Only use BOOK ANSWER.
[/SYSTEM]
[USER]
BOOK ANSWER:
{book}

QUESTION:
{question}

FORMAT:
📘 Book Answer:
🧠 Teacher Explanation:
✏️ Solved Example:
✅ Final Answer:
[/USER]
[ASSISTANT]
"""

    out = llm(prompt, max_tokens=200, temperature=0.2)
    return out["choices"][0]["text"].strip()
