# ================================
# Punjab Exam AI - Hybrid System
# GitHub: code only
# HuggingFace: model + FAISS store
# GPU + Server ready
# ================================

import os, pickle, faiss, re
import numpy as np
from sentence_transformers import SentenceTransformer
from llama_cpp import Llama
from huggingface_hub import hf_hub_download

# ========= 0. SETTINGS =========

HF_REPO = "rameezqadeer19/punjab-exam-rag"
BASE_DIR = "hf_assets"
GPU_LAYERS = int(os.getenv("GPU_LAYERS", "35"))   # 0 = CPU, 20–50 = GPU
N_THREADS = int(os.getenv("N_THREADS", "8"))

os.makedirs(BASE_DIR, exist_ok=True)

# ========= 1. DOWNLOAD FROM HUGGINGFACE =========

print("\n🔄 Downloading assets from HuggingFace...")

GGUF_PATH = hf_hub_download(
    repo_id=HF_REPO,
    filename="model/mistral.gguf",
    local_dir=BASE_DIR
)

CHUNKS_PATH = hf_hub_download(
    repo_id=HF_REPO,
    filename="rag_store/chunks.pkl",
    local_dir=BASE_DIR
)

INDEX_PATH = hf_hub_download(
    repo_id=HF_REPO,
    filename="rag_store/index.faiss",
    local_dir=BASE_DIR
)

print("✅ Assets ready")

# ========= 2. LOAD FAISS + CHUNKS =========

print("🔄 Loading exam knowledge base...")

with open(CHUNKS_PATH, "rb") as f:
    chunks = pickle.load(f)

index = faiss.read_index(INDEX_PATH)
embedder = SentenceTransformer("all-MiniLM-L6-v2")

print(f"✅ Knowledge base loaded | Chunks: {len(chunks)} | FAISS: {index.ntotal}")

# ========= 3. LOAD LLaMA =========

print("🔄 Loading LLaMA model...")

llm = Llama(
    model_path=GGUF_PATH,
    n_ctx=2048,
    n_threads=N_THREADS,
    n_gpu_layers=GPU_LAYERS
)

print(f"✅ LLaMA loaded | GPU layers: {GPU_LAYERS}")

# ========= 4. RETRIEVER =========

def retrieve(query, top_k=5):
    q_emb = embedder.encode([query], normalize_embeddings=True)
    scores, ids = index.search(q_emb.astype("float32"), top_k)

    results = []
    for i, idx in enumerate(ids[0]):
        if idx == -1:
            continue

        r = chunks[idx]
        results.append({
            "text": r["text"],
            "source": r.get("source",""),
            "chunk_id": idx,
            "score": float(scores[0][i])
        })

    return results

# ========= 5. BOOK ANSWER BUILDER =========

def build_book_answer(retrieved, max_chars=600):
    parts, total = [], 0

    for r in retrieved:
        t = re.sub(r"\s+", " ", r["text"]).strip()

        if len(t) < 40:
            continue

        if total + len(t) > max_chars:
            t = t[:max_chars-total]

        parts.append(t)
        total += len(t)

        if total >= max_chars:
            break

    return " ".join(parts)

# ========= 6. PROMPT BUILDER =========

def build_prompt(book_answer, question):

    system = """You are a strict school teacher.
Only use the BOOK ANSWER.
If not found, say: NOT FOUND IN BOOK.
Explain simply like a teacher.
"""

    return f"""<s>[SYSTEM]
{system}
[/SYSTEM]
[USER]
BOOK ANSWER:
{book_answer}

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

# ========= 7. MAIN ASK FUNCTION =========

def ask_llama(question, top_k=5):
    retrieved = retrieve(question, top_k)

    if not retrieved:
        return "❌ NOT FOUND IN BOOK.", []

    book_answer = build_book_answer(retrieved)

    if len(book_answer) < 50:
        return "❌ NOT FOUND IN BOOK.", retrieved

    prompt = build_prompt(book_answer, question)

    out = llm(prompt, max_tokens=250, temperature=0.2)
    answer = out["choices"][0]["text"].strip()

    return answer, retrieved

# ========= 8. TERMINAL MODE =========

if __name__ == "__main__":
    print("\n🎓 PUNJAB EXAM AI READY (TERMINAL MODE)")
    print("Type a question.  Type 'exit' to stop.\n")

    while True:
        q = input("❓ Question: ").strip()
        if q.lower() == "exit":
            break

        ans, src = ask_llama(q)

        print("\n=========== ANSWER ===========\n")
        print(ans)

        print("\n=========== SOURCES ==========\n")
        for s in src:
            print(s["source"], "| chunk", s["chunk_id"], "| score", round(s["score"],3))

        print("\n==============================\n")
