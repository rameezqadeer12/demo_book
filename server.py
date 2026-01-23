from fastapi import FastAPI
from pydantic import BaseModel
from hybridsystem import ask_llama

app = FastAPI(title="Punjab Exam AI")

class Question(BaseModel):
    question: str

@app.get("/")
def home():
    return {"status": "Punjab Exam AI running"}

@app.post("/ask")
def ask(q: Question):
    answer, sources = ask_llama(q.question)
    return {
        "question": q.question,
        "answer": answer,
        "sources": sources
    }
