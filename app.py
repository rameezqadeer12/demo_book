# app.py

from fastapi import FastAPI, Form, Request, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from exam_core import ask

# 🔐 YOUR PERMANENT API KEY
API_KEY = "PUNJAB-EXAM-2026"

app = FastAPI()


# ---------------------------
# SIMPLE FRONTEND (NO KEY)
# ---------------------------
HTML = """
<html>
<head>
<title>Punjab Exam AI</title>
<style>
body { font-family: Arial; background:#f4f6fb; }
.box { width:800px; margin:auto; margin-top:40px; background:white; padding:20px; border-radius:10px; }
textarea { width:100%; height:70px; font-size:16px; }
pre { background:#111; color:#0f0; padding:12px; white-space:pre-wrap; }
button { padding:10px 20px; font-size:16px; }
</style>
</head>
<body>
<div class="box">
<h2>📘 Punjab Exam AI – Demo</h2>
<form method="post">
<textarea name="q" placeholder="Type one exam question..."></textarea><br><br>
<button>Ask</button>
</form>
<pre>{ans}</pre>
</div>
</body>
</html>
"""

@app.get("/", response_class=HTMLResponse)
def home():
    return HTML.format(ans="")

@app.post("/", response_class=HTMLResponse)
def web_ask(q: str = Form(...)):
    ans = ask(q)
    return HTML.format(ans=ans)


# ---------------------------
# 🔐 PROTECTED API (KEY REQUIRED)
# ---------------------------
@app.post("/api/ask")
async def api_ask(request: Request, question: str = Form(...)):

    user_key = request.headers.get("X-API-KEY")

    if user_key != API_KEY:
        raise HTTPException(status_code=401, detail="❌ Invalid API Key")

    answer = ask(question)

    return JSONResponse({
        "question": question,
        "answer": answer,
        "status": "success"
    })
