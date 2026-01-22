# app.py

from fastapi import FastAPI, Form
from fastapi.responses import HTMLResponse
from exam_core import ask

app = FastAPI()

HTML = """
<html>
<head>
<title>Punjab Exam AI</title>
<style>
body { font-family: Arial; background:#f4f6fb; }
.box { width:800px; margin:auto; margin-top:40px; background:white; padding:20px; border-radius:10px; }
textarea { width:100%; height:70px; }
pre { background:#111; color:#0f0; padding:10px; white-space:pre-wrap; }
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
def ask_q(q: str = Form(...)):
    ans = ask(q)
    return HTML.format(ans=ans)
