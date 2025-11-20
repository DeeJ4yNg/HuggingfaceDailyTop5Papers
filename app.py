import os
import json
import requests
import re
from io import BytesIO
from flask import Flask, render_template
from pypdf import PdfReader
from openai import OpenAI

app = Flask(__name__)

API_URL = "https://huggingface.co/api/daily_papers"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_API_BASE = os.getenv("OPENAI_API_BASE")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

client = None
if OPENAI_API_KEY:
    client = OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_API_BASE)

def fetch_papers(limit: int = 5):
    r = requests.get(f"{API_URL}?limit={limit}")
    r.raise_for_status()
    return r.json()

def extract_json(text: str) -> str:
    t = text.strip()
    if t.startswith("```"):
        t = t.strip("`")
        t = t.replace("json", "").strip()
    s = t.find("{")
    e = t.rfind("}")
    if s == -1 or e == -1:
        return "{}"
    return t[s:e+1]

def ensure_list(value):
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        return [x.strip() for x in value.split("\n") if x.strip()]
    return []

def chunk_text(text: str, size: int = 12000, overlap: int = 500):
    chunks = []
    i = 0
    n = len(text)
    while i < n:
        end = min(i + size, n)
        chunks.append(text[i:end])
        if end == n:
            break
        i = end - overlap
        if i < 0:
            i = 0
    return chunks

def llm_summary_json(title: str, body: str) -> dict:
    if not client:
        sents = [x.strip() for x in re.split(r'(?<=[.!?])\s+', body) if x.strip()]
        issue = sents[0] if sents else ""
        solution = sents[1] if len(sents) > 1 else ""
        steps = sents[2:6] if len(sents) > 2 else []
        impact = sents[-1] if sents else ""
        return {"en": {"issue":issue, "solution":solution, "steps":steps, "impact":impact}}
    messages = [
        {"role": "system", "content": "Return only JSON with keys: issue, solution, steps, impact. Steps should be a concise list."},
        {"role": "user", "content": f"Title: {title}\n\nPaper:\n{body}"}
    ]
    resp = client.chat.completions.create(model=OPENAI_MODEL, messages=messages, temperature=0.7, max_tokens=900)
    out = resp.choices[0].message.content if resp and resp.choices else "{}"
    data_text = extract_json(out)
    try:
        data = json.loads(data_text)
    except Exception:
        data = {"issue":"", "solution":"", "steps":[], "impact":""}
    return {
        "en": {
            "issue": str(data.get("issue", "")),
            "solution": str(data.get("solution", "")),
            "steps": ensure_list(data.get("steps", [])),
            "impact": str(data.get("impact", "")),
        }
    }

def dedup(values):
    seen = set()
    out = []
    for v in values:
        k = v.strip()
        if not k:
            continue
        if k not in seen:
            seen.add(k)
            out.append(k)
    return out

def translate_json_to_zh(obj: dict) -> dict:
    if not client:
        return obj
    prompt = (
        "Translate the following JSON values into Simplified Chinese. "
        "Do not change keys. Keep list structure. Return only JSON.\n\n" +
        json.dumps(obj, ensure_ascii=False)
    )
    messages = [
        {"role": "system", "content": "You are a professional translator. Translate values to Simplified Chinese. Return only JSON with the same keys."},
        {"role": "user", "content": prompt}
    ]
    resp = client.chat.completions.create(model=OPENAI_MODEL, messages=messages, temperature=0.1, max_tokens=900)
    out = resp.choices[0].message.content if resp and resp.choices else "{}"
    data_text = extract_json(out)
    try:
        data = json.loads(data_text)
    except Exception:
        data = obj
    return data

def translate_text_zh(text: str) -> str:
    if not client:
        return text
    messages = [
        {"role": "system", "content": "Translate to Simplified Chinese. Return only translated text."},
        {"role": "user", "content": text}
    ]
    resp = client.chat.completions.create(model=OPENAI_MODEL, messages=messages, temperature=0.1, max_tokens=300)
    return (resp.choices[0].message.content or text) if resp and resp.choices else text

def summarize_chunks(title: str, text: str) -> dict:
    result = llm_summary_json(title, text)
    en = result.get("en", {"issue":"", "solution":"", "steps":[], "impact":""})
    zh = translate_json_to_zh(en)
    return {"en": en, "zh": zh}

def summarize_paper(title: str, abstract: str, full_text: str | None = None):
    text = full_text if full_text else abstract
    if not text:
        return {"issue":"", "solution":"", "steps":[], "impact":""}
    return summarize_chunks(title, text)

def normalize_authors(a):
    if isinstance(a, list):
        return ", ".join([str(x) for x in a])
    if isinstance(a, str):
        return a
    return ""

def arxiv_link(p):
    aid = get_arxiv_id(p)
    if aid:
        return f"https://arxiv.org/abs/{aid}"
    return p.get("paperUrl") or p.get("url") or "#"

def arxiv_id_from_url(u: str | None):
    if not u:
        return None
    m = re.search(r"arxiv\.org\/(?:abs|pdf)\/(\d{4}\.\d{4,5})(?:v\d+)?", u)
    return m.group(1) if m else None

def get_arxiv_id(p):
    paper = p.get("paper") or {}
    return (
        p.get("arxivId")
        or paper.get("id")
        or arxiv_id_from_url(p.get("paperUrl"))
        or arxiv_id_from_url(p.get("url"))
    )

def arxiv_pdf_url(arxiv_id: str):
    return f"https://arxiv.org/pdf/{arxiv_id}.pdf"

def fetch_full_text(arxiv_id: str) -> str:
    try:
        pdf_url = arxiv_pdf_url(arxiv_id)
        r = requests.get(pdf_url, timeout=(10, 120))
        r.raise_for_status()
        reader = PdfReader(BytesIO(r.content))
        pages = [p.extract_text() or "" for p in reader.pages]
        return "\n".join(pages)
    except Exception:
        return ""

@app.route("/")
def index():
    papers = fetch_papers(5)
    items = []
    for p in papers:
        abstract = p.get("summary") or p.get("ai_summary") or ""
        aid = get_arxiv_id(p)
        full_text = fetch_full_text(aid) if aid else ""
        title = p.get("title", "")
        s = summarize_paper(title, abstract, full_text)
        title_zh = translate_text_zh(title) if client else title
        items.append({
            "title": title,
            "titleZh": title_zh,
            "authors": normalize_authors(p.get("authors")),
            "publishedAt": p.get("publishedAt", ""),
            "thumbnail": p.get("thumbnail", ""),
            "arxiv": arxiv_link(p),
            "summary": s
        })
    return render_template("index.html", items=items)

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=7860, debug=True)