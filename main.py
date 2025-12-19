import os
import json
import re
import asyncio
from io import BytesIO
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta

# 加载.env文件
from dotenv import load_dotenv
load_dotenv(override=True)

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from pypdf import PdfReader
from openai import AsyncOpenAI

app = FastAPI()

templates = Jinja2Templates(directory="templates")

API_URL = "https://huggingface.co/api/daily_papers"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_API_BASE = os.getenv("OPENAI_API_BASE")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

# File to store shown papers history
SHOWN_PAPERS_FILE = "shown_papers.json"
# File to cache processed papers
CACHE_FILE = "papers_cache.json"

client = None
if OPENAI_API_KEY:
    client = AsyncOpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_API_BASE)

# Global variable to hold cached papers
cached_papers = None

def load_shown_papers():
    """Load the list of papers that have been shown before"""
    if os.path.exists(SHOWN_PAPERS_FILE):
        try:
            with open(SHOWN_PAPERS_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except:
            pass
    return []

def save_shown_papers(paper_ids):
    """Save the list of shown paper IDs"""
    shown_papers = load_shown_papers()
    shown_papers.extend(paper_ids)
    # Remove duplicates
    shown_papers = list(set(shown_papers))
    with open(SHOWN_PAPERS_FILE, 'w', encoding='utf-8') as f:
        json.dump(shown_papers, f, indent=2)

def save_papers_cache(papers):
    """Save processed papers to cache file"""
    with open(CACHE_FILE, 'w', encoding='utf-8') as f:
        json.dump(papers, f, indent=2, default=str)

def load_papers_cache():
    """Load processed papers from cache file"""
    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except:
            pass
    return None

async def fetch_papers(limit: int = 5):
    try:
        async with httpx.AsyncClient() as http_client:
            # Get all papers and sort by upvotes descending
            r = await http_client.get(f"{API_URL}?limit=50")  # Get more papers to ensure we find the top ones
            r.raise_for_status()
            papers = r.json()
            print(f"Fetched {len(papers)} papers from API")
            
            # Load shown papers history
            shown_papers = load_shown_papers()
            print(f"Loaded {len(shown_papers)} shown papers")
            
            # Filter out papers that have been shown before
            new_papers = []
            for paper in papers:
                paper_id = paper['paper']['id'] if 'paper' in paper and 'id' in paper['paper'] else paper['id']
                if paper_id not in shown_papers:
                    new_papers.append(paper)
            
            print(f"Filtered to {len(new_papers)} new papers")
            
            # Sort papers by upvotes descending
            if new_papers and isinstance(new_papers, list) and 'paper' in new_papers[0] and 'upvotes' in new_papers[0]['paper']:
                new_papers.sort(key=lambda x: x['paper'].get('upvotes', 0), reverse=True)
                print("Sorted papers by upvotes descending")
            
            # Return only the top N papers
            selected_papers = new_papers[:limit]
            
            # Save the IDs of the selected papers to shown history
            paper_ids = [p['paper']['id'] if 'paper' in p and 'id' in p['paper'] else p['id'] for p in selected_papers]
            save_shown_papers(paper_ids)
            print(f"Saved {len(paper_ids)} papers to shown history")
            
            return selected_papers
    except Exception as e:
        print(f"Error fetching papers: {str(e)}")
        return []

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

async def llm_summary_json(title: str, body: str) -> dict:
    if not client:
        # Fallback if no OpenAI client
        sents = [x.strip() for x in re.split(r'(?<=[.!?])\s+', body) if x.strip()]
        issue = sents[0] if sents else ""
        solution = sents[1] if len(sents) > 1 else ""
        steps = sents[2:6] if len(sents) > 2 else []
        impact = sents[-1] if sents else ""
        return {"en": {"issue": issue, "solution": solution, "steps": steps, "impact": impact}}
    
    messages = [
        {"role": "system", "content": "You are an AI researcher that summarizes research papers. The paper is in English. Help user to summarize the paper into JSON format. And return only JSON with keys: issue, solution, steps, impact. Steps should be a concise list."},
        {"role": "user", "content": f"Title: {title}\n\nPaper:\n{body}"}
    ]
    try:
        resp = await client.chat.completions.create(model=OPENAI_MODEL, messages=messages, temperature=0.7)
        out = resp.choices[0].message.content if resp and resp.choices else "{}"
        data_text = extract_json(out)
        data = json.loads(data_text)
    except Exception:
        data = {"issue": "", "solution": "", "steps": [], "impact": ""}
    
    return {
        "en": {
            "issue": str(data.get("issue", "")),
            "solution": str(data.get("solution", "")),
            "steps": ensure_list(data.get("steps", [])),
            "impact": str(data.get("impact", "")),
        }
    }

async def translate_json_to_zh(obj: dict) -> dict:
    if not client:
        return obj
    prompt = (
        "Translate the following JSON values into Simplified Chinese. "
        "Do not change keys. Keep list structure. Return only JSON.\n\n" +
        json.dumps(obj, ensure_ascii=False)
    )
    messages = [
        {"role": "system", "content": "You are a professional translator and AI researcher. Translate values to Simplified Chinese. Return only JSON with the same keys."},
        {"role": "user", "content": prompt}
    ]
    try:
        resp = await client.chat.completions.create(model=OPENAI_MODEL, messages=messages, temperature=0.1, max_tokens=900)
        out = resp.choices[0].message.content if resp and resp.choices else "{}"
        data_text = extract_json(out)
        data = json.loads(data_text)
    except Exception as e:
        print(f"Translation error: {str(e)}")
        data = obj
    return data

async def translate_text_zh(text: str) -> str:
    if not client:
        return text
    messages = [
        {"role": "system", "content": "You are a professional translator. Translate the given text to Simplified Chinese. Return ONLY the translated text, NOTHING ELSE. No explanations, no options, no additional comments, no original text. Just the direct translation."},
        {"role": "user", "content": text}
    ]
    try:
        resp = await client.chat.completions.create(model=OPENAI_MODEL, messages=messages, temperature=0.1, max_tokens=300)
        translated = (resp.choices[0].message.content or text) if resp and resp.choices else text
        # Clean up any duplicates or extra content
        lines = translated.split('\n')
        cleaned = []
        seen = set()
        for line in lines:
            line = line.strip()
            if line and line not in seen:
                seen.add(line)
                cleaned.append(line)
        return ' '.join(cleaned) if cleaned else text
    except Exception:
        return text

async def summarize_chunks(title: str, text: str) -> dict:
    # Logic: summarize then translate
    # Note: original code only summarizes the first chunk effectively or passes full text to llm_summary_json?
    # Original llm_summary_json takes 'body'. 'summarize_paper' calls 'summarize_chunks'.
    # But 'summarize_chunks' in original code just calls 'llm_summary_json(title, text)'.
    # It doesn't actually chunk in the original code snippet I saw, despite the name 'summarize_chunks' being there and 'chunk_text' being defined.
    # Wait, let me check the original 'summarize_chunks' implementation again.
    # 129: def summarize_chunks(title: str, text: str) -> dict:
    # 130:    result = llm_summary_json(title, text)
    # It seems the original code intended to chunk but currently just passes the whole text (or abstract).
    # I will stick to the original behavior for now to avoid complexity, 
    # but I'll make it async.
    
    result = await llm_summary_json(title, text)
    en = result.get("en", {"issue": "", "solution": "", "steps": [], "impact": ""})
    zh = await translate_json_to_zh(en)
    return {"en": en, "zh": zh}

async def summarize_paper(title: str, abstract: str, full_text: str | None = None):
    text = full_text if full_text else abstract
    if not text:
        return {"issue": "", "solution": "", "steps": [], "impact": ""}
    # If text is too long, we might need to truncate or chunk.
    # For now, let's just take the first 12000 chars to be safe with tokens if full_text is used.
    if len(text) > 200000:
        text = text[:200000]
    return await summarize_chunks(title, text)

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

async def fetch_full_text(arxiv_id: str) -> str:
    try:
        pdf_url = arxiv_pdf_url(arxiv_id)
        async with httpx.AsyncClient() as client:
            r = await client.get(pdf_url, timeout=30.0, follow_redirects=True)
            r.raise_for_status()
            # PdfReader is synchronous, but for small files it's fast. 
            # For strict async, run in executor.
            loop = asyncio.get_event_loop()
            text = await loop.run_in_executor(None, _read_pdf, r.content)
            return text
    except Exception:
        return ""

def _read_pdf(content: bytes) -> str:
    try:
        reader = PdfReader(BytesIO(content))
        pages = [p.extract_text() or "" for p in reader.pages]
        return "\n".join(pages)
    except Exception:
        return ""

async def process_paper(p: dict) -> dict:
    abstract = p.get("summary") or p.get("ai_summary") or ""
    aid = get_arxiv_id(p)
    
    full_text = ""
    if aid:
        full_text = await fetch_full_text(aid)
        
    title = p.get("title", "")
    
    # Run summarization and translation concurrently if possible? 
    # Actually summarize needs to finish before translation (usually), but translation of title can be parallel.
    
    # We can parallelize title translation and summary generation with error handling
    print(f"Processing paper {title} with arxiv id {aid}")
    
    async def safe_summarize():
        try:
            return await summarize_paper(title, abstract, full_text)
        except Exception as e:
            print(f"Error summarizing paper {title}: {str(e)}")
            return {"en": {"issue": "", "solution": "", "steps": [], "impact": ""}, "zh": {"issue": "", "solution": "", "steps": [], "impact": ""}}

    async def safe_translate():
        try:
            if client:
                return await translate_text_zh(title)
            return title
        except Exception as e:
            print(f"Error translating title {title}: {str(e)}")
            return title

    s_task = safe_summarize()
    t_task = safe_translate()
    
    # Wait for both tasks to complete even if one fails
    s, title_zh = await asyncio.gather(s_task, t_task)
    print(f"Finished processing paper {title} with arxiv id {aid}")

    return {
        "title": title,
        "titleZh": title_zh,
        "authors": normalize_authors(p.get("authors")),
        "publishedAt": p.get("publishedAt", ""),
        "thumbnail": p.get("thumbnail", ""),
        "arxiv": arxiv_link(p),
        "upvotes": p.get("paper", {}).get("upvotes", 0),
        "summary": s
    }

async def initialize_papers():
    """Initialize papers cache when app starts"""
    global cached_papers
    print("Initializing papers cache...")
    
    # Try to load from cache first
    cached_papers = load_papers_cache()
    
    if cached_papers:
        print(f"Loaded {len(cached_papers)} papers from cache")
        return
    
    # If no cache, fetch and process papers
    print("No cache found, fetching new papers...")
    papers = await fetch_papers(5)
    print(f"Processing {len(papers)} papers")
    
    # Process all papers in parallel!
    tasks = [process_paper(p) for p in papers]
    items = await asyncio.gather(*tasks)
    
    print(f"Processed {len(items)} papers successfully")
    cached_papers = items
    
    # Save to cache file
    save_papers_cache(items)
    print("Saved papers to cache")

@app.on_event("startup")
async def startup_event():
    """Run when app starts"""
    await initialize_papers()

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    try:
        print("Starting index route")
        
        if not cached_papers:
            # Fallback if cache is not initialized
            await initialize_papers()
        
        print(f"Using cached papers: {len(cached_papers)} papers")
        # Debug: Print all paper titles
        print("Papers being passed to template:")
        for i, paper in enumerate(cached_papers):
            print(f"  {i+1}. {paper['title']}")
        
        # Get current date
        current_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        return templates.TemplateResponse("index.html", {"request": request, "items": cached_papers, "current_date": current_date})
    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        print(f"Error in index route: {str(e)}")
        print(f"Traceback: {error_trace}")
        # 返回一个简单的错误页面
        error_html = """
        <html>
            <head><title>Error</title></head>
            <body>
                <h1>Internal Server Error</h1>
                <p>{error}</p>
            </body>
        </html>
        """.format(error=str(e))
        return HTMLResponse(content=error_html, status_code=500)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=7860, reload=True)
