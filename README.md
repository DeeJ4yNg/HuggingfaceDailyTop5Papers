# Daily AI Papers Summarizer

Fetches the top 5 daily papers from Hugging Face Daily Papers API, downloads the full arXiv PDF, sends the full paper text to an LLM (OpenAI API format), generates structured English summaries, and translates them into Simplified Chinese. Renders a polished Bootstrap UI with arXiv links.

## Features
- Fetches latest papers from `https://huggingface.co/api/daily_papers`.
- Extracts arXiv ID from `arxivId`, `paper.id`, or `paperUrl`/`url`.
- Downloads the full PDF and extracts text using `pypdf`.
- Full-context summarization via an OpenAI-compatible Chat Completions API.
- Bilingual output: English summary is translated into Simplified Chinese.
- Clean, responsive UI built with Bootstrap.
- Paper caching system to avoid redundant processing
- Async processing for improved performance

## Prerequisites
- Python 3.10+ (3.11 recommended)
- Internet access to reach Hugging Face API and arXiv PDFs
- OpenAI‑compatible API credentials (key, base URL, model)

## Quickstart (Windows PowerShell)
1. Create a virtual environment and install dependencies:
   ```powershell
   python -m venv .venv
   .\.venv\Scripts\python.exe -m pip install -r requirements.txt
   ```
2. Configure your OpenAI‑compatible endpoint:
   ```powershell
   $env:OPENAI_API_KEY = "<your_key>"
   $env:OPENAI_API_BASE = "<https://your.openai-compatible.endpoint>"
   $env:OPENAI_MODEL = "gpt-4o-mini"   # or your provider's model name
   ```
3. Run the server:
   ```powershell
   .\.venv\Scripts\python.exe main.py
   ```
4. Open `http://127.0.0.1:7860/`.

## Quickstart (uv)
```bash
uv venv
uv pip install -r requirements.txt
export OPENAI_API_KEY="<your_key>"
export OPENAI_API_BASE="<https://your.openai-compatible.endpoint>"
export OPENAI_MODEL="gpt-4o-mini"
uv run python main.py
```
Visit `http://127.0.0.1:7860/`.

## Quickstart (conda)
```bash
conda create -n hotpapers python=3.11
conda activate hotpapers
pip install -r requirements.txt
export OPENAI_API_KEY="<your_key>"
export OPENAI_API_BASE="<https://your.openai-compatible.endpoint>"
export OPENAI_MODEL="gpt-4o-mini"
python main.py
```
Visit `http://127.0.0.1:7860/`.

## Quickstart (macOS/Linux)
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export OPENAI_API_KEY="<your_key>"
export OPENAI_API_BASE="<https://your.openai-compatible.endpoint>"
export OPENAI_MODEL="gpt-4o-mini"
python main.py
```
Visit `http://127.0.0.1:7860/`.

## Configuration
- `OPENAI_API_KEY`: API key for your LLM provider (OpenAI API format)
- `OPENAI_API_BASE`: Base URL for the Chat Completions endpoint
- `OPENAI_MODEL`: Model name to use
- Paper count: `fetch_papers(5)` in `main.py`
- Port/host: Configured in `main.py` at the bottom

## How It Works
- Fetch papers from Hugging Face: `main.py`
- Extract arXiv ID: `main.py`
- Build arXiv link: `main.py`
- Download and extract full PDF text: `main.py`
- Full‑context English summary: `main.py`
- Translate JSON to Simplified Chinese: `main.py`
- Render UI with bilingual content: `templates/index.html`
- Caches processed papers in `papers_cache.json`
- Tracks shown papers in `shown_papers.json`

## Notes
- Full‑context input may exceed smaller model context windows; choose a model with sufficient capacity.
- If the LLM is not configured, the app falls back to heuristic summaries derived from the abstract.
- Keep your API key secret; do not commit it to source control.

## Troubleshooting
- PDF extraction issues: ensure `pypdf` is installed and that the arXiv PDF is accessible.
- Network/403 errors: retry later or check endpoint credentials.
- Slow summarization: large PDFs and full‑context requests take time; consider upgrading model or infra.

## License
This project is licensed under the MIT License. See the `LICENSE` file for details.

## Acknowledgements
- Hugging Face Daily Papers API for paper discovery
- arXiv for open access papers