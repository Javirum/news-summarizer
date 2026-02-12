# News Summarizer

A multi-provider news summarizer that fetches top headlines from [NewsAPI](https://newsapi.org/) and uses both OpenAI and Anthropic LLMs to generate summaries and sentiment analysis, with built-in cost tracking, rate limiting, fallback support, and a FastAPI web interface.

## What It Does

1. **Fetches news articles** from NewsAPI by category (`technology`, `business`, `health`, `general`)
2. **Summarizes each article** in 2-3 sentences using OpenAI (`gpt-4o-mini`)
3. **Analyzes sentiment** of each summary using Anthropic (`claude-sonnet-4-20250514`)
4. **Stores processed results** in a local database for later search and analysis
5. **Tracks costs** per request and enforces a configurable daily budget
6. **Falls back** to a secondary provider when the primary one fails
7. Supports both **CLI workflows** and a **web dashboard** (FastAPI + Jinja templates)

## Project Structure

```
news-summarizer/
├── main.py                 # CLI entry point (interactive prompts)
├── webapp.py               # FastAPI app (dashboard + fetch + trends pages)
├── config.py               # Configuration & environment variable loading
├── news_api.py             # NewsAPI client with rate limiting
├── llm_providers.py        # OpenAI + Anthropic clients, cost tracker, fallback logic
├── summarizer.py           # Sync & async summarization pipeline
├── database.py             # SQLite persistence and search helpers
├── cache.py                # Response caching utilities
├── templates/              # Jinja templates for the web app
├── static/style.css        # Web app styles
├── test_summarizer.py      # Unit tests (pytest)
├── requirements.txt        # Python dependencies
└── .gitignore
```

## Setup

### 1. Clone the repository

```bash
git clone <repo-url>
cd news-summarizer
```

### 2. Create a virtual environment

```bash
python -m venv venv
source venv/bin/activate  # macOS/Linux
# or
venv\Scripts\activate     # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure API keys

Create a `.env` file in the project root:

```env
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
NEWS_API_KEY=your_newsapi_key

# Optional
ENVIRONMENT=development
MAX_RETRIES=3
REQUEST_TIMEOUT=30
DAILY_BUDGET=5.00
```

You can get API keys from:
- **OpenAI**: https://platform.openai.com/api-keys
- **Anthropic**: https://console.anthropic.com/settings/keys
- **NewsAPI**: https://newsapi.org/register

## How to Run

### CLI mode (interactive)

```bash
python main.py
```

You will be prompted for:
- News category (`technology`, `business`, `health`, `general`)
- Number of articles (1-10)
- Whether to use async processing

### Web app mode (FastAPI)

```bash
uvicorn webapp:app --reload
```

Then open `http://127.0.0.1:8000`.

Web routes:
- `/` article list + search
- `/article/{id}` article detail page
- `/fetch` fetch/process new articles from a form
- `/trends` sentiment/source/date trend dashboard

### Run individual modules

```bash
# Test NewsAPI connection
python news_api.py

# Test LLM providers
python llm_providers.py

# Test async summarizer
python summarizer.py
```

### Run tests

```bash
pytest test_summarizer.py -v
```

## Example Output

```
================================================================================
NEWS SUMMARIZER - Multi-Provider Edition
================================================================================

Enter news category (technology/business/health/general): technology
How many articles to process? (1-10): 2
Use async processing? (y/n): n

Fetching 2 articles from category: technology
✓ Fetched 2 articles from News API

Processing 2 articles...

Processing: Apple Announces New AI Features for iPhone...
  → Summarizing with OpenAI...
  ✓ Summary generated
  → Analyzing sentiment with Anthropic...
  ✓ Sentiment analyzed

Processing: Google Launches Quantum Computing Milestone...
  → Summarizing with OpenAI...
  ✓ Summary generated
  → Analyzing sentiment with Anthropic...
  ✓ Sentiment analyzed

================================================================================
NEWS SUMMARY REPORT
================================================================================

1. Apple Announces New AI Features for iPhone
   Source: TechCrunch | Published: 2026-02-12T10:30:00Z
   URL: https://example.com/article1

   SUMMARY:
   Apple unveiled a suite of new AI-powered features for the iPhone,
   including enhanced Siri capabilities and on-device language processing.
   The update is expected to roll out later this year.

   SENTIMENT:
   Overall sentiment: Positive. Confidence: 85%. The tone is optimistic
   and forward-looking, reflecting excitement about technological advancement.

   ----------------------------------------------------------------------------

2. Google Launches Quantum Computing Milestone
   Source: The Verge | Published: 2026-02-11T14:00:00Z
   URL: https://example.com/article2

   SUMMARY:
   Google has achieved a new milestone in quantum computing, demonstrating
   a processor that can solve specific problems faster than classical
   supercomputers. Researchers say this brings practical applications closer.

   SENTIMENT:
   Overall sentiment: Positive. Confidence: 90%. The key emotional tone
   is one of scientific optimism and achievement.

   ----------------------------------------------------------------------------

================================================================================
COST SUMMARY
================================================================================
Total requests: 4
Total cost: $0.0012
Total tokens: 1,847
  Input: 1,203
  Output: 644
Average cost per request: $0.000300
================================================================================

✓ Processing complete!
```

## Cost Analysis

### Per-token pricing (used by the cost tracker)

| Model | Input (per 1M tokens) | Output (per 1M tokens) |
|---|---|---|
| `gpt-4o-mini` | $0.15 | $0.60 |
| `gpt-4o` | $2.50 | $10.00 |
| `claude-sonnet-4-20250514` | $3.00 | $15.00 |

### Estimated cost per article

Each article makes **2 API calls**: one to OpenAI (summary) and one to Anthropic (sentiment).

| Step | Model | ~Input tokens | ~Output tokens | ~Cost |
|---|---|---|---|---|
| Summarize | `gpt-4o-mini` | ~300 | ~100 | ~$0.0001 |
| Sentiment | `claude-sonnet-4-20250514` | ~200 | ~80 | ~$0.0018 |
| **Total per article** | | | | **~$0.002** |

### Scaling estimates

| Articles | Estimated cost |
|---|---|
| 5 | ~$0.01 |
| 50 | ~$0.10 |
| 500 | ~$1.00 |

### Cost controls

- **Daily budget**: Configurable via `DAILY_BUDGET` env var (default: $5.00). The app raises an exception if the budget is exceeded and warns at 90% usage.
- **Rate limiting**: Built-in per-provider rate limits (OpenAI: 500 RPM, Anthropic: 50 RPM, NewsAPI: 100 RPM).
- **Token limiting**: Article content is truncated to 500 characters before being sent to LLMs.
