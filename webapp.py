"""FastAPI web interface for the news summarizer."""
import re
from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from summarizer import NewsSummarizer


app = FastAPI(title="News Summarizer")
app.mount("/static", StaticFiles(directory="static"), name="static")

templates = Jinja2Templates(directory="templates")

summarizer = NewsSummarizer()


def extract_sentiment(text):
    """Extract sentiment category from free-text LLM output.

    Tries regex for 'overall sentiment: X' first, then keyword fallback.
    Returns one of: positive, negative, neutral, mixed, unknown.
    """
    if not text:
        return "unknown"
    lower = text.lower()

    # Regex: look for "overall sentiment: <word>" or "sentiment: <word>"
    match = re.search(r"(?:overall\s+)?sentiment[:\s]+\*{0,2}(positive|negative|neutral|mixed)\*{0,2}", lower)
    if match:
        return match.group(1)

    # Keyword fallback
    for keyword in ("positive", "negative", "neutral", "mixed"):
        if keyword in lower:
            return keyword

    return "unknown"


templates.env.filters["extract_sentiment"] = extract_sentiment


@app.get("/", response_class=HTMLResponse)
async def index(request: Request, q: str = ""):
    """Article list with optional search."""
    if q.strip():
        articles = summarizer.db.search_articles(q.strip())
    else:
        articles = summarizer.db.get_all_articles()
    return templates.TemplateResponse("index.html", {
        "request": request,
        "articles": articles,
        "query": q,
    })


@app.get("/article/{article_id}", response_class=HTMLResponse)
async def article_detail(request: Request, article_id: int):
    """Single article detail view."""
    row = summarizer.db.conn.execute(
        "SELECT * FROM articles WHERE id = ?", (article_id,)
    ).fetchone()
    article = dict(row) if row else None
    return templates.TemplateResponse("article.html", {
        "request": request,
        "article": article,
    })


@app.get("/trends", response_class=HTMLResponse)
async def trends(request: Request):
    """Trend analysis dashboard."""
    articles = summarizer.db.get_all_articles()

    # Sentiment distribution
    sentiment_counts = {}
    for a in articles:
        s = extract_sentiment(a.get("sentiment", ""))
        sentiment_counts[s] = sentiment_counts.get(s, 0) + 1

    # Articles by source
    source_counts = {}
    for a in articles:
        src = a.get("source", "Unknown")
        source_counts[src] = source_counts.get(src, 0) + 1

    # Articles over time (by date)
    date_counts = {}
    for a in articles:
        processed = a.get("processed_at", "")
        day = processed[:10] if processed else "unknown"
        date_counts[day] = date_counts.get(day, 0) + 1

    return templates.TemplateResponse("trends.html", {
        "request": request,
        "total": len(articles),
        "sentiment_counts": sentiment_counts,
        "source_counts": source_counts,
        "date_counts": dict(sorted(date_counts.items())),
        "max_sentiment": max(sentiment_counts.values()) if sentiment_counts else 1,
        "max_source": max(source_counts.values()) if source_counts else 1,
        "max_date": max(date_counts.values()) if date_counts else 1,
    })


@app.get("/fetch", response_class=HTMLResponse)
async def fetch_form(request: Request):
    """Form to trigger a new article fetch."""
    return templates.TemplateResponse("fetch.html", {
        "request": request,
        "result": None,
    })


@app.post("/fetch", response_class=HTMLResponse)
async def fetch_articles(
    request: Request,
    category: str = Form("technology"),
    num_articles: int = Form(3),
):
    """Fetch and process new articles, then redirect to index."""
    num_articles = max(1, min(10, num_articles))
    articles = summarizer.news_api.fetch_top_headlines(
        category=category,
        max_articles=num_articles,
    )
    count = 0
    if articles:
        results = summarizer.process_articles(articles)
        count = len(results)

    return templates.TemplateResponse("fetch.html", {
        "request": request,
        "result": {"category": category, "count": count},
    })


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
