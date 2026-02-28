from services.newsapi_service import fetch_from_newsapi
from services.rss_service import fetch_from_rss

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from pydantic import BaseModel
from typing import List

from services.scrape_service import extract_article_text

from fastapi.responses import StreamingResponse
import io
import csv

import time
import feedparser


import os
import resend

from services.processing_service import (
    filter_by_date_range,
    deduplicate_articles,
    rank_articles,
    classify_category,
    estimate_reading_time,
    
)

from services.gemini_service import (
    summarize_article,
    summarize_article_deep,
    summarize_executive,
)


# -----------------------------
# Request Models
# -----------------------------
class RunAgentRequest(BaseModel):
    fromDate: str
    toDate: str
    sources: List[str]


class ArticleSummaryRequest(BaseModel):
    title: str
    snippet: str

class ExportRequest(BaseModel):
    recipient: str
    executive_summary: str | None = None
    articles: List[dict]

class UrlSummaryRequest(BaseModel):
    title: str
    url: str

class SourceTestRequest(BaseModel):
    source_name: str

class AskAIRequest(BaseModel):
    question: str
    articles: List[dict]

# -----------------------------
# FastAPI App
# -----------------------------
app = FastAPI(title="News Intelligence Agent API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # later we will lock this down
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# -----------------------------
# Routes
# -----------------------------
@app.get("/")
def health():
    return {"status": "ok", "message": "Backend is running"}


@app.post("/agent/run")
def run_agent(payload: RunAgentRequest):
    # -----------------------------
    # 1) Fetch articles (ONLY selected sources)
    # -----------------------------

    # NewsAPI (only if user selected "reuters")
    newsapi_articles = []
    if "reuters" in payload.sources:
        newsapi_articles = fetch_from_newsapi(payload.fromDate, payload.toDate)

    # RSS (only selected RSS sources)
    rss_articles = fetch_from_rss(payload.sources)

    all_articles = newsapi_articles + rss_articles

    # -----------------------------
    # 2) Filter, dedupe, rank
    # -----------------------------
    all_articles = filter_by_date_range(all_articles, payload.fromDate, payload.toDate)
    all_articles = deduplicate_articles(all_articles)
    all_articles = rank_articles(all_articles)

    # -----------------------------
    # 3) Add category field (for UI tabs + filters)
    # -----------------------------
    for a in all_articles:
        a["category"] = classify_category(a)

        text_for_reading = (
            a.get("content")
            or a.get("description")
            or a.get("summary")
            or a.get("snippet")
            or ""
        )

        a["reading_time"] = estimate_reading_time(text_for_reading)

    # -----------------------------
    # 4) Generate short summaries (top 10 only)
    # -----------------------------
    TOP_N_FOR_SUMMARY = 10

    for i in range(min(TOP_N_FOR_SUMMARY, len(all_articles))):
        a = all_articles[i]

        # IMPORTANT: use real text fields
        text = (
            a.get("content")
            or a.get("description")
            or a.get("summary")
            or a.get("snippet")
            or ""
        )

        # Save raw text (for debugging + deep summary fallback)
        a["raw_text"] = text

        

        # Short summary (2-3 lines)
        a["summary"] = summarize_article(a.get("title", ""), text)

    # -----------------------------
    # 5) Executive summary (only top 10 ranked)
    # -----------------------------
    executive_summary = summarize_executive(all_articles[:10])

    return {
        "executive_summary": executive_summary,
        "articles": all_articles[:40],
         "total_fetched": len(all_articles)
    }


@app.post("/agent/article-summary")
def article_summary(payload: ArticleSummaryRequest):
    """
    Old endpoint: deep summary from snippet sent by frontend.
    """
    deep = summarize_article_deep(payload.title, payload.snippet)

    if deep:
        return {"deep_summary": deep}

    return {"deep_summary": payload.snippet[:300]}


@app.post("/agent/deep-summary-url")
def deep_summary_url(payload: UrlSummaryRequest):
    """
    Best endpoint:
    Extracts full article text from URL and summarizes it.
    """
    text = extract_article_text(payload.url)

    if not text or len(text.strip()) < 80:
        return {"deep_summary": "Could not extract enough text from this website."}

    deep = summarize_article_deep(payload.title, text)
    return {"deep_summary": deep}


# email export route
# -----------------------------
# Email export route (REAL SEND)
# -----------------------------


@app.post("/agent/export/email")
def export_email(payload: ExportRequest):

    resend.api_key = os.getenv("RESEND_API_KEY")

    # ---------- Build HTML ----------
    html = """
    <html>
    <head>
        <style>
            body { font-family: Arial; padding: 40px; }
            h2 { border-bottom: 2px solid #ddd; padding-bottom: 5px; }
            .article { margin-bottom: 30px; }
            .meta { color: #666; font-size: 13px; }
        </style>
    </head>
    <body>
    """

    if payload.executive_summary:
        html += f"""
        <h2>Executive Summary</h2>
        <p>{payload.executive_summary}</p>
        <hr/>
        """

    for a in payload.articles:
        html += f"""
        <div class="article">
            <h3>{a.get("title","")}</h3>
            <div class="meta">
                Source: {a.get("source","")} |
                Published: {a.get("publishedAt","")}
            </div>
            <p>{a.get("summary","")}</p>
            <p><a href="{a.get("url","")}">Read Full Article</a></p>
        </div>
        """

    html += "</body></html>"

    # ---------- SEND EMAIL ----------
    try:
        resend.Emails.send({
            "from": "News Agent <onboarding@resend.dev>",  # change after domain setup
            "to": payload.recipient,  # we will fix frontend next
            "subject": "Your AI News Digest",
            "html": html
        })

        return {"status": "Email sent successfully"}

    except Exception as e:
        print("EMAIL ERROR:", e)
        return {"status": "Email failed", "error": str(e)}

#csv export

@app.post("/agent/export/csv")
def export_csv(payload: ExportRequest):

    output = io.StringIO()
    writer = csv.writer(output)

    writer.writerow(["Title", "Source", "Published At", "Category", "URL"])

    for a in payload.articles:
        writer.writerow([
            a.get("title",""),
            a.get("source",""),
            a.get("publishedAt",""),
            a.get("category",""),
            a.get("url",""),
        ])

    output.seek(0)

    return StreamingResponse(
        output,
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=history_export.csv"}
    )

# For testing new sources (not in production yet)
@app.post("/agent/test-source")
def test_source(payload: SourceTestRequest):
    start = time.time()

    try:
        article_count = 0
        latest_date = None
        preview_titles = []

        source_name = payload.source_name

        # ------------------------
        # RSS SOURCES
        # ------------------------
        from services.rss_service import RSS_FEEDS

        for feed_id, display_name, feed_url in RSS_FEEDS:
            # Match using display name (BBC, TechCrunch, etc.)
            if display_name == source_name:
                feed = feedparser.parse(feed_url)

                article_count = len(feed.entries)

                if feed.entries:
                    latest_date = feed.entries[0].get("published", "Unknown")
                    preview_titles = [
                        entry.get("title", "")
                        for entry in feed.entries[:3]
                    ]
                break

        # ------------------------
        # NEWSAPI (Reuters via NewsAPI)
        # ------------------------
        if source_name == "Reuters":
            from services.newsapi_service import fetch_from_newsapi

            articles = fetch_from_newsapi("2026-02-20", "2026-02-23")
            article_count = len(articles)

            if articles:
                latest_date = articles[0].get("publishedAt")
                preview_titles = [
                    a.get("title") for a in articles[:3]
                ]

        latency = round((time.time() - start) * 1000)

        if article_count == 0:
            status = "failed"
        elif latency > 2000:
            status = "slow"
        else:
            status = "working"

        return {
            "status": status,
            "latency_ms": latency,
            "articles_found": article_count,
            "latest_article_date": latest_date,
            "preview_titles": preview_titles
        }

    except Exception as e:
        print("TEST ERROR:", e)
        return {
            "status": "failed",
            "error": str(e)
        }


@app.post("/agent/ask")
def ask_ai(payload: AskAIRequest):
    from openai import OpenAI
    import os

    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    client = OpenAI(api_key=OPENAI_API_KEY)

    if not payload.articles:
        return {"insight": "No articles available.", "matched_ids": []}

    # Prepare article context (lightweight)
    articles_context = "\n".join([
        f"ID: {a.get('id')} | Title: {a.get('title')} | Summary: {a.get('summary')}"
        for a in payload.articles
    ])

    prompt = f"""
You are analyzing today's news articles.

User Question:
{payload.question}

Here are today's articles:
{articles_context}

TASK:
1. Write a short 2 sentence insight answering the question.
2. Return ONLY a comma-separated list of matching article IDs.

FORMAT STRICTLY:
INSIGHT:
<your 2 sentence answer>

MATCHED_IDS:
id1,id2,id3

Do NOT invent IDs.
Only use IDs from the provided list.
"""

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You analyze news for professionals."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=300,
        )

        text = response.choices[0].message.content.strip()

        # Parse output
        insight_part = ""
        matched_ids = []

        if "MATCHED_IDS:" in text:
            parts = text.split("MATCHED_IDS:")
            insight_part = parts[0].replace("INSIGHT:", "").strip()
            ids_part = parts[1].strip()
            matched_ids = [i.strip() for i in ids_part.split(",") if i.strip()]

        return {
            "insight": insight_part,
            "matched_ids": matched_ids
        }

    except Exception as e:
        print("ASK AI ERROR:", e)
        return {"insight": "AI analysis failed.", "matched_ids": []}