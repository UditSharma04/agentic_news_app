from datetime import datetime, timezone
from dateutil import parser
import re
import math

import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def parse_date(date_str: str):
    """
    Tries to parse RSS dates + ISO dates safely.
    Returns datetime in UTC or None.
    """
    if not date_str:
        return None
    try:
        dt = parser.parse(date_str)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except Exception:
        return None


def filter_by_date_range(articles, from_date: str, to_date: str):
    start = parse_date(from_date)
    end = parse_date(to_date)

    if not start or not end:
        return articles

    filtered = []
    for a in articles:
        pub = parse_date(a.get("publishedAt", ""))
        if not pub:
            continue

        if start <= pub <= end:
            filtered.append(a)

    return filtered



def normalize_title(title: str):
    t = (title or "").lower().strip()
    t = re.sub(r"[^a-z0-9\s]", "", t)
    t = re.sub(r"\s+", " ", t)
    return t


def deduplicate_articles(articles):
    """
    Simple dedup based on normalized title similarity.
    Returns:
      - merged_articles (unique list)
    """
    seen = {}
    merged = []

    for a in articles:
        key = normalize_title(a.get("title", ""))

        if not key:
            continue

        if key in seen:
            # duplicate found
            original = seen[key]
            original["coverageCount"] = original.get("coverageCount", 1) + 1
            original["isDuplicateGroup"] = True
        else:
            seen[key] = a
            merged.append(a)

    return merged

CREDIBILITY_WEIGHTS = {
    # Tier 1 – Global Journalism
    "BBC": 5,
    "Reuters": 4,  # slightly lower since NewsAPI can fail

    # Tier 2 – High Quality Tech Journalism
    "MIT Technology Review": 5,
    "Ars Technica": 4,
    "Wired": 4,

    # Tier 3 – Tech Industry / Startup Focus
    "TechCrunch": 3,
    "The Verge": 3,
    "VentureBeat": 3,
    "The Register": 3,

    # Tier 4 – Community Curated
    "Hacker News": 2,
}




def rank_articles(articles):

    def score(a):
        coverage = a.get("coverageCount", 1)
        source = a.get("source", "")
        weight = CREDIBILITY_WEIGHTS.get(source, 1)

        # -----------------------
        # Recency Score
        # -----------------------
        published = a.get("publishedAt")

        recency_score = 0
        if published:
            try:
                published_dt = datetime.fromisoformat(published.replace("Z", "+00:00"))
                hours_old = (datetime.utcnow() - published_dt).total_seconds() / 3600

                # Fresh articles get more score
                if hours_old < 6:
                    recency_score = 5
                elif hours_old < 24:
                    recency_score = 3
                elif hours_old < 48:
                    recency_score = 1
                else:
                    recency_score = 0

            except:
                recency_score = 0

        # -----------------------
        # Final Score Formula
        # -----------------------
        return (coverage * 8) + (weight * 3) + recency_score

    return sorted(articles, key=score, reverse=True)

import re

CATEGORY_KEYWORDS = {
    "Tech": [
        "ai", "artificial intelligence", "openai", "google", "microsoft",
        "apple", "iphone", "android", "startup", "tech", "software",
        "chip", "nvidia", "tesla", "cloud", "aws", "cyber", "hacker",
        "data breach", "security", "ransomware"
    ],
    "Business": [
        "stock", "stocks", "market", "inflation", "interest rate", "gdp",
        "rbi", "fed", "earnings", "revenue", "profit", "loss", "bank",
        "ipo", "share", "economy"
    ],
    "World": [
        "war", "ukraine", "russia", "china", "israel", "gaza",
        "united nations", "nato", "election", "president", "government",
        "parliament", "diplomatic"
    ],
    "Sports": [
        "cricket", "football", "fifa", "ipl", "world cup", "match",
        "tournament", "goal", "coach", "player", "team"
    ],
    "Security": [
        "breach", "cyber", "hacked", "ransomware", "attack",
        "malware", "phishing", "leak", "vulnerability"
    ],
}



# def classify_category(article: dict) -> str:
#     text = f"{article.get('title','')} {article.get('summary','')}".lower()
#     text = re.sub(r"\s+", " ", text)

#     scores = {}

#     for category, keywords in CATEGORY_KEYWORDS.items():
#         score = 0
#         for kw in keywords:
#             if re.search(rf"\b{re.escape(kw)}\b", text):
#                 score += 1
#         scores[category] = score

#     best = max(scores, key=scores.get)

#     # Require at least 2 strong matches
#     if scores[best] < 2:
#         return "World"

#     return best
def classify_category(article: dict) -> str:
    title = article.get("title", "")
    summary = article.get("summary", "") or article.get("description", "")

    prompt = f"""
You are a precise news classifier.

Classify this article into ONE of the following categories:
Tech, Business, World, Sports, Security.

Respond with ONLY one word from:
Tech
Business
World
Sports
Security

Article:
Title: {title}
Summary: {summary}
"""

    valid_categories = ["Tech", "Business", "World", "Sports", "Security"]

    # ----------------------------
    # 1️⃣ Try GPT First
    # ----------------------------
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You classify news articles accurately."},
                {"role": "user", "content": prompt}
            ],
            temperature=0,
            max_tokens=10,
        )

        result = response.choices[0].message.content.strip()

        if result in valid_categories:
            return result

    except Exception as e:
        print("GPT classification failed:", e)

    # ----------------------------
    # 2️⃣ Fallback: Manual Rule-Based
    # ----------------------------
    import re

    text = f"{title} {summary}".lower()
    text = re.sub(r"\s+", " ", text)

    scores = {}

    for category, keywords in CATEGORY_KEYWORDS.items():
        score = 0
        for kw in keywords:
            if re.search(rf"\b{re.escape(kw)}\b", text):
                score += 1
        scores[category] = score

    best = max(scores, key=scores.get)

    if scores[best] == 0:
        return "World"

    return best

def estimate_reading_time(text: str):
    if not text:
        return "10 sec read"

    words = len(text.split())

    # Custom UX-friendly scaling
    seconds = words * 0.5   # 30 words → 15 sec
    seconds = int(math.ceil(seconds / 5) * 5)  # round to nearest 5 sec

    if seconds < 10:
        seconds = 10

    if seconds < 60:
        return f"{seconds} sec read"

    minutes = seconds // 60
    remaining_seconds = seconds % 60

    if minutes < 3:
        return f"{minutes} min {remaining_seconds} sec read"

    return f"{minutes} min read"