import feedparser
import requests
from bs4 import BeautifulSoup
import urllib.parse
from datetime import datetime
import random

def fetch_google_news_rss(query: str, max_results: int = 10) -> list:
    """
    Fetches news from Google News RSS feed.
    """
    encoded_query = urllib.parse.quote(f"{query} stock news india")
    rss_url = f"https://news.google.com/rss/search?q={encoded_query}&hl=en-IN&gl=IN&ceid=IN:en"
    
    try:
        feed = feedparser.parse(rss_url)
        articles = []
        for entry in feed.entries[:max_results]:
            articles.append({
                "title": entry.title,
                "link": entry.link,
                "published": entry.published,
                "source": entry.source.title if 'source' in entry else "Google News",
                "summary": entry.summary if 'summary' in entry else ""
            })
        return articles
    except Exception as e:
        print(f"Error fetching RSS: {e}")
        return []

def get_mock_news(ticker: str) -> list:
    """
    Provides fallback mock news data for development and testing.
    """
    mock_headlines = [
        f"{ticker} reports record Q3 earnings, beating analyst estimates.",
        f"Market concerns grow over {ticker}'s debt restructuring plan.",
        f"{ticker} announces major new expansion into emerging markets.",
        f"Regulatory probe hits {ticker}, shares tumble.",
        f"Analysts upgrade {ticker} to Strong Buy after recent product launch.",
        f"Inflation fears weigh heavily on {ticker} and broader sector.",
        f"{ticker} CEO steps down unexpectedly; board searches for replacement."
    ]
    
    articles = []
    for i in range(5):
        headline = random.choice(mock_headlines)
        articles.append({
            "title": headline,
            "link": f"https://mocknews.com/{ticker}/article{i}",
            "published": datetime.now().strftime("%a, %d %b %Y %H:%M:%S GMT"),
            "source": "Mock Financial News",
            "summary": f"This is a mocked news summary for {headline}."
        })
    return articles

def get_news(ticker: str, use_mock: bool = False) -> list:
    """
    Main function to get news. Tries RSS, falls back to mock if empty or if use_mock is True.
    """
    if use_mock:
        return get_mock_news(ticker)
        
    articles = fetch_google_news_rss(ticker.replace(".NS", ""))
    if not articles:
        print("RSS feed returned no articles. Falling back to mock data.")
        return get_mock_news(ticker)
        
    return articles
