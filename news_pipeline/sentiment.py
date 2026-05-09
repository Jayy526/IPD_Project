from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import pandas as pd

# Initialize VADER analyzer
analyzer = SentimentIntensityAnalyzer()

def analyze_sentiment(text: str) -> dict:
    """
    Analyzes the sentiment of a given text using VADER.
    Returns a dictionary with pos, neu, neg, and compound scores.
    """
    if not text:
        return {"pos": 0.0, "neu": 1.0, "neg": 0.0, "compound": 0.0}
    
    scores = analyzer.polarity_scores(text)
    return scores

def process_news_sentiment(articles: list) -> dict:
    """
    Processes a list of articles and aggregates sentiment scores.
    Returns aggregated metrics.
    """
    if not articles:
        return {
            "avg_compound": 0.0,
            "positive_count": 0,
            "negative_count": 0,
            "neutral_count": 0,
            "news_frequency": 0
        }
        
    compound_scores = []
    pos_count = 0
    neg_count = 0
    neu_count = 0
    
    for article in articles:
        # Combine title and summary for analysis
        text = f"{article.get('title', '')} {article.get('summary', '')}"
        scores = analyze_sentiment(text)
        
        compound = scores['compound']
        compound_scores.append(compound)
        
        # Classify based on compound score
        if compound >= 0.05:
            pos_count += 1
        elif compound <= -0.05:
            neg_count += 1
        else:
            neu_count += 1
            
    avg_compound = sum(compound_scores) / len(compound_scores) if compound_scores else 0.0
    
    return {
        "avg_compound": avg_compound,
        "positive_count": pos_count,
        "negative_count": neg_count,
        "neutral_count": neu_count,
        "news_frequency": len(articles)
    }
