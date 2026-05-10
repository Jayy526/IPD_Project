from transformers import pipeline
import pandas as pd
import logging

logger = logging.getLogger(__name__)

# Initialize FinBERT analyzer
try:
    analyzer = pipeline("sentiment-analysis", model="ProsusAI/finbert", top_k=None)
except Exception:
    # Fallback for older transformers versions
    analyzer = pipeline("sentiment-analysis", model="ProsusAI/finbert", return_all_scores=True)

def analyze_sentiment(text: str) -> dict:
    """
    Analyzes the sentiment of a given text using FinBERT.
    Returns a dictionary with pos, neu, neg, and compound scores.
    """
    if not text:
        return {"pos": 0.0, "neu": 1.0, "neg": 0.0, "compound": 0.0}
    
    # Truncate to avoid exceeding model's max token length (512)
    words = text.split()
    if len(words) > 300:
        text = " ".join(words[:300])
        
    try:
        results = analyzer(text)[0]
    except Exception as e:
        logger.error(f"FinBERT analysis failed: {e}")
        return {"pos": 0.0, "neu": 1.0, "neg": 0.0, "compound": 0.0}
        
    scores = {"pos": 0.0, "neu": 0.0, "neg": 0.0, "compound": 0.0}
    
    for score_dict in results:
        label = score_dict['label']
        if label == 'positive':
            scores["pos"] = score_dict['score']
        elif label == 'neutral':
            scores["neu"] = score_dict['score']
        elif label == 'negative':
            scores["neg"] = score_dict['score']
            
    # Calculate a compound score mathematically equivalent to VADER's logic
    scores["compound"] = scores["pos"] - scores["neg"]
    
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
