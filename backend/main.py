from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import pandas as pd
import numpy as np

# Import our pipelines
from data_pipeline.market_data import get_processed_data
from news_pipeline.news_fetcher import get_news
from news_pipeline.sentiment import process_news_sentiment
from models.xgboost_model import get_or_train_model, predict_latest
from xai.explainer import generate_shap_explanations

app = FastAPI(title="Explainable AI Stock Advisor API")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Helper to convert NaN and inf to None for JSON serialization
def clean_dict_for_json(d: dict) -> dict:
    clean = {}
    for k, v in d.items():
        if isinstance(v, float) and (pd.isna(v) or np.isinf(v)):
            clean[k] = None
        elif isinstance(v, dict):
            clean[k] = clean_dict_for_json(v)
        else:
            clean[k] = v
    return clean

@app.get("/")
def read_root():
    return {"message": "Explainable AI Stock Advisor API is running."}

@app.get("/api/analyze/{ticker}")
def analyze_stock(ticker: str, force_retrain: bool = Query(False, description="Force retrain the model")):
    """
    Complete analysis pipeline:
    1. Fetch stock data and calculate indicators.
    2. Fetch news and calculate sentiment.
    3. Train or load XGBoost model.
    4. Predict future movement (BUY/HOLD/SELL).
    5. Generate SHAP explanations.
    """
    try:
        # 1. Data Pipeline
        df = get_processed_data(ticker, period="1y")
        if df.empty:
            raise HTTPException(status_code=404, detail="No market data found.")
            
        # Extract last 30 days of market data for the chart
        chart_data = df.tail(60).reset_index()
        chart_data['Date'] = chart_data['Date'].astype(str)
        chart_records = chart_data.to_dict(orient='records')
        
        # 2. News Pipeline
        articles = get_news(ticker, use_mock=False)
        news_sentiment = process_news_sentiment(articles)
        
        # 3. Model Pipeline
        model, feature_names, X = get_or_train_model(ticker, df, news_sentiment, force_retrain=force_retrain)
        
        # 4. Prediction
        prediction_results = predict_latest(model, feature_names, X)
        
        # 5. XAI
        classes = ["SELL", "HOLD", "BUY"]
        pred_idx = classes.index(prediction_results["recommendation"])
        
        latest_features = prediction_results["latest_features"]
        shap_explanation = generate_shap_explanations(model, feature_names, latest_features, pred_idx)
        
        response_data = {
            "ticker": ticker.upper(),
            "recommendation": prediction_results["recommendation"],
            "confidence": prediction_results["confidence"],
            "probabilities": prediction_results["probabilities"],
            "current_price": float(df['Close'].iloc[-1]),
            "sentiment": news_sentiment,
            "articles": articles,
            "chart_data": chart_records,
            "explanation": shap_explanation
        }
        
        return clean_dict_for_json(response_data)
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
