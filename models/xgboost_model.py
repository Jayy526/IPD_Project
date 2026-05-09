import xgboost as xgb
import pandas as pd
import numpy as np
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

MODEL_CACHE_DIR = "model_cache"
if not os.path.exists(MODEL_CACHE_DIR):
    os.makedirs(MODEL_CACHE_DIR)

# Define feature columns we will use for training
FEATURES = [
    'RSI', 'SMA_20', 'SMA_50', 'SMA_200', 'ATR', 'ROC', 'Price_Change_Pct',
    'avg_compound', 'positive_count', 'negative_count', 'neutral_count', 'news_frequency'
]

def prepare_features(df: pd.DataFrame, news_sentiment: dict) -> pd.DataFrame:
    """
    Combines market data and news sentiment into a single feature dataframe.
    """
    df_features = df.copy()
    
    # Broadcast sentiment features to the dataframe
    # In a real historical backtest, you'd align dates. 
    # Here, we append the latest sentiment to recent rows for short-term prediction.
    df_features['avg_compound'] = news_sentiment.get('avg_compound', 0.0)
    df_features['positive_count'] = news_sentiment.get('positive_count', 0)
    df_features['negative_count'] = news_sentiment.get('negative_count', 0)
    df_features['neutral_count'] = news_sentiment.get('neutral_count', 0)
    df_features['news_frequency'] = news_sentiment.get('news_frequency', 0)
    
    # Keep only the features we need
    available_features = [f for f in FEATURES if f in df_features.columns]
    
    # Handle missing MACD and BBands if they exist in df but weren't in FEATURES explicitly
    if 'MACD_12_26_9' in df_features.columns:
        available_features.append('MACD_12_26_9')
    if 'MACDs_12_26_9' in df_features.columns:
        available_features.append('MACDs_12_26_9')
        
    return df_features[available_features]

def train_model(ticker: str, X: pd.DataFrame, y: pd.Series) -> xgb.XGBClassifier:
    """
    Trains an XGBoost model for the given ticker and saves it to cache.
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1,
        objective='multi:softprob',
        num_class=3, # 0: SELL, 1: HOLD, 2: BUY
        random_state=42,
        eval_metric='mlogloss'
    )
    
    model.fit(X_train, y_train)
    
    # Evaluate
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(f"[{ticker}] Model trained. Accuracy: {acc:.2f}")
    
    # Save model
    model_path = os.path.join(MODEL_CACHE_DIR, f"{ticker}_model.joblib")
    joblib.dump(model, model_path)
    
    # Save feature names for SHAP
    features_path = os.path.join(MODEL_CACHE_DIR, f"{ticker}_features.joblib")
    joblib.dump(X.columns.tolist(), features_path)
    
    return model

def load_model(ticker: str) -> tuple:
    """
    Loads a cached model and its feature names if they exist.
    """
    model_path = os.path.join(MODEL_CACHE_DIR, f"{ticker}_model.joblib")
    features_path = os.path.join(MODEL_CACHE_DIR, f"{ticker}_features.joblib")
    
    if os.path.exists(model_path) and os.path.exists(features_path):
        model = joblib.load(model_path)
        feature_names = joblib.load(features_path)
        return model, feature_names
    return None, None

def get_or_train_model(ticker: str, df: pd.DataFrame, news_sentiment: dict, force_retrain: bool = False) -> tuple:
    """
    Retrieves the cached model or trains a new one if not found or forced.
    Returns (model, feature_names, X).
    """
    X = prepare_features(df, news_sentiment)
    y = df['Target'] if 'Target' in df.columns else None
    
    model, feature_names = load_model(ticker)
    
    if model is None or force_retrain:
        if y is None or y.isna().all():
            raise ValueError("No target labels found for training.")
            
        print(f"Training new model for {ticker}...")
        model = train_model(ticker, X, y)
        feature_names = X.columns.tolist()
    else:
        print(f"Loaded cached model for {ticker}.")
        
    return model, feature_names, X

def predict_latest(model: xgb.XGBClassifier, feature_names: list, X: pd.DataFrame) -> dict:
    """
    Predicts the next movement for the latest row of data.
    Returns recommendation, confidence score, and the specific row used.
    """
    # Get the latest row
    latest_row = X[feature_names].iloc[-1:]
    
    # Predict probabilities
    probs = model.predict_proba(latest_row)[0]
    
    # 0: SELL, 1: HOLD, 2: BUY
    classes = ["SELL", "HOLD", "BUY"]
    pred_idx = np.argmax(probs)
    
    recommendation = classes[pred_idx]
    confidence = float(probs[pred_idx] * 100)
    
    return {
        "recommendation": recommendation,
        "confidence": confidence,
        "probabilities": {
            "SELL": float(probs[0]),
            "HOLD": float(probs[1]),
            "BUY": float(probs[2])
        },
        "latest_features": latest_row
    }
