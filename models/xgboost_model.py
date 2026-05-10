import os

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

MODEL_CACHE_DIR = "model_cache"
if not os.path.exists(MODEL_CACHE_DIR):
    os.makedirs(MODEL_CACHE_DIR)

FEATURES = [
    "RSI",
    "SMA_20",
    "SMA_50",
    "SMA_200",
    "ATR",
    "ROC",
    "Price_Change_Pct",
    "avg_compound",
    "positive_count",
    "negative_count",
    "neutral_count",
    "news_frequency",
]

MODEL_CACHE_SUFFIX = "catboost"


def prepare_features(df: pd.DataFrame, news_sentiment: dict) -> pd.DataFrame:
    """Combine market data and news sentiment into a single feature dataframe."""
    df_features = df.copy()

    df_features["avg_compound"] = news_sentiment.get("avg_compound", 0.0)
    df_features["positive_count"] = news_sentiment.get("positive_count", 0)
    df_features["negative_count"] = news_sentiment.get("negative_count", 0)
    df_features["neutral_count"] = news_sentiment.get("neutral_count", 0)
    df_features["news_frequency"] = news_sentiment.get("news_frequency", 0)

    available_features = [feature for feature in FEATURES if feature in df_features.columns]

    if "MACD_12_26_9" in df_features.columns:
        available_features.append("MACD_12_26_9")
    if "MACDs_12_26_9" in df_features.columns:
        available_features.append("MACDs_12_26_9")

    return df_features[available_features]


def _model_path(ticker: str) -> str:
    return os.path.join(MODEL_CACHE_DIR, f"{ticker}_{MODEL_CACHE_SUFFIX}_model.joblib")


def _features_path(ticker: str) -> str:
    return os.path.join(MODEL_CACHE_DIR, f"{ticker}_{MODEL_CACHE_SUFFIX}_features.joblib")


def train_model(ticker: str, X: pd.DataFrame, y: pd.Series) -> CatBoostClassifier:
    """Train a CatBoost classifier for the given ticker and save it to cache."""
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    model = CatBoostClassifier(
        iterations=200,
        depth=6,
        learning_rate=0.08,
        loss_function="MultiClass",
        eval_metric="MultiClass",
        random_seed=42,
        verbose=False,
        allow_writing_files=False,
    )

    model.fit(X_train, y_train)

    preds = model.predict(X_test).astype(int).ravel()
    acc = accuracy_score(y_test, preds)
    print(f"[{ticker}] Model trained. Accuracy: {acc:.2f}")

    joblib.dump(model, _model_path(ticker))
    joblib.dump(X.columns.tolist(), _features_path(ticker))

    return model


def load_model(ticker: str) -> tuple:
    """Load a cached model and its feature names if they exist."""
    model_path = _model_path(ticker)
    features_path = _features_path(ticker)

    if os.path.exists(model_path) and os.path.exists(features_path):
        model = joblib.load(model_path)
        feature_names = joblib.load(features_path)
        return model, feature_names
    return None, None


def get_or_train_model(ticker: str, df: pd.DataFrame, news_sentiment: dict, force_retrain: bool = False) -> tuple:
    """Retrieve the cached model or train a new one if not found or forced."""
    X = prepare_features(df, news_sentiment)
    y = df["Target"] if "Target" in df.columns else None

    model, feature_names = load_model(ticker)

    if model is None or force_retrain:
        if y is None or y.isna().all():
            raise ValueError("No target labels found for training.")

        print(f"Training new CatBoost model for {ticker}...")
        model = train_model(ticker, X, y)
        feature_names = X.columns.tolist()
    else:
        print(f"Loaded cached CatBoost model for {ticker}.")

    return model, feature_names, X


def predict_latest(model: CatBoostClassifier, feature_names: list, X: pd.DataFrame) -> dict:
    """Predict the next movement for the latest row of data."""
    latest_row = X[feature_names].iloc[-1:]
    probs = model.predict_proba(latest_row)[0]

    classes = ["SELL", "HOLD", "BUY"]
    pred_idx = int(np.argmax(probs))

    return {
        "recommendation": classes[pred_idx],
        "confidence": float(probs[pred_idx] * 100),
        "probabilities": {
            "SELL": float(probs[0]),
            "HOLD": float(probs[1]),
            "BUY": float(probs[2]),
        },
        "latest_features": latest_row,
    }
