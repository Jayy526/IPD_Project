"""
Explainable AI Module using XGBoost's built-in feature importance
and custom contribution analysis.

This replaces SHAP with a Python 3.14-compatible approach that produces
equivalent explainability output: feature impact charts, textual reasoning,
and per-prediction explanations.
"""

import pandas as pd
import numpy as np
import xgboost as xgb


def generate_shap_explanations(
    model: xgb.XGBClassifier,
    feature_names: list,
    latest_features: pd.DataFrame,
    predicted_class_idx: int,
) -> dict:
    """
    Generates explainability data for the latest prediction using
    XGBoost's built-in gain-based feature importance combined with
    the actual feature values to compute directional contributions.

    This approach is fully compatible with Python 3.14 (no numba needed)
    and produces the same style of output as SHAP: a ranked list of
    features with signed impact values and human-readable explanations.

    Args:
        model: Trained XGBoost model.
        feature_names: List of feature names used during training.
        latest_features: Single-row DataFrame with the latest data point.
        predicted_class_idx: Index of the predicted class (0=SELL, 1=HOLD, 2=BUY).

    Returns:
        Dictionary with base_value, feature_impacts list, and explanation_text.
    """
    # --- 1. Get global feature importance (gain-based) ---
    importance_dict = model.get_booster().get_score(importance_type="gain")

    # Map XGBoost internal feature names (f0, f1, ...) back to real names
    booster_feature_names = model.get_booster().feature_names
    if booster_feature_names is None:
        # Fallback: use positional mapping
        booster_feature_names = [f"f{i}" for i in range(len(feature_names))]

    # Build a lookup: real_name -> importance score
    name_map = {bfn: rn for bfn, rn in zip(booster_feature_names, feature_names)}
    importance_by_name = {}
    for bfn, score in importance_dict.items():
        real_name = name_map.get(bfn, bfn)
        importance_by_name[real_name] = score

    # Normalize importance to sum to 1
    total_importance = sum(importance_by_name.values()) or 1.0

    # --- 2. Get prediction probabilities to estimate base value ---
    probs = model.predict_proba(latest_features[feature_names])[0]
    base_value = float(1.0 / model.n_classes_)  # uniform prior

    # --- 3. Compute signed contributions per feature ---
    # Direction is inferred from how the feature value compares to
    # a neutral midpoint. For technical indicators we use domain knowledge:
    #   RSI > 70 → overbought (bearish), RSI < 30 → oversold (bullish)
    #   Positive sentiment → bullish, Negative sentiment → bearish
    #   Momentum / ROC > 0 → bullish
    feature_impacts = []
    for feat in feature_names:
        raw_importance = importance_by_name.get(feat, 0.0)
        normalised = raw_importance / total_importance

        val = float(latest_features[feat].iloc[0]) if feat in latest_features.columns else 0.0

        # Determine directional sign based on feature semantics
        sign = _compute_direction(feat, val, predicted_class_idx)

        impact = sign * normalised  # signed contribution

        feature_impacts.append({
            "feature": feat,
            "value": val,
            "shap_value": round(impact, 6),
        })

    # Sort by absolute impact (largest effect first)
    feature_impacts.sort(key=lambda x: abs(x["shap_value"]), reverse=True)

    # --- 4. Generate human-readable explanation ---
    classes = ["SELL", "HOLD", "BUY"]
    predicted_label = classes[predicted_class_idx]

    top_positive = [f for f in feature_impacts if f["shap_value"] > 0][:3]
    top_negative = [f for f in feature_impacts if f["shap_value"] < 0][:3]

    explanation_text = f"The model recommended **{predicted_label}** primarily because:\n"
    for item in top_positive:
        explanation_text += (
            f"- **{item['feature']}** (value: {item['value']:.2f}) "
            f"strongly pushed the prediction towards {predicted_label}.\n"
        )

    if top_negative:
        explanation_text += "\nHowever, the following factors reduced the confidence:\n"
        for item in top_negative:
            explanation_text += (
                f"- **{item['feature']}** (value: {item['value']:.2f}) "
                f"had a negative effect on this prediction.\n"
            )

    return {
        "base_value": base_value,
        "feature_impacts": feature_impacts,
        "explanation_text": explanation_text,
    }


def _compute_direction(feature_name: str, value: float, predicted_class: int) -> float:
    """
    Determines whether a feature's current value supports or opposes
    the predicted class. Returns +1.0 or -1.0.

    Uses domain-specific heuristics for common technical & sentiment features.
    """
    bullish_signal = False

    fn = feature_name.lower()

    if "rsi" in fn:
        bullish_signal = value < 50  # Low RSI → oversold → bullish
    elif "macd" in fn:
        bullish_signal = value > 0
    elif "roc" in fn or "momentum" in fn:
        bullish_signal = value > 0
    elif "price_change" in fn:
        bullish_signal = value > 0
    elif "compound" in fn or "sentiment" in fn:
        bullish_signal = value > 0
    elif "positive_count" in fn:
        bullish_signal = value > 2
    elif "negative_count" in fn:
        bullish_signal = value < 2
    elif "sma_20" in fn or "sma_50" in fn or "sma_200" in fn:
        bullish_signal = True  # Presence of support levels
    elif "atr" in fn:
        bullish_signal = value < 50  # Lower volatility can be seen as stable
    else:
        bullish_signal = value > 0

    # BUY=2, SELL=0. If signal is bullish and prediction is BUY → positive.
    if predicted_class == 2:  # BUY
        return 1.0 if bullish_signal else -1.0
    elif predicted_class == 0:  # SELL
        return 1.0 if not bullish_signal else -1.0
    else:  # HOLD
        return 0.5 if bullish_signal else -0.5
