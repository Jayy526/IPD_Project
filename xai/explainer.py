"""
Explainable AI module combining CatBoost feature attributions with DICE
counterfactual examples.

The attribution layer keeps the ranked feature impact output, while the
counterfactual layer adds concrete feature changes that could move the
prediction toward an alternative class.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier

try:
    import dice_ml
except ImportError:  # pragma: no cover - handled at runtime
    dice_ml = None


CLASS_LABELS = ["SELL", "HOLD", "BUY"]
MAX_BACKGROUND_ROWS = 200
MAX_COUNTERFACTUALS = 3


def generate_shap_dice_explanations(
    model: CatBoostClassifier,
    feature_names: list,
    latest_features: pd.DataFrame,
    predicted_class_idx: int,
    reference_data: pd.DataFrame | None = None,
) -> dict:
    """Build a combined CatBoost + DICE explanation payload."""
    shap_explanation = _build_shap_explanation(model, feature_names, latest_features, predicted_class_idx)
    dice_explanation = _build_dice_explanation(
        model=model,
        feature_names=feature_names,
        latest_features=latest_features,
        predicted_class_idx=predicted_class_idx,
        reference_data=reference_data,
    )

    return {
        **shap_explanation,
        "explanation_model": "CatBoost + DICE",
        "dice": dice_explanation,
    }


def generate_shap_explanations(
    model: CatBoostClassifier,
    feature_names: list,
    latest_features: pd.DataFrame,
    predicted_class_idx: int,
    reference_data: pd.DataFrame | None = None,
) -> dict:
    """Backward-compatible alias for the combined CatBoost + DICE output."""
    return generate_shap_dice_explanations(
        model=model,
        feature_names=feature_names,
        latest_features=latest_features,
        predicted_class_idx=predicted_class_idx,
        reference_data=reference_data,
    )


def _build_shap_explanation(
    model: CatBoostClassifier,
    feature_names: list,
    latest_features: pd.DataFrame,
    predicted_class_idx: int,
) -> dict:
    """Generate feature importance output from CatBoost."""
    importance_values = np.asarray(model.get_feature_importance(prettified=False), dtype=float)

    importance_by_name = {
        feature: float(importance_values[idx]) if idx < len(importance_values) else 0.0
        for idx, feature in enumerate(feature_names)
    }

    total_importance = sum(importance_by_name.values()) or 1.0
    base_value = float(1.0 / model.classes_count_) if getattr(model, "classes_count_", None) else float(1.0 / 3.0)

    feature_impacts = []
    for feat in feature_names:
        raw_importance = importance_by_name.get(feat, 0.0)
        normalised = raw_importance / total_importance

        val = float(latest_features[feat].iloc[0]) if feat in latest_features.columns else 0.0
        sign = _compute_direction(feat, val, predicted_class_idx)
        impact = sign * normalised

        feature_impacts.append({
            "feature": feat,
            "value": val,
            "shap_value": round(impact, 6),
        })

    feature_impacts.sort(key=lambda x: abs(x["shap_value"]), reverse=True)

    predicted_label = CLASS_LABELS[predicted_class_idx]
    top_positive = [f for f in feature_impacts if f["shap_value"] > 0][:3]
    top_negative = [f for f in feature_impacts if f["shap_value"] < 0][:3]

    explanation_text = f"The model recommended **{predicted_label}** primarily because:\n"
    for item in top_positive:
        explanation_text += (
            f"- **{item['feature']}** (value: {item['value']:.2f}) "
            f"pushed the prediction towards {predicted_label}.\n"
        )

    if top_negative:
        explanation_text += "\nThe following factors reduced the confidence:\n"
        for item in top_negative:
            explanation_text += (
                f"- **{item['feature']}** (value: {item['value']:.2f}) "
                f"worked against this prediction.\n"
            )

    return {
        "base_value": base_value,
        "feature_impacts": feature_impacts,
        "explanation_text": explanation_text,
    }


def _build_dice_explanation(
    model: xgb.XGBClassifier,
    feature_names: list,
    latest_features: pd.DataFrame,
    predicted_class_idx: int,
    reference_data: pd.DataFrame | None,
) -> dict:
    """Generate DICE counterfactuals for the latest prediction."""
    if dice_ml is None:
        return {
            "available": False,
            "message": "dice-ml is not installed in this environment.",
            "counterfactuals": [],
        }

    if reference_data is None or reference_data.empty:
        return {
            "available": False,
            "message": "Counterfactuals require reference data from the training slice.",
            "counterfactuals": [],
        }

    background = reference_data[feature_names].dropna().tail(MAX_BACKGROUND_ROWS).copy()
    if background.empty:
        return {
            "available": False,
            "message": "No clean reference rows were available for DICE.",
            "counterfactuals": [],
        }

    background = background.drop_duplicates().reset_index(drop=True)
    background["target"] = model.predict(background[feature_names])

    query_instance = latest_features[feature_names].copy()
    query_instance = query_instance.fillna(background.median(numeric_only=True))

    desired_class_idx = _select_counterfactual_class(model, query_instance, predicted_class_idx)
    if desired_class_idx == predicted_class_idx:
        return {
            "available": False,
            "message": "A contrasting DICE class could not be selected.",
            "counterfactuals": [],
        }

    try:
        data_interface = dice_ml.Data(
            dataframe=background,
            continuous_features=feature_names,
            outcome_name="target",
        )
        model_interface = dice_ml.Model(model=model, backend="sklearn", model_type="classifier")
        explainer = dice_ml.Dice(data_interface, model_interface, method="random")
        counterfactuals = explainer.generate_counterfactuals(
            query_instances=query_instance,
            total_CFs=MAX_COUNTERFACTUALS,
            desired_class=int(desired_class_idx),
            features_to_vary="all",
            verbose=False,
        )
    except Exception as exc:
        return {
            "available": False,
            "message": f"DICE counterfactual generation failed: {exc}",
            "counterfactuals": [],
        }

    if not getattr(counterfactuals, "cf_examples_list", None):
        return {
            "available": False,
            "message": "DICE did not return any counterfactual examples.",
            "counterfactuals": [],
        }

    cf_example = counterfactuals.cf_examples_list[0]
    final_cfs_df = getattr(cf_example, "final_cfs_df", None)
    if final_cfs_df is None or final_cfs_df.empty:
        return {
            "available": False,
            "message": "DICE did not find a valid counterfactual for this prediction.",
            "counterfactuals": [],
        }

    current_row = query_instance.iloc[0]
    cf_records = []
    for rank, (_, cf_row) in enumerate(final_cfs_df.head(MAX_COUNTERFACTUALS).iterrows(), start=1):
        changes = []
        feature_values = {}

        for feat in feature_names:
            current_value = _serialize_value(current_row.get(feat))
            target_value = _serialize_value(cf_row.get(feat))
            feature_values[feat] = target_value

            if not _values_match(current_value, target_value):
                changes.append({
                    "feature": feat,
                    "from": current_value,
                    "to": target_value,
                    "delta": _difference(current_value, target_value),
                })

        cf_records.append({
            "rank": rank,
            "target_class": CLASS_LABELS[desired_class_idx],
            "changes": changes[:5],
            "feature_values": feature_values,
        })

    summary = _build_counterfactual_summary(cf_records, predicted_class_idx, desired_class_idx)
    return {
        "available": True,
        "message": summary,
        "target_class": CLASS_LABELS[desired_class_idx],
        "counterfactuals": cf_records,
    }


def _build_counterfactual_summary(counterfactuals: list, predicted_class_idx: int, desired_class_idx: int) -> str:
    predicted_label = CLASS_LABELS[predicted_class_idx]
    desired_label = CLASS_LABELS[desired_class_idx]

    if not counterfactuals:
        return f"No DICE counterfactuals were generated to move from {predicted_label} toward {desired_label}."

    first_cf = counterfactuals[0]
    changes = first_cf.get("changes", [])[:3]
    if not changes:
        return f"DICE found a counterfactual route from {predicted_label} toward {desired_label}."

    change_bits = []
    for change in changes:
        feature = change["feature"]
        from_value = change["from"]
        to_value = change["to"]
        if isinstance(from_value, (int, float)) and isinstance(to_value, (int, float)):
            change_bits.append(f"{feature} {from_value:.2f} -> {to_value:.2f}")
        else:
            change_bits.append(f"{feature} {from_value} -> {to_value}")

    joined_changes = ", ".join(change_bits)
    return f"To move the prediction from {predicted_label} toward {desired_label}, DICE suggests adjusting {joined_changes}."


def _select_counterfactual_class(
    model: xgb.XGBClassifier,
    query_instance: pd.DataFrame,
    predicted_class_idx: int,
) -> int:
    probabilities = model.predict_proba(query_instance)[0]
    ranked_classes = np.argsort(probabilities)[::-1]

    for class_idx in ranked_classes:
        if int(class_idx) != int(predicted_class_idx):
            return int(class_idx)

    return int(predicted_class_idx)


def _compute_direction(feature_name: str, value: float, predicted_class: int) -> float:
    """Determine whether a feature value supports or opposes the prediction."""
    bullish_signal = False

    fn = feature_name.lower()

    if "rsi" in fn:
        bullish_signal = value < 50
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
        bullish_signal = True
    elif "atr" in fn:
        bullish_signal = value < 50
    else:
        bullish_signal = value > 0

    if predicted_class == 2:
        return 1.0 if bullish_signal else -1.0
    if predicted_class == 0:
        return 1.0 if not bullish_signal else -1.0
    return 0.5 if bullish_signal else -0.5


def _serialize_value(value):
    if isinstance(value, np.generic):
        return value.item()
    if pd.isna(value):
        return None
    return value


def _values_match(left, right) -> bool:
    if left is None and right is None:
        return True
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return bool(np.isclose(left, right))
    return left == right


def _difference(left, right):
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return round(float(right) - float(left), 6)
    return None
