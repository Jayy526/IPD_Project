# Explainable AI Stock Advisor for Indian Markets

An end-to-end full-stack Explainable AI (XAI) platform for Indian stock market investment recommendations. This project provides BUY/HOLD/SELL recommendations, confidence scores, and SHAP-based explanations for NSE-listed Indian stocks.

## Architecture

The project is divided into a robust FastAPI backend and a modern Streamlit frontend.

- **Backend (FastAPI)**: Handles data fetching, sentiment analysis, ML prediction, and XAI generation.
- **Frontend (Streamlit)**: A dark-themed fintech dashboard for visualizing the analysis.
- **Machine Learning**: XGBoost model trained dynamically based on technical indicators and VADER sentiment analysis of financial news.
- **Explainable AI**: SHAP (SHapley Additive exPlanations) is used to interpret model predictions.

## Setup Instructions

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the Backend API
Start the FastAPI server:
```bash
uvicorn backend.main:app --reload --port 8000
```
*The API will be available at `http://localhost:8000/docs`.*

### 3. Run the Frontend Dashboard
In a separate terminal, start Streamlit:
```bash
streamlit run frontend/app.py
```
*The dashboard will automatically open in your browser.*

## Project Structure

- `frontend/` - Streamlit application and UI logic.
- `backend/` - FastAPI backend server routing.
- `models/` - XGBoost training, prediction, and local caching.
- `data_pipeline/` - Market data fetching (`yfinance`) and technical indicators.
- `news_pipeline/` - News fetching (Google News RSS) and Sentiment analysis (VADER).
- `xai/` - SHAP explanations logic.
- `model_cache/` - Local storage for dynamically trained XGBoost models.
