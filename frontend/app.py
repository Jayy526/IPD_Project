import streamlit as st
import requests
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd

# Page Configuration
st.set_page_config(
    page_title="XAI Stock Advisor",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Dark Fintech Theme
st.markdown("""
<style>
    .reportview-container {
        background: #0E1117;
    }
    .main {
        background-color: #0E1117;
    }
    h1, h2, h3, h4, h5, h6 {
        color: #E0E6ED;
        font-family: 'Inter', sans-serif;
    }
    .metric-card {
        background-color: #1E2530;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #2B3544;
    }
    .stAlert {
        border-radius: 8px;
    }
    .news-card {
        background-color: #1A212D;
        padding: 1rem;
        border-radius: 8px;
        margin-bottom: 1rem;
        border-left: 4px solid #3B82F6;
    }
</style>
""", unsafe_allow_html=True)

API_URL = "http://localhost:8000/api/analyze"

def fetch_data(ticker, force_retrain):
    with st.spinner(f"Analyzing {ticker}... This may take a minute if training a new model."):
        try:
            response = requests.get(f"{API_URL}/{ticker}?force_retrain={str(force_retrain).lower()}")
            response.raise_for_status()
            return response.json()
        except Exception as e:
            st.error(f"Error fetching data from API: {e}")
            return None

def plot_candlestick(chart_data):
    df = pd.DataFrame(chart_data)
    
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                        vertical_spacing=0.03, subplot_titles=('OHLC', 'Volume'), 
                        row_width=[0.2, 0.7])

    fig.add_trace(go.Candlestick(x=df['Date'],
                open=df['Open'], high=df['High'],
                low=df['Low'], close=df['Close'],
                name='Price'), row=1, col=1)

    fig.add_trace(go.Bar(x=df['Date'], y=df['Volume'], name='Volume', marker_color='#3B82F6'), row=2, col=1)

    fig.update_layout(
        template='plotly_dark',
        height=600,
        margin=dict(l=0, r=0, t=30, b=0),
        xaxis_rangeslider_visible=False,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )
    return fig

def plot_gauge(confidence, recommendation):
    color = "green" if recommendation == "BUY" else "red" if recommendation == "SELL" else "yellow"
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = confidence,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Confidence Score"},
        gauge = {
            'axis': {'range': [0, 100]},
            'bar': {'color': color},
            'steps': [
                {'range': [0, 33], 'color': "rgba(255, 255, 255, 0.1)"},
                {'range': [33, 66], 'color': "rgba(255, 255, 255, 0.2)"},
                {'range': [66, 100], 'color': "rgba(255, 255, 255, 0.3)"}],
        }))
    fig.update_layout(template='plotly_dark', height=300, margin=dict(l=10, r=10, t=40, b=10), paper_bgcolor='rgba(0,0,0,0)')
    return fig

# --- UI Layout ---

st.title("📈 Explainable AI Stock Advisor")
st.markdown("Indian Stock Market Analysis & Prediction Platform")

with st.sidebar:
    st.header("Search Parameters")
    ticker_input = st.text_input("Enter NSE Ticker", value="RELIANCE.NS").upper()
    force_retrain = st.checkbox("Force Model Retraining")
    analyze_btn = st.button("Analyze Stock", type="primary", use_container_width=True)
    
    st.markdown("---")
    st.markdown("### Supported Examples:")
    st.markdown("- RELIANCE.NS")
    st.markdown("- TCS.NS")
    st.markdown("- HDFCBANK.NS")
    st.markdown("- INFY.NS")

if analyze_btn:
    data = fetch_data(ticker_input, force_retrain)
    
    if data:
        # 1. Top Metrics
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Ticker", data["ticker"])
        col2.metric("Current Price", f"₹{data['current_price']:.2f}")
        
        rec = data["recommendation"]
        rec_color = "normal"
        if rec == "BUY":
            st.toast(f"Strong BUY Signal for {data['ticker']}", icon="✅")
            
        col3.metric("AI Recommendation", rec)
        col4.metric("Confidence", f"{data['confidence']:.1f}%")
        
        st.markdown("---")
        
        # 2. Main Layout (Left: Chart, Right: AI/XAI)
        left_col, right_col = st.columns([2, 1])
        
        with left_col:
            st.subheader("📊 Market Activity")
            fig_candle = plot_candlestick(data["chart_data"])
            st.plotly_chart(fig_candle, use_container_width=True)
            
            st.subheader("📰 Financial News Sentiment")
            sent = data["sentiment"]
            s_col1, s_col2, s_col3 = st.columns(3)
            s_col1.metric("Positive News", sent["positive_count"])
            s_col2.metric("Negative News", sent["negative_count"])
            s_col3.metric("Neutral News", sent["neutral_count"])
            
            st.markdown("#### Latest Headlines")
            for article in data["articles"][:3]:
                st.markdown(f"""
                <div class="news-card">
                    <h4><a href="{article['link']}" target="_blank" style="color: #60A5FA; text-decoration: none;">{article['title']}</a></h4>
                    <p style="color: #9CA3AF; font-size: 0.9em;">{article['source']} | {article['published']}</p>
                </div>
                """, unsafe_allow_html=True)
                
        with right_col:
            st.subheader("🤖 AI Prediction")
            fig_gauge = plot_gauge(data["confidence"], data["recommendation"])
            st.plotly_chart(fig_gauge, use_container_width=True)
            
            # 3. Explainable AI Section
            st.subheader("🧠 Explainable AI (SHAP)")
            st.info("Why did the AI make this decision?")
            
            xai = data["explanation"]
            st.markdown(xai["explanation_text"])
            
            st.markdown("#### Feature Importance (Top Impactors)")
            # Create a horizontal bar chart for feature impacts
            impacts = xai["feature_impacts"][:8] # top 8
            
            features = [i["feature"] for i in impacts]
            shap_vals = [i["shap_value"] for i in impacts]
            colors = ['#10B981' if val > 0 else '#EF4444' for val in shap_vals]
            
            fig_shap = go.Figure(go.Bar(
                x=shap_vals,
                y=features,
                orientation='h',
                marker_color=colors
            ))
            fig_shap.update_layout(
                template='plotly_dark',
                height=350,
                margin=dict(l=0, r=0, t=10, b=0),
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                xaxis_title="SHAP Value (Impact on Prediction)"
            )
            st.plotly_chart(fig_shap, use_container_width=True)

else:
    st.info("👈 Enter an Indian Stock Ticker (e.g., RELIANCE.NS) and click 'Analyze Stock' to begin.")
