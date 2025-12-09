import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from scipy.stats import norm
from datetime import datetime, timedelta

# --- 1. KONFIGURACJA UI (BLOOMBERG STYLE) ---
st.set_page_config(layout="wide", page_title="QUANT LAB PRO", page_icon="🦅")

st.markdown("""
<style>
    /* Dark Mode Terminal Vibe */
    .stApp { background-color: #000000; color: #e0e0e0; }
    
    /* Metrics Styling */
    div[data-testid="stMetric"] {
        background-color: #111111;
        border: 1px solid #333;
        padding: 10px;
        border-radius: 4px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
    }
    div[data-testid="stMetricValue"] {
        font-family: 'Consolas', 'Courier New', monospace;
        font-size: 1.5rem !important;
    }
    
    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #0a0a0a;
        border-right: 1px solid #333;
    }
    
    /* Custom Headers */
    h1, h2, h3 { 
        color: #00ff88 !important; 
        font-family: 'Roboto Mono', monospace; 
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    /* Plotly Chart Container */
    .js-plotly-plot { border: 1px solid #222; }
</style>
""", unsafe_allow_html=True)

# --- 2. SIDEBAR - STEROWANIE ---
with st.sidebar:
    st.header("⚙️ PARAMETRY FUNDUSZU")
    selected_asset = st.selectbox("Aktywo Główne", ["EURUSD=X", "GBPUSD=X", "BTC-USD", "GC=F", "^GSPC"])
    lookback = st.slider("Okres Analizy (Dni)", 30, 730, 365)
    
    st.markdown("---")
    st.markdown("### 📡 Feed Status")
    st.success("IBKR API: DISCONNECTED (Sim Mode)")
    st.success("Yahoo Data: CONNECTED")

# --- 3. SILNIK DANYCH (ROZSZERZONY) ---
@st.cache_data(ttl=300)
def get_data(ticker, period_days):
    # Pobieranie głównego aktywa
    end_date = datetime.now()
    start_date = end_date - timedelta(days=period_days)
    
    # Główne dane (OHLC)
    df = yf.download(ticker, start=start_date, end=end_date, interval="1d", progress=False)
    
    # Flatten columns for newer yfinance versions
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(1)
        
    df['Returns'] = df['Close'].pct_change()
    df['Vol'] = df['Returns'].rolling(20).std()
    
    # Wskaźniki Techniczne do wykresu
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    df['SMA_200'] = df['Close'].rolling(window=200).mean()
    
    # Bollinger Bands
    df['BB_Mid'] = df['Close'].rolling(20).mean()
    df['BB_Std'] = df['Close'].rolling(20).std()
    df['BB_Upper'] = df['BB_Mid'] + (2 * df['BB_Std'])
    df['BB_Lower'] = df['BB_Mid'] - (2 * df['BB_Std'])
    
    df = df.dropna()

    # Dane Makro do korelacji (Benchmarki)
    tickers_macro = ["DX-Y.NYB", "^TNX", "^VIX", "BTC-USD"]
    data_macro = yf.download(tickers_macro, period="1y", interval="1d", progress=False)['Close']
    
    # Dane intraday do Heatmapy
    data_h = yf.download(ticker, period="59d", interval="1h", progress=False)
    if isinstance(data_h.columns, pd.MultiIndex):
        data_h.columns = data_h.columns.droplevel(1)
        
    return df, data_macro, data_h

# --- 4. ALGORYTMY QUANTOWE (TWOJE + ULEPSZENIA) ---

def calculate_hurst(series):
    lags = range(2, 100)
    # Zabezpieczenie przed zerami
    tau = [np.sqrt(np.std(np.subtract(series[lag:], series[:-lag]))) for lag in lags]
    poly = np.polyfit(np.log(lags), np.log(tau), 1)
    return poly[0] * 2.0

def detect_regime(df):
    X = df[['Returns', 'Vol']].copy()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    df['Regime'] = kmeans.fit_predict(X_scaled)
    return df['Regime'].iloc[-1]

# --- 5. DASHBOARD LAYOUT ---

try:
    df, df_macro, df_h = get_data(selected_asset, lookback)
    
    # A. NAGŁÓWEK TYPU "HEADS-UP DISPLAY"
    c1, c2, c3, c4, c5 = st.columns(5)
    
    curr_price = df['Close'].iloc[-1]
    prev_price = df['Close'].iloc[-2]
    delta = curr_price - prev_price
    
    hurst = calculate_hurst(df['Close'].values)
    regime = detect_regime(df)
    
    c1.metric(f"{selected_asset}", f"{curr_price:.4f}", f"{delta:.4f}")
    c2.metric("Hurst Exp (Fraktal)", f"{hurst:.2f}", "Trend" if hurst > 0.55 else "Mean Rev")
    c3.metric("Volatility (20d)", f"{df['Vol'].iloc[-1]*100:.2f}%")
    c4.metric("Market Regime", f"Cluster #{regime}")
    
    # RSI Calculation (Szybkie)
    delta_close = df['Close'].diff()
    gain = (delta_close.where(delta_close > 0, 0)).rolling(14).mean()
    loss = (-delta_close.where(delta_close < 0, 0)).rolling(14).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs)).iloc[-1]
    c5.metric("RSI (14)", f"{rsi:.1f}", "Overbought" if rsi>70 else "Oversold" if rsi<30 else "Neutral")

    # B. GŁÓWNY PANEL CHARTINGOWY
    st.markdown("---")
    col_chart, col_stats = st.columns([3, 1])
    
    with col_chart:
        st.subheader(f"📊 {selected_asset} PRICE ACTION & ALGO LEVELS")
        
        fig = go.Figure()
        
        # Candlestick
        fig.add_trace(go.Candlestick(x=df.index,
                        open=df['Open'], high=df['High'],
                        low=df['Low'], close=df['Close'],
                        name='OHLC'))
        
        # Bollinger Bands
        fig.add_trace(go.Scatter(x=df.index, y=df['BB_Upper'], line=dict(color='rgba(255, 255, 255, 0.3)', width=1), name='BB Upper'))
        fig.add_trace(go.Scatter(x=df.index, y=df['BB_Lower'], line=dict(color='rgba(255, 255, 255, 0.3)', width=1), name='BB Lower', fill='tonexty', fillcolor='rgba(255, 255, 255, 0.05)'))
        
        # SMAs
        fig.add_trace(go.Scatter(x=df.index, y=df['SMA_50'], line=dict(color='orange', width=1), name='SMA 50'))
        fig.add_trace(go.Scatter(x=df.index, y=df['SMA_200'], line=dict(color='blue', width=2), name='SMA 200'))
        
        fig.update_layout(
            template='plotly_dark',
            height=600,
            xaxis_rangeslider_visible=False,
            margin=dict(l=0, r=0, t=0, b=0),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_stats:
        st.subheader("🔗 MACIERZ KORELACJI")
        
        # Łączenie danych do korelacji
        combined_df = df_macro.copy()
        combined_df[selected_asset] = df['Close']
        corr = combined_df.corr().iloc[::-1] # Odwrócenie dla lepszego wyglądu
        
        fig_corr = px.imshow(corr, 
                             text_auto=".2f", 
                             color_continuous_scale='RdBu_r', 
                             aspect="auto",
                             title="Correlation Heatmap (1Y)")
        fig_corr.update_layout(template='plotly_dark', height=300)
        st.plotly_chart(fig_corr, use_container_width=True)
        
        st.subheader("🎯 Z-SCORE DISTRIBUTION")
        # Gaussian Distribution z Twojego kodu
        mu = df['Returns'].mean()
        sigma = df['Returns'].std()
        x_axis = np.linspace(mu - 4*sigma, mu + 4*sigma, 100)
        y_axis = norm.pdf(x_axis, mu, sigma)
        
        fig_gauss = go.Figure()
        fig_gauss.add_trace(go.Scatter(x=x_axis, y=y_axis, mode='lines', fill='tozeroy', line=dict(color='#00ff88')))
        fig_gauss.add_vline(x=df['Returns'].iloc[-1], line_width=2, line_color="red", annotation_text="TODAY")
        fig_gauss.update_layout(template='plotly_dark', height=250, showlegend=False, margin=dict(l=0,r=0,t=30,b=0))
        st.plotly_chart(fig_gauss, use_container_width=True)

    # C. ANALIZA SEZONOWOŚCI (INTRA-DAY ALPHA)
    st.markdown("---")
    st.subheader("🕒 ALPHA HOURS: GDY CZAS DAJE PRZEWAGĘ")
    
    df_h['Hour'] = df_h.index.hour
    df_h['DayOfWeek'] = df_h.index.day_name()
    df_h['Returns_Pips'] = df_h['Close'].pct_change() * 10000 
    
    # Sortowanie dni
    days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']
    heatmap_data = df_h.groupby(['DayOfWeek', 'Hour'])['Returns_Pips'].mean().unstack().reindex(days)
    
    fig_heat = px.imshow(
        heatmap_data,
        labels=dict(x="Hour (UTC)", y="Day", color="Mean Pips"),
        color_continuous_scale="RdYlGn",
        origin='upper'
    )
    fig_heat.update_layout(template='plotly_dark', height=350)
    st.plotly_chart(fig_heat, use_container_width=True)

except Exception as e:
    st.error(f"SYSTEM ERROR: {e}")
    st.info("Spróbuj wybrać inny ticker lub odśwież aplikację.")
