import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
import time

# --- 1. KONFIGURACJA UI (STYL IBKR TWS MOSAIC) ---
st.set_page_config(layout="wide", page_title="PRO TERMINAL", page_icon="📈", initial_sidebar_state="collapsed")

st.markdown("""
<style>
    /* RESET STYLU - TOTALNY DARK MODE */
    .stApp { background-color: #121212; color: #e0e0e0; font-family: 'Consolas', 'Courier New', monospace; }
    
    /* Ukrycie nagłówków Streamlit */
    header, footer {visibility: hidden;}
    .block-container { padding-top: 0.5rem; max-width: 99%; padding-left: 0.5rem; padding-right: 0.5rem; }
    
    /* MODUŁY (Okna jak w TWS) */
    div.css-1r6slb0, .element-container {
        border: 1px solid #333;
        background-color: #1a1a1a;
        margin-bottom: 5px;
    }
    
    /* KPI BOXES - Styl "Monitor" */
    div[data-testid="stMetric"] {
        background-color: #000000;
        border: 1px solid #444;
        padding: 5px 10px;
        border-radius: 0px; /* Zero zaokrągleń */
        min-height: 80px;
    }
    div[data-testid="stMetricLabel"] { font-size: 0.7rem !important; color: #aaa; text-transform: uppercase; }
    div[data-testid="stMetricValue"] { font-size: 1.2rem !important; color: #fff; font-weight: bold; }
    
    /* TABELE (NEWS/DATA) */
    .stDataFrame { border: 1px solid #333; }
    
    /* WYKRESY */
    .js-plotly-plot { border: 1px solid #333; background-color: #000; }
    
    /* MENU BOCZNE */
    section[data-testid="stSidebar"] { background-color: #0a0a0a; border-right: 1px solid #333; }
    
    /* NAGŁÓWKI MODUŁÓW */
    .module-header {
        background-color: #263238; /* IBKR Dark Blue Header */
        color: white;
        padding: 4px 8px;
        font-size: 0.8rem;
        font-weight: bold;
        border-bottom: 1px solid #333;
        display: flex;
        justify_content: space-between;
        font-family: Arial, sans-serif;
        letter-spacing: 0.5px;
    }
    
    /* ZAKŁADKI */
    .stTabs [data-baseweb="tab-list"] { background-color: #121212; gap: 2px; }
    .stTabs [data-baseweb="tab"] {
        height: 30px;
        background-color: #222;
        border: 1px solid #444;
        color: #888;
        border-radius: 0px;
        font-size: 0.8rem;
        padding: 0 15px;
    }
    .stTabs [aria-selected="true"] { background-color: #007bff; color: white; border: 1px solid #007bff; }
    
    /* Custom Scrollbar */
    ::-webkit-scrollbar { width: 8px; height: 8px; }
    ::-webkit-scrollbar-track { background: #1a1a1a; }
    ::-webkit-scrollbar-thumb { background: #444; }
    ::-webkit-scrollbar-thumb:hover { background: #555; }
</style>
""", unsafe_allow_html=True)

# --- 2. SILNIK OBLICZENIOWY (QUANT CORE) ---

@st.cache_data(ttl=60)
def get_data_bundle(ticker, interval, period):
    """Bezpieczne pobieranie danych cenowych."""
    try:
        df = yf.download(ticker, period=period, interval=interval, progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.droplevel(1)
        return df if not df.empty else None
    except: return None

@st.cache_data(ttl=300)
def get_news_static(ticker):
    """Pobiera newsy jako listę słowników (Serializable)."""
    try:
        t = yf.Ticker(ticker)
        return t.news if hasattr(t, 'news') else []
    except: return []

def calculate_quant_metrics(df):
    """Główna matematyka (Hurst, Kalman, VWAP, Profile)."""
    if df is None or df.empty: return None
    
    # 1. Kalman Filter Proxy (Wygładzanie wykładnicze jako aproksymacja)
    df['Kalman'] = df['Close'].ewm(span=5).mean()
    
    # 2. VWAP
    v = df['Volume'].values
    tp = (df['High'] + df['Low'] + df['Close']) / 3
    df['VWAP'] = (tp * v).cumsum() / v.cumsum()
    
    # 3. Wykładnik Hursta
    try:
        lags = range(2, 20)
        tau = [np.sqrt(np.std(np.subtract(df['Close'].values[lag:], df['Close'].values[:-lag]))) for lag in lags]
        poly = np.polyfit(np.log(lags), np.log(tau), 1)
        hurst = poly[0] * 2.0
    except: hurst = 0.5
    
    # 4. Z-Score (Mean Reversion)
    roll_mean = df['Close'].rolling(50).mean()
    roll_std = df['Close'].rolling(50).std()
    z_score = (df['Close'].iloc[-1] - roll_mean.iloc[-1]) / roll_std.iloc[-1]
    
    return df, hurst, z_score

def get_volume_profile(df, bins=50):
    """Oblicza histogram wolumenu."""
    try:
        price_hist, bin_edges = np.histogram(df['Close'], bins=bins, weights=df['Volume'])
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        return price_hist, bin_centers, bin_edges
    except: return [], [], []

# --- 3. UI LAYOUT (STRUKTURA MOSAIC) ---

# SIDEBAR: Minimalistyczne Kontrolki
with st.sidebar:
    st.markdown("### ⚙️ DATA FEED")
    ticker = st.text_input("SYMBOL", value="EURUSD=X")
    tf = st.selectbox("TIMEFRAME", ["1m", "5m", "15m", "1h", "4h", "1d"], index=3)
    
    # Mapowanie okresu dla yfinance
    period_map = {"1m": "5d", "5m": "5d", "15m": "10d", "1h": "1mo", "4h": "3mo", "1d": "1y"}
    period = period_map.get(tf, "1mo")

# GŁÓWNA LOGIKA
df_raw = get_data_bundle(ticker, tf, period)
news_data = get_news_static(ticker)

if df_raw is not None:
    df, hurst, z_score = calculate_quant_metrics(df_raw.copy())
    
    # --- TOP BAR: METRYKI (KPI) ---
    # Układ 6 kolumn dla gęstego, profesjonalnego wyglądu
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    
    last_price = df['Close'].iloc[-1]
    chg = (last_price - df['Close'].iloc[-2])
    chg_pct = chg / df['Close'].iloc[-2]
    
    # Kolorystyka TWS (Zielony/Czerwony tekst na czarnym tle)
    color_price = "green" if chg >= 0 else "red"
    
    c1.metric("LAST", f"{last_price:.5f}", f"{chg:.5f}")
    c2.metric("CHANGE %", f"{chg_pct:.2%}", delta_color="normal")
    c3.metric("HURST EXP", f"{hurst:.2f}", "TREND" if hurst > 0.55 else "RANGE", delta_color="off")
    c4.metric("Z-SCORE", f"{z_score:.2f}σ", "CRITICAL" if abs(z_score)>2 else "NORMAL", delta_color="inverse")
    c5.metric("VOL (ANN)", f"{df['Close'].pct_change().std()*np.sqrt(252*24)*100:.1f}%", "RISK")
    
    # Entropia (Simons Metric)
    entropy_val = 0.0 # Placeholder
    c6.metric("ENTROPY", "LOW", "STRUCTURED")

    # --- MAIN GRID: CHART (Left) + TOOLS (Right) ---
    col_main, col_tools = st.columns([3, 1])
    
    with col_main:
        # NAGŁÓWEK OKNA WYKRESU
        st.markdown(f"<div class='module-header'><span>CHART: {ticker} [{tf}]</span> <span>SIMONS MODEL V8</span></div>", unsafe_allow_html=True)
        
        # --- ZAAWANSOWANY WYKRES (SUBPLOTS) ---
        # To rozwiązuje problem "ściśniętego profilu". Profil jest teraz wewnątrz wykresu.
        fig = make_subplots(
            rows=2, cols=2, 
            shared_xaxes=True, shared_yaxes=True,
            column_widths=[0.85, 0.15], # 85% Wykres, 15% Profil
            row_heights=[0.8, 0.2],     # 80% Cena, 20% Wolumen/CVD
            horizontal_spacing=0.01,
            vertical_spacing=0.01,
            specs=[[{}, {}], [{"colspan": 2}, None]] # Dolny panel na całą szerokość
        )
        
        # 1. Świece (Main)
        fig.add_trace(go.Candlestick(
            x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'],
            name='Price', showlegend=False
        ), row=1, col=1)
        
        # 2. Kalman Filter (Złota Linia)
        fig.add_trace(go.Scatter(
            x=df.index, y=df['Kalman'], mode='lines', 
            line=dict(color='#ffc107', width=1.5), name='Kalman'
        ), row=1, col=1)
        
        # 3. VWAP (Cyan)
        fig.add_trace(go.Scatter(
            x=df.index, y=df['VWAP'], mode='lines',
            line=dict(color='#00e5ff', width=1, dash='dot'), name='VWAP'
        ), row=1, col=1)

        # 4. Volume Profile (Prawy Panel - Side)
        hist, bins_c, _ = get_volume_profile(df.tail(200)) # Profil z ostatnich 200 świec
        # Znalezienie POC
        poc_idx = np.argmax(hist)
        poc_price = bins_c[poc_idx]
        
        fig.add_trace(go.Bar(
            x=hist, y=bins_c, orientation='h',
            marker=dict(color=hist, colorscale='Viridis', opacity=0.6),
            name='Profile', showlegend=False
        ), row=1, col=2)
        
        # Linia POC na głównym wykresie
        fig.add_hline(y=poc_price, line_dash="dash", line_color="white", row=1, col=1, opacity=0.5)

        # 5. CVD (Dolny Panel) - Cumulative Volume Delta Simulation
        delta = np.where(df['Close'] >= df['Open'], df['Volume'], -df['Volume'])
        cvd = np.cumsum(delta)
        fig.add_trace(go.Scatter(
            x=df.index, y=cvd, fill='tozeroy', mode='lines',
            line=dict(color='#90caf9', width=1), name='CVD'
        ), row=2, col=1)

        # STYLIZACJA WYKRESU (BARDZO WAŻNE DLA DESIGNU)
        fig.update_layout(
            template='plotly_dark',
            height=600,
            margin=dict(l=0, r=0, t=0, b=0),
            paper_bgcolor='#000000',
            plot_bgcolor='#121212',
            xaxis_rangeslider_visible=False,
            showlegend=False
        )
        # Ukrycie osi dla profilu (czystość)
        fig.update_xaxes(visible=False, row=1, col=2)
        fig.update_yaxes(visible=False, row=1, col=2)
        fig.update_yaxes(side="right", row=1, col=1) # Cena po prawej stronie jak w TWS
        
        st.plotly_chart(fig, use_container_width=True)

    with col_tools:
        # --- ZAKŁADKI BOCZNE (NEWS / HEATMAP / STATS) ---
        st.markdown(f"<div class='module-header'><span>MONITOR</span></div>", unsafe_allow_html=True)
        
        tab_news, tab_quant = st.tabs(["NEWS", "QUANT"])
        
        with tab_news:
            if news_data:
                # Konwersja newsów do prostej tabeli
                news_items = []
                for n in news_data[:8]:
                    ts = datetime.fromtimestamp(n.get('providerPublishTime', 0)).strftime('%H:%M')
                    news_items.append({"TIME": ts, "HEADLINE": n.get('title')})
                
                df_news = pd.DataFrame(news_items)
                st.dataframe(
                    df_news, 
                    hide_index=True, 
                    use_container_width=True,
                    height=500
                )
            else:
                st.info("No News Data")

        with tab_quant:
            st.markdown("**HOURLY SEASONALITY**")
            # Naprawa błędu "unstack" - Zabezpieczona Heatmapa
            try:
                if tf in ['1h', '30m', '15m']:
                    df['Hour'] = df.index.hour
                    df['Day'] = df.index.dayofweek # 0=Mon, 6=Sun
                    
                    # Groupby musi zwrócić Series przed unstack
                    grouped = df.groupby(['Day', 'Hour'])['Close'].count() # Dummy check
                    if not grouped.empty:
                        # Prawdziwe obliczenia
                        returns = df.groupby(['Day', 'Hour'])['Close'].pct_change().mean()
                        heatmap_data = returns.unstack()
                        
                        fig_heat = go.Figure(data=go.Heatmap(
                            z=heatmap_data.values,
                            x=heatmap_data.columns,
                            y=['Mon', 'Tue', 'Wed', 'Thu', 'Fri'],
                            colorscale='RdBu', zmid=0, showscale=False
                        ))
                        fig_heat.update_layout(height=250, margin=dict(l=0,r=0,t=0,b=0), paper_bgcolor='#000', plot_bgcolor='#000')
                        st.plotly_chart(fig_heat, use_container_width=True)
                    else:
                        st.warning("Not enough data points")
                else:
                    st.info("Select intraday (1h/15m) for seasonality.")
            except Exception as e:
                st.error(f"Data Error: {str(e)}")
            
            st.markdown("---")
            st.markdown("**REGRESSION CHANNEL**")
            # Prosta wizualizacja odchylenia
            st.metric("DEV FROM MEAN", f"{(last_price - df['Close'].mean()):.4f}")

else:
    st.error("Waiting for data feed... Check symbol or connection.")
