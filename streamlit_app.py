import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import linregress
from datetime import datetime

# --- 1. KONFIGURACJA UI (IBKR MOSAIC STYLE) ---
st.set_page_config(layout="wide", page_title="EURUSD QUANT DESK", page_icon="🦅", initial_sidebar_state="collapsed")

st.markdown("""
<style>
    /* GLOBALNY DARK MODE - STYL TWS */
    .stApp { background-color: #121212; color: #e0e0e0; font-family: 'Consolas', 'Roboto Mono', monospace; }
    
    /* Ukrycie nagłówków i stopek Streamlit */
    header, footer {visibility: hidden;}
    .block-container { padding: 0.5rem 1rem; max-width: 100%; }
    
    /* KPI METRICS (Kwadratowe, Techniczne) */
    div[data-testid="stMetric"] {
        background-color: #000000;
        border: 1px solid #333;
        padding: 5px 10px;
        border-radius: 0px;
        min-height: 80px;
    }
    div[data-testid="stMetricLabel"] { font-size: 0.7rem !important; color: #888; text-transform: uppercase; letter-spacing: 1px; }
    div[data-testid="stMetricValue"] { font-size: 1.3rem !important; color: #fff; font-weight: 700; }
    
    /* MODUŁY I KONTENERY */
    .quant-module { border: 1px solid #333; background-color: #161616; margin-bottom: 10px; }
    .module-header {
        background-color: #263238; /* IBKR Blue */
        color: #fff;
        padding: 4px 8px;
        font-size: 0.75rem;
        font-weight: bold;
        display: flex; justify_content: space-between;
        font-family: Arial, sans-serif;
    }
    
    /* WYKRESY */
    .js-plotly-plot { border: 1px solid #333; }
    
    /* ZAKŁADKI */
    .stTabs [data-baseweb="tab-list"] { gap: 2px; background-color: #121212; }
    .stTabs [data-baseweb="tab"] {
        height: 30px; background-color: #222; border: 1px solid #333; color: #888; border-radius: 0px; font-size: 0.8rem;
    }
    .stTabs [aria-selected="true"] { background-color: #007bff; color: white; border-top: 2px solid #007bff; }
</style>
""", unsafe_allow_html=True)

# --- 2. QUANT ENGINE (ALGORITHMS) ---

@st.cache_data(ttl=60)
def get_market_data(ticker, interval, period):
    """Pobiera dane OHLCV."""
    try:
        df = yf.download(ticker, period=period, interval=interval, progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.droplevel(1)
        return df if not df.empty else None
    except: return None

@st.cache_data(ttl=300)
def get_news_feed(ticker):
    """Parsuje newsy do bezpiecznej listy."""
    try:
        t = yf.Ticker(ticker)
        raw_news = t.news if hasattr(t, 'news') else []
        clean_news = []
        for n in raw_news[:10]:
            clean_news.append({
                "Czas": datetime.fromtimestamp(n.get('providerPublishTime', 0)).strftime('%H:%M'),
                "Nagłówek": n.get('title'),
                "Źródło": n.get('publisher')
            })
        return clean_news
    except: return []

def calculate_advanced_metrics(df):
    """Główny silnik obliczeniowy."""
    if df is None: return None
    
    # 1. Kalman Filter Proxy (Wygładzona EWM)
    df['Kalman'] = df['Close'].ewm(span=8).mean()
    
    # 2. VWAP (Volume Weighted Average Price)
    v = df['Volume'].values
    tp = (df['High'] + df['Low'] + df['Close']) / 3
    df['VWAP'] = (tp * v).cumsum() / v.cumsum()
    
    # 3. Synthetic CVD (Order Flow Proxy)
    # Jeśli Close > Open -> Wolumen Kupna, inaczej Sprzedaży (uproszczenie dla yfinance)
    delta = np.where(df['Close'] >= df['Open'], df['Volume'], -df['Volume'])
    df['Delta'] = delta
    df['CVD'] = np.cumsum(delta)
    
    # 4. Rolling Z-Score (Probability Bands)
    window = 50
    roll_mean = df['Close'].rolling(window).mean()
    roll_std = df['Close'].rolling(window).std()
    df['Z_Score'] = (df['Close'] - roll_mean) / roll_std
    df['Upper_2SD'] = roll_mean + (2 * roll_std)
    df['Lower_2SD'] = roll_mean - (2 * roll_std)
    
    # 5. Hurst Exponent (Local Trend Memory)
    try:
        series = df['Close'].tail(100).values
        lags = range(2, 20)
        tau = [np.sqrt(np.std(np.subtract(series[lag:], series[:-lag]))) for lag in lags]
        poly = np.polyfit(np.log(lags), np.log(tau), 1)
        hurst = poly[0] * 2.0
    except: hurst = 0.5
    
    return df, hurst

def get_volume_profile(df, bins=40):
    """Generuje profil wolumenu z ostatnich N barów."""
    try:
        price_hist, bin_edges = np.histogram(df['Close'], bins=bins, weights=df['Volume'])
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        poc_idx = np.argmax(price_hist)
        poc_price = bin_centers[poc_idx]
        return price_hist, bin_centers, poc_price
    except: return [], [], 0

# --- 3. DASHBOARD LAYOUT ---

# SIDEBAR CONTROL
with st.sidebar:
    st.markdown("### 📡 DATA FEED")
    ticker = st.text_input("INSTRUMENT", value="EURUSD=X")
    tf = st.selectbox("INTERWAŁ", ["1m", "5m", "15m", "1h", "4h"], index=2)
    
    period_map = {"1m": "5d", "5m": "5d", "15m": "10d", "1h": "1mo", "4h": "3mo"}
    
    st.markdown("---")
    st.info("System operuje w trybie analizy pojedynczego rynku. Wszystkie wskaźniki są skalibrowane pod EURUSD.")

# MAIN LOOP
df_raw = get_market_data(ticker, tf, period_map.get(tf, "1mo"))
news_data = get_news_feed(ticker)

if df_raw is not None:
    df, hurst = calculate_advanced_metrics(df_raw.copy())
    
    # --- HEADER: HIGH DENSITY KPI ---
    last_close = df['Close'].iloc[-1]
    prev_close = df['Close'].iloc[-2]
    change = last_close - prev_close
    z_val = df['Z_Score'].iloc[-1]
    
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    
    c1.metric("CENA (SPOT)", f"{last_close:.5f}", f"{change:.5f}")
    c2.metric("REŻIM (HURST)", f"{hurst:.2f}", "TREND" if hurst > 0.55 else "RANGE", delta_color="off")
    
    z_col = "inverse" if abs(z_val) > 2 else "normal"
    c3.metric("Z-SCORE (50)", f"{z_val:.2f}σ", "OVEREXTENDED" if abs(z_val)>2 else "FAIR", delta_color=z_col)
    
    # Volatility
    vol = df['Close'].pct_change().std() * np.sqrt(252*24) * 100
    c4.metric("VOLATILITY", f"{vol:.2f}%", "ANNUALIZED")
    
    # Delta (Order Flow)
    last_delta = df['Delta'].iloc[-1]
    c5.metric("LAST DELTA", f"{last_delta/1000:.1f}K", "AGRESJA", delta_color="normal")
    
    # Scoring (Placeholder pod Etap 3)
    quant_score = (hurst * 5) + ( -1 * z_val if abs(z_val) > 2 else 0 )
    c6.metric("QUANT SCORE", f"{quant_score:.1f}", "WEAK" if quant_score < 3 else "STRONG")

    # --- MAIN WORKSPACE ---
    col_chart, col_intel = st.columns([3, 1])
    
    with col_chart:
        # NAGŁÓWEK MODUŁU
        st.markdown(f"<div class='module-header'><span>WYKRES GŁÓWNY: {ticker} [{tf}]</span> <span>MOSAIC V8.0</span></div>", unsafe_allow_html=True)
        
        # OBLICZENIA PROFILU (Ostatnie 150 barów sesji)
        hist, bins, poc = get_volume_profile(df.tail(150))
        
        # KONSTRUKCJA ZAAWANSOWANEGO WYKRESU (GRID 2x2)
        # To rozwiązuje problem ściskania profilu - jest on teraz wewnątrz siatki wykresu
        fig = make_subplots(
            rows=2, cols=2,
            shared_xaxes=True, shared_yaxes=True,
            column_widths=[0.85, 0.15], # 85% Wykres, 15% Profil
            row_heights=[0.75, 0.25],   # 75% Cena, 25% CVD
            horizontal_spacing=0.01, vertical_spacing=0.02,
            specs=[[{}, {}], [{"colspan": 2}, None]] # Dolny wykres rozciągnięty
        )
        
        # 1. ŚWIECE
        fig.add_trace(go.Candlestick(
            x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'],
            name='OHLC', showlegend=False
        ), row=1, col=1)
        
        # 2. QUANT OVERLAYS (Kalman + Bands)
        fig.add_trace(go.Scatter(x=df.index, y=df['Kalman'], mode='lines', line=dict(color='#ffd700', width=1.5), name='Kalman'), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['VWAP'], mode='lines', line=dict(color='#00e5ff', width=1, dash='dot'), name='VWAP'), row=1, col=1)
        
        # Z-Score Bands (Statystyczne granice)
        fig.add_trace(go.Scatter(x=df.index, y=df['Upper_2SD'], line=dict(color='gray', width=1), name='+2SD'), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['Lower_2SD'], line=dict(color='gray', width=1), fill='tonexty', fillcolor='rgba(255,255,255,0.02)', name='-2SD'), row=1, col=1)

        # 3. VOLUME PROFILE (Side Panel)
        fig.add_trace(go.Bar(
            x=hist, y=bins, orientation='h',
            marker=dict(color=hist, colorscale='Viridis', opacity=0.5),
            name='Vol Profile', showlegend=False
        ), row=1, col=2)
        
        # POC Line (Point of Control)
        fig.add_hline(y=poc, line_dash="dash", line_color="white", row=1, col=1, opacity=0.6, annotation_text="POC")

        # 4. SYNTHETIC CVD (Dolny Panel)
        # Kolorowanie tła w zależności od trendu CVD
        fig.add_trace(go.Scatter(
            x=df.index, y=df['CVD'], mode='lines', fill='tozeroy',
            line=dict(color='#90caf9', width=1), name='CVD'
        ), row=2, col=1)

        # STYLIZACJA WYKRESU (TWS STYLE)
        fig.update_layout(
            template='plotly_dark',
            height=600,
            margin=dict(l=0, r=0, t=0, b=0),
            paper_bgcolor='#000', plot_bgcolor='#121212',
            xaxis_rangeslider_visible=False,
            showlegend=False
        )
        # Ukrycie osi profilu
        fig.update_xaxes(visible=False, row=1, col=2)
        fig.update_yaxes(visible=False, row=1, col=2)
        # Cena po prawej stronie (jak w profesjonalnych terminalach)
        fig.update_yaxes(side="right", row=1, col=1)
        
        st.plotly_chart(fig, use_container_width=True)

    with col_intel:
        # ZAKŁADKI BOCZNE
        tab_context, tab_news = st.tabs(["KONTEKST (QUANT)", "NEWSY"])
        
        with tab_context:
            st.markdown(f"<div class='module-header'><span>SEASONALITY HEATMAP</span></div>", unsafe_allow_html=True)
            
            # HEATMAPA GODZINOWA (Bezpieczna)
            try:
                if tf in ['1h', '30m', '15m']:
                    df['Hour'] = df.index.hour
                    df['Day'] = df.index.dayofweek
                    
                    # Sprawdzenie czy mamy dość danych
                    grouped = df.groupby(['Day', 'Hour'])['Close'].count()
                    if len(grouped) > 20:
                        pivot = df.groupby(['Day', 'Hour'])['Close'].pct_change().mean().unstack()
                        fig_heat = go.Figure(data=go.Heatmap(
                            z=pivot.values, x=pivot.columns, y=['Pon', 'Wt', 'Śr', 'Czw', 'Pt'],
                            colorscale='RdBu', zmid=0, showscale=False
                        ))
                        fig_heat.update_layout(height=250, margin=dict(l=0,r=0,t=0,b=0), paper_bgcolor='#000')
                        st.plotly_chart(fig_heat, use_container_width=True)
                    else:
                        st.caption("Zbieranie danych do heatmapy...")
                else:
                    st.info("Heatmapa dostępna dla interwałów H1/M15")
            except Exception as e:
                st.error("Błąd danych heatmapy")

            st.markdown("---")
            st.markdown("**SYGNAŁY INTEL:**")
            if abs(z_val) > 2.0:
                st.warning(f"⚠️ CENA EKSTREMALNA ({z_val:.2f}σ). Szukaj Mean Reversion.")
            if hurst > 0.6:
                st.success("🌊 SILNY TREND. Nie graj pod prąd.")
            if last_delta > 0 and change < 0:
                st.info("🛒 ABSORPCJA: Cena spada, delty dodatnie.")

        with tab_news:
            if news_data:
                st.dataframe(pd.DataFrame(news_data), hide_index=True, use_container_width=True, height=500)
            else:
                st.write("Brak nowych wiadomości.")

else:
    st.error("Brak danych. Sprawdź połączenie.")
