import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import norm
from datetime import datetime
import pytz

# --- 1. KONFIGURACJA UI (ULTRA MINIMAL QUANT) ---
st.set_page_config(layout="wide", page_title="EURUSD QUANTUM CORE", page_icon="⚛️", initial_sidebar_state="collapsed")

st.markdown("""
<style>
    /* GLOBAL RESET - DEEP BLACK */
    .stApp { background-color: #000000; color: #c0c0c0; font-family: 'Roboto Mono', monospace; }
    .block-container { padding: 0.5rem; max-width: 100%; }
    header, footer {visibility: hidden;}
    
    /* MODUŁY */
    .module-container { background-color: #0a0a0a; border: 1px solid #222; margin-bottom: 10px; padding: 5px; }
    .module-header {
        color: #00bcd4; font-size: 0.8rem; font-weight: bold; letter-spacing: 1px;
        margin-bottom: 5px; border-bottom: 1px solid #222; padding-bottom: 2px;
        text-transform: uppercase;
    }
    
    /* KPI METRICS (Kompaktowe) */
    div[data-testid="stMetric"] {
        background-color: #080808; border: 1px solid #222; padding: 8px; border-radius: 0px;
    }
    div[data-testid="stMetricLabel"] { font-size: 0.6rem !important; color: #666; }
    div[data-testid="stMetricValue"] { font-size: 1.2rem !important; color: #eee; font-weight: 700; }
    
    /* Visualization Container */
    .viz-container {
        border: 1px solid #333;
        background: radial-gradient(circle at center, #111 0%, #000 100%);
        height: 600px;
    }
    
    /* TABS & TABLES */
    .stTabs [data-baseweb="tab-list"] { gap: 1px; background-color: #000; }
    .stTabs [data-baseweb="tab"] { height: 30px; background-color: #111; color: #888; border: 1px solid #222; border-radius: 0px; font-size: 0.7rem; }
    .stTabs [aria-selected="true"] { background-color: #00bcd4; color: #000; }
    .dataframe { font-size: 0.7rem !important; }
</style>
""", unsafe_allow_html=True)

# --- 2. QUANT ENGINE ---
BERLIN_TZ = pytz.timezone('Europe/Berlin')

@st.cache_data(ttl=60)
def get_data(ticker, interval, period):
    try:
        df = yf.download(ticker, period=period, interval=interval, progress=False)
        if df.empty: return None
        if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.droplevel(1)
        if df.index.tz is None: df.index = df.index.tz_localize('UTC')
        df.index = df.index.tz_convert(BERLIN_TZ)
        return df
    except: return None

def calculate_stats(df):
    if df is None: return None, None, None, None
    
    # VWAP (Cena ważona wolumenem - nasze "Fair Value")
    v = df['Volume'].values
    tp = (df['High'] + df['Low'] + df['Close']) / 3
    df['VWAP'] = (tp * v).cumsum() / v.cumsum()
    current_vwap = df['VWAP'].iloc[-1]
    
    # Odchylenie Standardowe (Zmienność)
    std_dev = df['Close'].std()
    
    # Z-Score (Odległość od VWAP w odchyleniach)
    last_price = df['Close'].iloc[-1]
    z_score = (last_price - current_vwap) / std_dev
    
    # Hurst Exponent
    try:
        lags = range(2, 20)
        ts = df['Close'].tail(100).values
        tau = [np.sqrt(np.std(np.subtract(ts[lag:], ts[:-lag]))) for lag in lags]
        poly = np.polyfit(np.log(lags), np.log(tau), 1)
        hurst = poly[0] * 2.0
    except: hurst = 0.5

    return df, current_vwap, std_dev, z_score, hurst

# --- 3. DASHBOARD LAYOUT ---

with st.sidebar:
    st.markdown("### ⚛️ QUANT CORE")
    ticker = st.text_input("SYMBOL", "EURUSD=X")
    tf = st.selectbox("TIMEFRAME", ["5m", "15m", "1h"], index=1)
    period_map = {"5m":"5d", "15m":"1mo", "1h":"3mo"}

df_raw = get_data(ticker, tf, period_map.get(tf, "1mo"))

if df_raw is not None:
    df, vwap, std, z, hurst = calculate_stats(df_raw.copy())
    last_price = df['Close'].iloc[-1]
    
    # --- TOP KPI ---
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("EURUSD PRICE", f"{last_price:.5f}", f"{(last_price - df['Close'].iloc[-2]):.5f}")
    c2.metric("FAIR VALUE (VWAP)", f"{vwap:.5f}", f"Dist: {(last_price-vwap)*10000:.1f} pips")
    c3.metric("STATISTICAL Z-SCORE", f"{z:.2f}σ", "EXTREME" if abs(z)>2 else "VALUE ZONE", delta_color="inverse" if abs(z)>2 else "off")
    c4.metric("MARKET MEMORY (HURST)", f"{hurst:.2f}", "TRENDING" if hurst>0.55 else "MEAN REV")
    c5.metric("VOLATILITY (STD)", f"{std*10000:.1f} pips", "RANGE RISK")
    
    # --- MAIN VISUALIZATION: THE PROBABILITY WELL ---
    col_main, col_side = st.columns([3, 1])
    
    with col_main:
        st.markdown(f"<div class='module-header'>STATISTICAL MARKET STRUCTURE (NO-CHART VIEW)</div>", unsafe_allow_html=True)
        
        # Generowanie Krzywej Gaussa na bazie danych rynkowych
        x_axis = np.linspace(df['Low'].min(), df['High'].max(), 500)
        # Używamy VWAP jako średniej i STD z danych
        gaussian_curve = norm.pdf(x_axis, vwap, std)
        
        # Normalizacja krzywej do wizualizacji (żeby była ładnie wysoka)
        gaussian_curve = gaussian_curve / gaussian_curve.max()

        fig = go.Figure()
        
        # 1. TŁO - Krzywa Rozkładu Prawdopodobieństwa (Gradient)
        fig.add_trace(go.Scatter(
            x=x_axis, y=gaussian_curve,
            mode='lines', fill='tozeroy',
            line=dict(color='rgba(0, 188, 212, 0.2)', width=2),
            fillcolor='rgba(0, 188, 212, 0.1)',
            name='Probability Density'
        ))
        
        # 2. STREFY STATYSTYCZNE (Pionowe linie)
        # VWAP (Środek)
        fig.add_vline(x=vwap, line_width=2, line_dash="dash", line_color="#00bcd4", annotation_text="VWAP (Fair Value)", annotation_position="top right")
        
        # +1 / -1 STD (Value Area)
        fig.add_vline(x=vwap + std, line_width=1, line_color="rgba(0,255,0,0.3)", annotation_text="+1σ")
        fig.add_vline(x=vwap - std, line_width=1, line_color="rgba(0,255,0,0.3)", annotation_text="-1σ")
        
        # +2 / -2 STD (Extreme Zones)
        fig.add_vline(x=vwap + 2*std, line_width=2, line_color="rgba(255,50,50,0.5)", annotation_text="+2σ (Sell Zone)")
        fig.add_vline(x=vwap - 2*std, line_width=2, line_color="rgba(255,50,50,0.5)", annotation_text="-2σ (Buy Zone)")
        
        # 3. AKTUALNA CENA (Kursor)
        # Kolor kursora zależy od Z-Score
        cursor_color = "#ff3333" if z > 2 else "#00ff00" if z < -2 else "#ffffff"
        
        fig.add_vline(x=last_price, line_width=4, line_color=cursor_color)
        # Dodanie punktu na szczycie linii dla efektu "celownika"
        current_prob = norm.pdf(last_price, vwap, std) / norm.pdf(vwap, vwap, std)
        fig.add_trace(go.Scatter(
            x=[last_price], y=[current_prob],
            mode='markers', marker=dict(color=cursor_color, size=15, line=dict(color='white', width=2)),
            name='CURRENT PRICE'
        ))

        # Stylizacja
        fig.update_layout(
            template='plotly_dark', height=550,
            margin=dict(l=0,r=0,t=30,b=0),
            paper_bgcolor='#0a0a0a', plot_bgcolor='#0a0a0a',
            xaxis_title="CENA (Oś Wartości)", yaxis_title="Gęstość Prawdopodobieństwa (Płynność)",
            showlegend=False
        )
        # Ukrywamy oś Y (liczby nie są istotne, liczy się kształt)
        fig.update_yaxes(showticklabels=False, showgrid=False)
        fig.update_xaxes(showgrid=True, gridcolor='#222')
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Interpretacja pod wizualizacją
        if abs(z) < 1:
            st.info("ℹ️ RYNEK W RÓWNOWADZE. Cena blisko VWAP. Brak przewagi statystycznej.")
        elif z > 2:
            st.error("🚨 EKSTREMALNE WYKUPIENIE (>2σ). Prawdopodobieństwo powrotu do środka jest wysokie.")
        elif z < -2:
            st.success("🚨 EKSTREMALNE WYPRZEDANIE (<-2σ). Prawdopodobieństwo odbicia jest wysokie.")
        else:
            st.warning("⚠️ Cena w strefie ruchu (między 1σ a 2σ). Obserwuj Hurst Exponent dla kierunku.")

    with col_side:
        st.markdown(f"<div class='module-header'>CONTEXT & LIQUIDITY</div>", unsafe_allow_html=True)
        
        # Tu zostawiamy klasyczny Volume Profile jako uzupełnienie
        hist, bin_edges = np.histogram(df['Close'], bins=70, weights=df['Volume'])
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        colors = ['#00bcd4' if p >= vwap else '#e91e63' for p in bin_centers]
        
        fig_vp = go.Figure(go.Bar(
            x=hist, y=bin_centers, orientation='h',
            marker=dict(color=colors, opacity=0.5), showlegend=False
        ))
        fig_vp.add_hline(y=last_price, line_color="white", line_width=2)
        fig_vp.update_layout(
            template='plotly_dark', height=550, margin=dict(l=0,r=0,t=0,b=0),
            paper_bgcolor='#0a0a0a', plot_bgcolor='#0a0a0a',
            xaxis_visible=False, yaxis_title="Cena"
        )
        st.plotly_chart(fig_vp, use_container_width=True)

else:
    st.error("Oczekiwanie na dane...")
