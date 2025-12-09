import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pytz

# --- 1. KONFIGURACJA STRONY (Full Width, Dark) ---
st.set_page_config(layout="wide", page_title="EURUSD Trader Terminal", page_icon="📉")

# --- 2. ADVANCED CSS (Terminal Look) ---
# To zmienia domyślny wygląd Streamlit na styl "Platforma Tradingowa"
st.markdown("""
<style>
    /* Ogólny reset */
    .stApp { background-color: #0b0e11; }
    
    /* Zmniejszenie marginesów strony (żeby wykorzystać cały ekran) */
    .block-container { padding-top: 1rem; padding-bottom: 0rem; padding-left: 1rem; padding-right: 1rem; }
    
    /* Stylizacja metryk (Kafelki na górze) */
    div[data-testid="stMetric"] {
        background-color: #1e222d;
        border: 1px solid #333;
        padding: 10px;
        border-radius: 5px;
        color: white;
    }
    [data-testid="stMetricLabel"] { font-size: 0.8rem; color: #8b949e; }
    [data-testid="stMetricValue"] { font-size: 1.2rem; font-family: 'Roboto Mono', monospace; }
    
    /* Ukrycie domyślnego menu hamburgera i stopki */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Cieńsze odstępy między kolumnami */
    div[data-testid="column"] { gap: 0.5rem; }
</style>
""", unsafe_allow_html=True)

# --- 3. PANEL BOCZNY ---
with st.sidebar:
    st.header("⚙️ Settings")
    interval = st.select_slider("Timeframe", options=["1m", "5m", "15m", "30m", "1h", "4h"], value="5m")
    st.info("💡 Aplikacja pobiera dane z opóźnieniem (yfinance). Do scalpingu na żywo zalecane API IBKR/Alpaca.")
    st.divider()
    st.write("Wskaźniki:")
    show_pivots = st.checkbox("Pivot Points", value=True)
    show_vwap = st.checkbox("VWAP Estimate", value=True)

# --- 4. DATA ENGINE (Naprawa 'nan') ---
@st.cache_data(ttl=60)
def get_pro_data(interval):
    # Mapowanie interwału na okres wsteczny
    period_map = {"1m": "2d", "5m": "5d", "15m": "1mo", "30m": "1mo", "1h": "3mo", "4h": "1y"}
    period = period_map[interval]
    
    # Pobieramy osobno, żeby łatwiej kontrolować błędy
    # EURUSD
    eur = yf.download("EURUSD=X", period=period, interval=interval, progress=False)
    
    # DXY (Indeks Dolara) - często ma luki na interwałach < 1h
    dxy = yf.download("DX-Y.NYB", period=period, interval=interval, progress=False)
    
    # US 10Y Yields
    tnx = yf.download("^TNX", period=period, interval=interval, progress=False)

    return eur, dxy, tnx

# Pobieranie
try:
    df_eur, df_dxy, df_tnx = get_pro_data(interval)
    
    # --- DATA CLEANING (Kluczowe dla 'nan') ---
    # Jeśli yfinance zwraca MultiIndex, spłaszczamy go
    if isinstance(df_eur.columns, pd.MultiIndex): df_eur.columns = df_eur.columns.get_level_values(0)
    if isinstance(df_dxy.columns, pd.MultiIndex): df_dxy.columns = df_dxy.columns.get_level_values(0)
    if isinstance(df_tnx.columns, pd.MultiIndex): df_tnx.columns = df_tnx.columns.get_level_values(0)

    # Wypełnianie braków (Forward Fill) - naprawia wykresy DXY
    df_dxy = df_dxy.resample(interval.replace('m', 'min')).ffill().reindex(df_eur.index, method='ffill')
    df_tnx = df_tnx.resample(interval.replace('m', 'min')).ffill().reindex(df_eur.index, method='ffill')

    # Ostatnie wartości
    last_price = df_eur['Close'].iloc[-1]
    prev_close = df_eur['Close'].iloc[-2]
    dxy_val = df_dxy['Close'].iloc[-1]
    tnx_val = df_tnx['Close'].iloc[-1]
    
    # Korelacja (ostatnie 50 świec)
    corr_dxy = df_eur['Close'].tail(50).corr(df_dxy['Close'].tail(50))

except Exception as e:
    st.error(f"Błąd danych: {e}")
    st.stop()

# --- 5. OBLICZENIA TECHNICZNE ---
# Pivot Points (Standard)
high_d = df_eur['High'].iloc[-200:].max() # Uproszczone High z sesji
low_d = df_eur['Low'].iloc[-200:].min()
close_d = df_eur['Close'].iloc[-1]
pp = (high_d + low_d + close_d) / 3
r1 = (2 * pp) - low_d
s1 = (2 * pp) - high_d

# --- 6. INTERFEJS (GRID LAYOUT) ---

# GÓRNY PASEK (METRYKI)
c1, c2, c3, c4 = st.columns(4)
c1.metric("EUR/USD", f"{last_price:.5f}", f"{(last_price-prev_close)*10000:.1f} pips", delta_color="normal")
c2.metric("DXY Index", f"{dxy_val:.2f}", delta_color="inverse") # Czerwony jak rośnie (bo źle dla EUR)
c3.metric("Korelacja DXY", f"{corr_dxy:.2f}")
c4.metric("US 10Y Yield", f"{tnx_val:.3f}%")

# GŁÓWNA SEKCJA (WYKRES + SIDEBAR)
col_chart, col_data = st.columns([3, 1]) # Podział ekranu 75% / 25%

with col_chart:
    # --- WYKRES PROFESJONALNY ---
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.8, 0.2], vertical_spacing=0.03)

    # Świece
    fig.add_trace(go.Candlestick(x=df_eur.index,
                                 open=df_eur['Open'], high=df_eur['High'],
                                 low=df_eur['Low'], close=df_eur['Close'],
                                 name="Price"), row=1, col=1)
    
    # Pivot Points
    if show_pivots:
        fig.add_hline(y=pp, line_dash="dash", line_color="yellow", annotation_text="Pivot", row=1, col=1)
        fig.add_hline(y=r1, line_dash="dot", line_color="red", annotation_text="R1", row=1, col=1)
        fig.add_hline(y=s1, line_dash="dot", line_color="green", annotation_text="S1", row=1, col=1)

    # VWAP (Uproszczony - Rolling Mean)
    if show_vwap:
        df_eur['VWAP_Est'] = df_eur['Close'].rolling(20).mean()
        fig.add_trace(go.Scatter(x=df_eur.index, y=df_eur['VWAP_Est'], 
                                 line=dict(color='#2980b9', width=2), name="Trend (SMA20)"), row=1, col=1)

    # RSI (Dół)
    delta = df_eur['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    
    fig.add_trace(go.Scatter(x=df_eur.index, y=rsi, line=dict(color='#9b59b6', width=2), name="RSI 14"), row=2, col=1)
    fig.add_hline(y=70, line_dash="dot", line_color="gray", row=2, col=1)
    fig.add_hline(y=30, line_dash="dot", line_color="gray", row=2, col=1)

    # STYLIZACJA WYKRESU (Brak przerw, ciemny motyw)
    fig.update_layout(
        height=650,
        margin=dict(l=10, r=10, t=10, b=10),
        template="plotly_dark",
        paper_bgcolor="#1e222d", # Tło kontenera
        plot_bgcolor="#121417",  # Tło wykresu
        xaxis_rangeslider_visible=False,
        showlegend=False
    )
    
    # Usunięcie weekendów (Rangebreaks)
    fig.update_xaxes(
        rangebreaks=[dict(bounds=["sat", "mon"])], # Ukrywa weekendy
        gridcolor="#333"
    )
    fig.update_yaxes(gridcolor="#333")

    st.plotly_chart(fig, use_container_width=True)

with col_data:
    # --- PANEL BOCZNY (MARKET CONTEXT) ---
    st.markdown("### 📊 Market Depth")
    
    # Symulacja "Order Book" (bo yfinance tego nie daje)
    # Wyliczamy siłę kupujących/sprzedających na podstawie ostatnich świec
    bull_power = df_eur[df_eur['Close'] > df_eur['Open']]['Volume'].tail(10).sum()
    bear_power = df_eur[df_eur['Close'] < df_eur['Open']]['Volume'].tail(10).sum()
    total_vol = bull_power + bear_power
    
    if total_vol > 0:
        bull_pct = (bull_power / total_vol) * 100
        bear_pct = (bear_power / total_vol) * 100
    else:
        bull_pct = 50
        bear_pct = 50

    st.write("Volume Imbalance (Last 10 candles)")
    st.progress(int(bull_pct))
    c_side1, c_side2 = st.columns(2)
    c_side1.caption(f"Buyers: {bull_pct:.0f}%")
    c_side2.caption(f"Sellers: {bear_pct:.0f}%")
    
    st.divider()
    
    # Analiza Sesji
    st.markdown("### 🕒 Session")
    now_utc = datetime.now(pytz.utc)
    hour = now_utc.hour
    
    if 7 <= hour < 16:
        st.success("🇬🇧 London Open")
    else:
        st.markdown("🇬🇧 London Closed")
        
    if 12 <= hour < 21:
        st.success("🇺🇸 New York Open")
    else:
        st.markdown("🇺🇸 New York Closed")
        
    st.divider()
    st.markdown("### 📝 Levels")
    st.code(f"""
    R1: {r1:.5f}
    PV: {pp:.5f}
    S1: {s1:.5f}
    """, language="text")
