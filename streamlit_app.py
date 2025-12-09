import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from datetime import datetime
import pytz

# --- 1. KONFIGURACJA STRONY (Ultra Wide) ---
st.set_page_config(layout="wide", page_title="EURUSD Pro Terminal", page_icon="🦅", initial_sidebar_state="expanded")

# --- 2. ZAAWANSOWANY CSS (Inspiracja UIverse + TradingView) ---
st.markdown("""
<style>
    /* Reset marginesów - wykorzystujemy każdy piksel */
    .block-container {
        padding-top: 0rem;
        padding-bottom: 0rem;
        padding-left: 0.5rem;
        padding-right: 0.5rem;
        max-width: 100%;
    }
    
    /* Tło aplikacji */
    .stApp { background-color: #0e1117; }
    
    /* CUSTOM CARDS (Zamiast st.metric) */
    .kpi-card {
        background-color: #161b22;
        border: 1px solid #30363d;
        border-radius: 8px;
        padding: 10px;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
        margin-bottom: 10px;
    }
    .kpi-label { font-size: 0.8rem; color: #8b949e; text-transform: uppercase; letter-spacing: 1px; }
    .kpi-value { font-size: 1.4rem; font-family: 'Roboto Mono', monospace; font-weight: bold; color: #e6edf3; }
    .kpi-delta-up { color: #2ea043; font-size: 0.9rem; }
    .kpi-delta-down { color: #f85149; font-size: 0.9rem; }
    
    /* Ukrycie paska header Streamlit (dla czystego wyglądu) */
    header {visibility: hidden;}
    
    /* Stylizacja Sidebara */
    [data-testid="stSidebar"] {
        background-color: #0d1117;
        border-right: 1px solid #30363d;
    }
</style>
""", unsafe_allow_html=True)

# --- 3. PANEL BOCZNY ---
with st.sidebar:
    st.markdown("### 🦅 Sniper Settings")
    
    interval = st.selectbox("Timeframe", ["1m", "5m", "15m", "1h", "4h", "1d"], index=1)
    
    st.divider()
    
    st.markdown("Expected Trading Range")
    st.info("Obliczamy zakres na podstawie ATR (zmienności).")
    
    show_ema = st.toggle("EMA Cloud (20/50)", value=True)
    show_pivots = st.toggle("Pivot Points", value=True)
    
    if st.button("🔄 Force Refresh", use_container_width=True):
        st.cache_data.clear()

# --- 4. ENGINE DANYCH ---
@st.cache_data(ttl=30)
def get_market_data(interval):
    # Okresy dopasowane do scalpingu
    p_map = {"1m": "2d", "5m": "5d", "15m": "1mo", "1h": "3mo", "4h": "1y", "1d": "2y"}
    
    try:
        # Pobieramy dane
        tickers = "EURUSD=X DX-Y.NYB ^TNX"
        data = yf.download(tickers, period=p_map[interval], interval=interval, group_by='ticker', progress=False)
        
        # Obsługa MultiIndex i czyszczenie
        eur = data['EURUSD=X'].copy()
        dxy = data['DX-Y.NYB'].copy()
        tnx = data['^TNX'].copy()
        
        # FILL NAN: Kluczowe dla DXY i Yields (One mają luki, Forex nie)
        # Metoda: Resample do interwału + Forward Fill
        dxy = dxy.resample(interval.replace('m', 'min')).ffill().reindex(eur.index, method='ffill')
        tnx = tnx.resample(interval.replace('m', 'min')).ffill().reindex(eur.index, method='ffill')
        
        return eur, dxy, tnx
    except Exception as e:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

df_eur, df_dxy, df_tnx = get_market_data(interval)

# --- 5. INTERFEJS: TOP BAR (HTML/CSS Metrics) ---
if not df_eur.empty:
    last = df_eur['Close'].iloc[-1]
    prev = df_eur['Close'].iloc[-2]
    change = last - prev
    pct = (change / prev) * 100
    color_cls = "kpi-delta-up" if change >= 0 else "kpi-delta-down"
    sign = "+" if change >= 0 else ""
    
    dxy_last = df_dxy['Close'].iloc[-1]
    tnx_last = df_tnx['Close'].iloc[-1]
    
    # KORELACJA (30 świec)
    corr = df_eur['Close'].tail(30).corr(df_dxy['Close'].tail(30))
    corr_color = "#f85149" if corr > -0.5 else "#2ea043" # Czerwony jeśli korelacja słabnie (dla EURUSD powinna być silnie ujemna)

    # Używamy HTML columns zamiast st.metric dla lepszego layoutu
    cols = st.columns(4)
    
    # KPI 1: EURUSD
    cols[0].markdown(f"""
    <div class="kpi-card">
        <div class="kpi-label">EUR / USD</div>
        <div class="kpi-value">{last:.5f}</div>
        <div class="{color_cls}">{sign}{change*10000:.1f} pips ({pct:.2f}%)</div>
    </div>
    """, unsafe_allow_html=True)
    
    # KPI 2: DXY
    cols[1].markdown(f"""
    <div class="kpi-card">
        <div class="kpi-label">DOLLAR INDEX (DXY)</div>
        <div class="kpi-value">{dxy_last:.2f}</div>
        <div style="font-size: 0.8rem; color: #8b949e;">Inverse Driver</div>
    </div>
    """, unsafe_allow_html=True)

    # KPI 3: CORRELATION
    cols[2].markdown(f"""
    <div class="kpi-card">
        <div class="kpi-label">DXY CORRELATION</div>
        <div class="kpi-value" style="color: {corr_color}">{corr:.2f}</div>
        <div style="font-size: 0.8rem; color: #8b949e;">Target: < -0.80</div>
    </div>
    """, unsafe_allow_html=True)
    
    # KPI 4: SESSION
    now_hour = datetime.now(pytz.utc).hour
    session_name = "Asian / Off"
    session_color = "#8b949e"
    if 7 <= now_hour < 16: session_name, session_color = "LONDON", "#2ea043"
    if 12 <= now_hour < 21: session_name, session_color = "NEW YORK", "#d29922"
    if 12 <= now_hour < 16: session_name, session_color = "⚡ OVERLAP", "#f85149"
    
    cols[3].markdown(f"""
    <div class="kpi-card">
        <div class="kpi-label">ACTIVE SESSION</div>
        <div class="kpi-value" style="color: {session_color}">{session_name}</div>
        <div style="font-size: 0.8rem; color: #8b949e;">UTC: {now_hour}:00</div>
    </div>
    """, unsafe_allow_html=True)

    # --- 6. WYKRES GŁÓWNY (TradingView Style) ---
    
    # Tworzenie subplotów: Główny (0.8) + Wolumen/Oscylator (0.2)
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.02, row_heights=[0.8, 0.2])

    # A. ŚWIECE
    fig.add_trace(go.Candlestick(
        x=df_eur.index,
        open=df_eur['Open'], high=df_eur['High'], low=df_eur['Low'], close=df_eur['Close'],
        name="Price",
        increasing_line_color='#2ea043', increasing_fillcolor='#2ea043', # GitHub Green
        decreasing_line_color='#f85149', decreasing_fillcolor='#f85149'  # GitHub Red
    ), row=1, col=1)

    # B. WSKAŹNIKI (EMA CLOUD)
    if show_ema:
        ema20 = df_eur['Close'].ewm(span=20).mean()
        ema50 = df_eur['Close'].ewm(span=50).mean()
        fig.add_trace(go.Scatter(x=df_eur.index, y=ema20, line=dict(color='#2196F3', width=1), name="EMA 20"), row=1, col=1)
        fig.add_trace(go.Scatter(x=df_eur.index, y=ema50, line=dict(color='#FF9800', width=1), name="EMA 50"), row=1, col=1)

    # C. PIVOT POINTS
    if show_pivots:
        # Prosta kalkulacja dzienna
        high_d = df_eur['High'].max()
        low_d = df_eur['Low'].min()
        close_d = df_eur['Close'].iloc[-1]
        pp = (high_d + low_d + close_d) / 3
        fig.add_hline(y=pp, line_dash="dash", line_color="white", opacity=0.3, row=1, col=1, annotation_text="PV")

    # D. DOLNY PANEL (RSI lub Wolumen)
    # Ponieważ Forex nie ma realnego wolumenu w yfinance, używamy RSI jako proxy momentum
    delta = df_eur['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    
    fig.add_trace(go.Scatter(x=df_eur.index, y=rsi, line=dict(color='#A020F0', width=2), name="RSI"), row=2, col=1)
    fig.add_hline(y=70, line_dash="dot", line_color="#333", row=2, col=1)
    fig.add_hline(y=30, line_dash="dot", line_color="#333", row=2, col=1)

    # E. LAYOUT CONFIG (Kluczowe dla UX)
    fig.update_layout(
        height=650, # Wysoki wykres
        margin=dict(l=0, r=50, t=10, b=0), # Margines po prawej na etykiety cen
        paper_bgcolor="#0e1117", # Tło zgodne z aplikacją
        plot_bgcolor="#0e1117",
        showlegend=False,
        dragmode='pan', # Domyślnie "rączka" do przesuwania
        xaxis=dict(
            rangeslider=dict(visible=False), # Ukrywamy suwak na dole
            type="date",
            showgrid=True, gridcolor="#1f242d"
        ),
        yaxis=dict(
            side="right", # Cena po prawej stronie (jak w TradingView)
            showgrid=True, gridcolor="#1f242d",
            tickformat=".5f"
        ),
        yaxis2=dict(
            side="right",
            showgrid=False,
            range=[0, 100] # RSI Range
        )
    )
    
    # Ukrywanie weekendów
    fig.update_xaxes(rangebreaks=[dict(bounds=["sat", "mon"])])

    # WYŚWIETLANIE Z KONFIGURACJĄ ZOOM
    st.plotly_chart(fig, use_container_width=True, config={
        'scrollZoom': True,       # Przybliżanie kółkiem myszy
        'displayModeBar': True,   # Pasek narzędzi
        'modeBarButtonsToRemove': ['lasso2d', 'select2d'],
        'displaylogo': False
    })

    # --- 7. MARKET DEPTH & NEWS (Split Layout) ---
    c_left, c_right = st.columns([2, 1])
    
    with c_left:
        st.markdown("##### 📝 Notatki / Poziomy")
        st.text_area("Trading Plan", height=100, placeholder="Np. Czekam na retest poziomu 1.0550, Stop Loss poniżej wicka...")
        
    with c_right:
        st.markdown("##### 📊 Imbalance (Trend)")
        # Prosta wizualizacja trendu na ostatnich 10 świecach
        last_10 = df_eur.tail(10)
        bullish = len(last_10[last_10['Close'] > last_10['Open']])
        bearish = len(last_10) - bullish
        
        st.write(f"Ostatnie 10 świec: {bullish} Up / {bearish} Down")
        st.progress(bullish / 10) # Pasek postępu (Bullishness)

else:
    st.warning("Pobieranie danych... Jeśli trwa to długo, odśwież stronę.")
