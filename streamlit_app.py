import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
from datetime import datetime
import pytz

# --- KONFIGURACJA STRONY ---
st.set_page_config(layout="wide", page_title="EURUSD Pro Trader", page_icon="💶")

# --- STYLIZACJA (Dark Mode & Forex Colors) ---
st.markdown("""
<style>
    .stApp { background-color: #0e1117; }
    div.stButton > button:first-child { background-color: #2e86de; color: white; }
    [data-testid="stMetricValue"] { font-family: 'Roboto Mono', monospace; }
</style>
""", unsafe_allow_html=True)

# --- PANEL BOCZNY (Ustawienia) ---
st.sidebar.header("⚙️ Konfiguracja")
interval = st.sidebar.select_slider("Interwał świec", options=["1m", "5m", "15m", "1h", "1d"], value="5m")
# Mapowanie interwału na okres pobierania danych (żeby wykres był czytelny)
period_map = {"1m": "1d", "5m": "5d", "15m": "1mo", "1h": "3mo", "1d": "1y"}

if st.sidebar.button("🔄 Odśwież dane"):
    st.cache_data.clear()

# --- FUNKCJA POBIERANIA DANYCH ---
@st.cache_data(ttl=60) # Cache na 60 sekund
def get_data(interval, period):
    # EURUSD=X: Para główna
    # DX-Y.NYB: Dollar Index (Główna korelacja odwrotna)
    # ^TNX: 10-Year Treasury Yield (Obligacje USA)
    # ^DE10Y: Niemieckie obligacje (Spread obligacji steruje EURUSD)
    tickers = "EURUSD=X DX-Y.NYB ^TNX" 
    data = yf.download(tickers, period=period, interval=interval, group_by='ticker', auto_adjust=True, prepost=True)
    return data

# --- LOGIKA APLIKACJI ---
st.title("🦅 EUR/USD Sniper Dashboard")

try:
    df = get_data(interval, period_map[interval])
    
    if not df.empty:
        # Wyciągamy dane do zmiennych
        eur = df['EURUSD=X']
        dxy = df['DX-Y.NYB']
        tnx = df['^TNX']

        # --- KPI SECTION (Górna belka) ---
        last_price = eur['Close'].iloc[-1]
        prev_price = eur['Close'].iloc[-2]
        change = last_price - prev_price
        pct_change = (change / prev_price) * 100
        
        # Obliczanie korelacji (ostatnie 30 świec)
        # Uwaga: yfinance może mieć różne długości danych, wyrównujemy
        min_len = min(len(eur), len(dxy))
        corr_dxy = eur['Close'].tail(min_len).corr(dxy['Close'].tail(min_len))

        # Kolumny z metrykami
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("EUR/USD Cena", f"{last_price:.5f}", f"{pct_change:.3f}%")
        k2.metric("Dollar Index (DXY)", f"{dxy['Close'].iloc[-1]:.2f}", delta_color="inverse")
        k3.metric("Korelacja (30 okresów)", f"{corr_dxy:.2f}")
        k4.metric("US 10Y Yields", f"{tnx['Close'].iloc[-1]:.3f}%")

        # --- WYKRESY (ANALIZA TECHNICZNA I KORELACJE) ---
        
        # Tworzymy układ 2 wierszy (Główny wykres + DXY pod spodem)
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                            vertical_spacing=0.05, row_heights=[0.7, 0.3],
                            subplot_titles=(f"EURUSD Price Action ({interval})", "DXY Correlation Check"))

        # 1. Świece EURUSD
        fig.add_trace(go.Candlestick(x=eur.index,
                                     open=eur['Open'], high=eur['High'],
                                     low=eur['Low'], close=eur['Close'],
                                     name="EURUSD"), row=1, col=1)

        # 2. SMA 50 (Trend krótkoterminowy)
        sma50 = eur['Close'].rolling(window=50).mean()
        fig.add_trace(go.Scatter(x=eur.index, y=sma50, mode='lines', 
                                 line=dict(color='orange', width=1), name="SMA 50"), row=1, col=1)

        # 3. DXY Line (Na dolnym panelu)
        fig.add_trace(go.Scatter(x=dxy.index, y=dxy['Close'], mode='lines',
                                 line=dict(color='#d63031', width=2), name="DXY Index"), row=2, col=1)

        # Ustawienia wyglądu wykresu
        fig.update_layout(height=700, template="plotly_dark", xaxis_rangeslider_visible=False)
        st.plotly_chart(fig, use_container_width=True)

        # --- PANEL SESYJNY (Timing) ---
        st.markdown("---")
        c1, c2 = st.columns(2)
        
        with c1:
            st.subheader("🌍 Status Sesji")
            # Czas UTC
            now_utc = datetime.now(pytz.utc)
            hour = now_utc.hour
            
            # Prosta logika sesji
            london = 7 <= hour < 16
            ny = 12 <= hour < 21
            
            if london and ny:
                st.error("⚡ LONDON / NEW YORK OVERLAP - UWAŻAJ NA ZMIENNOŚĆ!")
            elif london:
                st.warning("🇬🇧 Sesja Londyńska")
            elif ny:
                st.warning("🇺🇸 Sesja Nowojorska")
            else:
                st.info("🌙 Sesja Azjatycka / Niska płynność")
                
        with c2:
            st.subheader("📋 Notatki Tradera")
            st.text_area("Plan na dziś:", height=100, placeholder="Np. Czekam na retest poziomu 1.0850...")

    else:
        st.warning("Brak danych. Rynki mogą być zamknięte lub problem z API.")

except Exception as e:
    st.error(f"Wystąpił błąd: {e}")
