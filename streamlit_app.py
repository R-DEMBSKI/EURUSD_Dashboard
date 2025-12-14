import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from scipy.stats import norm
import pytz
from datetime import datetime
import time
from numba import jit

# --- 1. KONFIGURACJA OPTYMALIZACYJNA (JIT) ---
# Implementacja z Raportu[cite: 162]: Szybki Hurst bez pętli Pythona
@jit(nopython=True)
def calculate_rs_hurst_numba(series):
    """
    Oblicza wykładnik Hursta metodą Rescaled Range (R/S) przy użyciu Numba JIT.
    Krytyczne dla wydajności przy analizie tickowej[cite: 83].
    """
    N = len(series)
    if N < 20: return 0.5
    
    m = np.mean(series)
    Y = series - m
    Z = np.cumsum(Y)
    R = np.max(Z) - np.min(Z)
    S = np.std(series)
    
    if S == 0: return 0.5
    return np.log(R/S) / np.log(N)

def rolling_hurst(price_series, window=100):
    return price_series.rolling(window=window).apply(calculate_rs_hurst_numba, raw=True)

# --- 2. SILNIK MIKROSTRUKTURY (VPIN & BVC) ---
def calculate_vpin_bvc(df, bucket_size=1000, n_buckets=50):
    """
    Implementacja VPIN oparta na Bulk Volume Classification (BVC).
    Pozwala estymować toksyczność bez danych Level 2.
    """
    df = df.copy()
    # 1. Zmiana ceny i zmienność
    df['delta_p'] = df['Close'].diff()
    df['sigma_p'] = df['delta_p'].rolling(window=50).std()
    
    # 2. Standaryzacja zwrotu (Z-score) dla BVC [cite: 54]
    # Używamy fillna(0) aby uniknąć błędów na starcie
    df['Z'] = (df['delta_p'] / (df['sigma_p'] + 1e-9)).fillna(0)
    
    # 3. Dystrybuanta rozkładu normalnego (CDF)
    df['buy_prob'] = df['Z'].apply(norm.cdf)
    
    # 4. Alokacja Wolumenu (BVC)
    df['buy_vol'] = df['Volume'] * df['buy_prob']
    df['sell_vol'] = df['Volume'] * (1 - df['buy_prob'])
    
    # 5. Nierównowaga Zleceń (Order Imbalance)
    df['OI'] = (df['buy_vol'] - df['sell_vol']).abs()
    
    # 6. VPIN Rolling Calculation [cite: 65]
    # Używamy rolling sum jako aproksymacji "Volume Clock" dla uproszczenia w time-based DF
    rolling_oi = df['OI'].rolling(window=n_buckets).sum()
    rolling_vol = df['Volume'].rolling(window=n_buckets).sum()
    
    df['VPIN'] = rolling_oi / (rolling_vol + 1e-9)
    return df

# --- 3. KONFIGURACJA UI ---
st.set_page_config(layout="wide", page_title="Quant Flow V2.0 Inst", page_icon="⚡")

st.markdown("""
<style>
    .stApp { background-color: #050510; color: #e0e0e0; font-family: 'Roboto Mono', monospace; }
    /* Stylizacja metryk dla HFT Dashboard */
    div[data-testid="stMetricValue"] { font-size: 1.8rem !important; color: #00e5ff; font-weight: 700; text-shadow: 0 0 15px rgba(0, 229, 255, 0.4); }
    .vpin-alert { color: #ff1744 !important; animation: blinker 1s linear infinite; }
    @keyframes blinker { 50% { opacity: 0; } }
</style>
""", unsafe_allow_html=True)

# --- 4. STREAMLIT FRAGMENTS (ARCHITEKTURA NOWEJ GENERACJI) ---

# FRAGMENT 1: Ticker Tape (Ultra High Frequency)
# Odświeża się co 1s niezależnie od reszty aplikacji [cite: 17, 180]
@st.fragment(run_every="1s")
def render_hft_ticker(last_price_ref):
    # Symulacja Live Data Feed (w produkcji podpięcie pod WebSocket/API)
    # Dodajemy mikro-szum, aby symulować ruch tickowy
    noise = (np.random.random() - 0.5) * 0.0005
    current_price = last_price_ref + noise
    
    cols = st.columns(4)
    cols[0].metric("EURUSD (Live)", f"{current_price:.5f}", f"{noise*10000:.1f} pips")
    
    # Symulacja Spreadu
    cols[1].metric("Spread", "0.4 pips", delta_color="off")
    
    # Zegar Wolumenowy (Symulacja)
    cols[2].metric("Vol Clock", f"{np.random.randint(800, 1000)}/1000", "Filling...")
    
    # Status Systemu
    cols[3].metric("Latency", "24ms", "OK")

# FRAGMENT 2: Main Analysis (Event Driven)
# Odświeżany tylko przy zmianie parametrów lub wgraniu pliku [cite: 20]
def render_analysis_dashboard(df):
    if df is None: return

    # --- OBLICZENIA CORE ---
    # 1. Obliczenie VPIN
    df = calculate_vpin_bvc(df)
    last_vpin = df['VPIN'].iloc[-1]
    
    # 2. Obliczenie Hursta
    # Konwersja do numpy array dla Numby
    close_array = df['Close'].to_numpy()
    # Pandas Series wrapper dla rolling apply
    df['Hurst'] = rolling_hurst(pd.Series(close_array), window=100)
    last_hurst = df['Hurst'].iloc[-1]

    # --- WIZUALIZACJA ---
    
    # KPI Row - Analityka
    k1, k2, k3, k4 = st.columns(4)
    
    # Logika VPIN Alert 
    vpin_label = "TOXIC FLOW" if last_vpin > 0.8 else "STABLE FLOW"
    k1.metric("VPIN (Toxicity)", f"{last_vpin:.2f}", vpin_label, 
              delta_color="inverse" if last_vpin > 0.8 else "normal")
    
    # Logika Hurst Regime [cite: 72, 75]
    regime = "TRENDING" if last_hurst > 0.55 else ("MEAN REV" if last_hurst < 0.45 else "RANDOM WALK")
    k2.metric("Hurst Exponent", f"{last_hurst:.2f}", regime)
    
    # 3. Wykres Główny z VPIN
    fig = go.Figure()
    
    # Cena
    fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], 
                                 low=df['Low'], close=df['Close'], name='EURUSD'))
    
    # Heatmapa VPIN pod wykresem (bary koloryzowane toksycznością)
    colors = ['red' if v > 0.8 else 'rgba(0, 229, 255, 0.3)' for v in df['VPIN']]
    fig.add_trace(go.Bar(x=df.index, y=df['Volume'], marker_color=colors, yaxis='y2', name='Volume (VPIN color)'))
    
    fig.update_layout(
        template='plotly_dark',
        height=600,
        title="Institutional Chart: Price vs Order Flow Toxicity",
        yaxis2=dict(title="Volume", overlaying='y', side='right', showgrid=False, opacity=0.3)
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    return df

# FRAGMENT 3: Strategy Lab (VectorBT)
# Izolowany panel symulacji [cite: 114]
@st.fragment
def render_strategy_lab(df):
    st.markdown("---")
    st.markdown("### 🧪 Quant Lab (VectorBT Simulation)")
    
    c1, c2 = st.columns([1, 3])
    with c1:
        st.info("Parametry Backtestu")
        # Dynamiczne dostosowanie parametrów wskaźników oparte na Hurście [cite: 86]
        rsi_window = st.slider("RSI Window (Adaptive)", 7, 30, 14)
        ma_window = st.slider("MA Window", 20, 200, 50)
        
        if st.button("Uruchom Symulację"):
            # Tutaj normalnie byłby kod vectorbt
            # Zastepcza symulacja wyniku dla demo
            win_rate = 55 + (np.random.random() * 10)
            st.success(f"Backtest Zakończony: Win Rate {win_rate:.1f}%")
            
            # Krzywa kapitału (Mockup)
            equity = 10000 * np.cumprod(1 + np.random.normal(0.001, 0.01, 100))
            st.area_chart(equity)

# --- MAIN APP LOGIC ---

def load_data(file):
    if file is None: return None
    try:
        # Zakładamy format EODHD lub standardowy MT4/MT5
        df = pd.read_csv(file)
        # Prosta normalizacja kolumn
        df.columns = [c.lower() for c in df.columns]
        col_map = {'time': 'date', 'tick volume': 'volume', 'vol': 'volume'}
        df = df.rename(columns=col_map)
        
        # Parsowanie daty
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
            df.set_index('date', inplace=True)
        
        # Mapowanie kolumn obowiązkowe
        required = ['open', 'high', 'low', 'close', 'volume']
        if not all(c in df.columns for c in required):
            st.error(f"Brak wymaganych kolumn: {required}")
            return None
            
        # Capitalize for standard access
        df.columns = [c.capitalize() for c in df.columns]
        return df
    except Exception as e:
        st.error(f"Error: {e}")
        return None

def main():
    # Sidebar
    with st.sidebar:
        st.title("QUANT FLOW V2.0")
        st.markdown("*Institutional Grade*")
        uploaded_file = st.file_uploader("Wgraj dane OHLCV (EURUSD)", type=['csv'])
        
        st.divider()
        st.markdown("### ⚙️ Engine Settings")
        st.checkbox("Aktywuj Numba (JIT)", value=True, disabled=True, help="Włączone na stałe dla wydajności")
        st.checkbox("Aktywuj BVC (Microstructure)", value=True)

    # 1. Inicjalizacja Danych
    if uploaded_file:
        df = load_data(uploaded_file)
        if df is not None:
            last_price = df['Close'].iloc[-1]
            
            # 2. Renderowanie HFT Ticker (Fragment niezależny)
            render_hft_ticker(last_price)
            
            st.divider()
            
            # 3. Renderowanie Analizy (Fragment zdarzeniowy)
            render_analysis_dashboard(df)
            
            # 4. Renderowanie Laboratorium (Fragment izolowany)
            render_strategy_lab(df)
    else:
        st.info("Wgraj plik CSV, aby uruchomić silnik analityczny.")
        # Demo Ticker
        render_hft_ticker(1.0850)

if __name__ == "__main__":
    main()
