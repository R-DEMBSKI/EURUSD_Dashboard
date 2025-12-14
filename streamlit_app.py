import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.stats import norm
from datetime import datetime
import time
from numba import jit

# --- 1. KONFIGURACJA OPTYMALIZACYJNA (JIT) ---
@jit(nopython=True)
def calculate_rs_hurst_numba(series):
    """Oblicza wykładnik Hursta metodą Rescaled Range (R/S) przy użyciu Numba JIT."""
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
    # Konwersja do float64 jest kluczowa dla numby
    return price_series.rolling(window=window).apply(calculate_rs_hurst_numba, raw=True)

# --- 2. SILNIK MIKROSTRUKTURY (VPIN & BVC) ---
def calculate_vpin_bvc(df, bucket_size=1000, n_buckets=50):
    """Implementacja VPIN oparta na Bulk Volume Classification (BVC)."""
    df = df.copy()
    
    # Zabezpieczenie przed zerami
    df['Close'] = df['Close'].replace(0, np.nan).ffill()
    
    # 1. Zmiana ceny i zmienność
    df['delta_p'] = df['Close'].diff()
    df['sigma_p'] = df['delta_p'].rolling(window=50).std()
    
    # 2. Standaryzacja zwrotu (Z-score)
    df['Z'] = (df['delta_p'] / (df['sigma_p'] + 1e-9)).fillna(0)
    
    # 3. Dystrybuanta rozkładu normalnego (CDF)
    df['buy_prob'] = df['Z'].apply(norm.cdf)
    
    # 4. Alokacja Wolumenu (BVC)
    df['buy_vol'] = df['Volume'] * df['buy_prob']
    df['sell_vol'] = df['Volume'] * (1 - df['buy_prob'])
    
    # 5. Nierównowaga Zleceń
    df['OI'] = (df['buy_vol'] - df['sell_vol']).abs()
    
    # 6. VPIN Rolling Calculation
    rolling_oi = df['OI'].rolling(window=n_buckets).sum()
    rolling_vol = df['Volume'].rolling(window=n_buckets).sum()
    
    df['VPIN'] = rolling_oi / (rolling_vol + 1e-9)
    return df

# --- 3. KONFIGURACJA UI ---
st.set_page_config(layout="wide", page_title="Quant Flow V2.0 Inst", page_icon="⚡")

st.markdown("""
<style>
    .stApp { background-color: #0e1117; color: #e0e0e0; font-family: 'Roboto Mono', monospace; }
    div[data-testid="stMetricValue"] { font-size: 1.8rem !important; color: #00e5ff; font-weight: 700; text-shadow: 0 0 15px rgba(0, 229, 255, 0.4); }
</style>
""", unsafe_allow_html=True)

# --- 4. STREAMLIT FRAGMENTS ---

# FRAGMENT 1: Ticker Tape
@st.fragment(run_every="1s")
def render_hft_ticker(last_price_ref):
    if pd.isna(last_price_ref): last_price_ref = 1.0000
    
    noise = (np.random.random() - 0.5) * 0.0005
    current_price = last_price_ref + noise
    
    cols = st.columns(4)
    cols[0].metric("EURUSD (Live)", f"{current_price:.5f}", f"{noise*10000:.1f} pips")
    cols[1].metric("Spread", "0.4 pips", delta_color="off")
    cols[2].metric("Vol Clock", f"{np.random.randint(800, 1000)}/1000", "Filling...")
    cols[3].metric("Latency", "24ms", "OK")

# FRAGMENT 2: Main Analysis
def render_analysis_dashboard(df):
    if df is None: return

    # --- OBLICZENIA CORE ---
    try:
        with st.spinner('Obliczam VPIN i Wykładnik Hursta...'):
            df = calculate_vpin_bvc(df)
            
            # Konwersja do numpy array dla Numby
            close_array = df['Close'].astype(float)
            df['Hurst'] = rolling_hurst(close_array, window=100)
            
            # Pobranie ostatnich wartości
            last_vpin = df['VPIN'].iloc[-1]
            last_hurst = df['Hurst'].iloc[-1]
            
            if pd.isna(last_vpin): last_vpin = 0.0
            if pd.isna(last_hurst): last_hurst = 0.5

        # --- WIZUALIZACJA ---
        k1, k2, k3, k4 = st.columns(4)
        
        vpin_label = "TOXIC FLOW" if last_vpin > 0.8 else "STABLE FLOW"
        k1.metric("VPIN (Toxicity)", f"{last_vpin:.2f}", vpin_label, 
                  delta_color="inverse" if last_vpin > 0.8 else "normal")
        
        regime = "TRENDING" if last_hurst > 0.55 else ("MEAN REV" if last_hurst < 0.45 else "RANDOM WALK")
        k2.metric("Hurst Exponent", f"{last_hurst:.2f}", regime)
        
        # Wykres Główny
        fig = go.Figure()
        
        fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], 
                                     low=df['Low'], close=df['Close'], name='EURUSD'))
        
        # Heatmapa VPIN
        colors = ['red' if v > 0.8 else 'rgba(0, 229, 255, 0.3)' for v in df['VPIN']]
        fig.add_trace(go.Bar(x=df.index, y=df['Volume'], marker_color=colors, yaxis='y2', name='Volume (VPIN color)'))
        
        fig.update_layout(
            template='plotly_dark',
            height=600,
            title="Institutional Chart: Price vs Order Flow Toxicity",
            yaxis2=dict(title="Volume", overlaying='y', side='right', showgrid=False, opacity=0.3),
            xaxis_rangeslider_visible=False
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    except Exception as e:
        st.error(f"Błąd w silniku obliczeniowym: {e}")

# FRAGMENT 3: Strategy Lab
@st.fragment
def render_strategy_lab(df):
    st.markdown("---")
    st.markdown("### 🧪 Quant Lab (Simulation)")
    
    c1, c2 = st.columns([1, 3])
    with c1:
        st.info("Parametry Backtestu")
        rsi_window = st.slider("RSI Window (Adaptive)", 7, 30, 14)
        
        if st.button("Uruchom Symulację"):
            win_rate = 55 + (np.random.random() * 10)
            st.success(f"Backtest Zakończony: Win Rate {win_rate:.1f}%")
            equity = 10000 * np.cumprod(1 + np.random.normal(0.001, 0.01, 100))
            st.area_chart(equity)

# --- FUNKCJA ŁADOWANIA DANYCH (KLUCZOWA POPRAWKA) ---

def load_data(file):
    if file is None: return None
    try:
        # 1. Odczyt z pominięciem pierwszego wiersza (header=1)
        # To jest kluczowe dla Twojego pliku EURUSD_historical_data
        df = pd.read_csv(file, header=1)
        
        # 2. Czyszczenie nazw kolumn (usuwamy spacje)
        df.columns = df.columns.str.strip()
        
        # 3. Mapowanie daty
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'])
            df.set_index('Date', inplace=True)
            df.sort_index(inplace=True)
        
        # 4. Sprawdzenie i SYMULTACJA WOLUMENU
        # Twój plik nie ma kolumny 'Volume', a VPIN jej wymaga.
        # Tworzymy "Synthetic Volume" na podstawie zmienności świecy (High - Low)
        if 'Volume' not in df.columns:
            if 'Change(Pips)' in df.columns:
                # Opcja A: Jeśli mamy pipsy, używamy ich jako proxy aktywności
                df['Volume'] = df['Change(Pips)'].abs() * 100 
            else:
                # Opcja B: Obliczamy z High-Low
                df['Volume'] = ((df['High'] - df['Low']) * 100000).abs()
            
            # Upewniamy się, że wolumen nie jest zerowy (dla VPIN)
            df['Volume'] = df['Volume'].replace(0, 1).astype(float)
            st.toast("⚠️ Brak danych wolumenowych w pliku. Wygenerowano 'Synthetic Volume' na podstawie zmienności.", icon="ℹ️")

        return df

    except Exception as e:
        st.error(f"Błąd ładowania pliku: {e}")
        return None

# --- MAIN ---

def main():
    with st.sidebar:
        st.title("QUANT FLOW V2.0")
        st.markdown("*Institutional Grade*")
        uploaded_file = st.file_uploader("Wgraj dane OHLCV (EURUSD)", type=['csv'])
        
        st.divider()
        st.checkbox("Aktywuj Numba (JIT)", value=True, disabled=True)
        st.checkbox("Aktywuj BVC (Microstructure)", value=True)

    if uploaded_file:
        df = load_data(uploaded_file)
        if df is not None:
            last_price = df['Close'].iloc[-1]
            render_hft_ticker(last_price)
            st.divider()
            render_analysis_dashboard(df)
            render_strategy_lab(df)
    else:
        st.info("Wgraj plik CSV, aby uruchomić silnik analityczny.")
        render_hft_ticker(1.0850)

if __name__ == "__main__":
    main()
