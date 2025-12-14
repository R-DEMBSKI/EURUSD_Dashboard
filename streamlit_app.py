import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from scipy.stats import norm
import time
from datetime import datetime, timedelta

# --- 1. KONFIGURACJA UI (QUANT LAB DARK) ---
st.set_page_config(
    layout="wide", 
    page_title="QUANTFLOW V2.0 INSTITUTIONAL", 
    page_icon="⚡", 
    initial_sidebar_state="expanded"
)

# Custom CSS - Institutional Grade
st.markdown("""
<style>
    /* GLOBAL THEME */
    .stApp { background-color: #050510; color: #e0e0e0; font-family: 'Roboto Mono', monospace; }
    
    /* METRICS */
    div[data-testid="stMetric"] { background-color: #0f111a; border: 1px solid #2a2d3a; padding: 10px; border-radius: 6px; }
    div[data-testid="stMetricLabel"] { font-size: 0.75rem !important; color: #888; text-transform: uppercase; letter-spacing: 1px; }
    div[data-testid="stMetricValue"] { font-size: 1.6rem !important; color: #00e5ff; font-weight: 700; text-shadow: 0 0 15px rgba(0, 229, 255, 0.4); }
    div[data-testid="stMetricDelta"] { font-size: 0.8rem !important; }

    /* AI TERMINAL */
    .ai-terminal {
        font-family: 'Courier New', monospace;
        background-color: #080808;
        border: 1px solid #333;
        border-left: 3px solid #d500f9;
        padding: 15px;
        color: #00ff00;
        font-size: 0.85rem;
        margin-bottom: 20px;
        box-shadow: 0 0 20px rgba(213, 0, 249, 0.05);
        white-space: pre-wrap;
    }
    
    /* SECTION HEADERS */
    .section-header { 
        color: #00bcd4; 
        font-weight: 800; 
        font-size: 1.2rem; 
        text-transform: uppercase; 
        margin-top: 30px; 
        margin-bottom: 15px; 
        border-bottom: 1px solid #333; 
        padding-bottom: 5px; 
        letter-spacing: 2px;
    }
    
    /* TABLE STYLING */
    div[data-testid="stDataFrame"] { border: 1px solid #333; border-radius: 5px; }
</style>
""", unsafe_allow_html=True)

# --- 2. ZAAWANSOWANY SILNIK MATEMATYCZNY (HFT/MICROSTRUCTURE) ---

def calculate_hurst(series, window=100):
    """
    Oblicza Rolling Hurst Exponent (R/S Analysis) dla wykrywania reżimu rynku.
    H < 0.5: Mean Reversion (Powrót do średniej)
    H ~ 0.5: Random Walk (Szum)
    H > 0.5: Trending (Trend)
    """
    # Uproszczona implementacja wektorowa dla wydajności
    def get_hurst_scalar(ts):
        if len(ts) < 20: return 0.5
        lags = range(2, 20)
        tau = [np.sqrt(np.std(np.subtract(ts[lag:], ts[:-lag]))) for lag in lags]
        poly = np.polyfit(np.log(lags), np.log(tau), 1)
        return poly[0] * 2.0 

    return series.rolling(window=window).apply(get_hurst_scalar, raw=True)

def calculate_vpin_bvc(df, bucket_size_vol=1000, rolling_window=50):
    """
    Volume-Synchronized Probability of Informed Trading (VPIN)
    Używa algorytmu Bulk Volume Classification (BVC) do estymacji wolumenu kupna/sprzedaży.
    """
    df = df.copy()
    
    # 1. Zmiana ceny i zmienność
    df['Price_Change'] = df['Close'].diff()
    df['Sigma'] = df['Price_Change'].rolling(window=rolling_window).std()
    
    # 2. Standaryzacja (Z-Score) zmian ceny
    # Unikamy dzielenia przez zero dodając epsilon
    df['Z_Score_Price'] = df['Price_Change'] / (df['Sigma'] + 1e-9)
    
    # 3. Prawdopodobieństwo inicjacji przez kupującego (CDF rozkładu normalnego)
    df['Prob_Buy'] = norm.cdf(df['Z_Score_Price'])
    
    # 4. BVC: Podział wolumenu
    df['Vol_Buy'] = df['Volume'] * df['Prob_Buy']
    df['Vol_Sell'] = df['Volume'] * (1 - df['Prob_Buy'])
    
    # 5. Order Imbalance (Nierównowaga)
    df['Order_Imbalance'] = (df['Vol_Buy'] - df['Vol_Sell']).abs()
    
    # 6. VPIN = Rolling Sum(Imbalance) / Rolling Sum(Volume)
    # W wersji uproszczonej używamy okna czasowego jako aproksymacji "Volume Buckets"
    # dla płynności interfejsu (prawdziwy VPIN wymagałby resamplingu po wolumenie)
    df['VPIN'] = df['Order_Imbalance'].rolling(window=rolling_window).sum() / \
                 df['Volume'].rolling(window=rolling_window).sum()
                 
    return df

def detect_fvg(df):
    """Wykrywa Fair Value Gaps (Nierównowagi cenowe)"""
    fvgs = []
    if len(df) < 3: return fvgs
    
    # Wektoryzacja dla szybkości
    highs = df['High'].values
    lows = df['Low'].values
    times = df.index
    
    for i in range(2, len(df)):
        # Bullish FVG: Low[i] > High[i-2]
        if lows[i] > highs[i-2]:
            fvgs.append({
                'type': 'bull', 
                'top': lows[i], 
                'bottom': highs[i-2], 
                'x0': times[i-2],
                'x1': times[i]
            })
        # Bearish FVG: High[i] < Low[i-2]
        elif highs[i] < lows[i-2]:
            fvgs.append({
                'type': 'bear', 
                'top': lows[i-2], 
                'bottom': highs[i], 
                'x0': times[i-2],
                'x1': times[i]
            })
    return fvgs

def process_data(df):
    if df.empty: return df
    
    # Podstawowe metryki
    df['LogRet'] = np.log(df['Close'] / df['Close'].shift(1))
    df['VWAP'] = (df['Volume'] * (df['High'] + df['Low'] + df['Close']) / 3).cumsum() / df['Volume'].cumsum()
    
    # Z-Score Ceny (Mean Reversion Signal)
    df['Price_Z'] = (df['Close'] - df['Close'].rolling(20).mean()) / df['Close'].rolling(20).std()
    
    # Zaawansowane (z Raportu)
    df = calculate_vpin_bvc(df)
    df['Hurst'] = calculate_hurst(df['Close'], window=50) # Krótsze okno dla reaktywności
    
    return df

# --- 3. LOADER DANYCH ---

def smart_data_loader(uploaded_file):
    if uploaded_file is None: 
        return generate_mock_data() # Fallback do demo
        
    try:
        uploaded_file.seek(0)
        # Próba detekcji formatu
        try:
            df = pd.read_csv(uploaded_file)
        except:
            df = pd.read_csv(uploaded_file, sep=';') # Czasem MT4 eksportuje z ;
            
        # Czyszczenie kolumn
        df.columns = df.columns.str.strip().str.lower()
        col_map = {
            'time': 'date', 'datetime': 'date', 
            'open': 'open', 'high': 'high', 'low': 'low', 'close': 'close', 
            'vol': 'volume', 'volume': 'volume', 'tickvol': 'volume'
        }
        df = df.rename(columns=col_map)
        
        # Wymagane kolumny
        req = ['date', 'open', 'high', 'low', 'close']
        if not all(c in df.columns for c in req):
            st.error(f"Brak wymaganych kolumn. Znaleziono: {list(df.columns)}")
            return None
            
        df['date'] = pd.to_datetime(df['date'])
        df = df.set_index('date').sort_index()
        
        # Fallback dla volume
        if 'volume' not in df.columns: df['volume'] = 1000
        
        # Capitalize columns for internal logic
        df.columns = [c.capitalize() for c in df.columns]
        
        return df
    except Exception as e:
        st.error(f"Error: {e}")
        return None

def generate_mock_data():
    """Generuje dane syntetyczne GBM do demo"""
    dates = pd.date_range(end=datetime.now(), periods=500, freq='1min')
    np.random.seed(42)
    # Geometric Brownian Motion
    dt = 1/500
    mu = 0.0001
    sigma = 0.01
    b = np.random.normal(0, 1, 500)
    W = b.cumsum()
    t = np.linspace(0, 1, 500)
    drift = (mu - 0.5 * sigma**2) * t
    diffusion = sigma * W
    price = 1.0850 * np.exp(drift + diffusion)
    
    df = pd.DataFrame(index=dates)
    df['Close'] = price
    df['Open'] = price + np.random.normal(0, 0.0002, 500)
    df['High'] = np.maximum(df['Open'], df['Close']) + abs(np.random.normal(0, 0.0002, 500))
    df['Low'] = np.minimum(df['Open'], df['Close']) - abs(np.random.normal(0, 0.0002, 500))
    df['Volume'] = np.random.randint(50, 2000, 500)
    return df

# --- 4. STREAMLIT FRAGMENTS (NOWOŚĆ - ARCHITEKTURA HFT) ---

@st.fragment(run_every="2s")
def live_ticker_simulation(last_close, last_vpin):
    """
    Symulacja High-Frequency Tickera.
    Używa st.fragment, aby odświeżać tylko ten element co 1s bez przeładowania całej aplikacji.
    """
    # Symulacja ruchu ceny "pomiędzy świecami"
    noise = np.random.normal(0, 0.00005)
    live_price = last_close * (1 + noise)
    spread = 0.00008 + abs(noise) * 0.1
    
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("EURUSD LIVE", f"{live_price:.5f}", f"{noise*10000:.1f} pips")
    c2.metric("SPREAD", f"{spread*10000:.1f} pips", "Raw ECN")
    
    # Kolorowanie VPIN (Toksyczność)
    vpin_color = "normal"
    if last_vpin > 0.7: vpin_color = "off" # Red/Inverse in metric usually
    c3.metric("VPIN (TOXICITY)", f"{last_vpin:.2f}", "Risk Level", delta_color=vpin_color)
    
    # Zegar
    now = datetime.now().strftime("%H:%M:%S")
    c4.metric("SERVER TIME (UTC)", now, "London Session")

# --- 5. LOGIKA GŁÓWNA ---

# SIDEBAR
with st.sidebar:
    st.markdown("## ⚡ QUANT FLOW V2")
    st.markdown("Isntitutional Grade Dashboard")
    
    uploaded_file = st.file_uploader("Upload Market Data (CSV)", type=['csv'])
    
    st.markdown("---")
    st.markdown("### ⚙️ Engine Config")
    show_fvg = st.checkbox("Show FVG (Imbalance)", value=True)
    show_vwap = st.checkbox("Show VWAP", value=True)
    hurst_threshold = st.slider("Hurst Trend Threshold", 0.5, 0.8, 0.6)
    
    st.markdown("---")
    st.info("System Ready. Loaded modules: VPIN, Hurst, FVG, GBM.")

# DATA LOADING & CALCS
df = smart_data_loader(uploaded_file)
if df is None: st.stop()
df = process_data(df)

# POBRANIE OSTATNICH WARTOŚCI DLA TICKERA
last_close_val = df['Close'].iloc[-1]
last_vpin_val = df['VPIN'].iloc[-1] if not np.isnan(df['VPIN'].iloc[-1]) else 0.5

# --- UI LAYOUT ---

# 1. LIVE TICKER (FRAGMENT)
live_ticker_simulation(last_close_val, last_vpin_val)

st.markdown("---")

# 2. MAIN CHARTS (PLOTLY)
tab1, tab2, tab3 = st.tabs(["⚡ TRADING DESK", "🧠 DEEP LAB", "📊 DATA FEED"])

with tab1:
    # ANALIZA AI
    curr_hurst = df['Hurst'].iloc[-1]
    curr_vpin = df['VPIN'].iloc[-1]
    curr_price_z = df['Price_Z'].iloc[-1]
    
    # Generowanie Narracji
    narrative = []
    narrative.append(f"Analyzing {len(df)} candles...")
    
    # Analiza Fraktalna
    if curr_hurst > 0.6:
        narrative.append(f"► REGIME: TRENDING DETECTED (Hurst: {curr_hurst:.2f} > 0.5).")
        narrative.append("  ACTION: Use Pullback strategies. Disable Mean Reversion bots.")
    elif curr_hurst < 0.4:
        narrative.append(f"► REGIME: MEAN REVERSION (Hurst: {curr_hurst:.2f} < 0.5).")
        narrative.append("  ACTION: Look for fading moves at Bollinger Bands.")
    else:
        narrative.append(f"► REGIME: RANDOM WALK / NOISE (Hurst: {curr_hurst:.2f}). Cash is King.")
        
    # Analiza Toksyczności
    if curr_vpin > 0.7:
        narrative.append(f"⚠️ CRITICAL: HIGH FLOW TOXICITY (VPIN: {curr_vpin:.2f}).")
        narrative.append("  MARKET MAKER RISK: High probability of spread widening or flash crash.")
    else:
        narrative.append(f"► FLOW: STABLE (VPIN: {curr_vpin:.2f}). Liquidity is sufficient.")
        
    ai_text = "\n".join(narrative)
    st.markdown(f"<div class='ai-terminal'>SYSTEM LOG:\n{ai_text}</div>", unsafe_allow_html=True)

    # WYKRES GŁÓWNY
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, 
                        vertical_spacing=0.03, row_heights=[0.6, 0.2, 0.2],
                        subplot_titles=("Price Action & FVG", "VPIN (Order Flow Toxicity)", "Hurst Exponent (Regime)"))

    # Row 1: Price
    fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='Price'), row=1, col=1)
    if show_vwap:
        fig.add_trace(go.Scatter(x=df.index, y=df['VWAP'], line=dict(color='#ffab00', width=1.5), name='VWAP'), row=1, col=1)

    # FVG
    if show_fvg:
        fvgs = detect_fvg(df)
        for fvg in fvgs[-50:]: # Pokaż tylko ostatnie 50 FVG żeby nie zamulić
            color = 'rgba(0, 255, 0, 0.15)' if fvg['type'] == 'bull' else 'rgba(255, 0, 0, 0.15)'
            fig.add_shape(type="rect", x0=fvg['x0'], x1=df.index[-1] + timedelta(minutes=10), 
                          y0=fvg['bottom'], y1=fvg['top'], fillcolor=color, line_width=0, row=1, col=1)

    # Row 2: VPIN
    fig.add_trace(go.Scatter(x=df.index, y=df['VPIN'], fill='tozeroy', line=dict(color='#d500f9', width=1), name='VPIN'), row=2, col=1)
    fig.add_hline(y=0.7, line_dash="dot", line_color="red", row=2, col=1, annotation_text="Toxic Threshold")

    # Row 3: Hurst
    fig.add_trace(go.Scatter(x=df.index, y=df['Hurst'], line=dict(color='#00e5ff', width=1), name='Hurst'), row=3, col=1)
    fig.add_hline(y=0.5, line_color="gray", row=3, col=1)
    
    fig.update_layout(height=800, template='plotly_dark', margin=dict(l=0, r=0, t=30, b=0), showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.markdown("<div class='section-header'>Microstructure Lab</div>", unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    
    # Rozkład Wolumenu (BVC Analysis)
    buy_vol_total = df['Vol_Buy'].sum()
    sell_vol_total = df['Vol_Sell'].sum()
    
    fig_pie = go.Figure(data=[go.Pie(labels=['Buy Pressure', 'Sell Pressure'], values=[buy_vol_total, sell_vol_total], hole=.6)])
    fig_pie.update_layout(template='plotly_dark', title="Total Volume Delta (BVC Model)")
    fig_pie.update_traces(marker=dict(colors=['#00e5ff', '#ff1744']))
    c1.plotly_chart(fig_pie, use_container_width=True)
    
    # Histogram Zwrotów
    fig_hist = px.histogram(df, x='LogRet', nbins=100, title="Return Distribution (Fat Tails Check)")
    fig_hist.update_layout(template='plotly_dark')
    c2.plotly_chart(fig_hist, use_container_width=True)
    
    st.markdown("### Backtest Symulacyjny (Vector Logic)")
    st.markdown("Strategia: Kupuj gdy Hurst > 0.6 (Trend) i Cena > VWAP.")
    
    # Vector Backtest Logic
    df['Signal'] = np.where((df['Hurst'] > 0.6) & (df['Close'] > df['VWAP']), 1, 0)
    df['Strategy_Ret'] = df['Signal'].shift(1) * df['LogRet']
    df['Equity'] = (1 + df['Strategy_Ret']).cumprod()
    
    fig_eq = go.Figure()
    fig_eq.add_trace(go.Scatter(x=df.index, y=df['Equity'], mode='lines', name='Equity Curve', line=dict(color='#00ff00')))
    fig_eq.update_layout(height=300, template='plotly_dark', title="Symulowana Krzywa Kapitału")
    st.plotly_chart(fig_eq, use_container_width=True)

with tab3:
    st.dataframe(df.tail(100).style.format("{:.5f}"), use_container_width=True)
