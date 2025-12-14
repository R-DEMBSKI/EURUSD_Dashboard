import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from streamlit_option_menu import option_menu
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import NearestNeighbors
from scipy.stats import norm
import pytz
from datetime import datetime

# --- 1. KONFIGURACJA UI (QUANT LAB DARK) ---
st.set_page_config(layout="wide", page_title="QUANTFLOW INSTITUTIONAL", page_icon="⚡", initial_sidebar_state="expanded")

# Custom CSS
st.markdown("""
<style>
    /* GLOBAL THEME */
    .stApp { background-color: #050510; color: #e0e0e0; font-family: 'Roboto Mono', monospace; }
    .block-container { padding-top: 1rem; padding-bottom: 2rem; }
    
    /* UKRYCIE STANDARDOWYCH ELEMENTÓW */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* KPIS & METRICS */
    div[data-testid="stMetric"] { background-color: #0f111a; border: 1px solid #2a2d3a; padding: 10px; border-radius: 6px; }
    div[data-testid="stMetricLabel"] { font-size: 0.7rem !important; color: #888; text-transform: uppercase; }
    div[data-testid="stMetricValue"] { font-size: 1.4rem !important; color: #00e5ff; font-weight: 700; text-shadow: 0 0 10px rgba(0, 229, 255, 0.3); }
    
    /* AI TERMINAL STYLE */
    .ai-terminal {
        font-family: 'Courier New', monospace;
        background-color: #0a0a0a;
        border: 1px solid #333;
        border-left: 3px solid #d500f9;
        padding: 15px;
        color: #00ff00;
        font-size: 0.85rem;
        margin-bottom: 20px;
        box-shadow: 0 0 15px rgba(213, 0, 249, 0.1);
    }
    .warning-text { color: #ffab00; }
    .bearish-text { color: #ff1744; }
    .bullish-text { color: #00e5ff; }

    /* SECTIONS */
    .section-header { color: #00bcd4; font-weight: 800; font-size: 1.1rem; text-transform: uppercase; margin-top: 20px; margin-bottom: 10px; border-bottom: 1px solid #333; padding-bottom: 5px; }
</style>
""", unsafe_allow_html=True)

# --- 2. ZAAWANSOWANE OBLICZENIA QUANT (MATH ENGINE) ---

def calculate_institutional_metrics(df):
    if df.empty: return df
    
    # 1. Log Returns (Rozkład Normalny)
    df['LogRet'] = np.log(df['Close'] / df['Close'].shift(1))
    
    # 2. Zmienność i Z-Score (Mean Reversion)
    df['Volatility'] = df['LogRet'].rolling(window=20).std()
    df['Z_Score'] = (df['Close'] - df['Close'].rolling(window=20).mean()) / (df['Close'].rolling(window=20).std())
    
    # 3. Market Efficiency Ratio (Chaos Theory)
    # Efficiency = Net Change / Sum of Absolute Changes
    window = 14
    change = df['Close'].diff(window).abs()
    volatility = df['Close'].diff().abs().rolling(window).sum()
    df['Efficiency_Ratio'] = change / volatility
    
    # 4. Synthetic Delta (Szacowanie Presji)
    # Close > Open -> Buy Pressure approximation
    # Close < Open -> Sell Pressure approximation
    df['Delta_Est'] = np.where(df['Close'] > df['Open'], 
                               df['Volume'] * ((df['Close'] - df['Low']) / (df['High'] - df['Low'] + 1e-9)),
                               -df['Volume'] * ((df['High'] - df['Close']) / (df['High'] - df['Low'] + 1e-9)))
    
    # VWAP
    v = df['Volume'].values
    tp = (df['High'] + df['Low'] + df['Close']) / 3
    cum_vol = v.cumsum()
    cum_vol[cum_vol == 0] = 1 
    df['VWAP'] = (tp * v).cumsum() / cum_vol
    
    return df

def monte_carlo_simulation(df, horizon=20, simulations=200):
    last_price = df['Close'].iloc[-1]
    last_vol = df['LogRet'].iloc[-50:].std() # Zmienność z ostatnich 50 świec
    
    simulation_df = pd.DataFrame()
    
    for x in range(simulations):
        # Brownian Motion
        daily_vol = last_vol 
        price_series = [last_price]
        
        for y in range(horizon):
            price = price_series[-1] * (1 + np.random.normal(0, daily_vol))
            price_series.append(price)
            
        simulation_df[x] = price_series
        
    # Obliczanie kwantyli (Stożek)
    mean_path = simulation_df.mean(axis=1)
    upper_95 = simulation_df.quantile(0.95, axis=1)
    lower_05 = simulation_df.quantile(0.05, axis=1)
    
    # Daty przyszłe
    last_date = df.index[-1]
    time_delta = last_date - df.index[-2]
    future_dates = [last_date + i * time_delta for i in range(len(mean_path))]
    
    return future_dates, mean_path, upper_95, lower_05

def generate_ai_narrative(df):
    last = df.iloc[-1]
    z_score = last['Z_Score']
    eff = last['Efficiency_Ratio']
    rsi = 50 # Placeholder, można dodać obliczenie RSI
    
    report = []
    
    # 1. Regime Detection
    if eff > 0.6: report.append(f"MARKET REGIME: TRENDING (Eff: {eff:.2f}). Structure is organized.")
    elif eff < 0.3: report.append(f"MARKET REGIME: CHAOS/NOISE (Eff: {eff:.2f}). Algorithmic trading risky.")
    else: report.append(f"MARKET REGIME: TRANSITIONAL (Eff: {eff:.2f}).")
    
    # 2. Volatility Alert
    if abs(z_score) > 2.0:
        report.append(f"CRITICAL ALERT: Price Z-Score is {z_score:.2f}. EXTREME DEVIATION. Expect Mean Reversion.")
    
    # 3. Bias
    if last['Close'] > last['VWAP']:
        report.append("INSTITUTIONAL BIAS: BULLISH (Price > VWAP).")
    else:
        report.append("INSTITUTIONAL BIAS: BEARISH (Price < VWAP).")
        
    return "\n".join(report)

# --- 3. LOADER I HELPERY ---

def smart_data_loader(uploaded_file):
    try:
        uploaded_file.seek(0)
        # Szybka detekcja nagłówka dla MT4
        first_line = uploaded_file.readline().decode('utf-8')
        uploaded_file.seek(0)
        header_row = 1 if "Historical Data" in first_line else 0
        
        df = pd.read_csv(uploaded_file, header=header_row, index_col=False, engine='python')
        df.columns = df.columns.str.strip()
        df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
        
        col_map = {
            'Time': 'Date', 'date': 'Date', 'DATE': 'Date',
            'open': 'Open', 'OPEN': 'Open', 'high': 'High', 'HIGH': 'High',
            'low': 'Low', 'LOW': 'Low', 'close': 'Close', 'CLOSE': 'Close',
            'vol': 'Volume', 'VOL': 'Volume', 'volume': 'Volume', 'Tick Volume': 'Volume'
        }
        df = df.rename(columns=col_map)
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
            df = df.dropna(subset=['Date']).sort_values('Date').set_index('Date')
        
        if 'Volume' not in df.columns: df['Volume'] = 1000
        return df
    except Exception as e:
        return None

def detect_fvg(df):
    fvgs = []
    if len(df) < 3: return fvgs
    for i in range(2, len(df)):
        try:
            if df['Low'].iloc[i] > df['High'].iloc[i-2]:
                fvgs.append({'type': 'bull', 'top': df['Low'].iloc[i], 'bottom': df['High'].iloc[i-2], 'x0': df.index[i-2]})
            elif df['High'].iloc[i] < df['Low'].iloc[i-2]:
                fvgs.append({'type': 'bear', 'top': df['Low'].iloc[i-2], 'bottom': df['High'].iloc[i], 'x0': df.index[i-2]})
        except: continue
    return fvgs

def generate_mock_data():
    dates = pd.date_range(end=pd.Timestamp.now(), periods=500, freq='H')
    np.random.seed(42)
    price = 1.1000 * np.exp(np.cumsum(np.random.normal(0.0001, 0.002, 500)))
    df = pd.DataFrame(index=dates)
    df['Close'] = price
    df['Open'] = price * 1.0005
    df['High'] = price * 1.0015
    df['Low'] = price * 0.9985
    df['Volume'] = np.random.randint(100, 5000, 500)
    return df

# --- 4. GŁÓWNA APLIKACJA ---

# SIDEBAR CONTROLS
with st.sidebar:
    st.markdown("## ⚡ QUANT FLOW V2.0")
    uploaded_file = st.file_uploader("📂 Wgraj Dane (CSV)", type=['csv'])
    
    st.markdown("### ⚙️ Engine Layer")
    show_fvg = st.toggle("Show FVG (Smart Money)", value=True)
    show_monte = st.toggle("Monte Carlo Simulation", value=True)
    show_delta = st.toggle("Synthetic Delta", value=True)
    
    st.markdown("### 🧠 AI Analyst Log")
    ai_placeholder = st.empty()

# LOAD DATA
if uploaded_file:
    df = smart_data_loader(uploaded_file)
    src_label = "USER DATA"
else:
    df = generate_mock_data()
    src_label = "SIMULATION"

if df is None or df.empty:
    st.error("Błąd danych.")
    st.stop()

# PRZELICZANIE METRYK
df = calculate_institutional_metrics(df)
fvgs = detect_fvg(df)

# GLOBAL CLOCK HEADER
cols_time = st.columns(4)
now_utc = datetime.now(pytz.utc)
zones = {'WARSAW': 'Europe/Warsaw', 'LONDON': 'Europe/London', 'NEW YORK': 'America/New_York', 'TOKYO': 'Asia/Tokyo'}

for i, (city, zone) in enumerate(zones.items()):
    tz = pytz.timezone(zone)
    local_time = now_utc.astimezone(tz).strftime("%H:%M:%S")
    cols_time[i].metric(city, local_time, border=True)

st.markdown("---")

# KPI ROW
last_price = df['Close'].iloc[-1]
chg = (last_price / df['Close'].iloc[-2] - 1) * 100
vol_z = df['Z_Score'].iloc[-1]

k1, k2, k3, k4 = st.columns(4)
k1.metric("ASSET", "EURUSD", src_label)
k2.metric("PRICE", f"{last_price:.5f}", f"{chg:+.2f}%")
k3.metric("VOLATILITY Z-SCORE", f"{vol_z:.2f}", "Sigma")
k4.metric("EFFICIENCY", f"{df['Efficiency_Ratio'].iloc[-1]:.2f}", "Fractal Dim")

# GENERATE AI REPORT
ai_report = generate_ai_narrative(df)
ai_placeholder.markdown(f"<div class='ai-terminal'>{ai_report.replace(chr(10), '<br>')}</div>", unsafe_allow_html=True)

# --- VISUALIZATION TAB ---
tabs = option_menu(None, ["Quant Chart", "Chronos Heatmap", "Monte Carlo Lab"], 
    icons=['graph-up', 'clock', 'activity'], orientation="horizontal")

if tabs == "Quant Chart":
    # 1. MAIN CHART
    fig = go.Figure()
    
    # Candles
    fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='Price'))
    
    # VWAP
    fig.add_trace(go.Scatter(x=df.index, y=df['VWAP'], line=dict(color='orange', width=1.5), name='VWAP'))
    
    # FVG
    if show_fvg:
        for fvg in fvgs[-20:]:
            color = 'rgba(0, 255, 100, 0.15)' if fvg['type'] == 'bull' else 'rgba(255, 0, 50, 0.15)'
            fig.add_shape(type="rect", x0=fvg['x0'], x1=df.index[-1], y0=fvg['bottom'], y1=fvg['top'], 
                          fillcolor=color, line_width=0)

    # Monte Carlo Fan (Overlay)
    if show_monte:
        fd, mean_p, up_95, lo_05 = monte_carlo_simulation(df)
        # Rysujemy chmurę (Cloud)
        fig.add_trace(go.Scatter(x=fd, y=up_95, mode='lines', line=dict(width=0), showlegend=False))
        fig.add_trace(go.Scatter(x=fd, y=lo_05, mode='lines', line=dict(width=0), fill='tonexty', 
                                 fillcolor='rgba(138, 43, 226, 0.2)', name='Probability Cloud (90%)'))
        fig.add_trace(go.Scatter(x=fd, y=mean_p, mode='lines', line=dict(color='#d500f9', dash='dot'), name='Mean Path'))

    fig.update_layout(height=600, template='plotly_dark', xaxis_rangeslider_visible=False, margin=dict(l=0,r=0),
                      title="Institutional Price Action & AI Structure")
    st.plotly_chart(fig, use_container_width=True)
    
    # 2. SYNTHETIC DELTA
    if show_delta:
        fig_delta = go.Figure()
        colors = ['#00e5ff' if x >= 0 else '#ff1744' for x in df['Delta_Est']]
        fig_delta.add_trace(go.Bar(x=df.index, y=df['Delta_Est'], marker_color=colors, name='Delta Pressure'))
        fig_delta.update_layout(height=200, template='plotly_dark', margin=dict(t=0, b=0), title="Synthetic Delta Pressure (Buy/Sell Flow)")
        st.plotly_chart(fig_delta, use_container_width=True)

elif tabs == "Chronos Heatmap":
    st.markdown("<div class='section-header'>TIME & VOLATILITY STATISTICS</div>", unsafe_allow_html=True)
    
    # Przygotowanie danych do Heatmapy
    df['Day'] = df.index.day_name()
    df['Hour'] = df.index.hour
    df['Range'] = df['High'] - df['Low'] # Zmienność
    
    # Pivot Table: Dzień vs Godzina -> Średnia Zmienność
    heatmap_data = df.pivot_table(index='Day', columns='Hour', values='Range', aggfunc='mean')
    days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']
    heatmap_data = heatmap_data.reindex(days_order)
    
    fig_hm = px.imshow(heatmap_data, labels=dict(x="Godzina (UTC)", y="Dzień", color="Volatility (Pips)"),
                       color_continuous_scale='Magma', aspect='auto')
    fig_hm.update_layout(height=600, template='plotly_dark', title="Market Heartbeat: Kiedy handlować?")
    st.plotly_chart(fig_hm, use_container_width=True)
    st.info("💡 Jaśniejsze pola oznaczają statystycznie wyższą zmienność. Ciemne pola to czas konsolidacji (nie handluj).")

elif tabs == "Monte Carlo Lab":
    st.markdown("<div class='section-header'>PROBABILISTIC FUTURE MODELING</div>", unsafe_allow_html=True)
    fd, mean_p, up_95, lo_05 = monte_carlo_simulation(df, simulations=500)
    
    fig_mc = go.Figure()
    # Symulacja 50 losowych ścieżek dla tła
    last_price = df['Close'].iloc[-1]
    last_vol = df['LogRet'].iloc[-50:].std()
    for _ in range(50):
        path = [last_price]
        for _ in range(len(fd)-1):
            path.append(path[-1] * (1 + np.random.normal(0, last_vol)))
        fig_mc.add_trace(go.Scatter(x=fd, y=path, mode='lines', line=dict(color='rgba(255,255,255,0.05)', width=1), showlegend=False))
        
    fig_mc.add_trace(go.Scatter(x=fd, y=mean_p, line=dict(color='#00e5ff', width=3), name='Expected Path'))
    fig_mc.update_layout(height=500, template='plotly_dark', title="Monte Carlo: 500 Simulations")
    st.plotly_chart(fig_mc, use_container_width=True)
