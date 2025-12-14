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

def generate_ai_narrative(df, correlation_info=None):
    last = df.iloc[-1]
    z_score = last['Z_Score']
    eff = last['Efficiency_Ratio']
    
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

    # 4. Correlation Insight
    if correlation_info:
        report.append(f"INTER-MARKET LINK: Correlation is {correlation_info:.2f}. " + 
                      ("Strong Coupling." if abs(correlation_info) > 0.7 else "Decoupled/Divergent."))
        
    return "\n".join(report)

# --- 3. LOADER I HELPERY ---

def smart_data_loader(uploaded_file):
    if uploaded_file is None: return None
    try:
        uploaded_file.seek(0)
        # Szybka detekcja nagłówka dla plików typu "EURUSD Historical Data"
        first_line = uploaded_file.readline().decode('utf-8')
        uploaded_file.seek(0)
        
        # Jeśli pierwsza linia to tytuł, nagłówek jest w drugiej (index 1)
        header_row = 1 if "Historical Data" in first_line else 0
        
        # Używamy engine='python' i on_bad_lines='skip' dla bezpieczeństwa
        df = pd.read_csv(uploaded_file, header=header_row, index_col=False, engine='python')
        
        # Czyszczenie nazw kolumn
        df.columns = df.columns.str.strip()
        df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
        
        # Mapowanie nazw
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
        
        # Obsługa braku wolumenu (np. dane dzienne)
        if 'Volume' not in df.columns: df['Volume'] = 1000
        
        # Wymagane kolumny
        required = ['Open', 'High', 'Low', 'Close']
        if not all(c in df.columns for c in required):
            st.error(f"Błąd formatu pliku. Wymagane kolumny: {required}. Znaleziono: {df.columns.tolist()}")
            return None
            
        return df
    except Exception as e:
        st.error(f"Critical Error loading file: {e}")
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
    st.markdown("## ⚡ QUANT FLOW INST")
    
    # 1. Main Nav
    selected_page = option_menu(
        menu_title=None,
        options=["Dashboard", "Correlations", "Chronos", "Deep Lab"],
        icons=["speedometer2", "intersect", "clock-history", "cpu"],
        default_index=0,
        styles={
            "container": {"padding": "0!important", "background-color": "#0a0a0a"},
            "icon": {"color": "#00e5ff", "font-size": "14px"}, 
            "nav-link": {"font-size": "14px", "text-align": "left", "margin": "5px", "--hover-color": "#222"},
            "nav-link-selected": {"background-color": "#222", "border-left": "3px solid #00e5ff"},
        }
    )
    
    st.markdown("---")
    
    # 2. File Uploads
    st.markdown("### 📂 Data Feeds")
    uploaded_file = st.file_uploader("Asset A (Main)", type=['csv'], key="main_file")
    uploaded_ref_file = st.file_uploader("Asset B (Correlation)", type=['csv'], key="ref_file")
    
    st.markdown("### ⚙️ Engine Layers")
    show_fvg = st.checkbox("Show FVG", value=True)
    show_monte = st.checkbox("Monte Carlo", value=True)
    show_delta = st.checkbox("Synthetic Delta", value=True)
    
    st.markdown("---")
    st.markdown("### 🧠 AI Analyst Log")
    ai_placeholder = st.empty()

# LOAD DATA (Main Asset)
if uploaded_file:
    df = smart_data_loader(uploaded_file)
    src_label = uploaded_file.name
else:
    df = generate_mock_data()
    src_label = "SIMULATION"

if df is None or df.empty:
    st.stop()

# LOAD DATA (Reference Asset for Correlation)
df_ref = None
if uploaded_ref_file:
    df_ref = smart_data_loader(uploaded_ref_file)

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
corr_val = 0

# Obliczanie korelacji jeśli jest drugi plik
if df_ref is not None:
    # Merge on Index (Time)
    merged = pd.merge(df, df_ref, left_index=True, right_index=True, suffixes=('', '_REF'))
    if not merged.empty:
        corr_val = merged['Close'].rolling(50).corr(merged['Close_REF']).iloc[-1]
    else:
        st.warning("Brak pokrywających się dat między plikami!")

k1, k2, k3, k4 = st.columns(4)
k1.metric("ASSET A", src_label[:10], "Primary")
k2.metric("PRICE", f"{last_price:.5f}", f"{chg:+.2f}%")
k3.metric("VOLATILITY Z", f"{vol_z:.2f}", "Sigma")
k4.metric("CORRELATION", f"{corr_val:.2f}", "vs Asset B" if df_ref is not None else "No Data")

# GENERATE AI REPORT
ai_report = generate_ai_narrative(df, correlation_info=corr_val if df_ref is not None else None)
ai_placeholder.markdown(f"<div class='ai-terminal'>{ai_report.replace(chr(10), '<br>')}</div>", unsafe_allow_html=True)

# --- VISUALIZATION TABS ---

if selected_page == "Dashboard":
    # 1. MAIN CHART
    fig = go.Figure()
    
    # Candles
    fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='Price'))
    
    # VWAP
    fig.add_trace(go.Scatter(x=df.index, y=df['VWAP'], line=dict(color='orange', width=1.5), name='VWAP'))
    
    # FVG
    if show_fvg:
        for fvg in fvgs[-30:]:
            color = 'rgba(0, 255, 100, 0.15)' if fvg['type'] == 'bull' else 'rgba(255, 0, 50, 0.15)'
            fig.add_shape(type="rect", x0=fvg['x0'], x1=df.index[-1], y0=fvg['bottom'], y1=fvg['top'], 
                          fillcolor=color, line_width=0)

    # Monte Carlo Fan
    if show_monte:
        fd, mean_p, up_95, lo_05 = monte_carlo_simulation(df)
        fig.add_trace(go.Scatter(x=fd, y=up_95, mode='lines', line=dict(width=0), showlegend=False))
        fig.add_trace(go.Scatter(x=fd, y=lo_05, mode='lines', line=dict(width=0), fill='tonexty', 
                                 fillcolor='rgba(138, 43, 226, 0.2)', name='Probability Cloud (90%)'))
        fig.add_trace(go.Scatter(x=fd, y=mean_p, mode='lines', line=dict(color='#d500f9', dash='dot'), name='MC Prediction'))

    fig.update_layout(height=600, template='plotly_dark', xaxis_rangeslider_visible=False, margin=dict(l=0,r=0),
                      title="Institutional Price Action")
    st.plotly_chart(fig, use_container_width=True)
    
    # 2. SYNTHETIC DELTA
    if show_delta:
        fig_delta = go.Figure()
        colors = ['#00e5ff' if x >= 0 else '#ff1744' for x in df['Delta_Est']]
        fig_delta.add_trace(go.Bar(x=df.index, y=df['Delta_Est'], marker_color=colors, name='Delta Pressure'))
        fig_delta.update_layout(height=200, template='plotly_dark', margin=dict(t=0, b=0), title="Synthetic Delta Pressure")
        st.plotly_chart(fig_delta, use_container_width=True)

elif selected_page == "Correlations":
    st.markdown("<div class='section-header'>INTER-MARKET CORRELATION LAB</div>", unsafe_allow_html=True)
    
    if df_ref is not None:
        merged = pd.merge(df, df_ref, left_index=True, right_index=True, suffixes=('', '_REF'))
        
        # Normalize for visualization
        norm_main = (merged['Close'] - merged['Close'].min()) / (merged['Close'].max() - merged['Close'].min())
        norm_ref = (merged['Close_REF'] - merged['Close_REF'].min()) / (merged['Close_REF'].max() - merged['Close_REF'].min())
        
        fig_corr = go.Figure()
        fig_corr.add_trace(go.Scatter(x=merged.index, y=norm_main, name=f"Asset A ({src_label})", line=dict(color='#00e5ff')))
        fig_corr.add_trace(go.Scatter(x=merged.index, y=norm_ref, name="Asset B (Ref)", line=dict(color='#ffab00', dash='dot')))
        fig_corr.update_layout(height=500, template='plotly_dark', title="Normalized Price Comparison")
        st.plotly_chart(fig_corr, use_container_width=True)
        
        # Rolling Correlation Chart
        rolling_corr = merged['Close'].rolling(50).corr(merged['Close_REF'])
        fig_roll = px.area(x=merged.index, y=rolling_corr, title="50-Period Rolling Correlation Coefficient")
        fig_roll.update_layout(height=300, template='plotly_dark', yaxis_range=[-1, 1])
        st.plotly_chart(fig_roll, use_container_width=True)
    else:
        st.info("ℹ️ Wgraj drugi plik CSV w panelu bocznym (Asset B), aby analizować korelacje.")

elif selected_page == "Chronos":
    st.markdown("<div class='section-header'>TIME & VOLATILITY STATISTICS</div>", unsafe_allow_html=True)
    
    df['Day'] = df.index.day_name()
    df['Hour'] = df.index.hour
    df['Range'] = df['High'] - df['Low']
    
    heatmap_data = df.pivot_table(index='Day', columns='Hour', values='Range', aggfunc='mean')
    days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']
    heatmap_data = heatmap_data.reindex(days_order)
    
    fig_hm = px.imshow(heatmap_data, labels=dict(x="Godzina (UTC)", y="Dzień", color="Volatility (Pips)"),
                       color_continuous_scale='Magma', aspect='auto')
    fig_hm.update_layout(height=600, template='plotly_dark', title="Market Heartbeat: Kiedy handlować?")
    st.plotly_chart(fig_hm, use_container_width=True)

elif selected_page == "Deep Lab":
    st.markdown("<div class='section-header'>MONTE CARLO SIMULATION LAB</div>", unsafe_allow_html=True)
    fd, mean_p, up_95, lo_05 = monte_carlo_simulation(df, simulations=500)
    
    fig_mc = go.Figure()
    last_price = df['Close'].iloc[-1]
    last_vol = df['LogRet'].iloc[-50:].std()
    
    # 50 losowych ścieżek
    for _ in range(50):
        path = [last_price]
        for _ in range(len(fd)-1):
            path.append(path[-1] * (1 + np.random.normal(0, last_vol)))
        fig_mc.add_trace(go.Scatter(x=fd, y=path, mode='lines', line=dict(color='rgba(255,255,255,0.05)', width=1), showlegend=False))
        
    fig_mc.add_trace(go.Scatter(x=fd, y=mean_p, line=dict(color='#00e5ff', width=3), name='Expected Path'))
    fig_mc.update_layout(height=500, template='plotly_dark', title="Monte Carlo: 500 Future Paths")
    st.plotly_chart(fig_mc, use_container_width=True)
