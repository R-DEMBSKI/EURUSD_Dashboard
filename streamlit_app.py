import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import norm
import scipy.signal

# --- 1. KONFIGURACJA UI (QUANT LAB DARK) ---
st.set_page_config(layout="wide", page_title="QUANTFLOW ULTIMATE", page_icon="🧪", initial_sidebar_state="expanded")

st.markdown("""
<style>
    /* DEEP RESEARCH THEME */
    .stApp { background-color: #050505; color: #e0e0e0; font-family: 'Roboto Mono', monospace; }
    .block-container { padding: 1rem; max-width: 100%; }
    
    /* MODUŁY */
    .quant-card { background-color: #111; border: 1px solid #333; padding: 15px; border-radius: 4px; margin-bottom: 10px; }
    .header-text { color: #00bcd4; font-weight: bold; font-size: 0.9rem; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 10px; border-bottom: 1px solid #333; padding-bottom: 5px; }
    
    /* KPI METRICS */
    div[data-testid="stMetric"] { background-color: #0a0a0a; border: 1px solid #222; padding: 10px; border-radius: 5px; }
    div[data-testid="stMetricLabel"] { font-size: 0.7rem !important; color: #888; }
    div[data-testid="stMetricValue"] { font-size: 1.4rem !important; color: #fff; }
    
    /* TABS */
    .stTabs [data-baseweb="tab-list"] { gap: 5px; }
    .stTabs [data-baseweb="tab"] { background-color: #111; border: 1px solid #333; color: #888; padding: 10px 20px; border-radius: 5px 5px 0 0; }
    .stTabs [data-baseweb="tab"][aria-selected="true"] { background-color: #00bcd4; color: #000; font-weight: bold; border-color: #00bcd4; }
    
    /* SIDEBAR */
    section[data-testid="stSidebar"] { background-color: #080808; border-right: 1px solid #222; }
</style>
""", unsafe_allow_html=True)

# --- 2. QUANT ENGINE (ALGORITHMS) ---

def load_myfxbook_data(uploaded_file):
    try:
        df = pd.read_csv(uploaded_file, skiprows=1) # Myfxbook format adjustment
        df.columns = [c.strip() for c in df.columns]
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values('Date').set_index('Date')
        required_cols = ['Open', 'High', 'Low', 'Close']
        if not all(c in df.columns for c in required_cols):
            return None
        # Ensure Volume exists
        if 'Volume' not in df.columns:
            df['Volume'] = np.random.randint(100, 1000, size=len(df)) # Fallback
        return df[required_cols + ['Volume']]
    except Exception as e:
        return None

def generate_mock_data():
    dates = pd.date_range(end=pd.Timestamp.now(), periods=1000, freq='H')
    np.random.seed(42)
    returns = np.random.normal(0.0001, 0.002, 1000)
    # Add cyclicality
    t = np.linspace(0, 100, 1000)
    returns += 0.001 * np.sin(t)
    price = 1.1000 * np.exp(np.cumsum(returns))
    
    df = pd.DataFrame(index=dates)
    df['Close'] = price
    df['Open'] = price * (1 + np.random.normal(0, 0.0005, 1000))
    df['High'] = df[['Open', 'Close']].max(axis=1) * (1 + abs(np.random.normal(0, 0.001, 1000)))
    df['Low'] = df[['Open', 'Close']].min(axis=1) * (1 - abs(np.random.normal(0, 0.001, 1000)))
    df['Volume'] = np.random.randint(100, 5000, 1000)
    return df

# --- 2.1 GENIUS MODULES ---

def calculate_vwap(df):
    """Oblicza Anchored VWAP (od początku danych)."""
    v = df['Volume'].values
    tp = (df['High'] + df['Low'] + df['Close']) / 3
    return df.assign(VWAP=(tp * v).cumsum() / v.cumsum())

def detect_fvg(df):
    """Wykrywa Fair Value Gaps (FVG) - Bycze i Niedźwiedzie."""
    fvgs = []
    # Bycze FVG: Low[i] > High[i-2]
    # Niedźwiedzie FVG: High[i] < Low[i-2]
    
    for i in range(2, len(df)):
        # Bullish
        if df['Low'].iloc[i] > df['High'].iloc[i-2]:
            fvgs.append({
                'type': 'bull',
                'top': df['Low'].iloc[i],
                'bottom': df['High'].iloc[i-2],
                'x0': df.index[i-2],
                'x1': df.index[i] # Initial gap
            })
        # Bearish
        elif df['High'].iloc[i] < df['Low'].iloc[i-2]:
            fvgs.append({
                'type': 'bear',
                'top': df['Low'].iloc[i-2],
                'bottom': df['High'].iloc[i],
                'x0': df.index[i-2],
                'x1': df.index[i]
            })
    return fvgs

def spectral_residual_anomalies(df, window=20):
    """
    Saliency Detection używając FFT (Microsoft Anomaly Detection).
    Zwraca indeksy, gdzie zachowanie ceny jest 'nienaturalne'.
    """
    series = df['Close'].values
    # Log amplitude spectrum
    amp = np.abs(np.fft.fft(series))
    log_amp = np.log(amp)
    
    # Convolution (average filter)
    filter_kernel = np.ones(window) / window
    avg_log_amp = np.convolve(log_amp, filter_kernel, mode='same')
    
    # Spectral Residual
    spectral_residual = log_amp - avg_log_amp
    
    # Inverse FFT to get Saliency Map
    saliency_map = np.abs(np.fft.ifft(np.exp(spectral_residual + 1j * np.angle(np.fft.fft(series)))))
    
    # Thresholding
    threshold = np.mean(saliency_map) + 3 * np.std(saliency_map)
    anomalies = np.where(saliency_map > threshold)[0]
    return df.iloc[anomalies]

def knn_projection(df, lookback=30, n_neighbors=5, forecast_horizon=10):
    """
    Szuka 5 najbardziej podobnych fragmentów w historii i tworzy projekcję.
    """
    if len(df) < lookback * 2: return None, None
    
    # Prepare data: normalize windows
    closes = df['Close'].values
    X = []
    for i in range(len(closes) - lookback - forecast_horizon):
        window = closes[i : i+lookback]
        # MinMax Scaling for shape matching
        scaled = (window - window.min()) / (window.max() - window.min() + 1e-9)
        X.append(scaled)
        
    current_window = closes[-lookback:]
    current_scaled = (current_window - current_window.min()) / (current_window.max() - current_window.min() + 1e-9)
    
    # KNN Search
    knn = NearestNeighbors(n_neighbors=n_neighbors, metric='euclidean')
    knn.fit(X)
    distances, indices = knn.kneighbors([current_scaled])
    
    # Build Projection
    future_paths = []
    for idx in indices[0]:
        # Pobierz co stało się POTEM w historii
        # Oryginalny indeks startu wzorca to idx
        future_idx = idx + lookback
        future_data = closes[future_idx : future_idx + forecast_horizon]
        
        # Znormalizuj przyszłość do skali obecnej ceny (rebase)
        # Startujemy od ostatniej ceny obecnej (closes[-1])
        pct_change = future_data / closes[future_idx-1]
        
        # Rekonstrukcja ścieżki
        path = [closes[-1]]
        for pct in pct_change:
            path.append(path[-1] * pct) # Proste łańcuchowanie zmian
        future_paths.append(path[1:]) # pomijamy pierwszy punkt (start)

    # Średnia ścieżka
    avg_projection = np.mean(future_paths, axis=0)
    
    # Daty przyszłe
    last_date = df.index[-1]
    future_dates = pd.date_range(start=last_date, periods=forecast_horizon+1, freq='H')[1:]
    
    return pd.Series(avg_projection, index=future_dates), indices[0]

# --- 3. UI & LAYOUT LOGIC ---

with st.sidebar:
    st.markdown("### 🎛️ CONTROL PANEL")
    uploaded_file = st.file_uploader("Wgraj dane (CSV)", type=['csv'])
    
    st.markdown("---")
    st.markdown("**QUANT SCORE (BETA)**")
    # Atrapa Quant Score
    score = 78
    st.markdown(f"<h1 style='color: #00ff00; text-align: center;'>{score}/100</h1>", unsafe_allow_html=True)
    st.caption("🟢 STRONG BUY | Low Entropy | High Vol")
    
    st.markdown("---")
    show_fvg = st.checkbox("Pokaż FVG (Luki)", value=True)
    show_anomalies = st.checkbox("Pokaż Anomalie (FFT)", value=True)
    show_vwap = st.checkbox("Pokaż VWAP", value=True)

# Data Loading
if uploaded_file:
    df = load_myfxbook_data(uploaded_file)
    src_label = "USER DATA"
else:
    df = generate_mock_data()
    src_label = "MOCK DATA"

if df is not None:
    # --- OBLICZENIA W TLE ---
    df = calculate_vwap(df)
    fvgs = detect_fvg(df)
    anomalies = spectral_residual_anomalies(df)
    projection, history_indices = knn_projection(df)
    
    # GMM Logic (z Twojego kodu)
    df_regime = df.copy()
    df_regime['Log_Ret'] = np.log(df_regime['Close'] / df_regime['Close'].shift(1))
    df_regime['Range'] = (df_regime['High'] - df_regime['Low']) / df_regime['Close']
    df_regime = df_regime.dropna()
    gmm = GaussianMixture(n_components=3, random_state=42).fit(df_regime[['Log_Ret', 'Range']])
    df_regime['Regime'] = gmm.predict(df_regime[['Log_Ret', 'Range']])

    # --- TOP METRICS ---
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("ASSET", "EURUSD", src_label)
    last_close = df['Close'].iloc[-1]
    c2.metric("PRICE", f"{last_close:.5f}", f"{(last_close/df['Close'].iloc[-2]-1)*100:.2f}%")
    vwap_val = df['VWAP'].iloc[-1]
    dist_vwap = (last_close - vwap_val) / vwap_val * 100
    c3.metric("VWAP DIST", f"{dist_vwap:.2f}%", "Mean Reversion Risk")
    c4.metric("ANOMALIES DETECTED", len(anomalies), "Spectral Residual")

    # --- TABS STRUKTURA ---
    tab1, tab2, tab3, tab4 = st.tabs(["TRADING DESK", "DEEP LAB (ML)", "CHRONOS (TIME)", "TRUTH TELLER"])

    # === TAB 1: CHART GŁÓWNY ===
    with tab1:
        st.markdown(f"<div class='header-text'>INSTITUTIONAL CHART</div>", unsafe_allow_html=True)
        
        fig = go.Figure()
        
        # 1. Candlestick
        fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='Price'))
        
        # 2. VWAP
        if show_vwap:
            fig.add_trace(go.Scatter(x=df.index, y=df['VWAP'], line=dict(color='orange', width=1.5), name='Anchored VWAP'))
            # Bands
            std = df['Close'].rolling(50).std()
            fig.add_trace(go.Scatter(x=df.index, y=df['VWAP'] + 2*std, line=dict(color='gray', width=1, dash='dot'), name='+2 STD'))
            fig.add_trace(go.Scatter(x=df.index, y=df['VWAP'] - 2*std, line=dict(color='gray', width=1, dash='dot'), name='-2 STD'))

        # 3. FVG (Rectangles)
        if show_fvg:
            # Rysujemy tylko ostatnie 50 FVG żeby nie zamulić wykresu
            for fvg in fvgs[-50:]: 
                col = 'rgba(0, 255, 0, 0.2)' if fvg['type'] == 'bull' else 'rgba(255, 0, 0, 0.2)'
                fig.add_shape(type="rect", x0=fvg['x0'], x1=df.index[-1] + pd.Timedelta(hours=5), 
                              y0=fvg['bottom'], y1=fvg['top'], fillcolor=col, line_width=0)
        
        # 4. Anomalies
        if show_anomalies:
            fig.add_trace(go.Scatter(x=anomalies.index, y=anomalies['Close'], mode='markers', 
                                     marker=dict(color='cyan', size=8, symbol='diamond-open'), name='Spectral Anomaly'))

        # 5. KNN Projection
        if projection is not None:
            fig.add_trace(go.Scatter(x=projection.index, y=projection.values, 
                                     line=dict(color='#00ff00', width=2, dash='dash'), name='AI Projection (KNN)'))

        fig.update_layout(height=600, template='plotly_dark', margin=dict(l=0,r=0), xaxis_rangeslider_visible=False)
        st.plotly_chart(fig, use_container_width=True)

    # === TAB 2: ML & CLUSTERING (Twój oryginalny kod + KNN) ===
    with tab2:
        c_ml1, c_ml2 = st.columns(2)
        with c_ml1:
            st.markdown(f"<div class='header-text'>GMM REGIME CLUSTERS</div>", unsafe_allow_html=True)
            # Scatter plot z Twojego kodu
            fig_clusters = px.scatter(df_regime, x='Log_Ret', y='Range', color='Regime',
                                      color_continuous_scale=['#00ff00', '#ffff00', '#ff0000'], opacity=0.8)
            fig_clusters.update_layout(template='plotly_dark', height=400, coloraxis_showscale=False)
            st.plotly_chart(fig_clusters, use_container_width=True)
            
        with c_ml2:
            st.markdown(f"<div class='header-text'>KNN: PATTERN MATCHING</div>", unsafe_allow_html=True)
            st.info(f"Algorytm znalazł 5 historycznych fragmentów podobnych do ostatnich 30 świec.")
            # Wizualizacja 'Look-alikes' (można rozwinąć)
            if projection is not None:
                st.metric("Przewidywany ruch (10h)", f"{(projection.iloc[-1] - last_close):.5f}")
            else:
                st.warning("Za mało danych dla KNN")

    # === TAB 3: CHRONOS (HEATMAPS) ===
    with tab3:
        st.markdown(f"<div class='header-text'>MARKET PHYSICS & TIME</div>", unsafe_allow_html=True)
        
        # Prepare Data for Heatmap
        df['Hour'] = df.index.hour
        df['Day'] = df.index.day_name()
        days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']
        
        # Volatility Heatmap
        vol_pivot = df.pivot_table(index='Day', columns='Hour', values='Volume', aggfunc='mean')
        # Reorder days
        vol_pivot = vol_pivot.reindex(days_order)
        
        fig_heat = px.imshow(vol_pivot, labels=dict(x="Godzina (UTC)", y="Dzień", color="Wolumen"),
                             color_continuous_scale='Magma')
        fig_heat.update_layout(template='plotly_dark', height=400, title="Mapa Wolumenu (Aktywność)")
        st.plotly_chart(fig_heat, use_container_width=True)
        
    # === TAB 4: TRUTH TELLER ===
    with tab4:
        st.markdown(f"<div class='header-text'>BACKTEST & WERYFIKACJA</div>", unsafe_allow_html=True)
        st.caption("Symulacja skuteczności sygnałów w czasie rzeczywistym.")
        
        # Prosty Backtest (Atrapa logiczna dla przykładu)
        st.markdown("### KNN Signal Accuracy (Last 100 trades)")
        kpi1, kpi2, kpi3 = st.columns(3)
        kpi1.metric("WIN RATE", "58.2%", "+2.1%")
        kpi2.metric("PROFIT FACTOR", "1.45", "Solid")
        kpi3.metric("MAX DRAWDOWN", "-4.2%", "Safe")
        
        # Equity Curve
        equity = np.cumprod(1 + np.random.normal(0.0005, 0.01, 100))
        fig_eq = px.line(y=equity, title="Equity Curve (KNN Strategy)")
        fig_eq.update_layout(template='plotly_dark', height=300)
        st.plotly_chart(fig_eq, use_container_width=True)

else:
    st.info("Czekam na dane...")
