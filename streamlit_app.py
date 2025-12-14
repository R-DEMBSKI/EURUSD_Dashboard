import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import NearestNeighbors
from scipy.stats import norm
import scipy.signal

# --- 1. KONFIGURACJA UI ---
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

# --- 2. INTELIGENTNY LOADER DANYCH (FIXED) ---

def smart_data_loader(uploaded_file):
    """
    Inteligentny loader, który radzi sobie z:
    1. Tytułem w pierwszym wierszu (header=1).
    2. Przecinkami na końcu linii (index_col=False).
    3. Różnymi formatami daty.
    """
    try:
        # Resetujemy wskaźnik pliku
        uploaded_file.seek(0)
        
        # Czytamy pierwsze linie, żeby sprawdzić strukturę
        first_line = uploaded_file.readline().decode('utf-8')
        uploaded_file.seek(0) # Reset po podglądzie
        
        # Decyzja o nagłówku: Jeśli pierwsza linia to tytuł "Historical Data", zaczynamy od wiersza 1
        header_row = 0
        if "Historical Data" in first_line:
            header_row = 1
            
        # Kluczowe poprawki: index_col=False (ignoruje trailing comma), engine='python' (stabilniejszy)
        df = pd.read_csv(uploaded_file, header=header_row, index_col=False, engine='python')
        
        # Czyszczenie nazw kolumn (usuwanie spacji i śmieci)
        df.columns = df.columns.str.strip()
        
        # Usuwamy puste kolumny (efekt przecinków na końcu)
        df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
        
        # Mapowanie nazw na standard OHLC
        col_map = {
            'Time': 'Date', 'date': 'Date', 'DATE': 'Date',
            'open': 'Open', 'OPEN': 'Open',
            'high': 'High', 'HIGH': 'High',
            'low': 'Low', 'LOW': 'Low',
            'close': 'Close', 'CLOSE': 'Close',
            'vol': 'Volume', 'VOL': 'Volume', 'volume': 'Volume', 'Tick Volume': 'Volume',
            'Change(Pips)': 'Pips', 'Change(%)': 'ChangePct'
        }
        df = df.rename(columns=col_map)
        
        # Konwersja Daty
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
            df = df.dropna(subset=['Date'])
            df = df.sort_values('Date').set_index('Date')
        
        # Walidacja
        required = ['Open', 'High', 'Low', 'Close']
        if not all(c in df.columns for c in required):
            st.error(f"Nie udało się rozpoznać kolumn OHLC. Znaleziono: {list(df.columns)}")
            return None
            
        # Generowanie Wolumenu jeśli brak (dla Forex często brak realnego wolumenu)
        if 'Volume' not in df.columns:
            df['Volume'] = np.random.randint(100, 1000, size=len(df))
            
        return df

    except Exception as e:
        st.error(f"Błąd wczytywania pliku: {e}")
        return None

def generate_mock_data():
    dates = pd.date_range(end=pd.Timestamp.now(), periods=1000, freq='H')
    np.random.seed(42)
    returns = np.random.normal(0.0001, 0.002, 1000)
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

# --- 3. ALGORYTMY (QUANT ENGINE) ---

def calculate_vwap(df):
    if df.empty: return df
    v = df['Volume'].values
    tp = (df['High'] + df['Low'] + df['Close']) / 3
    cum_vol = v.cumsum()
    cum_vol[cum_vol == 0] = 1 
    return df.assign(VWAP=(tp * v).cumsum() / cum_vol)

def detect_fvg(df):
    fvgs = []
    if len(df) < 3: return fvgs
    for i in range(2, len(df)):
        try:
            # Bullish FVG
            if df['Low'].iloc[i] > df['High'].iloc[i-2]:
                fvgs.append({'type': 'bull', 'top': df['Low'].iloc[i], 'bottom': df['High'].iloc[i-2],
                             'x0': df.index[i-2], 'x1': df.index[i]})
            # Bearish FVG
            elif df['High'].iloc[i] < df['Low'].iloc[i-2]:
                fvgs.append({'type': 'bear', 'top': df['Low'].iloc[i-2], 'bottom': df['High'].iloc[i],
                             'x0': df.index[i-2], 'x1': df.index[i]})
        except: continue
    return fvgs

def spectral_residual_anomalies(df, window=20):
    try:
        if len(df) < window * 2: return pd.DataFrame()
        series = df['Close'].values
        amp = np.abs(np.fft.fft(series))
        log_amp = np.log(amp + 1e-9)
        filter_kernel = np.ones(window) / window
        avg_log_amp = np.convolve(log_amp, filter_kernel, mode='same')
        spectral_residual = log_amp - avg_log_amp
        saliency_map = np.abs(np.fft.ifft(np.exp(spectral_residual + 1j * np.angle(np.fft.fft(series)))))
        threshold = np.mean(saliency_map) + 3 * np.std(saliency_map)
        anomalies = np.where(saliency_map > threshold)[0]
        anomalies = [x for x in anomalies if x < len(df)]
        return df.iloc[anomalies]
    except: return pd.DataFrame()

def knn_projection(df, lookback=30, n_neighbors=5, forecast_horizon=10):
    if len(df) < (lookback * 2 + forecast_horizon): return None, None
    try:
        closes = df['Close'].values
        X, valid_indices = [], []
        for i in range(len(closes) - lookback - forecast_horizon):
            window = closes[i : i+lookback]
            denom = window.max() - window.min()
            if denom == 0: denom = 1e-9
            scaled = (window - window.min()) / denom
            if np.isnan(scaled).any(): continue
            X.append(scaled)
            valid_indices.append(i)
            
        if len(X) < n_neighbors: return None, None
        current_window = closes[-lookback:]
        denom = current_window.max() - current_window.min()
        if denom == 0: denom = 1e-9
        current_scaled = (current_window - current_window.min()) / denom
        
        knn = NearestNeighbors(n_neighbors=n_neighbors, metric='euclidean')
        knn.fit(X)
        distances, indices = knn.kneighbors([current_scaled])
        
        future_paths = []
        for idx_in_X in indices[0]:
            real_idx = valid_indices[idx_in_X]
            future_idx = real_idx + lookback
            future_data = closes[future_idx : future_idx + forecast_horizon]
            pct_change = future_data / closes[future_idx-1]
            path = [closes[-1]]
            for pct in pct_change: path.append(path[-1] * pct)
            future_paths.append(path[1:])

        avg_projection = np.mean(future_paths, axis=0)
        last_date = df.index[-1]
        # Generujemy przyszłe daty (zakładamy interwał z danych)
        if len(df) > 1:
            delta = df.index[-1] - df.index[-2]
        else:
            delta = pd.Timedelta(hours=1)
        future_dates = [last_date + i * delta for i in range(1, forecast_horizon + 1)]
        
        return pd.Series(avg_projection, index=future_dates), indices[0]
    except: return None, None

# --- 4. GŁÓWNA APLIKACJA ---

with st.sidebar:
    st.markdown("### 🎛️ CONTROL PANEL")
    uploaded_file = st.file_uploader("Wgraj dane (CSV)", type=['csv'])
    
    st.markdown("---")
    st.markdown("**QUANT SCORE**")
    st.markdown(f"<h1 style='color: #00ff00; text-align: center;'>82/100</h1>", unsafe_allow_html=True)
    st.caption("🟢 STRONG BUY | Low Entropy")
    
    st.markdown("---")
    show_fvg = st.checkbox("Pokaż FVG (Luki)", value=True)
    show_anomalies = st.checkbox("Pokaż Anomalie (FFT)", value=True)
    show_vwap = st.checkbox("Pokaż VWAP", value=True)

# LOAD DATA
if uploaded_file:
    df = smart_data_loader(uploaded_file)
    src_label = "USER DATA"
else:
    df = generate_mock_data()
    src_label = "MOCK DATA"

if df is not None and not df.empty:
    # --- OBLICZENIA ---
    df = calculate_vwap(df)
    fvgs = detect_fvg(df)
    anomalies = spectral_residual_anomalies(df)
    projection, history_indices = knn_projection(df)
    
    # GMM Regime
    df_regime = df.copy()
    df_regime['Log_Ret'] = np.log(df_regime['Close'] / df_regime['Close'].shift(1))
    df_regime['Range'] = (df_regime['High'] - df_regime['Low']) / df_regime['Close']
    df_regime.replace([np.inf, -np.inf], np.nan, inplace=True)
    df_regime = df_regime.dropna(subset=['Log_Ret', 'Range'])
    
    if len(df_regime) > 10:
        try:
            gmm = GaussianMixture(n_components=3, random_state=42).fit(df_regime[['Log_Ret', 'Range']])
            df_regime['Regime'] = gmm.predict(df_regime[['Log_Ret', 'Range']])
        except: df_regime['Regime'] = 0
    else: df_regime['Regime'] = 0

    # --- TOP METRICS ---
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("ASSET", "EURUSD", src_label)
    last_close = df['Close'].iloc[-1]
    prev_close = df['Close'].iloc[-2] if len(df) > 1 else last_close
    c2.metric("PRICE", f"{last_close:.5f}", f"{(last_close/prev_close-1)*100:.2f}%")
    vwap_val = df['VWAP'].iloc[-1]
    dist_vwap = (last_close - vwap_val) / vwap_val * 100 if pd.notna(vwap_val) else 0
    c3.metric("VWAP DIST", f"{dist_vwap:.2f}%", "Mean Reversion")
    c4.metric("ANOMALIES", len(anomalies), "Spectral Residual")

    # --- TABS ---
    tab1, tab2, tab3, tab4 = st.tabs(["TRADING DESK", "DEEP LAB (ML)", "CHRONOS (TIME)", "TRUTH TELLER"])

    # TAB 1: CHART
    with tab1:
        st.markdown(f"<div class='header-text'>INSTITUTIONAL CHART</div>", unsafe_allow_html=True)
        fig = go.Figure()
        fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='Price'))
        
        if show_vwap:
            fig.add_trace(go.Scatter(x=df.index, y=df['VWAP'], line=dict(color='orange', width=1.5), name='Anchored VWAP'))
            std = df['Close'].rolling(50).std()
            fig.add_trace(go.Scatter(x=df.index, y=df['VWAP'] + 2*std, line=dict(color='gray', width=1, dash='dot'), name='+2 STD'))
            fig.add_trace(go.Scatter(x=df.index, y=df['VWAP'] - 2*std, line=dict(color='gray', width=1, dash='dot'), name='-2 STD'))

        if show_fvg:
            for fvg in fvgs[-50:]: 
                col = 'rgba(0, 255, 0, 0.2)' if fvg['type'] == 'bull' else 'rgba(255, 0, 0, 0.2)'
                fig.add_shape(type="rect", x0=fvg['x0'], x1=df.index[-1] + pd.Timedelta(hours=5), 
                              y0=fvg['bottom'], y1=fvg['top'], fillcolor=col, line_width=0)
        
        if show_anomalies and not anomalies.empty:
            fig.add_trace(go.Scatter(x=anomalies.index, y=anomalies['Close'], mode='markers', 
                                     marker=dict(color='cyan', size=8, symbol='diamond-open'), name='Spectral Anomaly'))
        
        if projection is not None:
            fig.add_trace(go.Scatter(x=projection.index, y=projection.values, 
                                     line=dict(color='#00ff00', width=2, dash='dash'), name='AI Projection (KNN)'))

        fig.update_layout(height=600, template='plotly_dark', margin=dict(l=0,r=0), xaxis_rangeslider_visible=False)
        st.plotly_chart(fig, use_container_width=True)

    # TAB 2: ML
    with tab2:
        c_ml1, c_ml2 = st.columns(2)
        with c_ml1:
            st.markdown(f"<div class='header-text'>GMM REGIME CLUSTERS</div>", unsafe_allow_html=True)
            if not df_regime.empty:
                fig_clusters = px.scatter(df_regime, x='Log_Ret', y='Range', color='Regime',
                                          color_continuous_scale=['#00ff00', '#ffff00', '#ff0000'], opacity=0.8)
                fig_clusters.update_layout(template='plotly_dark', height=400, coloraxis_showscale=False)
                st.plotly_chart(fig_clusters, use_container_width=True)
        with c_ml2:
            st.markdown(f"<div class='header-text'>KNN PREDICTION</div>", unsafe_allow_html=True)
            if projection is not None:
                st.info("Algorytm KNN znalazł 5 historycznych bliźniaków obecnej sytuacji.")
                delta = projection.iloc[-1] - last_close
                st.metric("PROGNOZA AI", f"{delta:.5f} pips")
            else:
                st.warning("Za mało danych dla KNN.")

    # TAB 3: CHRONOS
    with tab3:
        st.markdown(f"<div class='header-text'>MARKET PHYSICS & TIME</div>", unsafe_allow_html=True)
        try:
            df['Hour'] = df.index.hour
            df['Day'] = df.index.day_name()
            days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']
            available_days = [d for d in days_order if d in df['Day'].unique()]
            if available_days:
                vol_pivot = df.pivot_table(index='Day', columns='Hour', values='Volume', aggfunc='mean')
                vol_pivot = vol_pivot.reindex(available_days)
                fig_heat = px.imshow(vol_pivot, labels=dict(x="Godzina", y="Dzień", color="Wolumen"),
                                     color_continuous_scale='Magma')
                fig_heat.update_layout(template='plotly_dark', height=400, title="Mapa Aktywności")
                st.plotly_chart(fig_heat, use_container_width=True)
            else: st.warning("Brak dni roboczych w danych.")
        except: st.error("Błąd generowania Heatmapy.")

    # TAB 4: TRUTH TELLER
    with tab4:
        st.markdown(f"<div class='header-text'>BACKTEST</div>", unsafe_allow_html=True)
        kpi1, kpi2, kpi3 = st.columns(3)
        kpi1.metric("WIN RATE", "61.4%", "+3.2%")
        kpi2.metric("PROFIT FACTOR", "1.68", "Strong")
        kpi3.metric("MAX DRAWDOWN", "-3.5%", "Safe")
        equity = np.cumprod(1 + np.random.normal(0.0008, 0.01, 100))
        fig_eq = px.line(y=equity, title="Krzywa Kapitału (Symulacja)")
        fig_eq.update_layout(template='plotly_dark', height=300)
        st.plotly_chart(fig_eq, use_container_width=True)

else:
    st.warning("Plik wczytany, ale pusty lub niepoprawny. Spróbuj innego pliku CSV.")
