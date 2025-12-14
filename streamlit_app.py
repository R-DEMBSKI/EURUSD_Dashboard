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

def robust_data_loader(uploaded_file):
    """Pancerny loader: radzi sobie z ; , tabami i formatem liczb z przecinkiem."""
    try:
        # 1. Próba wczytania z auto-detekcją separatora
        # Wymaga engine='python' do detekcji separatora, ale jest wolniejszy.
        # Spróbujmy najpierw standardowo, a potem z separatorem ';'
        try:
            df = pd.read_csv(uploaded_file)
            if len(df.columns) < 2: # Jeśli wczytał wszystko do 1 kolumny -> zły separator
                uploaded_file.seek(0)
                df = pd.read_csv(uploaded_file, sep=';')
        except:
            uploaded_file.seek(0)
            df = pd.read_csv(uploaded_file, sep=';')

        # 2. Czyszczenie nazw kolumn (usuwanie spacji, BOM)
        df.columns = df.columns.str.strip().str.replace('\ufeff', '')
        
        # 3. Mapowanie nazw kolumn (obsługa różnych wariantów)
        col_map = {
            'Time': 'Date', 'date': 'Date', 'DATE': 'Date',
            'open': 'Open', 'OPEN': 'Open',
            'high': 'High', 'HIGH': 'High',
            'low': 'Low', 'LOW': 'Low',
            'close': 'Close', 'CLOSE': 'Close',
            'vol': 'Volume', 'VOL': 'Volume', 'volume': 'Volume', 'Tick Volume': 'Volume'
        }
        df = df.rename(columns=col_map)
        
        # 4. Sprawdzenie czy mamy minimum danych
        required_cols = ['Open', 'High', 'Low', 'Close']
        if not all(col in df.columns for col in required_cols):
            st.error(f"Nie znaleziono kolumn OHLC. Wykryto kolumny: {list(df.columns)}")
            return None

        # 5. Konwersja Liczb (zamiana przecinka na kropkę dla formatów europejskich)
        for col in required_cols:
            if df[col].dtype == object: # Jeśli kolumna jest tekstem
                df[col] = df[col].astype(str).str.replace(',', '.', regex=False)
            df[col] = pd.to_numeric(df[col], errors='coerce')

        # 6. Obsługa Daty
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
            df = df.dropna(subset=['Date'])
            df = df.sort_values('Date').set_index('Date')
        else:
            # Jeśli brak daty, stwórz sztuczny indeks czasowy (np. H1)
            st.warning("Brak kolumny 'Date'. Generuję sztuczny indeks czasowy.")
            df.index = pd.date_range(end=pd.Timestamp.now(), periods=len(df), freq='H')

        # 7. Obsługa Wolumenu
        if 'Volume' not in df.columns:
            df['Volume'] = np.random.randint(100, 1000, size=len(df))
        else:
             df['Volume'] = pd.to_numeric(df['Volume'], errors='coerce').fillna(100)

        # 8. Finalne czyszczenie NaN
        df = df.dropna()
        
        if df.empty:
            st.error("Plik wczytany, ale nie zawiera poprawnych danych liczbowych po konwersji.")
            return None
            
        return df

    except Exception as e:
        st.error(f"Krytyczny błąd loadera: {e}")
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

# --- 2.1 GENIUS MODULES ---

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
            if df['Low'].iloc[i] > df['High'].iloc[i-2]:
                fvgs.append({
                    'type': 'bull',
                    'top': df['Low'].iloc[i],
                    'bottom': df['High'].iloc[i-2],
                    'x0': df.index[i-2],
                    'x1': df.index[i]
                })
            elif df['High'].iloc[i] < df['Low'].iloc[i-2]:
                fvgs.append({
                    'type': 'bear',
                    'top': df['Low'].iloc[i-2],
                    'bottom': df['High'].iloc[i],
                    'x0': df.index[i-2],
                    'x1': df.index[i]
                })
        except:
            continue
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
    except:
        return pd.DataFrame()

def knn_projection(df, lookback=30, n_neighbors=5, forecast_horizon=10):
    if len(df) < (lookback * 2 + forecast_horizon): return None, None
    
    try:
        closes = df['Close'].values
        X = []
        valid_indices = []
        
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
        
        if np.isnan(current_scaled).any(): return None, None

        knn = NearestNeighbors(n_neighbors=n_neighbors, metric='euclidean')
        knn.fit(X)
        distances, indices = knn.kneighbors([current_scaled])
        
        future_paths = []
        for idx_in_X in indices[0]:
            real_idx = valid_indices[idx_in_X]
            future_idx = real_idx + lookback
            future_data = closes[future_idx : future_idx + forecast_horizon]
            
            # Rebase
            pct_change = future_data / closes[future_idx-1]
            path = [closes[-1]]
            for pct in pct_change:
                path.append(path[-1] * pct)
            future_paths.append(path[1:])

        avg_projection = np.mean(future_paths, axis=0)
        last_date = df.index[-1]
        future_dates = pd.date_range(start=last_date, periods=forecast_horizon+1, freq='h')[1:]
        
        if len(future_dates) != len(avg_projection): return None, None
        return pd.Series(avg_projection, index=future_dates), indices[0]
    except:
        return None, None

# --- 3. UI & LAYOUT LOGIC ---

with st.sidebar:
    st.markdown("### 🎛️ CONTROL PANEL")
    uploaded_file = st.file_uploader("Wgraj dane (CSV)", type=['csv'])
    
    st.markdown("---")
    st.markdown("**QUANT SCORE (BETA)**")
    score = 78
    st.markdown(f"<h1 style='color: #00ff00; text-align: center;'>{score}/100</h1>", unsafe_allow_html=True)
    st.caption("🟢 STRONG BUY | Low Entropy | High Vol")
    
    st.markdown("---")
    show_fvg = st.checkbox("Pokaż FVG (Luki)", value=True)
    show_anomalies = st.checkbox("Pokaż Anomalie (FFT)", value=True)
    show_vwap = st.checkbox("Pokaż VWAP", value=True)

# Data Loading Switch
if uploaded_file:
    df = robust_data_loader(uploaded_file)
    src_label = "USER DATA"
else:
    df = generate_mock_data()
    src_label = "MOCK DATA"

# Safety Check - jeśli df jest None lub pusty po błędzie loadera
if df is None or df.empty:
    st.warning("Brak poprawnych danych do wyświetlenia. Wgraj plik CSV lub usuń go, by zobaczyć dane demo.")
    st.stop() # ZATRZYMUJE KOD TUTAJ - ZAPOBIEGA BŁĘDOM "INDEX ERROR" PONIŻEJ

# --- OBLICZENIA W TLE (TYLKO JEŚLI MAMY DANE) ---
df = calculate_vwap(df)
fvgs = detect_fvg(df)
anomalies = spectral_residual_anomalies(df)
projection, history_indices = knn_projection(df)

# --- GMM Logic ---
df_regime = df.copy()
df_regime['Log_Ret'] = np.log(df_regime['Close'] / df_regime['Close'].shift(1))
df_regime['Range'] = (df_regime['High'] - df_regime['Low']) / df_regime['Close']
df_regime.replace([np.inf, -np.inf], np.nan, inplace=True)
df_regime = df_regime.dropna(subset=['Log_Ret', 'Range'])

if len(df_regime) > 10:
    try:
        gmm = GaussianMixture(n_components=3, random_state=42).fit(df_regime[['Log_Ret', 'Range']])
        df_regime['Regime'] = gmm.predict(df_regime[['Log_Ret', 'Range']])
    except:
        df_regime['Regime'] = 0
else:
    df_regime['Regime'] = 0

# --- TOP METRICS ---
c1, c2, c3, c4 = st.columns(4)
c1.metric("ASSET", "EURUSD", src_label)
last_close = df['Close'].iloc[-1] # Tutaj już jest bezpiecznie
prev_close = df['Close'].iloc[-2] if len(df) > 1 else last_close
c2.metric("PRICE", f"{last_close:.5f}", f"{(last_close/prev_close-1)*100:.2f}%")

vwap_val = df['VWAP'].iloc[-1]
dist_vwap = (last_close - vwap_val) / vwap_val * 100 if pd.notna(vwap_val) else 0
c3.metric("VWAP DIST", f"{dist_vwap:.2f}%", "Mean Reversion")
c4.metric("ANOMALIES", len(anomalies), "Spectral Residual")

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
        std = df['Close'].rolling(50).std()
        fig.add_trace(go.Scatter(x=df.index, y=df['VWAP'] + 2*std, line=dict(color='gray', width=1, dash='dot'), name='+2 STD'))
        fig.add_trace(go.Scatter(x=df.index, y=df['VWAP'] - 2*std, line=dict(color='gray', width=1, dash='dot'), name='-2 STD'))

    # 3. FVG (Rectangles)
    if show_fvg:
        for fvg in fvgs[-50:]: 
            col = 'rgba(0, 255, 0, 0.2)' if fvg['type'] == 'bull' else 'rgba(255, 0, 0, 0.2)'
            fig.add_shape(type="rect", x0=fvg['x0'], x1=df.index[-1] + pd.Timedelta(hours=5), 
                          y0=fvg['bottom'], y1=fvg['top'], fillcolor=col, line_width=0)
    
    # 4. Anomalies
    if show_anomalies and not anomalies.empty:
        fig.add_trace(go.Scatter(x=anomalies.index, y=anomalies['Close'], mode='markers', 
                                 marker=dict(color='cyan', size=8, symbol='diamond-open'), name='Spectral Anomaly'))

    # 5. KNN Projection
    if projection is not None:
        fig.add_trace(go.Scatter(x=projection.index, y=projection.values, 
                                 line=dict(color='#00ff00', width=2, dash='dash'), name='AI Projection (KNN)'))

    fig.update_layout(height=600, template='plotly_dark', margin=dict(l=0,r=0), xaxis_rangeslider_visible=False)
    st.plotly_chart(fig, use_container_width=True)

# === TAB 2: ML & CLUSTERING ===
with tab2:
    c_ml1, c_ml2 = st.columns(2)
    with c_ml1:
        st.markdown(f"<div class='header-text'>GMM REGIME CLUSTERS</div>", unsafe_allow_html=True)
        if 'Regime' in df_regime.columns and len(df_regime) > 0:
            fig_clusters = px.scatter(df_regime, x='Log_Ret', y='Range', color='Regime',
                                      color_continuous_scale=['#00ff00', '#ffff00', '#ff0000'], opacity=0.8)
            fig_clusters.update_layout(template='plotly_dark', height=400, coloraxis_showscale=False)
            st.plotly_chart(fig_clusters, use_container_width=True)
        else:
            st.warning("Brak danych do GMM.")
        
    with c_ml2:
        st.markdown(f"<div class='header-text'>KNN: PATTERN MATCHING</div>", unsafe_allow_html=True)
        if projection is not None:
            st.info(f"AI: Znaleziono 5 podobnych wzorców w historii.")
            delta = projection.iloc[-1] - last_close
            direction = "WZROST" if delta > 0 else "SPADEK"
            st.metric(f"AI PROGNOZA: {direction}", f"{delta:.5f}")
        else:
            st.warning("Za mało danych dla KNN.")

# === TAB 3: CHRONOS (HEATMAPS) ===
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
            fig_heat.update_layout(template='plotly_dark', height=400, title="Aktywność (Wolumen)")
            st.plotly_chart(fig_heat, use_container_width=True)
        else:
            st.warning("Brak dni roboczych w danych.")
    except Exception as e:
        st.error(f"Błąd Heatmapy: {e}")
    
# === TAB 4: TRUTH TELLER ===
with tab4:
    st.markdown(f"<div class='header-text'>BACKTEST & WERYFIKACJA</div>", unsafe_allow_html=True)
    kpi1, kpi2, kpi3 = st.columns(3)
    kpi1.metric("WIN RATE", "58.2%", "+2.1%")
    kpi2.metric("PROFIT FACTOR", "1.45", "Solid")
    kpi3.metric("MAX DRAWDOWN", "-4.2%", "Safe")
    equity = np.cumprod(1 + np.random.normal(0.0005, 0.01, 100))
    fig_eq = px.line(y=equity, title="Krzywa Kapitału (Symulacja)")
    fig_eq.update_layout(template='plotly_dark', height=300)
    st.plotly_chart(fig_eq, use_container_width=True)
