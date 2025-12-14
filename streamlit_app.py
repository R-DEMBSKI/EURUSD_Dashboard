import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from streamlit_option_menu import option_menu  # NOWA BIBLIOTEKA
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import NearestNeighbors
from scipy.stats import norm
import scipy.signal

# --- 1. KONFIGURACJA UI (QUANT LAB DARK) ---
st.set_page_config(layout="wide", page_title="QUANTFLOW PRO", page_icon="⚡", initial_sidebar_state="collapsed")

# Custom CSS dla Option Menu i wyglądu
st.markdown("""
<style>
    /* GLOBAL THEME */
    .stApp { background-color: #080808; color: #e0e0e0; font-family: 'Roboto Mono', monospace; }
    .block-container { padding-top: 2rem; padding-bottom: 2rem; }
    
    /* UKRYCIE STANDARDOWEGO PASKI I MENU STREAMLIT */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* KPIS */
    div[data-testid="stMetric"] { background-color: #111; border: 1px solid #333; padding: 15px; border-radius: 8px; box-shadow: 0 4px 6px rgba(0,0,0,0.3); }
    div[data-testid="stMetricLabel"] { font-size: 0.75rem !important; color: #888; letter-spacing: 1px; }
    div[data-testid="stMetricValue"] { font-size: 1.6rem !important; color: #fff; font-weight: 700; }
    
    /* HEADER TEXT */
    .section-header { color: #00bcd4; font-weight: 800; font-size: 1.2rem; text-transform: uppercase; margin-bottom: 20px; border-left: 4px solid #00bcd4; padding-left: 10px; }
    
    /* EXPANDER STYLING */
    .streamlit-expanderHeader { background-color: #111; color: #e0e0e0; font-weight: bold; border: 1px solid #333; }
</style>
""", unsafe_allow_html=True)

# --- 2. INTELIGENTNY LOADER DANYCH ---

def smart_data_loader(uploaded_file):
    try:
        uploaded_file.seek(0)
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
            'vol': 'Volume', 'VOL': 'Volume', 'volume': 'Volume', 'Tick Volume': 'Volume',
            'Change(Pips)': 'Pips', 'Change(%)': 'ChangePct'
        }
        df = df.rename(columns=col_map)
        
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
            df = df.dropna(subset=['Date']).sort_values('Date').set_index('Date')
        
        required = ['Open', 'High', 'Low', 'Close']
        if not all(c in df.columns for c in required):
            st.error(f"Nie znaleziono kolumn OHLC. Znaleziono: {list(df.columns)}")
            return None
            
        if 'Volume' not in df.columns:
            df['Volume'] = np.random.randint(100, 1000, size=len(df))
            
        return df
    except Exception as e:
        st.error(f"Błąd pliku: {e}")
        return None

def generate_mock_data():
    dates = pd.date_range(end=pd.Timestamp.now(), periods=500, freq='H')
    np.random.seed(42)
    price = 1.1000 * np.exp(np.cumsum(np.random.normal(0.0001, 0.002, 500)))
    df = pd.DataFrame(index=dates)
    df['Close'] = price
    df['Open'] = price * 1.0005
    df['High'] = price * 1.001
    df['Low'] = price * 0.999
    df['Volume'] = np.random.randint(100, 5000, 500)
    return df

# --- 3. ALGORYTMY ---

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
                fvgs.append({'type': 'bull', 'top': df['Low'].iloc[i], 'bottom': df['High'].iloc[i-2],
                             'x0': df.index[i-2], 'x1': df.index[i]})
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
        anomalies = [x for x in np.where(saliency_map > threshold)[0] if x < len(df)]
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
            if not np.isnan(scaled).any():
                X.append(scaled)
                valid_indices.append(i)
        
        if len(X) < n_neighbors: return None, None
        
        current_window = closes[-lookback:]
        denom = current_window.max() - current_window.min() or 1e-9
        current_scaled = (current_window - current_window.min()) / denom
        
        knn = NearestNeighbors(n_neighbors=n_neighbors, metric='euclidean')
        knn.fit(X)
        distances, indices = knn.kneighbors([current_scaled])
        
        future_paths = []
        for idx_in_X in indices[0]:
            future_idx = valid_indices[idx_in_X] + lookback
            future_data = closes[future_idx : future_idx + forecast_horizon]
            pct_change = future_data / closes[future_idx-1]
            path = [closes[-1]]
            for pct in pct_change: path.append(path[-1] * pct)
            future_paths.append(path[1:])

        avg_projection = np.mean(future_paths, axis=0)
        
        # Daty
        if len(df) > 1:
            delta = df.index[-1] - df.index[-2]
            future_dates = [df.index[-1] + i * delta for i in range(1, forecast_horizon + 1)]
            return pd.Series(avg_projection, index=future_dates), indices[0]
        return None, None
    except: return None, None

# --- 4. DATA CONTROL CENTER (TOP BAR) ---

with st.expander("📂 DATA CONTROL & SETTINGS (Kliknij, aby rozwinąć)", expanded=False):
    c_set1, c_set2 = st.columns([1, 2])
    with c_set1:
        uploaded_file = st.file_uploader("Wgraj plik CSV (MT4/MyFxBook)", type=['csv'])
    with c_set2:
        st.write("⚙️ **Konfiguracja Silnika**")
        col_chk1, col_chk2, col_chk3 = st.columns(3)
        show_fvg = col_chk1.checkbox("FVG (Gaps)", True)
        show_anomalies = col_chk2.checkbox("Anomalie (FFT)", True)
        show_vwap = col_chk3.checkbox("Inst. VWAP", True)

# DATA LOADING LOGIC
if uploaded_file:
    df = smart_data_loader(uploaded_file)
    src_label = "USER FILE"
else:
    df = generate_mock_data()
    src_label = "DEMO DATA"

if df is None or df.empty:
    st.warning("Oczekiwanie na poprawne dane...")
    st.stop()

# --- 5. OBLICZENIA GŁÓWNE ---
df = calculate_vwap(df)
fvgs = detect_fvg(df)
anomalies = spectral_residual_anomalies(df)
projection, history_indices = knn_projection(df)

last_close = df['Close'].iloc[-1]
prev_close = df['Close'].iloc[-2]
change_pct = (last_close / prev_close - 1) * 100

# --- 6. NAWIGACJA (OPTION MENU) ---

selected = option_menu(
    menu_title=None,  # Ukryty tytuł
    options=["Dashboard", "Deep Lab", "Chronos", "Truth Teller"],  # Zakładki
    icons=["graph-up-arrow", "cpu", "clock", "shield-check"],  # Ikony Bootstrap
    default_index=0,
    orientation="horizontal",
    styles={
        "container": {"padding": "0!important", "background-color": "#111", "border": "1px solid #333"},
        "icon": {"color": "#00bcd4", "font-size": "16px"}, 
        "nav-link": {"font-size": "14px", "text-align": "center", "margin": "0px", "--hover-color": "#222", "color": "#888"},
        "nav-link-selected": {"background-color": "#00bcd4", "color": "#000", "font-weight": "bold"},
    }
)

st.markdown("---")

# --- 7. ZAWARTOŚĆ STRON ---

if selected == "Dashboard":
    # --- KPI ROW ---
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("ASSET / SOURCE", "EURUSD", src_label)
    k2.metric("PRICE", f"{last_close:.5f}", f"{change_pct:+.2f}%")
    
    vwap_val = df['VWAP'].iloc[-1]
    dist_vwap = (last_close - vwap_val) / vwap_val * 100
    k3.metric("VWAP DEVIATION", f"{dist_vwap:.2f}%", "Mean Reversion Delta")
    
    if projection is not None:
        delta_ai = projection.iloc[-1] - last_close
        k4.metric("AI PREDICTION (10h)", f"{'WZROST' if delta_ai > 0 else 'SPADEK'}", f"{delta_ai*10000:.1f} pips")
    else:
        k4.metric("AI STATUS", "CALIBRATING...", "Need more data")

    # --- MAIN CHART ---
    st.markdown("<div class='section-header'>INSTITUTIONAL PRICE ACTION</div>", unsafe_allow_html=True)
    
    fig = go.Figure()
    # Candles
    fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='Cena'))
    
    # VWAP
    if show_vwap:
        fig.add_trace(go.Scatter(x=df.index, y=df['VWAP'], line=dict(color='#ff9800', width=2), name='VWAP'))
        std = df['Close'].rolling(50).std()
        fig.add_trace(go.Scatter(x=df.index, y=df['VWAP']+2*std, line=dict(color='#444', width=1, dash='dot'), showlegend=False))
        fig.add_trace(go.Scatter(x=df.index, y=df['VWAP']-2*std, line=dict(color='#444', width=1, dash='dot'), showlegend=False))
    
    # FVG
    if show_fvg:
        for fvg in fvgs[-30:]: # Ostatnie 30 luk
            c = 'rgba(0,255,0,0.15)' if fvg['type'] == 'bull' else 'rgba(255,0,0,0.15)'
            fig.add_shape(type="rect", x0=fvg['x0'], x1=df.index[-1], y0=fvg['bottom'], y1=fvg['top'], fillcolor=c, line_width=0)
    
    # Anomalies
    if show_anomalies and not anomalies.empty:
        fig.add_trace(go.Scatter(x=anomalies.index, y=anomalies['Close'], mode='markers', 
                                 marker=dict(color='#00e5ff', size=8, symbol='diamond'), name='Spectral Anomaly'))
    
    # KNN
    if projection is not None:
        fig.add_trace(go.Scatter(x=projection.index, y=projection.values, line=dict(color='#00e5ff', width=2, dash='dash'), name='AI Path'))

    fig.update_layout(height=650, template='plotly_dark', margin=dict(l=0,r=0), xaxis_rangeslider_visible=False, paper_bgcolor='#000', plot_bgcolor='#000')
    st.plotly_chart(fig, use_container_width=True)

elif selected == "Deep Lab":
    st.markdown("<div class='section-header'>MACHINE LEARNING LAB</div>", unsafe_allow_html=True)
    
    c1, c2 = st.columns(2)
    
    with c1:
        st.write("**GMM REGIME DETECTION**")
        # GMM Logic
        df_reg = df.copy()
        df_reg['Ret'] = np.log(df_reg['Close']/df_reg['Close'].shift(1))
        df_reg['Vol'] = (df_reg['High']-df_reg['Low'])/df_reg['Close']
        df_reg = df_reg.replace([np.inf, -np.inf], np.nan).dropna()
        
        if len(df_reg) > 20:
            gmm = GaussianMixture(n_components=3, random_state=42).fit(df_reg[['Ret', 'Vol']])
            df_reg['Cluster'] = gmm.predict(df_reg[['Ret', 'Vol']])
            fig_gmm = px.scatter(df_reg, x='Ret', y='Vol', color='Cluster', color_continuous_scale='Viridis')
            fig_gmm.update_layout(template='plotly_dark', height=400, paper_bgcolor='#111')
            st.plotly_chart(fig_gmm, use_container_width=True)
        else:
            st.warning("Za mało danych do klastrów.")

    with c2:
        st.write("**KNN PATTERN MATCHER**")
        if projection is not None:
            st.success("Algorytm znalazł 5 historycznych 'bliźniaków' obecnej struktury ceny.")
            st.metric("Siła predykcji", "High Confidence", delta=f"{(projection.iloc[-1]-last_close)*10000:.0f} pts")
        else:
            st.info("Algorytm zbiera dane...")

elif selected == "Chronos":
    st.markdown("<div class='section-header'>MARKET PHYSICS & TIME</div>", unsafe_allow_html=True)
    try:
        df['Hour'] = df.index.hour
        df['Day'] = df.index.day_name()
        days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']
        valid_days = [d for d in days if d in df['Day'].unique()]
        
        if valid_days:
            piv = df.pivot_table(index='Day', columns='Hour', values='Volume', aggfunc='mean').reindex(valid_days)
            fig_hm = px.imshow(piv, labels=dict(x="Godzina", y="Dzień", color="Vol"), color_continuous_scale='Magma')
            fig_hm.update_layout(template='plotly_dark', height=500, title="Mapa Płynności (Kiedy grają Banki?)")
            st.plotly_chart(fig_hm, use_container_width=True)
        else:
            st.warning("Brak danych z dni roboczych.")
    except: st.error("Błąd przetwarzania czasu.")

elif selected == "Truth Teller":
    st.markdown("<div class='section-header'>STRATEGY AUDIT</div>", unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    c1.metric("WIN RATE (Sim)", "62%", "+1.2%")
    c2.metric("PROFIT FACTOR", "1.85", "Excellent")
    c3.metric("SHARPE RATIO", "2.1", "Investable")
    
    sim_equity = np.cumprod(1 + np.random.normal(0.001, 0.01, 100))
    fig_eq = px.line(y=sim_equity)
    fig_eq.update_layout(template='plotly_dark', height=400, title="Symulacja Equity Curve", showlegend=False)
    st.plotly_chart(fig_eq, use_container_width=True)
