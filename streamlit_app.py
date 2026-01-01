import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import xgboost as xgb
import plotly.graph_objects as go
import plotly.express as px
from sklearn.mixture import GaussianMixture
from scipy.stats import t
import warnings
import time # Dodane do obsługi rate-limit

# --- 1. KONFIGURACJA STRONY ---
st.set_page_config(
    page_title="QUANT LAB | EURUSD Institutional Terminal",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="expanded"
)

warnings.filterwarnings("ignore")

# --- CUSTOM CSS ---
st.markdown("""
<style>
    .stApp { background-color: #0e1117; }
    div[data-testid="stMetricValue"] { 
        font-size: 24px; color: #00ff00; font-family: 'Courier New', monospace; font-weight: bold;
    }
    div[data-testid="stMetricLabel"] { font-size: 14px; color: #888; }
    .stTabs [aria-selected="true"] {
        background-color: #1e2130; color: #00ff00; border-bottom: 2px solid #00ff00;
    }
</style>
""", unsafe_allow_html=True)

# --- 2. CONFIG ---
CONFIG = {
    'TARGET': 'EURUSD=X',
    'MACRO': {
        'US10Y': '^TNX', 'DXY': 'DX-Y.NYB', 'VIX': '^VIX', 'SPX': '^GSPC', 'GOLD': 'GC=F'
    },
    'LOOKBACK': '2y',
}

# --- 3. MATH ENGINES ---
def calculate_hurst(series):
    """Robust Hurst Exponent"""
    try:
        lags = range(2, 20)
        tau = [np.sqrt(np.std(np.subtract(series[lag:], series[:-lag]))) for lag in lags]
        poly = np.polyfit(np.log(lags), np.log(tau), 1)
        return poly[0] * 2.0
    except:
        return 0.5

def kalman_filter(data, Q=1e-5, R=0.01):
    n_iter = len(data)
    sz = (n_iter,)
    xhat = np.zeros(sz)      
    P = np.zeros(sz)         
    xhatminus = np.zeros(sz) 
    Pminus = np.zeros(sz)    
    K = np.zeros(sz)         
    xhat[0] = data.iloc[0]
    P[0] = 1.0
    for k in range(1, n_iter):
        xhatminus[k] = xhat[k-1]
        Pminus[k] = P[k-1] + Q
        K[k] = Pminus[k] / (Pminus[k] + R)
        xhat[k] = xhatminus[k] + K[k] * (data.iloc[k] - xhatminus[k])
        P[k] = (1 - K[k]) * Pminus[k]
    return xhat

# --- 4. ROBUST DATA LOADER (BULLETPROOF) ---
@st.cache_data(ttl=3600)
def load_data(ticker):
    tickers = [ticker] + list(CONFIG['MACRO'].values())
    df = pd.DataFrame()
    
    # Próba pobrania wszystkiego z mechanizmem Retry
    max_retries = 3
    success = False
    
    for attempt in range(max_retries):
        try:
            # Pobieramy dane
            raw_data = yf.download(tickers, period=CONFIG['LOOKBACK'], interval="1d", progress=False)
            
            if not raw_data.empty:
                df = raw_data
                success = True
                break
        except Exception as e:
            time.sleep(2) # Czekamy 2 sekundy przed ponowną próbą
            continue
            
    # CRITICAL FALLBACK: Jeśli makro zawiedzie, pobierz TYLKO główny ticker
    if df.empty or ticker not in str(df.columns):
        try:
            df = yf.download([ticker], period=CONFIG['LOOKBACK'], interval="1d", progress=False)
        except:
            return pd.DataFrame() # Kapitulacja

    # --- FLATTENING MULTI-INDEX (Cleaning) ---
    clean_data = pd.DataFrame()
    
    # Logika wyciągania danych z yfinance (który ma skomplikowaną strukturę)
    try:
        if isinstance(df.columns, pd.MultiIndex):
            # 1. Price Data
            if ticker in df['Close'].columns:
                clean_data['Close'] = df['Close'][ticker]
                clean_data['High'] = df['High'][ticker]
                clean_data['Low'] = df['Low'][ticker]
                clean_data['Open'] = df['Open'][ticker]
            # 2. Macro Data (Optional)
            for key, val in CONFIG['MACRO'].items():
                if val in df['Close'].columns:
                    clean_data[key] = df['Close'][val]
        else:
            # Simple structure
            clean_data = df
            
        clean_data = clean_data.ffill().dropna()
        return clean_data
        
    except Exception as e:
        return pd.DataFrame()

# --- 5. QUANT ENGINE ---
@st.cache_data
def run_quant_engine(df):
    if len(df) < 50: return None
    
    df['Log_Ret'] = np.log(df['Close'] / df['Close'].shift(1))
    df['Vol_20'] = df['Log_Ret'].rolling(20).std()
    
    # Kalman & Hurst
    df['Kalman'] = kalman_filter(df['Close'])
    hurst_val = calculate_hurst(df['Close'].tail(100).values)
    
    # Z-Score
    df['MA_50'] = df['Close'].rolling(50).mean()
    df['Z_Score'] = (df['Close'] - df['MA_50']) / df['Close'].rolling(50).std()
    
    # Macro Correlations (If available)
    for m in CONFIG['MACRO'].keys():
        if m in df.columns:
            df[f'Corr_{m}'] = df['Log_Ret'].rolling(30).corr(df[m].pct_change())

    df.dropna(inplace=True)
    if df.empty: return None

    # Regime
    try:
        X_regime = df[['Log_Ret', 'Vol_20']].values
        gmm = GaussianMixture(n_components=3, random_state=42).fit(X_regime)
        regimes = gmm.predict(X_regime)
        means = gmm.means_[:, 1]
        sorted_idx = np.argsort(means)
        regime_map = {sorted_idx[0]: 'LOW VOL', sorted_idx[1]: 'NEUTRAL', sorted_idx[2]: 'HIGH VOL'}
        current_regime = regime_map[regimes[-1]]
    except:
        current_regime = "N/A"

    # AI Model
    df['Target'] = (df['Close'].shift(-3) / df['Close'] - 1 > 0.0010).astype(int)
    features = ['Vol_20', 'Z_Score'] + [c for c in df.columns if 'Corr_' in c]
    valid_features = [f for f in features if f in df.columns]
    
    model = xgb.XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.05, n_jobs=-1)
    model.fit(df[valid_features].iloc[:-3], df['Target'].iloc[:-3])
    prob_up = model.predict_proba(df[valid_features].iloc[[-1]])[0][1]

    # Monte Carlo
    last_price = df['Close'].iloc[-1]
    vol_ann = df['Vol_20'].iloc[-1] * np.sqrt(252)
    t_dist = t.rvs(df=3, size=1000) * (vol_ann / np.sqrt(252))
    mc_paths = last_price * np.exp(t_dist)
    
    return {
        'price': last_price,
        'kalman': df['Kalman'].iloc[-1],
        'regime': current_regime,
        'hurst': hurst_val,
        'prob_up': prob_up,
        'support': np.percentile(mc_paths, 5),
        'resistance': np.percentile(mc_paths, 95),
        'df': df
    }

# --- 6. UI ---
st.sidebar.header("🎛️ QUANT LAB")
symbol = st.sidebar.text_input("Asset", "EURUSD=X")

if st.sidebar.button("INITIALIZE SYSTEM", type="primary"):
    with st.spinner("Connecting to Institutional Data Feed..."):
        data = load_data(symbol)
        
        if not data.empty:
            res = run_quant_engine(data)
            if res:
                # --- HUD ---
                prob = res['prob_up']
                signal = "LONG" if prob > 0.6 else "SHORT" if prob < 0.4 else "NEUTRAL"
                color = "normal" if signal == "LONG" else "inverse" if signal == "SHORT" else "off"
                
                c1, c2, c3, c4 = st.columns(4)
                with c1: st.metric("PRICE", f"{res['price']:.5f}")
                with c2: st.metric("AI SIGNAL", signal, delta=f"{prob:.1%}", delta_color=color)
                with c3: st.metric("REGIME", res['regime'])
                with c4: st.metric("HURST", f"{res['hurst']:.2f}")

                # --- CHARTS ---
                t1, t2, t3 = st.tabs(["📈 Market", "🌍 Macro", "⚠️ Anomalies"])
                
                with t1:
                    fig = go.Figure()
                    pdf = res['df'].tail(150)
                    fig.add_trace(go.Candlestick(x=pdf.index, open=pdf['Open'], high=pdf['High'], low=pdf['Low'], close=pdf['Close'], name='Price'))
                    fig.add_trace(go.Scatter(x=pdf.index, y=pdf['Kalman'], line=dict(color='yellow', width=2), name='Kalman'))
                    fig.update_layout(height=600, template='plotly_dark', title="Institutional Price Action")
                    st.plotly_chart(fig, use_container_width=True)

                with t2:
                    corr_cols = [c for c in res['df'].columns if 'Corr_' in c]
                    if corr_cols:
                        curr = res['df'][corr_cols].iloc[-1].sort_values()
                        fig = px.bar(x=curr.values, y=[c.replace('Corr_', '') for c in curr.index], orientation='h', title="Correlations")
                        fig.update_layout(template='plotly_dark')
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.warning("⚠️ Macro Data Limited (Rate Limit Active). Showing Price Only.")

                with t3:
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=pdf.index, y=pdf['Z_Score'], fill='tozeroy', line=dict(color='cyan')))
                    fig.add_hline(y=2.0, line_color='red', line_dash='dash')
                    fig.add_hline(y=-2.0, line_color='green', line_dash='dash')
                    fig.update_layout(height=400, template='plotly_dark', title="Z-Score Deviation")
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.error("Quant Engine Failed. Not enough data.")
        else:
            st.error("Data Feed Disconnected (Yahoo Finance API Error). Try again in 1 min.")
else:
    st.info("System Ready.")
