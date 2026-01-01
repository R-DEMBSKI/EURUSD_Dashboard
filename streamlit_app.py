import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import xgboost as xgb
import plotly.graph_objects as go
import plotly.express as px
from sklearn.mixture import GaussianMixture
from sklearn.linear_model import LinearRegression
from scipy.stats import t
import warnings
import time

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

def calculate_fair_value(df):
    """Oblicza Wartość Godziwą (Fair Value) na podstawie regresji makroekonomicznej"""
    # Potrzebujemy DXY i US10Y jako minimum
    try:
        macro_cols = [c for c in df.columns if c in ['DXY', 'US10Y', 'SPX', 'GOLD']]
        if not macro_cols: return None
        
        # Przygotowanie danych (zmiany procentowe dla stacjonarności, potem rekonstrukcja)
        # Dla uproszczenia w wersji v4 używamy znormalizowanych cen do wykrycia dywergencji
        df_norm = df.copy()
        
        # Target
        y = df_norm['Close']
        X = df_norm[macro_cols]
        
        # Prosta regresja na ostatnich 60 dniach (Rolling Window Regression)
        # Symulujemy "Fair Value" jako liniową kombinację czynników makro
        model = LinearRegression()
        model.fit(X, y)
        fair_value = model.predict(X)
        return fair_value
    except:
        return None

# --- 4. DATA LOADER ---
@st.cache_data(ttl=3600)
def load_data(ticker):
    tickers = [ticker] + list(CONFIG['MACRO'].values())
    df = pd.DataFrame()
    
    for attempt in range(3):
        try:
            raw_data = yf.download(tickers, period=CONFIG['LOOKBACK'], interval="1d", progress=False)
            if not raw_data.empty:
                df = raw_data
                break
        except:
            time.sleep(1)
            continue
            
    if df.empty: return pd.DataFrame()

    clean_data = pd.DataFrame()
    try:
        # Obsługa MultiIndex
        if isinstance(df.columns, pd.MultiIndex):
            # Price
            if ticker in df['Close'].columns:
                clean_data['Close'] = df['Close'][ticker]
                clean_data['High'] = df['High'][ticker]
                clean_data['Low'] = df['Low'][ticker]
                clean_data['Open'] = df['Open'][ticker]
            # Macro
            for key, val in CONFIG['MACRO'].items():
                if val in df['Close'].columns:
                    clean_data[key] = df['Close'][val]
        else:
            clean_data = df
            
        return clean_data.ffill().dropna()
    except:
        return pd.DataFrame()

# --- 5. QUANT ENGINE ---
@st.cache_data
def run_quant_engine(df):
    if len(df) < 50: return None
    
    df['Log_Ret'] = np.log(df['Close'] / df['Close'].shift(1))
    df['Vol_20'] = df['Log_Ret'].rolling(20).std()
    
    # Kalman
    df['Kalman'] = kalman_filter(df['Close'])
    
    # Hurst
    hurst_val = calculate_hurst(df['Close'].tail(100).values)
    
    # Fair Value Model (NOWOŚĆ)
    fv = calculate_fair_value(df)
    if fv is not None:
        df['Fair_Value'] = fv
        df['FV_Gap'] = df['Close'] - df['Fair_Value']
    else:
        df['Fair_Value'] = np.nan
        df['FV_Gap'] = 0.0

    # Z-Score
    df['MA_50'] = df['Close'].rolling(50).mean()
    df['Z_Score'] = (df['Close'] - df['MA_50']) / df['Close'].rolling(50).std()
    
    # Correlations
    for m in CONFIG['MACRO'].keys():
        if m in df.columns:
            df[f'Corr_{m}'] = df['Log_Ret'].rolling(30).corr(df[m].pct_change())

    df.dropna(inplace=True)
    if df.empty: return None

    # Regime & AI
    X_regime = df[['Log_Ret', 'Vol_20']].values
    try:
        gmm = GaussianMixture(n_components=3, random_state=42).fit(X_regime)
        regimes = gmm.predict(X_regime)
        means = gmm.means_[:, 1]
        sorted_idx = np.argsort(means)
        regime_map = {sorted_idx[0]: 'LOW VOL', sorted_idx[1]: 'NEUTRAL', sorted_idx[2]: 'HIGH VOL'}
        current_regime = regime_map[regimes[-1]]
    except:
        current_regime = "N/A"

    df['Target'] = (df['Close'].shift(-3) / df['Close'] - 1 > 0.0010).astype(int)
    features = ['Vol_20', 'Z_Score'] + [c for c in df.columns if 'Corr_' in c]
    valid_features = [f for f in features if f in df.columns]
    
    model = xgb.XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.05, n_jobs=-1)
    model.fit(df[valid_features].iloc[:-3], df['Target'].iloc[:-3])
    prob_up = model.predict_proba(df[valid_features].iloc[[-1]])[0][1]

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
        'fv_gap': df['FV_Gap'].iloc[-1],
        'support': np.percentile(mc_paths, 5),
        'resistance': np.percentile(mc_paths, 95),
        'df': df
    }

# --- 6. UI ---
st.sidebar.header("🎛️ QUANT LAB")
symbol = st.sidebar.text_input("Asset", "EURUSD=X")

if st.sidebar.button("INITIALIZE SYSTEM", type="primary"):
    with st.spinner("Calculating Institutional Fair Value..."):
        data = load_data(symbol)
        
        if not data.empty:
            res = run_quant_engine(data)
            if res:
                # --- HUD ---
                prob = res['prob_up']
                signal = "LONG" if prob > 0.6 else "SHORT" if prob < 0.4 else "NEUTRAL"
                color = "normal" if signal == "LONG" else "inverse" if signal == "SHORT" else "off"
                
                # Fair Value Logic
                fv_gap = res['fv_gap']
                fv_desc = "Undervalued" if fv_gap < 0 else "Overvalued"
                
                c1, c2, c3, c4 = st.columns(4)
                with c1: st.metric("PRICE", f"{res['price']:.5f}")
                with c2: st.metric("AI SIGNAL", signal, delta=f"{prob:.1%}", delta_color=color)
                with c3: st.metric("REGIME", res['regime'])
                with c4: st.metric("FAIR VALUE GAP", f"{fv_gap:.4f}", delta=fv_desc, delta_color="inverse")

                # --- CHARTS ---
                t1, t2, t3, t4 = st.tabs(["📈 Market", "💎 Fair Value", "🌍 Macro", "⚠️ Anomalies"])
                
                with t1:
                    fig = go.Figure()
                    pdf = res['df'].tail(150)
                    fig.add_trace(go.Candlestick(x=pdf.index, open=pdf['Open'], high=pdf['High'], low=pdf['Low'], close=pdf['Close'], name='Price'))
                    fig.add_trace(go.Scatter(x=pdf.index, y=pdf['Kalman'], line=dict(color='yellow', width=2), name='Kalman Filter'))
                    fig.update_layout(height=600, template='plotly_dark', title="Institutional Price Action", margin=dict(l=0, r=0, t=30, b=0))
                    st.plotly_chart(fig, use_container_width=True)

                with t2:
                    st.markdown("### 💎 Institutional Fair Value Model")
                    st.info("Model porównuje cenę spot EURUSD do 'syntetycznej' ceny wynikającej z DXY, Rentowności i S&P500. Rozjazdy (Gaps) to okazje.")
                    fig_fv = go.Figure()
                    pdf = res['df'].tail(200)
                    
                    # Price vs Fair Value
                    fig_fv.add_trace(go.Scatter(x=pdf.index, y=pdf['Close'], line=dict(color='white', width=1), name='Actual Price'))
                    if 'Fair_Value' in pdf.columns:
                        fig_fv.add_trace(go.Scatter(x=pdf.index, y=pdf['Fair_Value'], line=dict(color='orange', width=2, dash='dash'), name='Macro Fair Value'))
                        
                        # Wypełnienie chmury
                        fig_fv.add_trace(go.Scatter(
                            x=pdf.index, y=pdf['Fair_Value'],
                            fill=None, mode='lines', line_color='orange', showlegend=False
                        ))
                        fig_fv.add_trace(go.Scatter(
                            x=pdf.index, y=pdf['Close'],
                            fill='tonexty', mode='lines', line_color='white', showlegend=False,
                            fillcolor='rgba(255, 0, 0, 0.1)' if fv_gap > 0 else 'rgba(0, 255, 0, 0.1)'
                        ))

                    fig_fv.update_layout(height=500, template='plotly_dark', title="Fair Value Divergence")
                    st.plotly_chart(fig_fv, use_container_width=True)

                with t3:
                    corr_cols = [c for c in res['df'].columns if 'Corr_' in c]
                    if corr_cols:
                        curr = res['df'][corr_cols].iloc[-1].sort_values()
                        fig = px.bar(x=curr.values, y=[c.replace('Corr_', '') for c in curr.index], orientation='h', title="Real-time Correlations")
                        fig.update_layout(template='plotly_dark')
                        st.plotly_chart(fig, use_container_width=True)

                with t4:
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=pdf.index, y=pdf['Z_Score'], fill='tozeroy', line=dict(color='cyan')))
                    fig.add_hline(y=2.0, line_color='red', line_dash='dash')
                    fig.add_hline(y=-2.0, line_color='green', line_dash='dash')
                    fig.update_layout(height=400, template='plotly_dark', title="Statistical Reversion (Z-Score)")
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.error("Quant Engine Failed. Not enough data.")
        else:
            st.error("Data Feed Disconnected. Try again later.")
else:
    st.info("System Ready.")
