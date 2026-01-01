import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import xgboost as xgb
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.mixture import GaussianMixture
from scipy.stats import t
import warnings

warnings.filterwarnings("ignore")

# --- 1. KONFIGURACJA STRONY (Musi być pierwsza) ---
st.set_page_config(
    page_title="QUANT LAB | Institutional Terminal",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS dla "Bloomberg Feel"
st.markdown("""
<style>
    .stApp { background-color: #0e1117; }
    div[data-testid="stMetricValue"] { font-size: 24px; color: #00ff00; font-family: 'Courier New', monospace; }
    div[data-testid="stMetricLabel"] { font-size: 14px; color: #888; }
</style>
""", unsafe_allow_html=True)

# --- 2. CONFIG & CACHING ---
CONFIG = {
    'TARGET': 'EURUSD=X',
    'MACRO': {'US10Y': '^TNX', 'VIX': '^VIX', 'SPX': '^GSPC', 'GOLD': 'GC=F'},
    'LOOKBACK': '2y',
    'BEST_PARAMS': {'eta': 0.05, 'max_depth': 5, 'objective': 'binary:logistic', 'eval_metric': 'logloss'}
}

@st.cache_data(ttl=3600)
def load_data(ticker):
    tickers = [ticker] + list(CONFIG['MACRO'].values())
    # Flatten columns fix for new yfinance
    df = yf.download(tickers, period=CONFIG['LOOKBACK'], interval="1d", progress=False)
    
    # Obsługa MultiIndex w nowym yfinance
    data = pd.DataFrame()
    if isinstance(df.columns, pd.MultiIndex):
        try:
            data['Close'] = df['Close'][ticker]
            data['High'] = df['High'][ticker]
            data['Low'] = df['Low'][ticker]
            data['Open'] = df['Open'][ticker]
            for key, val in CONFIG['MACRO'].items():
                if val in df['Close'].columns:
                    data[key] = df['Close'][val]
        except KeyError:
             st.error("Data mapping error. Check tickers.")
             return pd.DataFrame()
    else:
        # Fallback
        data = df
        
    data = data.ffill().dropna()
    return data

@st.cache_data
def run_quant_engine(df):
    # 1. Feature Engineering
    df['Log_Ret'] = np.log(df['Close'] / df['Close'].shift(1))
    df['Vol_20'] = df['Log_Ret'].rolling(20).std()
    
    # Momentum & RSI
    for w in [5, 14]:
        df[f'Mom_{w}'] = df['Close'].pct_change(w)
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(w).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(w).mean()
        rs = gain / loss
        df[f'RSI_{w}'] = 100 - (100 / (1 + rs))

    df.dropna(inplace=True)

    # 2. GMM Regime Detection
    X_regime = df[['Log_Ret', 'Vol_20']].values
    gmm = GaussianMixture(n_components=3, random_state=42).fit(X_regime)
    regimes = gmm.predict(X_regime)
    # Sort regimes by volatility (0=Low, 2=High)
    means = gmm.means_[:, 1]
    sorted_idx = np.argsort(means)
    regime_map = {sorted_idx[0]: 'LOW VOL (Green)', sorted_idx[1]: 'NORMAL (Orange)', sorted_idx[2]: 'HIGH VOL (Red)'}
    current_regime = regime_map[regimes[-1]]

    # 3. XGBoost Signal
    # Target: Price Up in 3 days > spread
    df['Target'] = (df['Close'].shift(-3) / df['Close'] - 1 > 0.0010).astype(int)
    
    features = [c for c in df.columns if 'Mom' in c or 'Vol' in c or 'RSI' in c]
    X = df[features].iloc[:-3] # Drop last rows with NaN target
    y = df['Target'].iloc[:-3]
    
    model = xgb.XGBClassifier(**CONFIG['BEST_PARAMS'], n_estimators=100)
    model.fit(X, y)
    
    last_row = df[features].iloc[[-1]]
    prob_up = model.predict_proba(last_row)[0][1]
    
    # 4. Monte Carlo (Student-t)
    last_price = df['Close'].iloc[-1]
    vol_ann = df['Vol_20'].iloc[-1] * np.sqrt(252)
    sims = 1000
    t_dist = t.rvs(df=3, size=sims)
    shocks = t_dist * (vol_ann / np.sqrt(252))
    mc_paths = last_price * np.exp(shocks) # 1 day forecast
    
    support = np.percentile(mc_paths, 5)
    resistance = np.percentile(mc_paths, 95)

    return {
        'price': last_price,
        'regime': current_regime,
        'prob_up': prob_up,
        'support': support,
        'resistance': resistance,
        'df': df
    }

# --- 3. SIDEBAR ---
st.sidebar.header("🎛️ Control Panel")
symbol = st.sidebar.text_input("Asset", "EURUSD=X")

if st.sidebar.button("RUN ANALYSIS", type="primary"):
    with st.spinner("Processing Quantum Data..."):
        data = load_data(symbol)
        if not data.empty:
            res = run_quant_engine(data)
            
            # --- 4. MAIN DASHBOARD (HUD) ---
            
            # Logic Sygnału
            signal = "WAIT"
            color = "off"
            if res['prob_up'] > 0.60: 
                signal = "STRONG BUY"
                color = "normal" # Green in standard theme
            elif res['prob_up'] < 0.40: 
                signal = "STRONG SELL"
                color = "inverse" # Red-ish usually
                
            # Top Metrics Row
            c1, c2, c3, c4 = st.columns(4)
            with c1: st.metric("PRICE", f"{res['price']:.5f}", delta=None)
            with c2: st.metric("SIGNAL", signal, delta=f"{res['prob_up']:.0%}")
            with c3: st.metric("REGIME", res['regime'])
            with c4: st.metric("RANGE (MC)", f"{res['resistance']-res['support']:.4f}")

            # --- 5. INTERACTIVE CHARTS (PLOTLY) ---
            
            # Main Chart with Monte Carlo Zones
            fig = go.Figure()

            # Candlestick
            plot_df = res['df'].tail(100)
            fig.add_trace(go.Candlestick(x=plot_df.index,
                            open=plot_df['Open'], high=plot_df['High'],
                            low=plot_df['Low'], close=plot_df['Close'],
                            name='Price'))

            # Monte Carlo Lines (Projection)
            # Rysujemy proste linie od ostatniej ceny do prognozy
            last_date = plot_df.index[-1]
            next_date = last_date + pd.Timedelta(days=1)
            
            fig.add_trace(go.Scatter(x=[last_date, next_date], y=[res['price'], res['resistance']],
                                     mode='lines', line=dict(color='red', dash='dot'), name='Res (95%)'))
            
            fig.add_trace(go.Scatter(x=[last_date, next_date], y=[res['price'], res['support']],
                                     mode='lines', line=dict(color='green', dash='dot'), name='Sup (5%)'))

            fig.update_layout(title='Market Structure & Quantum Zones', 
                              height=600, 
                              template='plotly_dark',
                              xaxis_rangeslider_visible=False)
            
            st.plotly_chart(fig, use_container_width=True)
            
            # --- 6. DETAILS TABS ---
            t1, t2 = st.tabs(["📊 Macro Context", "🧠 AI Diagnostics"])
            
            with t1:
                st.info("Macro Correlations (Coming soon in v28)")
                # Tu można wrzucić heatmapę w plotly
                
            with t2:
                col1, col2 = st.columns(2)
                col1.progress(res['prob_up'], text="Bullish Probability")
                col2.write(f"**Predicted Support:** {res['support']:.5f}")
                col2.write(f"**Predicted Resistance:** {res['resistance']:.5f}")

        else:
            st.error("No data found.")
else:
    st.info("👈 Click **RUN ANALYSIS** to start the engine.")
