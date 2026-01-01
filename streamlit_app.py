import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import xgboost as xgb
import plotly.graph_objects as go
from sklearn.mixture import GaussianMixture
from scipy.stats import t
import warnings

# Wyciszenie ostrzeżeń dla czystości logów
warnings.filterwarnings("ignore")

# --- 1. KONFIGURACJA STRONY (Musi być na samej górze) ---
st.set_page_config(
    page_title="QUANT LAB | Institutional Terminal",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS dla "Bloomberg/Institutional Feel"
st.markdown("""
<style>
    .stApp { background-color: #0e1117; }
    /* Stylizacja metryk HUD */
    div[data-testid="stMetricValue"] { font-size: 24px; color: #00ff00; font-family: 'Courier New', monospace; }
    div[data-testid="stMetricLabel"] { font-size: 14px; color: #888; }
    /* Pasek postępu - kolor */
    .stProgress > div > div > div > div { background-color: #00ff00; }
</style>
""", unsafe_allow_html=True)

# --- 2. CONFIG & CACHING ---
CONFIG = {
    'TARGET': 'EURUSD=X',
    'MACRO': {'US10Y': '^TNX', 'VIX': '^VIX', 'SPX': '^GSPC', 'GOLD': 'GC=F'},
    'LOOKBACK': '2y',
    # Parametry modelu (można stroić przez Optuna offline)
    'BEST_PARAMS': {'eta': 0.05, 'max_depth': 5, 'objective': 'binary:logistic', 'eval_metric': 'logloss'}
}

@st.cache_data(ttl=3600)
def load_data(ticker):
    """Pobiera dane i obsługuje MultiIndex z yfinance"""
    tickers = [ticker] + list(CONFIG['MACRO'].values())
    try:
        df = yf.download(tickers, period=CONFIG['LOOKBACK'], interval="1d", progress=False)
    except Exception as e:
        st.error(f"Error downloading data: {e}")
        return pd.DataFrame()
    
    # Obsługa struktury danych (Flattening)
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
             st.error("Data mapping error. Check tickers or API.")
             return pd.DataFrame()
    else:
        # Fallback dla prostej struktury
        data = df
        
    data = data.ffill().dropna()
    return data

@st.cache_data
def run_quant_engine(df):
    """Główny silnik obliczeniowy: Feature Eng -> GMM -> XGBoost -> Monte Carlo"""
    
    # --- A. Feature Engineering ---
    df['Log_Ret'] = np.log(df['Close'] / df['Close'].shift(1))
    df['Vol_20'] = df['Log_Ret'].rolling(20).std()
    
    # Momentum & RSI
    for w in [5, 14, 20]:
        df[f'Mom_{w}'] = df['Close'].pct_change(w)
        # RSI Calculation
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(w).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(w).mean()
        rs = gain / loss
        df[f'RSI_{w}'] = 100 - (100 / (1 + rs))

    df.dropna(inplace=True)

    if df.empty:
        return None

    # --- B. Regime Detection (GMM) ---
    # Używamy zmienności i zwrotów do klastrowania
    X_regime = df[['Log_Ret', 'Vol_20']].values
    gmm = GaussianMixture(n_components=3, random_state=42).fit(X_regime)
    regimes = gmm.predict(X_regime)
    
    # Sortowanie reżimów po średniej zmienności (0=Low, 2=High)
    means = gmm.means_[:, 1]
    sorted_idx = np.argsort(means)
    regime_map = {sorted_idx[0]: 'LOW VOL (Green)', sorted_idx[1]: 'NORMAL (Orange)', sorted_idx[2]: 'HIGH VOL (Red)'}
    current_regime = regime_map[regimes[-1]]

    # --- C. AI Signal (XGBoost) ---
    # Target: Cena za 3 dni > Cena dzisiaj + spread (0.0010)
    df['Target'] = (df['Close'].shift(-3) / df['Close'] - 1 > 0.0010).astype(int)
    
    features = [c for c in df.columns if 'Mom' in c or 'Vol' in c or 'RSI' in c]
    # Usuwamy ostatnie wiersze bez targetu do treningu
    X = df[features].iloc[:-3] 
    y = df['Target'].iloc[:-3]
    
    model = xgb.XGBClassifier(**CONFIG['BEST_PARAMS'], n_estimators=100, n_jobs=-1)
    model.fit(X, y)
    
    # Predykcja na ostatnim wierszu
    last_row_features = df[features].iloc[[-1]]
    prob_up = model.predict_proba(last_row_features)[0][1]
    
    # --- D. Monte Carlo (Student-t Fat Tails) ---
    last_price = df['Close'].iloc[-1]
    vol_ann = df['Vol_20'].iloc[-1] * np.sqrt(252)
    sims = 1000
    # Rozkład t-Studenta (df=3) lepiej modeluje "grube ogony" niż rozkład normalny
    t_dist = t.rvs(df=3, size=sims)
    shocks = t_dist * (vol_ann / np.sqrt(252))
    mc_paths = last_price * np.exp(shocks) # Prognoza na 1 dzień
    
    support = np.percentile(mc_paths, 5)   # 95% pewności, że nie spadnie niżej
    resistance = np.percentile(mc_paths, 95) # 95% pewności, że nie wzrośnie wyżej

    return {
        'price': last_price,
        'regime': current_regime,
        'prob_up': prob_up,
        'support': support,
        'resistance': resistance,
        'df': df,
        'vol_ann': vol_ann
    }

# --- 3. UI & SIDEBAR ---
st.sidebar.header("🎛️ Control Panel")
symbol = st.sidebar.text_input("Asset", "EURUSD=X")
if st.sidebar.button("🚀 RUN ANALYSIS", type="primary"):
    
    with st.spinner("Processing Quantum Data..."):
        data = load_data(symbol)
        
        if not data.empty:
            res = run_quant_engine(data)
            
            if res:
                # --- 4. MAIN DASHBOARD (HUD) ---
                
                # Logika Sygnału
                signal = "WAIT"
                delta_color = "off"
                
                # Progi decyzyjne
                if res['prob_up'] > 0.60: 
                    signal = "STRONG BUY"
                    delta_color = "normal"
                elif res['prob_up'] < 0.40: 
                    signal = "STRONG SELL"
                    delta_color = "inverse"
                else:
                    signal = "NEUTRAL / RANGE"

                # Wyświetlanie Metryk (HUD)
                c1, c2, c3, c4 = st.columns(4)
                with c1: st.metric("PRICE", f"{res['price']:.5f}")
                with c2: st.metric("SIGNAL", signal, delta=f"{res['prob_up']:.1%}", delta_color=delta_color)
                with c3: st.metric("REGIME", res['regime'])
                with c4: st.metric("RANGE (MC)", f"{(res['resistance']-res['support'])*10000:.0f} pips")

                # --- 5. INTERACTIVE CHART (PLOTLY) ---
                fig = go.Figure()

                # Świece
                plot_df = res['df'].tail(120) # Pokaż ostatnie 120 dni
                fig.add_trace(go.Candlestick(
                    x=plot_df.index,
                    open=plot_df['Open'], high=plot_df['High'],
                    low=plot_df['Low'], close=plot_df['Close'],
                    name='Price'
                ))

                # Monte Carlo Cones (Projekcja)
                last_date = plot_df.index[-1]
                next_date = last_date + pd.Timedelta(days=1)
                
                # Linie wsparcia i oporu
                fig.add_trace(go.Scatter(
                    x=[last_date, next_date], y=[res['price'], res['resistance']],
                    mode='lines', line=dict(color='red', dash='dot', width=2), name='Resistance (95%)'
                ))
                
                fig.add_trace(go.Scatter(
                    x=[last_date, next_date], y=[res['price'], res['support']],
                    mode='lines', line=dict(color='green', dash='dot', width=2), name='Support (5%)'
                ))

                fig.update_layout(
                    title='Market Structure & Quantum Zones', 
                    height=600, 
                    template='plotly_dark',
                    xaxis_rangeslider_visible=False,
                    margin=dict(l=20, r=20, t=50, b=20)
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # --- 6. DETAILS TABS ---
                t1, t2 = st.tabs(["📊 Macro & Risk", "🧠 AI Diagnostics"])
                
                with t1:
                    c_vol, c_var = st.columns(2)
                    c_vol.info(f"Annualized Volatility: **{res['vol_ann']*100:.2f}%**")
                    c_var.warning("Macro Correlations coming in v29 (Data Feed Required)")
                    
                with t2:
                    col1, col2 = st.columns(2)
                    # FIX: Rzutowanie na float, aby uniknąć błędu StreamlitAPIException
                    prob_val = float(res['prob_up'])
                    col1.progress(prob_val, text=f"Bullish Probability: {prob_val:.1%}")
                    
                    col2.markdown(f"""
                    **AI Forecast Levels:**
                    * 🔴 Resistance: `{res['resistance']:.5f}`
                    * 🟢 Support: `{res['support']:.5f}`
                    """)
            else:
                st.error("Not enough data to calculate metrics.")
        else:
            st.error("Failed to load data. Check ticker symbol.")
else:
    st.info("👈 Select settings in the sidebar and click **RUN ANALYSIS** to start.")
