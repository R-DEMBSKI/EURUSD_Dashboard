import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import xgboost as xgb
import plotly.graph_objects as go
import plotly.express as px
from sklearn.mixture import GaussianMixture
from scipy.stats import t, zscore
from statsmodels.tsa.stattools import adfuller
import warnings

# --- 1. KONFIGURACJA STRONY ---
st.set_page_config(
    page_title="QUANT LAB | EURUSD Institutional Terminal",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="expanded"
)

warnings.filterwarnings("ignore")

# --- CUSTOM CSS (Bloomberg Style) ---
st.markdown("""
<style>
    .stApp { background-color: #0e1117; }
    
    /* HUD Metrics */
    div[data-testid="stMetricValue"] { 
        font-size: 24px; 
        color: #00ff00; 
        font-family: 'Courier New', monospace; 
        font-weight: bold;
    }
    div[data-testid="stMetricLabel"] { 
        font-size: 14px; 
        color: #888; 
    }
    
    /* Tables */
    .stDataFrame { border: 1px solid #333; }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] { gap: 24px; }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #0e1117;
        border-radius: 4px 4px 0px 0px;
        gap: 1px;
        padding-top: 10px;
        padding-bottom: 10px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #1e2130;
        color: #00ff00;
        border-bottom: 2px solid #00ff00;
    }
</style>
""", unsafe_allow_html=True)

# --- 2. ADVANCED CONFIG ---
CONFIG = {
    'TARGET': 'EURUSD=X',
    # Dodajemy kluczowe aktywa dla FX: Indeks Dolara, Ropa (korelacja inflacyjna), Złoto
    'MACRO': {
        'US10Y': '^TNX',       # US Treasury Yield
        'DXY': 'DX-Y.NYB',     # Dollar Index (Inverse correlation)
        'VIX': '^VIX',         # Volatility
        'SPX': '^GSPC',        # Risk Sentiment
        'GOLD': 'GC=F'         # Precious Metals
    },
    'LOOKBACK': '2y',
}

# --- 3. MATHEMATICAL ENGINES ---

def calculate_hurst(series, lags=range(2, 20)):
    """Oblicza Hurst Exponent - kluczowy dla określenia czy rynek trenduje (H>0.5) czy wraca do średniej (H<0.5)"""
    tau = [np.sqrt(np.std(np.subtract(series[lag:], series[:-lag]))) for lag in lags]
    poly = np.polyfit(np.log(lags), np.log(tau), 1)
    return poly[0] * 2.0

def kalman_filter(data, Q=1e-5, R=0.01):
    """Prosty jednowymiarowy Filtr Kalmana do wygładzania ceny (lepszy niż MA)"""
    n_iter = len(data)
    sz = (n_iter,)
    xhat = np.zeros(sz)      # a posteriori estimate of x
    P = np.zeros(sz)         # a posteriori error estimate
    xhatminus = np.zeros(sz) # a priori estimate of x
    Pminus = np.zeros(sz)    # a priori error estimate
    K = np.zeros(sz)         # gain or blending factor

    xhat[0] = data.iloc[0]
    P[0] = 1.0

    for k in range(1, n_iter):
        # Time Update
        xhatminus[k] = xhat[k-1]
        Pminus[k] = P[k-1] + Q

        # Measurement Update
        K[k] = Pminus[k] / (Pminus[k] + R)
        xhat[k] = xhatminus[k] + K[k] * (data.iloc[k] - xhatminus[k])
        P[k] = (1 - K[k]) * Pminus[k]

    return xhat

# --- 4. DATA LOADER ---
@st.cache_data(ttl=3600)
def load_data(ticker):
    tickers = [ticker] + list(CONFIG['MACRO'].values())
    try:
        df = yf.download(tickers, period=CONFIG['LOOKBACK'], interval="1d", progress=False)
    except Exception as e:
        st.error(f"Data Feed Error: {e}")
        return pd.DataFrame()

    data = pd.DataFrame()
    # Robust multi-index handling
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
             if ticker in df['Close'].columns:
                 data['Close'] = df['Close'][ticker]
    else:
        data = df

    data = data.ffill().dropna()
    return data

# --- 5. QUANT ENGINE ---
@st.cache_data
def run_quant_engine(df):
    # A. Feature Engineering
    df['Log_Ret'] = np.log(df['Close'] / df['Close'].shift(1))
    df['Vol_20'] = df['Log_Ret'].rolling(20).std()
    
    # Kalman Filter (The "True" Price)
    df['Kalman'] = kalman_filter(df['Close'])
    
    # Hurst Exponent (Rolling 100 days)
    # Obliczamy tylko dla ostatniego okna, aby oszczędzić zasoby
    hurst_val = calculate_hurst(df['Close'].tail(100).values)
    
    # Z-Score Anomalies (vs 50MA)
    df['MA_50'] = df['Close'].rolling(50).mean()
    df['Z_Score'] = (df['Close'] - df['MA_50']) / df['Close'].rolling(50).std()
    
    # Macro Correlations (Rolling 30d)
    for m in CONFIG['MACRO'].keys():
        if m in df.columns:
            df[f'Corr_{m}'] = df['Log_Ret'].rolling(30).corr(df[m].pct_change())

    df.dropna(inplace=True)
    if df.empty: return None

    # B. Regime Detection (GMM)
    X_regime = df[['Log_Ret', 'Vol_20']].values
    gmm = GaussianMixture(n_components=3, random_state=42).fit(X_regime)
    regimes = gmm.predict(X_regime)
    
    # Dynamic Mapping (High Vol is usually negative returns in Equities, but mixed in FX)
    means = gmm.means_[:, 1] # Volatility dimension
    sorted_idx = np.argsort(means)
    regime_map = {
        sorted_idx[0]: 'LOW VOL (Trending)', 
        sorted_idx[1]: 'NEUTRAL', 
        sorted_idx[2]: 'HIGH VOL (Mean Rev)'
    }
    current_regime = regime_map[regimes[-1]]

    # C. XGBoost Signal
    df['Target'] = (df['Close'].shift(-3) / df['Close'] - 1 > 0.0010).astype(int)
    features = ['Vol_20', 'Z_Score'] + [c for c in df.columns if 'Corr_' in c]
    
    X = df[features].iloc[:-3]
    y = df['Target'].iloc[:-3]
    
    model = xgb.XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.05, n_jobs=-1)
    model.fit(X, y)
    
    last_row = df[features].iloc[[-1]]
    prob_up = model.predict_proba(last_row)[0][1]

    # D. Monte Carlo (Student-t)
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
        'df': df,
        'vol_ann': vol_ann
    }

# --- 6. UI LAYOUT ---
st.sidebar.header("🎛️ QUANT LAB")
symbol = st.sidebar.text_input("Asset Class", "EURUSD=X")

if st.sidebar.button("INITIALIZE SYSTEM", type="primary"):
    with st.spinner("Crunching Institutional Data..."):
        data = load_data(symbol)
        
        if not data.empty:
            res = run_quant_engine(data)
            
            if res:
                # --- HEADER METRICS ---
                prob = res['prob_up']
                signal = "LONG" if prob > 0.6 else "SHORT" if prob < 0.4 else "NEUTRAL"
                signal_color = "normal" if signal == "LONG" else "inverse" if signal == "SHORT" else "off"
                
                # Hurst Interpretation
                hurst_desc = "Trending" if res['hurst'] > 0.55 else "Mean Reverting" if res['hurst'] < 0.45 else "Random Walk"

                c1, c2, c3, c4 = st.columns(4)
                with c1: st.metric("PRICE", f"{res['price']:.5f}", delta=f"{(res['price'] - data['Close'].iloc[-2]):.5f}")
                with c2: st.metric("AI SIGNAL", signal, delta=f"{prob:.1%}", delta_color=signal_color)
                with c3: st.metric("REGIME", res['regime'])
                with c4: st.metric("HURST EXP", f"{res['hurst']:.2f}", delta=hurst_desc, delta_color="off")

                # --- TABS FOR DEEP DIVE ---
                tab_main, tab_macro, tab_anom = st.tabs(["📈 Market Structure", "🌍 Macro & Correl", "⚠️ Anomalies"])
                
                with tab_main:
                    # Plotly Chart with Kalman & Monte Carlo
                    fig = go.Figure()
                    plot_df = res['df'].tail(150)
                    
                    # Candlesticks
                    fig.add_trace(go.Candlestick(
                        x=plot_df.index, open=plot_df['Open'], high=plot_df['High'],
                        low=plot_df['Low'], close=plot_df['Close'], name='Price'
                    ))
                    
                    # Kalman Filter (Yellow Line - Institutional Trend)
                    fig.add_trace(go.Scatter(
                        x=plot_df.index, y=plot_df['Kalman'], 
                        mode='lines', line=dict(color='yellow', width=2), name='Kalman Filter'
                    ))
                    
                    # MC Projections
                    next_date = plot_df.index[-1] + pd.Timedelta(days=1)
                    fig.add_trace(go.Scatter(x=[plot_df.index[-1], next_date], y=[res['price'], res['resistance']], 
                                             line=dict(color='red', dash='dot'), name='Res (95%)'))
                    fig.add_trace(go.Scatter(x=[plot_df.index[-1], next_date], y=[res['price'], res['support']], 
                                             line=dict(color='#00ff00', dash='dot'), name='Supp (5%)'))

                    fig.update_layout(height=600, template='plotly_dark', title="Institutional Price Action & Kalman Filter")
                    st.plotly_chart(fig, use_container_width=True)

                with tab_macro:
                    st.markdown("### 🔗 Macro Correlations (Rolling 30-Day)")
                    # Heatmap of Correlations
                    corr_cols = [c for c in res['df'].columns if 'Corr_' in c]
                    if corr_cols:
                        curr_corr = res['df'][corr_cols].iloc[-1].sort_values()
                        
                        # Visual Bar Chart for Correlations
                        fig_corr = px.bar(
                            x=curr_corr.values, y=[c.replace('Corr_', '') for c in curr_corr.index],
                            orientation='h', color=curr_corr.values, color_continuous_scale='RdBu',
                            range_color=[-1, 1], title="EURUSD Correlation Drivers"
                        )
                        fig_corr.update_layout(template='plotly_dark')
                        st.plotly_chart(fig_corr, use_container_width=True)
                        
                        st.info("💡 **Insight:** Silna negatywna korelacja z DXY (Indeks Dolara) potwierdza klasyczny ruch Risk-On/Off. Jeśli korelacja z GOLD słabnie, EURUSD jest sterowany czysto przez stopy procentowe.")

                with tab_anom:
                    st.markdown("### ⚠️ Z-Score & Statistical Anomalies")
                    # Z-Score Chart
                    fig_z = go.Figure()
                    fig_z.add_trace(go.Scatter(x=plot_df.index, y=plot_df['Z_Score'], fill='tozeroy', 
                                              line=dict(color='cyan'), name='Z-Score (Deviation)'))
                    fig_z.add_hline(y=2.0, line_color='red', line_dash='dash', annotation_text="Overbought (2σ)")
                    fig_z.add_hline(y=-2.0, line_color='green', line_dash='dash', annotation_text="Oversold (-2σ)")
                    fig_z.update_layout(height=400, template='plotly_dark', title="Statistical Mean Reversion Pressure")
                    st.plotly_chart(fig_z, use_container_width=True)
                    
                    if abs(res['df']['Z_Score'].iloc[-1]) > 2:
                        st.error("🚨 **CRITICAL ANOMALY:** Cena jest odchylona o ponad 2 odchylenia standardowe. Statystycznie wysokie prawdopodobieństwo powrotu do średniej (Mean Reversion).")
                    else:
                        st.success("✅ Market Conditions Normal. Brak ekstremalnych anomalii statystycznych.")

            else:
                st.error("Insufficient Data.")
else:
    st.info("System Ready. Waiting for User Input.")
