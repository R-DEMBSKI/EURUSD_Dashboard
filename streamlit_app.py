import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import xgboost as xgb
import plotly.graph_objects as go
import plotly.express as px
from sklearn.mixture import GaussianMixture
from scipy.stats import t
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
    'MACRO': {
        'US10Y': '^TNX',       # US Treasury Yield
        'DXY': 'DX-Y.NYB',     # Dollar Index
        'VIX': '^VIX',         # Volatility
        'SPX': '^GSPC',        # Risk Sentiment
        'GOLD': 'GC=F'         # Precious Metals
    },
    'LOOKBACK': '2y',
}

# --- 3. MATHEMATICAL ENGINES ---

def calculate_hurst(series, lags=range(2, 20)):
    """Oblicza Hurst Exponent - określa czy rynek trenduje"""
    # Sprawdzenie czy mamy wystarczająco dużo danych
    if len(series) < 20:
        return 0.5
        
    tau = []
    valid_lags = []
    for lag in lags:
        if len(series) > lag:
            diff = np.subtract(series[lag:], series[:-lag])
            tau.append(np.sqrt(np.std(diff)))
            valid_lags.append(lag)
            
    if len(valid_lags) < 2:
        return 0.5
        
    try:
        poly = np.polyfit(np.log(valid_lags), np.log(tau), 1)
        return poly[0] * 2.0
    except:
        return 0.5

def kalman_filter(data, Q=1e-5, R=0.01):
    """Filtr Kalmana do wygładzania ceny"""
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

# --- 4. DATA LOADER ---
@st.cache_data(ttl=3600)
def load_data(ticker):
    tickers = [ticker] + list(CONFIG['MACRO'].values())
    try:
        # Request data
        df = yf.download(tickers, period=CONFIG['LOOKBACK'], interval="1d", progress=False)
    except Exception as e:
        st.error(f"Data Feed Error: {e}")
        return pd.DataFrame()

    data = pd.DataFrame()
    
    # Obsługa MultiIndex (Nowe yfinance zwraca skomplikowane nagłówki)
    if isinstance(df.columns, pd.MultiIndex):
        try:
            # Próba wyciągnięcia danych dla głównego tickera
            # Sprawdzamy czy ticker jest na pierwszym poziomie
            if ticker in df.columns.get_level_values(1): # Struktura: Price, Ticker
                 data['Close'] = df['Close'][ticker]
                 data['High'] = df['High'][ticker]
                 data['Low'] = df['Low'][ticker]
                 data['Open'] = df['Open'][ticker]
            # Czasami struktura jest odwrotna lub płaska
            elif ticker in df['Close'].columns:
                 data['Close'] = df['Close'][ticker]
                 data['High'] = df['High'][ticker]
                 data['Low'] = df['Low'][ticker]
                 data['Open'] = df['Open'][ticker]
            
            # Pobieranie MACRO
            for key, val in CONFIG['MACRO'].items():
                if val in df['Close'].columns:
                    data[key] = df['Close'][val]
                
        except Exception as e:
             # Fallback: Jeśli struktura jest inna, bierzemy po prostu Close
             if ticker in df['Close'].columns:
                 data['Close'] = df['Close'][ticker]
                 data['High'] = df['High'][ticker]
                 data['Low'] = df['Low'][ticker]
                 data['Open'] = df['Open'][ticker]
    else:
        # Stara struktura yfinance
        data = df

    # Finalne czyszczenie
    data = data.ffill().dropna()
    return data

# --- 5. QUANT ENGINE ---
@st.cache_data
def run_quant_engine(df):
    # --- A. Feature Engineering ---
    df['Log_Ret'] = np.log(df['Close'] / df['Close'].shift(1))
    df['Vol_20'] = df['Log_Ret'].rolling(20).std()
    
    # Kalman Filter
    try:
        df['Kalman'] = kalman_filter(df['Close'])
    except:
        df['Kalman'] = df['Close'].rolling(5).mean() # Fallback
    
    # Hurst Exponent (na ostatnich 100 świecach)
    hurst_val = 0.5
    try:
        if len(df) > 100:
            hurst_val = calculate_hurst(df['Close'].tail(100).values)
    except:
        pass
    
    # Z-Score Anomalies
    df['MA_50'] = df['Close'].rolling(50).mean()
    df['Z_Score'] = (df['Close'] - df['MA_50']) / df['Close'].rolling(50).std()
    
    # Macro Correlations
    for m in CONFIG['MACRO'].keys():
        if m in df.columns:
            df[f'Corr_{m}'] = df['Log_Ret'].rolling(30).corr(df[m].pct_change())

    df.dropna(inplace=True)
    if df.empty: return None

    # --- B. Regime Detection ---
    X_regime = df[['Log_Ret', 'Vol_20']].values
    
    try:
        gmm = GaussianMixture(n_components=3, random_state=42).fit(X_regime)
        regimes = gmm.predict(X_regime)
        means = gmm.means_[:, 1]
        sorted_idx = np.argsort(means)
        regime_map = {
            sorted_idx[0]: 'LOW VOL (Trending)', 
            sorted_idx[1]: 'NEUTRAL', 
            sorted_idx[2]: 'HIGH VOL (Mean Rev)'
        }
        current_regime = regime_map[regimes[-1]]
    except:
        current_regime = "N/A"

    # --- C. AI Signal (XGBoost) ---
    df['Target'] = (df['Close'].shift(-3) / df['Close'] - 1 > 0.0010).astype(int)
    features = ['Vol_20', 'Z_Score'] + [c for c in df.columns if 'Corr_' in c]
    
    # Walidacja cech
    valid_features = [f for f in features if f in df.columns]
    
    X = df[valid_features].iloc[:-3]
    y = df['Target'].iloc[:-3]
    
    prob_up = 0.5
    if not X.empty and len(X) > 50:
        model = xgb.XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.05, n_jobs=-1)
        model.fit(X, y)
        last_row = df[valid_features].iloc[[-1]]
        prob_up = model.predict_proba(last_row)[0][1]

    # --- D. Monte Carlo ---
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
    with st.spinner("Processing Institutional Data..."):
        data = load_data(symbol)
        
        if not data.empty:
            res = run_quant_engine(data)
            
            if res:
                # --- HEADER METRICS ---
                prob = res['prob_up']
                signal = "LONG" if prob > 0.6 else "SHORT" if prob < 0.4 else "NEUTRAL"
                signal_color = "normal" if signal == "LONG" else "inverse" if signal == "SHORT" else "off"
                
                hurst_desc = "Trending" if res['hurst'] > 0.55 else "Mean Reverting" if res['hurst'] < 0.45 else "Random Walk"

                c1, c2, c3, c4 = st.columns(4)
                with c1: st.metric("PRICE", f"{res['price']:.5f}")
                with c2: st.metric("AI SIGNAL", signal, delta=f"{prob:.1%}", delta_color=signal_color)
                with c3: st.metric("REGIME", res['regime'])
                with c4: st.metric("HURST EXP", f"{res['hurst']:.2f}", delta=hurst_desc, delta_color="off")

                # --- TABS ---
                tab_main, tab_macro, tab_anom = st.tabs(["📈 Market Structure", "🌍 Macro & Correl", "⚠️ Anomalies"])
                
                with tab_main:
                    fig = go.Figure()
                    plot_df = res['df'].tail(150)
                    
                    # Candlesticks
                    fig.add_trace(go.Candlestick(
                        x=plot_df.index, open=plot_df['Open'], high=plot_df['High'],
                        low=plot_df['Low'], close=plot_df['Close'], name='Price'
                    ))
                    
                    # Kalman Filter
                    fig.add_trace(go.Scatter(
                        x=plot_df.index, y=plot_df['Kalman'], 
                        mode='lines', line=dict(color='yellow', width=2), name='Kalman Filter'
                    ))
                    
                    # Projections
                    next_date = plot_df.index[-1] + pd.Timedelta(days=1)
                    fig.add_trace(go.Scatter(x=[plot_df.index[-1], next_date], y=[res['price'], res['resistance']], 
                                             line=dict(color='red', dash='dot'), name='Res (95%)'))
                    fig.add_trace(go.Scatter(x=[plot_df.index[-1], next_date], y=[res['price'], res['support']], 
                                             line=dict(color='#00ff00', dash='dot'), name='Supp (5%)'))

                    fig.update_layout(height=600, template='plotly_dark', title="Institutional Price Action & Kalman Filter",
                                      margin=dict(l=20, r=20, t=50, b=20))
                    st.plotly_chart(fig, use_container_width=True)

                with tab_macro:
                    st.markdown("### 🔗 Macro Correlations (30-Day Rolling)")
                    corr_cols = [c for c in res['df'].columns if 'Corr_' in c]
                    if corr_cols:
                        curr_corr = res['df'][corr_cols].iloc[-1].sort_values()
                        fig_corr = px.bar(
                            x=curr_corr.values, y=[c.replace('Corr_', '') for c in curr_corr.index],
                            orientation='h', color=curr_corr.values, color_continuous_scale='RdBu',
                            range_color=[-1, 1], title="EURUSD Drivers"
                        )
                        fig_corr.update_layout(template='plotly_dark')
                        st.plotly_chart(fig_corr, use_container_width=True)
                    else:
                        st.info("Brak wystarczających danych makro do obliczenia korelacji.")

                with tab_anom:
                    st.markdown("### ⚠️ Z-Score & Statistical Anomalies")
                    fig_z = go.Figure()
                    fig_z.add_trace(go.Scatter(x=plot_df.index, y=plot_df['Z_Score'], fill='tozeroy', 
                                              line=dict(color='cyan'), name='Z-Score'))
                    fig_z.add_hline(y=2.0, line_color='red', line_dash='dash')
                    fig_z.add_hline(y=-2.0, line_color='green', line_dash='dash')
                    fig_z.update_layout(height=400, template='plotly_dark', title="Mean Reversion Pressure")
                    st.plotly_chart(fig_z, use_container_width=True)

            else:
                st.error("Insufficient Data - Market Closed or Ticker Invalid.")
        else:
            st.error("Failed to load data.")
else:
    st.info("System Ready. Awaiting Input.")
