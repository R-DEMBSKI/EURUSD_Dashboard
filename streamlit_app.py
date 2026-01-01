import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import xgboost as xgb
import plotly.graph_objects as go
import plotly.express as px
from sklearn.linear_model import LinearRegression
from scipy.stats import t
import warnings
import time

# --- 1. KONFIGURACJA STRONY ---
st.set_page_config(
    page_title="QUANT LAB | Institutional MTF Terminal",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Wyciszenie ostrzeżeń
warnings.filterwarnings("ignore")

# --- CUSTOM CSS ---
st.markdown("""
<style>
    .stApp { background-color: #0e1117; }
    /* HUD */
    div[data-testid="stMetricValue"] { 
        font-size: 24px; color: #00ff00; font-family: 'Courier New', monospace; font-weight: bold;
    }
    div[data-testid="stMetricLabel"] { font-size: 13px; color: #888; }
    
    /* Tables & Tabs */
    .stTabs [aria-selected="true"] {
        background-color: #1e2130; color: #00ff00; border-bottom: 2px solid #00ff00;
    }
    /* Matrix Box */
    .matrix-box {
        padding: 10px; border-radius: 5px; text-align: center; font-weight: bold; margin-bottom: 10px; font-family: 'Courier New';
    }
</style>
""", unsafe_allow_html=True)

# --- 2. CONFIG ---
CONFIG = {
    'TARGET': 'EURUSD=X',
    'MACRO': {
        'US10Y': '^TNX', 'DXY': 'DX-Y.NYB', 'VIX': '^VIX', 'SPX': '^GSPC', 'GOLD': 'GC=F'
    },
    'LOOKBACK': '730d', # 2 lata historii dla stabilności
}

# --- 3. MATH ENGINES ---
def garman_klass_volatility(df):
    """Zaawansowany estymator zmienności (High/Low/Open/Close)"""
    # Unikamy zer i ujemnych wartości w logarytmach
    try:
        log_hl = np.log(df['High'] / df['Low'])**2
        log_co = np.log(df['Close'] / df['Open'])**2
        return np.sqrt(0.5 * log_hl - (2 * np.log(2) - 1) * log_co)
    except:
        return df['Close'].pct_change().rolling(20).std()

def calculate_hurst(series):
    """Oblicza Hurst Exponent (Mean Rev < 0.5 < Trending)"""
    try:
        lags = range(2, 20)
        tau = [np.sqrt(np.std(np.subtract(series[lag:], series[:-lag]))) for lag in lags]
        poly = np.polyfit(np.log(lags), np.log(tau), 1)
        return poly[0] * 2.0
    except:
        return 0.5

def kalman_filter(data, Q=1e-5, R=0.01):
    """Filtr Kalmana - 'Prawdziwa' cena instrumentu"""
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

# --- 4. DATA LOADER (MTF ENABLED) ---
@st.cache_data(ttl=3600)
def load_data(ticker):
    tickers = [ticker] + list(CONFIG['MACRO'].values())
    
    # 1. Pobranie danych DZIENNYCH (Daily) - Próba retry
    df_d = pd.DataFrame()
    for _ in range(3):
        try:
            raw = yf.download(tickers, period="2y", interval="1d", progress=False)
            if not raw.empty:
                df_d = raw
                break
        except:
            time.sleep(1)
            
    # 2. Pobranie danych TYGODNIOWYCH (Weekly) - dla MTF Context
    df_w = pd.DataFrame()
    try:
        df_w = yf.download(ticker, period="2y", interval="1wk", progress=False)
    except:
        pass

    # Cleaning Daily
    clean_d = pd.DataFrame()
    if not df_d.empty:
        try:
            # Obsługa złożonego MultiIndex z yfinance
            if isinstance(df_d.columns, pd.MultiIndex):
                # Price Data
                if ticker in df_d['Close'].columns:
                    clean_d['Close'] = df_d['Close'][ticker]
                    clean_d['High'] = df_d['High'][ticker]
                    clean_d['Low'] = df_d['Low'][ticker]
                    clean_d['Open'] = df_d['Open'][ticker]
                # Macro Data
                for key, val in CONFIG['MACRO'].items():
                    if val in df_d['Close'].columns:
                        clean_d[key] = df_d['Close'][val]
            else:
                clean_d = df_d # Prosta struktura
        except:
            return None, None
            
    # Cleaning Weekly
    clean_w = pd.DataFrame()
    if not df_w.empty:
        try:
            if isinstance(df_w.columns, pd.MultiIndex):
                clean_w['Close'] = df_w['Close'][ticker]
            else:
                clean_w['Close'] = df_w['Close']
        except:
            pass

    return clean_d.ffill().dropna(), clean_w.ffill().dropna()

# --- 5. QUANT ENGINE ---
@st.cache_data
def run_quant_engine(df, df_w):
    if len(df) < 50: return None
    
    # --- A. Technicals ---
    df['Log_Ret'] = np.log(df['Close'] / df['Close'].shift(1))
    
    # Garman-Klass Volatility (Institutional Grade)
    df['Vol_GK'] = garman_klass_volatility(df)
    
    # Kalman
    df['Kalman'] = kalman_filter(df['Close'])
    
    # Hurst
    hurst_val = calculate_hurst(df['Close'].tail(100).values)
    
    # --- B. MTF Logic (Weekly Context) ---
    # Obliczamy Weekly Trend (SMA 20 na tygodniowym)
    weekly_trend = "NEUTRAL"
    if not df_w.empty:
        df_w['MA_20'] = df_w['Close'].rolling(20).mean()
        if len(df_w) > 20:
            last_w_close = df_w['Close'].iloc[-1]
            last_w_ma = df_w['MA_20'].iloc[-1]
            weekly_trend = "BULLISH" if last_w_close > last_w_ma else "BEARISH"

    # Daily Trend (SMA 50 na dziennym)
    df['MA_50'] = df['Close'].rolling(50).mean()
    daily_trend = "BULLISH" if df['Close'].iloc[-1] > df['MA_50'].iloc[-1] else "BEARISH"

    # Confluence Check
    confluence = "MIXED / CHOPPY"
    if weekly_trend == "BULLISH" and daily_trend == "BULLISH": confluence = "STRONG UPTREND"
    elif weekly_trend == "BEARISH" and daily_trend == "BEARISH": confluence = "STRONG DOWNTREND"
    elif weekly_trend == "BULLISH" and daily_trend == "BEARISH": confluence = "PULLBACK (Buy Dip)"
    elif weekly_trend == "BEARISH" and daily_trend == "BULLISH": confluence = "RELIEF RALLY (Sell Rip)"
    
    # --- C. Fair Value Model ---
    try:
        macro_cols = [c for c in df.columns if c in ['DXY', 'US10Y', 'SPX']]
        if macro_cols:
            # Regresja liniowa na ostatnich 60 dniach
            window = 60
            if len(df) > window:
                model = LinearRegression()
                X = df[macro_cols].iloc[-window:]
                y = df['Close'].iloc[-window:]
                model.fit(X, y)
                # Predykcja na całym zbiorze
                df['Fair_Value'] = model.predict(df[macro_cols])
                df['FV_Gap'] = df['Close'] - df['Fair_Value']
            else:
                df['FV_Gap'] = 0.0
        else:
            df['FV_Gap'] = 0.0
    except:
        df['FV_Gap'] = 0.0

    # --- D. Z-Score ---
    df['Z_Score'] = (df['Close'] - df['MA_50']) / df['Close'].rolling(50).std()

    # --- E. Correlations ---
    for m in CONFIG['MACRO'].keys():
        if m in df.columns:
            df[f'Corr_{m}'] = df['Log_Ret'].rolling(30).corr(df[m].pct_change())

    # --- F. AI Model (XGBoost) ---
    df['Target'] = (df['Close'].shift(-3) / df['Close'] - 1 > 0.0010).astype(int)
    features = ['Vol_GK', 'Z_Score'] + [c for c in df.columns if 'Corr_' in c]
    valid_features = [f for f in features if f in df.columns]
    
    model = xgb.XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.05, n_jobs=-1)
    # Trenujemy na danych bez ostatnich 3 dni (brak targetu)
    model.fit(df[valid_features].iloc[:-3], df['Target'].iloc[:-3])
    prob_up = model.predict_proba(df[valid_features].iloc[[-1]])[0][1]

    # --- G. Monte Carlo ---
    last_price = df['Close'].iloc[-1]
    vol_ann = df['Vol_GK'].iloc[-1] * np.sqrt(252)
    # Student-t distribution for fat tails
    t_dist = t.rvs(df=3, size=1000) * (vol_ann / np.sqrt(252))
    mc_paths = last_price * np.exp(t_dist)

    return {
        'price': last_price,
        'kalman': df['Kalman'].iloc[-1],
        'weekly_trend': weekly_trend,
        'daily_trend': daily_trend,
        'confluence': confluence,
        'hurst': hurst_val,
        'prob_up': prob_up,
        'fv_gap': df['FV_Gap'].iloc[-1],
        'df': df,
        'support': np.percentile(mc_paths, 5),
        'resistance': np.percentile(mc_paths, 95)
    }

# --- 6. UI ---
st.sidebar.header("🎛️ QUANT LAB PRO")
symbol = st.sidebar.text_input("Asset", "EURUSD=X")

if st.sidebar.button("INITIALIZE SYSTEM", type="primary"):
    with st.spinner("Processing Multi-Timeframe Matrix..."):
        df_d, df_w = load_data(symbol)
        
        if df_d is not None and not df_d.empty:
            res = run_quant_engine(df_d, df_w)
            if res:
                # --- HUD ---
                prob = res['prob_up']
                signal = "LONG" if prob > 0.6 else "SHORT" if prob < 0.4 else "NEUTRAL"
                color = "normal" if signal == "LONG" else "inverse" if signal == "SHORT" else "off"
                
                c1, c2, c3, c4 = st.columns(4)
                with c1: st.metric("PRICE", f"{res['price']:.5f}")
                with c2: st.metric("AI CONFIDENCE", f"{prob:.1%}", delta=signal, delta_color=color)
                with c3: st.metric("FV GAP", f"{res['fv_gap']:.4f}", delta="Fair Value Diff", delta_color="inverse")
                with c4: st.metric("HURST EXP", f"{res['hurst']:.2f}", delta="Regime", delta_color="off")

                # --- MTF MATRIX ---
                st.markdown("---")
                st.markdown("#### 🌐 Institutional MTF Matrix")
                m1, m2, m3 = st.columns(3)
                
                # Dynamiczne kolory dla Matrixa
                def get_color(trend):
                    return "#00ff00" if trend == "BULLISH" else "#ff0000" if trend == "BEARISH" else "#888"
                
                m1.markdown(f"<div class='matrix-box' style='border: 1px solid {get_color(res['weekly_trend'])}; color: {get_color(res['weekly_trend'])}'>WEEKLY (Macro)<br>{res['weekly_trend']}</div>", unsafe_allow_html=True)
                m2.markdown(f"<div class='matrix-box' style='border: 1px solid {get_color(res['daily_trend'])}; color: {get_color(res['daily_trend'])}'>DAILY (Tactical)<br>{res['daily_trend']}</div>", unsafe_allow_html=True)
                
                conf_color = "#00ff00" if "UP" in res['confluence'] else "#ff0000" if "DOWN" in res['confluence'] else "#ffa500"
                m3.markdown(f"<div class='matrix-box' style='border: 1px solid {conf_color}; color: {conf_color}'>CONFLUENCE<br>{res['confluence']}</div>", unsafe_allow_html=True)

                # --- CHARTS & TABS ---
                t1, t2, t3 = st.tabs(["📈 Price & Fair Value", "📊 Volatility & Z-Score", "🌍 Macro Correlations"])
                
                with t1:
                    fig = go.Figure()
                    pdf = res['df'].tail(150)
                    
                    # Świece
                    fig.add_trace(go.Candlestick(x=pdf.index, open=pdf['Open'], high=pdf['High'], low=pdf['Low'], close=pdf['Close'], name='Price'))
                    
                    # Kalman
                    fig.add_trace(go.Scatter(x=pdf.index, y=pdf['Kalman'], line=dict(color='yellow', width=2), name='Kalman Trend'))
                    
                    # Fair Value Cloud
                    if 'Fair_Value' in pdf.columns and pdf['Fair_Value'].iloc[-1] != 0:
                         fig.add_trace(go.Scatter(x=pdf.index, y=pdf['Fair_Value'], line=dict(color='orange', dash='dash', width=1), name='Fair Value'))

                    fig.update_layout(height=600, template='plotly_dark', title="Institutional Price Action", margin=dict(l=0,r=0,t=30,b=0))
                    st.plotly_chart(fig, use_container_width=True)

                with t2:
                    c_vol, c_z = st.columns(2)
                    with c_vol:
                        fig_v = px.line(pdf, x=pdf.index, y='Vol_GK', title="Garman-Klass Volatility (Risk)")
                        fig_v.update_traces(line_color='#ff00ff') # Magenta line
                        fig_v.update_layout(height=350, template='plotly_dark')
                        st.plotly_chart(fig_v, use_container_width=True)
                        
                    with c_z:
                        fig_z = go.Figure()
                        fig_z.add_trace(go.Scatter(x=pdf.index, y=pdf['Z_Score'], fill='tozeroy', line=dict(color='cyan')))
                        fig_z.add_hline(y=2.0, line_color='red', line_dash='dash')
                        fig_z.add_hline(y=-2.0, line_color='green', line_dash='dash')
                        fig_z.update_layout(height=350, template='plotly_dark', title="Z-Score (Reversion)")
                        st.plotly_chart(fig_z, use_container_width=True)

                with t3:
                    corr_cols = [c for c in res['df'].columns if 'Corr_' in c]
                    if corr_cols:
                        curr = res['df'][corr_cols].iloc[-1].sort_values()
                        fig = px.bar(x=curr.values, y=[c.replace('Corr_', '') for c in curr.index], orientation='h', title="Real-time Macro Drivers")
                        fig.update_layout(template='plotly_dark')
                        st.plotly_chart(fig, use_container_width=True)

            else:
                st.error("Quant Engine Failed. Not enough data.")
        else:
            st.error("Data Feed Disconnected. Please try again.")
else:
    st.info("System Ready. Waiting for connection...")
