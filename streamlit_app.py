import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import xgboost as xgb
import plotly.graph_objects as go
import plotly.express as px
from sklearn.linear_model import LinearRegression
from scipy.stats import t
from scipy.signal import argrelextrema
import warnings
import time

# --- 1. KONFIGURACJA STRONY ---
st.set_page_config(
    page_title="QUANT LAB | Smart Money Terminal",
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
    
    /* Tabs */
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
    'LOOKBACK': '730d', 
}

# --- 3. MATH ENGINES ---
def calculate_kelly(prob_win, win_loss_ratio=1.5):
    """Kryterium Kelly'ego dla wielkości pozycji (Money Management)"""
    # f = (p(b+1) - 1) / b
    # Zastosowanie Half-Kelly dla bezpieczeństwa (Standard w funduszach)
    q = 1 - prob_win
    f = (win_loss_ratio * prob_win - q) / win_loss_ratio
    return max(0, f) * 0.5 

def find_liquidity_levels(df, window=5):
    """Wykrywa fraktalne poziomy wsparcia i oporu (Liquidity Pools / Fractals)"""
    highs = df['High'].values
    lows = df['Low'].values
    
    # Lokalne maksima i minima (Fractals)
    max_idx = argrelextrema(highs, np.greater, order=window)[0]
    min_idx = argrelextrema(lows, np.less, order=window)[0]
    
    resistance_levels = highs[max_idx]
    support_levels = lows[min_idx]
    
    # Zwracamy tylko poziomy blisko obecnej ceny (ostatnie 3 miesiące)
    recent_limit = len(df) - 60
    
    rel_res = [x for i, x in enumerate(resistance_levels) if max_idx[i] > recent_limit]
    rel_sup = [x for i, x in enumerate(support_levels) if min_idx[i] > recent_limit]
    
    return rel_res, rel_sup

def garman_klass_volatility(df):
    try:
        log_hl = np.log(df['High'] / df['Low'])**2
        log_co = np.log(df['Close'] / df['Open'])**2
        return np.sqrt(0.5 * log_hl - (2 * np.log(2) - 1) * log_co)
    except:
        return df['Close'].pct_change().rolling(20).std()

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

# --- 4. DATA LOADER ---
@st.cache_data(ttl=3600)
def load_data(ticker):
    tickers = [ticker] + list(CONFIG['MACRO'].values())
    
    # Retry mechanism
    df_d = pd.DataFrame()
    for _ in range(3):
        try:
            raw = yf.download(tickers, period="2y", interval="1d", progress=False)
            if not raw.empty:
                df_d = raw
                break
        except:
            time.sleep(1)
            
    # Weekly Data for MTF
    df_w = pd.DataFrame()
    try:
        df_w = yf.download(ticker, period="2y", interval="1wk", progress=False)
    except:
        pass

    # Cleaning
    clean_d = pd.DataFrame()
    if not df_d.empty:
        try:
            if isinstance(df_d.columns, pd.MultiIndex):
                if ticker in df_d['Close'].columns:
                    clean_d['Close'] = df_d['Close'][ticker]
                    clean_d['High'] = df_d['High'][ticker]
                    clean_d['Low'] = df_d['Low'][ticker]
                    clean_d['Open'] = df_d['Open'][ticker]
                for key, val in CONFIG['MACRO'].items():
                    if val in df_d['Close'].columns:
                        clean_d[key] = df_d['Close'][val]
            else:
                clean_d = df_d
        except:
            return None, None
            
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
    
    # Technicals
    df['Log_Ret'] = np.log(df['Close'] / df['Close'].shift(1))
    df['Vol_GK'] = garman_klass_volatility(df)
    df['Kalman'] = kalman_filter(df['Close'])
    hurst_val = calculate_hurst(df['Close'].tail(100).values)
    
    # MTF Logic
    weekly_trend = "NEUTRAL"
    if not df_w.empty:
        df_w['MA_20'] = df_w['Close'].rolling(20).mean()
        if len(df_w) > 20:
            weekly_trend = "BULLISH" if df_w['Close'].iloc[-1] > df_w['MA_20'].iloc[-1] else "BEARISH"

    df['MA_50'] = df['Close'].rolling(50).mean()
    daily_trend = "BULLISH" if df['Close'].iloc[-1] > df['MA_50'].iloc[-1] else "BEARISH"

    confluence = "MIXED"
    if weekly_trend == "BULLISH" and daily_trend == "BULLISH": confluence = "STRONG UPTREND"
    elif weekly_trend == "BEARISH" and daily_trend == "BEARISH": confluence = "STRONG DOWNTREND"
    
    # Fair Value
    try:
        macro_cols = [c for c in df.columns if c in ['DXY', 'US10Y', 'SPX']]
        if macro_cols:
            window = 60
            if len(df) > window:
                model = LinearRegression()
                X = df[macro_cols].iloc[-window:]
                y = df['Close'].iloc[-window:]
                model.fit(X, y)
                df['Fair_Value'] = model.predict(df[macro_cols])
                df['FV_Gap'] = df['Close'] - df['Fair_Value']
            else: df['FV_Gap'] = 0.0
        else: df['FV_Gap'] = 0.0
    except: df['FV_Gap'] = 0.0

    # Z-Score & Correlations
    df['Z_Score'] = (df['Close'] - df['MA_50']) / df['Close'].rolling(50).std()
    for m in CONFIG['MACRO'].keys():
        if m in df.columns:
            df[f'Corr_{m}'] = df['Log_Ret'].rolling(30).corr(df[m].pct_change())

    # AI Model
    df['Target'] = (df['Close'].shift(-3) / df['Close'] - 1 > 0.0010).astype(int)
    features = ['Vol_GK', 'Z_Score'] + [c for c in df.columns if 'Corr_' in c]
    valid_features = [f for f in features if f in df.columns]
    
    model = xgb.XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.05, n_jobs=-1)
    model.fit(df[valid_features].iloc[:-3], df['Target'].iloc[:-3])
    prob_up = model.predict_proba(df[valid_features].iloc[[-1]])[0][1]

    # Monte Carlo & Levels
    last_price = df['Close'].iloc[-1]
    vol_ann = df['Vol_GK'].iloc[-1] * np.sqrt(252)
    t_dist = t.rvs(df=3, size=1000) * (vol_ann / np.sqrt(252))
    mc_paths = last_price * np.exp(t_dist)
    
    # Liquidity Levels (SMC)
    res_levels, sup_levels = find_liquidity_levels(df)

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
        'resistance': np.percentile(mc_paths, 95),
        'res_levels': res_levels,
        'sup_levels': sup_levels
    }

# --- 6. UI ---
st.sidebar.header("🎛️ QUANT LAB ELITE")
symbol = st.sidebar.text_input("Asset", "EURUSD=X")

# Sidebar - Risk Calculator
st.sidebar.markdown("---")
st.sidebar.subheader("💰 Smart Money Risk")
account_size = st.sidebar.number_input("Account Equity ($)", value=10000)
# Suwak ryzyka (Kelly fraction)
risk_per_trade = st.sidebar.slider("Risk Tolerance (Kelly Fraction)", 0.1, 1.0, 0.5)

if st.sidebar.button("INITIALIZE SYSTEM", type="primary"):
    with st.spinner("Analyzing Liquidity & Smart Money Flows..."):
        df_d, df_w = load_data(symbol)
        
        if df_d is not None and not df_d.empty:
            res = run_quant_engine(df_d, df_w)
            if res:
                # --- HUD ---
                prob = res['prob_up']
                signal = "LONG" if prob > 0.6 else "SHORT" if prob < 0.4 else "NEUTRAL"
                color = "normal" if signal == "LONG" else "inverse" if signal == "SHORT" else "off"
                
                # Kelly Calc
                # Jeśli sygnał jest Long, używamy prob, jeśli Short, używamy 1-prob
                win_prob = prob if signal == "LONG" else (1.0 - prob)
                
                # Obliczamy Kelly (Full)
                kelly_pct = calculate_kelly(win_prob) 
                # Skalujemy przez tolerancję ryzyka (np. Half Kelly)
                final_risk_pct = kelly_pct * risk_per_trade
                rec_position = account_size * final_risk_pct
                
                c1, c2, c3, c4 = st.columns(4)
                with c1: st.metric("PRICE", f"{res['price']:.5f}")
                with c2: st.metric("AI CONFIDENCE", f"{prob:.1%}", delta=signal, delta_color=color)
                with c3: st.metric("FV GAP", f"{res['fv_gap']:.4f}", delta="Mispricing", delta_color="inverse")
                with c4: st.metric("KELLY SIZE", f"${rec_position:.0f}", delta=f"{final_risk_pct:.1%} Risk")

                # --- MTF MATRIX ---
                st.markdown("---")
                m1, m2, m3 = st.columns(3)
                def get_color(trend): return "#00ff00" if trend == "BULLISH" else "#ff0000" if trend == "BEARISH" else "#888"
                
                m1.markdown(f"<div class='matrix-box' style='border: 1px solid {get_color(res['weekly_trend'])}; color: {get_color(res['weekly_trend'])}'>WEEKLY (Macro)<br>{res['weekly_trend']}</div>", unsafe_allow_html=True)
                m2.markdown(f"<div class='matrix-box' style='border: 1px solid {get_color(res['daily_trend'])}; color: {get_color(res['daily_trend'])}'>DAILY (Tactical)<br>{res['daily_trend']}</div>", unsafe_allow_html=True)
                conf_color = "#00ff00" if "UP" in res['confluence'] else "#ff0000" if "DOWN" in res['confluence'] else "#ffa500"
                m3.markdown(f"<div class='matrix-box' style='border: 1px solid {conf_color}; color: {conf_color}'>CONFLUENCE<br>{res['confluence']}</div>", unsafe_allow_html=True)

                # --- CHARTS ---
                t1, t2, t3 = st.tabs(["📊 SMC Liquidity", "🧠 Volatility & Z-Score", "🌍 Macro"])
                
                with t1:
                    fig = go.Figure()
                    pdf = res['df'].tail(150)
                    
                    # Candles
                    fig.add_trace(go.Candlestick(x=pdf.index, open=pdf['Open'], high=pdf['High'], low=pdf['Low'], close=pdf['Close'], name='Price'))
                    # Kalman
                    fig.add_trace(go.Scatter(x=pdf.index, y=pdf['Kalman'], line=dict(color='yellow', width=2), name='Kalman Trend'))
                    # Fair Value
                    if 'Fair_Value' in pdf.columns:
                         fig.add_trace(go.Scatter(x=pdf.index, y=pdf['Fair_Value'], line=dict(color='orange', dash='dash', width=1), name='Fair Value'))
                    
                    # Liquidity Levels (Smart Money Fractals)
                    # Pokazujemy tylko 3 ostatnie poziomy, aby nie zaśmiecać wykresu
                    for level in res['res_levels'][-3:]: 
                        fig.add_hline(y=level, line_color='red', line_width=1, line_dash='dot', annotation_text="Liquidity (Sell-Side)", annotation_position="top right")
                    for level in res['sup_levels'][-3:]: 
                        fig.add_hline(y=level, line_color='green', line_width=1, line_dash='dot', annotation_text="Liquidity (Buy-Side)", annotation_position="bottom right")

                    fig.update_layout(height=650, template='plotly_dark', title="Institutional Chart (SMC + Liquidity Pools)", margin=dict(l=0,r=0,t=30,b=0))
                    # FIX: Użycie width="stretch" zamiast use_container_width
                    st.plotly_chart(fig, use_container_width=True) 

                with t2:
                    c_vol, c_z = st.columns(2)
                    with c_vol:
                        fig_v = px.line(pdf, x=pdf.index, y='Vol_GK', title="Garman-Klass Volatility")
                        fig_v.update_traces(line_color='#ff00ff')
                        fig_v.update_layout(height=350, template='plotly_dark')
                        st.plotly_chart(fig_v, use_container_width=True)
                        
                    with c_z:
                        fig_z = go.Figure()
                        fig_z.add_trace(go.Scatter(x=pdf.index, y=pdf['Z_Score'], fill='tozeroy', line=dict(color='cyan')))
                        fig_z.add_hline(y=2.0, line_color='red', line_dash='dash')
                        fig_z.add_hline(y=-2.0, line_color='green', line_dash='dash')
                        fig_z.update_layout(height=350, template='plotly_dark', title="Z-Score Reversion")
                        st.plotly_chart(fig_z, use_container_width=True)

                with t3:
                    corr_cols = [c for c in res['df'].columns if 'Corr_' in c]
                    if corr_cols:
                        curr = res['df'][corr_cols].iloc[-1].sort_values()
                        fig = px.bar(x=curr.values, y=[c.replace('Corr_', '') for c in curr.index], orientation='h', title="Macro Drivers")
                        fig.update_layout(template='plotly_dark')
                        st.plotly_chart(fig, use_container_width=True)

            else:
                st.error("Engine Error. Try again.")
        else:
            st.error("Data Feed Disconnected.")
else:
    st.info("System Ready.")
