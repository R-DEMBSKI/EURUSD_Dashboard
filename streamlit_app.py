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

# --- 1. CONFIGURATION ---
st.set_page_config(
    page_title="QUANT LAB | Modular Terminal",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="collapsed" # Zwijamy pasek, by dać więcej miejsca danym
)

warnings.filterwarnings("ignore")

# --- CUSTOM CSS (Data Tables & Matrix) ---
st.markdown("""
<style>
    .stApp { background-color: #0e1117; }
    
    /* HUD Metrics */
    div[data-testid="stMetricValue"] { 
        font-size: 26px; color: #00ff00; font-family: 'Courier New', monospace; font-weight: bold;
    }
    div[data-testid="stMetricLabel"] { font-size: 13px; color: #888; }
    
    /* Tables */
    div[data-testid="stDataFrame"] { width: 100%; }
    
    /* Matrix Box */
    .matrix-box {
        padding: 15px; border-radius: 8px; text-align: center; 
        font-weight: bold; margin-bottom: 10px; font-family: 'Courier New';
        border: 1px solid #333; background-color: #161b22;
    }
    
    /* Headers */
    h3 { border-bottom: 2px solid #333; padding-bottom: 10px; color: #eee; }
</style>
""", unsafe_allow_html=True)

# --- 2. CONFIG ---
CONFIG = {
    'TARGET': 'EURUSD=X',
    'MACRO': {'US10Y': '^TNX', 'DXY': 'DX-Y.NYB', 'SPX': '^GSPC', 'GOLD': 'GC=F'},
    'LOOKBACK': '730d', 
}

# --- 3. MATH ENGINES ---
def calculate_kelly(prob_win, win_loss_ratio=1.5):
    q = 1 - prob_win
    f = (win_loss_ratio * prob_win - q) / win_loss_ratio
    return max(0, f) * 0.5 

def find_liquidity_levels(df, window=5):
    highs = df['High'].values
    lows = df['Low'].values
    max_idx = argrelextrema(highs, np.greater, order=window)[0]
    min_idx = argrelextrema(lows, np.less, order=window)[0]
    
    recent_limit = len(df) - 60
    rel_res = [highs[i] for i in max_idx if i > recent_limit]
    rel_sup = [lows[i] for i in min_idx if i > recent_limit]
    return sorted(list(set(rel_res))), sorted(list(set(rel_sup)))

def garman_klass_volatility(df):
    try:
        log_hl = np.log(df['High'] / df['Low'])**2
        log_co = np.log(df['Close'] / df['Open'])**2
        return np.sqrt(0.5 * log_hl - (2 * np.log(2) - 1) * log_co)
    except:
        return df['Close'].pct_change().rolling(20).std()

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
    df_d = pd.DataFrame()
    for _ in range(3):
        try:
            raw = yf.download(tickers, period="2y", interval="1d", progress=False)
            if not raw.empty:
                df_d = raw
                break
        except:
            time.sleep(1)
            
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
            model = LinearRegression()
            X = df[macro_cols].iloc[-window:]
            y = df['Close'].iloc[-window:]
            model.fit(X, y)
            df['Fair_Value'] = model.predict(df[macro_cols])
            df['FV_Gap'] = df['Close'] - df['Fair_Value']
        else: df['FV_Gap'] = 0.0
    except: df['FV_Gap'] = 0.0

    # Z-Score
    df['Z_Score'] = (df['Close'] - df['MA_50']) / df['Close'].rolling(50).std()

    # AI Model
    df['Target'] = (df['Close'].shift(-3) / df['Close'] - 1 > 0.0010).astype(int)
    features = ['Vol_GK', 'Z_Score']
    valid_features = [f for f in features if f in df.columns]
    
    model = xgb.XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.05, n_jobs=-1)
    model.fit(df[valid_features].iloc[:-3], df['Target'].iloc[:-3])
    prob_up = model.predict_proba(df[valid_features].iloc[[-1]])[0][1]

    # Liquidity Levels
    res_levels, sup_levels = find_liquidity_levels(df)

    return {
        'price': df['Close'].iloc[-1],
        'kalman': df['Kalman'].iloc[-1],
        'weekly_trend': weekly_trend,
        'daily_trend': daily_trend,
        'confluence': confluence,
        'prob_up': prob_up,
        'fv_gap': df['FV_Gap'].iloc[-1],
        'df': df,
        'res_levels': res_levels,
        'sup_levels': sup_levels
    }

# --- 6. UI LAYOUT ---
st.title("🏛️ QUANT LAB | Institutional Terminal")

if st.button("🔄 REFRESH DATA STREAM", type="primary"):
    with st.spinner("Accessing Institutional Feeds..."):
        df_d, df_w = load_data("EURUSD=X")
        
        if df_d is not None and not df_d.empty:
            res = run_quant_engine(df_d, df_w)
            if res:
                # --- A. HUD (HEADS UP DISPLAY) ---
                prob = res['prob_up']
                signal = "LONG" if prob > 0.6 else "SHORT" if prob < 0.4 else "NEUTRAL"
                color = "normal" if signal == "LONG" else "inverse" if signal == "SHORT" else "off"
                kelly = calculate_kelly(prob if prob > 0.5 else 1-prob) * 10000 * 0.5 # Zakładamy konto 10k
                
                c1, c2, c3, c4 = st.columns(4)
                with c1: st.metric("SPOT PRICE", f"{res['price']:.5f}")
                with c2: st.metric("AI SIGNAL", signal, delta=f"{prob:.1%}", delta_color=color)
                with c3: st.metric("KELLY SIZE (10k)", f"${kelly:.0f}", delta="Risk Adjusted", delta_color="off")
                with c4: st.metric("FAIR VALUE GAP", f"{res['fv_gap']:.4f}", delta="Macro Divergence", delta_color="inverse")

                st.markdown("---")

                # --- B. SPLIT VIEW (Tactical vs Strategic) ---
                col_tactical, col_strategic = st.columns([6, 4]) # 60% vs 40% szerokości
                
                with col_tactical:
                    st.subheader("🔭 Tactical Execution (Liquidity & Candles)")
                    fig_tac = go.Figure()
                    pdf = res['df'].tail(100)
                    
                    # Czyste świece
                    fig_tac.add_trace(go.Candlestick(x=pdf.index, open=pdf['Open'], high=pdf['High'], low=pdf['Low'], close=pdf['Close'], name='Price'))
                    
                    # Tylko poziomy Liquidity (Fractals)
                    for level in res['res_levels'][-3:]: 
                        fig_tac.add_hline(y=level, line_color='red', line_width=1, line_dash='dash', annotation_text="Liquidity Sell")
                    for level in res['sup_levels'][-3:]: 
                        fig_tac.add_hline(y=level, line_color='#00ff00', line_width=1, line_dash='dash', annotation_text="Liquidity Buy")
                        
                    fig_tac.update_layout(height=500, template='plotly_dark', margin=dict(l=0,r=0,t=30,b=0))
                    # FIX: Używamy nowego parametru, aby usunąć warning
                    st.plotly_chart(fig_tac, use_container_width=True)

                with col_strategic:
                    st.subheader("🧠 Strategic Context (Fair Value)")
                    fig_str = go.Figure()
                    pdf = res['df'].tail(150)
                    
                    # Linia ceny zamiast świec (dla czystości trendu)
                    fig_str.add_trace(go.Scatter(x=pdf.index, y=pdf['Close'], mode='lines', line=dict(color='white', width=1), name='Price'))
                    # Kalman
                    fig_str.add_trace(go.Scatter(x=pdf.index, y=pdf['Kalman'], mode='lines', line=dict(color='yellow', width=2), name='Kalman Trend'))
                    # Fair Value
                    if 'Fair_Value' in pdf.columns:
                        fig_str.add_trace(go.Scatter(x=pdf.index, y=pdf['Fair_Value'], mode='lines', line=dict(color='orange', dash='dot'), name='Macro FV'))
                        
                    fig_str.update_layout(height=500, template='plotly_dark', margin=dict(l=0,r=0,t=30,b=0))
                    st.plotly_chart(fig_str, use_container_width=True)

                # --- C. DATA INTEL & MATRIX (Tabelki) ---
                st.markdown("---")
                st.subheader("📋 Institutional Data Feed")
                
                tab1, tab2 = st.tabs(["🔢 Key Levels & Matrix", "📊 Volatility & Z-Score"])
                
                with tab1:
                    m1, m2, m3 = st.columns(3)
                    # Matrix w HTML dla wyglądu
                    def get_color(trend): return "#00ff00" if trend == "BULLISH" else "#ff0000" if trend == "BEARISH" else "#888"
                    m1.markdown(f"<div class='matrix-box' style='color:{get_color(res['weekly_trend'])}'>WEEKLY<br>{res['weekly_trend']}</div>", unsafe_allow_html=True)
                    m2.markdown(f"<div class='matrix-box' style='color:{get_color(res['daily_trend'])}'>DAILY<br>{res['daily_trend']}</div>", unsafe_allow_html=True)
                    m3.markdown(f"<div class='matrix-box' style='color:orange'>CONFLUENCE<br>{res['confluence']}</div>", unsafe_allow_html=True)
                    
                    # Tabela z poziomami (Dataframe)
                    st.caption("Active Liquidity Pools (Stop Loss Clusters)")
                    liquidity_data = {
                        "Type": ["Resistance (Sell)"]*len(res['res_levels'][-3:]) + ["Support (Buy)"]*len(res['sup_levels'][-3:]),
                        "Price Level": res['res_levels'][-3:] + res['sup_levels'][-3:],
                        "Distance (pips)": [f"{(p - res['price'])*10000:.1f}" for p in res['res_levels'][-3:]] + [f"{(p - res['price'])*10000:.1f}" for p in res['sup_levels'][-3:]]
                    }
                    df_levels = pd.DataFrame(liquidity_data).sort_values(by="Price Level", ascending=False)
                    st.dataframe(df_levels, use_container_width=True, hide_index=True)

                with tab2:
                    c_z, c_v = st.columns(2)
                    with c_z:
                        st.caption("Z-Score (Mean Reversion Pressure)")
                        fig_z = go.Figure()
                        fig_z.add_trace(go.Scatter(x=pdf.index, y=pdf['Z_Score'], fill='tozeroy', line=dict(color='cyan')))
                        fig_z.add_hline(y=2.0, line_color='red', line_dash='dash')
                        fig_z.add_hline(y=-2.0, line_color='green', line_dash='dash')
                        fig_z.update_layout(height=250, template='plotly_dark', margin=dict(l=0,r=0,t=10,b=0))
                        st.plotly_chart(fig_z, use_container_width=True)
                    with c_v:
                        st.caption("Institutional Volatility (Garman-Klass)")
                        fig_v = px.line(pdf, x=pdf.index, y='Vol_GK')
                        fig_v.update_traces(line_color='magenta')
                        fig_v.update_layout(height=250, template='plotly_dark', margin=dict(l=0,r=0,t=10,b=0))
                        st.plotly_chart(fig_v, use_container_width=True)

            else:
                st.error("Engine failure. Data insufficient.")
        else:
            st.error("Feed Disconnected. Try again.")
else:
    st.info("System Ready. Click REFRESH to start.")
