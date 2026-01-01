import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import xgboost as xgb
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots # <--- NAPRAWIONY IMPORT
from sklearn.linear_model import LinearRegression
from scipy.stats import t
from scipy.signal import argrelextrema
import warnings
import time

# --- 1. CONFIGURATION ---
st.set_page_config(
    page_title="QUANT LAB | Institutional V10",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="collapsed"
)
warnings.filterwarnings("ignore")

# --- CUSTOM CSS ---
st.markdown("""
<style>
    .stApp { background-color: #0e1117; }
    
    /* HUD Metrics */
    div[data-testid="stMetricValue"] { 
        font-size: 28px; font-family: 'Courier New'; font-weight: bold;
    }
    
    /* Matrix Box */
    .matrix-box { 
        padding: 15px; border-radius: 8px; text-align: center; 
        font-weight: bold; margin-bottom: 10px; font-family: 'Courier New'; 
        border: 1px solid #444; background-color: #161b22; font-size: 16px;
    }
    
    /* Table Styling */
    .stDataFrame { width: 100%; }
</style>
""", unsafe_allow_html=True)

# --- 2. MATH ENGINES ---

def calculate_kelly(prob_win, win_loss_ratio=1.5):
    """Kryterium Kelly'ego dla zarządzania kapitałem"""
    q = 1 - prob_win
    f = (win_loss_ratio * prob_win - q) / win_loss_ratio
    return max(0, f) * 0.5 # Half-Kelly dla bezpieczeństwa

def find_liquidity_levels(df, window=5):
    """Wykrywa Smart Money Fractals (Liquidity Pools)"""
    highs = df['High'].values
    lows = df['Low'].values
    max_idx = argrelextrema(highs, np.greater, order=window)[0]
    min_idx = argrelextrema(lows, np.less, order=window)[0]
    
    recent_limit = len(df) - 60
    levels = []
    for i in max_idx:
        if i > recent_limit: levels.append({'price': float(highs[i]), 'type': 'Sell Liquidity (Res)', 'color': 'red'})
    for i in min_idx:
        if i > recent_limit: levels.append({'price': float(lows[i]), 'type': 'Buy Liquidity (Sup)', 'color': '#00ff00'})
    return levels

def find_fair_value_gaps(df):
    """Wykrywa strefy FVG (Imbalance)"""
    fvgs = []
    lookback = 150
    subset = df.tail(lookback)
    for i in range(2, len(subset)):
        curr_idx = subset.index[i]
        prev_idx = subset.index[i-2]
        if subset['Low'].iloc[i] > subset['High'].iloc[i-2]:
            fvgs.append({'start': prev_idx, 'end': curr_idx, 'top': subset['Low'].iloc[i], 'bottom': subset['High'].iloc[i-2], 'color': 'rgba(0, 255, 0, 0.15)'})
        elif subset['High'].iloc[i] < subset['Low'].iloc[i-2]:
            fvgs.append({'start': prev_idx, 'end': curr_idx, 'top': subset['Low'].iloc[i-2], 'bottom': subset['High'].iloc[i], 'color': 'rgba(255, 0, 0, 0.15)'})
    return fvgs[-10:]

def garman_klass_volatility(df):
    """Lepszy estymator zmienności niż odchylenie standardowe"""
    try:
        log_hl = np.log(df['High'] / df['Low'])**2
        log_co = np.log(df['Close'] / df['Open'])**2
        return np.sqrt(0.5 * log_hl - (2 * np.log(2) - 1) * log_co)
    except:
        return df['Close'].pct_change().rolling(20).std()

def kalman_filter(series):
    """Filtr Kalmana do wygładzania trendu"""
    xhat = np.zeros(len(series))
    P = np.zeros(len(series))
    xhatminus = np.zeros(len(series))
    Pminus = np.zeros(len(series))
    K = np.zeros(len(series))
    Q = 1e-5
    R = 0.01**2
    xhat[0] = series.iloc[0]
    P[0] = 1.0
    for k in range(1, len(series)):
        xhatminus[k] = xhat[k-1]
        Pminus[k] = P[k-1] + Q
        K[k] = Pminus[k] / (Pminus[k] + R)
        xhat[k] = xhatminus[k] + K[k] * (series.iloc[k] - xhatminus[k])
        P[k] = (1 - K[k]) * Pminus[k]
    return xhat

# --- 3. DATA LOADER (SEQUENTIAL & SAFE) ---
@st.cache_data(ttl=3600)
def load_data():
    main_ticker = 'EURUSD=X'
    macro_tickers = {'US10Y': '^TNX', 'DXY': 'DX-Y.NYB', 'GOLD': 'GC=F'}
    
    main_df = pd.DataFrame()
    
    # 1. EURUSD (Critical)
    try:
        main_df = yf.download(main_ticker, period="2y", interval="1d", progress=False)
        if isinstance(main_df.columns, pd.MultiIndex):
            temp = pd.DataFrame()
            temp['Close'] = main_df['Close'][main_ticker]
            temp['Open'] = main_df['Open'][main_ticker]
            temp['High'] = main_df['High'][main_ticker]
            temp['Low'] = main_df['Low'][main_ticker]
            main_df = temp
        main_df = main_df.ffill().dropna()
    except:
        return None, None

    # 2. Weekly Data (For MTF Matrix)
    weekly_df = pd.DataFrame()
    try:
        weekly_df = yf.download(main_ticker, period="2y", interval="1wk", progress=False)
        if isinstance(weekly_df.columns, pd.MultiIndex):
            weekly_df = weekly_df.xs(main_ticker, axis=1, level=1)
    except:
        pass

    # 3. Macro (Sequential to avoid Rate Limit)
    for name, ticker in macro_tickers.items():
        time.sleep(1.5) # Pauza
        try:
            aux = yf.download(ticker, period="2y", interval="1d", progress=False)
            if not aux.empty:
                val = aux['Close'][ticker] if isinstance(aux.columns, pd.MultiIndex) else aux['Close']
                main_df[name] = val
        except:
            pass

    return main_df.ffill(), weekly_df.ffill()

# --- 4. ANALYTICS ENGINE ---
def run_analytics(df, df_w):
    if df is None or len(df) < 50: return None
    
    # Technicals
    df['Log_Ret'] = np.log(df['Close'] / df['Close'].shift(1))
    df['Vol_GK'] = garman_klass_volatility(df)
    df['Kalman'] = kalman_filter(df['Close'])
    df['MA_50'] = df['Close'].rolling(50).mean()
    df['Z_Score'] = (df['Close'] - df['MA_50']) / df['Close'].rolling(50).std()
    
    # SMC Logic
    liquidity = find_liquidity_levels(df)
    fvgs = find_fair_value_gaps(df)
    
    # MTF Matrix Logic
    weekly_trend = "NEUTRAL"
    if not df_w.empty:
        df_w['MA_20'] = df_w['Close'].rolling(20).mean()
        if len(df_w) > 20:
            weekly_trend = "BULLISH" if df_w['Close'].iloc[-1] > df_w['MA_20'].iloc[-1] else "BEARISH"
    
    daily_trend = "BULLISH" if df['Close'].iloc[-1] > df['MA_50'].iloc[-1] else "BEARISH"
    
    confluence = "MIXED"
    if weekly_trend == "BULLISH" and daily_trend == "BULLISH": confluence = "STRONG UPTREND"
    elif weekly_trend == "BEARISH" and daily_trend == "BEARISH": confluence = "STRONG DOWNTREND"

    # Macro FV
    fv_gap = 0.0
    macro_ok = False
    if 'US10Y' in df.columns:
        clean = df[['Close', 'US10Y']].dropna()
        if len(clean) > 60:
            model = LinearRegression().fit(clean[['US10Y']].iloc[-60:], clean['Close'].iloc[-60:])
            df.loc[clean.index, 'Macro_FV'] = model.predict(clean[['US10Y']])
            fv_gap = df['Close'].iloc[-1] - df['Macro_FV'].iloc[-1]
            macro_ok = True

    # Correlations
    for m in ['US10Y', 'DXY', 'GOLD']:
        if m in df.columns:
            df[f'Corr_{m}'] = df['Log_Ret'].rolling(30).corr(df[m].pct_change())

    # AI Signal
    features = ['Vol_GK', 'Z_Score']
    valid_feat = [f for f in features if f in df.columns]
    df['Target'] = (df['Close'].shift(-3) / df['Close'] - 1 > 0.0010).astype(int)
    
    model = xgb.XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.05, n_jobs=-1)
    model.fit(df[valid_feat].iloc[:-3], df['Target'].iloc[:-3])
    prob_up = model.predict_proba(df[valid_feat].iloc[[-1]])[0][1]

    return {
        'price': df['Close'].iloc[-1],
        'kalman': df['Kalman'].iloc[-1],
        'weekly_trend': weekly_trend,
        'daily_trend': daily_trend,
        'confluence': confluence,
        'prob_up': prob_up,
        'fv_gap': fv_gap,
        'macro_ok': macro_ok,
        'df': df,
        'liquidity': liquidity,
        'fvgs': fvgs
    }

# --- 5. UI LAYOUT ---
st.title("🏛️ QUANT LAB | Institutional V10")

if st.button("🚀 INITIALIZE FULL SYSTEM", type="primary"):
    with st.spinner("Aggregating Institutional Feeds..."):
        df, df_w = load_data()
        
        if df is not None:
            res = run_analytics(df, df_w)
            
            if res:
                # --- 1. HUD (Naprawione Kolory) ---
                prob = res['prob_up']
                if prob > 0.55:
                    bias = "BULLISH"
                    bias_color = "normal" # Zielony w Streamlit
                elif prob < 0.45:
                    bias = "BEARISH"
                    bias_color = "inverse" # Czerwony w Streamlit (dla delty)
                else:
                    bias = "NEUTRAL"
                    bias_color = "off"

                kelly = calculate_kelly(prob if prob > 0.5 else 1-prob) * 10000 * 0.5
                
                c1, c2, c3, c4 = st.columns(4)
                with c1: st.metric("EURUSD PRICE", f"{res['price']:.5f}")
                with c2: st.metric("SMC BIAS", bias, delta=f"{prob:.1%}", delta_color=bias_color)
                with c3: st.metric("KELLY RISK (10k)", f"${kelly:.0f}", delta="Position Size", delta_color="off")
                with c4: st.metric("FAIR VALUE GAP", f"{res['fv_gap']:.4f}", delta="Macro Model", delta_color="inverse")

                # --- 2. MATRIX ---
                st.markdown("---")
                m1, m2, m3 = st.columns(3)
                def get_color(t): return "#00ff00" if t == "BULLISH" else "#ff0000" if t == "BEARISH" else "#888"
                
                m1.markdown(f"<div class='matrix-box' style='color:{get_color(res['weekly_trend'])}'>WEEKLY TREND<br>{res['weekly_trend']}</div>", unsafe_allow_html=True)
                m2.markdown(f"<div class='matrix-box' style='color:{get_color(res['daily_trend'])}'>DAILY TREND<br>{res['daily_trend']}</div>", unsafe_allow_html=True)
                
                conf_c = "#00ff00" if "UP" in res['confluence'] else "#ff0000" if "DOWN" in res['confluence'] else "orange"
                m3.markdown(f"<div class='matrix-box' style='color:{conf_c}'>CONFLUENCE<br>{res['confluence']}</div>", unsafe_allow_html=True)

                # --- 3. MAIN CHART (SMC + FVG + Kalman) ---
                st.markdown("### 🔭 Institutional Price Action")
                fig = go.Figure()
                pdf = res['df'].tail(150)
                
                # Candles
                fig.add_trace(go.Candlestick(x=pdf.index, open=pdf['Open'], high=pdf['High'], low=pdf['Low'], close=pdf['Close'], name='Price'))
                # Kalman
                fig.add_trace(go.Scatter(x=pdf.index, y=pdf['Kalman'], line=dict(color='yellow', width=2), name='Kalman Filter'))
                
                # FVG (Rectangles)
                for fvg in res['fvgs']:
                    if fvg['start'] >= pdf.index[0]:
                        fig.add_shape(type="rect", x0=fvg['start'], x1=pdf.index[-1] + pd.Timedelta(days=1), 
                                      y0=fvg['bottom'], y1=fvg['top'], fillcolor=fvg['color'], line=dict(width=0), layer="below")
                
                # Liquidity Lines
                for level in res['liquidity']:
                    if level['price'] > pdf['Low'].min() and level['price'] < pdf['High'].max():
                        c = "red" if "Sell" in level['type'] else "#00ff00"
                        fig.add_hline(y=level['price'], line_dash="dot", line_color=c, annotation_text=level['type'])

                fig.update_layout(height=700, template='plotly_dark', margin=dict(l=0,r=0,t=20,b=0), xaxis_rangeslider_visible=False)
                st.plotly_chart(fig, use_container_width=True)

                # --- 4. TABS & DATA ---
                t1, t2, t3 = st.tabs(["📋 Liquidity Data", "📊 Z-Score & Volatility", "🌍 Macro Correlations"])
                
                with t1:
                    st.caption("Active Liquidity Pools (Data Feed)")
                    liq_data = [{'Level': l['price'], 'Type': l['type'], 'Dist (pips)': f"{(l['price']-res['price'])*10000:.1f}"} for l in res['liquidity']]
                    st.dataframe(pd.DataFrame(liq_data).sort_values('Level', ascending=False), use_container_width=True)
                    
                with t2:
                    c_z, c_v = st.columns(2)
                    with c_z:
                        fig_z = go.Figure()
                        fig_z.add_trace(go.Scatter(x=pdf.index, y=pdf['Z_Score'], fill='tozeroy', line=dict(color='cyan')))
                        fig_z.add_hline(y=2.0, line_color='red', line_dash='dash')
                        fig_z.add_hline(y=-2.0, line_color='green', line_dash='dash')
                        fig_z.update_layout(height=300, template='plotly_dark', title="Z-Score (Overbought/Oversold)")
                        st.plotly_chart(fig_z, use_container_width=True)
                    with c_v:
                        fig_v = px.line(pdf, x=pdf.index, y='Vol_GK', title="Garman-Klass Volatility")
                        fig_v.update_traces(line_color='magenta')
                        fig_v.update_layout(height=300, template='plotly_dark')
                        st.plotly_chart(fig_v, use_container_width=True)
                        
                with t3:
                    if res['macro_ok']:
                        fig_m = make_subplots(specs=[[{"secondary_y": True}]])
                        fig_m.add_trace(go.Scatter(x=pdf.index, y=pdf['Close'], name="EURUSD"), secondary_y=False)
                        if 'US10Y' in pdf.columns:
                            fig_m.add_trace(go.Scatter(x=pdf.index, y=pdf['US10Y'], name="US10Y Yield", line=dict(color='orange')), secondary_y=True)
                        fig_m.update_layout(height=400, template='plotly_dark', title="Yield Divergence (Price vs Rates)")
                        st.plotly_chart(fig_m, use_container_width=True)
                        
                        corr_cols = [c for c in res['df'].columns if 'Corr_' in c]
                        if corr_cols:
                            curr = res['df'][corr_cols].iloc[-1].sort_values()
                            fig_c = px.bar(x=curr.values, y=[c.replace('Corr_', '') for c in curr.index], orientation='h', title="Live Correlations")
                            fig_c.update_layout(template='plotly_dark')
                            st.plotly_chart(fig_c, use_container_width=True)
                    else:
                        st.warning("Macro data unavailable (API Rate Limit). Price analysis active.")

            else:
                st.error("Engine failure. Data insufficient.")
        else:
            st.error("Feed Disconnected. Try again.")
else:
    st.info("System Ready. Click INITIALIZE.")
