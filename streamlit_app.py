import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import xgboost as xgb
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.linear_model import LinearRegression
from scipy.signal import argrelextrema
import warnings
from datetime import datetime, time as dt_time
import pytz

# --- 1. CONFIGURATION ---
st.set_page_config(
    page_title="QUANT LAB | Macro-Structural Terminal",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="collapsed"
)
warnings.filterwarnings("ignore")

# --- CUSTOM CSS ---
st.markdown("""
<style>
    .stApp { background-color: #0e1117; }
    div[data-testid="stMetricValue"] { font-size: 24px; color: #00ff00; font-family: 'Courier New'; font-weight: bold; }
    div[data-testid="stMetricLabel"] { font-size: 13px; color: #888; }
    .matrix-box { padding: 10px; border-radius: 5px; text-align: center; font-weight: bold; margin-bottom: 5px; font-family: 'Courier New'; border: 1px solid #333; }
</style>
""", unsafe_allow_html=True)

# --- 2. ADVANCED ENGINES ---

def get_ict_killzones(df):
    """Oznacza sesje handlowe (ICT Killzones) na wykresie"""
    # Zakładamy czas UTC w danych
    zones = []
    for index, row in df.iterrows():
        # London Killzone (07:00 - 10:00 UTC) - orientacyjnie
        if index.hour == 8 and index.minute == 0:
            zones.append({'time': index, 'type': 'London Open', 'color': 'rgba(0, 0, 255, 0.1)'})
        # NY Killzone (12:00 - 15:00 UTC)
        elif index.hour == 13 and index.minute == 0:
            zones.append({'time': index, 'type': 'NY Open', 'color': 'rgba(255, 165, 0, 0.1)'})
    return zones

def calculate_smart_money_divergence(df):
    """Wykrywa dywergencję między ceną a rentownościami (Bond Spread Logic)"""
    # Normalizacja (Min-Max) dla porównania wizualnego
    price_norm = (df['Close'] - df['Close'].min()) / (df['Close'].max() - df['Close'].min())
    
    # Odwracamy US10Y (bo wyższe rentowności = niższe EURUSD)
    if 'US10Y' in df.columns:
        yield_inv = -df['US10Y']
        yield_norm = (yield_inv - yield_inv.min()) / (yield_inv.max() - yield_inv.min())
        return price_norm, yield_norm
    return None, None

def find_liquidity_sweeps(df, window=5):
    """Zaawansowane wykrywanie płynności (Fractals + Wicks)"""
    highs = df['High'].values
    lows = df['Low'].values
    
    max_idx = argrelextrema(highs, np.greater, order=window)[0]
    min_idx = argrelextrema(lows, np.less, order=window)[0]
    
    levels = []
    # Filtrujemy tylko ostatnie 3 miesiące
    limit = len(df) - 100
    
    for i in max_idx:
        if i > limit: levels.append({'price': highs[i], 'type': 'Buy-Side Liquidity (Res)', 'color': 'red'})
    for i in min_idx:
        if i > limit: levels.append({'price': lows[i], 'type': 'Sell-Side Liquidity (Sup)', 'color': '#00ff00'})
        
    return levels

def kalman_filter(series):
    """Wygładzanie szumu rynkowego"""
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

# --- 3. DATA LOADER ---
@st.cache_data(ttl=3600)
def load_data():
    # Pobieramy EURUSD, US10Y (TNX), DXY
    tickers = ['EURUSD=X', '^TNX', 'DX-Y.NYB']
    
    data = pd.DataFrame()
    try:
        # Pobieranie danych (Retry logic)
        for _ in range(3):
            df = yf.download(tickers, period="2y", interval="1d", progress=False)
            if not df.empty:
                break
        
        if df.empty: return None
        
        # Flattening MultiIndex
        if isinstance(df.columns, pd.MultiIndex):
            data['Close'] = df['Close']['EURUSD=X']
            data['Open'] = df['Open']['EURUSD=X']
            data['High'] = df['High']['EURUSD=X']
            data['Low'] = df['Low']['EURUSD=X']
            
            if '^TNX' in df['Close'].columns:
                data['US10Y'] = df['Close']['^TNX']
            if 'DX-Y.NYB' in df['Close'].columns:
                data['DXY'] = df['Close']['DX-Y.NYB']
        else:
            data = df # Fallback
            
        return data.ffill().dropna()
    except Exception as e:
        return None

# --- 4. MAIN ENGINE ---
def run_analysis(df):
    if df is None or len(df) < 50: return None
    
    # 1. Kalman
    df['Kalman'] = kalman_filter(df['Close'])
    
    # 2. Fair Value (Regression on US10Y & DXY)
    if 'US10Y' in df.columns and 'DXY' in df.columns:
        X = df[['US10Y', 'DXY']].iloc[-60:] # Ostatnie 60 dni korelacji
        y = df['Close'].iloc[-60:]
        model = LinearRegression().fit(X, y)
        df['Fair_Value'] = model.predict(df[['US10Y', 'DXY']])
        fv_gap = df['Close'].iloc[-1] - df['Fair_Value'].iloc[-1]
    else:
        df['Fair_Value'] = np.nan
        fv_gap = 0.0
        
    # 3. Liquidity Levels
    liquidity = find_liquidity_sweeps(df)
    
    # 4. Bond Divergence (Visual Data)
    p_norm, y_norm = calculate_smart_money_divergence(df)
    
    # 5. AI Signal (Simple Momentum + Volatility)
    df['Ret'] = df['Close'].pct_change()
    vol = df['Ret'].rolling(20).std() * np.sqrt(252)
    momentum = df['Close'].iloc[-1] > df['Close'].iloc[-20]
    
    signal = "BULLISH" if momentum and fv_gap < 0 else "BEARISH"
    
    return {
        'price': df['Close'].iloc[-1],
        'fv_gap': fv_gap,
        'signal': signal,
        'vol': vol.iloc[-1],
        'df': df,
        'liquidity': liquidity,
        'p_norm': p_norm,
        'y_norm': y_norm
    }

# --- 5. UI ---
st.title("🏛️ QUANT LAB | Macro-Structural Integrator")

if st.button("🚀 INITIALIZE INSTITUTIONAL FEED", type="primary"):
    with st.spinner("Synchronizing Yield Spreads & Liquidity Pools..."):
        df = load_data()
        res = run_analysis(df)
        
        if res:
            # --- HUD ---
            c1, c2, c3, c4 = st.columns(4)
            with c1: st.metric("EURUSD SPOT", f"{res['price']:.5f}")
            with c2: st.metric("INSTITUTIONAL BIAS", res['signal'], delta="Macro Driven")
            with c3: st.metric("FAIR VALUE GAP", f"{res['fv_gap']:.4f}", delta="Mispricing", delta_color="inverse")
            with c4: st.metric("VOLATILITY (Ann)", f"{res['vol']:.1%}")
            
            st.markdown("---")
            
            # --- MAIN CHART (SMC Style) ---
            # Wykres świecowy z poziomami płynności
            fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                                vertical_spacing=0.05, row_heights=[0.7, 0.3],
                                subplot_titles=("Price Action & Liquidity Pools (SMC)", "Macro Divergence (Yields vs Price)"))
            
            # 1. Price Candles
            pdf = res['df'].tail(120)
            fig.add_trace(go.Candlestick(x=pdf.index, open=pdf['Open'], high=pdf['High'], low=pdf['Low'], close=pdf['Close'], name='Price'), row=1, col=1)
            
            # 2. Kalman Filter (Smart Money Trend)
            fig.add_trace(go.Scatter(x=pdf.index, y=pdf['Kalman'], line=dict(color='yellow', width=2), name='Kalman Trend'), row=1, col=1)
            
            # 3. Liquidity Lines (Horizontal)
            for level in res['liquidity']:
                if level['price'] > pdf['Low'].min() and level['price'] < pdf['High'].max(): # Pokaż tylko widoczne
                    fig.add_hline(y=level['price'], line_dash="dot", line_color=level['color'], 
                                  annotation_text=level['type'], row=1, col=1)

            # 4. Macro Divergence (Subplot)
            if res['p_norm'] is not None:
                # Price (Normalized)
                fig.add_trace(go.Scatter(x=pdf.index, y=res['p_norm'].tail(120), line=dict(color='white'), name='EURUSD (Norm)'), row=2, col=1)
                # Yields Inverted (Normalized)
                fig.add_trace(go.Scatter(x=pdf.index, y=res['y_norm'].tail(120), line=dict(color='orange'), name='US10Y Inv (Norm)'), row=2, col=1)
                
            fig.update_layout(height=800, template='plotly_dark', margin=dict(l=0,r=0,t=30,b=0), showlegend=False)
            st.plotly_chart(fig, width="stretch") # Fixed deprecated warning
            
            # --- DATA INSIGHTS ---
            st.info("💡 **Institutional Insight:** Dolny wykres (Macro Divergence) pokazuje prawdę. Jeśli pomarańczowa linia (Obligacje) spada, a biała (EURUSD) rośnie -> to jest fałszywy ruch (Divergence). Czekaj na powrót ceny.")
            
        else:
            st.error("Data Feed Error. Please try again in 1 minute.")
else:
    st.info("System Ready. Waiting for connection...")
