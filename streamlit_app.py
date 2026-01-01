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
import time

# --- 1. KONFIGURACJA STRONY ---
st.set_page_config(
    page_title="QUANT LAB | SMC Institutional Core",
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
    .matrix-box { padding: 8px; border-radius: 4px; text-align: center; margin-bottom: 5px; font-family: 'Courier New'; border: 1px solid #333; font-size: 14px;}
</style>
""", unsafe_allow_html=True)

# --- 2. SMC & MATH ENGINES ---

def find_fair_value_gaps(df):
    """Wykrywa luki FVG (Smart Money Imbalances) - Kluczowe dla ICT"""
    fvgs = []
    # Analiza ostatnich 200 świec
    lookback = 200
    subset = df.tail(lookback)
    
    for i in range(2, len(subset)):
        curr_idx = subset.index[i]
        prev_idx = subset.index[i-2]
        
        # Bullish FVG: Low świecy 3 > High świecy 1
        if subset['Low'].iloc[i] > subset['High'].iloc[i-2]:
            fvgs.append({
                'start_time': prev_idx,
                'end_time': curr_idx,
                'top': subset['Low'].iloc[i],
                'bottom': subset['High'].iloc[i-2],
                'type': 'Bullish FVG',
                'color': 'rgba(0, 255, 0, 0.15)'
            })
        # Bearish FVG: High świecy 3 < Low świecy 1
        elif subset['High'].iloc[i] < subset['Low'].iloc[i-2]:
            fvgs.append({
                'start_time': prev_idx,
                'end_time': curr_idx,
                'top': subset['Low'].iloc[i-2],
                'bottom': subset['High'].iloc[i],
                'type': 'Bearish FVG',
                'color': 'rgba(255, 0, 0, 0.15)'
            })
    return fvgs[-15:] # Zwróć ostatnie 15

def find_liquidity_pools(df, window=5):
    """Wykrywa poziomy płynności (Fractals)"""
    highs = df['High'].values
    lows = df['Low'].values
    max_idx = argrelextrema(highs, np.greater, order=window)[0]
    min_idx = argrelextrema(lows, np.less, order=window)[0]
    
    limit = len(df) - 60
    levels = []
    for i in max_idx:
        if i > limit: levels.append({'price': float(highs[i]), 'type': 'Sell Liquidity'})
    for i in min_idx:
        if i > limit: levels.append({'price': float(lows[i]), 'type': 'Buy Liquidity'})
    return levels

def kalman_filter(series):
    """Instytucjonalne wygładzanie ceny"""
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

# --- 3. ROBUST DATA LOADER (SEQUENTIAL & SAFE) ---
@st.cache_data(ttl=3600)
def load_data():
    # Definicja tickerów
    main_ticker = 'EURUSD=X'
    macro_tickers = {'US10Y': '^TNX', 'DXY': 'DX-Y.NYB'}
    
    main_df = pd.DataFrame()
    
    # 1. Pobierz Main Asset (EURUSD) - Krytyczne
    try:
        # Pobieramy tylko jeden ticker, to rzadziej wywołuje błąd
        main_df = yf.download(main_ticker, period="2y", interval="1d", progress=False)
        
        # Obsługa MultiIndex (Nowe yfinance)
        if isinstance(main_df.columns, pd.MultiIndex):
            temp = pd.DataFrame()
            temp['Close'] = main_df['Close'][main_ticker]
            temp['Open'] = main_df['Open'][main_ticker]
            temp['High'] = main_df['High'][main_ticker]
            temp['Low'] = main_df['Low'][main_ticker]
            main_df = temp
        
        main_df = main_df.ffill().dropna()
    except Exception as e:
        return None # Jeśli nie ma EURUSD, nie ma nic

    # 2. Pobierz Makro (Opcjonalne, z opóźnieniem dla bezpieczeństwa)
    time.sleep(2) # Czekamy 2 sekundy, żeby nie dostać Rate Limit
    
    try:
        us10 = yf.download(macro_tickers['US10Y'], period="2y", interval="1d", progress=False)
        if not us10.empty:
            val = us10['Close'][macro_tickers['US10Y']] if isinstance(us10.columns, pd.MultiIndex) else us10['Close']
            main_df['US10Y'] = val
    except:
        pass # Ignoruj błąd makro, jedziemy dalej

    time.sleep(2) # Kolejna pauza
    
    try:
        dxy = yf.download(macro_tickers['DXY'], period="2y", interval="1d", progress=False)
        if not dxy.empty:
            val = dxy['Close'][macro_tickers['DXY']] if isinstance(dxy.columns, pd.MultiIndex) else dxy['Close']
            main_df['DXY'] = val
    except:
        pass

    return main_df.ffill()

# --- 4. ANALYTICS ENGINE ---
def run_analytics(df):
    if df is None or len(df) < 50: return None
    
    # A. SMC Logic
    df['Kalman'] = kalman_filter(df['Close'])
    liquidity = find_liquidity_pools(df)
    fvgs = find_fair_value_gaps(df)
    
    # B. Macro Logic (Bond Spread)
    fv_gap = 0.0
    macro_available = False
    
    # Sprawdzamy czy udało się pobrać US10Y
    if 'US10Y' in df.columns:
        clean_df = df[['Close', 'US10Y']].dropna()
        if len(clean_df) > 60:
            # Regresja liniowa: Rentowność vs Cena
            X = clean_df[['US10Y']].iloc[-60:]
            y = clean_df['Close'].iloc[-60:]
            model = LinearRegression().fit(X, y)
            df.loc[clean_df.index, 'Macro_FV'] = model.predict(clean_df[['US10Y']])
            # Oblicz GAP na ostatniej świecy
            if pd.notna(df['Macro_FV'].iloc[-1]):
                fv_gap = df['Close'].iloc[-1] - df['Macro_FV'].iloc[-1]
                macro_available = True
    
    # C. Signal
    last_close = df['Close'].iloc[-1]
    kalman_val = df['Kalman'].iloc[-1]
    trend = "BULLISH" if last_close > kalman_val else "BEARISH"
    
    return {
        'price': last_close,
        'trend': trend,
        'fv_gap': fv_gap,
        'macro_ok': macro_available,
        'df': df,
        'liquidity': liquidity,
        'fvgs': fvgs
    }

# --- 5. DASHBOARD UI ---
st.title("🏛️ QUANT LAB | Smart Money V9")

if st.button("🚀 INITIALIZE FEED (SEQUENTIAL)", type="primary"):
    with st.spinner("Connecting to Liquidity Providers..."):
        df = load_data()
        
        if df is not None:
            res = run_analytics(df)
            
            if res:
                # --- HUD ---
                c1, c2, c3, c4 = st.columns(4)
                with c1: st.metric("EURUSD SPOT", f"{res['price']:.5f}")
                with c2: st.metric("SMC BIAS", res['trend'], delta="Kalman Trend")
                
                if res['macro_ok']:
                    with c3: st.metric("MACRO FV GAP", f"{res['fv_gap']:.4f}", delta="Bond Spread Model", delta_color="inverse")
                else:
                    with c3: st.metric("MACRO DATA", "OFFLINE", delta="API Limit", delta_color="off")
                    
                # Liquidity Proximity
                nearest_liq = min([abs(res['price'] - l['price']) for l in res['liquidity']]) if res['liquidity'] else 0
                with c4: st.metric("DIST. TO LIQUIDITY", f"{nearest_liq*10000:.1f} pips", delta="Stop Hunt Risk")

                # --- CHARTING ---
                st.markdown("---")
                
                # Konfiguracja subplots (Makro tylko jeśli dostępne)
                rows = 2 if res['macro_ok'] else 1
                row_heights = [0.7, 0.3] if res['macro_ok'] else [1.0]
                titles = ("Price Action (SMC + FVG)", "Macro Divergence (Yields)") if res['macro_ok'] else ("Price Action (SMC)",)
                
                fig = make_subplots(rows=rows, cols=1, shared_xaxes=True, vertical_spacing=0.05, row_heights=row_heights, subplot_titles=titles)
                
                pdf = res['df'].tail(120)
                
                # 1. Świece
                fig.add_trace(go.Candlestick(x=pdf.index, open=pdf['Open'], high=pdf['High'], low=pdf['Low'], close=pdf['Close'], name='Price'), row=1, col=1)
                
                # 2. Kalman
                fig.add_trace(go.Scatter(x=pdf.index, y=pdf['Kalman'], line=dict(color='yellow', width=1.5), name='Institutional Trend'), row=1, col=1)
                
                # 3. FVG Boxes (Smart Money Imbalance)
                for fvg in res['fvgs']:
                    if fvg['start_time'] >= pdf.index[0]:
                        fig.add_shape(type="rect",
                            x0=fvg['start_time'], x1=pdf.index[-1] + pd.Timedelta(days=2), 
                            y0=fvg['bottom'], y1=fvg['top'],
                            fillcolor=fvg['color'], line=dict(width=0), layer="below", row=1, col=1
                        )

                # 4. Liquidity Lines
                for level in res['liquidity']:
                    if level['price'] > pdf['Low'].min() and level['price'] < pdf['High'].max():
                        color = "red" if level['type'] == 'Sell Liquidity' else "#00ff00"
                        fig.add_hline(y=level['price'], line_dash="dot", line_color=color, line_width=1, row=1, col=1)

                # 5. Macro Panel (Jeśli dostępne)
                if res['macro_ok'] and 'US10Y' in pdf.columns:
                    # Normalizacja
                    p_norm = (pdf['Close'] - pdf['Close'].min()) / (pdf['Close'].max() - pdf['Close'].min())
                    y_inv = -pdf['US10Y'] # Odwrócone rentowności
                    y_norm = (y_inv - y_inv.min()) / (y_inv.max() - y_inv.min())
                    
                    fig.add_trace(go.Scatter(x=pdf.index, y=p_norm, line=dict(color='white', width=1), name='EURUSD (Norm)'), row=2, col=1)
                    fig.add_trace(go.Scatter(x=pdf.index, y=y_norm, line=dict(color='orange', width=1), name='US10Y Inv (Norm)'), row=2, col=1)

                fig.update_layout(height=700, template='plotly_dark', margin=dict(l=0,r=0,t=30,b=0), showlegend=False, xaxis_rangeslider_visible=False)
                st.plotly_chart(fig, width="stretch")
                
                st.info("💡 **Institutional Insight:** Zielone/Czerwone strefy na wykresie to **Fair Value Gaps (FVG)**. To tam Smart Money często 'rebalansują' cenę. Szukaj wejść, gdy cena wraca do tych stref zgodnie z trendem Kalmana.")

            else:
                st.error("Data processing error.")
        else:
            st.error("System Failure. Could not download EURUSD data.")
else:
    st.info("System Ready. Click INITIALIZE to start.")
