import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import zscore
from datetime import datetime, time
import pytz

# --- 1. KONFIGURACJA UI (IBKR TWS STYLE) ---
st.set_page_config(layout="wide", page_title="EURUSD BERLIN DESK", page_icon="🦅", initial_sidebar_state="collapsed")

st.markdown("""
<style>
    /* GLOBALNY RESET - GŁĘBOKA CZERŃ */
    .stApp { background-color: #000000; color: #e0e0e0; font-family: 'Consolas', 'Roboto Mono', monospace; }
    
    /* Ukrycie marginesów Streamlit */
    .block-container { padding: 0.5rem 1rem; max-width: 100%; }
    header, footer {visibility: hidden;}
    
    /* MODUŁY I NAGŁÓWKI (TWS BLUE HEADER) */
    .module-container { border: 1px solid #333; margin-bottom: 5px; background-color: #111; }
    .module-header {
        background-color: #1a237e; /* IBKR Deep Blue */
        color: #fff; padding: 3px 8px; font-size: 0.75rem; font-weight: bold;
        border-bottom: 1px solid #444; letter-spacing: 0.5px;
        display: flex; justify_content: space-between; align-items: center;
    }
    
    /* KPI METRICS (KWADRATOWE) */
    div[data-testid="stMetric"] {
        background-color: #0a0a0a; border: 1px solid #333; padding: 5px; border-radius: 0px;
    }
    div[data-testid="stMetricLabel"] { font-size: 0.65rem !important; color: #888; text-transform: uppercase; }
    div[data-testid="stMetricValue"] { font-size: 1.1rem !important; color: #fff; font-weight: 700; }
    
    /* TABELE (NEWS/CALENDAR) */
    .dataframe { font-size: 0.75rem !important; font-family: 'Arial', sans-serif; }
    
    /* ZAKŁADKI */
    .stTabs [data-baseweb="tab-list"] { gap: 1px; background-color: #000; }
    .stTabs [data-baseweb="tab"] {
        height: 25px; background-color: #222; color: #aaa; border: 1px solid #333; 
        border-bottom: none; border-radius: 0px; font-size: 0.75rem; padding: 0 10px;
    }
    .stTabs [aria-selected="true"] { background-color: #1a237e; color: white; border: 1px solid #1a237e; }
    
    /* SCROLLBAR */
    ::-webkit-scrollbar { width: 6px; height: 6px; }
    ::-webkit-scrollbar-track { background: #000; }
    ::-webkit-scrollbar-thumb { background: #333; }
</style>
""", unsafe_allow_html=True)

# --- 2. QUANT ENGINE ---

# Definicja strefy czasowej
BERLIN_TZ = pytz.timezone('Europe/Berlin')

@st.cache_data(ttl=60)
def get_data(ticker, interval, period):
    """Pobiera dane i konwertuje na czas Berlin."""
    try:
        df = yf.download(ticker, period=period, interval=interval, progress=False)
        if df.empty: return None
        if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.droplevel(1)
        
        # Konwersja czasu
        if df.index.tz is None:
            df.index = df.index.tz_localize('UTC')
        df.index = df.index.tz_convert(BERLIN_TZ)
        
        return df
    except Exception as e:
        return None

@st.cache_data(ttl=300)
def get_intel_data(ticker):
    """Pobiera Newsy i Kalendarz (Safe Mode)."""
    t = yf.Ticker(ticker)
    news = t.news[:6] if hasattr(t, 'news') else []
    # Kalendarz często jest pusty dla Forex, więc robimy fallback
    cal = t.calendar if hasattr(t, 'calendar') else {}
    return news, cal

def calculate_metrics(df):
    """Główna matematyka."""
    if df is None: return None
    
    # 1. Kalman Proxy & VWAP
    df['Kalman'] = df['Close'].ewm(span=8).mean()
    v = df['Volume'].values
    tp = (df['High'] + df['Low'] + df['Close']) / 3
    df['VWAP'] = (tp * v).cumsum() / v.cumsum()
    
    # 2. Z-Score (Mean Reversion)
    roll_mean = df['Close'].rolling(50).mean()
    roll_std = df['Close'].rolling(50).std()
    df['Z_Score'] = (df['Close'] - roll_mean) / roll_std
    df['Upper'] = roll_mean + 2*roll_std
    df['Lower'] = roll_mean - 2*roll_std
    
    # 3. Hurst
    try:
        lags = range(2, 20)
        ts = df['Close'].tail(100).values
        tau = [np.sqrt(np.std(np.subtract(ts[lag:], ts[:-lag]))) for lag in lags]
        poly = np.polyfit(np.log(lags), np.log(tau), 1)
        hurst = poly[0] * 2.0
    except: hurst = 0.5
    
    # 4. Liquidity Sessions (Berlin Time)
    # Asia: 22:00-07:00 | London: 08:00-16:00 | NY: 13:00-21:00
    df['Hour'] = df.index.hour
    conditions = [
        (df['Hour'] >= 8) & (df['Hour'] < 16), # London
        (df['Hour'] >= 13) & (df['Hour'] < 21), # NY
        (df['Hour'] >= 22) | (df['Hour'] < 7)   # Asia
    ]
    choices = ['London', 'NY', 'Asia']
    # Uwaga: NY nakłada się na London, tutaj prosta kategoryzacja
    df['Session'] = np.select(conditions, choices, default='Other')
    
    return df, hurst

def get_volume_profile(df):
    """Pionowy profil wolumenu."""
    try:
        price_hist, bin_edges = np.histogram(df['Close'], bins=50, weights=df['Volume'])
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        poc_idx = np.argmax(price_hist)
        return price_hist, bin_centers, bin_centers[poc_idx]
    except: return [], [], 0

# --- 3. DASHBOARD LAYOUT ---

# SIDEBAR
with st.sidebar:
    st.markdown("### 🦅 BERLIN QUANT")
    ticker = st.text_input("SYMBOL", "EURUSD=X")
    tf = st.selectbox("TIMEFRAME", ["1m", "5m", "15m", "1h"], index=2)
    period_map = {"1m":"5d", "5m":"5d", "15m":"1mo", "1h":"3mo"}

# DATA FETCH
df_raw = get_data(ticker, tf, period_map.get(tf, "1mo"))
news_data, cal_data = get_intel_data(ticker)

if df_raw is not None:
    df, hurst = calculate_metrics(df_raw.copy())
    
    # --- TOP KPI BAR ---
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    
    last = df['Close'].iloc[-1]
    chg = last - df['Close'].iloc[-2]
    z = df['Z_Score'].iloc[-1]
    
    c1.metric("EURUSD (BERLIN)", f"{last:.5f}", f"{chg:.5f}")
    c2.metric("REŻIM (HURST)", f"{hurst:.2f}", "TREND" if hurst>0.55 else "RNG", delta_color="off")
    c3.metric("Z-SCORE (50)", f"{z:.2f}σ", "EXTREME" if abs(z)>2 else "OK", delta_color="inverse")
    c4.metric("VOLATILITY", f"{df['Close'].pct_change().std()*100*np.sqrt(252*24):.2f}%", "ANN")
    
    # Current Session Logic
    curr_hour = datetime.now(BERLIN_TZ).hour
    curr_sess = "ASIA" if (curr_hour>=22 or curr_hour<7) else "LONDON" if (curr_hour>=8 and curr_hour<13) else "LDN/NY" if (curr_hour>=13 and curr_hour<17) else "NY"
    c5.metric("ACTIVE SESSION", curr_sess, f"{datetime.now(BERLIN_TZ).strftime('%H:%M')}")
    
    quant_sig = "LONG" if z < -2 else "SHORT" if z > 2 else "WAIT"
    c6.metric("QUANT SIGNAL", quant_sig, "MEAN REV" if quant_sig != "WAIT" else "")

    # --- MAIN GRID ---
    col_main, col_right = st.columns([3, 1])
    
    with col_main:
        st.markdown(f"<div class='module-header'><span>CHART: {ticker} [{tf}]</span> <span>BERLIN TIMEZONE</span></div>", unsafe_allow_html=True)
        
        # OBLICZENIA PROFILU
        hist, bins, poc = get_volume_profile(df.tail(200))
        
        # PLOTLY SUBPLOTS (Wykres + Profil + CVD)
        fig = make_subplots(
            rows=2, cols=2, 
            shared_xaxes=True, shared_yaxes=True,
            column_widths=[0.85, 0.15], 
            row_heights=[0.75, 0.25],
            horizontal_spacing=0.01, vertical_spacing=0.02,
            specs=[[{}, {}], [{"colspan": 2}, None]]
        )
        
        # 1. Cena
        fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='OHLC', showlegend=False), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['Kalman'], line=dict(color='#ffeb3b', width=1.5), name='Kalman'), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['VWAP'], line=dict(color='#00e5ff', width=1, dash='dot'), name='VWAP'), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['Upper'], line=dict(color='gray', width=1), name='Band'), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['Lower'], line=dict(color='gray', width=1), fill='tonexty', fillcolor='rgba(255,255,255,0.02)', name='Band'), row=1, col=1)
        
        # 2. Profil (Side)
        fig.add_trace(go.Bar(x=hist, y=bins, orientation='h', marker=dict(color=hist, colorscale='Viridis', opacity=0.4), showlegend=False), row=1, col=2)
        fig.add_hline(y=poc, line_dash="dash", line_color="white", row=1, col=1, opacity=0.7, annotation_text="POC")
        
        # 3. CVD (Dolny)
        delta = np.where(df['Close']>=df['Open'], df['Volume'], -df['Volume'])
        cvd = np.cumsum(delta)
        fig.add_trace(go.Scatter(x=df.index, y=cvd, fill='tozeroy', line=dict(color='#2196f3', width=1), name='CVD'), row=2, col=1)
        
        fig.update_layout(template='plotly_dark', height=550, margin=dict(l=0,r=0,t=0,b=0), xaxis_rangeslider_visible=False, paper_bgcolor='#000', plot_bgcolor='#111')
        fig.update_xaxes(visible=False, row=1, col=2); fig.update_yaxes(visible=False, row=1, col=2); fig.update_yaxes(side="right", row=1, col=1)
        st.plotly_chart(fig, use_container_width=True)

    with col_right:
        # PRAWY PANEL (Liquidity & Heatmap)
        st.markdown(f"<div class='module-header'><span>LIQUIDITY & SEASONALITY</span></div>", unsafe_allow_html=True)
        
        tab1, tab2 = st.tabs(["LIQUIDITY", "HEATMAP"])
        
        with tab1:
            # Wykres sesji
            sess_vol = df.groupby('Session')['Volume'].mean()
            fig_sess = go.Figure(go.Bar(
                x=sess_vol.index, y=sess_vol.values, 
                marker_color=['#9c27b0', '#2196f3', '#ff9800', '#4caf50'] # Kolory dla Asia, London, NY, Other
            ))
            fig_sess.update_layout(template='plotly_dark', height=200, margin=dict(l=0,r=0,t=0,b=0), title="Avg Vol by Session", paper_bgcolor='#000')
            st.plotly_chart(fig_sess, use_container_width=True)
            
            st.info(f"Obecna sesja: **{curr_sess}**")
            
        with tab2:
            # FIX HEATMAP (Pivot Table zamiast unstack)
            try:
                if tf in ['15m', '1h']:
                    df['Day'] = df.index.day_name()
                    # Pivot table jest bezpieczniejsza niż unstack
                    pivot = pd.pivot_table(df, values='Close', index='Day', columns='Hour', aggfunc=lambda x: (x.iloc[-1]/x.iloc[0])-1)
                    # Sortowanie dni
                    days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']
                    pivot = pivot.reindex(days)
                    
                    fig_heat = go.Figure(data=go.Heatmap(
                        z=pivot.values, x=pivot.columns, y=pivot.index,
                        colorscale='RdBu', zmid=0, showscale=False
                    ))
                    fig_heat.update_layout(height=200, margin=dict(l=0,r=0,t=0,b=0), paper_bgcolor='#000')
                    st.plotly_chart(fig_heat, use_container_width=True)
                else:
                    st.caption("Wybierz M15/H1 dla mapy.")
            except Exception as e:
                st.error("Zbieranie danych...")

    # --- BOTTOM DOCK (NEWS & CALENDAR) ---
    st.markdown("---")
    c_news, c_cal = st.columns(2)
    
    with c_news:
        st.markdown(f"<div class='module-header'><span>NEWS FEED (BERLIN)</span></div>", unsafe_allow_html=True)
        if news_data:
            news_items = []
            for n in news_data:
                ts = datetime.fromtimestamp(n.get('providerPublishTime', 0)).strftime('%H:%M')
                news_items.append({"TIME": ts, "HEADLINE": n.get('title'), "SOURCE": n.get('publisher')})
            st.dataframe(pd.DataFrame(news_items), hide_index=True, use_container_width=True)
        else:
            st.caption("Brak newsów.")
            
    with c_cal:
        st.markdown(f"<div class='module-header'><span>ECONOMIC CALENDAR</span></div>", unsafe_allow_html=True)
        if isinstance(cal_data, dict) and cal_data:
            st.json(cal_data)
        elif hasattr(cal_data, 'empty') and not cal_data.empty:
            st.dataframe(cal_data, use_container_width=True)
        else:
            st.info("Brak danych kalendarza w Yahoo Finance (Limitation). Użyj zewnętrznego źródła dla danych makro.")

else:
    st.error("Błąd pobierania danych. Sprawdź symbol.")
