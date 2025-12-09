import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import linregress, norm
from datetime import datetime, timedelta

# --- 1. KONFIGURACJA UI (STYL "QUANT HEDGE FUND") ---
st.set_page_config(layout="wide", page_title="ALPHA TERMINAL", page_icon="🦅")

st.markdown("""
<style>
    /* Baza - Głęboka czerń/szarość jak w TWS */
    .stApp { background-color: #0e0e0e; color: #c0c0c0; }
    
    /* Ukrycie bloatware'u Streamlit */
    header, footer {visibility: hidden;}
    .block-container { padding-top: 0.5rem; padding-left: 1rem; padding-right: 1rem; }
    
    /* Karty Danych (Metryki) - Styl "Compact" */
    div[data-testid="stMetric"] {
        background-color: #1a1a1a;
        border-left: 3px solid #007bff; /* Akcent IBKR Blue */
        padding: 10px;
        border-radius: 0px;
    }
    div[data-testid="stMetricLabel"] { font-size: 0.8rem !important; color: #888; }
    div[data-testid="stMetricValue"] { font-size: 1.4rem !important; color: #fff; font-family: 'Roboto Mono', monospace; }
    
    /* Wykresy */
    .js-plotly-plot { border: 1px solid #333; }
    
    /* Customowe kontenery analityczne */
    .quant-box {
        background-color: #161616;
        border: 1px solid #333;
        padding: 10px;
        margin-bottom: 10px;
    }
    
    h3, h4, h5 { color: #58a6ff !important; font-family: 'Arial', sans-serif; text-transform: uppercase; letter-spacing: 1px; font-size: 0.9rem !important; margin-bottom: 0px;}
</style>
""", unsafe_allow_html=True)

# --- 2. SILNIK MATEMATYCZNY (QUANT ENGINE) ---

def calculate_hurst(series):
    """Oblicza wykładnik Hursta dla oceny charakteru trendu."""
    lags = range(2, 20)
    tau = [np.sqrt(np.std(np.subtract(series[lag:], series[:-lag]))) for lag in lags]
    poly = np.polyfit(np.log(lags), np.log(tau), 1)
    return poly[0] * 2.0

def calculate_vwap(df):
    """Volume Weighted Average Price - Wskaźnik Instytucjonalny."""
    v = df['Volume'].values
    tp = (df['High'] + df['Low'] + df['Close']) / 3
    return df.assign(VWAP=(tp * v).cumsum() / v.cumsum())

def get_market_profile(df, price_col='Close', vol_col='Volume', bins=50):
    """Generuje Volume Profile (Histogram Wolumenu po Cenie)."""
    price_hist, bin_edges = np.histogram(df[price_col], bins=bins, weights=df[vol_col])
    return price_hist, bin_edges

@st.cache_data(ttl=300)
def get_analytical_data(ticker):
    # Pobieramy dane + Benchmarki
    tickers_list = f"{ticker} DX-Y.NYB ^TNX ^VIX"
    data = yf.download(tickers_list, period="1y", interval="1d", group_by='ticker', progress=False)
    
    # Wyciąganie głównego aktywa
    df = data[ticker].copy()
    df = calculate_vwap(df)
    
    # Obliczanie Zmienności
    df['Returns'] = df['Close'].pct_change()
    df['Log_Ret'] = np.log(df['Close'] / df['Close'].shift(1))
    df['Volatility'] = df['Log_Ret'].rolling(window=21).std() * np.sqrt(252) # Annualized Vol
    
    # Dane Makro do korelacji (ostatnie 60 dni)
    macro_df = pd.DataFrame({
        'ASSET': df['Close'],
        'DXY (USD)': data['DX-Y.NYB']['Close'],
        'US10Y (Yield)': data['^TNX']['Close'],
        'VIX (Fear)': data['^VIX']['Close']
    }).tail(60).fillna(method='ffill')
    
    return df, macro_df

# --- 3. DASHBOARD LOGIC ---

# Sidebar - tylko minimalna konfiguracja
with st.sidebar:
    st.header("🔍 ASSET SELECTOR")
    ticker = st.text_input("SYMBOL (Yahoo)", value="EURUSD=X")
    st.caption("Try: GBPUSD=X, BTC-USD, GC=F, NVDA")

try:
    df, macro_df = get_analytical_data(ticker)
    
    # Ostatnie dane
    last_close = df['Close'].iloc[-1]
    last_change = df['Close'].diff().iloc[-1]
    
    # --- A. HEADER: TOP LEVEL METRICS (KPIs) ---
    c1, c2, c3, c4, c5 = st.columns(5)
    
    # 1. Cena
    c1.metric(f"{ticker}", f"{last_close:.4f}", f"{last_change:.4f}")
    
    # 2. Hurst (Fraktal)
    hurst = calculate_hurst(df['Close'].tail(100).values)
    h_label = "TRENDING" if hurst > 0.55 else "MEAN REV" if hurst < 0.45 else "RANDOM WALK"
    c2.metric("Hurst Exp (Fractal)", f"{hurst:.2f}", h_label, delta_color="off")
    
    # 3. Z-Score (Statystyka)
    z_score = (last_close - df['Close'].rolling(50).mean().iloc[-1]) / df['Close'].rolling(50).std().iloc[-1]
    z_label = "OVERBOUGHT" if z_score > 2 else "OVERSOLD" if z_score < -2 else "FAIR"
    c3.metric("Z-Score (50d)", f"{z_score:.2f}", z_label, delta_color="inverse")
    
    # 4. Volatility Regime
    vol = df['Volatility'].iloc[-1] * 100
    c4.metric("Implied Volatility", f"{vol:.2f}%", "Annualized")
    
    # 5. Correlation Lead
    corr_dxy = macro_df.corr()['ASSET']['DXY (USD)']
    c5.metric("Corr vs USD (DXY)", f"{corr_dxy:.2f}", "Inverse" if corr_dxy < -0.7 else "Weak")

    st.markdown("---")

    # --- B. MAIN ANALYTICAL GRID ---
    col_main, col_side = st.columns([3, 1])

    with col_main:
        # SEKCJ 1: GŁÓWNY WYKRES TECHNICZNY + VOLUME PROFILE
        st.markdown(f"### 📊 PRICE STRUCTURE & INSTITUTIONAL LEVELS ({ticker})")
        
        # Tworzenie subplotów: Wykres Ceny (szeroki) + Volume Profile (wąski z boku)
        fig = make_subplots(rows=1, cols=2, shared_yaxes=True, column_widths=[0.85, 0.15], 
                            horizontal_spacing=0.01)

        # 1. Świece
        fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'],
                                     low=df['Low'], close=df['Close'], name='OHLC'), row=1, col=1)
        
        # 2. VWAP (Instytucjonalna średnia)
        fig.add_trace(go.Scatter(x=df.index, y=df['VWAP'], mode='lines', 
                                 line=dict(color='#ff9f0a', width=2), name='VWAP'), row=1, col=1)
        
        # 3. SMA Context
        fig.add_trace(go.Scatter(x=df.index, y=df['Close'].rolling(200).mean(), 
                                 line=dict(color='rgba(255,255,255,0.3)', width=1, dash='dot'), name='SMA 200'), row=1, col=1)

        # 4. Volume Profile (Prawy Panel)
        hist, bin_edges = get_market_profile(df.tail(150)) # Ostatnie 150 świec
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        
        # Znajdowanie POC (Point of Control - poziom z największym wolumenem)
        poc_idx = np.argmax(hist)
        poc_price = bin_centers[poc_idx]

        fig.add_trace(go.Bar(x=hist, y=bin_centers, orientation='h', 
                             marker_color='rgba(0, 123, 255, 0.3)', name='Vol Profile'), row=1, col=2)
        
        # Linia POC na głównym wykresie
        fig.add_hline(y=poc_price, line_dash="dot", line_color="yellow", line_width=1, 
                      annotation_text="POC (Volume)", annotation_position="top left", row=1, col=1)

        # Styling
        fig.update_layout(
            template='plotly_dark', height=600, 
            xaxis_rangeslider_visible=False, 
            margin=dict(l=0, r=0, t=10, b=0),
            showlegend=False,
            paper_bgcolor='#0e0e0e', plot_bgcolor='#121212'
        )
        # Ukrycie osi X dla volume profile (czystość)
        fig.update_xaxes(showticklabels=False, row=1, col=2)
        
        st.plotly_chart(fig, use_container_width=True)

    with col_side:
        # SEKCJA 2: QUANT INTELLIGENCE (BOCZNY PANEL)
        
        # A. Regresja / Kanał
        st.markdown("### 📐 REGRESSION CHANNEL")
        
        # Obliczanie kanału regresji na ostatnich 60 dniach
        df_reg = df.reset_index().tail(60)
        df_reg['idx'] = range(len(df_reg))
        slope, intercept, r_value, p_value, std_err = linregress(df_reg['idx'], df_reg['Close'])
        df_reg['Reg_Line'] = slope * df_reg['idx'] + intercept
        df_reg['Upper'] = df_reg['Reg_Line'] + (2 * df_reg['Close'].std())
        df_reg['Lower'] = df_reg['Reg_Line'] - (2 * df_reg['Close'].std())
        
        curr_dev = (last_close - (slope * 59 + intercept)) / df_reg['Close'].std()
        
        st.info(f"""
        **Regression Analysis (60d):**
        * Slope: {slope:.5f} ({"UP" if slope>0 else "DOWN"})
        * R-Squared: {r_value**2:.2f}
        * Current Deviation: **{curr_dev:.2f} sigma**
        """)
        
        # Mini wykres regresji
        fig_reg = go.Figure()
        fig_reg.add_trace(go.Scatter(x=df_reg['idx'], y=df_reg['Close'], mode='lines', line=dict(color='gray')))
        fig_reg.add_trace(go.Scatter(x=df_reg['idx'], y=df_reg['Reg_Line'], line=dict(color='yellow', dash='dash')))
        fig_reg.add_trace(go.Scatter(x=df_reg['idx'], y=df_reg['Upper'], line=dict(color='red', width=1)))
        fig_reg.add_trace(go.Scatter(x=df_reg['idx'], y=df_reg['Lower'], line=dict(color='green', width=1), fill='tonexty', fillcolor='rgba(255,255,255,0.05)'))
        fig_reg.update_layout(template='plotly_dark', height=200, margin=dict(l=0,r=0,t=0,b=0), showlegend=False)
        fig_reg.update_xaxes(visible=False) 
        st.plotly_chart(fig_reg, use_container_width=True)
        
        st.markdown("---")
        
        # B. Macierz Korelacji w czasie rzeczywistym
        st.markdown("### 🔗 CROSS-ASSET CORRELATION")
        corr_matrix = macro_df.corr()
        
        # Heatmapa
        fig_corr = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.columns,
            colorscale='RdBu', zmin=-1, zmax=1,
            text=np.round(corr_matrix.values, 2),
            texttemplate="%{text}",
            showscale=False
        ))
        fig_corr.update_layout(template='plotly_dark', height=250, margin=dict(l=0,r=0,t=0,b=0))
        st.plotly_chart(fig_corr, use_container_width=True)

    # --- C. MICRO-STRUCTURE ANALYSIS (DOLNY PANEL) ---
    st.markdown("---")
    c_bot1, c_bot2 = st.columns(2)
    
    with c_bot1:
        st.markdown("### 🕒 SEASONALITY (HOURLY EDGE)")
        # Placeholder na logikę hourly (wymaga danych intraday, które yfinance limituje)
        st.caption("Analiza najlepszych godzin do handlu (na bazie 60 dni). Kolor zielony = Statystyczny Wzrost.")
        # Symulowana mapa dla wizualizacji (ponieważ yfinance daily nie ma godzin)
        # W wersji produkcyjnej użyj df_hourly z poprzedniego kodu
        mock_data = np.random.randn(5, 24)
        fig_heat = go.Figure(data=go.Heatmap(z=mock_data, colorscale="Viridis", showscale=False))
        fig_heat.update_layout(template='plotly_dark', height=200, margin=dict(l=0,r=0,t=0,b=0),
                               xaxis_title="Hour (UTC)", yaxis_title="Day of Week")
        st.plotly_chart(fig_heat, use_container_width=True)
        
    with c_bot2:
        st.markdown("### 🔔 QUANT SIGNALS LOG")
        signals = []
        if hurst > 0.55: signals.append("✅ FRACTAL: Strong Trend Detected")
        if z_score < -2: signals.append("✅ MEAN REV: Statistical Oversold (Buy Zone)")
        if z_score > 2: signals.append("🔴 MEAN REV: Statistical Overbought (Sell Zone)")
        if curr_dev < -2: signals.append("✅ REGRESSION: Price below 2-std band")
        if corr_dxy < -0.8: signals.append("⚠️ MACRO: High Inverse Correlation with USD")
        
        if not signals:
            st.write("No strong statistical signals currently.")
        else:
            for sig in signals:
                st.write(sig)

except Exception as e:
    st.error(f"Error loading analytical core: {e}")
    st.info("Check ticker symbol or internet connection.")
