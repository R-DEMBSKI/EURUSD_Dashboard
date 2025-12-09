import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import linregress, entropy, zscore
from datetime import datetime, timedelta

# --- 1. KONFIGURACJA SYSTEMU (UI/UX) ---
st.set_page_config(layout="wide", page_title="QUANTUM ALPHA NODE", page_icon="🦅", initial_sidebar_state="collapsed")

# CSS: Professional Dark Theme & Typography
st.markdown("""
<style>
    /* Główny styl aplikacji */
    .stApp { background-color: #0b0c10; color: #c5c6c7; font-family: 'Roboto', sans-serif; }
    
    /* Ukrycie elementów standardowych */
    header, footer {visibility: hidden;}
    .block-container { padding-top: 0.5rem; max-width: 100%; }
    
    /* Stylizacja Tabs (Zakładek) */
    .stTabs [data-baseweb="tab-list"] { gap: 24px; }
    .stTabs [data-baseweb="tab"] {
        height: 50px; white-space: pre-wrap; background-color: #1f2833; border-radius: 4px 4px 0px 0px;
        gap: 1px; padding-top: 10px; padding-bottom: 10px; color: #fff;
    }
    .stTabs [aria-selected="true"] { background-color: #45a29e; color: #fff; }
    
    /* Metryki KPI */
    div[data-testid="stMetric"] {
        background-color: #1f2833; border: 1px solid #45a29e; padding: 10px; border-radius: 5px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
    }
    div[data-testid="stMetricLabel"] { font-size: 0.75rem !important; color: #66fcf1; text-transform: uppercase; letter-spacing: 1.2px; }
    div[data-testid="stMetricValue"] { font-size: 1.5rem !important; color: #fff; font-weight: 700; font-family: 'Consolas', monospace; }
    
    /* Tabele Newsów */
    .news-card {
        background-color: #1a1a1a; padding: 10px; margin-bottom: 8px; border-left: 3px solid #45a29e; border-radius: 2px;
    }
    .news-title { font-size: 0.9rem; font-weight: bold; color: #e0e0e0; }
    .news-meta { font-size: 0.7rem; color: #888; margin-top: 4px; }
    
    /* Nagłówki */
    h3, h4 { color: #66fcf1 !important; text-transform: uppercase; font-size: 1rem !important; border-bottom: 1px solid #333; padding-bottom: 5px; }
</style>
""", unsafe_allow_html=True)

# --- 2. SILNIK MATEMATYCZNY (QUANT ENGINE) ---

def calculate_hurst(series):
    """Oblicza wykładnik Hursta (H) - Miarę pamięci szeregu czasowego."""
    lags = range(2, 20)
    try:
        tau = [np.sqrt(np.std(np.subtract(series[lag:], series[:-lag]))) for lag in lags]
        poly = np.polyfit(np.log(lags), np.log(tau), 1)
        return poly[0] * 2.0
    except: return 0.5

def simple_kalman_filter(data, n_iter=5):
    """Estymator trendu (wygładzanie szumu) - Proxy dla filtru Kalmana."""
    return pd.Series(data).ewm(span=n_iter).mean()

def calculate_vwap(df):
    """Volume Weighted Average Price - Benchmark instytucjonalny."""
    v = df['Volume'].values
    tp = (df['High'] + df['Low'] + df['Close']) / 3
    return df.assign(VWAP=(tp * v).cumsum() / v.cumsum())

def get_market_profile(df, bins=50):
    """Tworzy profil wolumenu (rozkład ceny)."""
    price_hist, bin_edges = np.histogram(df['Close'], bins=bins, weights=df['Volume'])
    return price_hist, bin_edges

@st.cache_data(ttl=60) # Szybkie odświeżanie dla Daytradingu
def get_data_bundle(ticker, interval="15m", period="5d"):
    """Pobiera dane dla głównego interwału oraz kontekst makro."""
    
    # 1. Dane Główne (Intraday)
    try:
        df = yf.download(ticker, period=period, interval=interval, progress=False)
        if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.droplevel(1)
        if df.empty: return None, None, None
    except: return None, None, None

    # Inżynieria Cech
    df['Kalman'] = simple_kalman_filter(df['Close'].values)
    df = calculate_vwap(df)
    df['Returns'] = df['Close'].pct_change()
    df['Log_Ret'] = np.log(df['Close'] / df['Close'].shift(1))
    df['Vol_Ann'] = df['Log_Ret'].rolling(20).std() * np.sqrt(252 if interval=='1d' else 252*78) # Approx intraday vol
    
    # 2. Dane Contextowe (Daily - dla RRG i korelacji)
    context_tickers = f"{ticker} ^GSPC ^TNX DX-Y.NYB BTC-USD"
    df_context = yf.download(context_tickers, period="60d", interval="1d", group_by='ticker', progress=False)
    
    # 3. Pobranie obiektu Ticker dla Newsów i Kalendarza
    ticker_obj = yf.Ticker(ticker)
    
    return df, df_context, ticker_obj

# --- 3. DASHBOARD LOGIC ---

# Sidebar Konfiguracja
with st.sidebar:
    st.title("🎛️ STEROWNIA")
    ticker_input = st.text_input("AKTYWO", value="EURUSD=X")
    timeframe = st.selectbox("INTERWAŁ (Day Trading)", ["1m", "5m", "15m", "1h", "1d"], index=2)
    st.info("ℹ️ 1m/5m dostępne tylko dla ostatnich 7 dni (ograniczenie API Yahoo).")

# Pobieranie danych
with st.spinner('Synchronizacja z rynkiem...'):
    df, df_context, ticker_obj = get_data_bundle(ticker_input, interval=timeframe, period="5d" if timeframe in ["1m", "5m", "15m"] else "60d")

if df is not None and not df.empty:
    
    # Ostatnie ceny
    last_price = df['Close'].iloc[-1]
    prev_price = df['Close'].iloc[-2]
    delta = last_price - prev_price
    delta_pct = delta / prev_price
    
    # --- HEADER: KPI & ALERTS ---
    c1, c2, c3, c4, c5 = st.columns(5)
    
    c1.metric(f"{ticker_input} ({timeframe})", f"{last_price:.4f}", f"{delta_pct:.2%}")
    
    # Hurst (Fraktal)
    hurst = calculate_hurst(df['Close'].tail(50).values)
    hurst_state = "TRENDUJĄCY" if hurst > 0.55 else "KONSOLIDACJA" if hurst < 0.45 else "LOSOWY"
    c2.metric("Charakter Rynku (Hurst)", f"{hurst:.2f}", hurst_state, delta_color="off")
    
    # Volatility
    vol = df['Vol_Ann'].iloc[-1]
    c3.metric("Zmienność (Implied)", f"{vol*100:.2f}%", "Ryzyko")
    
    # Z-Score (Statystyczne Odchylenie)
    z_score = (last_price - df['Close'].rolling(50).mean().iloc[-1]) / df['Close'].rolling(50).std().iloc[-1]
    c4.metric("Z-Score (50 okresów)", f"{z_score:.2f}σ", "Overbought" if z_score > 2 else "Oversold" if z_score < -2 else "Fair", delta_color="inverse")
    
    # Bias (Trend Filtru Kalmana)
    kalman_slope = df['Kalman'].iloc[-1] - df['Kalman'].iloc[-2]
    bias = "BYCZY (Wzrosty)" if kalman_slope > 0 else "NIEDŹWIEDZI (Spadki)"
    c5.metric("Bias (Kalman Filter)", bias, f"{kalman_slope:.5f}", delta_color="normal")

    st.markdown("---")

    # --- MAIN TABS (STRUKTURA ZAKŁADEK) ---
    tab_chart, tab_intel, tab_quant = st.tabs(["📊 ANALIZA TECHNICZNA & WOLUMEN", "📰 INTELLIGENCE & NEWS", "🧮 QUANT LAB (RRG)"])

    # === TAB 1: WYKRES & WOLUMEN ===
    with tab_chart:
        col_main, col_profile = st.columns([5, 1])
        
        with col_main:
            # Subploty: Cena + Wolumen
            fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.8, 0.2])
            
            # 1. Świece
            fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='Cena'), row=1, col=1)
            # 2. Kalman (Trend)
            fig.add_trace(go.Scatter(x=df.index, y=df['Kalman'], mode='lines', line=dict(color='#f1c40f', width=2), name='Kalman Filter'), row=1, col=1)
            # 3. VWAP
            fig.add_trace(go.Scatter(x=df.index, y=df['VWAP'], mode='lines', line=dict(color='#00d2d3', width=1.5, dash='dot'), name='VWAP'), row=1, col=1)
            
            # 4. Wolumen (Kolorowany zmianą ceny)
            colors = ['#2ecc71' if r >= 0 else '#e74c3c' for r in df['Returns']]
            fig.add_trace(go.Bar(x=df.index, y=df['Volume'], marker_color=colors, name='Wolumen'), row=2, col=1)
            
            fig.update_layout(template='plotly_dark', height=600, margin=dict(l=0, r=0, t=10, b=0), showlegend=False, xaxis_rangeslider_visible=False, paper_bgcolor='#0b0c10', plot_bgcolor='#131416')
            st.plotly_chart(fig, use_container_width=True)
            
        with col_profile:
            st.markdown("### 🧬 PROFILE")
            # Volume Profile
            hist, bin_edges = get_market_profile(df.tail(100)) # Profil z ostatnich 100 świec
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
            
            # Znalezienie POC (Point of Control)
            poc_idx = np.argmax(hist)
            poc_price = bin_centers[poc_idx]
            
            fig_vp = go.Figure(go.Bar(
                x=hist, y=bin_centers, orientation='h',
                marker=dict(color=hist, colorscale='Tealgrn'), name='Volume Profile'
            ))
            fig_vp.add_hline(y=poc_price, line_dash="dash", line_color="white", annotation_text="POC")
            
            fig_vp.update_layout(template='plotly_dark', height=600, margin=dict(l=0,r=0,t=30,b=0), showlegend=False, xaxis_visible=False)
            st.plotly_chart(fig_vp, use_container_width=True)
            
            st.caption(f"**POC (Point of Control):** {poc_price:.4f}")
            st.caption("Poziom, na którym wymieniono najwięcej kontraktów. Działa jak magnes.")

    # === TAB 2: INTELLIGENCE & NEWS ===
    with tab_intel:
        c_news, c_cal = st.columns([2, 1])
        
        with c_news:
            st.markdown("### 📡 NAJNOWSZE WIADOMOŚCI (LIVE)")
            try:
                news_list = ticker_obj.news
                if news_list:
                    for news in news_list[:5]: # Top 5 newsów
                        # Konwersja timestamp
                        pub_time = datetime.fromtimestamp(news['providerPublishTime']).strftime('%Y-%m-%d %H:%M')
                        st.markdown(f"""
                        <div class="news-card">
                            <div class="news-title"><a href="{news['link']}" target="_blank" style="text-decoration:none; color:#e0e0e0;">{news['title']}</a></div>
                            <div class="news-meta">🕒 {pub_time} | 📢 {news['publisher']}</div>
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    st.info("Brak najnowszych wiadomości dla tego waloru w API.")
            except Exception as e:
                st.error(f"Błąd modułu newsów: {e}")

        with c_cal:
            st.markdown("### 📅 KALENDARZ EKONOMICZNY")
            st.caption("Najbliższe wydarzenia (Earnings/Splits)")
            try:
                cal = ticker_obj.calendar
                if cal is not None and not cal.empty:
                    st.dataframe(cal, use_container_width=True)
                else:
                    st.write("Brak nadchodzących wydarzeń korporacyjnych.")
            except:
                st.write("Dane kalendarza niedostępne.")
            
            st.markdown("---")
            st.markdown("### 🌍 FOREX HEATMAP (D1)")
            # Prosta heatmapa korelacji z kontekstu
            if 'Close' in df_context:
                # Obsługa MultiIndex dla wielu tickerów
                try:
                    # Sprawdzenie czy df_context ma poziomy (zależy od wersji yfinance)
                    if isinstance(df_context.columns, pd.MultiIndex):
                        corr_data = df_context.xs('Close', level=1, axis=1).corr()
                    else:
                        corr_data = df_context['Close'].corr() if 'Close' in df_context else df_context.corr()
                    
                    fig_corr = go.Figure(data=go.Heatmap(
                        z=corr_data.values, x=corr_data.columns, y=corr_data.columns,
                        colorscale='RdBu', zmin=-1, zmax=1
                    ))
                    fig_corr.update_layout(height=300, margin=dict(l=0,r=0,t=0,b=0))
                    st.plotly_chart(fig_corr, use_container_width=True)
                except Exception as e:
                    st.warning("Nie można wygenerować korelacji (zbyt mało danych).")

    # === TAB 3: QUANT LAB (RRG) ===
    with tab_quant:
        st.markdown("### 🌀 RELATIVE ROTATION GRAPH (RRG Concept)")
        st.markdown("""
        **Jak czytać ten wykres?**
        * **Oś X (RS-Ratio):** Siła trendu względem S&P 500. (Prawo = Silniejszy niż rynek).
        * **Oś Y (RS-Momentum):** Dynamika zmian tej siły. (Góra = Nabiera rozpędu).
        * **Ćwiartki:** 🟢 Leading (Liderzy) | 🟡 Weakening (Słabnący) | 🔴 Lagging (Maruderzy) | 🔵 Improving (Poprawiający się).
        """)
        
        # Symulacja RRG (prawdziwe wymagałoby bardzo złożonych obliczeń historycznych benchmarków)
        # Tutaj robimy uproszczony model: Return vs Volatility jako proxy dla Momentum vs Risk
        
        col_rrg, col_metrics = st.columns([3, 1])
        
        with col_rrg:
            # Przygotowanie danych do Scatter Plot
            rrg_data = []
            tickers_rrg = [ticker_input, "BTC-USD", "EURUSD=X", "GC=F", "^GSPC", "NVDA"]
            
            # Pobieramy dane snapshotowe (ostatnie 20 dni)
            data_rrg = yf.download(tickers_rrg, period="20d", interval="1d", progress=False)['Close']
            
            for t in tickers_rrg:
                if t in data_rrg.columns:
                    series = data_rrg[t]
                    ret = (series.iloc[-1] / series.iloc[0]) - 1 # Total Return
                    vol = series.pct_change().std() * np.sqrt(252) # Volatility
                    
                    # Normalizacja dla wykresu (symulacja środka wykresu)
                    rrg_data.append({
                        "Ticker": t,
                        "Return (Trend)": ret * 100,
                        "Risk (Vol)": vol * 100,
                        "Color": "cyan" if t == ticker_input else "gray"
                    })
            
            df_rrg = pd.DataFrame(rrg_data)
            
            fig_rrg = go.Figure()
            
            # Osie ćwiartek
            fig_rrg.add_vline(x=df_rrg['Return (Trend)'].mean(), line_dash="dot", line_color="#555")
            fig_rrg.add_hline(y=df_rrg['Risk (Vol)'].mean(), line_dash="dot", line_color="#555")
            
            fig_rrg.add_trace(go.Scatter(
                x=df_rrg['Return (Trend)'],
                y=df_rrg['Risk (Vol)'],
                mode='markers+text',
                text=df_rrg['Ticker'],
                textposition="top center",
                marker=dict(size=15, color=df_rrg['Color'].map(lambda x: '#45a29e' if x=='cyan' else '#666'))
            ))
            
            fig_rrg.update_layout(
                template='plotly_dark',
                title="Risk vs Return Map (20D)",
                xaxis_title="Zwrot (Trend) %",
                yaxis_title="Ryzyko (Zmienność) %",
                height=500,
                paper_bgcolor='#0b0c10',
                plot_bgcolor='#131416'
            )
            st.plotly_chart(fig_rrg, use_container_width=True)

        with col_metrics:
            st.markdown("#### 🔬 STATYSTYKA")
            st.info("Powyższa mapa pomaga zidentyfikować, czy handlujesz aktywem, które jest aktualnie 'w grze'. Idealnie chcesz być w prawym dolnym rogu (Wysoki Zwrot, Niskie Ryzyko) lub prawym górnym (Momentum).")
            
            entropy_val = entropy(df['Close'].pct_change().dropna().abs())
            st.metric("Entropia Shannona", f"{entropy_val:.2f}", "Poziom Chaosu")
            
            if entropy_val > 4.5:
                st.warning("⚠️ Rynek wysoce chaotyczny! Zmniejsz stawki.")
            else:
                st.success("✅ Rynek uporządkowany. Systemy trendowe skuteczne.")

else:
    st.error("Brak danych. Sprawdź symbol (np. EURUSD=X) lub połączenie internetowe.")
