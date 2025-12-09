import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import zscore
from datetime import datetime, timedelta

# --- 1. KONFIGURACJA UI (QUANT DARK MODE) ---
st.set_page_config(layout="wide", page_title="QUANTUM SIMONS NODE", page_icon="🦅", initial_sidebar_state="collapsed")

st.markdown("""
<style>
    /* Globalne tło i czcionka */
    .stApp { background-color: #080808; color: #d0d0d0; font-family: 'Roboto Mono', monospace; }
    
    /* Ukrycie standardowych elementów Streamlit */
    header, footer {visibility: hidden;}
    .block-container { padding-top: 0.5rem; max-width: 100%; }
    
    /* Zakładki (Tabs) - Styl Bloomberg */
    .stTabs [data-baseweb="tab-list"] { gap: 10px; background-color: #080808; padding: 10px 0; }
    .stTabs [data-baseweb="tab"] {
        height: 40px; white-space: pre-wrap; background-color: #1a1a1a; border: 1px solid #333;
        border-radius: 4px; gap: 1px; color: #888; font-size: 0.8rem;
    }
    .stTabs [aria-selected="true"] { background-color: #007bff; color: #fff; border-color: #007bff; }
    
    /* Karty KPI (Glassmorphism) */
    div[data-testid="stMetric"] {
        background-color: #111; border-left: 4px solid #007bff; padding: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.5);
    }
    div[data-testid="stMetricLabel"] { font-size: 0.7rem !important; color: #666; letter-spacing: 1px; }
    div[data-testid="stMetricValue"] { font-size: 1.4rem !important; color: #eee; font-weight: 700; }
    
    /* Wiadomości */
    .news-item { 
        padding: 10px; border-bottom: 1px solid #222; margin-bottom: 5px; 
        transition: background 0.3s;
    }
    .news-item:hover { background-color: #1a1a1a; }
    .news-link { color: #00bcd4; text-decoration: none; font-weight: bold; font-size: 0.9rem; }
    .news-meta { color: #555; font-size: 0.75rem; margin-top: 4px; }
</style>
""", unsafe_allow_html=True)

# --- 2. SILNIK MATEMATYCZNY (NAPRAWIONY CACHING) ---

@st.cache_data(ttl=60) # Cache tylko prostych danych (DataFrames)
def get_price_data(ticker, interval, period):
    """Pobiera wyłącznie dane cenowe (Safe for caching)."""
    try:
        df = yf.download(ticker, period=period, interval=interval, progress=False)
        if df.empty: return None
        # Obsługa MultiIndex (dla nowych wersji yfinance)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.droplevel(1)
        return df
    except Exception as e:
        return None

@st.cache_data(ttl=300) 
def get_market_intel(ticker):
    """Pobiera Newsy i Kalendarz jako proste słowniki (Serializable)."""
    try:
        t = yf.Ticker(ticker)
        news = t.news if hasattr(t, 'news') else []
        calendar = t.calendar if hasattr(t, 'calendar') else None
        
        # Konwersja kalendarza do prostego formatu
        cal_data = {}
        if calendar is not None and isinstance(calendar, dict):
            cal_data = calendar
        elif calendar is not None:
            cal_data = calendar.to_dict()
            
        return news, cal_data
    except:
        return [], {}

def calculate_indicators(df):
    """Oblicza wskaźniki Quant (Kalman, Hurst, CVD)."""
    if df is None: return None
    
    # 1. Filtr Kalmana (Estymacja Trendu)
    df['Kalman'] = df['Close'].ewm(span=5).mean() # Szybka estymacja
    
    # 2. VWAP
    v = df['Volume'].values
    tp = (df['High'] + df['Low'] + df['Close']) / 3
    df['VWAP'] = (tp * v).cumsum() / v.cumsum()
    
    # 3. CVD (Cumulative Volume Delta - Symulacja Order Flow)
    # Jeśli Close > Open -> Zakładamy przewagę kupujących (Delta +)
    # Jeśli Close < Open -> Zakładamy przewagę sprzedających (Delta -)
    df['Delta'] = np.where(df['Close'] >= df['Open'], df['Volume'] * 0.6, -df['Volume'] * 0.6)
    df['CVD'] = df['Delta'].cumsum()
    
    return df

def calculate_hurst(series):
    """Oblicza Wykładnik Hursta (Fraktalność)."""
    try:
        lags = range(2, 20)
        tau = [np.sqrt(np.std(np.subtract(series[lag:], series[:-lag]))) for lag in lags]
        poly = np.polyfit(np.log(lags), np.log(tau), 1)
        return poly[0] * 2.0
    except: return 0.5

def get_market_profile(df, bins=40):
    """Volume Profile."""
    try:
        price_hist, bin_edges = np.histogram(df['Close'], bins=bins, weights=df['Volume'])
        return price_hist, bin_edges
    except: return [], []

# --- 3. DASHBOARD LOGIC ---

# Sidebar
with st.sidebar:
    st.header("🎛️ STEROWNIA ALGO")
    ticker_input = st.text_input("SYMBOL", value="EURUSD=X")
    tf = st.selectbox("INTERWAŁ", ["1m", "5m", "15m", "1h", "1d"], index=2)
    st.caption("Pamiętaj: 1m dostępne tylko dla ostatnich 7 dni.")

# Główna pętla
try:
    # A. Pobieranie Danych
    period_map = {"1m": "5d", "5m": "5d", "15m": "10d", "1h": "60d", "1d": "1y"}
    df_raw = get_price_data(ticker_input, tf, period_map.get(tf, "1mo"))
    news_list, calendar_data = get_market_intel(ticker_input)
    
    if df_raw is not None:
        df = calculate_indicators(df_raw.copy())
        
        last_price = df['Close'].iloc[-1]
        change = (last_price - df['Close'].iloc[-2]) / df['Close'].iloc[-2]
        
        # --- PANEL KPI (METRYKI) ---
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric(f"{ticker_input}", f"{last_price:.5f}", f"{change:.2%}")
        
        # Hurst (Fraktal)
        hurst = calculate_hurst(df['Close'].tail(50).values)
        hurst_state = "TRENDUJĄCY" if hurst > 0.55 else "KONSOLA" if hurst < 0.45 else "SZUM"
        c2.metric("Fraktal (Hurst)", f"{hurst:.2f}", hurst_state, delta_color="off")
        
        # Z-Score
        z = (last_price - df['Close'].rolling(50).mean().iloc[-1]) / df['Close'].rolling(50).std().iloc[-1]
        c3.metric("Z-Score (50)", f"{z:.2f}σ", "Drogie" if z>2 else "Tanie" if z<-2 else "Fair", delta_color="inverse")
        
        # Order Flow Momentum
        delta_last = df['Delta'].iloc[-1]
        c4.metric("Volume Delta (Flow)", f"{delta_last/1000:.1f}K", "Kupujący" if delta_last > 0 else "Sprzedający")
        
        # Entropia / Zmienność
        vol = df['Close'].pct_change().std() * np.sqrt(252*24) # Approx intraday vol
        c5.metric("Zmienność", f"{vol*100:.2f}%", "Ryzyko")

        st.markdown("---")

        # --- ZAKŁADKI (TABS) ---
        tab_chart, tab_intel, tab_quant = st.tabs(["📈 SYSTEM TRANSAKCYJNY", "🌍 INTELLIGENCE (NEWS)", "🧮 QUANT LAB (RESEARCH)"])

        # === TAB 1: WYKRESY (Full Screen) ===
        with tab_chart:
            col_main, col_side = st.columns([4, 1])
            
            with col_main:
                # Subploty: Cena + CVD (Order Flow)
                fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.02, row_heights=[0.75, 0.25])
                
                # 1. Świece
                fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='Cena'), row=1, col=1)
                
                # 2. Kalman (Złoty)
                fig.add_trace(go.Scatter(x=df.index, y=df['Kalman'], mode='lines', line=dict(color='#ffd700', width=2), name='Kalman Filter'), row=1, col=1)
                
                # 3. VWAP (Turkus)
                fig.add_trace(go.Scatter(x=df.index, y=df['VWAP'], mode='lines', line=dict(color='#00e5ff', width=1, dash='dot'), name='VWAP'), row=1, col=1)

                # 4. CVD (Cumulative Volume Delta) - Dolny panel
                fig.add_trace(go.Scatter(x=df.index, y=df['CVD'], mode='lines', fill='tozeroy', 
                                         line=dict(color='#bdc3c7', width=1), name='Cum. Delta'), row=2, col=1)

                fig.update_layout(template='plotly_dark', height=650, margin=dict(l=0,r=0,t=10,b=0), 
                                  xaxis_rangeslider_visible=False, paper_bgcolor='#080808', plot_bgcolor='#111')
                st.plotly_chart(fig, use_container_width=True)

            with col_side:
                st.markdown("### 🧬 PROFILE")
                # Volume Profile z prawej
                hist, bin_edges = get_market_profile(df.tail(150))
                bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
                poc_idx = np.argmax(hist) if len(hist) > 0 else 0
                poc_price = bin_centers[poc_idx] if len(bin_centers) > 0 else 0
                
                fig_vp = go.Figure(go.Bar(
                    x=hist, y=bin_centers, orientation='h',
                    marker=dict(color=hist, colorscale='Viridis'), name='Volume'
                ))
                fig_vp.add_hline(y=poc_price, line_dash="dash", line_color="white", annotation_text="POC")
                fig_vp.update_layout(template='plotly_dark', height=650, margin=dict(l=0,r=0,t=30,b=0), 
                                     xaxis_visible=False, showlegend=False, paper_bgcolor='#080808')
                st.plotly_chart(fig_vp, use_container_width=True)

        # === TAB 2: INTELLIGENCE ===
        with tab_intel:
            c_news, c_cal = st.columns([2, 1])
            with c_news:
                st.subheader(f"📰 WIADOMOŚCI: {ticker_input}")
                if news_list:
                    for n in news_list[:7]:
                        pub_time = datetime.fromtimestamp(n.get('providerPublishTime', 0)).strftime('%H:%M %d-%m')
                        st.markdown(f"""
                        <div class="news-item">
                            <a href="{n.get('link')}" target="_blank" class="news-link">{n.get('title')}</a>
                            <div class="news-meta">🕒 {pub_time} | 📡 {n.get('publisher')}</div>
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    st.info("Brak wiadomości w feedzie Yahoo.")
            
            with c_cal:
                st.subheader("📅 KALENDARZ")
                if calendar_data:
                    # Próba konwersji na DataFrame dla ładnego wyświetlania
                    try:
                        cal_df = pd.DataFrame(calendar_data)
                        st.dataframe(cal_df, use_container_width=True)
                    except:
                        st.json(calendar_data) # Fallback
                else:
                    st.write("Brak nadchodzących wydarzeń.")

        # === TAB 3: QUANT LAB ===
        with tab_quant:
            st.markdown("### 🕒 HEATMAPA SEZONOWOŚCI (GODZINOWA)")
            st.caption("Analiza, w których godzinach cena statystycznie rośnie (Zielony) lub spada (Czerwony).")
            
            # Przygotowanie danych do Heatmapy (wymaga więcej danych niż 5 dni, więc symulujemy logikę na dostępnych)
            # W pełnej wersji pobrałbyś 60 dni danych 1h w tle.
            if tf in ['1h', '15m']:
                df['Hour'] = df.index.hour
                df['Day'] = df.index.day_name()
                # Sortowanie dni
                days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']
                pivot = df.groupby(['Day', 'Hour'])['Close'].pct_change().mean().unstack()
                pivot = pivot.reindex(days_order)
                
                fig_heat = go.Figure(data=go.Heatmap(
                    z=pivot.values, x=pivot.columns, y=pivot.index,
                    colorscale='RdBu', zmid=0
                ))
                fig_heat.update_layout(template='plotly_dark', height=400, title="Średnia Zmiana (%) wg Godziny i Dnia")
                st.plotly_chart(fig_heat, use_container_width=True)
            else:
                st.warning("Heatmapa wymaga interwału 1h lub 15m.")

            st.markdown("### 🧠 LOGIKA SYGNAŁÓW (JIM SIMONS STYLE)")
            col_s1, col_s2 = st.columns(2)
            
            with col_s1:
                st.info(f"""
                **Analiza Hurst ({hurst:.2f}):**
                * Jeśli H > 0.55: Rynek ma pamięć (Trend). **Graj wybicia (Breakouts).**
                * Jeśli H < 0.45: Rynek wraca do średniej. **Kupuj wsparcia, sprzedawaj opory.**
                * Aktualnie: **{hurst_state}**
                """)
            
            with col_s2:
                st.info(f"""
                **Analiza Z-Score ({z:.2f}σ):**
                * Mierzy odchylenie ceny od 'Fair Value'.
                * Powyżej 2.0σ: Rynek wykupiony statystycznie (Szukaj Shorta).
                * Poniżej -2.0σ: Rynek wyprzedany statystycznie (Szukaj Longa).
                """)

    else:
        st.error("Nie udało się pobrać danych. Sprawdź symbol (np. EURUSD=X, BTC-USD) lub połączenie.")
except Exception as e:
    st.error(f"Krytyczny błąd systemu: {e}")
