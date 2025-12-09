import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from scipy.stats import norm
from sklearn.mixture import GaussianMixture
from datetime import datetime
import pytz

# --- 1. KONFIGURACJA UI (POLSKI TERMINAL) ---
st.set_page_config(layout="wide", page_title="EURUSD COMMAND CENTER", page_icon="🦅", initial_sidebar_state="expanded")

st.markdown("""
<style>
    /* STYL GLOBALNY - GŁĘBOKA CZERŃ */
    .stApp { background-color: #000000; color: #e0e0e0; font-family: 'Roboto', sans-serif; }
    
    /* MODUŁY I KONTENERY */
    .block-container { padding-top: 1rem; padding-bottom: 5rem; max-width: 100%; }
    header, footer {visibility: hidden;}
    
    /* KPI METRICS (LIVE DATA) */
    div[data-testid="stMetric"] {
        background-color: #0b0b0b; 
        border: 1px solid #333; 
        padding: 10px; 
        border-radius: 4px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
    }
    div[data-testid="stMetricLabel"] { font-size: 0.7rem !important; color: #888; text-transform: uppercase; letter-spacing: 1px; }
    div[data-testid="stMetricValue"] { font-size: 1.4rem !important; color: #fff; font-weight: 700; font-family: 'Consolas', monospace; }
    
    /* OPISY KONCEPCYJNE */
    .concept-box {
        background-color: #1a1a1a;
        border-left: 3px solid #00bcd4;
        padding: 15px;
        margin-top: 10px;
        margin-bottom: 20px;
        font-size: 0.85rem;
        color: #ccc;
    }
    .concept-title { color: #00bcd4; font-weight: bold; font-size: 0.9rem; margin-bottom: 5px; }

    /* ZAKŁADKI */
    .stTabs [data-baseweb="tab-list"] { gap: 5px; background-color: #000; padding: 10px 0; }
    .stTabs [data-baseweb="tab"] {
        height: 40px; background-color: #111; color: #aaa; border: 1px solid #333; border-radius: 4px;
    }
    .stTabs [aria-selected="true"] { background-color: #00bcd4; color: #000; font-weight: bold; border-color: #00bcd4; }
    
    /* LIQUIDITY BAR */
    .liq-bar { height: 8px; width: 100%; background: linear-gradient(90deg, #333 0%, #00bcd4 50%, #333 100%); border-radius: 4px; margin-top: 5px; }
</style>
""", unsafe_allow_html=True)

# --- 2. SILNIK DANYCH (LIVE & HISTORY) ---

# Strefy Czasowe
TZ_BERLIN = pytz.timezone('Europe/Berlin')
TZ_LONDON = pytz.timezone('Europe/London')
TZ_NY = pytz.timezone('America/New_York')
TZ_TOKYO = pytz.timezone('Asia/Tokyo')

@st.cache_data(ttl=30)
def get_live_data(ticker="EURUSD=X"):
    """Pobiera świeże dane LIVE do paska KPI."""
    try:
        # Pobieramy ostatni dzień (1m) dla precyzji
        df = yf.download(ticker, period="1d", interval="1m", progress=False)
        if df.empty: return None
        if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.droplevel(1)
        return df.iloc[-1] # Zwraca ostatnią świecę
    except: return None

@st.cache_data(ttl=600)
def get_analysis_data(ticker, period="1mo", interval="1h"):
    """Pobiera dane do wykresów analitycznych (jeśli nie ma pliku)."""
    try:
        df = yf.download(ticker, period=period, interval=interval, progress=False)
        if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.droplevel(1)
        return df
    except: return None

def load_uploaded_csv(uploaded_file):
    """Parsuje plik użytkownika (Format Myfxbook/MT4)."""
    try:
        df = pd.read_csv(uploaded_file, skiprows=1) # Pomijamy nagłówek Myfxbook
        df.columns = [c.strip() for c in df.columns]
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values('Date').set_index('Date')
        return df
    except Exception as e:
        return None

# --- 3. ALGORYTMY QUANTOWE ---

def calculate_probability_well(df):
    """Oblicza krzywą Gaussa dla Live Struktury."""
    v = df['Volume'].values
    tp = (df['High'] + df['Low'] + df['Close']) / 3
    # VWAP jako "Fair Value"
    df['VWAP'] = (tp * v).cumsum() / v.cumsum()
    vwap = df['VWAP'].iloc[-1]
    std = df['Close'].std()
    last_price = df['Close'].iloc[-1]
    z_score = (last_price - vwap) / std
    return vwap, std, last_price, z_score

def analyze_regimes_ml(df):
    """Analiza reżimów (GMM Clustering) dla danych historycznych."""
    data = df.copy()
    data['Log_Ret'] = np.log(data['Close'] / data['Close'].shift(1))
    data['Range'] = (data['High'] - data['Low']) / data['Close']
    data = data.dropna()
    
    # Model GMM
    X = data[['Log_Ret', 'Range']].values
    model = GaussianMixture(n_components=3, random_state=42).fit(X)
    data['Regime'] = model.predict(X)
    
    # Sortowanie: 0=Niska Zmienność, 2=Wysoka
    vol_means = data.groupby('Regime')['Range'].mean().sort_values()
    mapping = {old: new for new, old in enumerate(vol_means.index)}
    data['Regime'] = data['Regime'].map(mapping)
    return data

def get_liquidity_status():
    """Zwraca poziom płynności i aktywne sesje."""
    now = datetime.now(TZ_BERLIN)
    h = now.hour
    
    sessions = []
    liquidity = 0
    
    if 9 <= h < 18: 
        sessions.append("LONDYN")
        liquidity += 40
    if 14 <= h < 22: 
        sessions.append("NOWY JORK")
        liquidity += 50
    if h >= 23 or h < 8: 
        sessions.append("TOKIO/SYDNEY")
        liquidity += 10
        
    liq_level = min(liquidity, 100)
    status = " | ".join(sessions) if sessions else "OFF-HOURS"
    return status, liq_level

# --- 4. LAYOUT APLIKACJI ---

# PANEL BOCZNY (USTAWIENIA)
with st.sidebar:
    st.markdown("### ⚙️ STEROWANIE")
    ticker = st.text_input("SYMBOL LIVE", "EURUSD=X")
    st.markdown("---")
    st.markdown("### 📂 LABORATORIUM DANYCH")
    st.info("Wgraj tutaj plik CSV z historią, aby aktywować analizę statystyczną w zakładce 'Laboratorium'.")
    uploaded_file = st.file_uploader("Wgraj plik (CSV)", type=['csv'])

# POBIERANIE DANYCH LIVE (ZAWSZE AKTYWNE)
live_candle = get_live_data(ticker)
liq_status, liq_val = get_liquidity_status()
now_berlin = datetime.now(TZ_BERLIN)

# --- SEKCJ 1: GÓRNY PASEK KPI (LIVE) ---
if live_candle is not None:
    c1, c2, c3, c4 = st.columns(4)
    
    # 1. Cena Live
    price = live_candle['Close']
    chg_pips = (live_candle['Close'] - live_candle['Open']) * 10000
    c1.metric("CENA LIVE (SPOT)", f"{price:.5f}", f"{chg_pips:.1f} pips")
    
    # 2. Zegary Światowe
    clocks = f"BER: {now_berlin.strftime('%H:%M')}\nLON: {datetime.now(TZ_LONDON).strftime('%H:%M')}\nNYC: {datetime.now(TZ_NY).strftime('%H:%M')}"
    c2.metric("ZEGARY RYNKOWE", now_berlin.strftime('%H:%M:%S'), f"{liq_status}")
    
    # 3. Płynność (Wizualizacja)
    c3.metric("PŁYNNOŚĆ SESJI", f"{liq_val}%", "ACTIVE")
    
    # 4. Zmienność Live (Range świecy)
    rng = (live_candle['High'] - live_candle['Low']) * 10000
    c4.metric("ZMIENNOŚĆ (1M)", f"{rng:.1f} pips", "MOMENTUM")

    # Pasek Płynności HTML
    st.markdown(f"<div style='width:{liq_val}%; height:4px; background-color:#00bcd4; border-radius:2px; margin-bottom:20px;'></div>", unsafe_allow_html=True)
else:
    st.error("Błąd połączenia z rynkiem Live.")

# --- SEKCJ 2: GŁÓWNY INTERFEJS (ZAKŁADKI) ---
tab_live, tab_lab = st.tabs(["⚡ WIZUALIZACJA STRUKTURY (LIVE)", "🧪 LABORATORIUM HISTORYCZNE (CSV)"])

# === ZAKŁADKA 1: LIVE STRUCTURE (Dla Day Tradera) ===
with tab_live:
    # Pobieramy dane intraday do budowy krzywej (ostatni miesiąc H1)
    df_live_analysis = get_analysis_data(ticker, period="1mo", interval="1h")
    
    if df_live_analysis is not None:
        vwap, std, last_p, z = calculate_probability_well(df_live_analysis)
        
        c_viz, c_info = st.columns([3, 1])
        
        with c_viz:
            # KRZYWA GAUSSA (Probability Well)
            x_axis = np.linspace(vwap - 4*std, vwap + 4*std, 500)
            y_axis = norm.pdf(x_axis, vwap, std)
            
            fig = go.Figure()
            # Tło (Rozkład)
            fig.add_trace(go.Scatter(x=x_axis, y=y_axis, fill='tozeroy', mode='lines', line=dict(color='#00bcd4', width=2), fillcolor='rgba(0, 188, 212, 0.1)', name='Płynność'))
            
            # Kursor Ceny
            cursor_col = "#ff3333" if abs(z)>2 else "#00ff00"
            fig.add_vline(x=last_p, line_width=4, line_color=cursor_col)
            
            # Linie VWAP i Odchyleń
            fig.add_vline(x=vwap, line_dash="dash", line_color="white", annotation_text="VWAP")
            fig.add_vline(x=vwap+2*std, line_color="red", line_width=1, annotation_text="+2σ (Sell)")
            fig.add_vline(x=vwap-2*std, line_color="green", line_width=1, annotation_text="-2σ (Buy)")
            
            fig.update_layout(
                template='plotly_dark', height=450, 
                title="MAPA PRAWDOPODOBIEŃSTWA (LIVE)",
                xaxis_title="Cena", yaxis_visible=False,
                margin=dict(l=0,r=0,t=40,b=0), paper_bgcolor='#0b0b0b'
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # OPIS KONCEPCYJNY
            st.markdown("""
            <div class='concept-box'>
                <div class='concept-title'>💡 KONCEPCJA: STUDNIA PRAWDOPODOBIEŃSTWA</div>
                Ten wykres zastępuje tradycyjne świece. Pokazuje <b>rozkład statystyczny</b> cen z ostatniego okresu.
                Szczyt krzywej to <b>VWAP (Fair Value)</b> – tam rynek czuje się najlepiej.
                Twój cel: Szukać okazji, gdy pionowy kursor (aktualna cena) znajduje się w <b>czerwonych strefach (>2σ)</b>.
                To oznacza, że cena jest statystycznie "naciągnięta" i istnieje wysokie prawdopodobieństwo powrotu do środka (Mean Reversion).
            </div>
            """, unsafe_allow_html=True)
            
        with c_info:
            st.markdown("### SYGNAŁY")
            st.metric("ODCHYLENIE Z-SCORE", f"{z:.2f}σ", "EXTREME" if abs(z)>2 else "NORMAL", delta_color="inverse")
            st.metric("DYSTANS DO VWAP", f"{abs(last_p - vwap)*10000:.0f} pips", "POTENCJAŁ")
            
            st.info("Jeśli Z-Score > 2.0 -> Statystycznie DROGO (Szukaj Shorta).")
            st.info("Jeśli Z-Score < -2.0 -> Statystycznie TANIO (Szukaj Longa).")

# === ZAKŁADKA 2: LABORATORIUM (Dla Quanta) ===
with tab_lab:
    if uploaded_file is not None:
        df_hist = load_uploaded_csv(uploaded_file)
        
        if df_hist is not None:
            # Analiza Reżimów
            df_regime = analyze_regimes_ml(df_hist)
            last_reg = df_regime.iloc[-1]
            
            col_l1, col_l2 = st.columns([2, 1])
            
            with col_l1:
                st.subheader("KLASTERYZACJA REŻIMÓW RYNKU (AI)")
                # Wykres Reżimów
                fig_clust = px.scatter(
                    df_regime, x='Log_Ret', y='Range', color='Regime',
                    color_continuous_scale=['#4caf50', '#ffeb3b', '#f44336'],
                    labels={'Log_Ret': 'Zwrot (Trend)', 'Range': 'Zmienność (Ryzyko)'},
                    title="Mapa Stanów Rynku (Każda kropka to sesja)"
                )
                fig_clust.update_layout(template='plotly_dark', height=400, paper_bgcolor='#0b0b0b')
                st.plotly_chart(fig_clust, use_container_width=True)
                
            with col_l2:
                st.subheader("STATYSTYKA PLIKU")
                st.write(f"**Liczba Sesji:** {len(df_hist)}")
                st.write(f"**Data Od:** {df_hist.index.min().date()}")
                st.write(f"**Data Do:** {df_hist.index.max().date()}")
                
                regime_counts = df_regime['Regime'].value_counts(normalize=True)
                st.write("**Rozkład Reżimów:**")
                st.write(f"🟢 Spokojny: {regime_counts.get(0,0)*100:.1f}%")
                st.write(f"🟡 Zmienny: {regime_counts.get(1,0)*100:.1f}%")
                st.write(f"🔴 Kryzysowy: {regime_counts.get(2,0)*100:.1f}%")

            # OPIS KONCEPCYJNY
            st.markdown("""
            <div class='concept-box'>
                <div class='concept-title'>💡 KONCEPCJA: CLUSTERING ZMIENNOŚCI (GMM)</div>
                Analiza danych historycznych używa uczenia maszynowego (Gaussian Mixture Model), aby podzielić historię na "stany".
                Zamiast patrzeć na wykres, patrzysz na strukturę rynku.
                <b>Zielone punkty</b> to sesje bezpieczne (trendowe). <b>Czerwone punkty</b> to sesje paniczne (wysokie ryzyko).
                Wiedząc, w jakim klastrze był rynek ostatnio, możesz przewidzieć, jak zachowa się jutro.
            </div>
            """, unsafe_allow_html=True)
            
        else:
            st.error("Błąd formatu pliku CSV. Upewnij się, że to format Myfxbook/MT4.")
    else:
        st.warning("⚠️ Ta sekcja wymaga danych historycznych.")
        st.markdown("W panelu bocznym (po lewej) znajdziesz przycisk **'Wgraj plik (CSV)'**. Użyj go, aby załadować swoje dane z Myfxbook i odblokować głęboką analizę.")
        
        # Placeholder demo
        st.caption("Przykładowy widok po wgraniu danych:")
        st.progress(0)
