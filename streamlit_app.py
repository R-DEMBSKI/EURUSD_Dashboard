import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from sklearn.mixture import GaussianMixture
from scipy.stats import norm

# --- 1. KONFIGURACJA UI (QUANT LAB DARK) ---
st.set_page_config(layout="wide", page_title="QUANT RESEARCH LAB", page_icon="🧪", initial_sidebar_state="expanded")

st.markdown("""
<style>
    /* DEEP RESEARCH THEME */
    .stApp { background-color: #050505; color: #e0e0e0; font-family: 'Roboto Mono', monospace; }
    .block-container { padding: 1rem; max-width: 100%; }
    
    /* MODUŁY */
    .quant-card { background-color: #111; border: 1px solid #333; padding: 15px; border-radius: 4px; margin-bottom: 10px; }
    .header-text { color: #00bcd4; font-weight: bold; font-size: 0.9rem; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 10px; border-bottom: 1px solid #333; padding-bottom: 5px; }
    
    /* KPI METRICS */
    div[data-testid="stMetric"] { background-color: #0a0a0a; border: 1px solid #222; padding: 10px; }
    div[data-testid="stMetricLabel"] { font-size: 0.7rem !important; color: #888; }
    div[data-testid="stMetricValue"] { font-size: 1.4rem !important; color: #fff; }
    
    /* UPLOADER */
    .stFileUploader { font-size: 0.8rem; }
</style>
""", unsafe_allow_html=True)

# --- 2. QUANT ENGINE (DATA PROCESSING & ML) ---

def load_myfxbook_data(uploaded_file):
    """Parsuje specyficzny format Myfxbook CSV."""
    try:
        # Myfxbook ma nagłówek w 2. linii (index 1) i często trailing comma
        df = pd.read_csv(uploaded_file, skiprows=1)
        
        # Czyszczenie nazw kolumn (usuwanie spacji)
        df.columns = [c.strip() for c in df.columns]
        
        # Konwersja daty
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values('Date').set_index('Date')
        
        # Selekcja i czyszczenie
        required_cols = ['Open', 'High', 'Low', 'Close']
        if not all(c in df.columns for c in required_cols):
            st.error(f"Brakuje kolumn OHLC. Znaleziono: {df.columns.tolist()}")
            return None
            
        return df[required_cols]
    except Exception as e:
        st.error(f"Błąd parsowania pliku: {e}")
        return None

def generate_mock_data():
    """Generuje dane demo, jeśli użytkownik nie wgrał pliku."""
    dates = pd.date_range(end=pd.Timestamp.now(), periods=500, freq='D')
    np.random.seed(42)
    # Symulacja trendów i szoków
    returns = np.random.normal(0.0001, 0.005, 500)
    # Dodanie "szoków" (Cluster zmienności)
    returns[100:150] *= 3 
    returns[300:350] *= 0.5
    
    price = 1.1000 * np.exp(np.cumsum(returns))
    
    df = pd.DataFrame(index=dates)
    df['Close'] = price
    df['Open'] = price * (1 + np.random.normal(0, 0.001, 500))
    df['High'] = df[['Open', 'Close']].max(axis=1) * (1 + abs(np.random.normal(0, 0.002, 500)))
    df['Low'] = df[['Open', 'Close']].min(axis=1) * (1 - abs(np.random.normal(0, 0.002, 500)))
    return df

def analyze_regimes(df, n_components=3):
    """
    GMM (Gaussian Mixture Model) do wykrywania ukrytych stanów rynku.
    Features: Volatility (Range) & Returns.
    """
    data = df.copy()
    data['Log_Ret'] = np.log(data['Close'] / data['Close'].shift(1))
    data['Range'] = (data['High'] - data['Low']) / data['Close']
    data['Vol_5d'] = data['Log_Ret'].rolling(5).std()
    data = data.dropna()
    
    # Feature Engineering dla modelu
    # Używamy Zmienności i Zwrotów do klasteryzacji
    X = data[['Log_Ret', 'Range']].values
    
    # Trenowanie GMM
    gmm = GaussianMixture(n_components=n_components, covariance_type='full', random_state=42)
    gmm.fit(X)
    data['Regime'] = gmm.predict(X)
    
    # Sortowanie reżimów po zmienności (0 = Low Vol, 2 = High Vol)
    vol_means = data.groupby('Regime')['Range'].mean().sort_values()
    mapping = {old: new for new, old in enumerate(vol_means.index)}
    data['Regime'] = data['Regime'].map(mapping)
    
    return data, gmm

# --- 3. LAYOUT & LOGIC ---

# SIDEBAR - DATA LOADER
with st.sidebar:
    st.markdown("### 📥 DATA INGESTION")
    uploaded_file = st.file_uploader("Wgraj CSV (Myfxbook/MT4)", type=['csv'])
    
    if uploaded_file is not None:
        df_raw = load_myfxbook_data(uploaded_file)
        source_label = "USER DATA"
    else:
        df_raw = generate_mock_data()
        source_label = "MOCK DATA (DEMO)"
        st.info("👆 Wgraj plik CSV, aby zastąpić dane demo.")

# MAIN ANALYSIS
if df_raw is not None:
    # Uruchomienie Silnika Analitycznego
    df_regime, model = analyze_regimes(df_raw)
    
    last_regime = df_regime['Regime'].iloc[-1]
    regime_labels = {0: "🟢 CALM / TREND", 1: "🟡 NERVOUS / CHOPPY", 2: "🔴 CRISIS / HIGH VOL"}
    curr_label = regime_labels.get(last_regime, "UNKNOWN")
    
    # --- HEADER KPI ---
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("DATA SOURCE", source_label, f"{len(df_raw)} bars")
    c2.metric("CURRENT REGIME", f"TYPE {last_regime}", curr_label, delta_color="off")
    c3.metric("MEAN DAILY RANGE", f"{df_regime['Range'].mean()*10000:.1f} pips", "VOLATILITY BASE")
    
    # Prob kontynuacji
    next_day_prob = df_regime[df_regime['Regime'] == last_regime]['Log_Ret'].mean()
    direction = "BULLISH" if next_day_prob > 0 else "BEARISH"
    c4.metric("STATISTICAL BIAS", direction, f"Exp: {next_day_prob*100:.2f}%")
    
    st.markdown("---")
    
    # --- GŁÓWNA WIZUALIZACJA: REGIME CLUSTERS (Zamiast wykresu ceny) ---
    col_viz, col_stats = st.columns([2, 1])
    
    with col_viz:
        st.markdown(f"<div class='header-text'>MAPA REŻIMÓW RYNKOWYCH (ML CLUSTERING)</div>", unsafe_allow_html=True)
        st.caption("Każda kropka to jedna sesja historyczna. Kolory oznaczają stany wykryte przez AI.")
        
        # Scatter plot: Returns vs Volatility
        fig_clusters = px.scatter(
            df_regime, x='Log_Ret', y='Range', color='Regime',
            color_continuous_scale=['#00ff00', '#ffff00', '#ff0000'],
            hover_data=['Close'], opacity=0.8,
            labels={'Log_Ret': 'Zwrot Dzienny (Log)', 'Range': 'Zmienność (High-Low)'}
        )
        
        # Dodanie punktu "DZIŚ"
        last_day = df_regime.iloc[-1]
        fig_clusters.add_trace(go.Scatter(
            x=[last_day['Log_Ret']], y=[last_day['Range']],
            mode='markers+text', marker=dict(color='white', size=15, line=dict(color='black', width=2)),
            text=["DZIŚ"], textposition="top center", name='CURRENT'
        ))
        
        fig_clusters.update_layout(
            template='plotly_dark', height=500,
            paper_bgcolor='#111', plot_bgcolor='#111',
            coloraxis_showscale=False,
            margin=dict(l=0, r=0, t=10, b=0)
        )
        st.plotly_chart(fig_clusters, use_container_width=True)
        
    with col_stats:
        st.markdown(f"<div class='header-text'>PRAWDOPODOBIEŃSTWO PRZEJŚĆ</div>", unsafe_allow_html=True)
        st.caption("Jaka jest szansa zmiany reżimu jutro?")
        
        # Obliczanie macierzy przejść
        df_regime['Next_Regime'] = df_regime['Regime'].shift(-1)
        trans_matrix = pd.crosstab(df_regime['Regime'], df_regime['Next_Regime'], normalize='index')
        
        # Heatmapa przejść
        fig_trans = go.Figure(data=go.Heatmap(
            z=trans_matrix.values,
            x=['Type 0', 'Type 1', 'Type 2'],
            y=['Type 0', 'Type 1', 'Type 2'],
            colorscale='Viridis', text=np.round(trans_matrix.values*100, 1),
            texttemplate="%{text}%"
        ))
        fig_trans.update_layout(
            height=250, margin=dict(l=0,r=0,t=0,b=0),
            paper_bgcolor='#111', title_font_size=10
        )
        st.plotly_chart(fig_trans, use_container_width=True)
        
        # Statystyka obecnego reżimu
        st.markdown(f"<div class='header-text'>EDGE W REŻIMIE {last_regime}</div>", unsafe_allow_html=True)
        subset = df_regime[df_regime['Regime'] == last_regime]
        win_rate = (subset['Log_Ret'] > 0).mean()
        st.write(f"**Win Rate (Up Days):** {win_rate*100:.1f}%")
        st.write(f"**Avg Volatility:** {subset['Range'].mean()*10000:.0f} pips")
        st.progress(win_rate)

    # --- HISTORICAL DISTRIBUTION (FAT TAILS) ---
    st.markdown("---")
    c1, c2 = st.columns([1, 1])
    
    with c1:
        st.markdown(f"<div class='header-text'>ROZKŁAD ZWROTÓW (HISTOGRAM)</div>", unsafe_allow_html=True)
        fig_hist = px.histogram(df_regime, x='Log_Ret', color='Regime', nbins=50, barmode='overlay')
        fig_hist.add_vline(x=0, line_color="white", line_dash="dash")
        fig_hist.update_layout(template='plotly_dark', height=300, paper_bgcolor='#111', margin=dict(t=10,b=0))
        st.plotly_chart(fig_hist, use_container_width=True)
        
    with c2:
        st.markdown(f"<div class='header-text'>Krzywa Kapitału (Symulacja Buy & Hold)</div>", unsafe_allow_html=True)
        df_regime['Equity'] = (1 + df_regime['Log_Ret']).cumprod()
        fig_eq = px.line(df_regime, y='Equity')
        fig_eq.update_layout(template='plotly_dark', height=300, paper_bgcolor='#111', margin=dict(t=10,b=0))
        st.plotly_chart(fig_eq, use_container_width=True)

else:
    st.warning("Oczekiwanie na dane...")
