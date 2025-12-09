import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from sklearn.mixture import GaussianMixture
from scipy.stats import norm

# --- 1. KONFIGURACJA UI (FIXED PADDING) ---
st.set_page_config(layout="wide", page_title="QUANT LAB EXPERT", page_icon="🧪", initial_sidebar_state="expanded")

st.markdown("""
<style>
    /* GLOBALNY STYL - DEEP BLACK */
    .stApp { background-color: #050505; color: #e0e0e0; font-family: 'Roboto Mono', monospace; }
    
    /* NAPRAWA PRZESUNIĘCIA (PADDING) */
    .block-container { 
        padding-top: 3rem !important; /* To naprawia uciętą górę */
        padding-bottom: 2rem;
        max-width: 100%; 
    }
    
    /* UKRYCIE ELEMENTÓW SYSTEMOWYCH */
    header {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* KONTEINERY */
    .quant-card { 
        background-color: #0f0f0f; 
        border: 1px solid #333; 
        padding: 15px; 
        border-radius: 2px; 
        margin-bottom: 15px;
    }
    .header-text { 
        color: #00e5ff; 
        font-weight: bold; 
        font-size: 0.85rem; 
        text-transform: uppercase; 
        letter-spacing: 1.5px; 
        margin-bottom: 15px; 
        border-bottom: 1px solid #333; 
        padding-bottom: 5px; 
    }
    
    /* KPI METRICS */
    div[data-testid="stMetric"] { 
        background-color: #111; 
        border: 1px solid #333; 
        padding: 10px; 
        border-radius: 0px;
    }
    div[data-testid="stMetricLabel"] { font-size: 0.7rem !important; color: #888; text-transform: uppercase; }
    div[data-testid="stMetricValue"] { font-size: 1.3rem !important; color: #fff; font-weight: 700; font-family: 'Consolas', monospace; }
    
    /* ZAKŁADKI */
    .stTabs [data-baseweb="tab-list"] { gap: 2px; background-color: #050505; }
    .stTabs [data-baseweb="tab"] {
        height: 35px; background-color: #1a1a1a; color: #aaa; border: 1px solid #333; border-radius: 0px; font-size: 0.8rem;
    }
    .stTabs [aria-selected="true"] { background-color: #00e5ff; color: #000; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

# --- 2. QUANT ENGINE (CORE) ---

def load_myfxbook_data(uploaded_file):
    try:
        df = pd.read_csv(uploaded_file, skiprows=1) # Myfxbook ma nagłówek w 2 linii
        df.columns = [c.strip() for c in df.columns]
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values('Date').set_index('Date')
        required = ['Open', 'High', 'Low', 'Close']
        if not all(c in df.columns for c in required): return None
        return df[required]
    except: return None

def generate_mock_data():
    dates = pd.date_range(end=pd.Timestamp.now(), periods=1000, freq='B') # Business days
    np.random.seed(42)
    returns = np.random.normal(0.0001, 0.006, 1000)
    # Dodanie sezonowości i szoków
    returns[::5] += 0.002 # Wtorki rosną (mock)
    returns[::20] *= 3 # Szok co miesiąc
    price = 1.1000 * np.exp(np.cumsum(returns))
    df = pd.DataFrame({'Close': price}, index=dates)
    df['Open'] = df['Close'] * (1 + np.random.normal(0, 0.001, 1000))
    df['High'] = df[['Open', 'Close']].max(axis=1) * 1.002
    df['Low'] = df[['Open', 'Close']].min(axis=1) * 0.998
    return df

def analyze_regimes(df):
    data = df.copy()
    data['Log_Ret'] = np.log(data['Close'] / data['Close'].shift(1))
    data['Range'] = (data['High'] - data['Low']) / data['Close']
    data = data.dropna()
    
    # GMM Clustering
    X = data[['Log_Ret', 'Range']].values
    gmm = GaussianMixture(n_components=3, covariance_type='full', random_state=42)
    gmm.fit(X)
    data['Regime'] = gmm.predict(X)
    
    # Sortowanie reżimów po zmienności (0=Low, 2=High)
    vol_means = data.groupby('Regime')['Range'].mean().sort_values()
    mapping = {old: new for new, old in enumerate(vol_means.index)}
    data['Regime'] = data['Regime'].map(mapping)
    
    return data

def calculate_volatility_cone(df):
    """Oblicza historyczne percentyle zmienności dla różnych okien."""
    windows = [5, 10, 20, 40, 60, 90]
    log_ret = np.log(df['Close'] / df['Close'].shift(1))
    
    cone_data = []
    current_vols = []
    
    for w in windows:
        # Rolling annualized vol
        roll_vol = log_ret.rolling(w).std() * np.sqrt(252)
        
        # Percentyle z całej historii
        min_v = roll_vol.min()
        q25 = roll_vol.quantile(0.25)
        median = roll_vol.median()
        q75 = roll_vol.quantile(0.75)
        max_v = roll_vol.max()
        
        # Aktualna zmienność dla tego okna
        curr_v = roll_vol.iloc[-1]
        
        cone_data.append({'Window': w, 'Min': min_v, 'Q25': q25, 'Median': median, 'Q75': q75, 'Max': max_v})
        current_vols.append({'Window': w, 'Current': curr_v})
        
    return pd.DataFrame(cone_data), pd.DataFrame(current_vols)

# --- 3. DASHBOARD LOGIC ---

with st.sidebar:
    st.markdown("### 📥 DATA INPUT")
    uploaded_file = st.file_uploader("Wgraj CSV (Myfxbook)", type=['csv'])
    if uploaded_file:
        df_raw = load_myfxbook_data(uploaded_file)
        source = "USER FILE"
    else:
        df_raw = generate_mock_data()
        source = "MOCK DATA"

if df_raw is not None:
    df = analyze_regimes(df_raw)
    
    # --- HEADER KPI ---
    last_row = df.iloc[-1]
    
    # VaR (Value at Risk) - 95% Confidence
    var_95 = np.percentile(df['Log_Ret'], 5) * 100 # dzienne ryzyko w %
    
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("ANALYZED BARS", f"{len(df)}", source)
    c2.metric("CURRENT REGIME", f"TYPE {int(last_row['Regime'])}", 
              "HIGH VOL" if last_row['Regime']==2 else "TREND" if last_row['Regime']==0 else "NOISE", 
              delta_color="off")
    c3.metric("DAILY VaR (95%)", f"{var_95:.2f}%", "DOWNSIDE RISK", delta_color="inverse")
    c4.metric("LAST RETURN", f"{last_row['Log_Ret']*100:.2f}%", f"Range: {last_row['Range']*10000:.0f} pips")
    
    st.markdown("---")

    # --- TABS FOR DEEP DIVE ---
    tab_regime, tab_season, tab_vol = st.tabs(["🧬 MARKET GENETICS (REGIMES)", "📅 SEASONALITY LAB", "⚠️ VOLATILITY CONE"])

    # === TAB 1: REGIMES (GMM) ===
    with tab_regime:
        c_viz, c_trans = st.columns([2, 1])
        with c_viz:
            st.markdown(f"<div class='header-text'>AI CLUSTERING: RETURNS vs RANGE</div>", unsafe_allow_html=True)
            fig_clust = px.scatter(
                df, x='Log_Ret', y='Range', color='Regime',
                color_continuous_scale=['#00ff00', '#ffff00', '#ff0000'],
                opacity=0.7, title=""
            )
            # Zaznacz ostatni punkt
            fig_clust.add_trace(go.Scatter(
                x=[last_row['Log_Ret']], y=[last_row['Range']],
                mode='markers', marker=dict(color='white', size=15, line=dict(color='black', width=2)),
                name='CURRENT'
            ))
            fig_clust.update_layout(template='plotly_dark', height=450, margin=dict(l=0,r=0,t=0,b=0), paper_bgcolor='#0f0f0f')
            st.plotly_chart(fig_clust, use_container_width=True)
            
        with c_trans:
            st.markdown(f"<div class='header-text'>TRANSITION PROBABILITIES</div>", unsafe_allow_html=True)
            df['Next_Regime'] = df['Regime'].shift(-1)
            trans_matrix = pd.crosstab(df['Regime'], df['Next_Regime'], normalize='index')
            
            fig_trans = go.Figure(data=go.Heatmap(
                z=trans_matrix.values, x=[0,1,2], y=[0,1,2],
                colorscale='Viridis', text=np.round(trans_matrix.values*100, 0), texttemplate="%{text}%"
            ))
            fig_trans.update_layout(height=450, margin=dict(l=0,r=0,t=0,b=0), title="Next Day Prob", paper_bgcolor='#0f0f0f')
            st.plotly_chart(fig_trans, use_container_width=True)

    # === TAB 2: SEASONALITY (NEW) ===
    with tab_season:
        st.markdown(f"<div class='header-text'>STATYSTYKA DNI TYGODNIA (EDGE FINDER)</div>", unsafe_allow_html=True)
        
        df['DayName'] = df.index.day_name()
        days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']
        
        # Box Plot dla zwrotów
        fig_box = px.box(df, x='DayName', y='Log_Ret', color='DayName', 
                         category_orders={'DayName': days_order}, points=False)
        fig_box.add_hline(y=0, line_color="white", line_dash="dash")
        fig_box.update_layout(template='plotly_dark', height=500, margin=dict(t=30), paper_bgcolor='#0f0f0f')
        st.plotly_chart(fig_box, use_container_width=True)
        
        # Tabela statystyk
        stats = df.groupby('DayName')['Log_Ret'].agg(['mean', 'std', 'count', lambda x: (x>0).mean()])
        stats.columns = ['Avg Return', 'Volatility', 'Count', 'Win Rate']
        stats['Win Rate'] = stats['Win Rate'].apply(lambda x: f"{x*100:.1f}%")
        stats = stats.reindex(days_order)
        st.dataframe(stats.style.background_gradient(subset=['Avg Return'], cmap='RdYlGn'), use_container_width=True)

    # === TAB 3: VOLATILITY CONE (NEW) ===
    with tab_vol:
        st.markdown(f"<div class='header-text'>STOŻEK ZMIENNOŚCI (HISTORICAL CONE)</div>", unsafe_allow_html=True)
        st.caption("Porównanie obecnej zmienności do historycznych ekstremów. Jeśli biała linia jest nisko = Rynek jest skompresowany (Squeeze).")
        
        cone_df, curr_vol_df = calculate_volatility_cone(df)
        
        fig_cone = go.Figure()
        
        # Max/Min (Szare tło)
        fig_cone.add_trace(go.Scatter(x=cone_df['Window'], y=cone_df['Max'], mode='lines', line=dict(width=0), showlegend=False))
        fig_cone.add_trace(go.Scatter(x=cone_df['Window'], y=cone_df['Min'], mode='lines', fill='tonexty', fillcolor='rgba(255,255,255,0.1)', line=dict(width=0), name='Min-Max Range'))
        
        # Quartiles (Ciemniejsze tło)
        fig_cone.add_trace(go.Scatter(x=cone_df['Window'], y=cone_df['Q75'], mode='lines', line=dict(width=0), showlegend=False))
        fig_cone.add_trace(go.Scatter(x=cone_df['Window'], y=cone_df['Q25'], mode='lines', fill='tonexty', fillcolor='rgba(0,188,212,0.2)', line=dict(width=0), name='Interquartile (Normal)'))
        
        # Mediana
        fig_cone.add_trace(go.Scatter(x=cone_df['Window'], y=cone_df['Median'], mode='lines', line=dict(color='cyan', dash='dash'), name='Median Vol'))
        
        # Current Volatility (To jest najważniejsze)
        fig_cone.add_trace(go.Scatter(x=curr_vol_df['Window'], y=curr_vol_df['Current'], mode='lines+markers', line=dict(color='white', width=3), name='CURRENT STATE'))
        
        fig_cone.update_layout(template='plotly_dark', height=500, xaxis_title="Okres (Dni)", yaxis_title="Zmienność Roczna", paper_bgcolor='#0f0f0f')
        st.plotly_chart(fig_cone, use_container_width=True)

else:
    st.info("Wgraj plik CSV aby rozpocząć analizę.")
