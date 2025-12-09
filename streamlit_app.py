import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from sklearn.linear_model import LinearRegression
from scipy.stats import norm

# --- 1. KONFIGURACJA UI ---
st.set_page_config(layout="wide", page_title="EURUSD Quant Lab", page_icon="🧪")

st.markdown("""
<style>
    .stApp { background-color: #050505; color: #e0e0e0; }
    .block-container { padding-top: 1rem; }
    
    /* Stylizacja metryk */
    div[data-testid="metric-container"] {
        background-color: #111;
        border: 1px solid #333;
        padding: 10px;
        border-radius: 5px;
    }
    
    /* Ukrycie elementów Streamlit */
    header, footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# --- 2. SILNIK POBIERANIA DANYCH ---
@st.cache_data(ttl=600) # Cache 10 min
def get_quant_data():
    # Pobieramy EURUSD, DXY i US 10Y Yield
    tickers = "EURUSD=X DX-Y.NYB ^TNX"
    data = yf.download(tickers, period="2y", interval="1d", group_by='ticker', progress=False)
    
    # Przetwarzanie MultiIndex
    df = pd.DataFrame()
    df['EURUSD'] = data['EURUSD=X']['Close']
    df['DXY'] = data['DX-Y.NYB']['Close']
    df['US10Y'] = data['^TNX']['Close']
    
    # Forward Fill dla braków w danych (święta itp.)
    df = df.fillna(method='ffill').dropna()
    return df

# --- 3. ALGORYTMY MATEMATYCZNE ---

def calculate_monte_carlo(prices, days_ahead=5, simulations=1000):
    """Generuje symulacje przyszłych cen oparte na zmienności historycznej (Geometric Brownian Motion)"""
    last_price = prices.iloc[-1]
    returns = prices.pct_change().dropna()
    daily_vol = returns.std()
    
    simulation_df = pd.DataFrame()
    
    for i in range(simulations):
        # Generowanie losowych szoków cenowych
        count = 0
        price_list = [last_price]
        price = last_price
        
        for d in range(days_ahead):
            # Wzór Blacka-Scholesa na ruch Browna
            shock = np.random.normal(0, daily_vol)
            price = price * (1 + shock)
            price_list.append(price)
            
        simulation_df[i] = price_list
        
    return simulation_df

def calculate_fair_value(df):
    """Regresja liniowa: Przewiduje cenę EURUSD na podstawie DXY i US10Y"""
    # X = Zmienne niezależne (DXY, Yields), y = Zmienna zależna (EURUSD)
    X = df[['DXY', 'US10Y']].tail(100) # Trenujemy na ostatnich 100 dniach
    y = df['EURUSD'].tail(100)
    
    model = LinearRegression()
    model.fit(X, y)
    
    # Obliczamy "Fair Value" dla aktualnych danych
    current_x = df[['DXY', 'US10Y']].iloc[-1].values.reshape(1, -1)
    fair_value = model.predict(current_x)[0]
    r_squared = model.score(X, y) # Jak dobrze model pasuje (0-1)
    
    return fair_value, r_squared

def calculate_volatility_cone(prices, window_sizes=[5, 20, 50, 100]):
    """Oblicza historyczną zmienność dla różnych horyzontów czasowych"""
    log_returns = np.log(prices / prices.shift(1))
    vol_cone = {}
    
    for window in window_sizes:
        # Zmienność roczna dla danego okna
        realized_vol = log_returns.rolling(window=window).std() * np.sqrt(252)
        # Bierzemy percentyle (Min, Max, Median)
        vol_cone[window] = {
            'min': realized_vol.min(),
            'max': realized_vol.max(),
            'median': realized_vol.median(),
            'current': realized_vol.iloc[-1]
        }
    return vol_cone

# --- 4. GŁÓWNY INTERFEJS ---

st.title("🧪 EUR/USD Quantitative Lab")
st.caption("Algorithmic Analysis & Statistical Modeling")

try:
    df = get_quant_data()
    current_price = df['EURUSD'].iloc[-1]
    
    # --- SEKCJA 1: FAIR VALUE MODEL (REGRESJA) ---
    fair_val, model_confidence = calculate_fair_value(df)
    deviation = current_price - fair_val
    
    # Koloryzacja odchylenia
    dev_color = "red" if deviation > 0 else "green" 
    # Jeśli cena > fair value (red) -> potencjalny short. Jeśli cena < fair value (green) -> potencjalny long.

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Aktualna Cena", f"{current_price:.4f}")
    col2.metric("Fair Value (Model)", f"{fair_val:.4f}", help="Cena wynikająca z regresji względem DXY i US10Y")
    col3.metric("Odchylenie (Mispricing)", f"{deviation:.4f}", delta_color="inverse")
    col4.metric("Siła Modelu (R²)", f"{model_confidence:.2f}", help="Powyżej 0.80 oznacza bardzo silną zależność fundamentalną")

    st.markdown("---")

    # --- SEKCJA 2: MONTE CARLO & PRAWDOPODOBIEŃSTWO ---
    st.subheader("🎲 Monte Carlo Simulation (Next 5 Days)")
    
    col_mc_chart, col_mc_stats = st.columns([3, 1])
    
    with col_mc_chart:
        # Generujemy symulację
        sim_data = calculate_monte_carlo(df['EURUSD'], days_ahead=5, simulations=500)
        
        fig_mc = go.Figure()
        # Rysujemy 500 linii (zmniejszona przezroczystość)
        for col in sim_data.columns[:100]: # Pokaż tylko 100 linii dla wydajności, statystyka liczona z całości
            fig_mc.add_trace(go.Scatter(y=sim_data[col], mode='lines', line=dict(color='#2A2A2A', width=1), showlegend=False, hoverinfo='skip'))
            
        # Dodajemy średnią ścieżkę
        mean_path = sim_data.mean(axis=1)
        fig_mc.add_trace(go.Scatter(y=mean_path, mode='lines', name="Mean Path", line=dict(color='#00CC96', width=3)))
        
        # Start
        fig_mc.add_hline(y=current_price, line_dash="dash", line_color="white", annotation_text="Start")

        fig_mc.update_layout(
            template="plotly_dark", 
            title="Projekcja ścieżek cenowych (Random Walk)",
            height=400,
            xaxis_title="Dni do przodu",
            yaxis_title="Cena Symulowana"
        )
        st.plotly_chart(fig_mc, use_container_width=True)
        
    with col_mc_stats:
        # Statystyka z symulacji
        final_prices = sim_data.iloc[-1]
        p95 = np.percentile(final_prices, 95)
        p50 = np.percentile(final_prices, 50)
        p05 = np.percentile(final_prices, 5)
        
        st.markdown("##### 🔮 Probability Cone")
        st.write(f"**Górny pułap (95% szans):**")
        st.code(f"{p95:.5f}")
        st.write(f"**Środek (Oczekiwana):**")
        st.code(f"{p50:.5f}")
        st.write(f"**Dolny pułap (5% szans):**")
        st.code(f"{p05:.5f}")
        
        prob_up = (final_prices > current_price).mean() * 100
        st.metric("Szansa na wzrost", f"{prob_up:.1f}%")

    st.markdown("---")
    
    # --- SEKCJA 3: DISTRIBUTION ANALYTICS (DZWON GAUSSA) ---
    col_dist, col_vol = st.columns(2)
    
    with col_dist:
        st.subheader("📊 Rozkład Zwrotów (Gaussian)")
        # Obliczamy dzienne zmiany procentowe
        returns = df['EURUSD'].pct_change().dropna() * 100
        curr_return = returns.iloc[-1]
        
        # Fitujemy rozkład normalny
        mu, std = norm.fit(returns)
        
        # Histogram
        fig_dist = px.histogram(returns, nbins=100, title="Czy dzisiejszy ruch to anomalia?", opacity=0.6)
        
        # Dodajemy dzisiejszy ruch
        fig_dist.add_vline(x=curr_return, line_color="yellow", line_width=3, annotation_text="DZIŚ")
        
        # Linie Sigmy (Odchylenia standardowe)
        fig_dist.add_vline(x=std, line_dash="dot", line_color="red", annotation_text="+1σ")
        fig_dist.add_vline(x=-std, line_dash="dot", line_color="red", annotation_text="-1σ")
        fig_dist.add_vline(x=2*std, line_dash="dot", line_color="red", annotation_text="+2σ")
        
        fig_dist.update_layout(template="plotly_dark", showlegend=False, xaxis_title="Zmiana %")
        st.plotly_chart(fig_dist, use_container_width=True)
        
        # Interpretacja
        z_score_now = (curr_return - mu) / std
        st.info(f"Dzisiejszy Z-Score: {z_score_now:.2f}. " + ("Ruch w normie." if abs(z_score_now) < 2 else "⚠️ STATYSTYCZNA ANOMALIA!"))

    with col_vol:
        st.subheader("⚡ Volatility Cone (Ryzyko)")
        vol_data = calculate_volatility_cone(df['EURUSD'])
        
        # Przygotowanie danych do wykresu
        windows = list(vol_data.keys())
        max_vols = [vol_data[w]['max'] for w in windows]
        min_vols = [vol_data[w]['min'] for w in windows]
        curr_vols = [vol_data[w]['current'] for w in windows]
        
        fig_vol = go.Figure()
        fig_vol.add_trace(go.Scatter(x=windows, y=max_vols, mode='lines+markers', name="Max Vol (History)", line=dict(color='red', dash='dot')))
        fig_vol.add_trace(go.Scatter(x=windows, y=min_vols, mode='lines+markers', name="Min Vol (History)", line=dict(color='green', dash='dot')))
        fig_vol.add_trace(go.Scatter(x=windows, y=curr_vols, mode='lines+markers', name="Current Vol", line=dict(color='yellow', width=3)))
        
        fig_vol.update_layout(
            template="plotly_dark", 
            title="Czy zmienność jest tania czy droga?",
            xaxis_title="Horyzont (Dni)",
            yaxis_title="Annualized Volatility"
        )
        st.plotly_chart(fig_vol, use_container_width=True)
        
        if curr_vols[0] < min_vols[0] * 1.2:
            st.success("Zmienność jest bardzo niska (Cisza przed burzą).")
        elif curr_vols[0] > max_vols[0] * 0.8:
            st.error("Ekstremalnie wysoka zmienność (Panika/Euforia).")
        else:
            st.warning("Zmienność w normie.")

except Exception as e:
    st.error(f"Błąd silnika obliczeniowego: {e}")
    st.write("Sprawdź połączenie z yfinance lub zaktualizuj biblioteki.")
