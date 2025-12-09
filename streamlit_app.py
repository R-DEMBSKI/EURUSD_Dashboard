import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from scipy.stats import norm

# --- 1. KONFIGURACJA UI ---
st.set_page_config(layout="wide", page_title="EURUSD Quant Lab AI", page_icon="🦅")

st.markdown("""
<style>
    /* Globalny styl terminala */
    .stApp { background-color: #050505; color: #c9d1d9; }
    .block-container { padding-top: 1rem; max-width: 98%; }
    
    /* Karty danych */
    div[data-testid="stMetric"] {
        background-color: #0d1117;
        border: 1px solid #30363d;
        padding: 15px;
        border-radius: 6px;
    }
    
    /* Nagłówki */
    h1, h2, h3 { color: #58a6ff !important; font-family: 'Roboto Mono', monospace; }
    
    /* Tabele */
    .dataframe { font-size: 0.8rem !important; font-family: 'Roboto Mono', monospace; }
    
    /* Ukrycie standardowych elementów */
    header, footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# --- 2. SILNIK DANYCH ---
@st.cache_data(ttl=600)
def get_data():
    # Pobieramy dane dzienne (dla makro) i godzinowe (dla sezonowości)
    tickers = "EURUSD=X DX-Y.NYB ^TNX"
    data = yf.download(tickers, period="2y", interval="1d", group_by='ticker', progress=False)
    
    # Dane godzinowe (ostatnie 60 dni - limit yfinance)
    data_h = yf.download("EURUSD=X", period="60d", interval="1h", progress=False)
    
    # Przetwarzanie D1
    df = pd.DataFrame()
    df['EURUSD'] = data['EURUSD=X']['Close']
    df['DXY'] = data['DX-Y.NYB']['Close']
    df['US10Y'] = data['^TNX']['Close']
    df['Returns'] = df['EURUSD'].pct_change()
    df['Vol'] = df['Returns'].rolling(20).std()
    df = df.fillna(method='ffill').dropna()
    
    return df, data_h

# --- 3. ALGORYTMY QUANTOWE ---

def calculate_hurst(series):
    """Oblicza Wykładnik Hursta (Fraktalność).
    H < 0.5 = Mean Reverting (Powrót do średniej - Konsolidacja)
    H ~ 0.5 = Random Walk (Szum)
    H > 0.5 = Trending (Trend)
    """
    lags = range(2, 100)
    tau = [np.sqrt(np.std(np.subtract(series[lag:], series[:-lag]))) for lag in lags]
    poly = np.polyfit(np.log(lags), np.log(tau), 1)
    return poly[0] * 2.0

def detect_market_regime(df):
    """Używa K-Means Clustering do wykrycia reżimu rynkowego (np. Wysoka Zmienność Spadkowa)"""
    X = df[['Returns', 'Vol']].copy()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Dzielimy rynek na 4 klastry (Reżimy)
    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    df['Regime'] = kmeans.fit_predict(X_scaled)
    
    # Analiza ostatniego stanu
    current_regime = df['Regime'].iloc[-1]
    
    # Opisujemy klastry statystycznie, żeby nadać im nazwy
    cluster_stats = df.groupby('Regime')[['Returns', 'Vol']].mean()
    
    # Prosta logika nazewnictwa na podstawie średnich zwrotów i zmienności klastra
    regime_name = "Nieznany"
    r_ret = cluster_stats.loc[current_regime, 'Returns']
    r_vol = cluster_stats.loc[current_regime, 'Vol']
    
    if r_vol < df['Vol'].mean():
        if r_ret > 0: regime_name = "🟢 Spokojny Wzrost"
        else: regime_name = "🔴 Spokojny Spadek / Konsolidacja"
    else:
        if r_ret > 0: regime_name = "🚀 Dynamiczny Wzrost (Breakout)"
        else: regime_name = "🩸 Dynamiczny Spadek (Krach)"
        
    return regime_name, df['Regime']

def calculate_fair_value(df):
    X = df[['DXY', 'US10Y']].tail(120)
    y = df['EURUSD'].tail(120)
    model = LinearRegression()
    model.fit(X, y)
    fair_val = model.predict(df[['DXY', 'US10Y']].iloc[-1].values.reshape(1, -1))[0]
    r2 = model.score(X, y)
    return fair_val, r2

# --- 4. DASHBOARD GŁÓWNY ---

st.title("🦅 EUR/USD Algorithmic Command Center")
st.caption("Advanced Mathematical Modeling & Machine Learning")

try:
    df, df_h = get_data()
    current_price = df['EURUSD'].iloc[-1]
    
    # --- A. PANEL AI & FRAKTALI (NOWOŚĆ) ---
    st.subheader("🧠 Sztuczna Inteligencja i Fraktale")
    
    # 1. Obliczenia
    hurst = calculate_hurst(df['EURUSD'].values)
    regime_name, regimes = detect_market_regime(df)
    
    # 2. Wyświetlanie KPI
    c1, c2, c3, c4 = st.columns(4)
    
    c1.metric("Cena EUR/USD", f"{current_price:.5f}")
    
    # Logika koloru Hursta
    h_delta = "Neutralny"
    if hurst < 0.45: h_delta = "Konsolidacja (Mean Reversion)"
    elif hurst > 0.55: h_delta = "Silny Trend"
    
    c2.metric("Wykładnik Hursta (H)", f"{hurst:.2f}", h_delta, delta_color="off")
    
    c3.metric("Wykryty Reżim Rynku (AI)", regime_name)
    
    fair_val, r2 = calculate_fair_value(df)
    mispricing = current_price - fair_val
    c4.metric("Odchylenie od Fair Value", f"{mispricing:.4f}", f"Model R²: {r2:.2f}")

    # --- B. FAIR VALUE & REGRESJA ---
    st.markdown("---")
    col_main, col_side = st.columns([3, 1])
    
    with col_main:
        st.subheader("📉 Model Wyceny (Linear Regression vs Market)")
        
        # Wykres Ceny vs Fair Value
        fig_fv = go.Figure()
        
        # Generujemy historyczne fair value do wykresu
        X_hist = df[['DXY', 'US10Y']]
        model_hist = LinearRegression().fit(X_hist, df['EURUSD'])
        df['FairValue_Hist'] = model_hist.predict(X_hist)
        
        fig_fv.add_trace(go.Scatter(x=df.index, y=df['EURUSD'], name="Cena Rynkowa", line=dict(color='#238636', width=2)))
        fig_fv.add_trace(go.Scatter(x=df.index, y=df['FairValue_Hist'], name="Fair Value (Model)", line=dict(color='#da3633', dash='dot')))
        
        # Wypełnienie między liniami
        fig_fv.add_trace(go.Scatter(
            x=df.index, y=df['EURUSD'],
            fill=None, mode='lines', line_color='rgba(0,0,0,0)', showlegend=False
        ))
        fig_fv.add_trace(go.Scatter(
            x=df.index, y=df['FairValue_Hist'],
            fill='tonexty', mode='lines', line_color='rgba(0,0,0,0)', 
            fillcolor='rgba(255, 255, 255, 0.05)', name="Mispricing Zone"
        ))

        fig_fv.update_layout(template="plotly_dark", height=450, title="Czy rynek kłamie? (Cena vs Fundamenty)", xaxis_title="")
        st.plotly_chart(fig_fv, use_container_width=True)

    with col_side:
        st.subheader("📊 Statystyka Gaussa")
        # Z-Score
        mu = df['Returns'].mean()
        sigma = df['Returns'].std()
        last_ret = df['Returns'].iloc[-1]
        z_score = (last_ret - mu) / sigma
        
        fig_gauss = go.Figure()
        x_axis = np.linspace(mu - 3*sigma, mu + 3*sigma, 100)
        y_axis = norm.pdf(x_axis, mu, sigma)
        
        fig_gauss.add_trace(go.Scatter(x=x_axis, y=y_axis, mode='lines', fill='tozeroy', line=dict(color='#58a6ff'), name='Rozkład'))
        fig_gauss.add_vline(x=last_ret, line_color="yellow", line_width=3, annotation_text="DZIŚ")
        
        fig_gauss.update_layout(template="plotly_dark", height=450, showlegend=False, title=f"Z-Score: {z_score:.2f}")
        st.plotly_chart(fig_gauss, use_container_width=True)

    # --- C. HEATMAPY SEZONOWOŚCI (ALGORYTMIA CZASOWA) ---
    st.markdown("---")
    st.subheader("📅 Algorytmiczna Sezonowość (Gdzie jest przewaga?)")
    
    c_heat1, c_heat2 = st.columns(2)
    
    with c_heat1:
        # Przygotowanie danych godzinowych
        df_h['Hour'] = df_h.index.hour
        df_h['DayOfWeek'] = df_h.index.day_name()
        df_h['Return'] = df_h['Close'].pct_change() * 10000 # Pips
        
        # Pivot Table: Dzień Tygodnia vs Godzina
        # Kolejność dni
        days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']
        heatmap_data = df_h.groupby(['DayOfWeek', 'Hour'])['Return'].mean().unstack()
        heatmap_data = heatmap_data.reindex(days_order)
        
        fig_heat = px.imshow(
            heatmap_data,
            labels=dict(x="Godzina (UTC)", y="Dzień", color="Śr. Zmiana (Pips)"),
            x=heatmap_data.columns,
            y=heatmap_data.index,
            color_continuous_scale="RdBu",
            origin='upper',
            title="Intraday Edge: Średnia zmiana ceny (Pips)"
        )
        fig_heat.update_layout(template="plotly_dark", height=400)
        st.plotly_chart(fig_heat, use_container_width=True)
        
    with c_heat2:
        st.info("""
        **Jak czytać Heatmapę?**
        Algorytm analizuje ostatnie 60 dni handlu godzinowego.
        * **Niebieskie Pola:** Godziny statystycznie wzrostowe.
        * **Czerwone Pola:** Godziny statystycznie spadkowe.
        
        Wykorzystaj to do timingu wejścia. Jeśli Twój sygnał LONG pokrywa się z niebieskim polem -> Prawdopodobieństwo rośnie.
        """)
        
        st.write(f"**Wnioski z analizy Hurst'a ({hurst:.2f}):**")
        if hurst < 0.45:
            st.warning("⚠️ RYNEK W KONSOLIDACJI. Unikaj strategii wybiciowych (Breakout). Stosuj strategie Mean Reversion (kupuj nisko, sprzedawaj wysoko w kanale).")
        elif hurst > 0.55:
            st.success("✅ RYNEK W TRENDZIE. Stosuj strategie podążania za trendem (Trend Following). Kupuj dołki w trendzie wzrostowym.")
        else:
            st.info("⚖️ RYNEK LOSOWY (Random Walk). Brak wyraźnej przewagi statystycznej. Obniż ryzyko.")

except Exception as e:
    st.error(f"Inicjalizacja systemu AI... (Błąd: {e})")
    st.write("System pobiera duże ilości danych do obliczeń ML. Proszę odświeżyć stronę za chwilę.")
