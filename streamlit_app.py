import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import linregress, entropy
from datetime import datetime, timedelta

# --- 1. KONFIGURACJA UI (STYL QUANT HEDGE FUND) ---
st.set_page_config(layout="wide", page_title="QUANTUM TERMINAL", page_icon="🦅", initial_sidebar_state="collapsed")

# CSS: High Density, Dark Mode, Responsywność
st.markdown("""
<style>
    /* Baza - Głęboka czerń */
    .stApp { background-color: #050505; color: #c0c0c0; font-family: 'Roboto Mono', monospace; }
    
    /* Ukrycie elementów systemowych */
    header, footer {visibility: hidden;}
    .block-container { padding-top: 0.5rem; padding-left: 1rem; padding-right: 1rem; max-width: 100%; }
    
    /* Karty KPI - Styl "Glassmorphism" */
    div[data-testid="stMetric"] {
        background-color: #111;
        border: 1px solid #333;
        padding: 10px;
        border-radius: 4px;
        transition: transform 0.2s;
    }
    div[data-testid="stMetric"]:hover {
        border-color: #007bff;
        transform: scale(1.02);
    }
    div[data-testid="stMetricLabel"] { font-size: 0.7rem !important; color: #888; text-transform: uppercase; letter-spacing: 1px; }
    div[data-testid="stMetricValue"] { font-size: 1.3rem !important; color: #fff; font-weight: 600; }
    
    /* Wykresy */
    .js-plotly-plot { border: 1px solid #222; border-radius: 4px; }
    
    /* Typografia Nagłówków */
    h1, h2, h3, h4, h5 { 
        color: #e0e0e0 !important; 
        font-family: 'Arial', sans-serif; 
        text-transform: uppercase; 
        letter-spacing: 1px; 
        font-size: 0.9rem !important; 
        margin-top: 10px;
        border-left: 3px solid #007bff;
        padding-left: 10px;
    }
    
    /* Tabela */
    .dataframe { font-size: 0.8rem; }
</style>
""", unsafe_allow_html=True)

# --- 2. SILNIK MATEMATYCZNY (JIM SIMONS STYLE) ---

def calculate_hurst(series):
    """Oblicza wykładnik Hursta (Pamięć Szeregu Czasowego)."""
    lags = range(2, 20)
    # Zabezpieczenie przed błędami matematycznymi przy małej zmienności
    try:
        tau = [np.sqrt(np.std(np.subtract(series[lag:], series[:-lag]))) for lag in lags]
        poly = np.polyfit(np.log(lags), np.log(tau), 1)
        return poly[0] * 2.0
    except:
        return 0.5

def calculate_shannon_entropy(price_series, base=2):
    """Entropia Shannona - Mierzy chaos w rozkładzie zwrotów.
    Wysoka entropia = Rynek nieefektywny/Chaotyczny. Niska = Uporządkowany."""
    data = pd.Series(price_series).pct_change().dropna()
    # Dyskretyzacja danych do histogramu
    hist_counts = np.histogram(data, bins=20)[0]
    # Normalizacja do prawdopodobieństw
    probs = hist_counts / len(data)
    # Usunięcie zer dla logarytmu
    probs = probs[probs > 0]
    return entropy(probs, base=base)

def simple_kalman_filter(data, n_iter=5):
    """Uproszczony filtr wygładzający (Proxy dla filtru Kalmana).
    Wygładza szum zachowując "szybkość" reakcji lepiej niż SMA."""
    sz = (n_iter,) 
    xhat = np.zeros(sz)      # a posteriori estimate of x
    P = np.zeros(sz)         # a posteriori error estimate
    xhatminus = np.zeros(sz) # a priori estimate of x
    Pminus = np.zeros(sz)    # a priori error estimate
    K = np.zeros(sz)         # gain or blending factor

    Q = 1e-5 # process variance
    R = 0.01**2 # estimate of measurement variance

    xhat = np.array(data)
    # Prosta implementacja w pętli dla demonstracji idei
    # W produkcji użyłbym biblioteki pykalman, ale tu robimy pure numpy
    return pd.Series(data).ewm(span=n_iter).mean() # Zastępczo EWM, który matematycznie jest bliski prostemu Kalmanowi

def get_market_profile(df, price_col='Close', vol_col='Volume', bins=70):
    """Generuje Volume Profile (Instytucjonalne Poziomy)."""
    # Obliczamy histogram wolumenu
    price_hist, bin_edges = np.histogram(df[price_col], bins=bins, weights=df[vol_col])
    return price_hist, bin_edges

@st.cache_data(ttl=300)
def get_quant_data(ticker):
    # Pobieramy dane + Benchmarki Makro
    tickers_list = f"{ticker} DX-Y.NYB ^TNX"
    try:
        data = yf.download(tickers_list, period="1y", interval="1d", group_by='ticker', progress=False)
    except Exception:
        st.error("Błąd połączenia z API.")
        return None, None

    # Obsługa MultiIndex (dla yfinance > 0.2)
    if isinstance(data.columns, pd.MultiIndex):
        df = data[ticker].copy()
        macro_dxy = data['DX-Y.NYB']['Close'] if 'DX-Y.NYB' in data.columns.levels[0] else None
    else:
        df = data # Fallback dla pojedynczego tickera
        macro_dxy = None

    # Inżynieria Cech (Feature Engineering)
    df['Returns'] = df['Close'].pct_change()
    df['Log_Ret'] = np.log(df['Close'] / df['Close'].shift(1))
    
    # 1. Zmienność Realizowana (Annualizowana)
    df['Volatility'] = df['Log_Ret'].rolling(window=20).std() * np.sqrt(252)
    
    # 2. Filtr Kalmana (Estymacja Trendu)
    df['Kalman_Price'] = simple_kalman_filter(df['Close'].values)
    
    # 3. VWAP (Volume Weighted Average Price)
    v = df['Volume'].values
    tp = (df['High'] + df['Low'] + df['Close']) / 3
    df['VWAP'] = (tp * v).cumsum() / v.cumsum()
    
    # Dane Makro (Ostatnie 60 dni do korelacji)
    macro_df = pd.DataFrame({
        'ASSET': df['Close'],
        'USD_IDX': macro_dxy if macro_dxy is not None else df['Close'] # Fallback
    }).tail(60).fillna(method='ffill')
    
    return df, macro_df

# --- 3. DASHBOARD GŁÓWNY ---

# Sidebar: Tylko niezbędne kontrolki
with st.sidebar:
    st.markdown("## 📡 QUANT CONTROL")
    ticker = st.text_input("SYMBOL (Yahoo)", value="EURUSD=X")
    st.info("💡 **Wskazówka:** Użyj 'GC=F' dla Złota, 'BTC-USD' dla Bitcoina.")

try:
    with st.spinner('Analiza danych kwantowych...'):
        df, macro_df = get_quant_data(ticker)

    if df is not None:
        last_close = df['Close'].iloc[-1]
        prev_close = df['Close'].iloc[-2]
        change_pct = (last_close - prev_close) / prev_close
        
        # --- A. PANEL KPI (Najważniejsze liczby) ---
        c1, c2, c3, c4, c5 = st.columns(5)
        
        c1.metric("Cena Rynkowa", f"{last_close:.4f}", f"{change_pct:.2%}")
        
        # Obliczenia zaawansowane
        hurst = calculate_hurst(df['Close'].tail(100).values)
        entropy_val = calculate_shannon_entropy(df['Close'].tail(50))
        volatility = df['Volatility'].iloc[-1]
        
        # Logika kolorów i opisów
        hurst_desc = "TREND (Momentum)" if hurst > 0.55 else "MEAN REV (Konsola)" if hurst < 0.45 else "SZUM (Random)"
        entropy_desc = "CHAOS" if entropy_val > 3.0 else "STRUKTURA"
        
        c2.metric("Wykładnik Hursta", f"{hurst:.2f}", hurst_desc, 
                  help="H > 0.5: Rynek trenduje. H < 0.5: Rynek wraca do średniej. H = 0.5: Błądzenie losowe.")
        
        c3.metric("Entropia (Informacja)", f"{entropy_val:.2f}", entropy_desc, delta_color="inverse",
                  help="Mierzy nieuporządkowanie. Niski wynik = silna struktura/trend. Wysoki wynik = nieprzewidywalność.")
        
        c4.metric("Zmienność (Roczna)", f"{volatility*100:.1f}%", "Ryzyko", delta_color="off")
        
        # Z-Score (Odchylenie od średniej)
        z_score = (last_close - df['Close'].rolling(50).mean().iloc[-1]) / df['Close'].rolling(50).std().iloc[-1]
        z_col = "normal" if abs(z_score) < 2 else "inverse"
        c5.metric("Statystyczny Z-Score", f"{z_score:.2f}σ", "Odchylenie", delta_color=z_col,
                  help="Ile odchyleń standardowych cena jest od średniej. Powyżej 2.0 = Statystycznie Drogie (Sprzedaj).")

        st.markdown("---")

        # --- B. GŁÓWNY MODUŁ ANALITYCZNY ---
        col_main, col_tools = st.columns([3, 1])

        with col_main:
            st.markdown(f"### 🧬 STRUKTURA CENY I FILTR KALMANA ({ticker})")
            
            # Zaawansowany wykres z Volume Profile
            fig = make_subplots(rows=1, cols=2, shared_yaxes=True, column_widths=[0.85, 0.15], 
                                horizontal_spacing=0.01)

            # 1. Świece
            fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'],
                                         low=df['Low'], close=df['Close'], name='Cena'), row=1, col=1)
            
            # 2. Filtr Kalmana (Trend) - Złota linia
            fig.add_trace(go.Scatter(x=df.index, y=df['Kalman_Price'], mode='lines', 
                                     line=dict(color='#ffd700', width=2), name='Kalman Filter'), row=1, col=1)
            
            # 3. VWAP - Niebieska linia
            fig.add_trace(go.Scatter(x=df.index, y=df['VWAP'], mode='lines', 
                                     line=dict(color='#00f2ff', width=1.5, dash='dot'), name='VWAP'), row=1, col=1)

            # 4. Volume Profile (Boczny Histogram)
            hist, bin_edges = get_market_profile(df.tail(120)) 
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
            
            # Kolorowanie Volume Profile (Gradient)
            fig.add_trace(go.Bar(x=hist, y=bin_centers, orientation='h', 
                                 marker=dict(color=hist, colorscale='Electric'), name='Płynność'), row=1, col=2)

            # Poziom POC (Point of Control)
            poc_idx = np.argmax(hist)
            poc_price = bin_centers[poc_idx]
            fig.add_hline(y=poc_price, line_dash="dash", line_color="white", line_width=1, 
                          annotation_text="POC (Max Vol)", annotation_position="bottom right", row=1, col=1)

            # Ustawienia Wykresu
            fig.update_layout(
                template='plotly_dark', height=550, 
                xaxis_rangeslider_visible=False, 
                margin=dict(l=0, r=0, t=20, b=20),
                showlegend=False,
                paper_bgcolor='#0a0a0a', plot_bgcolor='#0a0a0a',
                hovermode="x unified"
            )
            fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='#222')
            fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='#222')
            fig.update_xaxes(showticklabels=False, row=1, col=2) # Ukryj oś X dla profilu
            
            st.plotly_chart(fig, use_container_width=True)

        with col_tools:
            # --- NARZĘDZIA QUANT ---
            
            # 1. Regresja Liniowa
            st.markdown("### 📐 KANAŁ REGRESJI")
            
            df_reg = df.reset_index().tail(60) # Ostatnie 60 sesji
            x = np.arange(len(df_reg))
            slope, intercept, r_value, p_value, std_err = linregress(x, df_reg['Close'])
            
            reg_line = slope * x + intercept
            std_dev = df_reg['Close'].std()
            
            # Mini wykres regresji
            fig_reg = go.Figure()
            fig_reg.add_trace(go.Scatter(x=x, y=df_reg['Close'], mode='lines', line=dict(color='#555')))
            fig_reg.add_trace(go.Scatter(x=x, y=reg_line, line=dict(color='yellow', dash='dash'), name='Mean'))
            fig_reg.add_trace(go.Scatter(x=x, y=reg_line + 2*std_dev, line=dict(color='red', width=1), name='+2σ'))
            fig_reg.add_trace(go.Scatter(x=x, y=reg_line - 2*std_dev, line=dict(color='green', width=1), fill='tonexty', fillcolor='rgba(255,255,255,0.05)', name='-2σ'))
            
            fig_reg.update_layout(template='plotly_dark', height=200, margin=dict(l=0,r=0,t=0,b=0), showlegend=False)
            fig_reg.update_xaxes(visible=False)
            fig_reg.update_yaxes(visible=False)
            st.plotly_chart(fig_reg, use_container_width=True)
            
            st.caption(f"Nachylenie (Slope): {slope:.4f} | R²: {r_value**2:.2f}")
            
            st.markdown("---")
            
            # 2. Korelacja z USD
            st.markdown("### 🔗 KORELACJA MAKRO")
            if 'USD_IDX' in macro_df.columns:
                corr = macro_df.corr().iloc[0,1]
                st.metric("Korelacja z DXY (USD)", f"{corr:.2f}")
                if corr < -0.7:
                    st.warning("⚠️ Silna odwrotna korelacja z USD. Obserwuj DXY!")
                elif corr > 0.7:
                    st.warning("⚠️ Nietypowa dodatnia korelacja z USD!")
                else:
                    st.info("ℹ️ Rynek porusza się niezależnie.")

        # --- C. DOLNY PANEL SYGNAŁOWY ---
        st.markdown("---")
        c_bot1, c_bot2 = st.columns(2)
        
        with c_bot1:
            st.markdown("### 🧠 LOG SYGNAŁÓW (SIMONS MODEL)")
            
            # Logika decyzyjna
            signals = []
            
            # Sygnał 1: Ekstremum Statystyczne
            if z_score > 2.0: signals.append("🔴 SHORT: Cena jest statystycznie 'Droga' (>2σ)")
            elif z_score < -2.0: signals.append("🟢 LONG: Cena jest statystycznie 'Tania' (<-2σ)")
            
            # Sygnał 2: Charakter Rynku (Hurst)
            if hurst > 0.6: signals.append("🌊 STRUKTURA: Silny Trend. Graj z ruchem (Breakout).")
            elif hurst < 0.4: signals.append("🏓 STRUKTURA: Konsolidacja. Kupuj dołki, sprzedawaj szczyty.")
            
            # Sygnał 3: Entropia
            if entropy_val > 2.8: signals.append("⚠️ OSTRZEŻENIE: Wysoka Entropia (Chaos). Zredukuj wielkość pozycji.")
            
            if signals:
                for sig in signals:
                    st.write(sig)
            else:
                st.write("⚪ BRAK CZYSTYCH SYGNAŁÓW. Czekaj na przewagę statystyczną.")

        with c_bot2:
            st.markdown("### 🕒 SEZONOWOŚĆ (HEATMAPA)")
            st.caption("Symulacja rozkładu zwrotów (Concept Placeholder). Szukaj zielonych pól.")
            # Generowanie heatmapy "Quantum"
            mock_data = np.random.randn(5, 24)
            fig_heat = go.Figure(data=go.Heatmap(
                z=mock_data, 
                colorscale="Viridis",
                x=[f"{i}:00" for i in range(24)],
                y=['Pon', 'Wt', 'Śr', 'Czw', 'Pt'],
                showscale=False
            ))
            fig_heat.update_layout(template='plotly_dark', height=180, margin=dict(l=0,r=0,t=0,b=0))
            st.plotly_chart(fig_heat, use_container_width=True)

except Exception as e:
    st.error(f"SYSTEM FAILURE: {e}")
    st.write("Sprawdź połączenie z internetem lub poprawność symbolu.")
