import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from datetime import datetime
import pytz

# --- 1. KONFIGURACJA STRONY (Full Wide Mode) ---
st.set_page_config(layout="wide", page_title="EURUSD Terminal Pro", page_icon="🦅")

# --- 2. CSS STYLING (To jest klucz do tego wyglądu) ---
st.markdown("""
<style>
    /* TŁO I GŁÓWNY KONTENER */
    .stApp { background-color: #0E1117; } /* Bardzo ciemne tło */
    .block-container { 
        padding-top: 1rem; 
        padding-left: 1rem; 
        padding-right: 1rem; 
        max-width: 100%; 
    }

    /* UKRYCIE ELEMENTÓW STREAMLIT */
    header { visibility: hidden; }
    footer { visibility: hidden; }

    /* STYLIZACJA KAFELKÓW (Metrics) */
    .metric-box {
        background-color: #161B22; /* Trochę jaśniejszy szary */
        border: 1px solid #30363D;
        border-radius: 6px;
        padding: 15px;
        text-align: center;
        box-shadow: 0 2px 4px rgba(0,0,0,0.2);
    }
    .metric-label { font-size: 0.8rem; color: #8B949E; text-transform: uppercase; letter-spacing: 1px; }
    .metric-value { font-size: 1.6rem; font-family: 'Roboto Mono', monospace; font-weight: 700; color: #E6EDF3; }
    .metric-delta-pos { color: #3FB950; font-size: 0.9rem; font-weight: bold; }
    .metric-delta-neg { color: #F85149; font-size: 0.9rem; font-weight: bold; }

    /* STYLIZACJA PRZYCISKU SESJI (Pill Button) */
    .session-box {
        background-color: #1F6FEB;
        color: white;
        padding: 5px 15px;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: bold;
        display: inline-block;
        margin-top: 10px;
    }
    
    /* GLOBALNY TEXT */
    p, label, span { color: #C9D1D9; }
</style>
""", unsafe_allow_html=True)

# --- 3. DANE (SZYBKI CACHE) ---
@st.cache_data(ttl=30) # Odświeżanie co 30 sek
def get_data():
    # Pobieramy EURUSD, DXY i Zmienność
    tickers = "EURUSD=X DX-Y.NYB"
    data = yf.download(tickers, period="5d", interval="15m", group_by='ticker', progress=False)
    return data

# Pobranie danych
try:
    data = get_data()
    df_eur = data['EURUSD=X']
    df_dxy = data['DX-Y.NYB']
    
    # Obsługa braków danych (Forward Fill)
    df_dxy = df_dxy.resample('15min').ffill().reindex(df_eur.index, method='ffill')

    # Ostatnie wartości
    last_price = df_eur['Close'].iloc[-1]
    prev_price = df_eur['Close'].iloc[-2]
    change = last_price - prev_price
    pct_change = (change / prev_price) * 100
    
    dxy_last = df_dxy['Close'].iloc[-1]
    dxy_change = dxy_last - df_dxy['Close'].iloc[-2]

    # Obliczenie Volatility (ATR-like)
    high_low = df_eur['High'] - df_eur['Low']
    volatility = high_low.tail(10).mean() * 10000 # w pipsach

except Exception as e:
    st.error("Ładowanie danych...")
    st.stop()

# --- 4. LAYOUT: GÓRNY PASEK (KPIs) ---
# Używamy HTML/CSS zamiast st.metric dla idealnego wyglądu "kafelków"
col1, col2, col3, col4 = st.columns(4)

def kpi_card(label, value, delta, is_pct=False):
    color_class = "metric-delta-pos" if delta >= 0 else "metric-delta-neg"
    sign = "+" if delta >= 0 else ""
    delta_str = f"{sign}{delta:.2f}%" if is_pct else f"{sign}{delta:.4f}"
    return f"""
    <div class="metric-box">
        <div class="metric-label">{label}</div>
        <div class="metric-value">{value}</div>
        <div class="{color_class}">{delta_str}</div>
    </div>
    """

with col1: st.markdown(kpi_card("EUR / USD", f"{last_price:.5f}", pct_change, is_pct=True), unsafe_allow_html=True)
with col2: st.markdown(kpi_card("DOLLAR INDEX", f"{dxy_last:.2f}", dxy_change), unsafe_allow_html=True)
with col3: st.markdown(kpi_card("CORRELATION (50)", f"{df_eur['Close'].tail(50).corr(df_dxy['Close'].tail(50)):.2f}", 0.0), unsafe_allow_html=True)
with col4: st.markdown(kpi_card("AVG VOLATILITY (10)", f"{volatility:.1f} pips", 0.0), unsafe_allow_html=True)

st.markdown("---") # Cienka linia oddzielająca

# --- 5. LAYOUT: GŁÓWNA SEKCJA (WYKRES + SIDEBAR) ---
# Dzielimy ekran: Lewa (Wykres 75%) | Prawa (Market Depth 25%)
col_chart, col_sidebar = st.columns([3, 1])

with col_chart:
    # --- A. WYKRES GŁÓWNY (Stylizowany na Area Chart jak na screenie) ---
    st.markdown("##### 📉 EURUSD Price Action")
    
    fig = go.Figure()

    # Wykres liniowy z wypełnieniem (Area Chart)
    fig.add_trace(go.Scatter(
        x=df_eur.index, y=df_eur['Close'],
        mode='lines',
        fill='tozeroy', # Wypełnienie pod wykresem
        name='Price',
        line=dict(color='#2E9AFE', width=2), # Jasnoniebieski
        fillcolor='rgba(46, 154, 254, 0.1)' # Przezroczyste wypełnienie
    ))

    # Dodanie prostej średniej
    fig.add_trace(go.Scatter(
        x=df_eur.index, y=df_eur['Close'].rolling(50).mean(),
        mode='lines',
        name='SMA 50',
        line=dict(color='#FF9F1C', width=1, dash='dot')
    ))

    # Konfiguracja wyglądu (Grid, Tło)
    fig.update_layout(
        height=550,
        margin=dict(l=10, r=10, t=30, b=10),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='#0E1117',
        showlegend=False,
        xaxis=dict(showgrid=True, gridcolor='#1F242D', gridwidth=1),
        yaxis=dict(showgrid=True, gridcolor='#1F242D', gridwidth=1, side='right') # Cena po prawej
    )
    
    # Ukrywanie weekendów
    fig.update_xaxes(rangebreaks=[dict(bounds=["sat", "mon"])])
    
    st.plotly_chart(fig, use_container_width=True, config={'scrollZoom': True, 'displayModeBar': False})
    
    # --- B. WYKRES KORELACJI (Pod spodem, mniejszy) ---
    # Symulacja oscylatora na dole (np. RSI lub Correlation)
    st.markdown("##### 📊 DXY Correlation Check")
    fig_corr = go.Figure()
    fig_corr.add_trace(go.Scatter(
        x=df_eur.index, y=df_eur['Close'].rolling(30).corr(df_dxy['Close'].rolling(30)),
        line=dict(color='#F85149', width=1.5)
    ))
    fig_corr.update_layout(height=150, margin=dict(t=0, b=0, l=10, r=10), 
                           paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='#0E1117',
                           xaxis=dict(showticklabels=False, showgrid=False),
                           yaxis=dict(showgrid=True, gridcolor='#1F242D'))
    st.plotly_chart(fig_corr, use_container_width=True, config={'staticPlot': True})

with col_sidebar:
    # --- C. MARKET DEPTH (Symulacja UI z prawej strony) ---
    st.markdown("##### 🧱 Market Depth")
    
    # Tworzymy wykres słupkowy poziomy, symulujący Order Book
    # Generujemy sztuczne dane, aby wyglądało jak na screenie (Bid/Ask volume)
    depth_data = pd.DataFrame({
        "Price": [last_price + i*0.0005 for i in range(-5, 6)],
        "Volume": np.random.randint(100, 1000, 11),
        "Type": ["Ask"]*5 + ["Spread"] + ["Bid"]*5
    })
    
    colors = ['#F85149' if t == "Ask" else '#3FB950' for t in depth_data['Type']]
    
    fig_depth = go.Figure(go.Bar(
        x=depth_data['Volume'],
        y=depth_data['Price'],
        orientation='h', # Poziomy
        marker_color=colors,
        text=depth_data['Volume'],
        textposition='inside',
        insidetextanchor='middle'
    ))
    
    fig_depth.update_layout(
        height=400,
        margin=dict(l=0, r=0, t=0, b=0),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='#0E1117',
        xaxis=dict(visible=False),
        yaxis=dict(showticklabels=True, tickformat=".4f", color="#8B949E"),
        bargap=0.1
    )
    st.plotly_chart(fig_depth, use_container_width=True, config={'displayModeBar': False})

    # --- D. SESJA (Pill Button) ---
    st.markdown("##### 🌍 Status Sesji")
    
    utc_now = datetime.now(pytz.utc).hour
    session_label = "OFFLINE"
    if 7 <= utc_now < 16: session_label = "🇬🇧 LONDON OPEN"
    elif 12 <= utc_now < 21: session_label = "🇺🇸 NEW YORK OPEN"
    
    # HTML Pill Button
    st.markdown(f"""
    <div style="text-align: left;">
        <span class="session-box">{session_label}</span>
    </div>
    <div style="margin-top: 10px; font-size: 0.8rem; color: #8B949E;">
        Current UTC: {utc_now}:00
    </div>
    """, unsafe_allow_html=True)
    
    # Notatnik
    st.markdown("##### 📝 Notatki")
    st.text_area("Plan:", height=150, placeholder="Czekam na retest poziomu...", label_visibility="collapsed")
