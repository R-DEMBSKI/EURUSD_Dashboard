import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
import time

# --- 1. KONFIGURACJA UI (STYL TWS/BLOOMBERG) ---
st.set_page_config(layout="wide", page_title="TWS PRO SIMULATOR", page_icon="📈")

# CSS: Totalna konwersja wyglądu na styl Interactive Brokers
st.markdown("""
<style>
    /* Główne tło - Ciemny Szary TWS */
    .stApp { background-color: #1a1a1a; color: #e0e0e0; font-family: 'Arial', sans-serif; }
    
    /* Ukrycie paska Streamlit */
    header {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Kontenery (Moduły) */
    .css-1r6slb0, .css-12oz5g7 { padding: 0px 1rem; }
    
    /* Stylizacja Przycisków BUY/SELL */
    div.stButton > button:first-child {
        border-radius: 4px;
        font-weight: bold;
        border: none;
        width: 100%;
        padding: 10px;
    }
    
    /* Inputy numeryczne (wygląd terminala) */
    div[data-testid="stNumberInput"] input {
        background-color: #000;
        color: #fff;
        border: 1px solid #444;
        font-family: 'Consolas', monospace;
    }
    
    /* Tabele (Dataframes) */
    div[data-testid="stDataFrame"] {
        font-size: 0.8rem;
    }
    
    /* Nagłówki modułów */
    h3 {
        font-size: 1rem !important;
        background-color: #2d2d2d;
        padding: 5px 10px;
        border-top: 2px solid #007bff; /* Niebieska linia TWS */
        margin-bottom: 10px;
        color: white !important;
        font-weight: 600;
    }
    
    /* Kolorystyka zysków i strat */
    .positive { color: #00ff00; font-weight: bold; }
    .negative { color: #ff3333; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

# --- 2. LOGIKA SYMULATORA PORTFELA (SESSION STATE) ---
if 'portfolio' not in st.session_state:
    st.session_state.portfolio = pd.DataFrame(columns=['Symbol', 'Qty', 'Avg Price', 'Value', 'Unrealized P&L'])
if 'cash' not in st.session_state:
    st.session_state.cash = 100000.0  # Wirtualne 100k USD
if 'trade_log' not in st.session_state:
    st.session_state.trade_log = []

# --- 3. POBIERANIE DANYCH (YFINANCE) ---
@st.cache_data(ttl=60) # Odświeżanie co minutę
def get_market_data(ticker):
    data = yf.download(ticker, period="5d", interval="15m", progress=False)
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.droplevel(1)
    
    # Proste wskaźniki
    data['SMA20'] = data['Close'].rolling(20).mean()
    data['Upper'] = data['SMA20'] + 2*data['Close'].rolling(20).std()
    data['Lower'] = data['SMA20'] - 2*data['Close'].rolling(20).std()
    return data

# --- 4. LAYOUT GŁÓWNY ---
# Podział na 2 główne kolumny: LEWA (Order + Chart) i PRAWA (Portfolio + News)
col_left, col_right = st.columns([4, 6], gap="small")

ticker = "EURUSD=X" # Domyślny ticker
data = get_market_data(ticker)
current_price = data['Close'].iloc[-1]
prev_close = data['Close'].iloc[-2]
pct_change = (current_price - prev_close) / prev_close

# === LEWA KOLUMNA: ORDER ENTRY & CHART ===
with col_left:
    # A. ORDER ENTRY PANEL (Niebieski nagłówek)
    st.markdown("### 🛒 ORDER ENTRY (SIMULATION)")
    
    # Wiersz z tickerem i ceną
    c1, c2, c3 = st.columns([2, 2, 2])
    c1.text_input("Symbol", value="EURUSD", disabled=True)
    color_price = "green" if pct_change > 0 else "red"
    c2.markdown(f"<div style='text-align:center; font-size:1.5em; color:{color_price}; font-family:monospace'>{current_price:.5f}</div>", unsafe_allow_html=True)
    c3.markdown(f"<div style='text-align:center; color:{color_price}'>{pct_change:.2%}</div>", unsafe_allow_html=True)
    
    # Parametry zlecenia
    st.markdown("---")
    qc1, qc2 = st.columns(2)
    qty = qc1.number_input("QTY", min_value=1000, value=10000, step=1000)
    order_type = qc2.selectbox("Type", ["MKT", "LMT (Simulated)"])
    
    # Przyciski akcji
    b1, b2 = st.columns(2)
    
    # Logika BUY
    if b1.button("BUY 🔵", use_container_width=True):
        cost = qty * current_price
        if st.session_state.cash >= cost:
            st.session_state.cash -= cost
            
            # Dodaj do portfela
            new_trade = pd.DataFrame([{
                'Symbol': ticker, 'Qty': qty, 'Avg Price': current_price, 
                'Value': cost, 'Unrealized P&L': 0.0
            }])
            
            if ticker in st.session_state.portfolio['Symbol'].values:
                # Uśrednianie ceny (proste)
                idx = st.session_state.portfolio[st.session_state.portfolio['Symbol'] == ticker].index[0]
                old_qty = st.session_state.portfolio.at[idx, 'Qty']
                old_val = st.session_state.portfolio.at[idx, 'Value']
                st.session_state.portfolio.at[idx, 'Qty'] += qty
                st.session_state.portfolio.at[idx, 'Value'] += cost
                st.session_state.portfolio.at[idx, 'Avg Price'] = (old_val + cost) / (old_qty + qty)
            else:
                st.session_state.portfolio = pd.concat([st.session_state.portfolio, new_trade], ignore_index=True)
                
            st.success(f"BOUGHT {qty} @ {current_price:.5f}")
            st.session_state.trade_log.append(f"{datetime.now().strftime('%H:%M:%S')} BUY {qty} {ticker} @ {current_price}")
        else:
            st.error("INSUFFICIENT FUNDS")

    # Logika SELL (Zamknięcie pozycji)
    if b2.button("SELL 🔴", type="primary", use_container_width=True):
        if ticker in st.session_state.portfolio['Symbol'].values:
            idx = st.session_state.portfolio[st.session_state.portfolio['Symbol'] == ticker].index[0]
            current_qty = st.session_state.portfolio.at[idx, 'Qty']
            
            if current_qty >= qty:
                revenue = qty * current_price
                st.session_state.cash += revenue
                st.session_state.portfolio.at[idx, 'Qty'] -= qty
                st.session_state.portfolio.at[idx, 'Value'] -= (qty * st.session_state.portfolio.at[idx, 'Avg Price'])
                
                # Usuń jeśli 0
                if st.session_state.portfolio.at[idx, 'Qty'] == 0:
                    st.session_state.portfolio = st.session_state.portfolio.drop(idx)
                
                st.warning(f"SOLD {qty} @ {current_price:.5f}")
                st.session_state.trade_log.append(f"{datetime.now().strftime('%H:%M:%S')} SELL {qty} {ticker} @ {current_price}")
            else:
                st.error("NOT ENOUGH SHARES")
        else:
            st.error("NO POSITION")

    # B. CHART PANEL
    st.markdown("### 📉 CHART (15m)")
    
    fig = go.Figure()
    
    # Świece
    fig.add_trace(go.Candlestick(x=data.index,
                    open=data['Open'], high=data['High'],
                    low=data['Low'], close=data['Close'],
                    name='Price'))
    
    # Bollinger (Subtelne)
    fig.add_trace(go.Scatter(x=data.index, y=data['Upper'], line=dict(color='rgba(255,255,255,0.2)', width=1), showlegend=False))
    fig.add_trace(go.Scatter(x=data.index, y=data['Lower'], line=dict(color='rgba(255,255,255,0.2)', width=1), fill='tonexty', fillcolor='rgba(255,255,255,0.05)', showlegend=False))

    fig.update_layout(
        template='plotly_dark',
        height=400,
        margin=dict(l=0, r=0, t=10, b=0),
        xaxis_rangeslider_visible=False,
        paper_bgcolor='#1a1a1a',
        plot_bgcolor='#000000',
        font=dict(family="Consolas", size=10)
    )
    st.plotly_chart(fig, use_container_width=True)

# === PRAWA KOLUMNA: PORTFOLIO & MONITOR ===
with col_right:
    # C. PORTFOLIO PANEL
    st.markdown("### 💼 PORTFOLIO & ACCOUNT")
    
    # Account Summary (Top Bar)
    ac1, ac2, ac3 = st.columns(3)
    ac1.metric("Net Liquidity", f"${st.session_state.cash + st.session_state.portfolio['Value'].sum():,.2f}")
    ac2.metric("Cash Balance", f"${st.session_state.cash:,.2f}")
    
    # Aktualizacja P&L na żywo
    total_unrealized = 0
    if not st.session_state.portfolio.empty:
        for index, row in st.session_state.portfolio.iterrows():
            # Symulacja live price (dla uproszczenia bierzemy ostatnią cenę z wykresu)
            # W pełnej wersji tutaj byłoby zapytanie o cenę dla każdego tickera
            mkt_price = current_price if row['Symbol'] == ticker else row['Avg Price'] 
            val = row['Qty'] * mkt_price
            pnl = val - (row['Qty'] * row['Avg Price'])
            
            st.session_state.portfolio.at[index, 'Value'] = val
            st.session_state.portfolio.at[index, 'Unrealized P&L'] = pnl
            total_unrealized += pnl

    ac3.metric("Unrealized P&L", f"${total_unrealized:,.2f}", delta_color="normal")

    st.markdown("---")
    
    # Wyświetlanie Tabeli Portfela
    if not st.session_state.portfolio.empty:
        # Formatowanie tabeli
        st.dataframe(
            st.session_state.portfolio.style.format({
                "Qty": "{:,.0f}",
                "Avg Price": "{:.4f}",
                "Value": "{:,.2f}",
                "Unrealized P&L": "{:,.2f}"
            }).applymap(lambda x: 'color: #00ff00' if x > 0 else 'color: #ff3333', subset=['Unrealized P&L']),
            use_container_width=True,
            height=300
        )
    else:
        st.info("No active positions. Use Order Entry to trade.")

    # D. MARKET SCANNER / NEWS (Statyczna lista dla stylu)
    st.markdown("### 📰 NEWS & EVENTS")
    news_data = pd.DataFrame({
        "Time": ["12:05", "11:45", "10:30", "09:15", "08:00"],
        "Source": ["DJ", "RTRS", "BBG", "DJ", "RTRS"],
        "Headline": [
            "ECB Signals Rate Pause in December",
            "EURUSD breaks key resistance at 1.1650",
            "US Initial Jobless Claims lower than expected",
            "Tech Sector leads rally in pre-market",
            "Oil prices stabilize after inventory data"
        ]
    })
    st.dataframe(news_data, hide_index=True, use_container_width=True)

# Pasek stanu na dole
st.markdown("---")
st.caption(f"CONNECTED: {ticker} | DATA: DELAYED 15min (YFINANCE) | MODE: PAPER TRADING")
