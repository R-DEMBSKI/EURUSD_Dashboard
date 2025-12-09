import streamlit as st
import streamlit.components.v1 as components
import yfinance as yf
import pandas as pd
import numpy as np

# --- 1. KONFIGURACJA STRONY ---
st.set_page_config(layout="wide", page_title="EURUSD Quant Hub", page_icon="🧠")

# --- 2. CSS (Quant Style) ---
st.markdown("""
<style>
    .stApp { background-color: #0e1117; }
    .block-container { padding: 0.5rem 1rem; }
    
    /* Stylizacja tabel analitycznych */
    .dataframe { font-size: 0.8rem !important; }
    
    /* Nagłówki sekcji */
    h3 { border-bottom: 2px solid #2ea043; padding-bottom: 5px; margin-top: 0px; }
</style>
""", unsafe_allow_html=True)

# --- 3. WIDGETY TRADINGVIEW (HTML/JS) ---
# Funkcja pomocnicza do renderowania widgetów
def tv_chart_widget(symbol="FX:EURUSD", theme="dark"):
    # Kod embed z TradingView
    code = f"""
    <div class="tradingview-widget-container">
      <div id="tradingview_chart"></div>
      <script type="text/javascript" src="https://s3.tradingview.com/tv.js"></script>
      <script type="text/javascript">
      new TradingView.widget(
      {{
        "width": "100%",
        "height": 600,
        "symbol": "{symbol}",
        "interval": "5",
        "timezone": "Etc/UTC",
        "theme": "{theme}",
        "style": "1",
        "locale": "pl",
        "toolbar_bg": "#f1f3f6",
        "enable_publishing": false,
        "hide_side_toolbar": false,
        "allow_symbol_change": true,
        "container_id": "tradingview_chart"
      }});
      </script>
    </div>
    """
    components.html(code, height=600)

def tv_technical_widget(symbol="FX:EURUSD"):
    code = f"""
    <div class="tradingview-widget-container">
      <div class="tradingview-widget-container__widget"></div>
      <script type="text/javascript" src="https://s3.tradingview.com/external-embedding/embed-widget-technical-analysis.js" async>
      {{
      "interval": "5m",
      "width": "100%",
      "isTransparent": true,
      "height": 400,
      "symbol": "{symbol}",
      "showIntervalTabs": true,
      "displayMode": "single",
      "locale": "pl",
      "colorTheme": "dark"
    }}
      </script>
    </div>
    """
    components.html(code, height=400)

# --- 4. PYTHON ANALYTICAL BRAIN (To jest Twoje 'Centrum Obliczeniowe') ---
@st.cache_data(ttl=60)
def run_quant_analysis():
    # Pobieramy dane do matematyki
    tickers = "EURUSD=X DX-Y.NYB ^TNX ^DE10Y GC=F" 
    # ^DE10Y = Niemieckie obligacje (Spread US vs DE to klucz do EURUSD)
    
    data = yf.download(tickers, period="1mo", interval="1h", group_by='ticker', progress=False)
    
    # Przygotowanie DataFrame'ów
    eur = data['EURUSD=X']['Close']
    dxy = data['DX-Y.NYB']['Close']
    us10y = data['^TNX']['Close']
    de10y = data['^DE10Y']['Close'] # Może mieć braki, trzeba uważać
    gold = data['GC=F']['Close']
    
    # 1. Yield Spread Analysis (Fundamenty)
    # Jeśli US yields rosną szybciej niż DE yields -> Dolar zyskuje -> EURUSD spada
    # Musimy wyrównać indeksy (ffill)
    common_idx = eur.index
    us10y = us10y.reindex(common_idx).ffill()
    de10y = de10y.reindex(common_idx).ffill()
    spread = us10y - de10y # Spread rentowności
    
    # 2. Korelacje (Heatmap Data)
    df_corr = pd.DataFrame({
        "EURUSD": eur,
        "DXY": dxy.reindex(common_idx).ffill(),
        "Gold": gold.reindex(common_idx).ffill(),
        "Yield Spread": spread
    }).tail(50) # Ostatnie 50 godzin handlu
    
    correlation_matrix = df_corr.corr()
    
    # 3. Z-Score (Czy cena jest statystycznie "za tania" lub "za droga"?)
    # Liczymy na 20-okresowej średniej
    ma20 = eur.rolling(20).mean()
    std20 = eur.rolling(20).std()
    z_score = (eur.iloc[-1] - ma20.iloc[-1]) / std20.iloc[-1]
    
    # 4. Volatility Scanner (ATR - Average True Range)
    high = data['EURUSD=X']['High']
    low = data['EURUSD=X']['Low']
    tr = high - low
    atr_current = tr.tail(5).mean()
    atr_historical = tr.tail(100).mean()
    volatility_ratio = atr_current / atr_historical
    
    return correlation_matrix, z_score, spread.iloc[-1], volatility_ratio, eur.iloc[-1]

# --- 5. UKŁAD STRONY (LAYOUT) ---

st.title("🦅 EUR/USD Quant Command Center")

# Uruchamiamy mózg analityczny
try:
    corr_matrix, z_score, current_spread, vol_ratio, current_price = run_quant_analysis()
    
    # --- GÓRNY PASEK METRYK (Obliczone przez Pythona) ---
    col1, col2, col3, col4 = st.columns(4)
    
    col1.metric("EURUSD Cena", f"{current_price:.5f}")
    
    # Z-SCORE Logic
    z_color = "normal"
    if z_score > 2.0: z_color = "inverse" # Wykupiony (Overbought)
    if z_score < -2.0: z_color = "inverse" # Wyprzedany (Oversold)
    col2.metric("Statystyka (Z-Score)", f"{z_score:.2f}σ", delta_color=z_color, 
                help="Powyżej 2.0 = Statystycznie drogo (Możliwa korekta). Poniżej -2.0 = Statystycznie tanio.")
    
    # Spread Obligacji
    col3.metric("US-DE Yield Spread", f"{current_spread:.3f}%", 
                help="Różnica między obligacjami USA i Niemiec. Kluczowy driver fundamentalny.")
    
    # Zmienność
    vol_state = "Niska"
    if vol_ratio > 1.2: vol_state = "🔥 WYSOKA"
    elif vol_ratio < 0.8: vol_state = "💤 Uśpienie"
    col4.metric("Zmienność (vs Norm)", f"{vol_ratio:.2f}", vol_state)

except Exception as e:
    st.error(f"Inicjalizacja silnika analitycznego... (Może brakować danych historycznych) {e}")


# --- GŁÓWNY GRID ---
c_left, c_right = st.columns([3, 1]) # 75% Wykres / 25% Analiza

with c_left:
    st.markdown("### 📈 Live Market Data")
    # Tu wstawiamy potężny widget TradingView
    tv_chart_widget(symbol="FX:EURUSD", theme="dark")

with c_right:
    st.markdown("### 🧠 Quant Brain")
    
    # Widget Techniczny (Tachometer)
    st.caption("Sentyment Techniczny (Oscylatory + Średnie)")
    tv_technical_widget(symbol="FX:EURUSD")
    
    st.divider()
    
    # Tabela Korelacji (Wyliczona w Pythonie)
    st.caption("🔥 Correlation Matrix (Last 50h)")
    # Formatowanie tabeli kolorami
    if 'corr_matrix' in locals():
        st.dataframe(
            corr_matrix.style.background_gradient(cmap='RdYlGn', vmin=-1, vmax=1).format("{:.2f}"),
            use_container_width=True
        )
        st.info("💡 Jeśli EURUSD vs Yield Spread jest dodatnie (>0.5), rynek gra pod stopy procentowe.")

# --- DOLNY PANEL (MACRO) ---
st.markdown("### 📅 Kalendarz Ekonomiczny & News")

# Widget Kalendarza TradingView
calendar_code = """
<div class="tradingview-widget-container">
  <div class="tradingview-widget-container__widget"></div>
  <script type="text/javascript" src="https://s3.tradingview.com/external-embedding/embed-widget-events.js" async>
  {
  "colorTheme": "dark",
  "isTransparent": false,
  "width": "100%",
  "height": "400",
  "locale": "pl",
  "importanceFilter": "-1,0,1",
  "currencyFilter": "USD,EUR"
}
  </script>
</div>
"""
components.html(calendar_code, height=400)
