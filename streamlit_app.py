import streamlit as st
import streamlit.components.v1 as components

# --- 1. KONFIGURACJA STRONY ---
st.set_page_config(layout="wide", page_title="EURUSD Command Center", page_icon="🦅")

# --- 2. CSS (Professional Dark UI) ---
st.markdown("""
<style>
    .stApp { background-color: #0E1117; }
    .block-container { padding-top: 0.5rem; padding-left: 1rem; padding-right: 1rem; }
    
    /* Ukrycie standardowych elementów Streamlit */
    header {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Stylizacja Tabs (Zakładek) */
    .stTabs [data-baseweb="tab-list"] { gap: 10px; }
    .stTabs [data-baseweb="tab"] {
        background-color: #161B22;
        border-radius: 4px;
        color: #C9D1D9;
        padding: 10px 20px;
    }
    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        background-color: #238636;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# --- 3. HELPER FUNCTION (Renderowanie Widgetów) ---
def render_myfx_widget(html_code, height=400, scrolling=True):
    components.html(html_code, height=height, scrolling=scrolling)

# --- 4. DEFINICJE WIDGETÓW (Kod HTML od Ciebie) ---

# WIDGET 1: Market Hours
w_hours = """
<iframe src="https://widget.myfxbook.com/widget/market-hours.html" style="border: 0; width:100%; height:100%;"></iframe>
<div style="margin-top: 10px; text-align: center; font-family: roboto,sans-serif; font-size: 10px; color: #666;">
    Powered by Myfxbook.com
</div>
"""

# WIDGET 2: Top News
w_news = """
<iframe src="https://widget.myfxbook.com/widget/news.html" style="border: 0; width:100%; height:100%;"></iframe>
"""

# WIDGET 3: Toolbar (Specjalna obsługa)
w_toolbar = """
<div style="width: 100%; text-align: center;">
<script type="text/javascript" src="https://widgets.myfxbook.com/scripts/toolbar.js"></script>
</div>
"""

# WIDGET 4: Forex Rates
w_rates = """
<iframe src="https://widget.myfxbook.com/widget/market-quotes.html?symbols=AUDUSD,EURGBP,EURUSD,GBPUSD,USDCAD,USDCHF,USDJPY" style="border: 0; width:100%; height:100%;"></iframe>
"""

# WIDGET 5: Patterns (Szeroki)
w_patterns = """
<iframe src="https://widgets.myfxbook.com/widgets/patterns.html?symbols=1,2,5,3&indicators=27,28,29,30,31,32,33,34,35,36,37,38,39,40,42,43,44,45,46,47,48,50,51,52,53,54,56,57,58,59,61,60,49,62,63,64,65,66,67,68,69,70,41,71,72,73,74,75,26,21,22,24,25,23,76,77,20,78,79,80&tType=patterns&timeFrame=1" width="100%" height="100%" frameborder="0"></iframe>
"""

# WIDGET 6: Volatility
w_volatility = """
<iframe src="https://widgets.myfxbook.com/widgets/market-volatility.html?symbols=8,9,10,11,12,6,13,14,15,17,7,1,4,2,28,5,29,3,50,51&type=0" width="100%" height="100%" frameborder="0"></iframe>
"""

# WIDGET 7: Heat Map
w_heatmap = """
<iframe src="https://widgets.myfxbook.com/widgets/heat-map.html?symbols=8,9,10,11,12,6,13,14,15,17,7,1,4,2,28,5,29,3,50,51&type=0" width="100%" height="100%" frameborder="0"></iframe>
"""

# WIDGET 8: Correlation
w_correlation = """
<iframe src="https://widgets.myfxbook.com/widgets/market-correlation.html?rowSymbols=2,29,51&colSymbols=1&timeScale=1440" width="100%" height="100%" frameborder="0"></iframe>
"""

# WIDGET 9: Liquidity
w_liquidity = """
<iframe src="https://widgets.myfxbook.com/widgets/liquidity.html?" width="100%" height="100%" frameborder="0"></iframe>
"""

# --- 5. BUDOWA DASHBOARDU ---

# A. SIDEBAR (Stały podgląd)
with st.sidebar:
    st.markdown("### 🕒 Market Clocks")
    render_myfx_widget(w_hours, height=350)
    
    st.markdown("### 💱 Quotes Board")
    render_myfx_widget(w_rates, height=600)

# B. GŁÓWNY OBSZAR
st.title("🦅 EUR/USD Quant Terminal")

# Tabs (Główne kategorie analizy)
tab_tech, tab_vol, tab_macro = st.tabs(["📊 Technical & Patterns", "🔥 Volatility & Heatmap", "🌍 News & Correlation"])

# --- TAB 1: TECHNICALS ---
with tab_tech:
    st.markdown("#### 🧠 Automated Pattern Recognition")
    # Patterns to bardzo duży widget, dajemy mu full width
    render_myfx_widget(w_patterns, height=600)
    
    st.divider()
    
    col_liq, col_blank = st.columns([1, 1])
    with col_liq:
        st.markdown("#### 💧 Market Liquidity Estimates")
        render_myfx_widget(w_liquidity, height=400)
    with col_blank:
        st.info("💡 Liquidity Widget pokazuje szacowaną głębokość rynku. Niska płynność + Duży news = Slippage.")

# --- TAB 2: VOLATILITY & FLOW ---
with tab_vol:
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### ⚡ Market Volatility")
        render_myfx_widget(w_volatility, height=500)
        
    with col2:
        st.markdown("#### 🔥 Currency Heatmap")
        render_myfx_widget(w_heatmap, height=500)
    
    st.caption("Heatmap pokazuje siłę walut relatywnie do siebie. Jeśli EUR jest 'gorące' (zielone), a USD 'zimne' (czerwone) -> Silny sygnał LONG na EURUSD.")

# --- TAB 3: MACRO & CORRELATION ---
with tab_macro:
    col_news, col_corr = st.columns([1, 2])
    
    with col_news:
        st.markdown("#### 📰 Breaking News")
        render_myfx_widget(w_news, height=600)
        
    with col_corr:
        st.markdown("#### 🔗 Asset Correlation Matrix")
        render_myfx_widget(w_correlation, height=600)
