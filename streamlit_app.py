import streamlit as st
import streamlit.components.v1 as components

# --- 1. KONFIGURACJA STRONY (Ultra Wide & Dark) ---
st.set_page_config(
    layout="wide",
    page_title="EURUSD Quant Cockpit",
    page_icon="🦅",
    initial_sidebar_state="collapsed" # Zwijamy sidebar, żeby mieć więcej miejsca
)

# --- 2. ZAAWANSOWANY CSS (Professional UI/UX) ---
st.markdown("""
<style>
    /* Tło aplikacji */
    .stApp { background-color: #0a0c0f; } /* Bardzo głęboka czerń */
    
    /* Reset marginesów dla maksymalnego wykorzystania miejsca */
    .block-container {
        padding-top: 1rem;
        padding-bottom: 2rem;
        padding-left: 1rem;
        padding-right: 1rem;
        max-width: 100%;
    }
    
    /* Ukrycie standardowego nagłówka Streamlit */
    header {visibility: hidden;}
    
    /* STYLIZACJA KONTENERÓW DLA WIDGETÓW */
    /* Tworzymy "karty" dla każdego widgetu, żeby wyglądało to spójnie */
    [data-testid="stVerticalBlock"] > [style*="flex-direction: column;"] > [data-testid="stVerticalBlock"] {
        background-color: #161b22;
        border: 1px solid #30363d;
        border-radius: 8px;
        padding: 15px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
        margin-bottom: 1rem;
    }
    
    /* Nagłówki sekcji */
    h3, h4 {
        color: #e6edf3 !important;
        font-weight: 600;
        margin-bottom: 15px !important;
        border-bottom: 2px solid #238636; /* Zielony akcent */
        padding-bottom: 5px;
        font-size: 1.1rem !important;
    }
    
    /* Divider */
    hr { border-color: #30363d; }

    /* Hack dla iframe'ów - próba wymuszenia ciemniejszego otoczenia */
    iframe {
        background-color: #ffffff; /* Myfxbook wymusza biały, musimy to zaakceptować */
        border-radius: 4px;
    }
</style>
""", unsafe_allow_html=True)

# --- 3. HELPER FUNCTION ---
def render_widget(html_code, height):
    # Opakowujemy widget w div, żeby CSS mógł go złapać jako kontener
    components.html(html_code, height=height, scrolling=True)

# --- 4. DEFINICJE WIDGETÓW (Zwiększone wysokości) ---

# Row 1 Widgets
w_hours = """<iframe src="https://widget.myfxbook.com/widget/market-hours.html" style="border: 0; width:100%; height:100%;"></iframe>"""
w_news = """<iframe src="https://widget.myfxbook.com/widget/news.html" style="border: 0; width:100%; height:100%;"></iframe>"""

# Row 2 (Quant Data) Widgets - Zwiększona wysokość dla czytelności
w_heatmap = """<iframe src="https://widget.myfxbook.com/widget/heat-map.html?symbols=EURUSD,GBPUSD,USDJPY,USDCAD,AUDUSD,NZDUSD,USDCHF&type=0" width="100%" height="100%" frameborder="0"></iframe>"""
w_volatility = """<iframe src="https://widget.myfxbook.com/widget/market-volatility.html?symbols=EURUSD,GBPUSD,USDJPY,USDCAD,AUDUSD,NZDUSD,USDCHF&type=0" width="100%" height="100%" frameborder="0"></iframe>"""
w_correlation = """<iframe src="https://widget.myfxbook.com/widget/market-correlation.html?rowSymbols=EURUSD,GBPUSD,USDJPY&colSymbols=USDCAD,AUDUSD,USDCHF&timeScale=1440" width="100%" height="100%" frameborder="0"></iframe>"""

# Row 3 (Deep Dive) Widgets
w_patterns = """<iframe src="https://widgets.myfxbook.com/widgets/patterns.html?symbols=1&timeFrame=1" width="100%" height="100%" frameborder="0"></iframe>"""
w_liquidity = """<iframe src="https://widgets.myfxbook.com/widgets/liquidity.html?" width="100%" height="100%" frameborder="0"></iframe>"""
w_rates = """<iframe src="https://widget.myfxbook.com/widget/market-quotes.html?symbols=AUDUSD,EURGBP,EURUSD,GBPUSD,USDCAD,USDCHF,USDJPY" style="border: 0; width:100%; height:100%;"></iframe>"""


# --- 5. LAYOUT KOKPITU (SINGLE PAGE GRID) ---

st.markdown("### 🦅 EUR/USD Market Overview")

# --- ROW 1: STATUS & NEWS (Dzielimy ekran 1/3 do 2/3) ---
col_r1_1, col_r1_2 = st.columns([1, 2])

with col_r1_1:
    st.markdown("#### 🕒 Market Sessions")
    render_widget(w_hours, height=350)
    
with col_r1_2:
    st.markdown("#### 📰 Breaking Headlines")
    render_widget(w_news, height=350)

st.markdown("---")

# --- ROW 2: QUANT DATA CORE (Trzy równe kolumny) ---
st.markdown("### 🧠 Quantitative Flow Analysis")
col_r2_1, col_r2_2, col_r2_3 = st.columns(3)

with col_r2_1:
    st.markdown("#### 🔥 Currency Heatmap")
    render_widget(w_heatmap, height=500)
    st.caption("Zielone = Silne, Czerwone = Słabe. Szukaj par z największym kontrastem.")

with col_r2_2:
    st.markdown("#### ⚡ Volatility (Pips)")
    render_widget(w_volatility, height=500)
    st.caption("Wysoka zmienność = Większe ryzyko i większy potencjał.")

with col_r2_3:
    st.markdown("#### 🔗 Correlation Matrix (Daily)")
    render_widget(w_correlation, height=500)
    st.caption("Unikaj otwierania pozycji na parach skorelowanych > 80%.")

st.markdown("---")

# --- ROW 3: DEEP DIVE & REFERENCE (Dzielimy 2/1/1) ---
st.markdown("### 🎯 Technicals & Liquidity")
col_r3_1, col_r3_2, col_r3_3 = st.columns([2, 1, 1])

with col_r3_1:
    st.markdown("#### 📐 Auto-Chart Patterns (EURUSD M1)")
    # Przefiltrowałem widget tylko do EURUSD (symbols=1) i M1 dla daytradingu
    render_widget(w_patterns, height=600)

with col_r3_2:
    st.markdown("#### 💧 Market Liquidity")
    render_widget(w_liquidity, height=600)

with col_r3_3:
    st.markdown("#### 💱 Live Quotes")
    render_widget(w_rates, height=600)
