# @title 🏛️ QUANT LAB v26.0: THE CLOUD ARCHITECT (Streamlit App)
# PRZENIESIENIE: Lokalny Skrypt -> Profesjonalna Aplikacja Webowa

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Rectangle
import os
import csv

try:
    import seaborn as sns
    SEABORN_AVAILABLE = True
except ImportError:
    SEABORN_AVAILABLE = False

# --- KONFIGURACJA STREAMLIT ---
st.set_page_config(
    page_title="QUANT LAB | Institutional Terminal",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- KONFIGURACJA SYSTEMU ---
CONFIG = {
    'TARGET': 'EURUSD=X',
    'MACRO': {'US10Y': '^TNX', 'VIX': '^VIX', 'SPX': '^GSPC', 'GOLD': 'GC=F'},
    'LOOKBACK': '6mo',
    'INTERVAL': '1h',
    'LEDGER_FILE': 'quant_signals_log.csv',
    'BEST_PARAMS': {
        'eta': 0.05, 'max_depth': 5, 'objective': 'binary:logistic',
        'eval_metric': 'logloss', 'subsample': 0.8, 'colsample_bytree': 0.8
    }
}

# --- POMOCNICZE (Tytuł i Sidebar) ---
st.sidebar.title("🏛️ QUANT LAB v26.0")
st.sidebar.markdown("---")
st.sidebar.markdown("### ⚙️  System Settings")

# W Streamlit używamy sesji do przechowywania stanu, żeby nie restartować całego app'a
if "df_full" not in st.session_state:
    st.session_state.df_full = None
    st.session_state.alerts = []

# Sidebar Controls
symbol = st.sidebar.text_input("Target Pair", "EURUSD=X")
lookback_opt = st.sidebar.selectbox("History Depth", ["6mo", "1y", "2y"])
CONFIG['TARGET'] = symbol
CONFIG['LOOKBACK'] = lookback_opt

run_btn = st.sidebar.button("🚀 RUN LIVE ANALYSIS", type="primary")

# --- 1. DATA CACHING (Kluczowa zaleta Streamlit) ---
@st.cache_data(ttl=3600) # Cache'uj dane na 1 godzinę
def load_data(target, lookback, interval):
    tickers = [target] + list(CONFIG['MACRO'].values())
    raw = yf.download(tickers, period=lookback, interval=interval, progress=False)
    return raw

def prepare_ml_data(df):
    df = df.copy()
    df['Log_Ret'] = np.log(df['Close'] / df['Close'].shift(1))
    for w in [5, 10, 20]:
        df[f'Mom_{w}'] = df['Close'].pct_change(w)
        df[f'Vol_{w}'] = df['Log_Ret'].rolling(w).std()
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(w).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(w).mean()
        rs = gain / loss
        df[f'RSI_{w}'] = 100 - (100 / (1 + rs))
    for name, ticker in CONFIG['MACRO'].items():
        if name in df.columns:
            df[f'{name}_Ret'] = df[name].pct_change()
            df[f'{name}_Z'] = (df[name] - df[name].rolling(48).mean()) / df[name].rolling(48].std()
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    df['True_Range'] = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['ATR'] = df['True_Range'].rolling(14).mean()
    df.dropna(inplace=True)
    return df

# --- 2. ENGINES ---
class MacroEngine:
    @staticmethod
    def analyze(df):
        flows = []; score = 0
        definitions = {'US10Y': -1, 'VIX': -1, 'SPX': 1, 'GOLD': 1}
        for asset, impact in definitions.items():
            if asset in df.columns:
                z = df[f'{asset}_Z'].iloc[-1]
                r = df[f'{asset}_Ret'].iloc[-1]
                raw_impact = np.sign(r) * impact
                strength = min(5, int(abs(z) + 1))
                bias = "NEUTRAL"
                if raw_impact > 0: bias = "BULLISH"
                if raw_impact < 0: bias = "BEARISH"
                flows.append({'name': asset, 'bias': bias, 'strength': strength, 'z': z})
                if abs(z) > 0.5: score += int(np.sign(r)) * impact
        return flows, score

class MLEngine:
    @staticmethod
    def predict(df):
        future_ret = df['Close'].shift(-4) / df['Close'] - 1
        y = (future_ret > 0.0005).astype(int) 
        exclude_cols = ['Open', 'High', 'Low', 'Close']
        cols = [c for c in df.columns if c not in exclude_cols]
        X = df[cols]
        split = len(df) - 100
        model = xgb.XGBClassifier(**CONFIG['BEST_PARAMS'], n_estimators=300, use_label_encoder=False)
        model.fit(X.iloc[:split], y.iloc[:split])
        return model.predict_proba(X.iloc[[-1]])[0][1]

class SqueezeEngine:
    @staticmethod
    def detect(df):
        atr_curr = df['ATR'].iloc[-1]
        atr_avg = df['ATR'].iloc[-50:].mean()
        if atr_curr < atr_avg * 0.75:
            return (df['Low'].iloc[-48:].min(), df['High'].iloc[-48:].max(), 48)
        return None

# --- 3. DASHBOARD (WERSJA STREAMLIT) ---
class MasterDashboard:
    @staticmethod
    def plot(df, action, ml_prob, macro_flows, macro_score, squeeze_bounds, history_csv_path, alerts):
        fig = plt.figure(figsize=(24, 14), dpi=100)
        gs = gridspec.GridSpec(3, 3, height_ratios=[3, 2, 2])
        
        # 1. MAIN CHART
        ax_main = fig.add_subplot(gs[0, :])
        plot_df = df.iloc[-72:].copy()
        ax_main.plot(plot_df.index, plot_df['Close'], color='#e0e0e0', linewidth=2, label='EURUSD', zorder=2)
        
        if squeeze_bounds:
            low, high, width = squeeze_bounds
            rect = Rectangle((plot_df.index[-width], low), 
                             plot_df.index[-1] - plot_df.index[-width], high - low,
                             linewidth=1, edgecolor='#ffa726', facecolor='#ffa726', alpha=0.1, zorder=1)
            ax_main.add_patch(rect)
            ax_main.axhline(high, color='#ffa726', linestyle='--', alpha=0.5, label='Squeeze Res')
            ax_main.axhline(low, color='#ffa726', linestyle='--', alpha=0.5, label='Squeeze Sup')

        curr_price = df['Close'].iloc[-1]
        ax_main.scatter(plot_df.index[-1], curr_price, color='yellow', s=150, zorder=5, edgecolors='black', label='Current')
        ax_main.set_title(f"EURUSD LIVE | {curr_price:.5f} | SIGNAL: {action}", fontsize=18, color='white')
        ax_main.grid(True, color='#333333', linestyle=':', alpha=0.4)
        ax_main.legend(loc='upper left', fontsize=10)

        # 2. MACRO NEXUS
        ax_macro = fig.add_subplot(gs[1, 0])
        ax_macro.axis('off')
        macro_text = "🌐 GLOBAL MACRO NEXUS\n--------------------------\n"
        for f in macro_flows:
            c = "#00e676" if f['bias'] == "BULLISH" else "#ff5252" if f['bias'] == "BEARISH" else "#9e9e9e"
            bar = "█" * f['strength']
            macro_text += f"{f['name']}: {f['bias']}\n  {bar} (Z:{f['z']:.1f})\n\n"
        ax_macro.text(0.05, 0.9, macro_text, transform=ax_macro.transAxes, fontsize=11, 
                      verticalalignment='top', bbox=dict(boxstyle='round', facecolor='#212121', edgecolor='white'), color='white', fontfamily='monospace')

        # 3. CORRELATION MATRIX
        ax_corr = fig.add_subplot(gs[1, 1])
        assets = ['Close'] + list(CONFIG['MACRO'].keys())
        corr_df = df[assets].iloc[-100:].corr()
        if SEABORN_AVAILABLE:
            sns.heatmap(corr_df, annot=True, fmt=".2f", cmap='coolwarm', center=0, ax=ax_corr, cbar=False, linewidths=.5, linecolor='black', annot_kws={"size": 8})
        else:
            ax_corr.imshow(corr_df, cmap='coolwarm', vmin=-1, vmax=1)
        ax_corr.set_title("REAL-TIME CORRELATION (100h)", fontsize=11, color='white')
        ax_corr.tick_params(colors='white', labelsize=8)

        # 4. AI SENTIMENT HISTORY
        ax_sent = fig.add_subplot(gs[1, 2])
        if os.path.exists(history_csv_path):
            hist_df = pd.read_csv(history_csv_path)
            if 'Timestamp' not in hist_df.columns:
                 hist_df.columns = ['Timestamp', 'Decision', 'AI_Prob', 'Macro_Score', 'Price', 'Note']
            hist_df['Timestamp'] = pd.to_datetime(hist_df['Timestamp'])
            recent_hist = hist_df.tail(100)
            ax_sent.plot(recent_hist.index, recent_hist['AI_Prob'], color='#e040fb', linewidth=2, label='AI Confidence')
            ax_sent.fill_between(recent_hist.index, recent_hist['AI_Prob'], 0.5, color='#e040fb', alpha=0.2)
            ax_sent.axhline(0.5, color='gray', linestyle='--', linewidth=1)
            ax_sent.set_title("🕸️ AI SENTIMENT HISTORY (Memory)", fontsize=11, color='#e040fb')
            ax_sent.grid(True, color='#333', linestyle=':')
        ax_sent.tick_params(axis='x', labelsize=8)

        # 5. EXECUTION PANEL
        ax_exec = fig.add_subplot(gs[2, :])
        ax_exec.axis('off')
        
        panel_color = '#1b5e20' if "LONG" in action else '#b71c1c' if "SHORT" in action else '#263238'
        
        exec_text = (
            f"⚔️ ACTION PLAN: {action}\n"
            f"-----------------------------------------------------------------------------------------\n"
            f"AI CONVICTION: {ml_prob*100:.1f}% |  RISK:REWARD TARGET: 1:2 (ATR Based)\n"
            f"COMPOSITE MACRO SCORE: {macro_score}  |  REGIME: {'SQUEEZE' if squeeze_bounds else 'TREND'}\n"
            f"-----------------------------------------------------------------------------------------\n"
            f"💡 TRADER INSIGHT: "
        )
        
        insight = "Market is indecisive. Wait for break."
        if "LONG" in action: insight = "Strong Bullish momentum + Macro support. Look for pullbacks to buy."
        if "SHORT" in action: insight = "Bearish breakdown detected. Protect downside aggressively."
        if squeeze_bounds: insight = "Volatility compressed. Expect explosive move soon. Set alerts on Box edges."
        
        # ALERTS BLOCK
        if alerts:
            alert_str = "\n".join(alerts)
            insight += f"\n\n🔔 SENTINEL ALERTS:\n{alert_str}"
        
        exec_text += insight
        
        ax_exec.text(0.02, 0.5, exec_text, transform=ax_exec.transAxes, fontsize=14,
                     verticalalignment='center', bbox=dict(boxstyle='round', facecolor=panel_color, edgecolor='white', alpha=0.9), color='white', fontweight='bold', fontfamily='monospace')

        plt.tight_layout()
        return fig

# --- 4. LOGIKA APLIKACJI ---
if run_btn or st.session_state.df_full is not None:
    
    if run_btn:
        st.session_state.df_full = None # Reset cache if clicked

    if st.session_state.df_full is None:
        with st.spinner("📥 Fetching Data & Running AI Engines..."):
            # 1. Load Data
            raw = load_data(CONFIG['TARGET'], CONFIG['LOOKBACK'], CONFIG['INTERVAL'])
            
            df = pd.DataFrame()
            df['Open'] = raw['Open'][CONFIG['TARGET']]
            df['High'] = raw['High'][CONFIG['TARGET']]
            df['Low'] = raw['Low'][CONFIG['TARGET']]
            df['Close'] = raw['Close'][CONFIG['TARGET']]
            for name, ticker in CONFIG['MACRO'].items():
                df[name] = raw['Close'][ticker]
            df = df.ffill().dropna()
            df = prepare_ml_data(df)
            
            st.session_state.df_full = df
            
            # 2. Run Engines
            ml_prob = MLEngine.predict(df)
            macro_flows, macro_score = MacroEngine.analyze(df)
            squeeze_bounds = SqueezeEngine.detect(df)
            
            action = "WAIT / FLAT"
            if ml_prob < 0.30: action = "SHORT NOW"
            elif ml_prob > 0.70: action = "LONG NOW"
            elif squeeze_bounds: action = "WAIT (SQUEEZE PREP)"
            
            st.session_state.results = {
                'action': action, 'ml_prob': ml_prob, 'macro_flows': macro_flows,
                'macro_score': macro_score, 'squeeze_bounds': squeeze_bounds
            }
            
            # 3. Sentinel / Alerts (Opcjonalnie)
            st.session_state.alerts = [] # Tu można wstawić logikę sentinel

    df = st.session_state.df_full
    res = st.session_state.results
    alerts = st.session_state.alerts
    
    # --- WYŚWIETLANIE ---
    st.markdown(f"### 📊 SIGNAL: {res['action']} | AI PROB: {res['ml_prob']*100:.1f}% | MACRO SCORE: {res['macro_score']}")
    
    fig = MasterDashboard.plot(df, res['action'], res['ml_prob'], res['macro_flows'], 
                                res['macro_score'], res['squeeze_bounds'], CONFIG['LEDGER_FILE'], alerts)
    
    st.pyplot(fig)

else:
    st.info("👈 Select settings in the sidebar and click **RUN LIVE ANALYSIS**.")
